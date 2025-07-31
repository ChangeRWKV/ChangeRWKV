import argparse
import json
import os
import sys
import time
import warnings
import multiprocessing as mproc
import xlsxwriter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.ChangeRWKV import ChangeRWKV
from utils.dataset import ChangeDetectionDataset
from utils.func import clip_gradient
from utils.metric_tool import AvgMeter, ConfuseMatrixMeter
from utils.loss_f import BCEDICE_loss
from utils.lr_scheduler import get_scheduler

sys.path.append(os.path.abspath("."))
warnings.filterwarnings("ignore")


def parse_option():
    parser = argparse.ArgumentParser()
    # io
    parser.add_argument('--dataset', type=str, default='LEVIR')
    parser.add_argument('--data_dir', type=str, default='data/LEVIR/ABLabel')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--pretrain_dir', type=str, default='ckpt/vrwkv_b.pth')
    parser.add_argument('--output_dir', type=str, default='./output', help='output director')
    # model
    parser.add_argument('--model_size', type=str, default="base")
    parser.add_argument('--out_channels', type=int, default=1)
    # training
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--devices', type=int, default=4, help='gpu number')
    parser.add_argument('--epochs', type=int, default=200, help='epoch number')
    parser.add_argument('--num_workers', type=int, default=mproc.cpu_count(), help='num worker')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--warmup_epoch', type=int, default=20, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=10, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_steps', type=int, default=20,
                        help='for step scheduler. step size to decay lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.999, help='momentum for SGD')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')

    opt, unparsed = parser.parse_known_args()
    opt.output_dir = os.path.join(opt.output_dir, opt.dataset, f"ChangeRWKV-{opt.model_size[0]}")
    return opt


def build_loader(opt):
    train_data = ChangeDetectionDataset(opt.data_dir, "train")
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              pin_memory=True, drop_last=True)
    val_data = ChangeDetectionDataset(opt.data_dir, "val")
    val_loader = DataLoader(val_data, batch_size=opt.batch_size if opt.dataset != "WHU" else 4, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=True)
    return train_loader, val_loader


def build_model(opt):
    model = ChangeRWKV(img_size=opt.img_size, num_classes=opt.out_channels, size=opt.model_size)
    if opt.pretrain_dir is not None:
        pretrained_dict = torch.load(opt.pretrain_dir)
        model_dict = model.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            k = "encoder." + k
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    return model


def main(opt):
    train_loader, val_loader = build_loader(opt)
    print(f"Length of training dataset: {len(train_loader.dataset)}")
    print(f"Length of val dataset: {len(val_loader.dataset)}")
    model = nn.DataParallel(build_model(opt), device_ids=list(range(opt.devices))).cuda()

    params = 0
    for _, param in model.named_parameters():
        params += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}".format(params))

    train_hook = xlsxwriter.Workbook(f'ChangeRWKV-{opt.model_size[0]}_{opt.dataset}_train.xlsx')
    train_record = train_hook.add_worksheet()
    train_record.write('A1', 'epoch')
    train_record.write('B1', 'Pre')
    train_record.write('C1', 'Recall')
    train_record.write('D1', 'F1')
    train_record.write('E1', 'IoU')
    train_record.write('F1', 'acc')
    train_record.write('G1', 'loss')
    val_hook = xlsxwriter.Workbook(f'ChangeRWKV-{opt.model_size[0]}_{opt.dataset}_val.xlsx')
    val_record = val_hook.add_worksheet()
    val_record.write('A1', 'epoch')
    val_record.write('B1', 'Pre')
    val_record.write('C1', 'Recall')
    val_record.write('D1', 'F1')
    val_record.write('E1', 'IoU')
    val_record.write('F1', 'acc')
    val_record.write('G1', 'loss')
    row = 1
    col = 0
    with open(os.path.join(opt.output_dir, 'res.out'), 'w') as f:
        f.write('---------------------------------------------\n')

    # build optimizer
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), opt.lr / 10.0 * opt.batch_size, momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError(f"Unimplemented optimizer type: {opt.optim}")
    scheduler = get_scheduler(optimizer, len(train_loader), opt)
    # routine
    for epoch in range(1, opt.epochs + 1):
        tic = time.time()
        tool4metric = ConfuseMatrixMeter(n_class=2)
        print(f"Epoch {epoch}, train:")
        train(train_loader, model, optimizer, BCEDICE_loss, scheduler, epoch, tool4metric, train_record, row, col)
        print('Epoch {}, total time {:.2f}, learning_rate {}'.format(epoch, (time.time() - tic),
                                                                     optimizer.param_groups[0]['lr']))
        print(f"Epoch {epoch}, val:")
        val(val_loader, model, BCEDICE_loss, epoch, tool4metric, val_record, row, col)
        print('Epoch {}, total time {:.2f}'.format(epoch, (time.time() - tic)))
        torch.save(model.state_dict(), os.path.join(opt.output_dir, "last.pth"))
        print("Model saved: {}!".format(os.path.join(opt.output_dir, "last.pth")))
        if (epoch >= 30) & (epoch % 1 == 0):
            torch.save(model.state_dict(), os.path.join(opt.output_dir, f"Epoch{epoch}.pth"))
            print("Model saved: {}!".format(os.path.join(opt.output_dir, f"Epoch{epoch}.pth")))
        row = row + 1
    train_hook.close()
    val_hook.close()


def train(train_loader, model, optimizer, criterion, scheduler, epoch, tool4metric, train_record, row, col):
    tool4metric.clear()
    model.train()
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        imageA, imageB, gts = pack
        imageA = imageA.cuda().float()
        imageB = imageB.cuda().float()
        gts = gts.cuda().float()

        # forward
        pred_s = model(imageA, imageB)
        if torch.isnan(pred_s).any():
            del imageA, imageB, gts, pred_s
            raise Exception("nan occur")
        gts = torch.unsqueeze(gts, dim=1)
        loss = criterion(pred_s, gts)
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        scheduler.step()

        loss_record.update(loss.data, opt.batch_size)
        bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
        mask = gts.to('cpu').numpy().astype(int)
        tool4metric.update_cm(pr=bin_preds_mask, gt=mask)
        if i % 100 == 0 or i == len(train_loader):
            print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                  'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(train_loader), loss_record.show()))
    scores_dictionary = tool4metric.get_scores()
    print('---------------------------------------------')
    train_record.write(row, col, epoch)
    train_record.write(row, col + 1, scores_dictionary['precision'])
    train_record.write(row, col + 2, scores_dictionary['recall'])
    train_record.write(row, col + 3, scores_dictionary['F1'])
    train_record.write(row, col + 4, scores_dictionary['iou'])
    train_record.write(row, col + 5, scores_dictionary['acc'])
    train_record.write(row, col + 6, loss_record.show())


def val(val_loader, model, criterion, epoch, tool4metric, val_record, row, col):
    model.eval()
    tool4metric.clear()
    loss_record = AvgMeter()
    with torch.no_grad():
        for i, pack in enumerate(val_loader):
            imageA, imageB, gts = pack
            imageA = imageA.cuda().float()
            imageB = imageB.cuda().float()
            gts = gts.cuda().float()
            pred_s = model(imageA, imageB)
            if torch.isnan(pred_s).any():
                pred_s = torch.nan_to_num(pred_s, nan=1.0)
            gts = torch.unsqueeze(gts, dim=1)
            loss = criterion(pred_s, gts)
            bin_preds_mask = (pred_s.to('cpu') > 0.5).detach().numpy().astype(int)
            mask = gts.to('cpu').numpy().astype(int)
            tool4metric.update_cm(pr=bin_preds_mask, gt=mask)
            loss_record.update(loss.data, opt.batch_size)
            if i % 100 == 0 or i == len(val_loader):
                print('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],'
                      'Loss: {:.4f}'.format(epoch, opt.epochs, i, len(val_loader), loss_record.show()))
        scores_dictionary = tool4metric.get_scores()
        print("IoU for epoch {} is {}".format(epoch, scores_dictionary["iou"]))
        print("F1 for epoch {} is {}".format(epoch, scores_dictionary["F1"]))
        print("acc for epoch {} is {}".format(epoch, scores_dictionary["acc"]))
        print("precision for epoch {} is {}".format(epoch, scores_dictionary["precision"]))
        print("recall for epoch {} is {}".format(epoch, scores_dictionary["recall"]))
        print('---------------------------------------------')
        with open(os.path.join(opt.output_dir, 'res.out'), 'a') as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"IoU for epoch {epoch}: {scores_dictionary['iou']}\n")
            f.write(f"F1 for epoch {epoch}: {scores_dictionary['F1']}\n")
            f.write(f"acc for epoch {epoch}: {scores_dictionary['acc']}\n")
            f.write(f"precision for epoch {epoch}: {scores_dictionary['precision']}\n")
            f.write(f"recall for epoch {epoch}: {scores_dictionary['recall']}\n")
            f.write(f"Loss for epoch {epoch}: {loss_record.show()}\n")
            f.write('---------------------------------------------\n')
        val_record.write(row, col, epoch)
        val_record.write(row, col + 1, scores_dictionary['precision'])
        val_record.write(row, col + 2, scores_dictionary['recall'])
        val_record.write(row, col + 3, scores_dictionary['F1'])
        val_record.write(row, col + 4, scores_dictionary['iou'])
        val_record.write(row, col + 5, scores_dictionary['acc'])
        val_record.write(row, col + 6, loss_record.show())


if __name__ == '__main__':
    opt = parse_option()
    os.makedirs(opt.output_dir, exist_ok=True)
    path = os.path.join(opt.output_dir, 'config.json')
    with open(path, 'w') as f:
        json.dump(vars(opt), f, indent=2)
    print("Full config save to {}".format(path))
    main(opt)
