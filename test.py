import argparse
import os
import shutil
import warnings
from tqdm.auto import tqdm

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from model.ChangeRWKV import ChangeRWKV
from utils.metric_tool import ConfuseMatrixMeter
from utils.dataset import ChangeDetectionDataset
from utils.loss_f import BCEDICE_loss

warnings.warn('ignore')

parser = argparse.ArgumentParser()
# io
parser.add_argument('--model_path', type=str, default="ckpt/changerwkv-b_levir.pth", help='path to model file')
parser.add_argument('--dataset', type=str, default="LEVIR", help='test dataset')
parser.add_argument('--data_path', type=str, default='data/LEVIR/ABLabel')
parser.add_argument('--img_size', type=int, default=256, help='image size')
parser.add_argument('--save_path', type=str, default="preds/", help='save path of predictions')
# model
parser.add_argument('--model_size', type=str, default="base")
parser.add_argument('--out_channels', type=int, default=1)

opt = parser.parse_args()

test_data = ChangeDetectionDataset(opt.data_path, "test")
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
textfile = os.path.join(opt.data_path, "list/test.txt")
namelines = []
with open(textfile, 'r', encoding='utf-8') as file:
    for c in file.readlines():
        namelines.append(c.strip('\n').split(' ')[0])

model = ChangeRWKV(img_size=opt.img_size, num_classes=opt.out_channels, size=opt.model_size)
state_dict = torch.load(opt.model_path)
model_dict = model.state_dict()
load_key, no_load_key, temp_dict = [], [], {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)
model = model.cuda()
model.eval()
print(no_load_key)

bce_loss = 0.0
criterion = BCEDICE_loss
tool_metric = ConfuseMatrixMeter(n_class=2)

if opt.save_path is not None:
    if os.path.exists(opt.save_path):
        shutil.rmtree(opt.save_path)
    os.makedirs(opt.save_path)

with torch.no_grad():
    i = 0
    for imageA, imageB, gts in tqdm(test_loader):
        imageA = imageA.cuda().float()
        imageB = imageB.cuda().float()
        gts = gts.float()

        generated_mask = model(imageA, imageB)
        generated_mask = generated_mask.squeeze(1).cpu()
        bce_loss += criterion(generated_mask, gts)

        bin_genmask = (generated_mask.to('cpu') > 0.5).numpy().astype(int)
        out_png = bin_genmask.squeeze(0)

        if opt.save_path is not None:
            savename = os.path.join(opt.save_path, namelines[i])
            cv2.imwrite(savename, out_png)
        i += 1

        gts = gts.numpy()
        gts = gts.astype(int)
        tool_metric.update_cm(pr=bin_genmask, gt=gts)

    bce_loss /= len(test_loader)
    print("Test summary")
    print("Loss is {}".format(bce_loss))
    scores_dictionary = tool_metric.get_scores()
    print(scores_dictionary)
