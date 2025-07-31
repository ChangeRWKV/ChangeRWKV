from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)
    elif "step" in args.lr_scheduler:
        if len(args.lr_decay_epochs) > 0:
            lr_decay_epochs = args.lr_decay_epochs
        else:
            lr_decay_epochs = [args.lr_decay_steps * i for i in range(1, args.epochs // args.lr_decay_steps)]
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=args.lr_decay_rate,
            milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in lr_decay_epochs])
    else:
        raise NotImplementedError(f"scheduler {args.lr_scheduler} not supported")

    if args.warmup_epoch > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            after_scheduler=scheduler,
            total_epoch=args.warmup_epoch * n_iter_per_epoch)
    return scheduler
