import math

import torch


def build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': params, 'lr': args.lr, 'weight_decay': args.wd}], betas=(0.9, 0.95))
    return optimizer, params


def adjust_learning_rate(param_group, base_lr, epoch_progress, args):
    min_lr = 5e-6
    if epoch_progress < args.warmup_epochs:
        lr = base_lr * epoch_progress / args.warmup_epochs
    else:
        denom = max(args.num_epochs - args.warmup_epochs, 1e-8)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (epoch_progress - args.warmup_epochs) / denom))
    param_group['lr'] = lr
    return lr
