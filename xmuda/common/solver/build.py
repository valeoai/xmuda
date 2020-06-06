"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR


def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = ClipLR(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)

    return scheduler
