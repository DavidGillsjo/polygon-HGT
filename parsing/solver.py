import torch
from math import sqrt

def calculate_lr(cfg):
    return cfg.SOLVER.BASE_LR*(cfg.SOLVER.IMS_PER_BATCH/10)*cfg.SOLVER.NUM_GPUS


def make_optimizer(cfg, model):

    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        # Use size 10 as base
        # lr=cfg.SOLVER.BASE_LR*sqrt(cfg.SOLVER.IMS_PER_BATCH/10)
        lr=calculate_lr(cfg)
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        # if 'md_predictor' in key or 'st_predictor' in key or 'ed_predictor' in key:
        #     lr = cfg.SOLVER.BASE_LR*100.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    if cfg.SOLVER.OPTIMIZER == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    cfg.SOLVER.BASE_LR,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIMIZER == 'ADAM':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY,
                                     amsgrad=cfg.SOLVER.AMSGRAD)
    elif cfg.SOLVER.OPTIMIZER == 'RADAM':
        optimizer = torch.optim.RAdam(params, lr=cfg.SOLVER.BASE_LR,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise NotImplementedError()
    return optimizer

def make_lr_scheduler(cfg,optimizer):
    if cfg.SOLVER.LR_SCHEDULER == 'MULTISTEP':
        lr_scheduler =  torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=cfg.SOLVER.STEPS,gamma=cfg.SOLVER.GAMMA)
    elif cfg.SOLVER.LR_SCHEDULER == 'COSINE_ANNEALING':
        lr=calculate_lr(cfg)
        lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0 = cfg.SOLVER.FIRST_RESTART,
                                                                             T_mult = 2,
                                                                             eta_min = cfg.SOLVER.ETA_MIN_MULTIPLIER*lr)
    else:
        raise NotImplementedError()

    if cfg.SOLVER.WARMUP_EPOCHS > 0:
        warmup_sceduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                            start_factor=cfg.SOLVER.WARMUP_LR_MODIFIER,
                                                            total_iters=cfg.SOLVER.WARMUP_EPOCHS)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sceduler, lr_scheduler], milestones=[cfg.SOLVER.WARMUP_EPOCHS])

    return lr_scheduler
