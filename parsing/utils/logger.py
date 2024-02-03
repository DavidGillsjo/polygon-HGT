# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import wandb
import os.path as osp
import torch

class CudaTimer:
    def __init__(self, active = True):
        if active:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start_timer = self._start_timer_active
            self.end_timer = self._end_timer_active
        else:
            self.start_timer = self._start_timer_noop
            self.end_timer = self._end_timer_noop

    @classmethod
    def initiate_timer(cls, active = True):
        ctimer = cls(active=active)
        ctimer.start_timer()
        return ctimer


    def _start_timer_active(self):
        # Sync so we don't capture other work
        torch.cuda.synchronize()
        self.start.record()


    def _end_timer_active(self):
        self.end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        return self.start.elapsed_time(self.end)*1e-3

    def _start_timer_noop(self):
        pass

    def _end_timer_noop(self):
        return 0.0


def setup_logger(name, save_dir, out_file='log.txt', on_stdout = True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    if on_stdout:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, out_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def setup_noop_logger(name='noop'):
    logger = logging.getLogger(name)
    ch = logging.NullHandler()
    logger.addHandler(ch)
    return logger


#WANDB config here.
# os.environ["WANDB_API_KEY"] = "<Insert your key>"
project = "semantic-room-wireframe"
entity = "room-wireframe"

# def wandb_init_from_output_dir(cfg,output_dir, disable_wandb=False):
#     mode = 'disabled' if disable_wandb else 'online'
#     if resume:
#         wandb_run = _wandb_from_checkpoint(cfg, checkpointer, mode)
#     else:
#         wandb_run = _wandb_from_config(cfg, timestamp, mode)
#     checkpointer.wandb_id = wandb_run.id
#     return wandb_run

def wandb_init(cfg, checkpointer, resume=False, timestamp = '', disable_wandb = False):
    # Do not init wandb if there is no API key
    if "WANDB_API_KEY" not in os.environ:
        disable_wandb = True
    
    mode = 'disabled' if disable_wandb else 'online'
    if resume:
        wandb_run = _wandb_from_checkpoint(cfg, checkpointer, mode)
    else:
        wandb_run = _wandb_from_config(cfg, timestamp, mode)
    checkpointer.wandb_id = wandb_run.id
    return wandb_run

def _wandb_from_config(cfg, timestamp, mode):
    kwargs = dict(
        name = f'{cfg.EXPERIMENT.NAME}-{timestamp}',
        group = cfg.EXPERIMENT.GROUP,
        notes = cfg.EXPERIMENT.NOTES,
        dir = osp.join(cfg.OUTPUT_DIR),
        project = project,
        entity = entity,
        resume = 'never',
        config = cfg,
        mode = mode
    )
    return wandb.init(**kwargs)

def _wandb_from_checkpoint(cfg, checkpoint, mode):
    assert checkpoint.wandb_id
    kwargs = dict(
        id = checkpoint.wandb_id,
        dir = osp.join(cfg.OUTPUT_DIR),
        project = project,
        entity = entity,
        resume = 'must',
        config = cfg,
        mode = mode
    )
    return wandb.init(**kwargs)
