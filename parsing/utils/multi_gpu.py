import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from parsing.utils.metric_logger import MetricLogger as pMetricLogger
from torch.nn.parallel import DistributedDataParallel as DDP
import parsing.utils.logger as srw_log
import os
import torch
import logging
import cupy


def setup_context():
    mp.set_sharing_strategy('file_system')
    mp.set_start_method('spawn', force=True)

def is_multiprocessing():
    return dist.is_initialized()

def num_gpus():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1

def gather_object(data):
    if not dist.is_initialized():
        return [data]

    data_list = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(data_list, data)
    return data_list

def is_main_proc():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def barrier():
    if dist.is_initialized():
        dist.barrier()

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def get_device(cfg):
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return cfg.MODEL.DEVICE

def setup(rank, world_size):
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{rank}'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        cupy.cuda.Device(rank).use()
        torch.cuda.set_device(rank)

def wrap_model(model_bare):
    if dist.is_initialized():
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_bare)
        # model = model_bare
        rank = dist.get_rank()
        model = DDP(model_bare, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    else:
        model = model_bare

    return model

def dataloader(dataset, world_size, **loader_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        sampler = DistributedSampler(dataset, drop_last=False, shuffle=loader_kwargs['shuffle'])
        loader_kwargs['shuffle'] = False
        # loader_kwargs['persistent_workers'] = True
        # loader_kwargs['pin_memory'] = False
        loader_kwargs['sampler'] = sampler
        # loader_kwargs['num_workers'] = 0
    else:
        loader_kwargs['sampler'] = None


    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader

LOGGER_PREFIX = 'srw'
def setup_logger(save_dir, timestamp):
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = srw_log.setup_logger(f'{LOGGER_PREFIX}-{rank}', save_dir,
                          out_file=f'train-{rank}-{timestamp}.log',
                          on_stdout = (rank == 0))
    return logger

def get_logger():
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = logging.getLogger(f'{LOGGER_PREFIX}-{rank}')
    return logger


class MetricLogger(pMetricLogger):
    def wandb(self, *args, **kwargs):
        if is_main_proc():
            super().wandb(*args, **kwargs)

    def tensorboard(self, *args, **kwargs):
        if is_main_proc():
            super().tensorboard(*args, **kwargs)

def set_epoch(dataloader, epoch):
    if dist.is_initialized():
        dataloader.sampler.set_epoch(epoch)

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
