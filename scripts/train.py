import torch
import random
import numpy as np

from parsing.config import cfg
from parsing.utils.comm import to_device
from parsing.dataset import build_train_dataset, build_gnn_train_dataset, move_datasets
from parsing.detector import WireframeDetector
from parsing.plane_from_gt_detector import PlaneFromGtDetector
from parsing.modules.gnn import WireframeGNNClassifier
from parsing.solver import make_lr_scheduler, make_optimizer
from parsing.utils.logger import setup_logger, wandb_init, setup_noop_logger
from parsing.utils.miscellaneous import save_config
from parsing.utils.checkpoint import DetectronCheckpointer
import os
import os.path as osp
import time
from datetime import datetime, timedelta
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
import sys
from test import ModelTester

# Multi GPU support
import parsing.utils.multi_gpu as multi_gpu
import torch.multiprocessing as mp




class LossReducer(object):
    def __init__(self,cfg):
        # self.loss_keys = cfg.MODEL.LOSS_WEIGHTS.keys()
        self.loss_weights = dict(cfg.MODEL.LOSS_WEIGHTS)

    def __call__(self, loss_dict):
        total_loss = sum([self.loss_weights[k]*loss_dict[k]
        for k in self.loss_weights.keys()])

        return total_loss

    def apply_weight(self, loss_dict):
        return {k:self.loss_weights[k]*v for k,v in loss_dict.items()}


def train(rank, cfg, resume=False, timestamp = '', disable_wandb = False, test_queue = None):

    multi_gpu.setup(rank,cfg.SOLVER.NUM_GPUS)

    logger = multi_gpu.setup_logger(cfg.OUTPUT_DIR, timestamp)
    tb_logger = SummaryWriter(cfg.OUTPUT_DIR) if rank == 0 else None

    device = multi_gpu.get_device(cfg)
    val_period = cfg.SOLVER.VAL_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD



    # torch.cuda.set_device(device)

    if 'GNN' in cfg.MODEL.NAME.upper():
        model = WireframeGNNClassifier(cfg)
        train_dataset = build_gnn_train_dataset(cfg)
    elif 'GT_PLANE_CLASSIFIER' == cfg.MODEL.NAME.upper():
        model = PlaneFromGtDetector(cfg)
        train_dataset = build_train_dataset(cfg)
    elif 'HOURGLASS' == cfg.MODEL.NAME.upper():
        model = WireframeDetector(cfg)
        train_dataset = build_train_dataset(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.MODEL.NAME} not supported')

    model_bare = model.to(device)

    # For multi gpu
    model = multi_gpu.wrap_model(model_bare)

    optimizer = make_optimizer(cfg,model)
    scheduler = make_lr_scheduler(cfg,optimizer)

    loss_reducer = LossReducer(cfg)

    checkpointer = DetectronCheckpointer(cfg,
                                         model_bare,
                                         optimizer,
                                         scheduler = scheduler,
                                         save_dir=cfg.OUTPUT_DIR,
                                         save_to_disk=True,
                                         logger=logger)

    if resume:
        checkpointer.load()
        scheduler.step()
    elif getattr(cfg, 'CHECKPOINT', None):
        transfer = getattr(cfg, 'TRANSFER_LEARN', False)
        checkpointer.load(cfg.CHECKPOINT, use_latest=False, transfer = transfer)

    if rank == 0:
        wandb_run = wandb_init(cfg, checkpointer, resume=resume, timestamp = timestamp, disable_wandb = disable_wandb)
    else:
        wandb_run = None

    if val_period:
        val_dir = osp.join(cfg.OUTPUT_DIR, 'val')
        os.makedirs(val_dir, exist_ok=True)
        tester = ModelTester(cfg, validation=True, output_dir = val_dir, wandb_run = wandb_run, result_queue=test_queue)

    start_training_time = time.time()
    end = time.time()

    start_epoch = scheduler.last_epoch if resume else 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    epoch_size = len(train_dataset)

    global_iteration = epoch_size*start_epoch
    meters = multi_gpu.MetricLogger(" ")

    for epoch in range(start_epoch, max_epoch):
        model.train()
        multi_gpu.set_epoch(train_dataset, epoch)

        if rank == 0:
            wandb_run.log({'epoch': epoch, 'lr': optimizer.param_groups[0]["lr"]})


        for it, (images, annotations) in enumerate(train_dataset):
            data_time = time.time() - end
            images = to_device(images, device)
            annotations = to_device(annotations,device)
            loss_dict, timing_info = model(images,annotations)
            total_loss = loss_reducer(loss_dict)

            with torch.no_grad():
                weighted_losses = loss_reducer.apply_weight(loss_dict)
                loss_dict_reduced = {k:v.item() for k,v in loss_dict.items()}
                loss_reduced = total_loss.item()
                loss_dict_ratio = {'ratio_{}'.format(k):v.item()/(loss_reduced + 1e-10) for k,v in weighted_losses.items()}
                meters.update(loss=loss_reduced, **loss_dict_reduced, **loss_dict_ratio, **timing_info)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            global_iteration +=1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_batch = epoch_size*(max_epoch-epoch+1) - it +1
            eta_seconds = meters.time.global_avg*eta_batch
            eta_string = str(timedelta(seconds=int(eta_seconds)))

            if (it % 20 == 0 or it+1 == len(train_dataset)):
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        iter=it,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                meters.tensorboard(it + epoch*epoch_size, tb_logger, phase='train_batch')
                meters.wandb(epoch, it, wandb_run)


        if rank == 0 and checkpoint_period and not (epoch%checkpoint_period):
            checkpointer.save('model_{:05d}'.format(epoch))

        multi_gpu.barrier()
        scheduler.step()

        meters.tensorboard(epoch, tb_logger, phase='train')

        #Run validation
        if val_period and epoch > 0 and not (epoch%val_period):
            try:
                tester.test_model(model, epoch=epoch)
            except KeyboardInterrupt:
                raise
            except:
                logger.exception('Could not run validation for epoch {}'.format(epoch))

    total_training_time = time.time() - start_training_time
    total_time_str = str(timedelta(seconds=total_training_time))

    logger.info(
        "Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / (max_epoch)
        )
    )
    multi_gpu.barrier()
    multi_gpu.cleanup()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Layout HAWP Training')

    parser.add_argument("--config-file",
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        required=True
                        )
    parser.add_argument("--seed",
                        default=2,
                        type=int)
    parser.add_argument("--disable-wandb",
                        action='store_true')
    parser.add_argument("--disable-cudnn",
                        action='store_true')
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )


    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    resume_training = osp.exists(osp.join(cfg.OUTPUT_DIR, 'last_checkpoint'))
    timestamp = datetime.now().isoformat(timespec='seconds')
    if not resume_training:
        output_path = cfg.OUTPUT_DIR
        if cfg.EXPERIMENT.GROUP:
            output_path = osp.join(output_path, cfg.EXPERIMENT.GROUP)
        if cfg.EXPERIMENT.NAME:
            output_path = osp.join(output_path, cfg.EXPERIMENT.NAME)
        #Create timestamped folder in output dir
        cfg.OUTPUT_DIR = osp.join(output_path, timestamp)
        os.makedirs(cfg.OUTPUT_DIR)

    cfg.freeze()

    logger = setup_logger('srw', cfg.OUTPUT_DIR, out_file='train-{}.log'.format(timestamp))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file,"r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)

    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config-{}.yml'.format(timestamp))
    logger.info("Saving config into: {}".format(output_config_path))

    save_config(cfg, output_config_path)

    # Will move datasets if DATASET.TMP_DIR is specified in the config
    move_datasets(cfg, logger)

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    else:
        torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # train(cfg,
    #       resume = resume_training,
    #       timestamp = timestamp,
    #       disable_wandb = args.disable_wandb)
    multi_gpu.setup_context()
    test_queue = ModelTester.setup_queue()
    train_args = (cfg, resume_training, timestamp, args.disable_wandb, test_queue)

    if cfg.SOLVER.NUM_GPUS > 1:
        mp.spawn(
        train,
        args=train_args,
        nprocs=cfg.SOLVER.NUM_GPUS
        )
    else:
        train(0, *train_args)
