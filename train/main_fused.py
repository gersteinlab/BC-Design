import datetime
import os
import sys
sys.path.append(os.getcwd())
# Set environment variable for CUDA_LAUNCH_BLOCKING
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Set environment variable for device-side assertions
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# Set CUDA_VISIBLE_DEVICES to only use GPU 1 and 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ['NCCL_P2P_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from model_interface import MInterface
from data_interface import DInterface
from src.tools.logger import SetupCallback,BackupCodeCallback
import math
from shutil import ignore_patterns

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
torch.autograd.set_detect_anomaly(True)

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./train/results', type=str)
    # parser.add_argument('--ex_name', default='SurfProPiFold', type=str)
    parser.add_argument('--ex_name', default='BC-Design', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    
    parser.add_argument('--dataset', default='CATH4.2') # AF2DB_dataset, CATH_dataset
    parser.add_argument('--model_name', default='SBC2Model', 
        choices=['SBC2Model'])
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='onecycle')
    parser.add_argument('--offline', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='./data/')
    
    # Training parameters
    parser.add_argument('--epoch', default=20, type=int, help='end epoch')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')

    # Model parameters
    parser.add_argument('--use_dist', default=1, type=int)
    parser.add_argument('--use_product', default=0, type=int)

    # Checkpoint parameter
    parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to a checkpoint to resume training')

    args = parser.parse_args()
    return args




def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    
    callbacks.append(BackupCodeCallback(os.path.dirname(args.res_dir),logdir, ignore_patterns=ignore_patterns('results*', 'pdb*', 'metadata*', 'vq_dataset*')))
    

    metric = "recovery"
    sv_filename = 'best-{epoch:02d}-{recovery:.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=15,
        mode='max',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks



if __name__ == "__main__":
    args = create_parser()
    pl.seed_everything(args.seed)
    
    data_module = DInterface(**vars(args))
    data_module.setup()
    
    # gpu_count = torch.cuda.device_count()
    gpu_count = 4
    args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
    print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")

    model = MInterface(**vars(args))
    
    trainer_config = {
        'devices': gpu_count,
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp_find_unused_parameters_true',
        'precision': 32,
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': load_callbacks(args),
        'logger': plog.WandbLogger(
                    project = 'BC-Design',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = "_".join(args.ex_name.split("/")),
                    entity = "BC-Design"),
        'gradient_clip_val':1.0
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer_dict = vars(trainer_opt)
    trainer = Trainer(**trainer_dict)
    
    if args.checkpoint_path:
        trainer.fit(model, data_module, ckpt_path=args.checkpoint_path)
    else:
        trainer.fit(model, data_module)
    
    print(trainer_config)
