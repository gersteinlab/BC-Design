import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['NCCL_P2P_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from model_interface import MInterface
from data_interface import DInterface

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
torch.autograd.set_detect_anomaly(True)

def create_parser():
    checkpoint_path = './train/results/ncs-revision-ubc2-fullfinaltrain-strucmask0.1-bcmaskmax1.0-bs2-lr0.00002-epoch50-encoder12-bc2/checkpoints/last.ckpt'
    ex_name = 'UBC2Model'
    batch_size = 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default=ex_name, type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--dataset', default='CATH4.2')
    parser.add_argument('--model_name', default='UBC2Model',
                        choices=['UBC2Model', 'SBC2Model'])
    parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('--lr_scheduler', default='onecycle')
    parser.add_argument('--offline', default=0, type=int)
    parser.add_argument('--seed', default=111, type=int)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pad', default=1024, type=int)
    parser.add_argument('--min_length', default=40, type=int)
    parser.add_argument('--data_root', default='./data/')
    
    # Testing specific parameters
    parser.add_argument('--epoch', default=20, type=int, help='end epoch')
    parser.add_argument('--augment_eps', default=0.0, type=float, help='noise level')

    # Model parameters
    parser.add_argument('--use_dist', default=1, type=int)
    parser.add_argument('--use_product', default=0, type=int)

    # Checkpoint parameter
    parser.add_argument('--checkpoint_path', default=checkpoint_path, type=str, help='Path to a checkpoint to resume testing')

    parser.add_argument('--contrastive_pretrain', default=False, type=bool)

    args = parser.parse_args()
    return args

def load_callbacks(args):
    callbacks = []
    return callbacks

if __name__ == "__main__":
    args = create_parser()
    pl.seed_everything(args.seed)

    # Initialize data module and setup test data
    data_module = DInterface(**vars(args))
    data_module.setup()  # Ensure the test dataset is loaded

    gpu_count = 1
    print(f"Using {gpu_count} GPUs for testing")

    # Initialize the model
    model = MInterface(**vars(args))

    # Trainer configuration
    trainer_config = {
        'devices': gpu_count,
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp_find_unused_parameters_true',
        'precision': 32,
        'accelerator': 'gpu',
        'callbacks': load_callbacks(args),
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer_dict = vars(trainer_opt)
    trainer = Trainer(**trainer_dict)
    # model.custom_test()
    # Perform testing
    if args.checkpoint_path:
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        trainer.test(model, datamodule=data_module, ckpt_path=args.checkpoint_path)
    else:
        print("No checkpoint provided, testing with current model state")
    
    print(trainer_config)
