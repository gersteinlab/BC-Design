import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['NCCL_P2P_DISABLE'] = '1'

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from model_interface import MInterface
from data_interface import DInterface
from src.tools.logger import SetupCallback, BackupCodeCallback
import math
from shutil import ignore_patterns

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.strategies import DDPStrategy
torch.autograd.set_detect_anomaly(True)

def create_parser():
    # our best
    # checkpoint_path = './train/results/SBC2-sum3-minlrdiv1-bs4-lr00002-epoch10/checkpoints/best-epoch=16-recovery=0.877.ckpt'
    # ex_name = 'SBC2-sum3-minlrdiv1-bs4-lr00002-epoch20-test'
    # batch_size = 1



    # ablation
    # checkpoint_path = './train/results/ablat-ssc-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.386.ckpt'
    # ex_name = checkpoint_path.split('/')[3]
    # batch_size = 2

    # checkpoint_path = './train/results/ablat-e&v-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.836.ckpt'
    # ex_name = 'ablat-ev-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 1

    # checkpoint_path = './train/results/ablat-hydro&charge-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.322.ckpt'
    # ex_name = 'ablat-hydrocharge-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 1

    # checkpoint_path = './train/results/ablat-hydro-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=18-recovery=0.485.ckpt'
    # ex_name = 'ablat-hydro-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/ablat-charge-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.824.ckpt'
    # ex_name = 'ablat-charge-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 1

    checkpoint_path = './train/results/ablat-e-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.856.ckpt'
    ex_name = 'ablat-e-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    batch_size = 2

    # checkpoint_path = './train/results/ablat-v-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.848.ckpt'
    # ex_name = 'ablat-v-SBC2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2


    # hyperparam exp
    # gtlayers
    # checkpoint_path = './train/results/SBC2-gtlayers1-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.865.ckpt'
    # ex_name = 'SBC2-gtlayers1-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.870.ckpt'
    # ex_name = 'SBC2-gtlayers2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers4-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.871.ckpt'
    # ex_name = 'SBC2-gtlayers4-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers5-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.870.ckpt'
    # ex_name = 'SBC2-gtlayers5-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers6-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.869.ckpt'
    # ex_name = 'SBC2-gtlayers6-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers7-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.868.ckpt'
    # ex_name = 'SBC2-gtlayers7-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # mha
    # checkpoint_path = './train/results/SBC2-gtlayers3-mha2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.864.ckpt'
    # ex_name = 'SBC2-gtlayers3-mha2-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers3-mha3-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.868.ckpt'
    # ex_name = 'SBC2-gtlayers3-mha3-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # checkpoint_path = './train/results/SBC2-gtlayers3-mha4-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.841.ckpt'
    # ex_name = 'SBC2-gtlayers3-mha4-loss1,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2

    # loss weights
    # checkpoint_path = './train/results/SBC2-loss1,0.25,1-a100-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.873.ckpt'
    # ex_name = 'SBC2-loss1,0.25,1-a100-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2 

    # checkpoint_path = './train/results/SBC2-loss1,1,0.25-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.869.ckpt'
    # ex_name = 'SBC2-loss1,1,0.25-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2   

    # checkpoint_path = './train/results/SBC2-loss2,1,1-minlrdiv1c-bs4-lr00002-epoch20/checkpoints/best-epoch=19-recovery=0.872.ckpt'
    # ex_name = 'SBC2-loss2,1,1-minlrdiv1c-bs4-lr00002-epoch20'
    # batch_size = 2    

    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default=ex_name, type=str)
    # parser.add_argument('--ex_name', default='PiFold-test', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--dataset', default='CATH4.2SurfProPiFoldDense')
    # parser.add_argument('--dataset', default='TS50')
    # parser.add_argument('--dataset', default='TS500')
    # parser.add_argument('--dataset', default='CATH4.2')
    # parser.add_argument('--dataset', default='AFDB2000')
    parser.add_argument('--model_name', default='SBC2Model',
                        choices=['StructGNN', 'GraphTrans', 'GVP', 'GCA', 'AlphaDesign', 'ESMIF', 'PiFold', 
                                 'ProteinMPNN', 'KWDesign', 'E3PiFold', 'SurfProPiFold', 'SurfProPiFoldSurfaceOnly',
                                 'SurfProPiFoldDense', 'TestModel0831', 'TestModel0904', 'TestModel0907',
                                 'SBModel', 'SBCModel', 'SBC2Model'])
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
