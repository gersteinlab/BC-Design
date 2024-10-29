import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'  # or the master node IP
os.environ['MASTER_PORT'] = '12355'

import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from model_interface import MInterface
from data_interface import DInterface, MyDataLoader
from src.tools.logger import SetupCallback, BackupCodeCallback
import math
from shutil import ignore_patterns
from tqdm import tqdm
import pytorch_lightning as pl
import copy
from transformers import AutoTokenizer
import pickle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import DataLoader, DistributedSampler

# Add the current directory to the system path
sys.path.append(os.getcwd())

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32

residue_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='SBC2-sum3-minlrdiv1-bs4-lr00002-epoch20-test', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    # parser.add_argument('--dataset', default='CATH4.2SurfProPiFoldDense')
    # parser.add_argument('--dataset', default='TS50')
    # parser.add_argument('--dataset', default='TS500')
    parser.add_argument('--dataset', default='AFDB2000')
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
    parser.add_argument('--batch_size', default=1, type=int)
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
    parser.add_argument('--checkpoint_path', default='./train/results/SBC2-sum3-minlrdiv1-bs4-lr00002-epoch10/checkpoints/best-epoch=16-recovery=0.877.ckpt', type=str, help='Path to a checkpoint to resume testing')

    args = parser.parse_args()
    return args

# Function to load the model checkpoint
def load_model_checkpoint(model, checkpoint_path, rank):
    # Set map_location to the correct device (GPU) for the current process
    device = torch.device(f'cuda:{rank}')  # Use the correct GPU for the current rank

    # Load the checkpoint and map it to the correct device
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle case where model has 'module.' prefix due to multi-GPU wrapping
    state_dict = checkpoint['state_dict']
    # If the keys don't have 'module.' prefix and we're using DDP, add the prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            new_state_dict['module.' + k] = v
        else:
            new_state_dict[k] = v

    # Load state dict into the model
    model.load_state_dict(new_state_dict)
    return model


def create_baseline_batch(batch):
    baseline_batch = copy.deepcopy(batch)
    baseline_batch['_V'] = torch.randn_like(batch['_V'])
    baseline_batch['_E'] = torch.randn_like(batch['_E'])
    baseline_batch['features'] = torch.randn_like(batch['features'])
    return baseline_batch



def integrated_gradients(model, batch, steps=50):
    batch_id, E_idx = batch['batch_id'], batch['E_idx']
    B = len(batch_id.unique())  # Batch size
    model.eval()  # Ensure the model is in evaluation mode

    with torch.no_grad():
        orig_results = model(batch)
        orig_log_probs = orig_results['log_probs']
        pred_indices = orig_log_probs.argmax(dim=-1)

        # Convert pred_indices to residue types using the tokenizer
    residue_tokens = tokenizer.convert_ids_to_tokens(pred_indices, skip_special_tokens=True)

    # Group tokens by their residue types
    residue_groups = {res: [] for res in residue_types}
    for i, residue in enumerate(residue_tokens):
        if residue in residue_types:
            residue_groups[residue].append(i)
    
    # Prepare baseline (e.g., zeros)
    baseline_batch = create_baseline_batch(batch)  # Ensure same device

    # Initialize a dict to store results
    integrated_grads_dict = {}

    # Initialize lists to accumulate gradients over steps
    grads_biochem_accum = {}

    # Loop through each predicted index (token level)
    for j in range(B):
        node_indices = (batch_id == j).nonzero(as_tuple=True)[0]
        title = batch['title'][j]  # Unique title for each sample
        integrated_grads_dict[title] = {'grads': []}
        grads_biochem_accum[title] = {res: [] for res in residue_types + ['overall']}

    # Loop through each interpolation step
    for step in range(steps + 1):
        alpha = float(step) / steps
        scaled_batch = copy.deepcopy(baseline_batch)

        with torch.enable_grad():
            # Interpolate between baseline and actual inputs
            scaled_batch['features'] = (baseline_batch['features'] + alpha * (batch['features'] - baseline_batch['features'])).requires_grad_(True)
            # Forward pass for scaled input
            results = model(scaled_batch)
            log_probs = results['log_probs']

            # Compute gradients for each unique residue type
            for residue, indices in residue_groups.items():
                if not indices:
                    continue
                
                # Sum or mean the log probabilities for all tokens of this residue type
                target_output = log_probs[indices].mean()  # Use .mean() if you want to average

                # Compute gradients for this token
                model.zero_grad()
                target_output.backward(retain_graph=True)

                for j in range(B):
                    # Accumulate gradients for biochem (biochemical features)
                    grad_biochem = scaled_batch['features'].grad[j].data.clone().cpu()
                    grads_biochem_accum[batch['title'][j]][residue].append(grad_biochem)

            # Sum or mean the log probabilities for all tokens of this residue type
            target_output = log_probs.mean()  # Use .mean() if you want to average

            # Compute gradients for this token
            model.zero_grad()
            target_output.backward(retain_graph=True)

            for j in range(B):
                # Accumulate gradients for biochem (biochemical features)
                grad_biochem = scaled_batch['features'].grad[j].data.clone().cpu()
                grads_biochem_accum[batch['title'][j]]['overall'].append(grad_biochem)

    # After accumulating over all steps, calculate the Integrated Gradients
    for title in integrated_grads_dict.keys():
        integrated_grads_biochem = {}

        # Compute average gradients for each biochemical feature
        for k in grads_biochem_accum[title]:
            if grads_biochem_accum[title][k]:
                integrated_grads_biochem[k] = torch.stack(grads_biochem_accum[title][k]).mean(dim=0).cpu().numpy()

        # Store the integrated gradients for this sample
        token_integrated_grads = {
            'biochem': integrated_grads_biochem
        }

        integrated_grads_dict[title]['grads'] = token_integrated_grads

    return integrated_grads_dict


def setup_ddp(rank, world_size, args):
    ig_save_directory = f"ig_biochem_results_steps50/{args.ex_name}/{args.dataset}"
    if not os.path.exists(ig_save_directory):
        os.makedirs(ig_save_directory)
    # Initialize the process group for DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set the GPU to use

    # Initialize your model and move it to the correct GPU
    model = MInterface(**vars(args)).to(rank)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    # Load checkpoint if provided
    model = load_model_checkpoint(model, args.checkpoint_path, rank)

    # Initialize the dataloader
    data_module = DInterface(**vars(args))
    data_module.setup()

    # Get the dataset from the test DataLoader
    test_dataloader = data_module.test_dataloader()
    dataset = test_dataloader.dataset  # Extract the dataset

    # Use DistributedSampler for DDP
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    # Recreate the DataLoader using the original parameters, including the collate function
    dataloader = MyDataLoader(
        dataset,
        model_name=test_dataloader.model_name,
        batch_size=args.batch_size,  # Use batch size from args
        sampler=sampler,             # Apply DistributedSampler
        num_workers=args.num_workers, # Use number of workers from args
        pin_memory=True,             # Use pin memory
        collate_fn=test_dataloader.collate_fn  # Reuse the original collate function
    )

    # Test with integrated gradients
    model.eval()
    for batch in tqdm(dataloader):
        # batch = model.model._get_features(batch)
        batch = model.module.model._get_features(batch)

        integrated_grads_dict = integrated_gradients(model.module.model, batch, steps=50)

        batch_id, E_idx = batch['batch_id'], batch['E_idx']
        B = len(batch_id.unique())  # Batch size
        for j, title in enumerate(integrated_grads_dict):
            node_indices = (batch_id == j).nonzero(as_tuple=True)[0]
            integrated_grads_dict[title]['surface'] = batch['surface'][j].cpu().numpy()

            ig_path = os.path.join(ig_save_directory, f"{title}.pkl")
            with open(ig_path, 'wb') as f:
                pickle.dump(integrated_grads_dict[title], f)

    # Clean up DDP
    dist.destroy_process_group()


def run_ddp(args):
    world_size = torch.cuda.device_count()  # Get the number of available GPUs
    mp.spawn(setup_ddp, args=(world_size, args), nprocs=world_size, join=True)

# Main execution logic
if __name__ == "__main__":
    args = create_parser()
    # Run DDP if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with Distributed Data Parallel")
        run_ddp(args)
    else:
        # Fallback to single-GPU or CPU execution
        setup_ddp(0, 1, args)