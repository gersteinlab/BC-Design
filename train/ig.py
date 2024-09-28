import datetime
import os
import sys
sys.path.append(os.getcwd())
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
from tqdm import tqdm
import pytorch_lightning as pl
import copy
from transformers import AutoTokenizer
import pickle

# Add the current directory to the system path
sys.path.append(os.getcwd())

# Set the CUDA environment if needed
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Ensure correct GPUs are used

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32

residue_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./train/results', type=str)
    parser.add_argument('--ex_name', default='SBC2-sum3-minlrdiv1-bs4-lr00002-epoch20-test', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    # parser.add_argument('--dataset', default='CATH4.2SurfProPiFoldDense')
    parser.add_argument('--dataset', default='TS50')
    # parser.add_argument('--dataset', default='TS500')
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
def load_model_checkpoint(model, checkpoint_path, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Load state dict into the model
    model.load_state_dict(checkpoint['state_dict'])  # 'state_dict' key might differ, verify in checkpoint
    return model

def create_baseline_batch(batch):
    baseline_batch = copy.deepcopy(batch)
    baseline_batch['_V'] = torch.randn_like(batch['_V'])
    baseline_batch['_E'] = torch.randn_like(batch['_E'])
    baseline_batch['features'] = torch.randn_like(batch['features'])
    return baseline_batch


def individual_integrated_gradients(model, batch, steps=50):
    batch_id, E_idx = batch['batch_id'], batch['E_idx']
    B = len(batch_id.unique())  # Batch size
    model.eval()  # Ensure the model is in evaluation mode

    with torch.no_grad():
        orig_results = model(batch)
        orig_log_probs = orig_results['log_probs']
        pred_indices = orig_log_probs.argmax(dim=-1)
    
    # Prepare baseline (e.g., zeros)
    baseline_batch = create_baseline_batch(batch)  # Ensure same device

    # Initialize a dict to store results
    integrated_grads_dict = {}

    # Initialize lists to accumulate gradients over steps
    grads_V_accum = {}
    grads_E_accum = {}
    grads_biochem_accum = {}

    # Loop through each predicted index (token level)
    for j in range(B):
        node_indices = (batch_id == j).nonzero(as_tuple=True)[0]
        title = batch['title'][j]  # Unique title for each sample
        integrated_grads_dict[title] = {'grads': []}
        grads_V_accum[title] = [[] for _ in range(len(node_indices))]  # One list per token
        grads_E_accum[title] = [[] for _ in range(len(node_indices))]
        grads_biochem_accum[title] = [[] for _ in range(len(node_indices))]

    # Loop through each interpolation step
    for step in tqdm(range(steps + 1)):
        alpha = float(step) / steps
        scaled_batch = copy.deepcopy(baseline_batch)

        with torch.enable_grad():
            # Interpolate between baseline and actual inputs
            scaled_batch['_V'] = (baseline_batch['_V'] + alpha * (batch['_V'] - baseline_batch['_V'])).requires_grad_(True)
            scaled_batch['_E'] = (baseline_batch['_E'] + alpha * (batch['_E'] - baseline_batch['_E'])).requires_grad_(True)
            scaled_batch['features'] = (baseline_batch['features'] + alpha * (batch['features'] - baseline_batch['features'])).requires_grad_(True)
            # Forward pass for scaled input
            results = model(scaled_batch)
            log_probs = results['log_probs']

            # Iterate over each token and sample to compute gradients
            for i, pred_index in enumerate(pred_indices):
                target_output = log_probs[i, pred_index]

                # Compute gradients for this token
                model.zero_grad()
                target_output.backward(retain_graph=True)

                for j in range(B):
                    node_indices = (batch_id == j).nonzero(as_tuple=True)[0]  # Get indices for the current sample (nodes)
                    min_node_id = node_indices.min().item()
                    src, dst = E_idx[0, :], E_idx[1, :]
                    local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

                    # Accumulate gradients for _V (nodes)
                    grad_V = scaled_batch['_V'].grad[node_indices].data.clone().cpu()
                    grads_V_accum[batch['title'][j]][i - min_node_id].append(grad_V)

                    # Accumulate gradients for _E (edges)
                    grad_E = scaled_batch['_E'].grad[local_edges_mask].data.clone().cpu()
                    grads_E_accum[batch['title'][j]][i - min_node_id].append(grad_E)

                    # Accumulate gradients for biochem (biochemical features)
                    grad_biochem = scaled_batch['features'].grad[j].data.clone().cpu()
                    grads_biochem_accum[batch['title'][j]][i - min_node_id].append(grad_biochem)

    # After accumulating over all steps, calculate the Integrated Gradients
    for title in integrated_grads_dict.keys():
        integrated_grads_V = []
        integrated_grads_E = []
        integrated_grads_biochem = []

        # Compute average gradients for each token
        for k in range(len(grads_V_accum[title])):
            integrated_grads_V.append(torch.stack(grads_V_accum[title][k]).mean(dim=0).cpu().numpy())
        
        # Compute average gradients for each edge
        for k in range(len(grads_E_accum[title])):
            integrated_grads_E.append(torch.stack(grads_E_accum[title][k]).mean(dim=0).cpu().numpy())

        # Compute average gradients for each biochemical feature
        for k in range(len(grads_biochem_accum[title])):
            integrated_grads_biochem.append(torch.stack(grads_biochem_accum[title][k]).mean(dim=0).cpu().numpy())

        # Store the integrated gradients for this sample
        token_integrated_grads = {
            '_V': integrated_grads_V,
            '_E': integrated_grads_E,
            'biochem': integrated_grads_biochem
        }

        integrated_grads_dict[title]['grads'] = token_integrated_grads

    return integrated_grads_dict


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
    grads_V_accum = {}
    grads_E_accum = {}
    grads_biochem_accum = {}

    # Loop through each predicted index (token level)
    for j in range(B):
        node_indices = (batch_id == j).nonzero(as_tuple=True)[0]
        title = batch['title'][j]  # Unique title for each sample
        integrated_grads_dict[title] = {'grads': []}
        grads_V_accum[title] = {res: [] for res in residue_types + ['overall']}  # One list per token
        grads_E_accum[title] = {res: [] for res in residue_types + ['overall']}
        grads_biochem_accum[title] = {res: [] for res in residue_types + ['overall']}

    # Loop through each interpolation step
    for step in range(steps + 1):
        alpha = float(step) / steps
        scaled_batch = copy.deepcopy(baseline_batch)

        with torch.enable_grad():
            # Interpolate between baseline and actual inputs
            scaled_batch['_V'] = (baseline_batch['_V'] + alpha * (batch['_V'] - baseline_batch['_V'])).requires_grad_(True)
            scaled_batch['_E'] = (baseline_batch['_E'] + alpha * (batch['_E'] - baseline_batch['_E'])).requires_grad_(True)
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
                    node_indices = (batch_id == j).nonzero(as_tuple=True)[0]  # Get indices for the current sample (nodes)
                    min_node_id = node_indices.min().item()
                    src, dst = E_idx[0, :], E_idx[1, :]
                    local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

                    # Accumulate gradients for _V (nodes)
                    grad_V = scaled_batch['_V'].grad[node_indices].data.clone().cpu()
                    grads_V_accum[batch['title'][j]][residue].append(grad_V)

                    # Accumulate gradients for _E (edges)
                    grad_E = scaled_batch['_E'].grad[local_edges_mask].data.clone().cpu()
                    grads_E_accum[batch['title'][j]][residue].append(grad_E)

                    # Accumulate gradients for biochem (biochemical features)
                    grad_biochem = scaled_batch['features'].grad[j].data.clone().cpu()
                    grads_biochem_accum[batch['title'][j]][residue].append(grad_biochem)

            # Sum or mean the log probabilities for all tokens of this residue type
            target_output = log_probs.mean()  # Use .mean() if you want to average

            # Compute gradients for this token
            model.zero_grad()
            target_output.backward(retain_graph=True)

            for j in range(B):
                node_indices = (batch_id == j).nonzero(as_tuple=True)[0]  # Get indices for the current sample (nodes)
                min_node_id = node_indices.min().item()
                src, dst = E_idx[0, :], E_idx[1, :]
                local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

                # Accumulate gradients for _V (nodes)
                grad_V = scaled_batch['_V'].grad[node_indices].data.clone().cpu()
                grads_V_accum[batch['title'][j]]['overall'].append(grad_V)

                # Accumulate gradients for _E (edges)
                grad_E = scaled_batch['_E'].grad[local_edges_mask].data.clone().cpu()
                grads_E_accum[batch['title'][j]]['overall'].append(grad_E)

                # Accumulate gradients for biochem (biochemical features)
                grad_biochem = scaled_batch['features'].grad[j].data.clone().cpu()
                grads_biochem_accum[batch['title'][j]]['overall'].append(grad_biochem)

    # After accumulating over all steps, calculate the Integrated Gradients
    for title in integrated_grads_dict.keys():
        integrated_grads_V = {}
        integrated_grads_E = {}
        integrated_grads_biochem = {}

        # Compute average gradients for each token
        for k in grads_V_accum[title]:
            if grads_V_accum[title][k]:
                integrated_grads_V[k] = torch.stack(grads_V_accum[title][k]).mean(dim=0).cpu().numpy()
        
        # Compute average gradients for each edge
        for k in grads_E_accum[title]:
            if grads_E_accum[title][k]:
                integrated_grads_E[k] = torch.stack(grads_E_accum[title][k]).mean(dim=0).cpu().numpy()

        # Compute average gradients for each biochemical feature
        for k in grads_biochem_accum[title]:
            if grads_biochem_accum[title][k]:
                integrated_grads_biochem[k] = torch.stack(grads_biochem_accum[title][k]).mean(dim=0).cpu().numpy()

        # Store the integrated gradients for this sample
        token_integrated_grads = {
            '_V': integrated_grads_V,
            '_E': integrated_grads_E,
            'biochem': integrated_grads_biochem
        }

        integrated_grads_dict[title]['grads'] = token_integrated_grads

    return integrated_grads_dict


# Main execution logic
if __name__ == "__main__":
    args = create_parser()
    ig_save_directory = f"ig_results/{args.ex_name}/{args.dataset}"
    if not os.path.exists(ig_save_directory):
        os.makedirs(ig_save_directory)
    
    # Seed for reproducibility
    pl.seed_everything(args.seed)

    # Initialize data module and setup test data
    data_module = DInterface(**vars(args))
    data_module.setup()  # Ensure the test dataset is loaded

    # Initialize the model
    model = MInterface(**vars(args))
    
    # Load checkpoint if provided
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(model, args.checkpoint_path, device)
    model.to(device)  # Move model to GPU if available
    
    # Test with integrated gradients
    dataloader = data_module.test_dataloader()  # Get the test dataloader
    model.eval()  # Set model to evaluation mode
    for batch in tqdm(dataloader):
        batch = model.model._get_features(batch)
        integrated_grads_dict = integrated_gradients(model.model, batch, steps=1)

        batch_id, E_idx = batch['batch_id'], batch['E_idx']
        B = len(batch_id.unique())  # Batch size
        for j, title in enumerate(integrated_grads_dict):
            node_indices = (batch_id == j).nonzero(as_tuple=True)[0]
            min_node_id = node_indices.min().item()
            src, dst = E_idx[0, :], E_idx[1, :]
            local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

            integrated_grads_dict[title]['X'] = batch['X'][node_indices, 1, :].cpu().numpy()
            integrated_grads_dict[title]['E_idx'] = E_idx[:, local_edges_mask].cpu().numpy()
            integrated_grads_dict[title]['surface'] = batch['surface'][j].cpu().numpy()

            ig_path = os.path.join(ig_save_directory, f"{title}.pkl")
            with open(ig_path, 'wb') as f:
                pickle.dump(integrated_grads_dict[title], f)