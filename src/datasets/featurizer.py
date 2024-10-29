import torch
import numpy as np
import random
import itertools
import torch.nn.functional as F
import math
import torch_geometric
import torch_cluster
from collections.abc import Mapping, Sequence
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from sklearn.neighbors import NearestNeighbors
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32


def pad_ss_connections(ss_connections, max_residues, max_surface_atoms):
    """ Pad ss_connections to the maximum number of residues and surface atoms in the batch """
    B = len(ss_connections)
    ss_connections_padded = torch.ones((B, max_residues, max_surface_atoms), dtype=torch.float32)
    for i, ss_connection in enumerate(ss_connections):
        ss_connections_padded[i, :ss_connection.shape[0], :ss_connection.shape[1]] = ss_connection
    return ss_connections_padded


def featurize_SBC2Model(batch):
    """ Pack and pad batch into torch tensors with surface and orig_surface downsampling to the minimum size """
    batch = [one for one in batch if one is not None]
    B = len(batch)
    if B == 0:
        return None

    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max(lengths)
    
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max]) - 1  # 1:需要被预测的掩码部分 0:可见部分
    chain_encoding = np.zeros([B, L_max]) - 1
    
    # Build the batch
    surfaces = []
    features = []
    orig_surfaces = []
    surface_lengths = []
    ss_connections = []
    correspondences = []
    
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1)  # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))  # [#atom, 4, 3]
        X[i, :, :, :] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        S[i, :l] = indices
        chain_mask[i, :l] = b['chain_mask']
        chain_encoding[i, :l] = b['chain_encoding']

        # Add surface, features, orig_surface
        surfaces.append(torch.tensor(b['surface'], dtype=torch.float32))
        features.append(torch.tensor(b['features'], dtype=torch.float32))
        orig_surfaces.append(torch.tensor(b['orig_surface'], dtype=torch.float32))
        surface_lengths.append(b['surface'].shape[0])

    # Find the minimum surface length in the batch
    min_surface_length = min(surface_lengths)

    # Downsample all surfaces, features, and orig_surfaces to the minimum surface length
    surfaces_downsampled = []
    features_downsampled = []
    orig_surfaces_downsampled = []
    
    for i, surface in enumerate(surfaces):
        surface_len = surface.shape[0]
        if surface_len > min_surface_length:
            # Randomly sample indices without replacement
            sampled_indices = random.sample(range(surface_len), min_surface_length)
            surfaces_downsampled.append(surface[sampled_indices])
            features_downsampled.append(features[i][sampled_indices])
            orig_surfaces_downsampled.append(orig_surfaces[i][sampled_indices])
        else:
            surfaces_downsampled.append(surface)
            features_downsampled.append(features[i])
            orig_surfaces_downsampled.append(orig_surfaces[i])

    # Stack the downsampled surfaces, features, and orig_surfaces
    surfaces_stacked = torch.stack(surfaces_downsampled, dim=0)
    features_stacked = torch.stack(features_downsampled, dim=0)
    orig_surfaces_stacked = torch.stack(orig_surfaces_downsampled, dim=0)

    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)  # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X) + np.nan

    for i, n in enumerate(numbers):
        X_new[i, :n, ::] = X[i][mask[i] == 1]
        S_new[i, :n] = S[i][mask[i] == 1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Calculate ss_connection based on X_new and downsampled orig_surface
    for i in range(B):
        ca_coords = X[i, :, 1, :]  # Extract CA coordinates from X_new (1 is for CA atom)
        surface_coords = orig_surfaces_stacked[i]
        
        # Use the mask to identify valid indices
        valid_indices = mask[i].astype(bool)  # mask[i] is 1 for valid indices, 0 otherwise
        valid_ca_coords = ca_coords[valid_indices]
        
        # Nearest neighbors search
        n_neighbors = max(1, int(8 * 175 / lengths[i]))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(surface_coords)
        # nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(surface_coords)
        distances, indices = nbrs.kneighbors(valid_ca_coords)
        
        ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
        
        # Fill ss_connection for valid CA coordinates
        for j, neighbors in zip(np.where(valid_indices)[0], indices):
            ss_connection[j, neighbors] = 1
        
        # Fill ss_connection for invalid CA coordinates
        ss_connection[~valid_indices, :] = 1
        
        ss_connections.append(torch.tensor(ss_connection, dtype=torch.float32))

        # 1. Calculate the distance matrix for valid_ca_coords
        ca_dist_matrix = np.linalg.norm(valid_ca_coords[:, None, :] - valid_ca_coords[None, :, :], axis=-1)
        max_dist = np.max(ca_dist_matrix)
        r = max_dist / 3  # 1/3 of max distance as radius
        
        # 2. Randomly sample 8 coords from valid_ca_coords
        sampled_indices = random.sample(range(valid_ca_coords.shape[0]), min(8, valid_ca_coords.shape[0]))
        
        batch_correspondences = []
        for sampled_idx in sampled_indices:
            # Get indices of CA atoms within radius r
            ca_neighbors = np.where(ca_dist_matrix[sampled_idx] < r)[0]
            
            # Get distances between the sampled CA atom and surface points
            ca_surface_dist_matrix = np.linalg.norm(valid_ca_coords[sampled_idx] - surface_coords.numpy(), axis=-1)
            
            # Get indices of surface points within radius r
            surface_neighbors = np.where(ca_surface_dist_matrix < r)[0]
            
            # Store the two sets of indices as tensors
            batch_correspondences.append([
                torch.tensor(ca_neighbors, dtype=torch.long),
                torch.tensor(surface_neighbors, dtype=torch.long)
            ])
        
        correspondences.append(batch_correspondences)


    # Pad ss_connections
    ss_connections_padded = pad_ss_connections(ss_connections, L_max, min_surface_length)

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    
    return {
        "title": [b['title'] for b in batch],
        "X": X,
        "S": S,
        "score": score,
        "mask": mask,
        "lengths": lengths,
        "chain_mask": chain_mask,
        "chain_encoding": chain_encoding,
        "surface": surfaces_stacked,
        "features": features_stacked,
        'ss_connection': ss_connections_padded,
        'correspondences': correspondences,
    }