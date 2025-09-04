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
from torch_geometric.nn.pool import knn_graph
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_sum
from transformers import AutoTokenizer
from sklearn.neighbors import NearestNeighbors
from src.tools import Rigid, Rotation, get_interact_feats
import copy
import json

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32



def pad_ss_connections(ss_connections, max_residues, max_surface_atoms):
    """ Pad ss_connections to the maximum number of residues and surface atoms in the batch """
    B = len(ss_connections)
    ss_connections_padded = torch.ones((B, max_residues, max_surface_atoms), dtype=torch.float32)
    for i, ss_connection in enumerate(ss_connections):
        ss_connections_padded[i, :ss_connection.shape[0], :ss_connection.shape[1]] = ss_connection
    return ss_connections_padded



def rbf(values, v_min, v_max, n_bins=16):
    """
    Returns RBF encodings in a new dimension at the end.
    """
    rbf_centers = torch.linspace(v_min, v_max, n_bins, device=values.device, dtype=values.dtype)
    rbf_centers = rbf_centers.view([1] * len(values.shape) + [-1])
    rbf_std = (v_max - v_min) / n_bins
    z = (values.unsqueeze(-1) - rbf_centers) / rbf_std
    return torch.exp(-z ** 2)


class MyTokenizer:
    def __init__(self):
        self.alphabet_protein = 'ACDEFGHIKLMNPQRSTVWY' # [X] for unknown token
        self.alphabet_RNA = 'AUGC'
    
    def encode(self, seq, RNA=False):
        if RNA:
            return [self.alphabet_RNA.index(s) for s in seq]
        else:
            return [self.alphabet_protein.index(s) for s in seq]
        
    def decode(self, indices, RNA=False):
        if RNA:
            return ' '.join([self.alphabet_RNA[i] for i in indices])
        else:
            return ' '.join([self.alphabet_protein[i] for i in indices])
        

class featurize_UBC2Model:
    def __init__(self, **kwargs) -> None:
        self.tokenizer = MyTokenizer()
        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
        self.virtual_frame_num = 3
        self.exp_backbone_noise_sd = kwargs.get('exp_backbone_noise_sd', 0.0)
        self.partial_design = kwargs.get('partial_design', False)
        self.design_region_path = kwargs.get('design_region_path', '')
        self.design_regions = None # Initialize as None
        self.ig_baseline_data = kwargs.get('ig_baseline_data', False)

        if self.partial_design:
            print(f"Partial design is enabled. Loading design regions from: {self.design_region_path}")
            try:
                with open(self.design_region_path, 'r') as f:
                    self.design_regions = json.load(f)
                print("Successfully loaded design regions.")
            except FileNotFoundError:
                print(f"⚠️ WARNING: Design region file not found at {self.design_region_path}. Partial design will be disabled.")
                self.partial_design = False
            except json.JSONDecodeError:
                print(f"⚠️ WARNING: Could not decode JSON from {self.design_region_path}. Partial design will be disabled.")
                self.partial_design = False

    def _get_features_persample(self, batch):
        # uniif struc featurizer
        for key in batch:
            try:
                batch[key] = batch[key][None,...]
            except:
                batch[key] = batch[key]
        S = []
        for seq in batch['seq']:
            S.extend(self.tokenizer.encode(seq))
        S = torch.tensor(S)
        
        X = torch.from_numpy(np.stack([np.concatenate(batch['N']),
                            np.concatenate(batch['CA']),
                            np.concatenate(batch['C']),
                            np.concatenate(batch['O'])], axis=1)).float()

        chain_mask = torch.from_numpy(np.concatenate(batch['chain_mask'])).float()
        chain_encoding = torch.from_numpy(np.concatenate(batch['chain_encoding'])).float()


        X, S = X.unsqueeze(0), S.unsqueeze(0)
        mask = torch.isfinite(torch.sum(X,(2,3))).float() # atom mask
        numbers = torch.sum(mask, axis=1).int()
        S_new = torch.zeros_like(S)
        X_new = torch.zeros_like(X)+torch.nan
        for i, n in enumerate(numbers):
            X_new[i,:n,::] = X[i][mask[i]==1]
            S_new[i,:n] = S[i][mask[i]==1]

        X = X_new
        S = S_new
        isnan = torch.isnan(X)
        mask = torch.isfinite(torch.sum(X,(2,3))).float()
        X[isnan] = 0.

        mask_bool = (mask==1)
        def node_mask_select(x):
            shape = x.shape
            x = x.reshape(shape[0], shape[1],-1)
            out = torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])
            out = out.reshape(-1,*shape[2:])
            return out

        batch_id = torch.arange(mask_bool.shape[0], device=mask_bool.device)[:,None].expand_as(mask_bool)
        seq = node_mask_select(S)
        X = node_mask_select(X)
        batch_id = node_mask_select(batch_id)
        C_a = X[:,1,:]
        
        edge_idx = knn_graph(C_a, k=30, batch=batch_id, loop=True, flow='target_to_source')

        
        
        N, CA, C = X[:,0], X[:,1], X[:,2]

        T = Rigid.make_transform_from_reference(N.float(), CA.float(), C.float())
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        T_ts = T[dst_idx,None].invert().compose(T[src_idx,None])

        # global virtual frames
        num_global = self.virtual_frame_num
        
        '''
        U的每一列，为原始空间中的坐标基向量
        R = U
        U2, S2, V2 = torch.svd((R@X_c.T)@(X_c@R.T))
        R@U == U2
        '''

        X_c = T._trans
        X_m = X_c.mean(dim=0, keepdim=True)
        X_c = X_c-X_m
        U,S,V = torch.svd(X_c.T@X_c)
        d = (torch.det(U) * torch.det(V)) < 0.0
        D = torch.zeros_like(V)
        D[ [0,1], [0,1]] = 1
        D[2,2] = -1*d+1*(~d)
        V = D@V
        R = torch.matmul(U, V.permute(0,1))


        rot_g = [R]*num_global
        trans_g = [X_m]*num_global
        
        feat = get_interact_feats(T, T_ts, X.float(), edge_idx, batch_id)
        _V, _E = feat['_V'], feat['_E']

        '''
        global_src: N+1,N+1,N+2,N+2,..N+B, N+B+1,N+B+1,N+B+2,N+B+2,..N+B+B
        global_dst: 0,  1,  2,  3,  ..N,   0,    1,    2,    3,    ..N
        batch_id_g: 1,  1,  2,  2,  ..B,   1,    1,    2,    2,    ..B
        '''
        T_g = Rigid(Rotation(torch.stack(rot_g)), torch.cat(trans_g,dim=0))
        num_nodes = scatter_sum(torch.ones_like(batch_id), batch_id)
        global_src = torch.cat([batch_id  +k*num_nodes.shape[0] for k in range(num_global)]) + num_nodes
        global_dst = torch.arange(batch_id.shape[0], device=batch_id.device).repeat(num_global)
        edge_idx_g = torch.stack([global_dst, global_src])
        edge_idx_g_inv = torch.stack([global_src, global_dst])
        edge_idx_g = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)

        batch_id_g = torch.zeros(num_global,dtype=batch_id.dtype)
        T_all = Rigid.cat([T, T_g], dim=0)

        idx, _ = edge_idx_g.min(dim=0)
        T_gs = T_all[idx,None].invert().compose(T_all[idx,None])

        rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(_E.shape[0],-1)
        rbf_gs = rbf(T_gs._trans.norm(dim=-1), 0, 50, 16)[:,0].view(edge_idx_g.shape[1],-1)

        _V_g = torch.arange(num_global)
        _E_g = torch.zeros([edge_idx_g.shape[1], 128])

        mask = torch.masked_select(mask, mask_bool)
        chain_features = (chain_encoding[edge_idx[0]] == chain_encoding[edge_idx[1]]).int()

        batch={
                'T':T,
                'T_g': T_g,
                'T_ts': T_ts,
                'T_gs': T_gs,
                'rbf_ts': rbf_ts,
                'rbf_gs': rbf_gs,
                'X':X,
                'chain_features': chain_features,
                '_V': _V,
                '_E': _E,
                '_V_g': _V_g,
                '_E_g': _E_g,
                'S':seq,
                'edge_idx':edge_idx,
                'edge_idx_g': edge_idx_g,
                'batch_id': batch_id,
                'batch_id_g': batch_id_g,
                'num_nodes': num_nodes,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding,
                'K_g': num_global}

        return batch
    
    def featurize(self,batch):
        if self.exp_backbone_noise_sd != 0:
            # print('backbone noise sd: ', self.exp_backbone_noise_sd)
            # Iterate over each protein sample in the batch list
            for protein_sample in batch:
                # List of keys corresponding to backbone atom coordinates
                coord_keys = ['N', 'CA', 'C', 'O']
                for key in coord_keys:
                    # Get the original coordinates (e.g., shape [num_residues, 3])
                    coords = protein_sample[key]
                    # Generate Gaussian noise with the same shape as the coordinates.
                    # The noise is centered at 0.0 with the specified standard deviation.
                    noise = np.random.normal(loc=0.0, scale=self.exp_backbone_noise_sd, size=coords.shape)
                    # Add the noise to the original coordinates and update the sample in place
                    protein_sample[key] = coords + noise

        if self.ig_baseline_data:
            for protein_sample in batch:
                # List of keys corresponding to backbone atom coordinates
                coord_keys = ['N', 'CA', 'C', 'O']
                for key in coord_keys:
                    coords = protein_sample[key]
                    protein_sample[key] = np.random.normal(loc=0.0, scale=15., size=coords.shape)

        # deepcopy batch
        batch_copy = copy.deepcopy(batch)
        res = []
        for one in batch:
            temp = self._get_features_persample(one)
            res.append(temp)
        res = self.custom_collate_fn(res)
        # sbc2 featurizer
        bc_batch = self.featurize_SBC2Model(batch_copy)
        # update bc_batch into res
        for key in bc_batch.keys():
            res[key] = bc_batch[key]
        return res
    
    def custom_collate_fn(self, batch):
        batch = [one for one in batch if one is not None]
        num_nodes = torch.cat([one['num_nodes'] for one in batch])
        shift = num_nodes.cumsum(dim=0)
        shift = torch.cat([torch.tensor([0], device=shift.device), shift], dim=0)
        def shift_node_idx(idx, num_node, shift_real, shift_virtual):
            mask = idx>=num_node
            shift_combine = (~mask)*(shift_real) + (mask)*(shift_virtual)
            return idx+shift_combine

        
        ret = {}
        for key in batch[0].keys():
            if batch[0][key] is None:
                continue
            
            if key in ['T', 'T_g', 'T_ts', 'T_gs']:
                T = Rigid.cat([one[key] for one in batch], dim=0)
                ret[key+'_rot'] = T._rots._rot_mats
                ret[key+'_trans'] = T._trans
            elif key in ['edge_idx']:
                ret[key] = torch.cat([one[key] + shift[idx] for idx, one in enumerate(batch)], dim=1)
            elif key in ['edge_idx_g']:
                edge_idx_g = []
                for idx, one in enumerate(batch):
                    shift_virtual = shift[-1] + idx*one['K_g']-num_nodes[idx]
                    src = shift_node_idx(one['edge_idx_g'][0], num_nodes[idx], shift[idx], shift_virtual)
                    dst_g = shift_node_idx(one['edge_idx_g'][1], num_nodes[idx], shift[idx], shift_virtual) 
                    edge_idx_g.append(torch.stack([src, dst_g]))
                ret[key] = torch.cat(edge_idx_g, dim=1)
                # edge_idx_g = torch.cat(edge_idx_g, dim=1)
                # edge_idx_g_inv = edge_idx_g.flip((0,))
                # ret[key] = torch.cat([edge_idx_g, edge_idx_g_inv], dim=1)
            elif key in ['batch_id', 'batch_id_g']:
                ret[key] = torch.cat([one[key] + idx for idx, one in enumerate(batch)])
            elif key in ['K_g']:
                pass
            else:
                ret[key] = torch.cat([one[key] for one in batch], dim=0)

        return ret

    def featurize_SBC2Model(self, batch):
        """ Pack and pad batch into torch tensors with surface and orig_surface downsampling to the minimum size """
        # batch = [one for one in batch if one is not None]
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
            # check if b['N'] is a list
            x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1)  # [#atom, 4, 3]
            # # check if x is [1, #atom, 4, 3]
            # if x.shape[0] == 1 and len(x.shape) == 4:
            #     # remove the 0th dimension
            #     x = x.squeeze(0)
            
            l = len(b['seq'])
            x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))  # [#atom, 4, 3]
            X[i, :, :, :] = x_pad

            # Convert to labels
            indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
            # indices = np.array(self.tokenizer.encode(b['seq']))
            S[i, :l] = indices
            chain_mask[i, :l] = b['chain_mask']
            chain_encoding[i, :l] = b['chain_encoding']

            # Add surface, features, orig_surface
            surfaces.append(torch.tensor(b['surface'], dtype=torch.float32))
            features.append(torch.tensor(b['features'], dtype=torch.float32))
            orig_surfaces.append(torch.tensor(b['orig_surface'], dtype=torch.float32))
            surface_lengths.append(b['surface'].shape[0])

            if self.partial_design and self.design_regions:
                protein_name = b['title']
                if protein_name in self.design_regions:
                    # 1. Get the necessary data
                    design_mask = torch.tensor(self.design_regions[protein_name], dtype=torch.bool)
                    
                    # Convert tensors to NumPy arrays for Scikit-learn (this is fast on CPU)
                    ca_coords = torch.tensor(b['CA'], dtype=torch.float32).numpy()
                    surface_coords = orig_surfaces[i].numpy()

                    if len(design_mask) != len(ca_coords):
                        print(f"⚠️ WARNING: Mismatch for '{protein_name}'. Mask length {len(design_mask)} != Residue count {len(ca_coords)}. Skipping masking.")
                    else:
                        # 2. Find the closest residue for each surface point (using NearestNeighbors)
                        # Build the tree from the residue coordinates
                        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ca_coords)
                        
                        # Find the index of the single nearest neighbor for each surface point
                        distances, indices = nbrs.kneighbors(surface_coords)
                        
                        # `indices` has shape [num_surface_points, 1], so flatten it
                        closest_residue_indices = indices.flatten()
                        
                        # 3. Create a mask for the surface points
                        # Use the NumPy array of indices to look up values in the PyTorch design_mask
                        surface_mask = design_mask[closest_residue_indices]
                        
                        # 4. Apply the mask to the features tensor
                        features[i][surface_mask] = float('nan')

                else:
                    print(f"⚠️ WARNING: Protein '{protein_name}' not found in design region file. Skipping masking for this sample.")

            if self.ig_baseline_data:
                features[i][:] = float('nan')

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

        # # Calculate and print the proportion of NaN values
        # nan_proportion = torch.isnan(features_stacked).sum() / features_stacked.numel()
        # print(f"Proportion of NaN values in features_stacked: {nan_proportion.item():.2%}")

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
        X_flattened = X[mask==1]
        lengths = torch.from_numpy(lengths)
        chain_mask = torch.from_numpy(chain_mask)
        chain_encoding = torch.from_numpy(chain_encoding)

        mask_bool = (mask==1)
        S = torch.masked_select(S, mask_bool)
        mask = torch.masked_select(mask, mask_bool)
        return {
            "title": [b['title'] for b in batch],
            "X": X,
            "X_flattened": X_flattened,
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
