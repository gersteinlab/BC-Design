import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from src.tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
import numpy as np
from transformers import AutoTokenizer
import math


pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']



class PointCloudMessagePassing(nn.Module):
    def __init__(self, feat_dim, edge_dim, l_max, num_scales, hidden_dim, aggregation='concat', num_heads=4, num_mha_layers=1):
        super(PointCloudMessagePassing, self).__init__()
        self.l_max = l_max
        self.num_scales = num_scales
        self.aggregation = aggregation
        self.num_heads = num_heads

        self.per_layer_dim = hidden_dim // 4

        # CLS token for biochemical features initialized with per_layer_dim
        self.biochem_cls_token = nn.Parameter(torch.zeros(1 + 8, self.per_layer_dim))  # Adjusted dimension

        # Linear layer for feature dimension adjustment
        self.input_fc = nn.Linear(feat_dim, self.per_layer_dim)

        # MHA module
        self.mha = nn.MultiheadAttention(embed_dim=self.per_layer_dim, num_heads=num_heads, batch_first=True)

        # fc for residue connection
        self.res_conn_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.per_layer_dim, hidden_dim)
        )

        # Feature aggregation after MHA
        self.fc = nn.Linear(self.per_layer_dim * num_scales, hidden_dim)

    def forward(self, surfaces, biochem_feats, correspondences):
        B, N, _ = surfaces.shape
        
        # Elevate the biochemical features
        biochem_feats = self.input_fc(biochem_feats)  # BxNx(per_layer_dim)

        # Add CLS token at the end of biochem_feats (Bx(N+1)x(per_layer_dim))
        cls_tokens = self.biochem_cls_token.expand(B, -1, -1)  # Expand CLS token for the batch
        biochem_feats = torch.cat([biochem_feats, cls_tokens], dim=1)  # Concatenated CLS token

        # Add a last row of infs and a last column of 0s to distances
        distances = torch.cdist(surfaces, surfaces)  # BxNxN

        # Compute the maximum distance to set dynamic radii
        max_distance = distances.max().item()
        thr_rs = [max_distance / 20 * i / 4 for i in range(1, 5)]  # Different scales of radii

        # Add 9 rows to the bottom of distances, all set to inf
        inf_rows = torch.full((B, 9, N), float('inf'), device=surfaces.device)  # (Bx9xN)
        distances = torch.cat([distances, inf_rows], dim=1)  # Bx(N+9)xN

        # Add 9 columns to the right of distances, with special handling
        inf_cols = torch.full((B, N + 9, 9), float('inf'), device=surfaces.device)  # Bx(N+9)x9

        # First column (corresponding to global CLS token) is all 0s
        inf_cols[:, :, 0] = 0

        # Vectorized filling of subarea CLS distances based on correspondences
        for i in range(B):
            # Get the neighbors from correspondences and subarea indices
            corr = correspondences[i]
            surface_neighbors = torch.cat([surf for _, surf in corr], dim=0)  # Concatenate all surface neighbors

            # Create indices for the subareas corresponding to surface neighbors
            subarea_idxs = torch.cat([torch.full_like(surf, j+1) for j, (_, surf) in enumerate(corr)], dim=0)

            # Assign distances for subarea CLS tokens to 0 where correspondences exist
            inf_cols[i, surface_neighbors, subarea_idxs] = 0

        # Concatenate the inf_cols to distances
        distances = torch.cat([distances, inf_cols], dim=2)  # Bx(N+9)x(N+9)

        # Set the diagonal of the last 9x9 block to 0
        distances[:, -9:, -9:] = float('inf')  # Set the entire 9x9 block to inf first
        distances[:, -9:, -9:].diagonal(dim1=-2, dim2=-1).fill_(0)  # Set only the diagonal values to 0

        N += 9  # Adjust N to N+9 since CLS tokens are added
        
        features_list = []
        
        for thr_r in thr_rs:
            # 1. Create a mask for points within the spherical region
            region_mask = distances < thr_r  # Bx(N+1)x(N+1) boolean mask
            
            # 2. Compute the number of neighbors for each point in the region (Bx(N+1))
            num_neighbors = region_mask.sum(dim=-1)  # Bx(N+1)
            
            # 3. Find the maximum number of neighbors to pad all regions to the same size
            max_neighbors = num_neighbors.max().item()  # The largest region size in this batch

            # 4. Downsample neighbors to 100 if max_neighbors > 100
            if max_neighbors > 100:
                # Step 1: Get the indices of the True values in region_mask (all neighbors)
                batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)

                # Step 2: Create a mask for the center points (rows) that have more than 100 neighbors
                over_limit_mask = num_neighbors > 100  # Bx(N+1) boolean mask where num_neighbors > 100
                
                # Step 3: Find the batch and center indices that have more than 100 neighbors
                over_limit_batch_idx, over_limit_center_idx = torch.nonzero(over_limit_mask, as_tuple=True)
                
                # Step 4: For these rows, get the neighbor indices and randomly sample 100 neighbors for each row
                downsampled_mask = region_mask.clone()
                
                for b_idx, c_idx in zip(over_limit_batch_idx, over_limit_center_idx):
                    # Find all neighbors for this center point
                    neighbor_indices = torch.nonzero(region_mask[b_idx, c_idx], as_tuple=False).squeeze()  # Get all neighbors
                    
                    # Randomly sample 100 neighbors
                    random_indices = torch.randperm(neighbor_indices.size(0), device=biochem_feats.device)[:100]  # Randomly select 100
                    selected_neighbors = neighbor_indices[random_indices]  # Select 100 neighbors
                    
                    # Reset region_mask for this point and update it with only the selected 100 neighbors
                    downsampled_mask[b_idx, c_idx] = False
                    downsampled_mask[b_idx, c_idx, selected_neighbors] = True
                
                # Update region_mask with the downsampled mask
                region_mask = downsampled_mask

            # Recompute num_neighbors and max_neighbors after downsampling
            num_neighbors = region_mask.sum(dim=-1)  # Bx(N+1)
            max_neighbors = num_neighbors.max().item()  # Limit max_neighbors to 100
            
            # 5. Get the indices of True values in region_mask
            batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)  # Extract indices of neighbors in the region
            
            # 6. Gather the biochemical features for these indices
            gathered_feats = biochem_feats[batch_idx, neighbor_idx]  # Gather the corresponding features from biochem_feats
           
            # 7. Generate sequential indices for each neighbor
            neighbor_offsets = torch.arange(num_neighbors.sum()).to(num_neighbors.device) - torch.repeat_interleave(torch.cumsum(num_neighbors.view(-1), dim=0) - num_neighbors.view(-1), num_neighbors.view(-1)).to(num_neighbors.device)

            # 8. Create a tensor to hold padded features for each region
            padded_feats = torch.zeros(B, N, max_neighbors, biochem_feats.shape[-1], device=biochem_feats.device)
            
            # Create a mask to indicate which points are real and which are padding
            padding_mask = torch.zeros(B, N, max_neighbors, device=biochem_feats.device, dtype=torch.bool)
            
            # 9. Scatter the gathered features into the padded_feats tensor using the generated sequential indices
            padded_feats[batch_idx, center_idx, neighbor_offsets] = gathered_feats
            
            # Update padding mask where neighbors exist
            padding_mask[batch_idx, center_idx, neighbor_offsets] = 1  # Mark valid neighbors
            
            # 10. Perform Multi-Head Attention (MHA)
            padded_feats_flat = padded_feats.view(B * N, max_neighbors, -1)  # (B*(N+1))xMaxNeighborsxFeatDim
            padding_mask_flat = ~padding_mask.view(B * N, max_neighbors)  # (B*(N+1))xMaxNeighbors, invert mask for MHA
            
            # Apply MHA over the padded regions
            attn_output, _ = self.mha(padded_feats_flat, padded_feats_flat, padded_feats_flat, key_padding_mask=padding_mask_flat)
            
            # 11. Perform pooling over the region (e.g., mean pooling over valid points)
            attn_output = attn_output.view(B, N, max_neighbors, -1)  # Bx(N+1)xMaxNeighborsxFeatDim
            pooled_feats = attn_output.masked_fill(~padding_mask.unsqueeze(-1), 0).sum(dim=2) / num_neighbors.unsqueeze(-1)  # Bx(N+1)xFeatDim

            features_list.append(pooled_feats)
        
        # 12. Concatenate features from different scales
        combined_feats = torch.cat(features_list, dim=-1)  # Bx(N+1)x(num_scales * per_layer_dim)

        # Add the residual connection and final projection to hidden_dim
        combined_feats = combined_feats + self.res_conn_mlp(biochem_feats)
        output_feats = self.fc(combined_feats)  # Bx(N+1)xhidden_dim
        
        return output_feats



class PointCloudMessagePassingMultiple(nn.Module):
    def __init__(self, feat_dim, edge_dim, l_max, num_scales, hidden_dim, aggregation='concat', num_heads=4, num_mha_layers=1):
        super(PointCloudMessagePassingMultiple, self).__init__()
        self.l_max = l_max
        self.num_scales = num_scales
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.num_mha_layers = num_mha_layers

        self.per_layer_dim = hidden_dim // 4

        # CLS token for biochemical features initialized with per_layer_dim
        self.biochem_cls_token = nn.Parameter(torch.zeros(1 + 8, self.per_layer_dim))  # Adjusted dimension

        # Linear layer for feature dimension adjustment
        self.input_fc = nn.Linear(feat_dim, self.per_layer_dim)

        # MHA module
        # Replace single MHA with multiple layers
        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.per_layer_dim, num_heads=num_heads, batch_first=True)
            for _ in range(self.num_mha_layers)
        ])

        # fc for residue connection
        self.res_conn_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.per_layer_dim, hidden_dim)
        )

        # Feature aggregation after MHA
        self.fc = nn.Linear(self.per_layer_dim * num_scales, hidden_dim)

    def forward(self, surfaces, biochem_feats, correspondences):
        B, N, _ = surfaces.shape
        
        # Elevate the biochemical features
        biochem_feats = self.input_fc(biochem_feats)  # BxNx(per_layer_dim)

        # Add CLS token at the end of biochem_feats (Bx(N+1)x(per_layer_dim))
        cls_tokens = self.biochem_cls_token.expand(B, -1, -1)  # Expand CLS token for the batch
        biochem_feats = torch.cat([biochem_feats, cls_tokens], dim=1)  # Concatenated CLS token

        # Add a last row of infs and a last column of 0s to distances
        distances = torch.cdist(surfaces, surfaces)  # BxNxN

        # Compute the maximum distance to set dynamic radii
        max_distance = distances.max().item()
        thr_rs = [max_distance / 20 * i / 4 for i in range(1, 5)]  # Different scales of radii

        # Add 9 rows to the bottom of distances, all set to inf
        inf_rows = torch.full((B, 9, N), float('inf'), device=surfaces.device)  # (Bx9xN)
        distances = torch.cat([distances, inf_rows], dim=1)  # Bx(N+9)xN

        # Add 9 columns to the right of distances, with special handling
        inf_cols = torch.full((B, N + 9, 9), float('inf'), device=surfaces.device)  # Bx(N+9)x9

        # First column (corresponding to global CLS token) is all 0s
        inf_cols[:, :, 0] = 0

        # Vectorized filling of subarea CLS distances based on correspondences
        for i in range(B):
            # Get the neighbors from correspondences and subarea indices
            corr = correspondences[i]
            surface_neighbors = torch.cat([surf for _, surf in corr], dim=0)  # Concatenate all surface neighbors

            # Create indices for the subareas corresponding to surface neighbors
            subarea_idxs = torch.cat([torch.full_like(surf, j+1) for j, (_, surf) in enumerate(corr)], dim=0)

            # Assign distances for subarea CLS tokens to 0 where correspondences exist
            inf_cols[i, surface_neighbors, subarea_idxs] = 0

        # Concatenate the inf_cols to distances
        distances = torch.cat([distances, inf_cols], dim=2)  # Bx(N+9)x(N+9)

        # Set the diagonal of the last 9x9 block to 0
        distances[:, -9:, -9:] = float('inf')  # Set the entire 9x9 block to inf first
        distances[:, -9:, -9:].diagonal(dim1=-2, dim2=-1).fill_(0)  # Set only the diagonal values to 0

        N += 9  # Adjust N to N+9 since CLS tokens are added
        
        features_list = []
        
        for thr_r in thr_rs:
            # 1. Create a mask for points within the spherical region
            region_mask = distances < thr_r  # Bx(N+1)x(N+1) boolean mask
            
            # 2. Compute the number of neighbors for each point in the region (Bx(N+1))
            num_neighbors = region_mask.sum(dim=-1)  # Bx(N+1)
            
            # 3. Find the maximum number of neighbors to pad all regions to the same size
            max_neighbors = num_neighbors.max().item()  # The largest region size in this batch

            # 4. Downsample neighbors to 100 if max_neighbors > 100
            if max_neighbors > 100:
                # Step 1: Get the indices of the True values in region_mask (all neighbors)
                batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)

                # Step 2: Create a mask for the center points (rows) that have more than 100 neighbors
                over_limit_mask = num_neighbors > 100  # Bx(N+1) boolean mask where num_neighbors > 100
                
                # Step 3: Find the batch and center indices that have more than 100 neighbors
                over_limit_batch_idx, over_limit_center_idx = torch.nonzero(over_limit_mask, as_tuple=True)
                
                # Step 4: For these rows, get the neighbor indices and randomly sample 100 neighbors for each row
                downsampled_mask = region_mask.clone()
                
                for b_idx, c_idx in zip(over_limit_batch_idx, over_limit_center_idx):
                    # Find all neighbors for this center point
                    neighbor_indices = torch.nonzero(region_mask[b_idx, c_idx], as_tuple=False).squeeze()  # Get all neighbors
                    
                    # Randomly sample 100 neighbors
                    random_indices = torch.randperm(neighbor_indices.size(0), device=biochem_feats.device)[:100]  # Randomly select 100
                    selected_neighbors = neighbor_indices[random_indices]  # Select 100 neighbors
                    
                    # Reset region_mask for this point and update it with only the selected 100 neighbors
                    downsampled_mask[b_idx, c_idx] = False
                    downsampled_mask[b_idx, c_idx, selected_neighbors] = True
                
                # Update region_mask with the downsampled mask
                region_mask = downsampled_mask

            # Recompute num_neighbors and max_neighbors after downsampling
            num_neighbors = region_mask.sum(dim=-1)  # Bx(N+1)
            max_neighbors = num_neighbors.max().item()  # Limit max_neighbors to 100
            
            # 5. Get the indices of True values in region_mask
            batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)  # Extract indices of neighbors in the region
            
            # 6. Gather the biochemical features for these indices
            gathered_feats = biochem_feats[batch_idx, neighbor_idx]  # Gather the corresponding features from biochem_feats
           
            # 7. Generate sequential indices for each neighbor
            neighbor_offsets = torch.arange(num_neighbors.sum()).to(num_neighbors.device) - torch.repeat_interleave(torch.cumsum(num_neighbors.view(-1), dim=0) - num_neighbors.view(-1), num_neighbors.view(-1)).to(num_neighbors.device)

            # 8. Create a tensor to hold padded features for each region
            padded_feats = torch.zeros(B, N, max_neighbors, biochem_feats.shape[-1], device=biochem_feats.device)
            
            # Create a mask to indicate which points are real and which are padding
            padding_mask = torch.zeros(B, N, max_neighbors, device=biochem_feats.device, dtype=torch.bool)
            
            # 9. Scatter the gathered features into the padded_feats tensor using the generated sequential indices
            padded_feats[batch_idx, center_idx, neighbor_offsets] = gathered_feats
            
            # Update padding mask where neighbors exist
            padding_mask[batch_idx, center_idx, neighbor_offsets] = 1  # Mark valid neighbors
            
            # 10. Perform Multi-Head Attention (MHA)
            padded_feats_flat = padded_feats.view(B * N, max_neighbors, -1)  # (B*(N+1))xMaxNeighborsxFeatDim
            padding_mask_flat = ~padding_mask.view(B * N, max_neighbors)  # (B*(N+1))xMaxNeighbors, invert mask for MHA
            
            # Apply MHA over the padded regions
            # Initialize attn_output with padded_feats_flat
            attn_output = padded_feats_flat

            # Apply multiple MHA layers
            for mha_layer in self.mha_layers:
                attn_output, _ = mha_layer(
                    attn_output, attn_output, attn_output, key_padding_mask=padding_mask_flat
                )
            
            # 11. Perform pooling over the region (e.g., mean pooling over valid points)
            attn_output = attn_output.view(B, N, max_neighbors, -1)  # Bx(N+1)xMaxNeighborsxFeatDim
            pooled_feats = attn_output.masked_fill(~padding_mask.unsqueeze(-1), 0).sum(dim=2) / num_neighbors.unsqueeze(-1)  # Bx(N+1)xFeatDim

            features_list.append(pooled_feats)
        
        # 12. Concatenate features from different scales
        combined_feats = torch.cat(features_list, dim=-1)  # Bx(N+1)x(num_scales * per_layer_dim)

        # Add the residual connection and final projection to hidden_dim
        combined_feats = combined_feats + self.res_conn_mlp(biochem_feats)
        output_feats = self.fc(combined_feats)  # Bx(N+1)xhidden_dim
        
        return output_feats



class GraphTransformerModelShare(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, dropout=0.):
        super(GraphTransformerModelShare, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
        # CLS token for structural features
        self.struct_cls_token = nn.Parameter(torch.zeros(1 + 8, hidden_dim))

        # Linear projections for multi-head attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Layer normalization for the FFN block
        self.layer_norm_ffn = nn.LayerNorm(hidden_dim)
        
        # MLP for node embeddings
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # MLP for edge weights (converts _E into 1D edge weights)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 1D output for edge weights
        )
    
    def forward(self, _V, _E, batch):
        # Unpack batch
        batch_id, E_idx, correspondences = batch['batch_id'], batch['E_idx'], batch['correspondences']
        inv_distance_matrices = batch['inv_distance_matrices']
        heat_kernel_pe = batch['heat_kernel_pe']
        padding_mask = batch['unflattened_mask'].bool()  # Shape: (B, max_nodes)

        B = len(batch_id.unique())  # Batch size
        max_nodes = inv_distance_matrices.size(1)  # Max number of nodes in any graph in the batch

        # Node embedding from _V using MLP
        node_embeds = self.node_mlp(_V)  # Apply MLP to node features

        # Add CLS token to node embeddings
        cls_tokens = self.struct_cls_token.expand(B, -1, -1)  # Shape: (B, 1, hidden_dim)
        padded_node_embeds = self._pad_and_stack(node_embeds, batch_id, max_nodes)  # Shape: (B, max_nodes, hidden_dim)
        padded_node_embeds = torch.cat([cls_tokens, padded_node_embeds], dim=1)  # Add CLS token at the start

        # Update max_nodes to account for CLS token
        max_nodes += 1 + 8
        
        # Edge weight transformation from _E using MLP
        edge_weights = self.edge_mlp(_E).squeeze(-1)  # Convert edge features to 1D weights
        padded_edge_weights = self._pad_and_stack_edges(edge_weights, batch_id, E_idx, max_nodes, correspondences)  # Shape: (B, max_nodes, max_nodes)

        # Apply MHA layers for n_layers
        for _ in range(self.n_layers):
            padded_node_embeds = self.multi_head_attention(
                padded_node_embeds, inv_distance_matrices, heat_kernel_pe, padded_edge_weights, padding_mask
            )
        
        return padded_node_embeds

    def multi_head_attention(self, node_embeds, inv_distance_matrices, heat_kernel_pe, transformed_edge_weights, padding_mask):
        # Batch size and max_nodes
        B, max_nodes, _ = node_embeds.size()
        d_k = self.hidden_dim // self.n_heads  # Dimension per head

        # Linear projections for Q, K, V
        Q = self.query_proj(node_embeds)  # (B, max_nodes, hidden_dim)
        K = self.key_proj(node_embeds)    # (B, max_nodes, hidden_dim)
        V = self.value_proj(node_embeds)  # (B, max_nodes, hidden_dim)

        # Reshape Q, K, V for multi-head attention: (B, n_heads, max_nodes, d_k)
        Q = Q.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)
        K = K.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)
        V = V.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)

        # Scaled dot-product attention: (QK^T / sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, n_heads, max_nodes, max_nodes)

        # Add inv_distance_matrices and heat_kernel_pe to attention scores
        # Add 0th row and column to heat_kernel_pe
        heat_kernel_pe = F.pad(heat_kernel_pe, (1 + 8, 0, 1 + 8, 0), "constant", 0)
        
        attn_scores = attn_scores * transformed_edge_weights.unsqueeze(1) + heat_kernel_pe.unsqueeze(1)
        attn_scores = torch.where(transformed_edge_weights.unsqueeze(1) == 0, -1e9, attn_scores)

        # Apply padding mask (set scores to a large negative value where padding mask is False)
        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1 + 8, 0), "constant", 1)  # Pad to include CLS token
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, max_nodes)
            attn_scores = attn_scores.masked_fill(~padding_mask, -1e9)  # Mask padded positions with large negative value

        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute the final weighted values
        attn_output = torch.matmul(attn_probs, V)  # (B, n_heads, max_nodes, d_k)

        # Concatenate heads and project the result back to hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, max_nodes, self.hidden_dim)  # (B, max_nodes, hidden_dim)
        attn_output = self.out_proj(attn_output)

        # Apply dropout and residual connection for attention output
        attn_output = self.dropout_layer(attn_output)
        attn_output = attn_output + node_embeds  # Residual connection for attention
        attn_output = self.layer_norm(attn_output)  # Layer normalization for attention output

        # Apply Feed-Forward Network (FFN) + residual connection + layer normalization
        ffn_output = self.ffn(attn_output)  # Feed-Forward network
        ffn_output = self.dropout_layer(ffn_output)  # Dropout after FFN
        ffn_output = ffn_output + attn_output  # Residual connection for FFN
        ffn_output = self.layer_norm_ffn(ffn_output)  # Layer normalization for FFN output

        return ffn_output

    def _pad_and_stack(self, features, batch_id, max_nodes):
        """Pad and stack node features to include CLS token."""
        B = batch_id.max().item() + 1  # Batch size
        padded = torch.zeros((B, max_nodes, self.hidden_dim), device=features.device)
        
        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            padded[i, :len(node_indices), :] = features[node_indices]
        
        return padded

    def _pad_and_stack_edges(self, edge_weights, batch_id, E_idx, max_nodes, correspondences):
        """Pad and stack edges to include interactions for CLS tokens."""
        B = batch_id.max().item() + 1  # Batch size
        padded_edges = torch.zeros((B, max_nodes, max_nodes), device=edge_weights.device)

        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            min_node_id = node_indices.min().item()

            src, dst = E_idx[0, :], E_idx[1, :]
            local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

            src_local = src[local_edges_mask] - min_node_id
            dst_local = dst[local_edges_mask] - min_node_id

            # Fill interactions for global CLS token (0th row/column)
            padded_edges[i, 0, 1:len(node_indices) + 1] = 1  # CLS -> Nodes
            padded_edges[i, 1:len(node_indices) + 1, 0] = 1  # Nodes -> CLS

            # Vectorized filling for subarea CLS tokens (1st to 8th rows/columns)
            ca_neighbors_list = [ca_neighbors + 1 + 8 for ca_neighbors, _ in correspondences[i]]  # Shifted CA neighbors
            subarea_idx = torch.arange(1, 1 + 8, device=padded_edges.device)  # Subarea indices 1 to 8

            # Create a tensor from the list of CA neighbors
            ca_neighbors_tensor = torch.cat(ca_neighbors_list).long()
            subarea_repeats = torch.repeat_interleave(subarea_idx, torch.tensor([len(ca) for ca in ca_neighbors_list], device=padded_edges.device))

            # Assign values to the subarea CLS -> CA neighbors
            padded_edges[i, subarea_repeats, ca_neighbors_tensor] = 1  # Subarea CLS -> CA Neighbors
            padded_edges[i, ca_neighbors_tensor, subarea_repeats] = 1  # CA Neighbors -> Subarea CLS

            # Fill the rest of the edges (adjusted for the shifted indices)
            padded_edges[i, src_local + 1 + 8, dst_local + 1 + 8] = edge_weights[local_edges_mask]
            padded_edges[i, dst_local + 1 + 8, src_local + 1 + 8] = edge_weights[local_edges_mask]  # Assuming undirected edges

        return padded_edges


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SBC2Model(nn.Module):
    def __init__(self, args, queue_size=64, **kwargs):
        """ Graph labeling network """
        super(SBC2Model, self).__init__()
        self.args = args
        self.augment_eps = args.augment_eps
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        num_encoder_layers = args.num_encoder_layers
        self.top_k = args.k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        self.dihedral_type = args.dihedral_type
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
        alphabet = [one for one in 'ACDEFGHIKLMNPQRSTVWYX']
        self.token_mask = torch.tensor([(one in alphabet) for one in self.tokenizer._token_to_id.keys()])

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            node_in += pair_num*self.num_rbf
        if self.args.node_angle:
            node_in += 12
        if self.args.node_direct:
            node_in += 9
        
        edge_in = 0
        if self.args.edge_dist:
            pair_num = 0
            if self.args.Ca_Ca:
                pair_num += 1
            if self.args.Ca_C:
                pair_num += 2
            if self.args.Ca_N:
                pair_num += 2
            if self.args.Ca_O:
                pair_num += 2
            if self.args.C_C:
                pair_num += 1
            if self.args.C_N:
                pair_num += 2
            if self.args.C_O:
                pair_num += 2
            if self.args.N_N:
                pair_num += 1
            if self.args.N_O:
                pair_num += 2
            if self.args.O_O:
                pair_num += 1

            edge_in += pair_num*self.num_rbf
        if self.args.edge_angle:
            edge_in += 4
        if self.args.edge_direct:
            edge_in += 12
        
        if self.args.use_gvp_feat:
            node_in = 12
            edge_in = 48-16
        
        edge_in += 16+16 # position encoding, chain encoding

        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)

        self.encoder = GraphTransformerModelShare(hidden_dim=hidden_dim, n_layers=self.args.gt_layers, n_heads=8)

        l_max = 2
        num_scales = 4
        # best
        self.surface_encoder = PointCloudMessagePassing(2, 1, l_max, num_scales, hidden_dim)
        # hyperparam exp
        # self.surface_encoder = PointCloudMessagePassingMultiple(2, 1, l_max, num_scales, hidden_dim, num_mha_layers=4)

        # New Transformer decoder and MLP for final prediction
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
        # self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.tokenizer._token_to_id))
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, len(self.tokenizer._token_to_id))
        # )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        # Temperature for contrastive learning
        self.temperature = 0.1
        self.queue_size = queue_size

        # Initialize queues for structural and biochemical CLS tokens
        self.struct_queue = nn.Parameter(torch.zeros(queue_size, hidden_dim), requires_grad=False)
        self.biochem_queue = nn.Parameter(torch.zeros(queue_size, hidden_dim), requires_grad=False)
        self.queue_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

        self._init_params()
    
    def forward(self, batch):
        ### pifold encoder
        h_V, h_P, P_idx, batch_id = batch['_V'], batch['_E'], batch['E_idx'], batch['batch_id']

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V_unflattened = self.encoder(h_V, h_P, batch)

        # Manually extract the CLS tokens (global + subarea) from the structure encoder
        struct_cls_tokens = h_V_unflattened[:, :9, :]  # First 9 tokens: global (0th) + subarea (1st to 8th)
        h_V_unflattened = h_V_unflattened[:, 9:, :]  # The rest of the node embeddings

        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        mask_unflattened = torch.zeros(batch_size, max_length, device=h_V.device)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()


        ### surface encoder
        surfaces, biochem_feats, correspondences = batch['surface'], batch['features'], batch['correspondences']
        h_surface = self.surface_encoder(surfaces, biochem_feats, correspondences)

        # Manually extract the CLS tokens (global + subarea) from the biochemical encoder
        biochem_cls_tokens = h_surface[:, -9:, :]  # Last 9 tokens: global (0th) + subarea (1st to 8th)
        h_surface = h_surface[:, :-9, :]  # The rest of the biochemical node embeddings


        ### new decoder
        ss_connection_mask = batch['ss_connection']
        ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

        # Transformer decoder to fuse h_V_unflattened and h_surface
        # Add positional encoding to the inputs of the Transformer decoder
        h_V_unflattened = self.positional_encoding(h_V_unflattened)
        
        decoder_output = self.transformer_decoder(
            h_V_unflattened, h_surface, 
            tgt_key_padding_mask=target_padding_mask, 
            memory_mask=ss_connection_mask
        )

        # Flatten decoder_output and remove padding
        mask = mask_unflattened.bool()
        decoder_output = decoder_output[mask]

        # Predict labels using MLP
        logits = self.mlp(decoder_output)
        log_probs = F.log_softmax(logits, dim=-1)

        # Contrastive learning
        contrastive_loss_global = self._contrastive_loss(struct_cls_tokens[:, 0, :], biochem_cls_tokens[:, 0, :])  # Global CLS
        contrastive_loss_subarea = self._contrastive_loss_subarea(struct_cls_tokens[:, 1:, :], biochem_cls_tokens[:, 1:, :])  # Subarea CLS
        contrastive_loss = contrastive_loss_global + contrastive_loss_subarea

        # Update queues with current batch global CLS tokens
        self._dequeue_and_enqueue(struct_cls_tokens[:, 0, :], biochem_cls_tokens[:, 0, :])

        return {'log_probs': log_probs, 'contrastive_loss': contrastive_loss}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, struct_cls_token, biochem_cls_token):
        """Append new CLS tokens to the queue and dequeue older ones."""
        batch_size = struct_cls_token.size(0)

        # Get current position in the queue
        ptr = int(self.queue_ptr)

        # Replace oldest entries with the new ones
        if ptr + batch_size > self.queue_size:
            ptr = 0
        self.struct_queue[ptr:ptr + batch_size, :] = struct_cls_token
        self.biochem_queue[ptr:ptr + batch_size, :] = biochem_cls_token

        # Move pointer and wrap-around if necessary
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def _contrastive_loss(self, struct_cls_token, biochem_cls_token):
        """Compute NT-Xent contrastive loss using queue-based negative sampling."""
        batch_size = struct_cls_token.size(0)

        # Normalize CLS tokens
        z_i = F.normalize(struct_cls_token, dim=-1)
        z_j = F.normalize(biochem_cls_token, dim=-1)

        # Normalize queue embeddings
        struct_queue_norm = F.normalize(self.struct_queue.clone().detach(), dim=-1)
        biochem_queue_norm = F.normalize(self.biochem_queue.clone().detach(), dim=-1)

        # Cosine similarity between current CLS tokens
        sim_ij = torch.matmul(z_i, z_j.T) / self.temperature  # (batch_size, batch_size)

        # Cosine similarity with negative samples from the queue
        sim_i_struct_queue = torch.matmul(z_i, biochem_queue_norm.T) / self.temperature  # (batch_size, queue_size)
        sim_j_biochem_queue = torch.matmul(z_j, struct_queue_norm.T) / self.temperature  # (batch_size, queue_size)

        # Combine positive and negative samples
        sim_matrix_i = torch.cat([sim_ij, sim_i_struct_queue], dim=1)  # (batch_size, batch_size + queue_size)
        sim_matrix_j = torch.cat([sim_ij.T, sim_j_biochem_queue], dim=1)  # (batch_size, batch_size + queue_size)

        # Create labels (positive samples on the diagonal)
        labels = torch.arange(batch_size).long().to(sim_matrix_i.device)

        # Contrastive loss for both modalities
        loss_i = F.cross_entropy(sim_matrix_i, labels)
        loss_j = F.cross_entropy(sim_matrix_j, labels)

        loss = (loss_i + loss_j) / 2.0
        return loss

    def _contrastive_loss_subarea(self, struct_subarea_cls_tokens, biochem_subarea_cls_tokens):
        """Compute contrastive loss for the subarea CLS tokens without using a queue, using only the current batch."""
        batch_size, num_subareas, hidden_dim = struct_subarea_cls_tokens.size()

        # Normalize CLS tokens
        z_i = F.normalize(struct_subarea_cls_tokens, dim=-1)
        z_j = F.normalize(biochem_subarea_cls_tokens, dim=-1)

        # Cosine similarity within the batch for subarea CLS tokens
        sim_ij = torch.matmul(z_i, z_j.transpose(1, 2)) / self.temperature  # (batch_size, num_subareas, num_subareas)

        # Create labels (positive samples on the diagonal)
        labels = torch.arange(num_subareas).long().to(sim_ij.device).unsqueeze(0).expand(batch_size, -1)

        # Reshape sim_ij and labels for efficient cross-entropy calculation
        sim_ij = sim_ij.view(batch_size * num_subareas, num_subareas)  # (batch_size * num_subareas, num_subareas)
        labels = labels.reshape(batch_size * num_subareas)  # (batch_size * num_subareas,)

        # Compute contrastive loss in one step
        loss = F.cross_entropy(sim_ij, labels)

        return loss
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _get_features(self, batch):
        S, score, X, mask, chain_mask, chain_encoding = batch['S'], batch['score'], batch['X'], batch['mask'], batch['chain_mask'], batch['chain_encoding']

        device = X.device
        mask_bool = (mask==1)
        B, N, _,_ = X.shape
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)

        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # sequence
        S = torch.masked_select(S, mask_bool)
        if score is not None:
            score = torch.masked_select(score, mask_bool)
        chain_mask = torch.masked_select(chain_mask, mask_bool)
        chain_encoding = torch.masked_select(chain_encoding, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X, self.dihedral_type) 
        V_angles = node_mask_select(V_angles)

        V_direct, E_direct, E_angles = _orientations_coarse_gl_tuple(X, E_idx)
        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)

        # distance
        atom_N = X[:,:,0,:]
        atom_Ca = X[:,:,1,:]
        atom_C = X[:,:,2,:]
        atom_O = X[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()
        
        pair_lst = []
        if self.args.Ca_Ca:
            pair_lst.append('Ca-Ca')
        if self.args.Ca_C:
            pair_lst.append('Ca-C')
            pair_lst.append('C-Ca')
        if self.args.Ca_N:
            pair_lst.append('Ca-N')
            pair_lst.append('N-Ca')
        if self.args.Ca_O:
            pair_lst.append('Ca-O')
            pair_lst.append('O-Ca')
        if self.args.C_C:
            pair_lst.append('C-C')
        if self.args.C_N:
            pair_lst.append('C-N')
            pair_lst.append('N-C')
        if self.args.C_O:
            pair_lst.append('C-O')
            pair_lst.append('O-C')
        if self.args.N_N:
            pair_lst.append('N-N')
        if self.args.N_O:
            pair_lst.append('N-O')
            pair_lst.append('O-N')
        if self.args.O_O:
            pair_lst.append('O-O')

        
        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))
        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        ### 1. Distance Matrix Calculation with Padding
        # Create a zero-padded distance matrix for each batch element
        max_nodes = mask.shape[1]  # Max number of nodes in any graph in the batch

        eps = 1e-6  # Small epsilon to prevent division by zero
        distance_matrices = []

        for b in range(B):
            num_nodes = int(mask[b].sum().item())  # Number of valid nodes in this graph
            coords = X_ca[b, :num_nodes, :]  # (num_nodes, 3)
            
            # Pairwise Euclidean distance
            dist_matrix = torch.cdist(coords, coords)  # Shape: (num_nodes, num_nodes)
            
            # Add epsilon to the distance matrix to avoid division by zero
            dist_matrix = dist_matrix + eps
            
            # Compute the inverse of the distance matrix
            inv_dist_matrix = 1.0 / dist_matrix  # Shape: (num_nodes, num_nodes)
            
            # Zero-padding to max_nodes
            padded_inv_dist_matrix = torch.zeros((max_nodes, max_nodes), device=device)
            padded_inv_dist_matrix[:num_nodes, :num_nodes] = inv_dist_matrix
            distance_matrices.append(padded_inv_dist_matrix)

        # Stack distance matrices to create a batch tensor
        inv_distance_matrices = torch.stack(distance_matrices, dim=0)  # Shape: (B, max_nodes, max_nodes)


        h_V = []
        if self.args.node_dist:
            h_V.append(V_dist)
        if self.args.node_angle:
            h_V.append(V_angles)
        if self.args.node_direct:
            h_V.append(V_direct)
        
        h_E = []
        if self.args.edge_dist:
            h_E.append(E_dist)
        if self.args.edge_angle:
            h_E.append(E_angles)
        if self.args.edge_direct:
            h_E.append(E_direct)
        
        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)
        
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        ### 2. Heat Kernel-based Positional Encoding
        # Create adjacency matrix from E_idx (edges) based on the per-graph cumulative indices
        heat_kernel_pe_list = []

        # Calculate the shift for each graph in the batch based on the number of nodes
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)

        for b in range(B):
            num_nodes = int(mask[b].sum().item())  # Number of valid nodes in this graph

            # Get the edge indices for this batch element
            start_idx = shift[b].item()  # Starting index of the graph in the batch
            end_idx = start_idx + num_nodes
            
            # Filter edges that belong to the current graph
            edge_indices_b = E_idx[:, (E_idx[0, :] >= start_idx) & (E_idx[0, :] < end_idx)]  # Get edges for this graph
            
            # Adjust edge indices back to the local range for this graph
            edge_indices_b = (edge_indices_b - start_idx).to(torch.long)
            
            # Initialize adjacency matrix for the current graph
            adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
            
            # Populate adjacency matrix with the local edges
            adj_matrix[edge_indices_b[0], edge_indices_b[1]] = 1 # right

            # Degree matrix
            degree_matrix = torch.diag(adj_matrix.sum(dim=1))

            # Laplacian matrix
            laplacian_matrix = degree_matrix - adj_matrix

            # Compute heat kernel: H(t) = exp(-t * L), where t = 1, using PyTorch matrix exponential
            heat_kernel = torch.linalg.matrix_exp(-laplacian_matrix)  # Directly compute matrix exponential

            # Zero-pad the heat kernel to max_nodes size
            padded_heat_kernel = torch.zeros((max_nodes, max_nodes), device=device)
            padded_heat_kernel[:num_nodes, :num_nodes] = heat_kernel
            heat_kernel_pe_list.append(padded_heat_kernel)

        # Stack heat kernel matrices for each graph in the batch
        heat_kernel_pe = torch.stack(heat_kernel_pe_list, dim=0)  # Shape: (B, max_nodes, max_nodes)

        
        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)
        
        d_chains = ((chain_encoding[dst.long()] - chain_encoding[src.long()])==0).long().reshape(-1)   
        chain_embed = self._idx_embeddings(d_chains)
        _E = torch.cat([_E, chain_embed], dim=-1)

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]
        unflattened_mask = mask
        mask = torch.masked_select(mask, mask_bool)
        batch.update({'X':X,
                'S':S,
                'score':score,
                '_V':_V,
                '_E':_E,
                'E_idx':E_idx,
                'batch_id': batch_id,
                'unflattened_mask': unflattened_mask,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding,
                'inv_distance_matrices': inv_distance_matrices,  # (B, max_nodes, max_nodes)
                'heat_kernel_pe': heat_kernel_pe,        # (B, max_nodes, max_nodes)
                })
        return batch
        
    def _positional_embeddings(self, E_idx, 
                               num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = E_idx[0]-E_idx[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
    
    def _idx_embeddings(self, d, 
                               num_embeddings=None):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=d.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d[:,None] * frequency[None,:]
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E