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
    def __init__(self, feat_dim, edge_dim, l_max, num_scales, hidden_dim, aggregation='concat', num_heads=4):
        super(PointCloudMessagePassing, self).__init__()
        self.l_max = l_max
        self.num_scales = num_scales
        self.aggregation = aggregation
        self.num_heads = num_heads

        self.per_layer_dim = hidden_dim // 4
        
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

    def forward(self, surfaces, biochem_feats):
        B, N, _ = surfaces.shape
        
        # Elevate the biochemical features
        biochem_feats = self.input_fc(biochem_feats)  # BxNx(hidden_dim // 4)
        
        # Compute pairwise distances (dense N x N format)
        distances = torch.cdist(surfaces, surfaces)  # BxNxN
        
        # Compute the maximum distance to set dynamic radii
        max_distance = distances.max().item()
        thr_rs = [max_distance / 20 * i / 4 for i in range(1, 5)]  # Different scales of radii
        
        features_list = []
        
        for thr_r in thr_rs:
            # 1. Create a mask for points within the spherical region
            region_mask = distances < thr_r  # BxNxN boolean mask
            
            # 2. Compute the number of neighbors for each point in the region (BxN)
            num_neighbors = region_mask.sum(dim=-1)  # BxN
            
            # 3. Find the maximum number of neighbors to pad all regions to the same size
            max_neighbors = num_neighbors.max().item()  # The largest region size in this batch

            # begin test sample
            # Downsample neighbors to 100 if max_neighbors > 100
            if max_neighbors > 100:
                # Step 1: Get the indices of the True values in region_mask (all neighbors)
                batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)

                # Step 2: Create a mask for the center points (rows) that have more than 100 neighbors
                over_limit_mask = num_neighbors > 100  # BxN boolean mask where num_neighbors > 100
                
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
            num_neighbors = region_mask.sum(dim=-1)  # BxN
            max_neighbors = num_neighbors.max().item()  # Limit max_neighbors to 100
            # end test sample
            
            # 4. Get the indices of True values in region_mask
            batch_idx, center_idx, neighbor_idx = torch.nonzero(region_mask, as_tuple=True)  # Extract indices of neighbors in the region
            
            # 5. Gather the biochemical features for these indices
            gathered_feats = biochem_feats[batch_idx, neighbor_idx]  # Gather the corresponding features from biochem_feats
           
            # 6. Generate sequential indices for each neighbor (e.g., 0, 1, 2, ...) for each point
            # Sequential neighbor indices (scatter index)
            # neighbor_offsets = torch.cat([torch.arange(n) for n in num_neighbors.view(-1)])  # Creates a sequential range for each point's neighbors
            neighbor_offsets = torch.arange(num_neighbors.sum()).to(num_neighbors.device) - torch.repeat_interleave(torch.cumsum(num_neighbors.view(-1), dim=0) - num_neighbors.view(-1), num_neighbors.view(-1)).to(num_neighbors.device)

            # 7. Create a tensor to hold padded features for each region
            padded_feats = torch.zeros(B, N, max_neighbors, biochem_feats.shape[-1], device=biochem_feats.device)
            
            # Create a mask to indicate which points are real and which are padding
            padding_mask = torch.zeros(B, N, max_neighbors, device=biochem_feats.device, dtype=torch.bool)
            
            # 8. Scatter the gathered features into the padded_feats tensor using the generated sequential indices
            padded_feats[batch_idx, center_idx, neighbor_offsets] = gathered_feats
            
            # Update padding mask where neighbors exist
            padding_mask[batch_idx, center_idx, neighbor_offsets] = 1  # Mark valid neighbors
            
            # 9. Perform Multi-Head Attention (MHA)
            padded_feats_flat = padded_feats.view(B * N, max_neighbors, -1)  # (B*N)xMaxNeighborsxFeatDim
            padding_mask_flat = ~padding_mask.view(B * N, max_neighbors)  # (B*N)xMaxNeighbors, invert mask for MHA
            
            # Apply MHA over the padded regions
            attn_output, _ = self.mha(padded_feats_flat, padded_feats_flat, padded_feats_flat, key_padding_mask=padding_mask_flat)
            
            # 10. Perform pooling over the region (e.g., mean pooling over valid points)
            attn_output = attn_output.view(B, N, max_neighbors, -1)  # BxNxMaxNeighborsxFeatDim
            pooled_feats = attn_output.masked_fill(~padding_mask.unsqueeze(-1), 0).sum(dim=2) / num_neighbors.unsqueeze(-1)  # BxNxFeatDim

            features_list.append(pooled_feats)
        
        # 11. Concatenate features from different scales
        combined_feats = torch.cat(features_list, dim=-1)  # BxNx(num_scales * per_layer_dim)

        # residue connection
        combined_feats = combined_feats + self.res_conn_mlp(biochem_feats)
        
        # 12. Final projection to hidden_dim
        output_feats = self.fc(combined_feats)  # BxNxhidden_dim
        
        return output_feats





class GraphTransformerModel(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_heads, dropout=0.):
        super(GraphTransformerModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        
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
        batch_id, E_idx = batch['batch_id'], batch['E_idx']
        inv_distance_matrices = batch['inv_distance_matrices']
        heat_kernel_pe = batch['heat_kernel_pe']
        padding_mask = batch['unflattened_mask'].bool()  # Shape: (B, max_nodes)

        B = len(batch_id.unique())  # Batch size
        max_nodes = inv_distance_matrices.size(1)  # Max number of nodes in any graph in the batch

        # 1. Node embedding from _V using MLP, and then pad/stack according to batch_id
        node_embeds = self.node_mlp(_V)  # Apply MLP to node features
        padded_node_embeds = self._pad_and_stack(node_embeds, batch_id, max_nodes)  # Shape: (B, max_nodes, hidden_dim)

        # 2. Edge weight transformation from _E using MLP, and then pad/stack according to batch_id and E_idx
        edge_weights = self.edge_mlp(_E).squeeze(-1)  # Convert edge features to 1D weights
        padded_edge_weights = self._pad_and_stack_edges(edge_weights, batch_id, E_idx, max_nodes)  # Shape: (B, max_nodes, max_nodes)

        # 3. Replace zero values in padded_edge_weights with -inf
        # transformed_edge_weights = torch.where(padded_edge_weights == 0, float('-inf'), padded_edge_weights)
        transformed_edge_weights = padded_edge_weights

        # 4. Apply MHA layers for n_layers
        for _ in range(self.n_layers):
            padded_node_embeds = self.multi_head_attention(padded_node_embeds, inv_distance_matrices, heat_kernel_pe, transformed_edge_weights, padding_mask)
        
        return padded_node_embeds

    def multi_head_attention(self, node_embeds, inv_distance_matrices, heat_kernel_pe, transformed_edge_weights, padding_mask):
        # Batch size and max_nodes
        B, max_nodes, _ = node_embeds.size()
        d_k = self.hidden_dim // self.n_heads  # Dimension per head

        # 1. Linear projections for Q, K, V
        Q = self.query_proj(node_embeds)  # (B, max_nodes, hidden_dim)
        K = self.key_proj(node_embeds)    # (B, max_nodes, hidden_dim)
        V = self.value_proj(node_embeds)  # (B, max_nodes, hidden_dim)

        # 2. Reshape Q, K, V for multi-head attention: (B, n_heads, max_nodes, d_k)
        Q = Q.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)
        K = K.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)
        V = V.view(B, max_nodes, self.n_heads, d_k).transpose(1, 2)  # (B, n_heads, max_nodes, d_k)

        # 3. Scaled dot-product attention: (QK^T / sqrt(d_k))
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, n_heads, max_nodes, max_nodes)

        # 4. Add inv_distance_matrices and heat_kernel_pe to attention scores
        # attn_scores = attn_scores + inv_distance_matrices.unsqueeze(1) + heat_kernel_pe.unsqueeze(1)  # Shape: (B, n_heads, max_nodes, max_nodes)

        # 5. Multiply by transformed_edge_weights (element-wise multiplication)
        # attn_scores = attn_scores * transformed_edge_weights.unsqueeze(1)  # Shape: (B, n_heads, max_nodes, max_nodes)
        # attn_scores = torch.where(transformed_edge_weights.unsqueeze(1) == 0, -1e9, attn_scores * transformed_edge_weights.unsqueeze(1))

        # mult edge feats then add pe
        attn_scores = attn_scores * transformed_edge_weights.unsqueeze(1) + heat_kernel_pe.unsqueeze(1)
        attn_scores = torch.where(transformed_edge_weights.unsqueeze(1) == 0, -1e9, attn_scores)

        # 6. Apply padding mask (set scores to a large negative value where padding mask is False)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, max_nodes)
            attn_scores = attn_scores.masked_fill(~padding_mask, -1e9)  # Mask padded positions with large negative value

        # 7. Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 8. Compute the final weighted values
        attn_output = torch.matmul(attn_probs, V)  # (B, n_heads, max_nodes, d_k)

        # 9. Concatenate heads and project the result back to hidden_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, max_nodes, self.hidden_dim)  # (B, max_nodes, hidden_dim)
        attn_output = self.out_proj(attn_output)

        # 10. Apply dropout and residual connection for attention output
        attn_output = self.dropout_layer(attn_output)
        attn_output = attn_output + node_embeds  # Residual connection for attention
        attn_output = self.layer_norm(attn_output)  # Layer normalization for attention output

        # 11. Apply Feed-Forward Network (FFN) + residual connection + layer normalization
        ffn_output = self.ffn(attn_output)  # Feed-Forward network
        ffn_output = self.dropout_layer(ffn_output)  # Dropout after FFN
        ffn_output = ffn_output + attn_output  # Residual connection for FFN
        ffn_output = self.layer_norm_ffn(ffn_output)  # Layer normalization for FFN output

        return ffn_output

    def _pad_and_stack(self, features, batch_id, max_nodes):
        B = batch_id.max().item() + 1  # Batch size
        padded = torch.zeros((B, max_nodes, self.hidden_dim), device=features.device)
        
        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            padded[i, :len(node_indices), :] = features[node_indices]
        
        return padded

    def _pad_and_stack_edges(self, edge_weights, batch_id, E_idx, max_nodes):
        B = batch_id.max().item() + 1  # Batch size
        padded_edges = torch.zeros((B, max_nodes, max_nodes), device=edge_weights.device)

        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            min_node_id = node_indices.min().item()

            src, dst = E_idx[0, :], E_idx[1, :]
            local_edges_mask = (src >= min_node_id) & (src < min_node_id + node_indices.size(0))

            src_local = src[local_edges_mask] - min_node_id
            dst_local = dst[local_edges_mask] - min_node_id

            padded_edges[i, src_local, dst_local] = edge_weights[local_edges_mask]
            padded_edges[i, dst_local, src_local] = edge_weights[local_edges_mask]  # Assuming undirected edges

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


class SBModel(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(SBModel, self).__init__()
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

        self.encoder = GraphTransformerModel(hidden_dim=hidden_dim, n_layers=3, n_heads=8)

        l_max = 2
        num_scales = 4
        self.surface_encoder = PointCloudMessagePassing(2, 1, l_max, num_scales, hidden_dim)

        # New Transformer decoder and MLP for final prediction
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.tokenizer._token_to_id))
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        self._init_params()

        self.encode_t = 0
        self.decode_t = 0
    
    def forward(self, batch):
        ### pifold encoder
        h_V, h_P, P_idx, batch_id = batch['_V'], batch['_E'], batch['E_idx'], batch['batch_id']

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V_unflattened = self.encoder(h_V, h_P, batch)

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
        surfaces, biochem_feats = batch['surface'], batch['features']
        h_surface = self.surface_encoder(surfaces, biochem_feats)



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

        # return log_probs, log_probs0
        return {'log_probs': log_probs}
        
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
        # distance_matrices = []
        
        # for b in range(B):
        #     num_nodes = int(mask[b].sum().item())  # Number of valid nodes in this graph
        #     coords = X_ca[b, :num_nodes, :]  # (num_nodes, 3)
            
        #     # Pairwise Euclidean distance
        #     dist_matrix = torch.cdist(coords, coords)  # Shape: (num_nodes, num_nodes)
            
        #     # Zero-padding to max_nodes
        #     padded_dist_matrix = torch.zeros((max_nodes, max_nodes), device=device)
        #     padded_dist_matrix[:num_nodes, :num_nodes] = dist_matrix
        #     distance_matrices.append(padded_dist_matrix)

        # distance_matrices = torch.stack(distance_matrices, dim=0)  # Shape: (B, max_nodes, max_nodes)

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
            adj_matrix[edge_indices_b[0], edge_indices_b[1]] = 0
            # for src, dst in edge_indices_b.t():  # Transpose to get pairs of edges
            #     adj_matrix[src, dst] = 1
            #     adj_matrix[dst, src] = 1  # Assuming undirected graph

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