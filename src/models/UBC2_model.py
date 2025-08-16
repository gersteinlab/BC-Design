import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_scatter import scatter_sum, scatter_softmax
from src.tools import gather_nodes, _dihedrals, _get_rbf, _orientations_coarse_gl_tuple, Rigid, Rotation
from src.datasets.featurizer import rbf
import numpy as np
from transformers import AutoTokenizer
import math
import copy


pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


def build_MLP(n_layers,dim_in, dim_hid, dim_out, dropout = 0.0, activation=nn.ReLU, normalize=True):
    if normalize:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.BatchNorm1d(dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    else:
        layers = [nn.Linear(dim_in, dim_hid), 
                nn.Dropout(dropout), 
                activation()]
    for _ in range(n_layers - 2):
        layers.append(nn.Linear(dim_hid, dim_hid))
        if normalize:
            layers.append(nn.BatchNorm1d(dim_hid))
        layers.append(nn.Dropout(dropout))
        layers.append(activation())
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class PointCloudMessagePassing(nn.Module):
    def __init__(self, args, feat_dim, edge_dim, l_max, num_scales, hidden_dim, aggregation='concat', num_heads=4, num_mha_layers=1, bc_dropout=0.0):
        super(PointCloudMessagePassing, self).__init__()
        self.l_max = l_max
        self.num_scales = num_scales
        self.aggregation = aggregation
        self.num_heads = num_heads

        self.per_layer_dim = hidden_dim // 4

        self.bc_mask_max_rate = args.bc_mask_max_rate
        self.bc_mask_how = args.bc_mask_how

        # CLS token for biochemical features initialized with per_layer_dim
        self.biochem_cls_token = nn.Parameter(torch.randn(1 + 8, self.per_layer_dim))  # Adjusted dimension

        # bc mask token
        self.bc_mask_token = nn.Parameter(torch.randn(1, self.per_layer_dim))

        # Linear layer for feature dimension adjustment
        self.input_fc = nn.Linear(feat_dim, self.per_layer_dim)

        # # MHA module
        # self.mha = nn.MultiheadAttention(embed_dim=self.per_layer_dim, num_heads=num_heads, batch_first=True)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.per_layer_dim,      # 输入特征维度
            nhead=num_heads,                # 多头注意力的头数
            dim_feedforward=self.per_layer_dim * 4, # FFN的隐藏层维度
            dropout=bc_dropout,
            batch_first=True               
        )
        self.attention_layers = TransformerEncoder(encoder_layer, num_layers=args.bc_encoder_layer)

        # fc for residue connection
        self.res_conn_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.per_layer_dim, hidden_dim)
        )

        # Feature aggregation after MHA
        self.fc = nn.Linear(self.per_layer_dim * num_scales, hidden_dim)

    def forward(self, surfaces, biochem_feats, correspondences):
        B, N, _ = surfaces.shape

        ###### for inference with only backbone structure, bc input will be all nan
        # Find rows (over N) where any feature is nan, for each batch
        nan_rows = torch.any(torch.isnan(biochem_feats), dim=-1)  # shape: (B, N)
        
        # Elevate the biochemical features
        biochem_feats = self.input_fc(biochem_feats)  # BxNx(per_layer_dim)

        ###### for inference with only backbone structure, bc input will be all nan
        biochem_feats[nan_rows] = self.bc_mask_token

        if self.training:
            # randomly select a probability between 0 and self.bc_mask_max_rate
            bc_mask_rate = torch.rand(B, device=biochem_feats.device) * self.bc_mask_max_rate
            # select the indices of the biochemical features to be masked
            bc_mask_indices = torch.rand(B, N, device=biochem_feats.device) < bc_mask_rate[:, None]
            # mask the biochemical features
            if self.bc_mask_how == 'token':
                biochem_feats[bc_mask_indices] = self.bc_mask_token
            elif self.bc_mask_how == 'gauss':
                biochem_feats[bc_mask_indices] = torch.randn_like(biochem_feats[bc_mask_indices])

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
            
            # # Apply MHA over the padded regions
            # attn_output, _ = self.mha(padded_feats_flat, padded_feats_flat, padded_feats_flat, key_padding_mask=padding_mask_flat)
            attn_output = self.attention_layers(padded_feats_flat, src_key_padding_mask=padding_mask_flat)
            
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


class GeoFeat(nn.Module):
    def __init__(self, geo_layer, num_hidden, virtual_atom_num, dropout=0.0):
        super(GeoFeat, self).__init__()
        self.__dict__.update(locals())
        self.virtual_atom = nn.Linear(num_hidden, virtual_atom_num*3)
        self.virtual_direct = nn.Linear(num_hidden, virtual_atom_num*3)
        # self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+272, num_hidden, num_hidden, dropout)
        self.we_condition = build_MLP(geo_layer, 4*virtual_atom_num*3+9+16+32, num_hidden, num_hidden, dropout)
        self.MergeEG = nn.Linear(num_hidden+num_hidden, num_hidden)

    def forward(self, h_V, h_E, T_ts, edge_idx, h_E_0):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        num_edge = src_idx.shape[0]
        num_atom = h_V.shape[0]
        # print('shape of h_V', h_V.shape)
        # print('shape of h_E', h_E.shape)
        # print('shape of T_ts', T_ts.shape)
        # print('shape of T_ts._rots._rot_mats', T_ts._rots._rot_mats.shape)
        # print('shape of T_ts._trans', T_ts._trans.shape)
        # print('T_ts[0]._rots._rot_mats', T_ts[0]._rots._rot_mats)
        # print('T_ts[0]._rots._quats', T_ts[0]._rots._quats)
        # print('T_ts[0]._trans', T_ts[0]._trans)
        # print('shape of edge_idx', edge_idx.shape)
        # print('shape of h_E_0', h_E_0.shape)

        # ==================== point cross attention =====================
        V_local = self.virtual_atom(h_V).view(num_atom,-1,3)
        V_edge = self.virtual_direct(h_E).view(num_edge,-1,3)
        # print('V_local', V_local.shape)
        # # print max of src_idx
        # print('max of src_idx', src_idx.max())
        Ks = torch.cat([V_edge,V_local[src_idx].view(num_edge,-1,3)], dim=1)
        Qt = T_ts.apply(Ks)
        Ks = Ks.view(num_edge,-1)
        Qt = Qt.reshape(num_edge,-1)
        V_edge = V_edge.reshape(num_edge,-1)
        quat_st = T_ts._rots._rot_mats[:, 0].reshape(num_edge, -1)


        RKs = torch.einsum('eij,enj->eni', T_ts._rots._rot_mats[:,0], V_local[src_idx].view(num_edge,-1,3))
        QRK = torch.einsum('enj,enj->en', V_local[dst_idx].view(num_edge,-1,3), RKs)

        # H = torch.cat([Ks, Qt, quat_st, T_ts.rbf, h_E_0], dim=1)
        H = torch.cat([Ks, Qt, quat_st, T_ts.rbf, QRK], dim=1)
        G_e = self.we_condition(H)
        h_E = self.MergeEG(torch.cat([h_E, G_e], dim=-1))
        return h_E



class PiFoldAttn(nn.Module):
    def __init__(self, attn_layer, num_hidden, num_V, num_E, dropout=0.0):
        super(PiFoldAttn, self).__init__()
        self.__dict__.update(locals())
        self.num_heads = 4
        self.W_V = nn.Sequential(nn.Linear(num_E, num_hidden),
                                nn.GELU())
                                
        self.Bias = nn.Sequential(
                                nn.Linear(2*num_V+num_E, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,self.num_heads))
        self.W_O = nn.Linear(num_hidden, num_V, bias=False)
        self.gate = nn.Linear(num_hidden, num_V)


    def forward(self, h_V, h_E, edge_idx):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]
        h_V_skip = h_V

        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        num_nodes = h_V.shape[0]
        
        w = self.Bias(torch.cat([h_V[src_idx], h_E, h_V[dst_idx]],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1,n_heads, d) 
        attend = scatter_softmax(attend_logits, index=src_idx, dim=0)
        h_V = scatter_sum(attend*V, src_idx, dim=0).view([num_nodes, -1])

        h_V_gate = F.sigmoid(self.gate(h_V))
        dh = self.W_O(h_V)*h_V_gate

        h_V = h_V_skip + dh
        return h_V


class UpdateNode(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense = nn.Sequential(
            nn.BatchNorm1d(num_hidden),
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden),
            nn.BatchNorm1d(num_hidden)
        )
        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden))
    
    def forward(self, h_V, batch_id):
        dh = self.dense(h_V)
        h_V = h_V + dh

        # # ============== global attn - virtual frame
        # print('batch_id', batch_id)
        uni = batch_id.unique()
        mat = (uni[:,None] == batch_id[None]).to(h_V.dtype)
        mat = mat/mat.sum(dim=1, keepdim=True)
        c_V = mat@h_V

        h_V = h_V * F.sigmoid(self.V_MLP_g(c_V))[batch_id]
        return h_V

class UpdateEdge(nn.Module):
    def __init__(self, edge_layer, num_hidden, dropout=0.1):
        super(UpdateEdge, self).__init__()
        self.W = build_MLP(edge_layer, num_hidden*3, num_hidden, num_hidden, dropout, activation=nn.GELU, normalize=False)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.pred_quat = nn.Linear(num_hidden,8)

    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_E = self.norm(h_E + self.W(h_EV))

        return h_E


class GeneralGNN(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer,
                 num_hidden, 
                 virtual_atom_num=32, 
                 dropout=0.1,
                 mask_rate=0.15):
        super(GeneralGNN, self).__init__()
        self.__dict__.update(locals())
        self.geofeat = GeoFeat(geo_layer, num_hidden, virtual_atom_num, dropout)
        self.attention = PiFoldAttn(attn_layer, num_hidden, num_hidden, num_hidden, dropout) 
        self.update_node = UpdateNode(num_hidden)
        self.update_edge = UpdateEdge(edge_layer, num_hidden, dropout)
        self.mask_token = nn.Embedding(2, num_hidden)
    
    def get_rand_idx(self, h_V, mask_rate):
        num_N = int(h_V.shape[0] * mask_rate)  # 要选择的样本数量，即15%
        indices = torch.randperm(h_V.shape[0], device=h_V.device)
        selected_indices = indices[:num_N]
        return selected_indices
        
    def forward(self, h_V, h_E, T_ts, edge_idx, batch_id, h_E_0):
        if self.training:
            selected_indices = self.get_rand_idx(h_V, self.mask_rate)
            h_V[selected_indices] = self.mask_token.weight[0]

            selected_indices = self.get_rand_idx(h_E, self.mask_rate)
            h_E[selected_indices] = self.mask_token.weight[1]

            # selected_indices = (selected_indices[:,None,None] == edge_idx[None]).sum(dim=(0,1))
            # h_E[selected_indices] = self.mask_token.weight[1]

            # h_V  = h_V + torch.rand_like(h_V)*self.mask_rate*1
            # h_E  = h_E + torch.rand_like(h_E)*self.mask_rate*1
        
        h_E = self.geofeat(h_V, h_E, T_ts, edge_idx, h_E_0)
        h_V = self.attention(h_V, h_E, edge_idx)
        h_V = self.update_node(h_V, batch_id)
        h_E = self.update_edge( h_V, h_E, T_ts, edge_idx, batch_id )
        return h_V, h_E


class StructureEncoder(nn.Module):
    def __init__(self, 
                 geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 encoder_layer,
                 hidden_dim, 
                 dropout=0,
                 mask_rate=0.15):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        self.__dict__.update(locals())
        self.encoder_layers = nn.ModuleList([GeneralGNN(geo_layer, 
                 attn_layer,
                 ffn_layer,
                 edge_layer, 
                 hidden_dim, 
                 dropout=dropout,
                 mask_rate=mask_rate) for i in range(encoder_layer)])
        self.s = nn.Linear(hidden_dim, 1)
    
    def forward(self, h_S,
                    T, 
                    h_V,
                    h_E,
                    T_ts, 
                    edge_idx,
                    batch_id, h_E_0):
        # No global frame handling needed - work only with local components
        outputs = []
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, T_ts, edge_idx, batch_id, h_E_0)
            outputs.append(h_V.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        S = F.sigmoid(self.s(outputs))
        output = torch.einsum('nkc, nkb -> nbc', outputs, S).squeeze(1)
        return output
        

class UniIFEncoder(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(UniIFEncoder, self).__init__()
        self.__dict__.update(locals())
        self.hidden_dim = args.hidden_dim
        geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate = args.geo_layer, args.attn_layer, args.node_layer, args.edge_layer, args.encoder_layer, args.hidden_dim, args.dropout, args.mask_rate
    
        self.node_embedding = build_MLP(2, 76, hidden_dim, hidden_dim)
        self.edge_embedding = build_MLP(2, 196+16, hidden_dim, hidden_dim)
        self.encoder = StructureEncoder(geo_layer, attn_layer, node_layer, edge_layer, encoder_layer, hidden_dim, dropout, mask_rate)
        # self.decoder = MLPDecoder(hidden_dim)
        self.chain_embeddings = nn.Embedding(2, 16)

        # CLS token for structural features
        self.struct_cls_token = nn.Parameter(torch.randn(1 + 8, hidden_dim))

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, batch, num_global=3):
        h_V, h_E, edge_idx, batch_id, chain_features = batch['_V'], batch['_E'], batch['edge_idx'], batch['batch_id'], batch['chain_features']
        correspondences = batch['correspondences']
        # Remove global virtual frame variables
        T = Rigid(Rotation(batch['T_rot']), batch['T_trans'])
        T_ts = Rigid(Rotation(batch['T_ts_rot']), batch['T_ts_trans'])
        # rbf_ts = batch['rbf_ts']
        # T_ts.rbf = rbf_ts
        h_E = torch.cat([h_E, self.chain_embeddings(chain_features)], dim=-1)

        h_E_0 = h_E

        node_embeds = self.node_embedding(h_V)

        # Prepare for adding CLS tokens
        B = len(batch_id.unique())   # Batch size
        max_nodes = 9 + max([(batch_id == i).sum().item() for i in range(B)])  # 9 CLS + max residues per batch
        # print('max_nodes-9', max_nodes-9)
        # print('max lenght', batch['lengths'].max().item())
        # Add CLS token embeddings to node embeddings
        # Directly add CLS tokens to the beginning of each batch in the original flattened node_embeds
        cls_tokens = self.struct_cls_token.expand(B, -1, -1)  # (B, 9, hidden_dim)
        # For each batch, prepend 9 CLS tokens to the corresponding node embeddings
        node_embeds_with_cls = []
        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            this_node_embeds = node_embeds[node_indices]  # (num_nodes_i, hidden_dim)
            this_cls_tokens = cls_tokens[i]  # (9, hidden_dim)
            node_embeds_with_cls.append(torch.cat([this_cls_tokens, this_node_embeds], dim=0))  # (9 + num_nodes_i, hidden_dim)

        h_V = torch.cat(node_embeds_with_cls, dim=0)  # (sum_i (9 + num_nodes_i), hidden_dim)

        batch_id_with_cls = []
        # batch_id identifies the which batch each node belongs to
        # get unique batch_id
        unique_batch_id = batch_id.unique()
        for i in unique_batch_id:
            # 9 i's before the original i's, use tensor
            batch_id_with_cls.append(torch.full((9 + (batch_id == i).sum().item(),), i, device=batch_id.device))
        batch_id_with_cls = torch.cat(batch_id_with_cls, dim=0)

        # h_E = self.edge_embedding(h_E).squeeze(-1)  # Convert edge features to 1D weights
        h_E = self.edge_embedding(h_E)
        h_E, edge_idx, T_ts = self._pad_and_stack_edges(h_E, batch_id, edge_idx, T_ts, max_nodes, correspondences)  # Shape: (total_edges, hidden_dim), (2, total_edges)

        h_S = None
        # No global frame features needed

        # Get structural node embeddings from encoder (without global frames)
        node_embeds = self.encoder(h_S,
                                T, 
                                h_V,
                                h_E,
                                T_ts, 
                                edge_idx,
                                batch_id_with_cls, h_E_0)

        unflattened_node_embeds = self._pad_and_stack(node_embeds, batch_id_with_cls, max_nodes)
        
        return unflattened_node_embeds
        

    def _get_features(self, batch):
        return batch

    def _pad_and_stack(self, features, batch_id, max_nodes):
        """Pad and stack node features."""
        B = batch_id.max().item() + 1  # Batch size
        padded = torch.zeros((B, max_nodes, self.hidden_dim), device=features.device)
        
        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            padded[i, :len(node_indices), :] = features[node_indices]
        
        return padded

    def _pad_and_stack_edges(self, edge_weights, batch_id, E_idx, T_ts, max_nodes, correspondences):
        """Convert edge features to include CLS tokens and return in 2D format with updated edge indices."""
        B = batch_id.max().item() + 1  # Batch size
        
        is_vector = True
        hidden_dim = edge_weights.shape[-1]

        # Collect all new edge features and indices
        new_edge_features = []
        new_edge_indices = []
        new_rot_mats = []
        new_trans = []
        # shape of T_ts._rots._rot_mats torch.Size([#edges, 1, 3, 3])                                                                 
        # shape of T_ts._trans torch.Size([#edges, 1, 3]) 
        
        # Calculate batch offsets for global node indexing
        batch_sizes = [(batch_id == i).sum().item() for i in range(B)]
        batch_offsets = [0]
        for i in range(B):
            batch_offsets.append(batch_offsets[-1] + batch_sizes[i] + 9)  # +9 for CLS tokens per batch

        for i in range(B):
            node_indices = (batch_id == i).nonzero(as_tuple=True)[0]
            min_node_id = node_indices.min().item()
            num_nodes = len(node_indices)

            src, dst = E_idx[0, :], E_idx[1, :]
            local_edges_mask = (src >= min_node_id) & (src < min_node_id + num_nodes)

            batch_offset = batch_offsets[i]

            # 1. Process original edges (shifted by +9 for CLS tokens)
            if local_edges_mask.any():
                src_local = src[local_edges_mask] - min_node_id + 9  # +9 for CLS tokens
                dst_local = dst[local_edges_mask] - min_node_id + 9  # +9 for CLS tokens
                
                # Add to global indices
                global_src = src_local + batch_offset
                global_dst = dst_local + batch_offset
                
                # Original edge features
                original_edge_features = edge_weights[local_edges_mask]
                new_edge_features.append(original_edge_features)
                
                new_edge_indices.append(torch.stack([global_src, global_dst]))
                # T_ts is a list, so we need to use nonzero indices to select elements
                local_edge_indices = local_edges_mask.nonzero(as_tuple=True)[0]
                new_rot_mats.append(T_ts._rots._rot_mats[local_edge_indices])
                new_trans.append(T_ts._trans[local_edge_indices])

                # 2. Determine CLS edge feature
                cls_edge_feature = torch.zeros(hidden_dim, device=edge_weights.device)
                # determine the T_ts for the CLS edge
                cls_rot_mats = torch.zeros_like(T_ts._rots._rot_mats[0])
                cls_trans = torch.zeros_like(T_ts._trans[0])

            # 3. Add global CLS token connections (index 0)
            global_cls_idx = batch_offset + 0  # Global CLS index for this batch
            global_node_indices = torch.arange(9, 9 + num_nodes, device=edge_weights.device) + batch_offset
            
            # CLS -> Nodes
            cls_to_nodes_src = torch.full((num_nodes,), global_cls_idx, device=edge_weights.device)
            cls_to_nodes_dst = global_node_indices
            cls_to_nodes_features = cls_edge_feature.unsqueeze(0).repeat(num_nodes, 1)
            
            new_edge_indices.append(torch.stack([cls_to_nodes_src, cls_to_nodes_dst]))
            new_edge_features.append(cls_to_nodes_features)
            new_rot_mats.append(cls_rot_mats.unsqueeze(0).repeat(num_nodes, 1, 1, 1))
            new_trans.append(cls_trans.unsqueeze(0).repeat(num_nodes, 1, 1))
            
            # Nodes -> CLS  
            new_edge_indices.append(torch.stack([cls_to_nodes_dst, cls_to_nodes_src]))
            new_edge_features.append(cls_to_nodes_features)
            new_rot_mats.append(cls_rot_mats.unsqueeze(0).repeat(num_nodes, 1, 1, 1))
            new_trans.append(cls_trans.unsqueeze(0).repeat(num_nodes, 1, 1))

            # 4. Add subarea CLS token connections (indices 1-8)
            if len(correspondences) > i and len(correspondences[i]) > 0:
                for sub_idx, (ca_neighbors, _) in enumerate(correspondences[i], start=1):  # Limit to 8 subareas
                    if len(ca_neighbors) > 0:
                        global_subarea_idx = batch_offset + sub_idx  # Global subarea CLS index
                        global_ca_neighbors = torch.tensor(ca_neighbors, device=edge_weights.device) + 9 + batch_offset
                        
                        # Subarea CLS -> CA neighbors
                        subarea_to_ca_src = torch.full((len(ca_neighbors),), global_subarea_idx, device=edge_weights.device)
                        subarea_to_ca_dst = global_ca_neighbors
                        subarea_to_ca_features = cls_edge_feature.unsqueeze(0).repeat(len(ca_neighbors), 1)
                        
                        new_edge_indices.append(torch.stack([subarea_to_ca_src, subarea_to_ca_dst]))
                        new_edge_features.append(subarea_to_ca_features)
                        new_rot_mats.append(cls_rot_mats.unsqueeze(0).repeat(len(ca_neighbors), 1, 1, 1))
                        new_trans.append(cls_trans.unsqueeze(0).repeat(len(ca_neighbors), 1, 1))
                        
                        # CA neighbors -> Subarea CLS
                        new_edge_indices.append(torch.stack([subarea_to_ca_dst, subarea_to_ca_src]))
                        new_edge_features.append(subarea_to_ca_features)
                        new_rot_mats.append(cls_rot_mats.unsqueeze(0).repeat(len(ca_neighbors), 1, 1, 1))
                        new_trans.append(cls_trans.unsqueeze(0).repeat(len(ca_neighbors), 1, 1))

        # Concatenate all edge features and indices
        if new_edge_features:
            final_edge_features = torch.cat(new_edge_features, dim=0)  # (total_edges, hidden_dim)
            final_edge_indices = torch.cat(new_edge_indices, dim=1)    # (2, total_edges)
            final_rot_mats = torch.cat(new_rot_mats, dim=0)
            final_trans = torch.cat(new_trans, dim=0)
            T_ts._rots._rot_mats = final_rot_mats
            T_ts._trans = final_trans
            rbf_ts = rbf(T_ts._trans.norm(dim=-1), 0, 50, 16)[:,0].view(final_edge_features.shape[0],-1)
            T_ts.rbf = rbf_ts
        else:
            # Handle empty case
            final_edge_features = torch.zeros((0, hidden_dim), device=edge_weights.device)
            final_edge_indices = torch.zeros((2, 0), device=edge_weights.device, dtype=torch.long)

        return final_edge_features, final_edge_indices, T_ts


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


class UBC2Model(nn.Module):
    def __init__(self, args, queue_size=64, **kwargs):
        """ Graph labeling network """
        super(UBC2Model, self).__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        dropout = args.dropout
        self.modal_mask_ratio = args.modal_mask_ratio
        self.contrastive_pretrain = args.contrastive_pretrain
        self.contrastive_pretrain_both = args.contrastive_pretrain_both
        self.contrastive_loss_global_alpha = args.contrastive_loss_global_alpha
        self.contrastive_loss_local_alpha = args.contrastive_loss_local_alpha

        self.if_warmup_train = args.if_warmup_train

        # self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
        self.tokenizer = MyTokenizer()

        # self.encoder = GraphTransformerModel(hidden_dim=hidden_dim, n_layers=self.args.gt_layers, n_heads=8)
        self.encoder = UniIFEncoder(args)

        l_max = 2
        num_scales = 4
        # best
        self.surface_encoder = PointCloudMessagePassing(args, 2, 1, l_max, num_scales, hidden_dim)
        # hyperparam exp
        # self.surface_encoder = PointCloudMessagePassingMultiple(2, 1, l_max, num_scales, hidden_dim, num_mha_layers=4)

        # New Transformer decoder and MLP for final prediction
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
        # self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 33)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)

        self.contrastive_learning = args.contrastive_learning

        # Temperature for contrastive learning
        self.temperature = 0.1
        self.queue_size = queue_size

        # Initialize queues for structural and biochemical CLS tokens
        self.struct_queue = nn.Parameter(torch.zeros(queue_size, hidden_dim), requires_grad=False)
        self.biochem_queue = nn.Parameter(torch.zeros(queue_size, hidden_dim), requires_grad=False)
        self.queue_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

        self._init_params()

        if self.contrastive_pretrain:
            modules_to_freeze = {
                "encoder": self.encoder,
                "transformer_decoder": self.transformer_decoder,
                "mlp": self.mlp
            }

            for name, module in modules_to_freeze.items():
                print(f"--- Freezing module: '{name}'")
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
        elif self.contrastive_pretrain_both:
            modules_to_freeze = {
                "transformer_decoder": self.transformer_decoder,
                "mlp": self.mlp
            }

            for name, module in modules_to_freeze.items():
                print(f"--- Freezing module: '{name}'")
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()            
        elif self.if_warmup_train:
            modules_to_freeze = {
                "encoder": self.encoder,
                "surface_encoder": self.surface_encoder
            }
            for name, module in modules_to_freeze.items():
                print(f"--- Freezing module: '{name}'")
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
            
    def forward(self, batch):
        batch_id = batch['batch_id']
        h_V_unflattened = self.encoder(batch)
        
        # Manually extract the CLS tokens (global + subarea) from the structure encoder
        struct_cls_tokens = h_V_unflattened[:, :9, :]  # First 9 tokens: global (0th) + subarea (1st to 8th)
        h_V_unflattened = h_V_unflattened[:, 9:, :]  # The rest of the node embeddings

        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        mask_unflattened = torch.zeros(batch_size, max_length, device=h_V_unflattened.device)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()

        ### surface encoder
        surfaces, biochem_feats, correspondences = batch['surface'], batch['features'], batch['correspondences']

        if self.training:
            # generate a random number between 0 and 1
            random_number = torch.rand(1).item()
            if random_number < self.modal_mask_ratio:
                # biochem_feats = torch.ones_like(biochem_feats) * 100
                biochem_feats = torch.randn_like(biochem_feats)

        ################## real start
        h_surface = self.surface_encoder(surfaces, biochem_feats, correspondences)

        # Manually extract the CLS tokens (global + subarea) from the biochemical encoder
        biochem_cls_tokens = h_surface[:, -9:, :]  # Last 9 tokens: global (0th) + subarea (1st to 8th)
        h_surface = h_surface[:, :-9, :]  # The rest of the biochemical node embeddings


        ### new decoder
        if self.contrastive_pretrain or self.contrastive_pretrain_both:
            log_probs = 0
        else:
            ss_connection_mask = batch['ss_connection']
            ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

            # Transformer decoder to fuse h_V_unflattened and h_surface
            # Add positional encoding to the inputs of the Transformer decoder
            h_V_unflattened = self.positional_encoding(h_V_unflattened)
            
            decoder_output = self.transformer_decoder(
                h_V_unflattened, h_surface, 
                tgt_key_padding_mask=target_padding_mask, 
                # ablation
                memory_mask=ss_connection_mask
            )
            ################## real end

            ################## exp
            # decoder_output = h_V_unflattened
            ################## exp

            # Flatten decoder_output and remove padding
            mask = mask_unflattened.bool()
            decoder_output = decoder_output[mask]

            # Predict labels using MLP
            logits = self.mlp(decoder_output)
            log_probs = F.log_softmax(logits, dim=-1)

        ################## real start
        # Contrastive learning
        if (self.training and random_number < self.modal_mask_ratio) or not self.contrastive_learning:
            contrastive_loss = 0
        else:
            contrastive_loss_global = self._contrastive_loss(struct_cls_tokens[:, 0, :], biochem_cls_tokens[:, 0, :])  # Global CLS
            contrastive_loss_subarea = self._contrastive_loss_subarea(struct_cls_tokens[:, 1:, :], biochem_cls_tokens[:, 1:, :])  # Subarea CLS
            contrastive_loss = self.contrastive_loss_global_alpha * contrastive_loss_global + self.contrastive_loss_local_alpha * contrastive_loss_subarea

        # Update queues with current batch global CLS tokens
        self._dequeue_and_enqueue(struct_cls_tokens[:, 0, :], biochem_cls_tokens[:, 0, :])
        ################## real end
        ################## exp
        # contrastive_loss = 0
        ################## exp

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

    def _get_features(self, batch):
        return batch
    
    
