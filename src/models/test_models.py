import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch_scatter
from src.tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
import numpy as np
from transformers import AutoTokenizer
import os
from src.modules.pifold_module import *
import gc
import time


pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


def compute_graph_laplacian(num_nodes, edge_index, edge_weight=None):
    adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0 if edge_weight is None else edge_weight
    degree = torch.sum(adj, dim=1)
    D = torch.diag(degree)
    L = D - adj
    return L

def compute_heat_kernel(L, t):
    H_t = torch.matrix_exp(-t * L)
    return H_t


def compute_heat_distribution(H_t, h_0):
    h_t = torch.matmul(H_t, h_0)
    return h_t


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


class EdgeEnhancedPositionalEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeEnhancedPositionalEncoder, self).__init__()
        self.mlp_eepe = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h_i, pe_i, edge_weights):
        # Compute edge-enhanced positional encoding
        eepe_i = self.mlp_eepe(torch.cat([pe_i.unsqueeze(1), edge_weights], dim=-1))
        h_aug = h_i + eepe_i
        return h_aug


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        # Linear transformation for computing attention scores
        self.W = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h_aug, edge_index):
        # Gather h_aug for each edge
        h_i = h_aug[edge_index[0]]  # Source node features [num_edges, hidden_dim]
        h_j = h_aug[edge_index[1]]  # Target node features [num_edges, hidden_dim]

        # Concatenate features and apply linear transformation
        h_concat = torch.cat([h_i, h_j], dim=-1)  # [num_edges, 2 * hidden_dim]
        Wh_ij = self.W(h_concat)  # [num_edges, hidden_dim]

        # Apply LeakyReLU
        e_ij = self.leaky_relu(Wh_ij)  # [num_edges, hidden_dim]

        # Normalize attention scores with scatter_softmax
        alpha_ij = torch_scatter.scatter_softmax(e_ij, edge_index[0], dim=0)  # [num_edges]

        return alpha_ij


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(MessagePassingLayer, self).__init__()
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, alpha_ij, h_aug_j, edge_index):
        # Apply linear transformation to the neighboring node features
        W_hoj = self.W_o(h_aug_j)  # [num_edges, hidden_dim]

        # Weight the neighboring features by the attention scores
        weighted_messages = alpha_ij * W_hoj  # [num_edges, hidden_dim]

        # Aggregate the messages for each node
        h_prime = torch_scatter.scatter_add(weighted_messages, edge_index[0], dim=0)  # [num_nodes, hidden_dim]

        return h_prime


class GatedMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedMLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(output_dim, momentum=0.1)

    def forward(self, x):
        # MLP part
        x_fc = self.fc(x)
        # Gate part (using sigmoid as a gating function)
        x_gate = torch.sigmoid(self.gate(x))
        # Element-wise multiplication (gate)
        x_out = x_fc * x_gate
        x_out = self.norm(x_out) 
        # Activation
        return self.activation(x_out)



class HeavyEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(HeavyEncoder, self).__init__()
        self.positional_encoder = EdgeEnhancedPositionalEncoder(hidden_dim)
        self.attention_layer = AttentionLayer(hidden_dim)
        self.message_passing_layer = MessagePassingLayer(hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim, momentum=0.1)

    def forward(self, h, pe, edge_weights, edge_index):
        # Step 1: Edge-enhanced positional encoding
        # h_aug = self.positional_encoder(h, pe, edge_weights)
        h_aug = h

        # Step 2: Attention
        alpha_ij = self.attention_layer(h_aug, edge_index)
        
        # Step 3: Message passing
        h_prime = self.message_passing_layer(alpha_ij, h_aug[edge_index[1]], edge_index)

        # Normalization
        h_prime = self.norm(h_prime) 
        
        return h_prime



class TestModel0831(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(TestModel0831, self).__init__()
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
        # # self.biochem_embedding = nn.Linear(2, hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)
        # # self.norm_biochem = nn.BatchNorm1d(hidden_dim)
        # Replacing with Gated MLP
        self.node_gated_mlp = GatedMLP(node_features, hidden_dim)
        self.edge_gated_mlp = GatedMLP(edge_features, hidden_dim)

        # self.W_v = nn.Sequential(
        #     nn.Linear(node_features, hidden_dim, bias=True),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim, bias=True),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim, bias=True)
        # )
        
        # self.W_e = nn.Linear(edge_features, hidden_dim, bias=True) 
        # self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_b = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )

        # New Transformer decoder and MLP for final prediction
        decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)

        # self.heavy_encoder = HeavyEncoder(hidden_dim)
        self.heavy_encoders = nn.ModuleList([HeavyEncoder(hidden_dim) for _ in range(3)])

        self.t = kwargs.get('t', 0.5)

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
        t1 = time.time()

        # h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        # h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        h_V = self.node_gated_mlp(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.edge_gated_mlp(self.norm_edges(self.edge_embedding(h_P)))

        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        h_V_unflattened = torch.zeros(batch_size, max_length, h_V.size(-1), device=h_V.device)
        mask_unflattened = torch.zeros(batch_size, max_length, device=h_V.device)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            h_V_unflattened[idx, :mask.sum()] = h_V[mask]
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()
        memory_padding_mask = ~batch['surface_mask']  # [batch_size, seq_len]
        t2 = time.time()




        ### surface encoder
        biochem_feats = batch['features']
        # h_surface = self.surface_encoder(surfaces, biochem_feats, pcs, memory_padding_mask)
        # h_biochem = self.W_b(self.norm_biochem(self.biochem_embedding(biochem_feats)))
        h_biochem = self.W_b(biochem_feats)
        # h_biochem[memory_padding_mask] = 0


        ### new decoder
        ss_connection_mask = batch['ss_connection']
        ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

        # Transformer decoder to fuse h_V_unflattened and h_surface
        # Add positional encoding to the inputs of the Transformer decoder
        h_V_unflattened = self.positional_encoding(h_V_unflattened)
        
        decoder_output = self.transformer_decoder(
            h_V_unflattened, h_biochem, 
            tgt_key_padding_mask=target_padding_mask, 
            memory_key_padding_mask=memory_padding_mask,
            memory_mask=ss_connection_mask
        )

        # Flatten decoder_output and remove padding
        mask = mask_unflattened.bool()
        decoder_output = decoder_output[mask]

        # # heavy encoder
        # # Step 1: Initialize edge weights for each node
        # num_nodes = h_V.size(0)
        # edge_weights = torch.zeros(num_nodes, h_P.size(1), device=h_P.device)

        # # Step 2: Accumulate edge embeddings for each node based on E_idx
        # # Use scatter_add to sum embeddings for corresponding source nodes
        # edge_weights = torch_scatter.scatter_add(h_P, P_idx[0], dim=0)

        # positional_encodings = batch['positional_encodings']
        # # encoder_output = self.heavy_encoder(decoder_output, positional_encodings, edge_weights, P_idx)
        # # Pass through the stack of HeavyEncoders
        # for encoder in self.heavy_encoders:
        #     decoder_output = encoder(decoder_output, positional_encodings, edge_weights, P_idx)

        encoder_output = decoder_output

        # Predict labels using MLP
        logits = self.mlp(encoder_output)
        log_probs = F.log_softmax(logits, dim=-1)


        
        ### original decoder
        # log_probs, logits = self.decoder(h_V, batch_id, self.token_mask)
                
        # log_probs, logits = self.decoder2(h_V, logits, batch_id)
        t3 = time.time()

        self.encode_t += t2-t1
        self.decode_t += t3-t2
        # return log_probs, log_probs0
        return {'log_probs': log_probs}
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
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
        
        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)
        
        d_chains = ((chain_encoding[dst.long()] - chain_encoding[src.long()])==0).long().reshape(-1)   
        chain_embed = self._idx_embeddings(d_chains)
        _E = torch.cat([_E, chain_embed], dim=-1)

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]

        mask = torch.masked_select(mask, mask_bool)
        batch.update({'X':X,
                'S':S,
                'score':score,
                '_V':_V,
                '_E':_E,
                'E_idx':E_idx,
                'batch_id': batch_id,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding})

        # Create positional encodings using heat kernel method
        positional_encodings = torch.zeros(X.size(0), device=X.device)

        unique_batch_ids = batch['batch_id'].unique(sorted=True)
        for b in unique_batch_ids:
            node_mask = (batch['batch_id'] == b)
            node_indices = torch.where(node_mask)[0]
            num_nodes = node_indices.size(0)

            edge_mask = torch.isin(E_idx[0], node_indices) & torch.isin(E_idx[1], node_indices)
            edges = E_idx[:, edge_mask]
            edges -= node_indices.min()  # Adjust edge indices to the local graph

            # Step 1: Compute Laplacian
            L = compute_graph_laplacian(num_nodes, edges)

            # Step 2: Compute heat kernel
            H_t = compute_heat_kernel(L, self.t)

            # Step 3: Compute initial heat vector h(0)
            node_coordinates = X[node_mask][:, 1, :]
            centroid = node_coordinates.mean(dim=0)  # Calculate the centroid
            dists = torch.norm(node_coordinates - centroid, dim=1)  # Compute distances to the centroid
            center_node_idx = torch.argmin(dists)  # Find the node closest to the centroid
            
            h_0 = torch.zeros(num_nodes, device=device)
            h_0[center_node_idx] = 1.0  # One-hot vector with the center node as 1

            # Step 4: Compute heat distribution h(t)
            h_t = compute_heat_distribution(H_t, h_0)

            # Step 5: Assign h_t to positional encodings
            positional_encodings[node_mask] = h_t

        batch.update({'positional_encodings': positional_encodings})

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



###################### NNNNNNNEEEEEEEEWWWWWWWWWWWW


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
from scipy.special import sph_harm
import numpy as np

# # 球谐基函数计算
# def spherical_harmonic_basis(l_max, theta, phi):
#     """
#     计算球谐基函数 Y_l^m(theta, phi)
    
#     参数:
#     - l_max: 最大阶数 l
#     - theta: 极角 (BxNxN)
#     - phi: 方位角 (BxNxN)
    
#     返回:
#     - basis: 球谐基函数值 (BxNxNx(2*l_max+1))
#     """
#     start_time = time.time()  # 记录起始时间
#     B, N, _ = theta.shape  # 获取批次大小 B 和点的数量 N
#     harmonics = []
    
#     for l in range(l_max + 1):
#         for m in range(-l, l + 1):
#             # 计算 scipy.special.sph_harm(m, l, phi, theta)
#             Y_lm = sph_harm(m, l, phi.cpu().numpy(), theta.cpu().numpy())  # scipy 计算返回复数
#             Y_lm = torch.tensor(np.real(Y_lm), dtype=torch.float32)  # 只取实部，BxNxN
#             harmonics.append(Y_lm)
    
#     # 将所有球谐基函数拼接为 (BxNxNx(2l_max+1))
#     harmonics = torch.stack(harmonics, dim=-1)  # BxNxNx(2l_max+1)
#     end_time = time.time()  # 记录结束时间
#     print(f"Time taken to compute spherical harmonic basis: {end_time - start_time:.4f} seconds")
    
#     return harmonics


def factorial(n):
    """Compute factorial using PyTorch for compatibility with GPUs."""
    return torch.exp(torch.lgamma(n + 1))

def associated_legendre_polynomial(l, m, x):
    """Compute the Associated Legendre Polynomial P_l^m(x) using PyTorch."""
    pmm = torch.ones_like(x) if m == 0 else (-1)**m * factorial(torch.tensor(2 * m - 1)) / (2**m * factorial(torch.tensor(m))) * (1 - x**2)**(m / 2)
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def spherical_harmonic(l, m, theta, phi):
    """Compute the spherical harmonic Y_l^m for given theta and phi using PyTorch."""
    legendre = associated_legendre_polynomial(l, abs(m), torch.cos(theta))
    if m < 0:
        legendre *= (-1)**m * factorial(torch.tensor(l - abs(m))) / factorial(torch.tensor(l + abs(m)))
    return torch.sqrt((2 * l + 1) / (4 * torch.pi) * factorial(torch.tensor(l - m)) / factorial(torch.tensor(l + m))) * legendre * torch.exp(1j * m * phi)

def spherical_harmonic_basis(l_max, theta, phi):
    """
    Compute spherical harmonic basis Y_l^m(theta, phi) using PyTorch.
    
    Parameters:
    - l_max: Maximum degree l
    - theta: Polar angle (BxNxN, radians)
    - phi: Azimuthal angle (BxNxN, radians)
    
    Returns:
    - harmonics: Spherical harmonic values (BxNxNx(2*l_max+1))
    """
    # start_time = time.time()
    
    B, N, _ = theta.shape  # Get batch size B and number of points N
    harmonics = []
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Calculate spherical harmonics using PyTorch
            Y_lm = spherical_harmonic(l, m, theta, phi)
            harmonics.append(Y_lm.real)  # If you only need the real part
    
    # Stack all harmonics into a tensor of shape (BxNxNx(2*l_max+1))
    harmonics = torch.stack(harmonics, dim=-1)
    
    # end_time = time.time()
    # print(f"Time taken to compute spherical harmonic basis: {end_time - start_time:.4f} seconds")
    
    return harmonics

# 计算相对球坐标系
def relative_spherical_coordinates(xyz_i, xyz_j):
    """
    计算相对位置的球坐标系表示
    
    参数:
    - xyz_i: 中心点坐标 (BxNx3)
    - xyz_j: 邻居点坐标 (BxNx3)
    
    返回:
    - r: 距离 (BxNxN)
    - theta: 极角 (BxNxN)
    - phi: 方位角 (BxNxN)
    """
    rel_pos = xyz_j - xyz_i  # 计算相对位置
    r = torch.norm(rel_pos, dim=-1)  # 距离
    theta = torch.atan2(rel_pos[..., 1], rel_pos[..., 0])  # 极角
    phi = torch.acos(rel_pos[..., 2] / (r + 1e-8))  # 方位角
    return r, theta, phi

# 基于距离的邻接矩阵生成函数，返回邻居对的索引和对应的距离
def compute_edges(distances, thr_r):
    """
    计算邻接列表和对应的距离值，只存储有边的点对
    
    参数:
    - distances: 点对之间的距离 (BxNxN)
    - thr_r: 阈值，距离小于此值的点对保留
    
    返回:
    - edge_indices: 邻接点对的索引 (2xE)，E 是有边的点对数
    - edge_distances: 对应点对的距离 (E)
    """
    B, N, _ = distances.shape
    
    # 找到小于阈值的邻居
    edge_mask = distances < thr_r  # BxNxN
    
    # 获取有边的点对索引 (num_edges, 3) -> (B, i, j)
    edge_indices = edge_mask.nonzero(as_tuple=False)  # (num_edges, 3)
    
    # 展平所有batch维度，将批次维度合并到索引中
    batch_indices = edge_indices[:, 0]  # 批次索引 B
    i_indices = edge_indices[:, 1]  # 源节点 i
    j_indices = edge_indices[:, 2]  # 目标节点 j
    
    # 重新排列为 2xE 形式 (源节点, 目标节点)，合并了batch维度
    edge_indices = torch.stack([i_indices + batch_indices * N, j_indices + batch_indices * N], dim=0)  # 2xE
    
    # 根据 edge_mask 获取对应的距离值 (E)
    edge_distances = distances[edge_mask]  # E
    
    return edge_indices, edge_distances

# Attention 计算模块
class AttentionNet(nn.Module):
    def __init__(self, feat_dim, edge_dim, sph_dim):
        super(AttentionNet, self).__init__()
        # 输入是 节点特征 + 边特征 (距离) + 球谐特征
        self.fc = nn.Sequential(
            nn.Linear(2 * feat_dim + edge_dim + sph_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, node_i_feats, node_j_feats, edge_feats):
        """
        计算每对节点之间的注意力权重
        
        参数:
        - node_i_feats: 源节点 i 的特征 (ExC)
        - node_j_feats: 目标节点 j 的特征 (ExC)
        - edge_feats: 边的特征 (ExF)，包括距离和球谐特征
        
        返回:
        - attn_weights: 经过 softmax 归一化的注意力权重 (E)
        """
        # 拼接节点 i 和节点 j 的特征，以及边特征
        combined_feats = torch.cat([node_i_feats, node_j_feats, edge_feats], dim=-1)  # Ex(C + C + F)
        
        # 计算注意力权重
        attn_weights = self.fc(combined_feats).squeeze(-1)  # Ex1 -> E
        
        return attn_weights

# 消息传递与聚合
def message_passing(node_feats, edge_indices, edge_feats, attention_net):
    """
    通过注意力机制进行消息传递
    
    参数:
    - node_feats: 节点特征 (BxNxC)，B批次，N节点，C特征维度
    - edge_indices: 邻接点对的索引 (2xE)，E是有边的点对数
    - edge_feats: 边的特征 (ExF)
    - attention_net: AttentionNet 网络
    
    返回:
    - new_node_feats: 更新后的节点特征 (BxNxC)
    """
    B, N, C = node_feats.shape
    E = edge_indices.shape[1]  # E 是边的数量

    # 将节点特征展平，合并batch维度为 B*N
    node_feats_flat = node_feats.view(B * N, C)  # B*NxC
    
    # 获取边对应的源节点 i 和目标节点 j 的特征
    node_i_feats = node_feats_flat[edge_indices[0]]  # ExC，源节点 i 的特征
    node_j_feats = node_feats_flat[edge_indices[1]]  # ExC，目标节点 j 的特征
    
    # 计算注意力权重
    attn_weights = attention_net(node_i_feats, node_j_feats, edge_feats)  # Ex1
    
    # 对每个源节点的邻居进行 softmax 归一化
    attn_weights = scatter_softmax(attn_weights, edge_indices[0], dim=0)  # 按源节点 i 进行 softmax
    
    # 聚合特征
    updated_feats = node_j_feats * attn_weights.unsqueeze(-1)  # ExC
    
    # 将结果根据节点 i 进行加权聚合
    new_node_feats = torch.zeros_like(node_feats_flat)
    new_node_feats = scatter_add(updated_feats, edge_indices[0], out=new_node_feats, dim=0)  # B*NxC
    
    # 还原回 BxNxC 形状
    new_node_feats = new_node_feats.view(B, N, C)
    
    return new_node_feats


# 主网络
class PointCloudMessagePassing(nn.Module):
    def __init__(self, feat_dim, edge_dim, l_max, num_scales, hidden_dim, aggregation='concat'):
        super(PointCloudMessagePassing, self).__init__()
        sph_dim = (l_max + 1)**2  # 球谐基函数的特征维度
        self.l_max = l_max
        self.num_scales = num_scales
        self.aggregation = aggregation

        # 4 层，每层输出 hidden_dim // 4
        self.per_layer_dim = hidden_dim // 4
        
        # 对 biochem_feats 升维到 hidden_dim // 4
        self.input_fc = nn.Linear(feat_dim, self.per_layer_dim)
        self.attention_net = AttentionNet(self.per_layer_dim, edge_dim, sph_dim)
        
        # 映射到hidden_dim
        self.fc = nn.Linear(self.per_layer_dim * num_scales, hidden_dim)
    
    def forward(self, surfaces, biochem_feats):
        """
        thr_rs 是一个列表，表示不同尺度的半径阈值
        """
        # 先将 biochem_feats 升维
        biochem_feats = self.input_fc(biochem_feats)  # BxNx(hidden_dim // 4)
        
        # 计算距离矩阵
        distances = torch.cdist(surfaces, surfaces)  # BxNxN 的距离矩阵
        
        # 计算 distances 的最大值
        max_distance = distances.max().item()
        
        # 动态生成 thr_rs
        thr_rs = [max_distance / 20 * i / 4 for i in range(1, 5)]  # 1/4max, 2/4max, 3/4max, max
        
        # 对不同的半径阈值进行消息传递
        features_list = []

        edge_indices_ls = []
        edge_distances_ls = []
        
        for thr_r in thr_rs:
            # 计算邻接点对和距离
            edge_indices, edge_distances = compute_edges(distances, thr_r)
            edge_indices_ls.append(edge_indices)
            edge_distances_ls.append(edge_distances)
        # del distances
        # gc.collect()

        # # Free up any cached memory on the GPU
        # torch.cuda.empty_cache()

        for i, thr_r in enumerate(thr_rs):  
            # print('thr_r: ', thr_r)
            edge_indices = edge_indices_ls[i]
            edge_distances = edge_distances_ls[i]
            # 计算相对球坐标
            _, theta, phi = relative_spherical_coordinates(surfaces.unsqueeze(2), surfaces.unsqueeze(1))
            
            # 计算球谐基函数特征 (BxNxNxS)
            sph_feats = spherical_harmonic_basis(self.l_max, theta, phi)  # BxNxNx(2l_max+1)
            sph_feats = sph_feats.to(edge_indices.device)
            
            # 提取与邻接点对对应的球谐特征
            edge_batch_indices = edge_indices[0] // surfaces.shape[1]  # 获取批次索引
            edge_i_indices = edge_indices[0] % surfaces.shape[1]  # 获取 i 索引
            edge_j_indices = edge_indices[1] % surfaces.shape[1]  # 获取 j 索引
            # print('# edges:', edge_j_indices.shape)
            
            # 根据 edge_indices 提取 sph_feats 中对应的球谐特征 (Ex(2l_max+1))
            sph_feats = sph_feats[edge_batch_indices, edge_i_indices, edge_j_indices]  # Ex(2l_max+1)
            
            # 将距离和球谐特征合并
            edge_feats = torch.cat([edge_distances.unsqueeze(-1), sph_feats], dim=-1)  # Ex(F)
            
            # 进行消息传递
            new_feats = message_passing(biochem_feats, edge_indices, edge_feats, self.attention_net)
            
            features_list.append(new_feats)
        
        # 特征聚合：按sum或concat方式
        combined_feats = torch.cat(features_list, dim=-1)  # BxNx(num_scales * (hidden_dim // 4))
        
        # 映射到hidden_dim
        output_feats = self.fc(combined_feats)  # BxNxhidden_dim
        
        return output_feats




class TestModel0904(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(TestModel0904, self).__init__()
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
        
        # self.full_atom_dis = args.full_atom_dis
        
        # node_in = 12

        # node_in = node_in + 9 + 576 # node_in + 9 + 576
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0       ,  0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]

        # prior_matrix = torch.rand(self.args.virtual_num,3)
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])
        # num_va = self.virtual_atoms.shape[0]
        # edge_in = (15 + 9 * num_va + (num_va - 1) * num_va) * 16 + 16 + 7 

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
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

            
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
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

        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, args.updating_edges, args.att_output_mlp, args.node_output_mlp, args.node_net, args.edge_net, args.node_context, args.edge_context)

        # self.surface_encoder = EGNN(in_node_nf=2, hidden_nf=256, out_node_nf=3,
        #                             in_edge_nf=0, device='cuda', n_layers=3, attention=True)
        # self.surface_encoder = SurfaceEncoder(dropout_prob=0.1)
        l_max = 2
        num_scales = 4
        self.surface_encoder = PointCloudMessagePassing(2, 1, l_max, num_scales, hidden_dim)

        # self.decoder = CNNDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu)
        # self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim, args.num_decoder_layers2, args.kernel_size2, args.act_type, args.glu)

        # self.decoder = MLPDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu, vocab=len(self.tokenizer._token_to_id))
        # self.chain_embed = nn.Embedding(2,16)

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
        t1 = time.time()

        # print("Input data type (h_V):", h_V.dtype)
        # for name, param in self.node_embedding.named_parameters():
        #     print(f"Parameter name: {name}, dtype: {param.dtype}")

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)

        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        h_V_unflattened = torch.zeros(batch_size, max_length, h_V.size(-1), device=h_V.device)
        mask_unflattened = torch.zeros(batch_size, max_length, device=h_V.device)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            h_V_unflattened[idx, :mask.sum()] = h_V[mask]
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()
        t2 = time.time()


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


        
        ### original decoder
        # log_probs, logits = self.decoder(h_V, batch_id, self.token_mask)
                
        # log_probs, logits = self.decoder2(h_V, logits, batch_id)
        t3 = time.time()

        self.encode_t += t2-t1
        self.decode_t += t3-t2
        # return log_probs, log_probs0
        return {'log_probs': log_probs}
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
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

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
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

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

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
        
        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)
        
        d_chains = ((chain_encoding[dst.long()] - chain_encoding[src.long()])==0).long().reshape(-1)   
        chain_embed = self._idx_embeddings(d_chains)
        _E = torch.cat([_E, chain_embed], dim=-1)

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]

        mask = torch.masked_select(mask, mask_bool)
        batch.update({'X':X,
                'S':S,
                'score':score,
                '_V':_V,
                '_E':_E,
                'E_idx':E_idx,
                'batch_id': batch_id,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding})
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




###################### 0907 #############


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax
import numpy as np


def factorial(n):
    """Compute factorial using PyTorch for compatibility with GPUs."""
    return torch.exp(torch.lgamma(n + 1))

def associated_legendre_polynomial(l, m, x):
    """Compute the Associated Legendre Polynomial P_l^m(x) using PyTorch."""
    pmm = torch.ones_like(x) if m == 0 else (-1)**m * factorial(torch.tensor(2 * m - 1)) / (2**m * factorial(torch.tensor(m))) * (1 - x**2)**(m / 2)
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll

def spherical_harmonic(l, m, theta, phi):
    """Compute the spherical harmonic Y_l^m for given theta and phi using PyTorch."""
    legendre = associated_legendre_polynomial(l, abs(m), torch.cos(theta))
    if m < 0:
        legendre *= (-1)**m * factorial(torch.tensor(l - abs(m))) / factorial(torch.tensor(l + abs(m)))
    return torch.sqrt((2 * l + 1) / (4 * torch.pi) * factorial(torch.tensor(l - m)) / factorial(torch.tensor(l + m))) * legendre * torch.exp(1j * m * phi)

def spherical_harmonic_basis(l_max, theta, phi):
    """
    Compute spherical harmonic basis Y_l^m(theta, phi) using PyTorch.
    
    Parameters:
    - l_max: Maximum degree l
    - theta: Polar angle (BxNxN, radians)
    - phi: Azimuthal angle (BxNxN, radians)
    
    Returns:
    - harmonics: Spherical harmonic values (BxNxNx(2*l_max+1))
    """
    # start_time = time.time()
    
    B, N, _ = theta.shape  # Get batch size B and number of points N
    harmonics = []
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Calculate spherical harmonics using PyTorch
            Y_lm = spherical_harmonic(l, m, theta, phi)
            harmonics.append(Y_lm.real)  # If you only need the real part
    
    # Stack all harmonics into a tensor of shape (BxNxNx(2*l_max+1))
    harmonics = torch.stack(harmonics, dim=-1)
    
    # end_time = time.time()
    # print(f"Time taken to compute spherical harmonic basis: {end_time - start_time:.4f} seconds")
    
    return harmonics

# 计算相对球坐标系
def relative_spherical_coordinates(xyz_i, xyz_j):
    """
    计算相对位置的球坐标系表示
    
    参数:
    - xyz_i: 中心点坐标 (BxNx3)
    - xyz_j: 邻居点坐标 (BxNx3)
    
    返回:
    - r: 距离 (BxNxN)
    - theta: 极角 (BxNxN)
    - phi: 方位角 (BxNxN)
    """
    rel_pos = xyz_j - xyz_i  # 计算相对位置
    r = torch.norm(rel_pos, dim=-1)  # 距离
    theta = torch.atan2(rel_pos[..., 1], rel_pos[..., 0])  # 极角
    phi = torch.acos(rel_pos[..., 2] / (r + 1e-8))  # 方位角
    return r, theta, phi


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




class TestModel0907(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(TestModel0907, self).__init__()
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
        
        # self.full_atom_dis = args.full_atom_dis
        
        # node_in = 12

        # node_in = node_in + 9 + 576 # node_in + 9 + 576
        prior_matrix = [
            [-0.58273431, 0.56802827, -0.54067466],
            [0.0       ,  0.83867057, -0.54463904],
            [0.01984028, -0.78380804, -0.54183614],
        ]

        # prior_matrix = torch.rand(self.args.virtual_num,3)
        self.virtual_atoms = nn.Parameter(torch.tensor(prior_matrix)[:self.args.virtual_num,:])
        # num_va = self.virtual_atoms.shape[0]
        # edge_in = (15 + 9 * num_va + (num_va - 1) * num_va) * 16 + 16 + 7 

        node_in = 0
        if self.args.node_dist:
            pair_num = 6
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
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

            
            if self.args.virtual_num>0:
                pair_num += self.args.virtual_num
                pair_num += self.args.virtual_num*(self.args.virtual_num-1)
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

        # self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, args.updating_edges, args.att_output_mlp, args.node_output_mlp, args.node_net, args.edge_net, args.node_context, args.edge_context)
        self.encoder = StructureEncoder(hidden_dim, 3, 0, args.updating_edges, args.att_output_mlp, args.node_output_mlp, args.node_net, args.edge_net, args.node_context, args.edge_context)

        # self.surface_encoder = EGNN(in_node_nf=2, hidden_nf=256, out_node_nf=3,
        #                             in_edge_nf=0, device='cuda', n_layers=3, attention=True)
        # self.surface_encoder = SurfaceEncoder(dropout_prob=0.1)
        l_max = 2
        num_scales = 4
        self.surface_encoder = PointCloudMessagePassing(2, 1, l_max, num_scales, hidden_dim)

        # self.decoder = CNNDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu)
        # self.decoder2 = CNNDecoder2(hidden_dim, hidden_dim, args.num_decoder_layers2, args.kernel_size2, args.act_type, args.glu)

        # self.decoder = MLPDecoder(hidden_dim, hidden_dim, args.num_decoder_layers1, args.kernel_size1, args.act_type, args.glu, vocab=len(self.tokenizer._token_to_id))
        # self.chain_embed = nn.Embedding(2,16)

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
        t1 = time.time()

        # print("Input data type (h_V):", h_V.dtype)
        # for name, param in self.node_embedding.named_parameters():
        #     print(f"Parameter name: {name}, dtype: {param.dtype}")

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.W_e(self.norm_edges(self.edge_embedding(h_P)))
        
        h_V, h_P = self.encoder(h_V, h_P, P_idx, batch_id)

        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        h_V_unflattened = torch.zeros(batch_size, max_length, h_V.size(-1), device=h_V.device)
        mask_unflattened = torch.zeros(batch_size, max_length, device=h_V.device)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            h_V_unflattened[idx, :mask.sum()] = h_V[mask]
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()
        t2 = time.time()


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


        
        ### original decoder
        # log_probs, logits = self.decoder(h_V, batch_id, self.token_mask)
                
        # log_probs, logits = self.decoder2(h_V, logits, batch_id)
        t3 = time.time()

        self.encode_t += t2-t1
        self.decode_t += t3-t2
        # return log_probs, log_probs0
        return {'log_probs': log_probs}
        
    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
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

        if self.args.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
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

        if self.args.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

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
        
        pos_embed = self._positional_embeddings(E_idx, 16)
        _E = torch.cat([_E, pos_embed], dim=-1)
        
        d_chains = ((chain_encoding[dst.long()] - chain_encoding[src.long()])==0).long().reshape(-1)   
        chain_embed = self._idx_embeddings(d_chains)
        _E = torch.cat([_E, chain_embed], dim=-1)

        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X = X[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]

        mask = torch.masked_select(mask, mask_bool)
        batch.update({'X':X,
                'S':S,
                'score':score,
                '_V':_V,
                '_E':_E,
                'E_idx':E_idx,
                'batch_id': batch_id,
                'mask': mask,
                'chain_mask': chain_mask,
                'chain_encoding': chain_encoding})
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











