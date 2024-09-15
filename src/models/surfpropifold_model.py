import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from src.tools import gather_nodes, _dihedrals, _get_rbf, _get_dist, _rbf, _orientations_coarse_gl_tuple
import numpy as np
from src.modules.pifold_module import *
from src.modules.surface_module import *
from transformers import AutoTokenizer
import os

# from src.models.SurfPro.fairseq.models.egnn import EGNN
# from src.models.SurfPro.fairseq.models.surface_protein_model import get_edges_batch

pair_lst = ['N-N', 'C-C', 'O-O', 'Cb-Cb', 'Ca-N', 'Ca-C', 'Ca-O', 'Ca-Cb', 'N-C', 'N-O', 'N-Cb', 'Cb-C', 'Cb-O', 'O-C', 'N-Ca', 'C-Ca', 'O-Ca', 'Cb-Ca', 'C-N', 'O-N', 'Cb-N', 'C-Cb', 'O-Cb', 'C-O']


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


class SurfProPiFold_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(SurfProPiFold_Model, self).__init__()
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
        self.surface_encoder = SurfaceEncoder()

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
        memory_padding_mask = ~batch['surface_mask']  # [batch_size, seq_len]
        t2 = time.time()



        ### surfpro encoder
        # '''
        # (done)surf_aa_embs: 4(batch size)x4316(#vertices of surface)x2(2 biochem feat), biochem feats
        # (done)coor_features: 4(batch size)x4316(#vertices of surface)x3(xyz coordinates) (after coor_features = coor_features.view(coor_features.size(0), -1, 3)), coordinates of surface vertices
        # (done)k: 30, #nearest neighbor to aggregate 
        # edges: list of 2 tensors, shape of each tensor: 517920, indicating source and target
        # '''
        # k = 30
        # bs = surf_aa_embs.shape[0]
        # edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)
        # h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)



        ### surface encoder
        surfaces, biochem_feats, pcs = batch['surface'], batch['features'], batch['pc']
        h_surface = self.surface_encoder(surfaces, biochem_feats, pcs, memory_padding_mask)



        ### new decoder
        ss_connection_mask = batch['ss_connection']
        ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

        # Transformer decoder to fuse h_V_unflattened and h_surface
        # Add positional encoding to the inputs of the Transformer decoder
        h_V_unflattened = self.positional_encoding(h_V_unflattened)
        
        decoder_output = self.transformer_decoder(
            h_V_unflattened, h_surface, 
            tgt_key_padding_mask=target_padding_mask, 
            memory_key_padding_mask=memory_padding_mask,
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
    


class SurfProPiFoldSurfaceOnly_Model(nn.Module):
    # def __init__(self, args, **kwargs):
    #     """ Graph labeling network """
    #     super(SurfProPiFoldSurfaceOnly_Model, self).__init__()
    #     self.args = args
    #     self.augment_eps = args.augment_eps
    #     node_features = args.node_features
    #     edge_features = args.edge_features
    #     hidden_dim = args.hidden_dim
    #     dropout = args.dropout
    #     num_encoder_layers = args.num_encoder_layers
    #     self.top_k = args.k_neighbors
    #     self.num_rbf = 16
    #     self.num_positional_embeddings = 16

    #     self.dihedral_type = args.dihedral_type
    #     self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
    #     alphabet = [one for one in 'ACDEFGHIKLMNPQRSTVWYX']

    #     self.surface_encoder = SurfaceEncoder()

    #     # New Transformer decoder and MLP for final prediction
    #     decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout, batch_first=True)
    #     self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
    #     self.mlp = nn.Sequential(
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.ReLU(),
    #         nn.Linear(hidden_dim, len(self.tokenizer._token_to_id))
    #     )

    #     self._init_params()

    #     self.encode_t = 0
    #     self.decode_t = 0
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(SurfProPiFoldSurfaceOnly_Model, self).__init__()
        self.args = args
        self.augment_eps = args.augment_eps
        node_features = args.node_features
        edge_features = args.edge_features
        hidden_dim = args.hidden_dim
        self.hidden_dim = args.hidden_dim
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
        self.surface_encoder = SurfaceEncoder()

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
        t1 = time.time()
        batch_id = batch['batch_id']
        # Unflatten h_V and mask to have batch dimension
        max_length = batch['lengths'].max().item()
        batch_size = len(batch['lengths'])
        h_V_unflattened = torch.randn(batch_size, max_length, self.hidden_dim)
        mask_unflattened = torch.zeros(batch_size, max_length)

        # Efficiently assign values to h_V_unflattened and mask_unflattened
        for idx in torch.unique(batch_id):
            mask = (batch_id == idx)
            mask_unflattened[idx, :mask.sum()] = 1
        # Create padding masks
        # target_padding_mask = (mask_unflattened == 0).to(h_V.device)  # [batch_size, seq_len]
        target_padding_mask = ~mask_unflattened.bool()
        memory_padding_mask = ~batch['surface_mask']  # [batch_size, seq_len]
        t2 = time.time()



        ### surface encoder
        surfaces, biochem_feats, pcs = batch['surface'], batch['features'], batch['pc']
        h_surface = self.surface_encoder(surfaces, biochem_feats, pcs, memory_padding_mask)



        ### new decoder
        ss_connection_mask = batch['ss_connection']
        ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

        # Transformer decoder to fuse h_V_unflattened and h_surface
        h_V_unflattened = h_V_unflattened.to(h_surface.device)
        mask_unflattened = mask_unflattened.to(h_surface.device)
        target_padding_mask = target_padding_mask.to(h_surface.device)
        decoder_output = self.transformer_decoder(
            h_V_unflattened, h_surface, 
            tgt_key_padding_mask=target_padding_mask, 
            memory_key_padding_mask=memory_padding_mask,
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
    


class SurfProPiFoldDense_Model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super(SurfProPiFoldDense_Model, self).__init__()
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
        self.surface_encoder = SurfaceEncoder()

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
        memory_padding_mask = ~batch['surface_mask']  # [batch_size, seq_len]
        t2 = time.time()



        ### surfpro encoder
        # '''
        # (done)surf_aa_embs: 4(batch size)x4316(#vertices of surface)x2(2 biochem feat), biochem feats
        # (done)coor_features: 4(batch size)x4316(#vertices of surface)x3(xyz coordinates) (after coor_features = coor_features.view(coor_features.size(0), -1, 3)), coordinates of surface vertices
        # (done)k: 30, #nearest neighbor to aggregate 
        # edges: list of 2 tensors, shape of each tensor: 517920, indicating source and target
        # '''
        # k = 30
        # bs = surf_aa_embs.shape[0]
        # edges = get_edges_batch(coor_features.size()[1], bs, coor_features.detach().cpu(), k)
        # h, coor_features = self.encoder(surf_aa_embs, coor_features, edges, None, k)



        ### surface encoder
        surfaces, biochem_feats, pcs = batch['surface'], batch['features'], batch['pc']
        h_surface = self.surface_encoder(surfaces, biochem_feats, pcs, memory_padding_mask)



        ### new decoder
        ss_connection_mask = batch['ss_connection']
        ss_connection_mask = ~ss_connection_mask.bool().repeat(8, 1, 1)

        # Transformer decoder to fuse h_V_unflattened and h_surface
        # Add positional encoding to the inputs of the Transformer decoder
        h_V_unflattened = self.positional_encoding(h_V_unflattened)
        
        decoder_output = self.transformer_decoder(
            h_V_unflattened, h_surface, 
            tgt_key_padding_mask=target_padding_mask, 
            memory_key_padding_mask=memory_padding_mask,
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