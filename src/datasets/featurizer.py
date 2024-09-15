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

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def shuffle_subset(n, p):
    n_shuffle = np.random.binomial(n, p)
    ix = np.arange(n)
    ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
    ix_subset_shuffled = np.copy(ix_subset)
    np.random.shuffle(ix_subset_shuffled)
    ix[ix_subset] = ix_subset_shuffled
    return ix


def featurize_AF(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.zeros([B, L_max])

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
            score[i,:l] = b['score'][idx_shuffle]
        else:
            S[i, :l] = indices
            score[i,:l] = b['score']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    score_new = np.zeros_like(score)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]
        score_new[i,:n] = score[i][mask[i]==1]

    X = X_new
    S = S_new
    score = score_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    return X, S, score, mask, lengths


def featurize_GTrans(batch):
    """ Pack and pad batch into torch tensors """
    # alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    batch = [one for one in batch if one is not None]
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])-1 # 1:需要被预测的掩码部分 0:可见部分
    chain_encoding = np.zeros([B, L_max])-1
    

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 4, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        # indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)


        S[i, :l] = indices
        chain_mask[i,:l] = b['chain_mask']
        chain_encoding[i,:l] = b['chain_encoding']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    
    return {"title": [b['title'] for b in batch],
            "X":X,
            "S":S,
            "score": score,
            "mask":mask,
            "lengths":lengths,
            "chain_mask":chain_mask,
            "chain_encoding":chain_encoding}


class featurize_GVP:
    def __init__(self, num_positional_embeddings=16, top_k=30, num_rbf=16):
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        # self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
        #                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
        #                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
        #                'N': 2, 'Y': 18, 'M': 12}
        # self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
    
    def featurize(self, batch):
        data_all = []
        for b in batch:
            if b is None:
                continue
            coords = torch.tensor(np.stack([b[c] for c in ['N', 'CA', 'C', 'O']], 1))
            seq = torch.tensor(np.array(tokenizer.encode(b['seq'], add_special_tokens=False)))
        
            mask = torch.isfinite(coords.sum(dim=(1,2)))
            coords[~mask] = np.inf
            
            X_ca = coords[:, 1].float()
            edge_index = torch_cluster.knn_graph(X_ca, k=self.top_k)
            
            pos_embeddings = self._positional_embeddings(edge_index) # [E, 16]
            E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]] # [E, 3]
            rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf) # [E, 16]
            
            dihedrals = self._dihedrals(coords)  # [n,6]
            orientations = self._orientations(X_ca) # [n,2,3]
            sidechains = self._sidechains(coords) # [n,3]
            
            node_s = dihedrals.float() # [n,6]

            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2).float() # [n, 3, 3]

            edge_s = torch.cat([rbf, pos_embeddings], dim=-1).float() # [E, 32]
            edge_v = _normalize(E_vectors).unsqueeze(-2).float() # [E, 1, 3]
            
            node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
            
            data = torch_geometric.data.Data(x=X_ca, seq=seq,
                                            node_s=node_s, node_v=node_v,
                                            edge_s=edge_s, edge_v=edge_v,
                                            edge_index=edge_index, mask=mask)
            data_all.append(data)
        return data_all
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    def _orientations(self, X):
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 
    
    def collate(self, batch):
        batch = self.featurize(batch)
        if (batch is None) or (len(batch)==0):
            return None
        
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


def featurize_ProteinMPNN(batch, is_testing=False, chain_dict=None, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None, pssm_dict=None, bias_by_res_dict=None):
    """ Pack and pad batch into torch tensors """
    batch = [one for one in batch if one is not None]
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0*np.ones([B, L_max, 21], dtype=np.float32) #1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.zeros([B, L_max])
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[b['name']] #masked_chains a list of chain letters to predict [A, D, F]
        else:
            # masked_chains = [item[-1:] for item in list(b) if item[:10]=='seq_chain_']
            masked_chains = ['']
            visible_chains = []
        # num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains
        #random.shuffle(all_chains)
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                # chain_seq = b[f'seq_chain_{letter}']
                chain_seq = b[f'seq{letter}']
                chain_seq = ''.join([a if a!='-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1]+chain_length)
                masked_chain_length_list.append(chain_length)
                # chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_coords = b
                chain_mask = np.ones(chain_length) #1.0 for masked
                # x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain = np.stack([chain_coords[c] for c in [f'N', f'CA', f'C', f'O']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict!=None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list)-1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict!=None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0])-1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet))== AA)[0][0] for AA in item[1]]).repeat(idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:,0], idx_[:,1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0*np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

       
        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict!=None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx+v[0][v_count]-1)#make 0 to be the first
                                tied_beta[start_idx+v[0][v_count]-1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx+v_-1)#make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)
 
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)
        m_pos = np.concatenate(fixed_position_mask_list,0) #[L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list,0) #[L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list,0) #[L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list, 0)  #[L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        if 'score' in b.keys():
            score[i, :l] = b['score']
        else:
            score[i, :l] = 100.0
 
        m_pad = np.pad(m, [[0, L_max-l]], 'constant', constant_values=(0.0, ))
        m_pos_pad = np.pad(m_pos, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list,0), [[0,L_max-l], [0, 0]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad
        chain_M_pos[i,:] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max-l]], 'constant', constant_values=(0.0, ))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))

        pssm_coef_all[i,:] = pssm_coef_pad
        pssm_bias_all[i,:] = pssm_bias_pad
        pssm_log_odds_all[i,:] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0,L_max-l], [0,0]], 'constant', constant_values=(0.0, ))
        bias_by_res_all[i,:] = bias_by_res_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    
    # mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    # numbers = np.sum(mask, axis=1).astype(np.int)
    # S_new = np.zeros_like(S)
    # X_new = np.zeros_like(X)+np.nan
    # for i, n in enumerate(numbers):
    #     X_new[i,:n,::] = X[i][mask[i]==1]
    #     S_new[i,:n] = S[i][mask[i]==1]

    # X = X_new
    # S = S_new
    
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32)

    jumps = ((residue_idx[:,1:]-residue_idx[:,:-1])==1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32)
    phi_mask = np.pad(jumps, [[0,0],[1,0]])
    psi_mask = np.pad(jumps, [[0,0],[0,1]])
    omega_mask = np.pad(jumps, [[0,0],[0,1]])
    dihedral_mask = np.concatenate([phi_mask[:,:,None], psi_mask[:,:,None], omega_mask[:,:,None]], -1) #[B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    score = torch.from_numpy(score).float()
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)

    if is_testing is False:
        return {"title": [b['title'] for b in batch],
                "X":X,
                "S":S,
                "score": score,
                "mask":mask,
                "lengths":lengths,
                "chain_M":chain_M,
                "chain_M_pos":chain_M_pos,
                "residue_idx":residue_idx,
                "chain_encoding_all":chain_encoding_all}
    else:
        return {"title": [b['title'] for b in batch],
                "X":X,
                "S":S,
                "score": score,
                "mask":mask,
                "lengths":lengths,
                "chain_M":chain_M,
                "chain_M_pos":chain_M_pos,
                "residue_idx":residue_idx,
                "chain_encoding_all":chain_encoding_all}
        
        



def featurize_Inversefolding(batch, shuffle_fraction=0.):
    """ Pack and pad batch into torch tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 3, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])-1 # 1:需要被预测的掩码部分 0:可见部分
    chain_encoding = np.zeros([B, L_max])-1

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b[c] for c in ['N', 'CA', 'C']], 1) # [#atom, 4, 3]
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, )) # [#atom, 3, 3]
        X[i,:,:,:] = x_pad

        # Convert to labels
        indices = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices
        
        chain_mask[i,:l] = b['chain_mask']
        chain_encoding[i,:l] = b['chain_encoding']

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32) # atom mask
    numbers = np.sum(mask, axis=1).astype(np.int)
    S_new = np.zeros_like(S)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]

    X = X_new
    S = S_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    return {"title": [b['title'] for b in batch],
            "X":X,
            "S":S,
            "score": score,
            "mask":mask,
            "lengths":lengths,
            "chain_mask":chain_mask,
            "chain_encoding":chain_encoding}


def featurize_SurfProPiFold(batch):
    """ Pack and pad batch into torch tensors """
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
    pcs = []
    surface_lengths = []
    ss_connections = []
    
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

        # Add surface and features
        surfaces.append(torch.tensor(b['surface'], dtype=torch.float32))
        features.append(torch.tensor(b['features'], dtype=torch.float32))
        pcs.append(torch.tensor(b['pc'], dtype=torch.float32))
        surface_lengths.append(b['surface'].shape[0])
        # ss_connections.append(torch.tensor(b['ss_connection'], dtype=torch.float32))

    surfaces_padded = pad_sequence(surfaces, batch_first=True, padding_value=0)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)

    # Pad ss_connections
    # ss_connections_padded = pad_ss_connections(ss_connections, L_max, max(surface_lengths))

    # Create surface padding mask
    surface_lengths = torch.tensor(surface_lengths, dtype=torch.long)
    max_surface_length = max(surface_lengths)
    surface_mask = torch.arange(max_surface_length).expand(len(surface_lengths), max_surface_length) < surface_lengths.unsqueeze(1)

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

    # Calculate ss_connection based on X_new and orig_surface
    for i in range(B):
        ca_coords = X[i, :, 1, :]  # Extract CA coordinates from X_new (1 is for CA atom)
        surface_coords = batch[i]['orig_surface']
        
        # Use the mask to identify valid indices
        valid_indices = mask[i].astype(bool)  # mask[i] is 1 for valid indices, 0 otherwise
        valid_ca_coords = ca_coords[valid_indices]
        
        # Nearest neighbors search
        # nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(surface_coords)
        nbrs = NearestNeighbors(n_neighbors=32, algorithm='ball_tree').fit(surface_coords)
        distances, indices = nbrs.kneighbors(valid_ca_coords)
        
        ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
        
        # Fill ss_connection for valid CA coordinates
        for j, neighbors in zip(np.where(valid_indices)[0], indices):
            ss_connection[j, neighbors] = 1
        
        # Fill ss_connection for invalid CA coordinates
        ss_connection[~valid_indices, :] = 1
        
        ss_connections.append(torch.tensor(ss_connection, dtype=torch.float32))

    # Pad ss_connections
    ss_connections_padded = pad_ss_connections(ss_connections, L_max, max(surface_lengths))

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    pcs = torch.stack(pcs)  # Combine pcs into a single tensor
    
    return {
        "title": [b['title'] for b in batch],
        "X": X,
        "S": S,
        "score": score,
        "mask": mask,
        "lengths": lengths,
        "chain_mask": chain_mask,
        "chain_encoding": chain_encoding,
        "surface": surfaces_padded,
        "features": features_padded,
        "pc": pcs,
        'surface_mask': surface_mask,
        'ss_connection': ss_connections_padded
    }


def pad_ss_connections(ss_connections, max_residues, max_surface_atoms):
    """ Pad ss_connections to the maximum number of residues and surface atoms in the batch """
    B = len(ss_connections)
    ss_connections_padded = torch.ones((B, max_residues, max_surface_atoms), dtype=torch.float32)
    for i, ss_connection in enumerate(ss_connections):
        ss_connections_padded[i, :ss_connection.shape[0], :ss_connection.shape[1]] = ss_connection
    return ss_connections_padded





def featurize_TestModel0904(batch):
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
    pcs = []
    orig_surfaces = []
    surface_lengths = []
    ss_connections = []
    
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
        nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(surface_coords)
        distances, indices = nbrs.kneighbors(valid_ca_coords)
        
        ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
        
        # Fill ss_connection for valid CA coordinates
        for j, neighbors in zip(np.where(valid_indices)[0], indices):
            ss_connection[j, neighbors] = 1
        
        # Fill ss_connection for invalid CA coordinates
        ss_connection[~valid_indices, :] = 1
        
        ss_connections.append(torch.tensor(ss_connection, dtype=torch.float32))

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
        'ss_connection': ss_connections_padded
    }





def featurize_TestModel0907(batch):
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
    pcs = []
    orig_surfaces = []
    surface_lengths = []
    ss_connections = []
    
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
        nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(surface_coords)
        distances, indices = nbrs.kneighbors(valid_ca_coords)
        
        ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
        
        # Fill ss_connection for valid CA coordinates
        for j, neighbors in zip(np.where(valid_indices)[0], indices):
            ss_connection[j, neighbors] = 1
        
        # Fill ss_connection for invalid CA coordinates
        ss_connection[~valid_indices, :] = 1
        
        ss_connections.append(torch.tensor(ss_connection, dtype=torch.float32))

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
        'ss_connection': ss_connections_padded
    }


def featurize_SBModel(batch):
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
    pcs = []
    orig_surfaces = []
    surface_lengths = []
    ss_connections = []
    
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
        'ss_connection': ss_connections_padded
    }


def featurize_SBModel1(batch):
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
    pcs = []
    orig_surfaces = []
    surface_lengths = []
    ss_connections = []
    
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

    # Conversion
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)


    # # Step 1: Construct the local orientation frame Q for all atoms in the batch
    # Q = compute_Q(X)  # Shape: (B, N, 3, 3) - local orientation frame for each atom in the batch

    # # Step 2: Extract CA coordinates and surface coordinates for all batch elements
    # ca_coords = X[:, :, 1, :]  # Shape: (B, N, 3) for CA atom coordinates

    # # Mask to identify valid atoms
    # valid_indices = mask.to(bool)  # Shape: (B, N)

    # # Surface coordinates and stacked original surfaces for all batch elements
    # surface_coords_batch = orig_surfaces_stacked  # Shape: (B, n_surface_points, 3)

    # # Step 3: Generate 8 quadrant vectors for all atoms in the batch
    # # Generate 8 combinations of signs (-1 or 1) to represent quadrants
    # signs = torch.tensor([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
    #                     [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]])

    # # Broadcast signs over the Q vectors to generate 8 quadrant vectors for each atom across the entire batch
    # quadrant_vectors = torch.einsum('qv, bnvk -> bnqvk', signs, Q)  # Shape: (B, N, 8, 3, 3)

    # # Compute the mean vector for each set of 3 quadrant vectors
    # quadrant_means = quadrant_vectors.mean(dim=3)  # Shape: (B, N, 8, 3)

    # # # Step 4: Project surface_coords onto each direction vector (for each atom across the batch)
    # # # First, transpose surface_coords for dot product calculation
    # # surface_coords_t_batch = surface_coords_batch.transpose(2, 1)  # Shape: (B, 3, n_surface_points)

    # # # Perform dot product projection for all atoms across the batch in a single step
    # # # quadrant_means shape: (B, N, 8, 3), surface_coords_t_batch shape: (B, 3, n_surface_points)
    # # projections = torch.einsum('bnqk,bkm->bnqm', [quadrant_means, surface_coords_t_batch])  # Shape: (B, N, 8, n_surface_points)

    # # Step 4: Subtract CA coordinates from surface coordinates
    # # Reshape ca_coords for broadcasting: (B, N, 3) -> (B, N, 1, 3)
    # ca_coords_expanded = ca_coords.unsqueeze(2)  # Shape: (B, N, 1, 3)

    # # Surface coordinates already have shape (B, n_surface_points, 3), so we reshape it to align the axes.
    # surface_coords_expanded = surface_coords_batch.unsqueeze(1)  # Shape: (B, 1, n_surface_points, 3)

    # # Subtract CA coordinates from surface coordinates
    # subtracted_coords = surface_coords_expanded - ca_coords_expanded  # Shape: (B, N, n_surface_points, 3)

    # # Step 5: Project the subtracted coordinates onto each quadrant mean vector
    # # quadrant_means shape: (B, N, 8, 3), subtracted_coords shape: (B, N, n_surface_points, 3)
    # projections = torch.einsum('bnqk,bnmk->bnqm', [quadrant_means, subtracted_coords])  # Shape: (B, N, 8, n_surface_points)


    # # Step 1: Mask non-positive values
    # positive_projections = torch.where(projections > 0, projections, torch.tensor(float('inf')).to(projections.device))

    # # Step 2: Identify quadrants where all values are non-positive
    # all_inf_mask = torch.isinf(positive_projections).all(dim=3)  # Shape: (B, N, 8)

    # # Step 3: Find argmin for positive values
    # min_projection_indices = torch.argmin(positive_projections, dim=3)  # Shape: (B, N, 8)
    # # top2_values, top2_indices = torch.topk(positive_projections, k=2, dim=3, largest=False)
    # # min_projection_indices = top2_indices.reshape(top2_indices.shape[0], top2_indices.shape[1], -1)  # Shape: (B, N, 16)
    # # all_inf_mask = all_inf_mask.repeat_interleave(2, dim=2)

    # # Step 4: Set indices for quadrants with all non-positive values to -1 (or another sentinel value)
    # # min_projection_indices[all_inf_mask] = -1  # or some other value indicating no valid projection
    # min_projection_indices = torch.where(all_inf_mask, torch.max(min_projection_indices, dim=2).values.unsqueeze(-1), min_projection_indices)

    # # Step 7: Initialize ss_connection for the entire batch
    # ss_connection = torch.zeros((ca_coords.shape[0], ca_coords.shape[1], surface_coords_batch.shape[1]))  # Shape: (B, N, n_surface_points)

    # # Fill ss_connection for all CA coordinates (valid and invalid) directly
    # # For valid atoms, set the nearest surface points to 1
    # ss_connection.scatter_(2, min_projection_indices, 1)

    # # Fill ss_connection for invalid CA coordinates (all ones for invalid atoms)
    # ss_connection[~valid_indices] = 1

    # # Final step: Pad ss_connections
    # ss_connections_padded = pad_ss_connections(ss_connection, L_max, min_surface_length)



    # Step 1: Construct the local orientation frame Q for all atoms in the batch
    Q = compute_Q(X)  # Shape: (B, N, 3, 3) - local orientation frame for each atom in the batch

    # Step 2: Extract CA coordinates and surface coordinates for all batch elements
    ca_coords = X[:, :, 1, :]  # Shape: (B, N, 3) for CA atom coordinates

    # Mask to identify valid atoms
    valid_indices = mask.to(bool)  # Shape: (B, N)

    # Surface coordinates and stacked original surfaces for all batch elements
    surface_coords_batch = orig_surfaces_stacked  # Shape: (B, n_surface_points, 3)

    # Step 3: Generate 8 combinations of signs (-1 or 1) to represent quadrants
    signs = torch.tensor([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                        [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]])

    # Step 4: Subtract CA coordinates from surface coordinates
    # Reshape ca_coords for broadcasting: (B, N, 3) -> (B, N, 1, 3)
    ca_coords_expanded = ca_coords.unsqueeze(2)  # Shape: (B, N, 1, 3)

    # Surface coordinates already have shape (B, n_surface_points, 3), so we reshape it to align the axes.
    surface_coords_expanded = surface_coords_batch.unsqueeze(1)  # Shape: (B, 1, n_surface_points, 3)

    # Subtract CA coordinates from surface coordinates
    subtracted_coords = surface_coords_expanded - ca_coords_expanded  # Shape: (B, N, n_surface_points, 3)

    # Step 5: Rotate the subtracted coordinates by Q to get local coordinates
    # Rotate subtracted coordinates into the local frame using Q (B, N, 3, 3) * (B, N, n_surface_points, 3)
    # We'll use torch.einsum for this matrix multiplication
    rotated_coords = torch.einsum('bnvk,bnmk->bnmv', Q, subtracted_coords)  # Shape: (B, N, n_surface_points, 3)

    # Step 6: Classify each surface point based on the signs of its rotated coordinates
    # Get the signs of x, y, z components to determine the quadrant
    signs_rotated_coords = torch.sign(rotated_coords)  # Shape: (B, N, n_surface_points, 3)
    signs_rotated_coords = signs_rotated_coords.int()  # Ensure the signs are integers (-1 or 1)

    # Compare signs_rotated_coords with the `signs` tensor (8 combinations)
    # We want to classify each point into one of the 8 quadrants
    # We'll create a mask for each quadrant by comparing the signs of each point with the 8 quadrant definitions

    # Reshape the signs tensor for broadcasting to compare with each point
    signs_expanded = signs.view(1, 1, 1, 8, 3).expand(rotated_coords.shape[0], rotated_coords.shape[1], rotated_coords.shape[2], -1, -1)  # Shape: (B, N, n_surface_points, 8, 3)

    # Compare signs of the points with the quadrant definitions to create a mask for each point
    quadrant_mask = (signs_rotated_coords.unsqueeze(3) == signs_expanded).all(dim=-1)  # Shape: (B, N, n_surface_points, 8)

    # Step 7: Calculate the Euclidean norm of each surface point in the local frame
    norms = torch.norm(rotated_coords, dim=-1)  # Shape: (B, N, n_surface_points)

    # Step 8: For each quadrant, find the surface point with the smallest norm
    # Mask norms by quadrant_mask and assign a high value (e.g., float('inf')) to points not in the quadrant
    norms_masked = torch.where(quadrant_mask, norms.unsqueeze(-1), torch.tensor(float('inf')).to(norms.device))  # Shape: (B, N, n_surface_points, 8)

    # Step 9: Handle quadrants with no valid surface points
    # Identify quadrants where all values are inf (no valid surface point)
    all_inf_mask = torch.isinf(norms_masked).all(dim=2)  # Shape: (B, N, 8)

    # Find the index of the minimum norm for each quadrant
    # min_projection_indices = torch.argmin(norms_masked, dim=2)  # Shape: (B, N, 8)
    top2_values, top2_indices = torch.topk(norms_masked, k=2, dim=2, largest=False)
    min_projection_indices = top2_indices.reshape(top2_indices.shape[0], top2_indices.shape[1], -1)  # Shape: (B, N, 16)
    all_inf_mask = all_inf_mask.repeat_interleave(2, dim=2)

    # Replace invalid indices with some valid index, e.g., max valid index
    min_projection_indices = torch.where(all_inf_mask, torch.max(min_projection_indices, dim=2).values.unsqueeze(-1), min_projection_indices)

    # Step 10: Initialize ss_connection for the entire batch
    ss_connection = torch.zeros((ca_coords.shape[0], ca_coords.shape[1], surface_coords_batch.shape[1]))  # Shape: (B, N, n_surface_points)

    # Fill ss_connection for all CA coordinates (valid and invalid) directly
    # For valid atoms, set the nearest surface points to 1
    ss_connection.scatter_(2, min_projection_indices, 1)

    # Fill ss_connection for invalid CA coordinates (all ones for invalid atoms)
    ss_connection[~valid_indices] = 1

    # Final step: Pad ss_connections
    ss_connections_padded = pad_ss_connections(ss_connection, L_max, min_surface_length)




    
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
        'ss_connection': ss_connections_padded
    }


def compute_Q(X):
    # X: Tensor of shape (B, N, 4, 3) where the 3rd dimension represents ['N', 'CA', 'C', 'O'] atoms.
    
    # Step 1: Compute vectors between consecutive atoms (N -> CA, CA -> C)
    v1 = X[:, :, 1, :] - X[:, :, 0, :]  # N -> CA vector
    v2 = X[:, :, 2, :] - X[:, :, 1, :]  # CA -> C vector

    # Step 2: Normalize v1 and v2
    v1 = v1 / torch.norm(v1, dim=-1, keepdim=True)
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Step 3: Compute the normal vector n as the cross product of v1 and v2
    n = torch.cross(v1, v2)
    n = n / torch.norm(n, dim=-1, keepdim=True)  # Normalize n

    # Step 4: Compute the bisector vector b1
    b1 = v1 - v2
    b1 = b1 / torch.norm(b1, dim=-1, keepdim=True)  # Normalize b1

    # Step 5: Compute the third vector as the cross product of b1 and n to ensure orthogonality
    b2 = torch.cross(b1, n)

    # Step 6: Stack b1, n, and b2 to form the local frame Q
    Q = torch.stack([b1, n, b2], dim=-2)  # Shape: (B, N, 3, 3)

    return Q


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