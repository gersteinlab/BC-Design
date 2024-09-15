import torch
import torch.nn as nn
import torch.nn.functional as F


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
        attn_scores = attn_scores + inv_distance_matrices.unsqueeze(1) + heat_kernel_pe.unsqueeze(1)  # Shape: (B, n_heads, max_nodes, max_nodes)

        # 5. Multiply by transformed_edge_weights (element-wise multiplication)
        # attn_scores = attn_scores * transformed_edge_weights.unsqueeze(1)  # Shape: (B, n_heads, max_nodes, max_nodes)
        attn_scores = torch.where(transformed_edge_weights.unsqueeze(1) == 0, float('-inf'), attn_scores * transformed_edge_weights.unsqueeze(1))

        # 6. Apply padding mask (set scores to a large negative value where padding mask is False)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, max_nodes)
            attn_scores = attn_scores.masked_fill(~padding_mask, float('-inf'))  # Mask padded positions with large negative value

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















import torch
import random
import numpy as np

def generate_test_data(B, max_nodes, hidden_dim):
    """
    Generates test data for a batch of B graphs, where each graph has a different number of nodes.
    
    :param B: Number of graphs in the batch
    :param max_nodes: Maximum number of nodes in any graph in the batch
    :param hidden_dim: The dimension of node and edge features
    :return: A dictionary representing a batch of data
    """
    batch = {}
    
    # Simulate node features (_V)
    _V = []
    batch_id = []
    total_nodes = 0  # To track total number of nodes across graphs
    cumulative_nodes = 0  # Cumulative sum of nodes for shifting edge indices
    E_idx = []
    _E = []
    edge_batch_id = []
    mask_list = []

    for i in range(B):
        # Ensure that each graph has a different number of nodes
        num_nodes = random.randint(5, max_nodes)  # Random number of nodes in each graph
        num_edges = random.randint(num_nodes - 1, num_nodes * 2)  # Random number of edges
        
        # Generate random node features
        _V.append(torch.randn(num_nodes, hidden_dim))  # Random node features
        batch_id.append(torch.full((num_nodes,), i))  # Batch ID for each node
        
        # Create a mask for the valid nodes (1 for valid, 0 for padding)
        mask = torch.zeros(max_nodes)
        mask[:num_nodes] = 1  # Mark valid nodes
        mask_list.append(mask)

        # Generate random edge features and edge indices
        edges = []
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            edges.append([src + cumulative_nodes, dst + cumulative_nodes])  # Shifted indices

        E_idx.append(torch.tensor(edges).t())  # Transpose to (2, num_edges)
        _E.append(torch.randn(num_edges, hidden_dim))  # Random edge features
        edge_batch_id.append(torch.full((num_edges,), i))  # Batch ID for each edge

        # Update cumulative node count for the next graph
        cumulative_nodes += num_nodes

    _V = torch.cat(_V, dim=0)  # Flatten all node features into one tensor
    batch_id = torch.cat(batch_id)  # Flatten batch_id into one tensor
    _E = torch.cat(_E, dim=0)  # Flatten all edge features into one tensor
    E_idx = torch.cat(E_idx, dim=1)  # Flatten edge indices into one tensor (2, total_edges)
    edge_batch_id = torch.cat(edge_batch_id)  # Flatten edge batch_id into one tensor

    # Simulate distance matrices (B, max_nodes, max_nodes)
    distance_matrices = []
    for i in range(B):
        num_nodes = (batch_id == i).sum().item()
        dist_matrix = torch.randn(num_nodes, num_nodes)
        dist_matrix = dist_matrix.abs()  # Make all distances positive
        padded_dist_matrix = torch.zeros(max_nodes, max_nodes)
        padded_dist_matrix[:num_nodes, :num_nodes] = dist_matrix
        distance_matrices.append(padded_dist_matrix)

    distance_matrices = torch.stack(distance_matrices, dim=0)  # Stack into (B, max_nodes, max_nodes)

    # Simulate heat kernel positional encodings (B, max_nodes, max_nodes)
    heat_kernel_pe = []
    for i in range(B):
        num_nodes = (batch_id == i).sum().item()
        heat_kernel = torch.randn(num_nodes, num_nodes)
        padded_heat_kernel = torch.zeros(max_nodes, max_nodes)
        padded_heat_kernel[:num_nodes, :num_nodes] = heat_kernel
        heat_kernel_pe.append(padded_heat_kernel)

    heat_kernel_pe = torch.stack(heat_kernel_pe, dim=0)  # Stack into (B, max_nodes, max_nodes)

    # Stack the masks into (B, max_nodes)
    mask = torch.stack(mask_list, dim=0)

    # Update batch dictionary
    batch.update({
        '_V': _V,                           # Flattened node features
        '_E': _E,                           # Flattened edge features
        'batch_id': batch_id,               # Batch ID for each node
        'E_idx': E_idx,                     # Edge indices (cumulative across graphs)
        'inv_distance_matrices': distance_matrices,  # Distance matrices
        'heat_kernel_pe': heat_kernel_pe,    # Heat kernel positional encoding
        'unflattened_mask': mask                        # Node mask (1 for valid, 0 for padding)
    })

    return batch


# Generate test data
hidden_dim = 128
max_nodes = 10
B = 3  # Batch size of 3 graphs (each with different number of nodes)

test_batch = generate_test_data(B, max_nodes, hidden_dim)

# Print test data keys and shapes for verification
for key, value in test_batch.items():
    print(f"{key}: {value.shape}")


hidden_dim = 128
n_layers = 4
n_heads = 8

# Create the model
model = GraphTransformerModel(hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads)

# Forward pass
output = model(test_batch['_V'], test_batch['_E'], test_batch)

# Check output shape
print(f"Output shape: {output.shape}")
