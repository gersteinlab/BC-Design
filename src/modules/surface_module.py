import torch
import torch.nn as nn

class SurfaceEncoder(nn.Module):
    # def __init__(self, vertex_dim=3, feature_dim=2, hidden_dim=256, local_layers=1, global_layers=2, num_heads=8, K=30, dropout_prob=0.3):
    def __init__(self, vertex_dim=3, feature_dim=2, hidden_dim=128, local_layers=1, global_layers=2, num_heads=8, K=30, dropout_prob=0.3):
        super(SurfaceEncoder, self).__init__()

        self.K = K
        self.hidden_dim = hidden_dim

        # Mapping matrix Wm with bias set to False
        self.Wm = nn.Linear(feature_dim, hidden_dim, bias=False)

        # Local Perspective Modeling - Equivariant Graph Convolutional Layer (EGCL)
        self.local_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2 + 1, hidden_dim) for _ in range(local_layers)
        ])

        self.local_activation = nn.SiLU()

        # Weight computation parameter
        self.weight_layer = nn.Linear(hidden_dim, 1)  # Single layer for weight computation

        # Gating mechanism
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(local_layers)
        ])

        # Dropout layers for local layers and gate layers
        self.local_dropout = nn.Dropout(dropout_prob)

        # Linear layer to project (256 + 3) to 256
        self.projection_layer = nn.Linear(hidden_dim + 3, hidden_dim)
        self.projection_dropout = nn.Dropout(dropout_prob)

        # Global Landscape Modeling - Frame Averaging Multi-Head Attention (FAMHA)
        # self.global_layers = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        #     for _ in range(global_layers)
        # ])
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=dropout_prob)
        self.global_encoder = nn.TransformerEncoder(encoder_layer, num_layers=global_layers)

        # self.projection_layer2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        # self.projection_dropout2 = nn.Dropout(dropout_prob)

    def forward(self, vertices, biochemical_features, principal_components, memory_padding_mask):
        device = vertices.device
        vertices = vertices.to(device)
        biochemical_features = biochemical_features.to(device)

        batch_size, num_vertices = vertices.shape[:2]

        # Local Perspective Modeling
        hidden_states = self.Wm(biochemical_features)

        for l in range(len(self.local_layers)):
            # Finding K-nearest neighbors
            distances = torch.cdist(vertices, vertices)  # [batch_size, N, N]
            knn_indices = distances.argsort(dim=-1)[:, :, :self.K]  # [batch_size, N, K]

            # Get distances between vertices and their neighbors
            vertices_expanded = vertices.unsqueeze(2).expand(-1, -1, self.K, -1)  # [batch_size, N, K, D]
            neighbor_vertices = torch.gather(vertices.unsqueeze(1).expand(-1, num_vertices, -1, -1), 2, knn_indices.unsqueeze(-1).expand(-1, -1, -1, vertices.size(-1)))  # [batch_size, N, K, D]
            dists = torch.norm(vertices_expanded - neighbor_vertices, dim=-1)  # [batch_size, N, K]

            # Prepare input features for the local layers
            hidden_states_expanded = hidden_states.unsqueeze(2).expand(-1, -1, self.K, -1)  # [batch_size, N, K, hidden_dim]
            neighbor_hidden_states = torch.gather(hidden_states.unsqueeze(1).expand(-1, num_vertices, -1, -1), 2, knn_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1)))  # [batch_size, N, K, hidden_dim]
            input_features = torch.cat([hidden_states_expanded, neighbor_hidden_states, dists.unsqueeze(-1)], dim=-1)  # [batch_size, N, K, hidden_dim * 2 + 1]

            # Apply local layers and activation
            m_prime_ij = self.local_activation(self.local_layers[l](input_features))  # [batch_size, N, K, hidden_dim]
            m_prime_ij = self.local_dropout(m_prime_ij)

            # Compute weights
            weights = torch.exp(self.weight_layer(m_prime_ij).squeeze(-1))  # [batch_size, N, K]
            weights = weights / weights.sum(dim=-1, keepdim=True)

            # Compute weighted sum of messages
            c_l_plus_1 = (weights.unsqueeze(-1) * m_prime_ij).sum(dim=2)  # [batch_size, N, hidden_dim]
            gate = self.gate_layers[l](c_l_plus_1)  # [batch_size, N, hidden_dim]
            gate = self.local_dropout(gate)
            hidden_states = hidden_states + gate * c_l_plus_1  # [batch_size, N, hidden_dim]

        # Define the 8 possible transformations using broadcasting
        signs = torch.tensor([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                              [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], device=device, dtype=torch.float32)

        # Stack principal components with different sign combinations using broadcasting
        signs = signs.unsqueeze(0).unsqueeze(3)  # Shape: [1, 8, 3, 1]
        principal_components = principal_components.unsqueeze(1).transpose(2, 3)  # Shape: [batch_size, 1, 3, D]
        transformed_vertices = torch.matmul(vertices.unsqueeze(2).unsqueeze(3), signs * principal_components.unsqueeze(1))  # Shape: [batch_size, N, 8, 1, 3]
        transformed_vertices = transformed_vertices.squeeze(3)  # Shape: [batch_size, N, 8, 3]

        # Concatenate hidden states with inverse transformed point cloud coordinates
        hidden_states_expanded = hidden_states.unsqueeze(2).expand(-1, -1, 8, -1)  # [batch_size, N, 8, hidden_dim]
        concatenated_input = torch.cat((hidden_states_expanded, transformed_vertices), dim=-1)  # [batch_size, N, 8, hidden_dim + 3]

        # Project concatenated input to hidden_dim
        projected_input = self.projection_layer(concatenated_input)  # [batch_size, N, 8, hidden_dim]
        projected_input = self.projection_dropout(projected_input)

        # Global Landscape Modeling (FAMHA)
        projected_input = projected_input.permute(0, 2, 1, 3).reshape(batch_size * 8, num_vertices, self.hidden_dim)  # [batch_size * 8, N, hidden_dim + 3]
        # for layer in self.global_layers:
        #     projected_input = layer(projected_input)  # [batch_size * 8, N, hidden_dim + 3]
        projected_input = self.global_encoder(
            projected_input,
            src_key_padding_mask=memory_padding_mask.repeat(8, 1)
        )  # [batch_size * 8, N, hidden_dim + 3]
        projected_input = projected_input.view(batch_size, 8, num_vertices, self.hidden_dim).mean(dim=1)  # [batch_size, N, hidden_dim + 3]
        # hidden_states = self.projection_layer2(projected_input)  
        # hidden_states = self.projection_dropout2(hidden_states)

        return hidden_states
