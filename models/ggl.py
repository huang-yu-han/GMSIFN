"""
Graph Generation Layer (GGL)

This module implements the self-adaptive graph construction mechanism
that dynamically builds graph structures based on node similarity.

Reference:
    Paper: Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for
           Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs
    Section: Graph Generation Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GGL(nn.Module):
    """
    Graph Generation Layer (GGL)

    Dynamically constructs graph structure by:
    1. Computing node similarity via learned transformation
    2. Selecting top-k neighbors for each node
    3. Generating edge weights via sigmoid normalization

    Args:
        top_k (int): Number of neighbors to select for each node
        dim (int): Dimension of node features

    Forward:
        x: Node features [batch_size, num_nodes, feat_dim]
        params: Optional list of parameters for meta-learning

    Returns:
        node_neighbor: Neighbor node features [B, N, K, feat_dim]
        bond_neighbor: Edge weights [B, N, K, 1]
    """

    def __init__(self, top_k, dim):
        super(GGL, self).__init__()
        self.top_k = top_k

        # Learnable parameters (Index 54-56 in full model)
        self.edge_weight = nn.Parameter(torch.randn(1))  # Index 54
        self.node_transform = nn.Linear(dim, 10)  # Index 55-56

    def forward(self, x, params=None):
        """
        Forward pass with adaptive graph construction

        Parameter mapping (when params is provided):
            params[0]: edge_weight (Index 54)
            params[1]: node_transform.weight (Index 55)
            params[2]: node_transform.bias (Index 56)
        """
        # Handle batch dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batch_size, num_nodes, feat_dim = x.shape

        # Extract parameters
        if params is not None:
            edge_weight = params[0]  # Index 54
            node_weight = params[1]  # Index 55
            node_bias = params[2]    # Index 56
        else:
            edge_weight = self.edge_weight
            node_weight = self.node_transform.weight
            node_bias = self.node_transform.bias

        # Step 1: Node feature transformation
        x_transformed = F.linear(x, node_weight, node_bias)
        x_transformed = F.leaky_relu(x_transformed)

        # Step 2: Similarity computation with learnable edge weight
        similarity = torch.matmul(x_transformed, x_transformed.transpose(1, 2))
        similarity = similarity * edge_weight.exp()  # Ensure positive weights

        # Step 3: Top-k neighbor selection (excluding self-connections)
        mask = torch.eye(num_nodes, dtype=torch.bool, device=x.device)
        similarity.masked_fill_(mask.unsqueeze(0), float('-inf'))
        topk_values, topk_indices = similarity.topk(self.top_k, dim=2)

        # Step 4: Construct neighbor features
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        node_neighbor = torch.gather(
            x.unsqueeze(1).expand(-1, num_nodes, -1, -1),
            dim=2,
            index=expanded_indices
        )

        # Step 5: Edge weight normalization
        bond_neighbor = torch.sigmoid(topk_values.unsqueeze(-1))

        return node_neighbor, bond_neighbor

    def parameters(self):
        """Return parameters in strict index order for meta-learning"""
        return [
            self.edge_weight,              # Index 54
            self.node_transform.weight,    # Index 55
            self.node_transform.bias       # Index 56
        ]
