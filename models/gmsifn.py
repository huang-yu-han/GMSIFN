"""
GMSIFN: Graph-based Multi-Sensor Information Fusion Network

Main model architecture implementing:
1. Input projection layer
2. Graph Generation Layer (GGL) for adaptive graph construction
3. GAT with GRU-based feature aggregation (2 iterations)
4. Graph-level attention with GRU

Reference:
    Paper: Dynamic Graph Meta-Learning with Multi-Sensor Spatial Dependencies for
           Cross-Category Small-Sample Fault Diagnosis in ZDJ9-RTAs
    Original file: ImGAT.py line 1593
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ggl import GGL


class GMSIFN(nn.Module):
    """
    GMSIFN Model Architecture

    Args:
        radius (int): Number of iterations for GAT aggregation (default: 2)
        T (int): Number of iterations for graph-level attention (default: 2)
        input_feature_dim (int): Node feature dimension after projection (default: 240)
        input_bond_dim (int): Edge feature dimension (default: 1 from GGL)
        fingerprint_dim (int): Hidden dimension for node embeddings (default: 240)
        output_units_num (int): Number of output classes (default: 16)
        p_dropout (float): Dropout probability (default: 0.2)
        top_k (int): Number of neighbors in GGL (default: 5)

    Forward:
        input: Input features [batch_size, num_nodes, 5120]
        params: List of model parameters (for meta-learning compatibility)

    Returns:
        output: Class predictions [batch_size, output_units_num]
    """

    def __init__(self, radius, T, input_feature_dim, input_bond_dim,
                 fingerprint_dim, output_units_num, p_dropout, top_k):
        super(GMSIFN, self).__init__()

        # === Node-level Components ===
        # Node feature projection (params 0-1)
        self.node_fc = nn.Linear(input_feature_dim, fingerprint_dim)

        # Neighbor feature projection (params 2-3)
        self.neighbor_fc = nn.Linear(input_feature_dim + input_bond_dim, fingerprint_dim)

        # GRU cells for iterative aggregation (params 4-11)
        self.GRUCell = nn.ModuleList([
            nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)
        ])

        # Attention alignment layers (params 12-15)
        self.align = nn.ModuleList([
            nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)
        ])

        # Attention transformation layers (params 16-19)
        self.attend = nn.ModuleList([
            nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)
        ])

        # === Graph-level Components (Forward) ===
        # GRU for graph aggregation (params 20-23)
        self.graph_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)

        # Graph-level attention (params 24-27)
        self.graph_align = nn.Linear(2 * fingerprint_dim, 1)
        self.graph_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        # === Graph-level Components (Backward) ===
        # GRU for backward aggregation (params 34-37)
        self.graph_GRUCell2 = nn.GRUCell(fingerprint_dim, fingerprint_dim)

        # Backward graph-level attention (params 30-33)
        self.graph_align2 = nn.Linear(2 * fingerprint_dim, 1)
        self.graph_attend2 = nn.Linear(fingerprint_dim, fingerprint_dim)

        # === Output Layer ===
        # Final classification layer (params 28-29)
        self.output = nn.Linear(fingerprint_dim, output_units_num)

        # === Preprocessing Layers ===
        # Input projection (params 38-39)
        self.node_input = nn.Linear(5120, input_feature_dim)

        # Graph Generation Layer (params 40-42)
        self.atrr = GGL(top_k=top_k, dim=input_feature_dim)

        # === Hyperparameters ===
        self.dropout = nn.Dropout(p=p_dropout)
        self.radius = radius
        self.T = T

    def forward(self, input, params, type=None):
        """
        Forward pass through GMSIFN

        Parameter Index Mapping (for meta-learning):
            0-1: node_fc
            2-3: neighbor_fc
            4-7: GRUCell[0]
            8-11: GRUCell[1]
            12-13: align[0]
            14-15: align[1]
            16-17: attend[0]
            18-19: attend[1]
            20-23: graph_GRUCell
            24-25: graph_align
            26-27: graph_attend
            28-31: graph_GRUCell2
            32-33: graph_align2
            34-35: graph_attend2
            36-37: output
            38-39: node_input
            40-42: atrr (GGL layer)
        """
        params = [p.cuda() for p in params]

        # === Stage 1: Input Projection ===
        node_list = input.squeeze()
        node_list = F.leaky_relu(F.linear(node_list, params[38], params[39]))

        # === Stage 2: Graph Generation ===
        node_neighbor, bond_neighbor = self.atrr(node_list, params[40:43])
        node_neighbor, bond_neighbor = node_neighbor.cuda(), bond_neighbor.cuda()

        batch_size, num_nodes, node_feat_dim = node_list.size()

        # === Stage 3: GAT Aggregation ===
        # Initial node feature projection
        node_feature = F.leaky_relu(F.linear(node_list, params[0], params[1]))

        # Concatenate neighbor node and edge features
        neighbor_feature = torch.cat([node_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(F.linear(neighbor_feature, params[2], params[3]))

        batch_size, num_nodes, max_neighbor_num, fingerprint_dim = neighbor_feature.shape

        # --- Hop 1 ---
        node_feature_expand = node_feature.unsqueeze(-2).expand(
            batch_size, num_nodes, max_neighbor_num, fingerprint_dim
        )
        feature_align = torch.cat([node_feature_expand, neighbor_feature], dim=-1)

        # Attention mechanism
        align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[12], params[13]))
        attention_weight = F.softmax(align_score, -2)

        # Weighted neighbor aggregation
        neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[16], params[17])
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        context = F.elu(context)

        # GRU update
        context_reshape = context.view(batch_size * num_nodes, fingerprint_dim)
        node_feature_reshape = node_feature.view(batch_size * num_nodes, fingerprint_dim)

        r = torch.sigmoid(
            F.linear(context_reshape, params[4][:fingerprint_dim], params[6][:fingerprint_dim]) +
            F.linear(node_feature_reshape, params[5][:fingerprint_dim], params[7][:fingerprint_dim])
        )
        z = torch.sigmoid(
            F.linear(context_reshape, params[4][fingerprint_dim:fingerprint_dim*2],
                     params[6][fingerprint_dim:fingerprint_dim*2]) +
            F.linear(node_feature_reshape, params[5][fingerprint_dim:fingerprint_dim*2],
                     params[7][fingerprint_dim:fingerprint_dim*2])
        )
        n = torch.tanh(
            F.linear(context_reshape, params[4][fingerprint_dim*2:], params[6][fingerprint_dim*2:]) +
            torch.mul(r, F.linear(node_feature_reshape, params[5][fingerprint_dim*2:],
                                  params[7][fingerprint_dim*2:]))
        )
        node_feature_reshape = torch.mul((1 - z), n) + torch.mul(node_feature_reshape, z)
        node_feature = node_feature_reshape.view(batch_size, num_nodes, fingerprint_dim)
        activated_features = F.relu(node_feature)

        # --- Hop 2 (if radius > 1) ---
        for d in range(self.radius - 1):
            neighbor_feature, _ = self.atrr(activated_features)

            node_feature_expand = activated_features.unsqueeze(-2).expand(
                batch_size, num_nodes, max_neighbor_num, fingerprint_dim
            )
            feature_align = torch.cat([node_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[14], params[15]))
            attention_weight = F.softmax(align_score, -2)

            neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[18], params[19])
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            context = F.elu(context)
            context_reshape = context.view(batch_size * num_nodes, fingerprint_dim)

            # GRU update
            r = torch.sigmoid(
                F.linear(context_reshape, params[8][:fingerprint_dim], params[10][:fingerprint_dim]) +
                F.linear(node_feature_reshape, params[9][:fingerprint_dim], params[11][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(context_reshape, params[8][fingerprint_dim:fingerprint_dim*2],
                         params[10][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(node_feature_reshape, params[9][fingerprint_dim:fingerprint_dim*2],
                         params[11][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(context_reshape, params[8][fingerprint_dim*2:], params[10][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(node_feature_reshape, params[9][fingerprint_dim*2:],
                                      params[11][fingerprint_dim*2:]))
            )
            node_feature_reshape = torch.mul((1 - z), n) + torch.mul(node_feature_reshape, z)
            node_feature = node_feature_reshape.view(batch_size, num_nodes, fingerprint_dim)
            activated_features = F.relu(node_feature)

        # === Stage 4: Graph-level Readout ===
        graph_feature = torch.sum(activated_features, dim=-2)
        activated_features_graph = F.relu(graph_feature)

        # Clone for legacy compatibility (backward GRU not used in output)
        activated_features_graph2 = activated_features_graph.clone()
        graph_feature2 = graph_feature.clone()

        # --- Forward Direction ---
        for t in range(self.T):
            graph_prediction_expand = activated_features_graph.unsqueeze(-2).expand(
                batch_size, num_nodes, fingerprint_dim
            )
            graph_align = torch.cat([graph_prediction_expand, activated_features], dim=-1)

            graph_align_score = F.leaky_relu(F.linear(graph_align, params[24], params[25]))
            graph_attention_weight = F.softmax(graph_align_score, -2)

            activated_features_transform = F.linear(self.dropout(activated_features), params[26], params[27])
            graph_context = torch.sum(torch.mul(graph_attention_weight, activated_features_transform), -2)
            graph_context = F.elu(graph_context)

            # GRU update
            r = torch.sigmoid(
                F.linear(graph_context, params[20][:fingerprint_dim], params[22][:fingerprint_dim]) +
                F.linear(graph_feature, params[21][:fingerprint_dim], params[23][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(graph_context, params[20][fingerprint_dim:fingerprint_dim*2],
                         params[22][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(graph_feature, params[21][fingerprint_dim:fingerprint_dim*2],
                         params[23][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(graph_context, params[20][fingerprint_dim*2:], params[22][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(graph_feature, params[21][fingerprint_dim*2:],
                                      params[23][fingerprint_dim*2:]))
            )
            graph_feature = torch.mul((1 - z), n) + torch.mul(graph_feature, z)
            activated_features_graph = F.relu(graph_feature)

        # --- Backward Direction ---
        activated_features_graph_reverse = torch.flip(activated_features_graph2, dims=[0])
        activated_features_reverse = torch.flip(activated_features, dims=[0])
        graph_feature_reverse = torch.flip(graph_feature2.clone(), dims=[0])

        for t in range(self.T):
            graph_prediction_expand = activated_features_graph_reverse.unsqueeze(-2).expand(
                batch_size, num_nodes, fingerprint_dim
            )
            graph_align = torch.cat([graph_prediction_expand, activated_features_reverse], dim=-1)

            graph_align_score = F.leaky_relu(F.linear(graph_align, params[32], params[33]))
            graph_attention_weight = F.softmax(graph_align_score, -2)

            activated_features_transform = F.linear(self.dropout(activated_features_reverse), params[34], params[35])
            graph_context = torch.sum(torch.mul(graph_attention_weight, activated_features_transform), -2)
            graph_context = F.elu(graph_context)

            # GRU update
            r = torch.sigmoid(
                F.linear(graph_context, params[28][:fingerprint_dim], params[30][:fingerprint_dim]) +
                F.linear(graph_feature_reverse, params[29][:fingerprint_dim], params[31][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(graph_context, params[28][fingerprint_dim:fingerprint_dim*2],
                         params[30][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(graph_feature_reverse, params[29][fingerprint_dim:fingerprint_dim*2],
                         params[31][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(graph_context, params[28][fingerprint_dim*2:], params[30][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(graph_feature_reverse, params[29][fingerprint_dim*2:],
                                      params[31][fingerprint_dim*2:]))
            )
            graph_feature_reverse = torch.mul((1 - z), n) + torch.mul(graph_feature_reverse, z)
            activated_features_graph_reverse = F.relu(graph_feature_reverse)

        # === Stage 5: Final Output ===
        graph_feature_all = graph_feature
        graph_prediction = F.linear(self.dropout(graph_feature_all), params[36], params[37])

        return graph_prediction

    def get_features(self, input, params, type=None):
        """Extract intermediate features for visualization (e.g., t-SNE)"""
        params = [p.cuda() for p in params]

        # Repeat forward pass up to graph feature extraction
        node_list = input.squeeze()
        node_list = F.leaky_relu(F.linear(node_list, params[38], params[39]))

        node_neighbor, bond_neighbor = self.atrr(node_list, params[40:43])
        node_neighbor, bond_neighbor = node_neighbor.cuda(), bond_neighbor.cuda()

        batch_size, num_nodes, node_feat_dim = node_list.size()

        node_feature = F.leaky_relu(F.linear(node_list, params[0], params[1]))
        neighbor_feature = torch.cat([node_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(F.linear(neighbor_feature, params[2], params[3]))

        batch_size, num_nodes, max_neighbor_num, fingerprint_dim = neighbor_feature.shape
        node_feature_expand = node_feature.unsqueeze(-2).expand(
            batch_size, num_nodes, max_neighbor_num, fingerprint_dim
        )
        feature_align = torch.cat([node_feature_expand, neighbor_feature], dim=-1)

        # Hop 1
        align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[12], params[13]))
        attention_weight = F.softmax(align_score, -2)
        neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[16], params[17])
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        context = F.elu(context)

        context_reshape = context.view(batch_size * num_nodes, fingerprint_dim)
        node_feature_reshape = node_feature.view(batch_size * num_nodes, fingerprint_dim)

        # GRU update
        r = torch.sigmoid(
            F.linear(context_reshape, params[4][:fingerprint_dim], params[6][:fingerprint_dim]) +
            F.linear(node_feature_reshape, params[5][:fingerprint_dim], params[7][:fingerprint_dim])
        )
        z = torch.sigmoid(
            F.linear(context_reshape, params[4][fingerprint_dim:fingerprint_dim*2],
                     params[6][fingerprint_dim:fingerprint_dim*2]) +
            F.linear(node_feature_reshape, params[5][fingerprint_dim:fingerprint_dim*2],
                     params[7][fingerprint_dim:fingerprint_dim*2])
        )
        n = torch.tanh(
            F.linear(context_reshape, params[4][fingerprint_dim*2:], params[6][fingerprint_dim*2:]) +
            torch.mul(r, F.linear(node_feature_reshape, params[5][fingerprint_dim*2:],
                                  params[7][fingerprint_dim*2:]))
        )
        node_feature_reshape = torch.mul((1 - z), n) + torch.mul(node_feature_reshape, z)
        node_feature = node_feature_reshape.view(batch_size, num_nodes, fingerprint_dim)
        activated_features = F.relu(node_feature)

        # Hop 2
        for d in range(self.radius - 1):
            neighbor_feature, _ = self.atrr(activated_features)
            node_feature_expand = activated_features.unsqueeze(-2).expand(
                batch_size, num_nodes, max_neighbor_num, fingerprint_dim
            )
            feature_align = torch.cat([node_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(F.linear(self.dropout(feature_align), params[14], params[15]))
            attention_weight = F.softmax(align_score, -2)

            neighbor_feature_transform = F.linear(self.dropout(neighbor_feature), params[18], params[19])
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            context = F.elu(context)
            context_reshape = context.view(batch_size * num_nodes, fingerprint_dim)

            # GRU update
            r = torch.sigmoid(
                F.linear(context_reshape, params[8][:fingerprint_dim], params[10][:fingerprint_dim]) +
                F.linear(node_feature_reshape, params[9][:fingerprint_dim], params[11][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(context_reshape, params[8][fingerprint_dim:fingerprint_dim*2],
                         params[10][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(node_feature_reshape, params[9][fingerprint_dim:fingerprint_dim*2],
                         params[11][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(context_reshape, params[8][fingerprint_dim*2:], params[10][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(node_feature_reshape, params[9][fingerprint_dim*2:],
                                      params[11][fingerprint_dim*2:]))
            )
            node_feature_reshape = torch.mul((1 - z), n) + torch.mul(node_feature_reshape, z)
            node_feature = node_feature_reshape.view(batch_size, num_nodes, fingerprint_dim)
            activated_features = F.relu(node_feature)

        # Graph-level aggregation (forward only)
        graph_feature = torch.sum(activated_features, dim=-2)
        activated_features_graph = F.relu(graph_feature)
        activated_features_graph2 = activated_features_graph.clone()
        graph_feature2 = graph_feature.clone()

        for t in range(self.T):
            graph_prediction_expand = activated_features_graph.unsqueeze(-2).expand(
                batch_size, num_nodes, fingerprint_dim
            )
            graph_align = torch.cat([graph_prediction_expand, activated_features], dim=-1)

            graph_align_score = F.leaky_relu(F.linear(graph_align, params[24], params[25]))
            graph_attention_weight = F.softmax(graph_align_score, -2)

            activated_features_transform = F.linear(self.dropout(activated_features), params[26], params[27])
            graph_context = torch.sum(torch.mul(graph_attention_weight, activated_features_transform), -2)
            graph_context = F.elu(graph_context)

            # GRU update
            r = torch.sigmoid(
                F.linear(graph_context, params[20][:fingerprint_dim], params[22][:fingerprint_dim]) +
                F.linear(graph_feature, params[21][:fingerprint_dim], params[23][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(graph_context, params[20][fingerprint_dim:fingerprint_dim*2],
                         params[22][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(graph_feature, params[21][fingerprint_dim:fingerprint_dim*2],
                         params[23][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(graph_context, params[20][fingerprint_dim*2:], params[22][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(graph_feature, params[21][fingerprint_dim*2:],
                                      params[23][fingerprint_dim*2:]))
            )
            graph_feature = torch.mul((1 - z), n) + torch.mul(graph_feature, z)
            activated_features_graph = F.relu(graph_feature)

        # Backward direction
        activated_features_graph_reverse = torch.flip(activated_features_graph2, dims=[0])
        activated_features_reverse = torch.flip(activated_features, dims=[0])
        graph_feature_reverse = torch.flip(graph_feature2.clone(), dims=[0])

        for t in range(self.T):
            graph_prediction_expand = activated_features_graph_reverse.unsqueeze(-2).expand(
                batch_size, num_nodes, fingerprint_dim
            )
            graph_align = torch.cat([graph_prediction_expand, activated_features_reverse], dim=-1)

            graph_align_score = F.leaky_relu(F.linear(graph_align, params[32], params[33]))
            graph_attention_weight = F.softmax(graph_align_score, -2)

            activated_features_transform = F.linear(self.dropout(activated_features_reverse), params[34], params[35])
            graph_context = torch.sum(torch.mul(graph_attention_weight, activated_features_transform), -2)
            graph_context = F.elu(graph_context)

            # GRU update
            r = torch.sigmoid(
                F.linear(graph_context, params[28][:fingerprint_dim], params[30][:fingerprint_dim]) +
                F.linear(graph_feature_reverse, params[29][:fingerprint_dim], params[31][:fingerprint_dim])
            )
            z = torch.sigmoid(
                F.linear(graph_context, params[28][fingerprint_dim:fingerprint_dim*2],
                         params[30][fingerprint_dim:fingerprint_dim*2]) +
                F.linear(graph_feature_reverse, params[29][fingerprint_dim:fingerprint_dim*2],
                         params[31][fingerprint_dim:fingerprint_dim*2])
            )
            n = torch.tanh(
                F.linear(graph_context, params[28][fingerprint_dim*2:], params[30][fingerprint_dim*2:]) +
                torch.mul(r, F.linear(graph_feature_reverse, params[29][fingerprint_dim*2:],
                                      params[31][fingerprint_dim*2:]))
            )
            graph_feature_reverse = torch.mul((1 - z), n) + torch.mul(graph_feature_reverse, z)
            activated_features_graph_reverse = F.relu(graph_feature_reverse)

        graph_feature_all = graph_feature
        return graph_feature_all
