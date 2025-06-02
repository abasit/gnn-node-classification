"""Graph Attention Network (GAT) layer implentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, add_self_loops


class GAT(MessagePassing):
    """Graph Attention Network layer.

    Reference: Veličković et al. "Graph Attention Networks" (2018)
    """
    def __init__(self, in_channels, out_channels, args, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.heads = getattr(args, "heads", 2)
        self.negative_slope = getattr(args, "negative_slope", 0.2)
        self.dropout = getattr(args, "dropout", 0.)

        # Linear transformations for multi-head attention
        self.lin_l = nn.Linear(in_channels, self.heads * out_channels, bias=False)
        self.lin_r = nn.Linear(in_channels, self.heads * out_channels, bias=False)

        # Attention parameters for each head
        self.att_l = nn.Parameter(torch.empty(self.heads, out_channels))
        self.att_r = nn.Parameter(torch.empty(self.heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        """
        Forward pass with multi-head attention.

        Args:
            x: Node features [N, in_channels]
            edge_index: COO graph [2, E]
            size:  optional tuple (num_src, num_dst) for bipartite graphs Tuple[int, int]

        Returns:
            Updated node features [N, heads * out_channels]
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)

        # Apply linear transformations and reshape for multi-head
        h_l = self.lin_l(x).view(N, H, C)  # Shape [N, H, C]
        h_r = self.lin_r(x).view(N, H, C)  # Shape [N, H, C]

        # Compute attention scores for each head
        alpha_l = (h_l * self.att_l).sum(dim=-1) # Shape [N, H]
        alpha_r = (h_r * self.att_r).sum(dim=-1) # Shape [N, H]

        # Add self-loops so each node attends to itself
        edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # Propagate messages with attention
        out = self.propagate(edge_index, x=(h_l, h_r),
                             alpha=(alpha_l, alpha_r), size=size)

        # Reshape output
        out = out.view(N, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        """Compute messages with attention weights."""
        # Compute unnormalized attention coefficients
        e_ij = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)

        # Apply softmax normalization over neighbors
        alpha = softmax(e_ij, index, ptr=ptr, num_nodes=size_i)

        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weight messages by attention coefficients
        out = x_j * alpha.unsqueeze(-1) # Shape [E, H, C]

        return out

    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using sum."""
        return torch_scatter.scatter(inputs, index, dim=0,
                                     dim_size=dim_size, reduce="sum")