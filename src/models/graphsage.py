"""GraphSAGE layer implementation using PyTorch Geometric."""
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn.conv import MessagePassing


class GraphSage(MessagePassing):
    """GraphSAGE layer implementation.

    Reference: Hamilton et al. "Inductive Representation Learning on Large Graphs" (2017)
    """
    def __init__(self, in_channels, out_channels, args, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.normalize = getattr(args, "normalize", True)
        bias = getattr(args, "bias", False)

        # Linear transformations for self and neighbor aggregation
        # self.lin_l is the linear transformation applied to the embedding
        # of the central node
        # self.lin_r is the linear transformation applied to the aggregated
        # message from the neighbors
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels]
            edge_index: COO graph LongTensor[2, E]
            size:  optional tuple (num_src, num_dst) for bipartite graphs Tuple[int, int]

        Returns:
            Updated node features [N, out_channels]
        """
        # Propagate messages from neighbors
        h_neigh = self.propagate(edge_index, x=(x, x), size=size)

        # Apply linear transformations and combine
        out = self.lin_l(x) + self.lin_r(h_neigh)

        # Apply L2 normalization if specified
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j):
        """Construct messages from neighbors."""
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using mean."""
        # The axis along which to index number of nodes
        node_dim = self.node_dim
        return torch_scatter.scatter(inputs, index, dim=node_dim,
                                     dim_size=dim_size, reduce='mean')