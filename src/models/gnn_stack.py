"""Flexible GNN Stack supporting multiple architectures."""
import torch.nn as nn
import torch.nn.functional as F
from .graphsage import GraphSage
from .gat import GAT


class GNNStack(nn.Module):
    """Flexible GNN architecture supporting GraphSAGE and GAT."""

    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()

        ConvModel = self.build_conv_model(args.model_type)
        self.heads = getattr(args, "heads", 1)

        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(ConvModel(input_dim, hidden_dim, args))

        # Hidden layers
        assert args.num_layers >= 1, 'Number of layers must be >= 1'
        for _ in range(args.num_layers-1):
            self.convs.append(ConvModel(self.heads * hidden_dim, hidden_dim, args))

        # Post-message-passing layers
        self.post_mp = nn.Sequential(
            nn.Linear(self.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.emb = emb

    def build_conv_model(self, model_type):
        """Get the appropriate convolution model."""
        model_type = model_type.lower()
        if model_type == 'graphsage':
            return GraphSage
        elif model_type == 'gat':
            return GAT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, data):
        """Forward pass through the GNN stack."""
        x, edge_index = data.x, data.edge_index

        # Pass through GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Post-processing
        x = self.post_mp(x)

        if self.emb:
            return x

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        """Compute negative log likelihood loss."""
        return F.nll_loss(pred, label)
