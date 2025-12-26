from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv


class SAGEResBlock(nn.Module):
    """Residual block with SAGEConv layers."""

    def __init__(self, in_channels: int, out_channels: int, feat_drop: float = 0.0):
        """Initializes SAGEResBlock.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            feat_drop: Feature dropout rate
        """
        super(SAGEResBlock, self).__init__()
        self.sage1 = SAGEConv(
            in_channels,
            out_channels,
            aggregator_type="mean",
            feat_drop=feat_drop,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.sage2 = SAGEConv(
            in_channels,
            out_channels,
            aggregator_type="pool",
            feat_drop=feat_drop,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, graph, x, edge_weight=None):
        """Forward pass.

        Args:
            graph: DGL graph
            x: Node features
            edge_weight: Edge weights

        Returns:
            Output features
        """
        out1 = self.sage1(graph, x, edge_weight)
        out1 = self.bn1(out1)

        out2 = self.sage2(graph, x, edge_weight)
        out2 = self.bn2(out2)

        out = self.relu(out1 + out2)
        return out


class ResSAGE(nn.Module):
    """Residual SAGE network for Set Cover problems."""

    def __init__(
        self,
        in_feats: int,
        hidden_sizes: Union[int, list[int]],
        number_classes: int,
        dropout: float,
    ):
        """Initializes ResSAGE.

        Args:
            in_feats: Input feature dimension
            hidden_sizes: Hidden layer dimensions
            number_classes: Number of output classes
            dropout: Dropout rate
        """
        super(ResSAGE, self).__init__()
        self.dropout_frac = dropout
        self.layers = nn.ModuleList()
        current_dim = in_feats

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        for hdim in hidden_sizes:
            self.layers.append(SAGEResBlock(current_dim, hdim))
            self.layers.append(nn.LeakyReLU())
            current_dim = hdim

        self.layers.append(
            SAGEConv(current_dim, number_classes, aggregator_type="mean")
        )

    def forward(self, graph, h, h0, edge_weight=None):
        """Forward pass.

        Args:
            graph: DGL graph
            h: Node features
            h0: Initial features
            edge_weight: Edge weights

        Returns:
            Tuple of (probabilities, logits)
        """
        h = torch.cat([h, h0], 1)

        for layer, norm in zip(self.layers[:-1][::2], self.layers[:-1][1::2]):
            h = layer(graph, h, edge_weight)
            h = norm(h)

        h = F.dropout(h, p=self.dropout_frac)
        h0 = self.layers[-1](graph, h, edge_weight)
        h = torch.sigmoid(h0)

        return h, h0
