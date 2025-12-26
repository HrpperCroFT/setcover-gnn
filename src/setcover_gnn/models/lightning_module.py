import logging
from typing import Dict, Optional, Tuple

import dgl
import lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..utils.graph_utils import pagerank_features
from .gnn_models import ResSAGE

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Dummy dataset for unsupervised training."""

    def __init__(self, size=1):
        """Initializes DummyDataset.

        Args:
            size: Dataset size
        """
        self.size = size

    def __len__(self):
        """Returns the length of dataset. It's a constant int value."""
        return self.size

    def __getitem__(self, idx):
        """Returns element on idx position. It's a constant torch.Tensor.

        Args:
            idx: element position
        """
        return torch.tensor([0])


class SetCoverGNN(pl.LightningModule):
    """PyTorch Lightning module for Set Cover GNN."""

    def __init__(
        self,
        qubo_matrix: torch.Tensor,
        graph: dgl.DGLGraph,
        dim_embedding: int = 10,
        hidden_dim: int = 31,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        prob_threshold: float = 0.5,
        A: float = 4.0,
        n_elements: int = 100,
        clip_grad_norm: float = 2.0,
        penalty_rate: float = 1e-3,
        **kwargs,
    ):
        """Initializes SetCoverGNN.

        Args:
            qubo_matrix: QUBO matrix
            graph: DGL graph
            dim_embedding: Dimension of node embeddings
            hidden_dim: Hidden dimension of GNN
            dropout: Dropout rate
            learning_rate: Learning rate
            prob_threshold: Threshold for binary conversion
            A: QUBO penalty parameter
            n_elements: Number of elements in Set Cover problem
            clip_grad_norm: Gradient clipping norm
            penalty_rate: Penalty rate for loss function
            **kwargs: Additional arguments
        """
        super().__init__()
        self.save_hyperparameters(ignore=["qubo_matrix", "graph"])

        self.qubo_matrix = qubo_matrix
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()

        input_dim = 3 * dim_embedding + 1

        self.model = ResSAGE(
            in_feats=input_dim,
            hidden_sizes=hidden_dim,
            number_classes=1,
            dropout=dropout,
        )

        self.h0 = self._initialize_h0()
        self.inputs = self._initialize_inputs(dim_embedding)

        self.best_bitstring: Optional[torch.Tensor] = None
        self.best_loss = float("inf")
        self.best_sums = float("inf")
        self.best_probs: Optional[torch.Tensor] = None

        self.training_step_outputs: list[torch.Tensor] = []

        self.penalty_rate = penalty_rate

        self.edge_weights = self._compute_edge_weights()

    def _initialize_inputs(self, dim_embedding: int) -> torch.Tensor:
        """Initializes input features for the graph.

        Args:
            dim_embedding: Embedding dimension

        Returns:
            Input features tensor
        """
        device = self.device
        inputs = torch.rand(
            (self.n_nodes, dim_embedding), dtype=self.qubo_matrix.dtype, device=device
        )

        walk = pagerank_features(self.graph.cpu().to_networkx(), 2 * dim_embedding).to(
            device
        )

        inputs = torch.cat([inputs, walk], 1)

        return inputs

    def _initialize_h0(self) -> torch.Tensor:
        """Initializes h0 tensor on the correct device.

        Returns:
            Initialized h0 tensor
        """
        return torch.zeros((self.n_nodes, 1), device=self.device)

    def _compute_edge_weights(self) -> torch.Tensor:
        """Computes edge weights from QUBO matrix.

        Returns:
            Edge weights tensor
        """
        edge_weight = (self.qubo_matrix - torch.diag(self.qubo_matrix.diag())) / 2
        edge_weight = edge_weight + edge_weight.T
        edge_weight = edge_weight[self.graph.edges()[0], self.graph.edges()[1]]
        return edge_weight.to(self.device)

    def _loss_func(self, probs: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """Custom loss function for QUBO optimization.

        Args:
            probs: Node probabilities
            epoch: Current epoch number

        Returns:
            Loss value
        """
        probs_ = torch.unsqueeze(probs, 1)
        lbd = epoch * self.penalty_rate
        penalty = (1 - probs) * probs
        cost = (probs_.T @ self.qubo_matrix @ probs_).squeeze() + lbd * penalty.sum()
        return cost

    def _check_device(self):
        """Ensures that all tensors are on self.device."""
        if self.graph.device != self.device:
            self.graph = self.graph.to(self.device)

        if self.h0.device != self.device:
            self.h0 = self.h0.to(self.device)

        if self.inputs != self.device:
            self.inputs = self.inputs.to(self.device)

        if self.edge_weights.device != self.device:
            self.edge_weights = self.edge_weights.to(self.device)

        if self.qubo_matrix.device != self.device:
            self.qubo_matrix = self.qubo_matrix.to(self.device)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Returns:
            Tuple of (probabilities, logits)
        """
        self._check_device()

        probs, h0 = self.model(
            self.graph, self.inputs, self.h0.detach(), self.edge_weights
        )
        self.h0 = h0.detach()

        return probs.squeeze(), h0

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Training step for PyTorch Lightning with automatic logging.

        Args:
            batch: Batch data
            batch_idx: Batch index

        Returns:
            Dictionary with loss
        """
        probs, _ = self()
        loss = self._loss_func(probs, self.current_epoch)

        bitstring = (probs.detach() >= self.hparams.prob_threshold) * 1
        current_loss = loss.detach().item()

        if current_loss < self.best_loss:
            sums = (
                self._loss_func(bitstring.to(torch.float32), self.current_epoch)
                + self.hparams.A * self.hparams.n_elements
            )

            if self.best_sums > sums:
                self.best_sums = sums
                self.best_loss = current_loss
                self.best_bitstring = bitstring
                self.best_probs = probs

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "best_loss",
            self.best_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("best_sums", self.best_sums, on_step=False, on_epoch=True, logger=True)

        self.training_step_outputs.append(loss)

        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        if not self.training_step_outputs:
            return

        epoch_loss = torch.stack(self.training_step_outputs).mean()

        if (self.current_epoch + 1) % 1000 == 0:
            logger.info(
                f"Epoch {self.current_epoch + 1}: Loss={epoch_loss:.4f}, "
                f"Best loss={self.best_loss:.4f}, Best sum={self.best_sums:.4f}"
            )

        self.training_step_outputs.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures optimizer with gradient clipping.

        Returns:
            Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Override optimizer step to add gradient clipping."""
        if self.hparams.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(), max_norm=self.hparams.clip_grad_norm, norm_type=2
            )

        optimizer.step(closure=optimizer_closure)

    def get_solution(self) -> torch.Tensor:
        """Gets the best found solution.

        Returns:
            Solution bitstring
        """
        if self.best_bitstring is None:
            probs, _ = self()
            return (probs.detach() >= self.hparams.prob_threshold) * 1
        return self.best_bitstring

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step.

        Args:
            batch: Batch data
            batch_idx: Batch index
            dataloader_idx: DataLoader index

        Returns:
            Predicted bitstring
        """
        probs, _ = self()
        bitstring = (probs >= self.hparams.prob_threshold) * 1
        return bitstring

    def train_dataloader(self):
        """Fictive DataLoader for Lightning.

        Returns:
            DataLoader
        """
        return DataLoader(DummyDataset(), batch_size=1)

    def val_dataloader(self):
        """Fictive DataLoader for validation.

        Returns:
            DataLoader
        """
        return DataLoader(DummyDataset(), batch_size=1)

    def test_dataloader(self):
        """Fictive DataLoader for testing.

        Returns:
            DataLoader
        """
        return DataLoader(DummyDataset(), batch_size=1)
