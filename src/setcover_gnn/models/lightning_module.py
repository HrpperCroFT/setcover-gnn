import torch
import torch.nn as nn
import lightning as pl
import dgl
from typing import Dict, Tuple, Optional
import logging

from .gnn_models import ResSAGE
from ..utils.graph_utils import pagerank_features

logger = logging.getLogger(__name__)


class SetCoverGNN(pl.LightningModule):
    """PyTorch Lightning module for Set Cover GNN."""
    
    def __init__(
        self,
        qubo_matrix: torch.Tensor,
        graph: dgl.DGLGraph,
        dim_embedding: int = 10,
        hidden_dim: int = 51,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        prob_threshold: float = 0.5,
        A: float = 4.0,
        n_elements: int = 100,
        clip_grad_norm: float = 2.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['qubo_matrix', 'graph'])
        
        self.qubo_matrix = qubo_matrix
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        
        # Calculate input features dimension
        input_dim = dim_embedding + 1 + 4 * dim_embedding
        
        # Initialize model
        self.model = ResSAGE(
            in_feats=input_dim,
            hidden_sizes=hidden_dim,
            number_classes=1,
            dropout=dropout
        )
        
        # Initialize state tensors
        self.register_buffer('h0', torch.zeros(self.n_nodes, 1))
        self.inputs = self._initialize_inputs(dim_embedding)
        
        # For tracking best solution
        self.best_bitstring: Optional[torch.Tensor] = None
        self.best_loss = float('inf')
        self.best_probs: Optional[torch.Tensor] = None
        
        # Track training metrics
        self.training_step_outputs = []
        
    def _initialize_inputs(self, dim_embedding: int) -> torch.Tensor:
        """Initialize input features for the graph."""
        device = self.device
        inputs = torch.rand(
            (self.n_nodes, dim_embedding), 
            dtype=self.qubo_matrix.dtype,
            device=device
        )
        
        # Add pagerank features
        walk = pagerank_features(
            self.graph.cpu().to_networkx(), 
            2 * dim_embedding
        ).to(device)
        
        inputs = torch.cat([
            inputs, 
            torch.ones_like(inputs), 
            torch.ones_like(inputs), 
            walk
        ], 1)
        
        return inputs
    
    def _compute_edge_weights(self) -> torch.Tensor:
        """Compute edge weights from QUBO matrix."""
        edge_weight = (self.qubo_matrix - torch.diag(self.qubo_matrix.diag())) / 2
        edge_weight = edge_weight + edge_weight.T
        edge_weight = edge_weight[self.graph.edges()[0], self.graph.edges()[1]]
        return edge_weight
    
    def _loss_func(self, probs: torch.Tensor, epoch: int = 0) -> torch.Tensor:
        """Custom loss function for QUBO optimization."""
        probs_ = torch.unsqueeze(probs, 1)
        lbd = epoch / 1e3
        penalty = (1 - probs) * probs
        cost = (probs_.T @ self.qubo_matrix @ probs_).squeeze() + lbd * penalty.sum()
        return cost
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        edge_weight = self._compute_edge_weights()
        
        probs, h0 = self.model(
            self.graph, 
            self.inputs, 
            self.h0.detach(), 
            edge_weight
        )
        
        # Update state for next iteration
        self.h0 = h0.detach()
        
        return probs.squeeze(), h0
    
    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """Training step for PyTorch Lightning."""
        probs, _ = self()
        loss = self._loss_func(probs, self.current_epoch)
        
        # Track best solution
        bitstring = (probs.detach() >= self.hparams.prob_threshold) * 1
        current_loss = loss.detach().item()
        
        if current_loss < self.best_loss:
            sums = self._loss_func(
                bitstring.to(torch.float32), 
                self.current_epoch
            ) + self.hparams.A * self.hparams.n_elements
            
            if self.best_loss > sums:
                self.best_loss = current_loss
                self.best_bitstring = bitstring
                self.best_probs = probs
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('best_loss', self.best_loss, prog_bar=True)
        
        # Store for epoch end
        self.training_step_outputs.append(loss)
        
        return {'loss': loss}
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        
        if self.current_epoch % 1000 == 0:
            logger.info(f'Epoch {self.current_epoch}: Loss={epoch_loss:.4f}, Best={self.best_loss:.4f}')
        
        # Clear memory
        self.training_step_outputs.clear()
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer with gradient clipping."""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        return optimizer
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """Override optimizer step to add gradient clipping."""
        optimizer.step(closure=optimizer_closure)
        
        # Clip gradients
        if self.hparams.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(
                self.parameters(), 
                max_norm=self.hparams.clip_grad_norm, 
                norm_type=2
            )
    
    def get_solution(self) -> torch.Tensor:
        """Get the best found solution."""
        if self.best_bitstring is None:
            probs, _ = self()
            return (probs.detach() >= self.hparams.prob_threshold) * 1
        return self.best_bitstring
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step."""
        probs, _ = self()
        bitstring = (probs >= self.hparams.prob_threshold) * 1
        return bitstring