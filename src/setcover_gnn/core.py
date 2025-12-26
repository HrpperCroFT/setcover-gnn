import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import dgl
import numpy as np
import torch

from .data.generation import generate_set_cover_instance
from .data.qubo_conversion import set_cover_to_qubo_qubovert
from .models.lightning_module import SetCoverGNN
from .training.trainer import train_setcover_gnn
from .utils.graph_utils import create_dgl_graph_from_qubo
from .utils.verification import greedy_set_cover, verify_set_cover


@dataclass
class SetCoverProblem:
    """Container for Set Cover problem instance."""

    n_elements: int
    subsets: List[Set[int]]
    qubo_matrix: Optional[torch.Tensor] = None
    graph: Optional[dgl.DGLGraph] = None

    def __post_init__(self):
        """Validates the problem instance."""
        if not self.subsets:
            raise ValueError("Subsets list cannot be empty")

        covered = set()
        for subset in self.subsets:
            covered.update(subset)

        if covered != set(range(1, self.n_elements + 1)):
            missing = set(range(1, self.n_elements + 1)) - covered
            raise ValueError(f"Subsets don't cover all elements. Missing: {missing}")


class SetCoverSolver:
    """Main solver class for Set Cover problems."""

    def __init__(
        self,
        device: str = "auto",
        seed: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes Set Cover solver.

        Args:
            device: Device to run on ('cuda', 'cpu', or 'auto')
            seed: Random seed for reproducibility
            dtype: Torch data type
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = dtype

    def generate_problem(
        self,
        n_elements: int,
        n_subsets: Optional[int] = None,
        coverage_factor: float = 1.5,
        A: float = 4.0,
        B: float = 1.5,
    ) -> SetCoverProblem:
        """Generates a random Set Cover problem.

        Args:
            n_elements: Number of elements in universal set
            n_subsets: Number of subsets (default: random between n/2 and 2n)
            coverage_factor: Controls density of coverage
            A: QUBO penalty weight for covering constraint
            B: QUBO weight for minimizing number of sets

        Returns:
            SetCoverProblem instance
        """
        subsets = generate_set_cover_instance(n_elements, n_subsets, coverage_factor)

        qubo_matrix = set_cover_to_qubo_qubovert(n_elements, subsets, A=A, B=B)

        graph = create_dgl_graph_from_qubo(qubo_matrix)
        qubo_matrix = qubo_matrix.to(self.device).to(self.dtype)
        graph = graph.to(self.device)

        return SetCoverProblem(
            n_elements=n_elements, subsets=subsets, qubo_matrix=qubo_matrix, graph=graph
        )

    def solve_greedy(self, problem: SetCoverProblem) -> Tuple[List[int], bool, int]:
        """Solves using greedy algorithm (baseline).

        Args:
            problem: SetCoverProblem instance

        Returns:
            Tuple of (solution_bitstring, is_valid, selected_count)
        """
        solution = greedy_set_cover(problem.n_elements, problem.subsets)
        is_valid, count = verify_set_cover(
            problem.n_elements, problem.subsets, solution
        )
        return solution, is_valid, count

    def solve_gnn(
        self,
        problem: SetCoverProblem,
        cfg,
        logger=None,
        dim_embedding: int = 10,
        hidden_dim: int = 51,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
        prob_threshold: float = 0.5,
        max_epochs: int = 60000,
        patience: int | None = 100,
        tolerance: float | None = 1e-4,
        **trainer_kwargs,
    ) -> Tuple[SetCoverGNN, List[int], Dict[str, Any]]:
        """Solves using GNN with MLflow logging.

        Args:
            problem: SetCoverProblem instance
            cfg: Configuration dictionary for logging
            logger: Optional logger instance (MLFlowLogger)
            dim_embedding: Dimension of node embeddings
            hidden_dim: Hidden dimension of GNN
            dropout: Dropout rate
            learning_rate: Learning rate
            prob_threshold: Threshold for binary conversion
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            tolerance: Loss tolerance for early stopping
            **trainer_kwargs: Additional trainer arguments

        Returns:
            Tuple of (trained_model, solution_bitstring, metrics)
        """
        if problem.qubo_matrix is None or problem.graph is None:
            raise ValueError("Problem must have qubo_matrix and graph")

        model = SetCoverGNN(
            qubo_matrix=problem.qubo_matrix,
            graph=problem.graph,
            dim_embedding=dim_embedding,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            prob_threshold=prob_threshold,
            A=4.0,
            n_elements=problem.n_elements,
        ).to(self.device)

        trained_model = train_setcover_gnn(
            model,
            cfg=cfg,
            logger=logger,
            max_epochs=max_epochs,
            patience=patience,
            tolerance=tolerance,
            accelerator="gpu" if self.device == "cuda" else "cpu",
            devices=1,
            **trainer_kwargs,
        )

        solution = trained_model.get_solution()
        solution_np = solution.cpu().numpy()[: len(problem.subsets)].tolist()

        is_valid, count = verify_set_cover(
            problem.n_elements, problem.subsets, solution_np
        )

        metrics = {
            "is_valid": is_valid,
            "selected_count": count,
            "solution": solution_np,
        }

        return trained_model, solution_np, metrics

    def solve(
        self, problem: SetCoverProblem, cfg=None, method: str = "gnn", **kwargs
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Solves Set Cover problem with specified method.

        Args:
            problem: SetCoverProblem instance
            cfg: Configuration dictionary for logging (required for 'gnn' method)
            method: 'gnn' or 'greedy'
            **kwargs: Additional arguments for the solver

        Returns:
            Tuple of (solution_bitstring, metrics)
        """
        if method == "greedy":
            solution, is_valid, count = self.solve_greedy(problem)
            metrics = {"is_valid": is_valid, "selected_count": count}
            return solution, metrics

        elif method == "gnn":
            if cfg is None:
                raise ValueError("Configuration (cfg) is required for GNN method")
            _, solution, metrics = self.solve_gnn(problem, cfg, **kwargs)
            return solution, metrics

        else:
            raise ValueError(f"Unknown method: {method}. Use 'gnn' or 'greedy'")
