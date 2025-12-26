from typing import List, Set

import torch
from qubovert.problems import SetCover
from qubovert.utils import qubo_to_matrix


def set_cover_to_qubo_qubovert(
    n: int, subsets: List[Set[int]], A: float = 2.0, B: float = 1.0
) -> torch.Tensor:
    """Convert Set Cover problem to QUBO matrix using qubovert.

    Args:
        n: Number of elements
        subsets: List of subsets
        A: Penalty weight for covering constraint
        B: Weight for minimizing number of sets

    Returns:
        QUBO matrix as torch.Tensor
    """
    covered = set()
    for subset in subsets:
        covered.update(subset)

    if covered != set(range(1, n + 1)):
        raise ValueError(
            "Subsets don't cover all elements. "
            f"Missing: {set(range(1, n + 1)) - covered}"
        )

    U = set(range(1, n + 1))
    V = subsets

    problem = SetCover(U, V)

    if not problem.is_coverable():
        raise ValueError("Cannot construct valid solution from given subsets")

    qubo_dict = problem.to_qubo(A=A, B=B)
    qubo_matrix = qubo_to_matrix(qubo_dict.Q)
    qubo_matrix = (qubo_matrix + qubo_matrix.T) / 2

    return torch.tensor(qubo_matrix, dtype=torch.float32)
