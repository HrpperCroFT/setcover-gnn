import pytest
import torch
from setcover_gnn.data.generation import generate_set_cover_instance
from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
from setcover_gnn.utils.verification import verify_set_cover

@pytest.mark.parametrize(("n", "num_sets"), [(10, 5), (8, 2), (4, 11)])
def test_generate_set_cover_instance(n, num_sets):
    """Test instance generation."""
    subsets = generate_set_cover_instance(n, num_sets=num_sets)
    
    assert len(subsets) == num_sets
    assert all(isinstance(s, set) for s in subsets)

    covered = set()
    for subset in subsets:
        covered.update(subset)
    
    assert covered == set(range(1, n + 1))

@pytest.mark.parametrize(("n", "subsets"), [
    (5, [{1, 2}, {2, 3}, {3, 4, 5}, {1, 5}])
])
def test_qubo_conversion(n, subsets):
    """Test QUBO conversion."""
    qubo = set_cover_to_qubo_qubovert(n, subsets, A=2.0, B=1.0)
    
    assert isinstance(qubo, torch.Tensor)
    assert qubo.shape == (len(subsets), len(subsets))
    assert torch.allclose(qubo, qubo.T)

@pytest.mark.parametrize(
        ("n", "subsets", "valid_solution", "count_valid", "invalid_solution", "count_invalid"),
        [
            (5, [{1, 2}, {3, 4}, {5}], [1, 1, 1], 3, [1, 0, 0], 1),
            ])
def test_verification(n, subsets, valid_solution, count_valid, invalid_solution, count_invalid):
    """Test solution verification."""

    is_valid, count = verify_set_cover(n, subsets, valid_solution)
    assert is_valid
    assert count == count_valid
    
    is_valid, count = verify_set_cover(n, subsets, invalid_solution)
    assert not is_valid
    assert count == count_invalid