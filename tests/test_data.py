import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from scripts.msc_parser import parse_msc_content
from setcover_gnn.data.generation import generate_set_cover_instance
from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
from setcover_gnn.utils.verification import verify_set_cover


@pytest.mark.parametrize(("n", "num_sets"), [(10, 5), (8, 2), (4, 11)])
def test_generate_set_cover_instance(n, num_sets):
    """Tests instance generation."""
    subsets = generate_set_cover_instance(n, num_sets=num_sets)

    assert len(subsets) == num_sets
    assert all(isinstance(s, set) for s in subsets)

    covered = set()
    for subset in subsets:
        covered.update(subset)

    assert covered == set(range(1, n + 1))


@pytest.mark.parametrize(("n", "subsets"), [(5, [{1, 2}, {2, 3}, {3, 4, 5}, {1, 5}])])
def test_qubo_conversion(n, subsets):
    """Tests QUBO conversion."""
    qubo = set_cover_to_qubo_qubovert(n, subsets, A=2.0, B=1.0)

    assert isinstance(qubo, torch.Tensor)
    assert torch.allclose(qubo, qubo.T)


@pytest.mark.parametrize(
    (
        "n",
        "subsets",
        "valid_solution",
        "count_valid",
        "invalid_solution",
        "count_invalid",
    ),
    [
        (5, [{1, 2}, {3, 4}, {5}], [1, 1, 1], 3, [1, 0, 0], 1),
    ],
)
def test_verification(
    n, subsets, valid_solution, count_valid, invalid_solution, count_invalid
):
    """Tests solution verification."""

    is_valid, count = verify_set_cover(n, subsets, valid_solution)
    assert is_valid
    assert count == count_valid

    is_valid, count = verify_set_cover(n, subsets, invalid_solution)
    assert not is_valid
    assert count == count_invalid


@pytest.mark.parametrize(
    ("content", "valid_n_elements", "valid_subsets"),
    [
        ("c This is a comment\np set 5 3\ns 1 2\ns 3 4\ns 5", 5, [{1, 2}, {3, 4}, {5}]),
    ],
)
def test_msc_parser(content, valid_n_elements, valid_subsets):
    """Tests MSC content parser."""

    n_elements, subsets = parse_msc_content(content)

    assert n_elements == valid_n_elements
    assert len(subsets) == len(valid_subsets)
    assert all(subsets[ind] == valid_subsets[ind] for ind in range(len(subsets)))


@pytest.mark.parametrize(
    ("content",),
    [("p set 5 2\ns 1 2\ns 3 4",)],
)
def test_msc_parser_invalid(content):
    """Tests MSC parser with invalid content."""

    with pytest.raises(ValueError, match="Not all elements covered"):
        parse_msc_content(content)


@pytest.fixture
def temp_dvc_repo():
    """Creates a temporary DVC repository for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    dvc_dir = temp_dir / ".dvc"
    dvc_dir.mkdir()

    data_dir = temp_dir / "data"
    data_dir.mkdir()

    test_dir = data_dir / "test"
    test_dir.mkdir()
    test_file = test_dir / "problem.msc"

    with open(test_file, "w") as f:
        f.write("p set 5 3\n")
        f.write("s 1 2\n")
        f.write("s 3 4\n")
        f.write("s 5\n")

    config_file = dvc_dir / "config"
    with open(config_file, "w") as f:
        f.write(
            "[core]\n\tremote = setcover_gnn_data\n\n"
            '[remote "setcover_gnn_data"]\n\turl = ../data\n'
        )

    yield temp_dir

    shutil.rmtree(temp_dir)
