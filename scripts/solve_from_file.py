#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any, Optional

import click
import hydra
from omegaconf import OmegaConf

from scripts.msc_parser import load_problem_from_dvc
from setcover_gnn.core import SetCoverProblem, SetCoverSolver
from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
from setcover_gnn.utils.graph_utils import create_dgl_graph_from_qubo


def solve_problem_from_file(
    input_file: str,
    repo_path: str,
    remote_name: str,
    use_dvc: bool,
    method: str = "gnn",
    config_path: str = "../configs",
    config_name: str = "config",
    **solver_kwargs,
) -> dict[str, Any]:
    """Solves Set Cover problem from file.

    Args:
        input_file: Path to .msc file in DVC repository
        repo_path: Path to DVC repository root
        remote_name: Name of DVC remote storage
        use_dvc: Whether to use DVC for file reading
        method: Solution method ('gnn' or 'greedy')
        config_path: Path to Hydra configs
        config_name: Config name
        **solver_kwargs: Additional solver parameters

    Returns:
        Dictionary with results
    """
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)

    click.echo(f"Loading problem from {input_file} via DVC...")
    n_elements, subsets = load_problem_from_dvc(
        file_path=input_file,
        repo_path=repo_path,
        remote_name=remote_name,
        use_dvc=use_dvc,
    )

    click.echo("Creating QUBO matrix...")
    qubo_matrix = set_cover_to_qubo_qubovert(
        n_elements, subsets, A=cfg.model.qubo.A, B=cfg.model.qubo.B
    )

    click.echo("Creating graph...")
    graph = create_dgl_graph_from_qubo(qubo_matrix)

    problem = SetCoverProblem(
        n_elements=n_elements, subsets=subsets, qubo_matrix=qubo_matrix, graph=graph
    )

    click.echo(f"Solving problem with {method} method...")
    solver = SetCoverSolver(device=cfg.get("device", "auto"), seed=cfg.get("seed"))

    if method == "gnn":
        _, solution, metrics = solver.solve_gnn(
            problem,
            dim_embedding=cfg.model.gnn.dim_embedding,
            hidden_dim=cfg.model.gnn.hidden_dim,
            dropout=cfg.model.gnn.dropout,
            learning_rate=cfg.model.gnn.learning_rate,
            prob_threshold=cfg.model.gnn.prob_threshold,
            max_epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
            tolerance=cfg.training.tolerance,
            **solver_kwargs,
        )
    else:
        solution, metrics = solver.solve(problem, method=method)

    results = {
        "input_file": input_file,
        "problem": {"n_elements": n_elements, "n_subsets": len(subsets)},
        "method": method,
        "solution": solution,
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    return results


@click.command()
@click.option(
    "--input-file",
    help="Path to .msc file in DVC repository. If not specified, taken from config.",
)
@click.option("--output-file", required=True, help="Path to save results (JSON)")
@click.option(
    "--method",
    default="gnn",
    type=click.Choice(["gnn", "greedy"]),
    help="Solution method",
)
@click.option(
    "--repo-path",
    help="Path to DVC repository root. If not specified, taken from config.",
)
@click.option(
    "--config-path", default="../configs", help="Path to Hydra configuration files"
)
@click.option("--config-name", default="config", help="Configuration file name")
def main(
    input_file: Optional[str],
    output_file: str,
    method: str,
    repo_path: Optional[str],
    config_path: str,
    config_name: str,
):
    """Solves Set Cover problem from file and saves result.

    Example usage:
        python scripts/solve_from_file.py --output-file results.json
    """
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)

    if repo_path is None:
        repo_path = cfg.dvc.get("repo_path", ".")

    if input_file is None:
        input_file = cfg.get("data_file", "frb30-15-msc/frb30-15-1.msc")

    use_dvc = cfg.dvc.get("use_dvc", True)
    remote_name = cfg.dvc.get("remote_name", "setcover_gnn_data")

    click.echo(f"Solving problem from file: {input_file}")
    click.echo(f"Method: {method}")
    click.echo(f"Repository path: {repo_path}")
    click.echo(f"Remote storage: {remote_name}")
    click.echo(f"Use DVC: {use_dvc}")
    click.echo(f"Config: {config_name}")

    results = solve_problem_from_file(
        input_file=input_file,
        repo_path=repo_path,
        remote_name=remote_name,
        use_dvc=use_dvc,
        method=method,
        config_path=config_path,
        config_name=config_name,
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    click.echo(f"Results saved to {output_path}")
    click.echo(f"Selected subsets count: {results['metrics']['selected_count']}")
    click.echo(f"Solution valid: {results['metrics']['is_valid']}")

    click.echo("\n" + "=" * 50)
    click.echo("BRIEF RESULTS:")
    click.echo(f"Method: {results['method']}")
    click.echo(f"Elements: {results['problem']['n_elements']}")
    click.echo(f"Subsets: {results['problem']['n_subsets']}")
    click.echo(f"Selected subsets: {results['metrics']['selected_count']}")
    click.echo(f"Solution valid: {'YES' if results['metrics']['is_valid'] else 'NO'}")
    click.echo("=" * 50)


if __name__ == "__main__":
    main()
