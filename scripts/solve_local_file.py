#!/usr/bin/env python3
"""Script to solve Set Cover problem from local file without DVC."""

import json
import logging
from pathlib import Path
from typing import Any

import click
import hydra
from omegaconf import OmegaConf

from setcover_gnn.core import SetCoverProblem, SetCoverSolver
from setcover_gnn.data.qubo_conversion import set_cover_to_qubo_qubovert
from setcover_gnn.utils.graph_utils import create_dgl_graph_from_qubo

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_msc_file(file_path: Path) -> tuple[int, list[set[int]]]:
    """Parses .msc file from local filesystem.

    Args:
        file_path: Path to .msc file

    Returns:
        Tuple of (n_elements, subsets)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.strip().split("\n")
    subsets = []
    n_elements = None
    n_subsets = None

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("c"):
            continue

        parts = line.split()

        if parts[0] == "p" and parts[1] == "set":
            if len(parts) != 4:
                raise ValueError(
                    f"Line {line_num}: Invalid parameter line format: {line}"
                )
            try:
                n_elements = int(parts[2])
                n_subsets = int(parts[3])
                logger.debug(f"Parameters: {n_elements} elements, {n_subsets} subsets")
            except ValueError:
                raise ValueError(f"Line {line_num}: Invalid numeric values: {line}")

        elif parts[0] == "s":
            try:
                subset_elements = {int(x) for x in parts[1:]}
                subsets.append(subset_elements)
                logger.debug(f"Subset {len(subsets)}: {len(subset_elements)} elements")
            except ValueError:
                raise ValueError(f"Line {line_num}: Invalid subset format: {line}")

        else:
            raise ValueError(f"Line {line_num}: Unknown line type: {line}")

    if n_elements is None or n_subsets is None:
        raise ValueError("File does not contain parameter line (p set ...)")

    if len(subsets) != n_subsets:
        raise ValueError(
            f"Subset count mismatch: expected {n_subsets}, got {len(subsets)}"
        )

    all_elements = set()
    for i, subset in enumerate(subsets, 1):
        for elem in subset:
            if not 1 <= elem <= n_elements:
                raise ValueError(
                    f"Subset {i}: element {elem} outside range 1..{n_elements}"
                )
        all_elements.update(subset)

    if all_elements != set(range(1, n_elements + 1)):
        missing = set(range(1, n_elements + 1)) - all_elements
        raise ValueError(f"Not all elements covered: missing {missing}")

    logger.info(f"Successfully parsed: {n_elements} elements, {n_subsets} subsets")
    return n_elements, subsets


def solve_problem_from_local_file(
    input_file: str,
    method: str = "gnn",
    config_path: str = "../configs",
    config_name: str = "config",
    **solver_kwargs,
) -> dict[str, Any]:
    """Solves Set Cover problem from local file.

    Args:
        input_file: Path to local .msc file
        method: Solution method ('gnn' or 'greedy')
        config_path: Path to Hydra configs
        config_name: Config name
        **solver_kwargs: Additional solver parameters

    Returns:
        Dictionary with results
    """
    with hydra.initialize(config_path=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)

    logger.info(f"Loading problem from local file: {input_file}")
    n_elements, subsets = parse_msc_file(Path(input_file))

    logger.info("Creating QUBO matrix...")
    qubo_matrix = set_cover_to_qubo_qubovert(
        n_elements, subsets, A=cfg.model.qubo.A, B=cfg.model.qubo.B
    )

    logger.info("Creating graph...")
    graph = create_dgl_graph_from_qubo(qubo_matrix)

    problem = SetCoverProblem(
        n_elements=n_elements, subsets=subsets, qubo_matrix=qubo_matrix, graph=graph
    )

    logger.info(f"Solving problem with {method} method...")
    solver = SetCoverSolver(device=cfg.get("device", "auto"), seed=cfg.get("seed"))

    if method == "gnn":
        _, solution, metrics = solver.solve_gnn(
            problem,
            cfg=cfg,
            logger=None,
            dim_embedding=cfg.model.gnn.dim_embedding,
            hidden_dim=cfg.model.gnn.hidden_dim,
            dropout=cfg.model.gnn.dropout,
            learning_rate=cfg.model.gnn.learning_rate,
            prob_threshold=cfg.model.gnn.prob_threshold,
            max_epochs=cfg.training.max_epochs,
            patience=cfg.training.patience,
            tolerance=cfg.training.tolerance,
            penalty_rate=cfg.training.penalty_rate,
            **solver_kwargs,
        )
    else:
        solution, metrics = solver.solve(problem, method=method)

    results = {
        "input_file": input_file,
        "problem": {
            "n_elements": n_elements,
            "n_subsets": len(subsets),
            "file_size_bytes": Path(input_file).stat().st_size,
        },
        "method": method,
        "solution": solution,
        "metrics": metrics,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

    return results


@click.command()
@click.option(
    "--input-file",
    required=True,
    help="Path to local .msc file",
)
@click.option(
    "--output-file",
    required=True,
    help="Path to save results (JSON)",
)
@click.option(
    "--method",
    default="gnn",
    type=click.Choice(["gnn", "greedy"]),
    help="Solution method",
)
@click.option(
    "--config-path",
    default="../configs",
    help="Path to Hydra configuration files",
    show_default=True,
)
@click.option(
    "--config-name",
    default="config",
    help="Configuration file name",
    show_default=True,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    input_file: str,
    output_file: str,
    method: str,
    config_path: str,
    config_name: str,
    verbose: bool,
) -> None:
    """Solves Set Cover problem from local file and saves result.

    Example usage:
        python scripts/solve_local_file.py --input-file data/problem.msc
            --output-file results.json
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    input_path = Path(input_file)
    if not input_path.exists():
        click.echo(f"Error: Input file not found: {input_file}", err=True)
        return
    if input_path.suffix != ".msc":
        click.echo(f"Warning: File extension is not .msc: {input_file}", err=True)

    click.echo(f"Solving problem from local file: {input_file}")
    click.echo(f"Method: {method}")
    click.echo(f"Config: {config_name}")

    try:
        results = solve_problem_from_local_file(
            input_file=input_file,
            method=method,
            config_path=config_path,
            config_name=config_name,
        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        click.echo(f"\nResults saved to: {output_path}")
        click.echo(
            f"Problem size: {results['problem']['n_elements']} elements, "
            f"{results['problem']['n_subsets']} subsets"
        )
        click.echo(f"Selected subsets: {results['metrics']['selected_count']}")
        click.echo(f"Solution valid: {results['metrics']['is_valid']}")

        click.echo("\n" + "=" * 60)
        click.echo("SOLUTION SUMMARY:")
        click.echo(f"Method: {results['method'].upper()}")
        click.echo(f"Input file: {results['input_file']}")
        click.echo(
            f"Problem: {results['problem']['n_elements']} elements, "
            f"{results['problem']['n_subsets']} subsets"
        )
        click.echo(f"Selected subsets: {results['metrics']['selected_count']}")
        click.echo(
            f"Solution valid: {'YES' if results['metrics']['is_valid'] else 'NO'}"
        )

        if method == "gnn":
            click.echo("GNN Parameters:")
            click.echo(
                f"  Hidden dim: {results['config']['model']['gnn']['hidden_dim']}"
            )
            click.echo(
                f"  Embedding dim: {results['config']['model']['gnn']['dim_embedding']}"
            )
            click.echo(
                f"  Learning rate: {results['config']['model']['gnn']['learning_rate']}"
            )
            click.echo(f"  Max epochs: {results['config']['training']['max_epochs']}")
        click.echo("=" * 60)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        return
    except ValueError as e:
        click.echo(f"Error parsing file: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"Error solving problem: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        return


if __name__ == "__main__":
    main()
