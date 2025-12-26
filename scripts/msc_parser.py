#!/usr/bin/env python3
import logging
from pathlib import Path

import dvc.api

logger = logging.getLogger(__name__)


def parse_msc_content(content: str) -> tuple[int, list[set[int]]]:
    """Parses content of .msc format file.

    File format:
    p set <n_elements> <n_subsets>
    s <element1> <element2> ... <elementK>

    Args:
        content: Content of .msc file

    Returns:
        Tuple of (n_elements, subsets)

    Raises:
        ValueError: If file format is invalid
    """
    subsets = []
    n_elements = None
    n_subsets = None

    lines = content.strip().split("\n")

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


def load_problem_from_dvc(
    file_path: str,
    repo_path: str = ".",
    remote_name: str = "setcover_gnn_data",
    use_dvc: bool = True,
) -> tuple[int, list[set[int]]]:
    """Loads a problem from file via DVC API.

    Args:
        file_path: Relative path to file in DVC repository
        repo_path: Path to DVC repository root
        remote_name: Name of DVC remote storage
        use_dvc: Whether to use DVC for file reading

    Returns:
        Tuple of (n_elements, subsets)
    """
    if use_dvc:
        try:
            logger.info(f"Reading file via DVC API: {file_path}")

            content = dvc.api.read(
                path=file_path,
                repo=repo_path,
                remote=remote_name,
                mode="r",
                encoding="utf-8",
            )

            return parse_msc_content(content)

        except ImportError:
            logger.error("DVC API not available. Install dvc: pip install dvc")
            raise
        except Exception as e:
            logger.error(f"Error reading file via DVC API: {e}")
            raise

    else:
        logger.warning("Using non-DVC mode (for testing only)")

        full_path = Path(repo_path) / file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        return parse_msc_content(content)
