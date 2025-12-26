#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import dvc.api
import gdown
import hydra
from dvc.repo import Repo
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def download_from_gdrive(remote_url: str, output_dir: Path) -> None:
    """Downloads a data folder from Google Drive.

    Args:
        remote_url: Google Drive folder URL
        output_dir: Local directory for saving data
    """
    logger.info(f"Downloading data from {remote_url} to {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    folder_id = None
    if "/folders/" in remote_url:
        parts = remote_url.split("/folders/")
        if len(parts) > 1:
            folder_id = parts[1].split("?")[0].split("/")[0]
    elif "id=" in remote_url:
        folder_id = remote_url.split("id=")[1].split("&")[0]

    if folder_id:
        logger.info(f"Google Drive folder ID: {folder_id}")

        gdown.download_folder(
            id=folder_id, output=str(output_dir), quiet=False, use_cookies=True
        )
        logger.info(f"Data successfully downloaded to {output_dir}")

        files = list(output_dir.rglob("*"))
        logger.info(f"Downloaded files: {len(files)}")
    else:
        raise ValueError(f"Could not extract folder ID from URL: {remote_url}")


def init_dvc_repo(repo_path: Path, remote_name: str = "setcover_gnn_data") -> Repo:
    """Initializes a DVC repository and configures remote storage.

    Args:
        repo_path: Path to repository root
        remote_name: Name of remote storage

    Returns:
        Initialized DVC repository
    """
    logger.info(f"Initializing DVC repository in {repo_path}...")

    dvc_dir = repo_path / ".dvc"

    if dvc_dir.exists():
        logger.info("DVC already initialized in this repository")
        repo = Repo(repo_path)
    else:
        repo = Repo.init(repo_path, no_scm=True, force=True)
        logger.info("DVC successfully initialized")

    try:
        with repo.config.edit() as conf:
            data_dir = repo_path / "data"
            conf["remote"][remote_name] = {"url": str(data_dir)}
            conf["core"]["remote"] = remote_name

            logger.info(f"Remote storage configured: {remote_name} -> {data_dir}")
            repo.scm.add([".dvc/config"])

    except Exception as e:
        logger.warning(f"Failed to configure remote storage: {e}")

    return repo


def setup_dvc_storage(repo: Repo, data_dir: Path, file_format: str) -> None:
    """Configures DVC storage and adds files.

    Args:
        repo: DVC repository
        data_dir: Data directory
        file_format: File format to include
    """
    logger.info(f"Setting up DVC storage for {data_dir}...")

    logger.info("Adding files to DVC...")

    added_files = []
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.name.endswith(file_format):
            try:
                rel_path = file_path.relative_to(repo.root_dir)
                repo.add(str(rel_path))
                added_files.append(str(rel_path))
                logger.debug(f"Added file: {rel_path}")
            except ValueError:
                logger.warning(f"File outside repository: {file_path}")
            except Exception as e:
                logger.error(f"Error adding file {file_path}: {e}")

    logger.info(f"Files added to DVC: {len(added_files)}")

    if added_files:
        repo.scm.add(["*.dvc", ".dvcignore", ".dvc/config"])
        logger.info("Changes committed to DVC")

        try:
            repo.push()
            logger.info("Data pushed to remote storage")
        except Exception as e:
            logger.warning(f"Failed to push data to remote storage: {e}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function for data downloading and DVC initialization.

    Args:
        cfg: Hydra configuration
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Data download configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    repo_path = Path(cfg.dvc.repo_path).resolve()
    if not repo_path.exists():
        logger.error(f"Repository path does not exist: {repo_path}")
        sys.exit(1)

    logger.info(f"Repository root: {repo_path}")

    data_dir = repo_path / "data"

    if cfg.dvc.get("remote_url"):
        remote_url = cfg.dvc.remote_url
        logger.info(f"Download URL: {remote_url}")

        try:
            download_from_gdrive(remote_url, data_dir)
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            sys.exit(1)
    else:
        logger.info("remote_url not specified, skipping data download")
        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            sys.exit(1)

    try:
        repo = init_dvc_repo(repo_path, cfg.dvc.get("remote_name", "setcover_gnn_data"))

        if data_dir.exists():
            setup_dvc_storage(repo, data_dir, cfg.data.file.get("format"))

        logger.info("DVC repository information:")
        logger.info(f"Root folder: {repo.root_dir}")
        logger.info(f"Remote storage: {repo.config.get('core', {}).get('remote')}")

        test_file = cfg.data.file.get("path")
        try:
            _ = dvc.api.read(
                path=test_file,
                repo=str(repo_path),
                remote=cfg.dvc.get("remote_name", "setcover_gnn_data"),
            )
            logger.info(f"File accessible via DVC API: {test_file}")
        except Exception as e:
            logger.warning(f"File not accessible via DVC API: {test_file} - {e}")

    except Exception as e:
        logger.error(f"Error initializing DVC: {e}")
        sys.exit(1)

    logger.info("Data download and DVC initialization completed successfully!")
    logger.info(f"Data available in: {data_dir}")
    logger.info("For working with DVC use commands:")
    logger.info(f"  cd {repo_path}")
    logger.info("  dvc status  # Check status")
    logger.info("  dvc push    # Push changes to remote storage")
    logger.info("  dvc pull    # Pull changes from remote storage")


if __name__ == "__main__":
    main()
