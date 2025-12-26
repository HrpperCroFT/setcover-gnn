#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

import click


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--host",
    default="127.0.0.1",
    help="Hostname to bind to",
    show_default=True,
)
@click.option(
    "--port",
    default=8080,
    type=int,
    help="Port to listen on",
    show_default=True,
)
@click.option(
    "--backend-store-uri",
    default="sqlite:///mlflow.db",
    help="Backend store URI",
    show_default=True,
)
@click.option(
    "--artifact-root",
    default="./mlflow_artifacts",
    help="Artifact root directory",
    show_default=True,
)
@click.option(
    "--experiment-name",
    default="setcover_experiments",
    help="Default experiment name",
    show_default=True,
)
def start_mlflow_server(
    host: str,
    port: int,
    backend_store_uri: str,
    artifact_root: str,
    experiment_name: str,
) -> None:
    """Starts MLflow tracking server with specified configuration.

    Args:
        host: Hostname to bind to
        port: Port to listen on
        backend_store_uri: Backend store URI
        artifact_root: Artifact root directory
        experiment_name: Default experiment name
    """
    Path(artifact_root).mkdir(parents=True, exist_ok=True)

    if backend_store_uri.startswith("sqlite:///"):
        db_path = backend_store_uri.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    click.echo("Starting MLflow server...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Backend store: {backend_store_uri}")
    click.echo(f"  Artifact root: {artifact_root}")
    click.echo(f"  Experiment: {experiment_name}")
    click.echo(f"\nAccess the UI at: http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the server")

    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        artifact_root,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"MLflow server failed: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nMLflow server stopped")
        sys.exit(0)


if __name__ == "__main__":
    start_mlflow_server()
