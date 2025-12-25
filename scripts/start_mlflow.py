#!/usr/bin/env python3
"""Start MLflow tracking server (CLI with Click)."""

import click
import subprocess
import sys
from pathlib import Path

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
    """
    Start MLflow tracking server with the specified configuration.
    """
    # Check if mlflow is installed
    try:
        import mlflow
    except ImportError:
        click.echo("Error: MLflow is not installed.", err=True)
        click.echo("Install it with: pip install mlflow", err=True)
        sys.exit(1)
    
    # Create directories
    Path(artifact_root).mkdir(parents=True, exist_ok=True)
    
    # Extract directory from SQLite URI
    if backend_store_uri.startswith("sqlite:///"):
        db_path = backend_store_uri.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Starting MLflow server...")
    click.echo(f"  Host: {host}")
    click.echo(f"  Port: {port}")
    click.echo(f"  Backend store: {backend_store_uri}")
    click.echo(f"  Artifact root: {artifact_root}")
    click.echo(f"  Experiment: {experiment_name}")
    click.echo(f"\nAccess the UI at: http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the server")
    
    # Build command
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", artifact_root,
        "--gunicorn-opts", "--timeout 120"
    ]
    
    try:
        # Start server
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"MLflow server failed: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\nMLflow server stopped")
        sys.exit(0)

if __name__ == "__main__":
    start_mlflow_server()