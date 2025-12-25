#!/usr/bin/env python3
"""Setup MLflow experiment for testing (CLI with Click)."""

import click
import mlflow

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--tracking-uri",
    default="http://127.0.0.1:8080",
    help="MLflow tracking URI",
    show_default=True,
)
@click.option(
    "--experiment-name",
    default="setcover_experiments",
    help="Experiment name",
    show_default=True,
)
def setup_experiment(tracking_uri: str, experiment_name: str) -> None:
    """
    Setup MLflow experiment with the given tracking URI and experiment name.
    Creates the experiment if it doesn't exist.
    """
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    click.echo(f"Tracking URI set to: {tracking_uri}")
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        click.echo(f"Creating experiment: {experiment_name}")
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location="./mlflow_artifacts"
        )
    else:
        experiment_id = experiment.experiment_id
        click.echo(f"Experiment '{experiment_name}' already exists (ID: {experiment_id})")
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name)
    
    click.echo(f"\nExperiment setup complete.")
    click.echo(f"Name: {experiment_name}")
    click.echo(f"ID: {experiment_id}")
    click.echo(f"\nStart MLflow server with:")
    click.echo(f"  ./scripts/start_mlflow.sh")
    click.echo(f"\nOr with Python:")
    click.echo(f"  python scripts/start_mlflow.py")

if __name__ == "__main__":
    setup_experiment()