import logging
from typing import Optional

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf

file_logger = logging.getLogger(__name__)


def create_trainer(
    cfg,
    logger=None,
    max_epochs: int = 60000,
    patience: int | None = 100,
    tolerance: float | None = 1e-4,
    checkpoint_dir: Optional[str] = None,
    **trainer_kwargs,
) -> pl.Trainer:
    """Creates a PyTorch Lightning trainer with MLflow logging.

    Args:
        cfg: Configuration dictionary or OmegaConf object
        logger: Optional logger instance. If None, will
                create MLFlowLogger if enabled in cfg.
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        tolerance: Loss change tolerance for early stopping
        checkpoint_dir: Directory to save checkpoints
        **trainer_kwargs: Additional arguments for Trainer

    Returns:
        Configured PyTorch Lightning Trainer
    """
    callbacks = []

    if patience is not None and tolerance is not None:
        early_stopping = EarlyStopping(
            monitor="train_loss",
            patience=patience,
            min_delta=tolerance,
            mode="min",
            check_on_train_epoch_end=True,
        )
        callbacks.append(early_stopping)

    if checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="train_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    if logger is None and cfg.logging.enabled and cfg.logging.backend == "mlflow":
        try:
            mlflow_config = OmegaConf.to_container(cfg.logging.mlflow, resolve=True)
            tags = mlflow_config.get("tags", {})

            if not isinstance(tags, dict):
                tags = {}

            logger = MLFlowLogger(
                experiment_name=str(mlflow_config["experiment_name"]),
                run_name=str(mlflow_config["run_name"]),
                tracking_uri=str(mlflow_config["tracking_uri"]),
                tags=tags,
            )
            file_logger.info(
                "MLflow logger initialized for experiment: "
                f"{mlflow_config['experiment_name']}"
            )

        except Exception as e:
            file_logger.warning(f"Failed to initialize MLflow logger: {e}")
            logger = None

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=100,
        logger=logger,
        **trainer_kwargs,
    )

    return trainer


def train_setcover_gnn(
    model: pl.LightningModule,
    cfg,
    logger=None,
    max_epochs: int = 60000,
    patience: int | None = 100,
    tolerance: float | None = 1e-4,
    **trainer_kwargs,
) -> pl.LightningModule:
    """Trains a SetCoverGNN model with MLflow logging.

    Args:
        model: SetCoverGNN model to train
        cfg: Configuration dictionary
        logger: Optional logger instance
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        tolerance: Loss tolerance for early stopping
        **trainer_kwargs: Additional trainer arguments

    Returns:
        Trained model
    """
    trainer = create_trainer(
        cfg=cfg,
        logger=logger,
        max_epochs=max_epochs,
        patience=patience,
        tolerance=tolerance,
        **trainer_kwargs,
    )

    trainer.fit(model)
    return model
