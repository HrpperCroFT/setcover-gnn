import lightning as pl
from lightning.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional


def create_trainer(
    max_epochs: int = 60000,
    patience: int = 100,
    tolerance: float = 1e-4,
    checkpoint_dir: Optional[str] = None,
    **trainer_kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with appropriate callbacks.
    
    Args:
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        tolerance: Loss change tolerance for early stopping
        checkpoint_dir: Directory to save checkpoints
        **trainer_kwargs: Additional arguments for Trainer
        
    Returns:
        Configured PyTorch Lightning Trainer
    """
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='train_loss',
        patience=patience,
        min_delta=tolerance,
        mode='min',
        check_on_train_epoch_end=True
    )
    callbacks.append(early_stopping)
    
    # Checkpoint callback
    if checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=100,
        **trainer_kwargs
    )
    
    return trainer


def train_setcover_gnn(
    model: pl.LightningModule,
    max_epochs: int = 60000,
    patience: int = 100,
    tolerance: float = 1e-4,
    **trainer_kwargs
) -> pl.LightningModule:
    """
    Train a SetCoverGNN model.
    
    Args:
        model: SetCoverGNN model to train
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        tolerance: Loss tolerance for early stopping
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Trained model
    """
    trainer = create_trainer(
        max_epochs=max_epochs,
        patience=patience,
        tolerance=tolerance,
        **trainer_kwargs
    )
    
    trainer.fit(model)
    return model