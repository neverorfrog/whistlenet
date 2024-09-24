import os

import lightning as L
from aim.pytorch_lightning import AimLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from config import TrainerConfig


class LightningTrainer(L.Trainer):  # type: ignore[misc]
    def __init__(self, config: TrainerConfig):

        self.config = config

        aim_logger = AimLogger(
            experiment=config.experiment,
            train_metric_prefix="train_",
            test_metric_prefix="test_",
            val_metric_prefix="val_",
        )

        self._checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=config.ckpt_path,
            save_top_k=1,
            mode="min",
            verbose=True,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            min_delta=config.min_delta,
            verbose=True,
        )

        super().__init__(
            max_epochs=config.epochs,
            callbacks=[self._checkpoint_callback, early_stopping_callback],
            logger=aim_logger,
        )

    def fit(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
        checkpoint_path = f"{self.config.ckpt_path}/last.ckpt"

        if self.config.resume_training and os.path.exists(checkpoint_path):
            ckpt_path = checkpoint_path
        else:
            ckpt_path = None

        super().fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            ckpt_path=ckpt_path,
        )

    def test(
        self, model: L.LightningModule, datamodule: L.LightningDataModule
    ) -> None:
        """Test the model on the test set."""
        best_ckpt_path = self._checkpoint_callback.best_model_path

        if not best_ckpt_path:
            raise ValueError(
                "Best model path not found. \
                Make sure the model was trained and saved."
            )

        super().test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=best_ckpt_path,
        )
