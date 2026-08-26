"""
Cashew Pest and Disease Diagnosis System
Phase C.2: Segmentation Model Trainer
Framework: TensorFlow / Keras
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)

from .training_config import SegmentationTrainingConfig
from .unet_model import build_unet, dice_coef, iou_metric, bce_dice_loss


class SegmentationTrainer:
    """Manages compilation, callback setup, and training execution of the U-Net."""

    def __init__(
        self,
        config: SegmentationTrainingConfig,
        experiment_dir: Path,
    ):
        self.config = config
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[tf.keras.Model] = None

    def build_and_compile_model(self) -> tf.keras.Model:
        """Constructs and compiles the U-Net with combined BCE-Dice loss and segmentation metrics."""
        self.model = build_unet(
            input_shape=self.config.input_shape,
            num_classes=self.config.num_classes,
            encoder_filters=self.config.encoder_filters,
            bottleneck_filters=self.config.bottleneck_filters,
            dropout_rate=self.config.dropout_rate,
        )

        loss_fn = bce_dice_loss(
            bce_weight=self.config.bce_weight,
            dice_weight=self.config.dice_weight,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                dice_coef,
                iou_metric,
                tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            ],
        )
        return self.model

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        train_steps: Optional[int] = None,
        val_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes U-Net training with production callbacks and history logging."""
        if self.model is None:
            self.build_and_compile_model()

        best_model_path = self.experiment_dir / "best_unet_model.keras"
        csv_log_path = self.experiment_dir / "training_history.csv"

        has_validation = (val_ds is not None and val_steps != 0)

        callbacks = [
            CSVLogger(str(csv_log_path)),
            ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_iou_metric" if has_validation else "iou_metric",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
        ]

        if has_validation:
            callbacks.extend([
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience,
                    min_lr=self.config.min_lr,
                    verbose=1,
                ),
            ])

        print(f"\n[TRAINER] Starting U-Net Training for up to {self.config.num_epochs} epochs...")
        print(f"  - Checkpoint Path : {best_model_path}")
        print(f"  - Has Validation  : {has_validation}")

        history = self.model.fit(
            train_ds,
            validation_data=val_ds if has_validation else None,
            epochs=self.config.num_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps if has_validation else None,
            callbacks=callbacks,
            verbose=1,
        )

        # Save final model
        final_model_path = self.experiment_dir / "final_unet_model.keras"
        self.model.save(str(final_model_path))
        print(f"[TRAINER] Final model saved to: {final_model_path}")

        # Convert history to serializable dict
        history_dict = {}
        for k, v in history.history.items():
            history_dict[k] = [float(val) for val in v]

        with open(self.experiment_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=4)

        return history_dict
