"""
Cashew Pest and Disease Diagnosis System
Phase 3: Modular Training Engine (TensorFlow / Keras)
Two-Stage Fine-Tuning, Optimizer Selector, Callbacks, Mixed Precision & Artifact Saving
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.utils import set_seed, get_logger, get_optimal_batch_size
from src.dataset import create_reproducible_splits, build_tf_data_pipelines
from src.models import build_keras_model, unfreeze_model_backbone
from src.loss import get_loss_function

logger = get_logger("TrainingEngine")


# ---------------------------------------------------------
# Optimizer Factory
# ---------------------------------------------------------
def get_optimizer(optimizer_name: str = "adam", learning_rate: float = 1e-4) -> keras.optimizers.Optimizer:
    """
    Constructs requested Keras optimizer.
    Supported options: 'adam', 'adamw', 'sgd'
    """
    name_lower = optimizer_name.lower()
    if name_lower == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif name_lower == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif name_lower == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer name: '{optimizer_name}'. Choices: ['adam', 'adamw', 'sgd']")


# ---------------------------------------------------------
# Training History & Curve Visualization Helpers
# ---------------------------------------------------------
def plot_and_save_training_curves(history_df: pd.DataFrame, experiment_dir: str):
    """
    Generates and saves Training & Validation Accuracy, Loss, and Learning Rate curves.
    """
    # 1. Loss & Accuracy Curves Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(history_df['epoch'], history_df['loss'], label='Train Loss', color='#2b5c8f', linewidth=2)
    if 'val_loss' in history_df.columns:
        axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', color='#e07a5f', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    acc_col = 'accuracy' if 'accuracy' in history_df.columns else 'categorical_accuracy'
    val_acc_col = 'val_accuracy' if 'val_accuracy' in history_df.columns else 'val_categorical_accuracy'

    if acc_col in history_df.columns:
        axes[1].plot(history_df['epoch'], history_df[acc_col], label='Train Accuracy', color='#2b5c8f', linewidth=2)
    if val_acc_col in history_df.columns:
        axes[1].plot(history_df['epoch'], history_df[val_acc_col], label='Val Accuracy', color='#81b29a', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    curves_path = os.path.join(experiment_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=300)
    plt.close()
    logger.info(f"[ARTIFACT SAVED] Training curves saved to: {curves_path}")

    # 2. Learning Rate Progression Plot
    if 'lr' in history_df.columns:
        plt.figure(figsize=(8, 4))
        plt.plot(history_df['epoch'], history_df['lr'], label='Learning Rate', color='#3d405b', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.title('Learning Rate Schedule Progression', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        lr_path = os.path.join(experiment_dir, "learning_rate_curve.png")
        plt.savefig(lr_path, dpi=300)
        plt.close()
        logger.info(f"[ARTIFACT SAVED] Learning rate plot saved to: {lr_path}")


# ---------------------------------------------------------
# Main Training Engine Function
# ---------------------------------------------------------
def train_model(
    model_index: int = 1,
    epochs: int = Config.EPOCHS,
    warmup_epochs: int = Config.WARMUP_EPOCHS,
    lr: float = Config.LEARNING_RATE,
    fine_tune_lr: float = Config.FINE_TUNE_LEARNING_RATE,
    optimizer_name: str = Config.OPTIMIZER,
    loss_name: str = "categorical_crossentropy",
    patience: int = Config.PATIENCE
):
    """
    Executes the complete Phase 3 TensorFlow/Keras Training Pipeline:
      1. Hardware check & Mixed Precision activation (if GPU is present).
      2. Loads Phase 2 tf.data pipelines and class weights.
      3. Builds Keras model architecture with frozen backbone (Stage 1 Warmup).
      4. Stage 1 Warmup Training: Trains top classification head.
      5. Stage 2 Fine-Tuning: Unfreezes backbone and trains with reduced learning rate.
      6. Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint (.keras), CSVLogger, TensorBoard.
      7. Exports all artifacts, plots, and metrics to Google Drive Experiments/<Model_Name>/.
    """
    # 1. System Setup & GPU Mixed Precision Activation
    set_seed(Config.SEED)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"[HARDWARE DETECTED] GPU Available: {[gpu.name for gpu in gpus]}")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logger.info("[MIXED PRECISION] Global policy set to 'mixed_float16'")
    else:
        logger.info("[HARDWARE DETECTED] CPU Mode active.")

    # Resolve Experiment Output Folder in Google Drive
    experiment_dir = Config.get_experiment_dir(model_index)
    folder_name, model_key = Config.MODEL_MAP[model_index]
    logger.info(f"=== Starting Training Experiment for Model #{model_index}: {folder_name} ===")
    logger.info(f"Experiment Artifact Directory: {experiment_dir}")

    # 2. Load Phase 2 Dataset & TensorFlow Data Pipelines
    split_info = create_reproducible_splits(seed=Config.SEED)
    batch_size = get_optimal_batch_size()
    tf_pipelines = build_tf_data_pipelines(split_info, batch_size=batch_size)

    train_ds = tf_pipelines["train"]
    val_ds = tf_pipelines["val"]
    
    num_classes = len(split_info["class_names"])
    
    # Format class weights dictionary for Keras fit()
    class_weights_dict = {i: float(w) for i, w in enumerate(split_info["class_weights"])}
    logger.info(f"Loaded Class Weights for Training: {class_weights_dict}")

    # 3. Define Callbacks & Save Paths
    best_model_path = os.path.join(experiment_dir, "best_model.keras")
    history_csv_path = os.path.join(experiment_dir, "history.csv")
    tensorboard_dir = os.path.join(experiment_dir, "logs")

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=False
        ),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=1
        )
    ]

    # 4. Stage 1: Warmup Training (Frozen Backbone)
    logger.info(f"\n--- STAGE 1: WARMUP TRAINING ({warmup_epochs} Epochs, Frozen Backbone) ---")
    model = build_keras_model(
        model_index=model_index,
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        trainable_backbone=False
    )

    optimizer = get_optimizer(optimizer_name, learning_rate=lr)
    loss_fn = get_loss_function(loss_name)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["categorical_accuracy"]
    )
    
    model.summary(print_fn=logger.info)

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # 5. Stage 2: Fine-Tuning (Unfrozen Backbone)
    remaining_epochs = max(0, epochs - warmup_epochs)
    if remaining_epochs > 0:
        logger.info(f"\n--- STAGE 2: FINE-TUNING ({remaining_epochs} Remaining Epochs, Unfrozen Backbone) ---")
        model = unfreeze_model_backbone(model)

        fine_tune_optimizer = get_optimizer(optimizer_name, learning_rate=fine_tune_lr)
        model.compile(
            optimizer=fine_tune_optimizer,
            loss=loss_fn,
            metrics=["categorical_accuracy"]
        )

        # Update CSVLogger callback to append history for Stage 2
        callbacks_list[3] = keras.callbacks.CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=True
        )

        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=epochs,
            class_weight=class_weights_dict,
            callbacks=callbacks_list,
            verbose=1
        )

    logger.info("[TRAINING COMPLETE] Best model checkpoint saved to: " + best_model_path)

    # 6. Export Training Curves & Summary Report
    if os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)
        plot_and_save_training_curves(history_df, experiment_dir)

        best_val_acc = float(history_df['val_categorical_accuracy'].max()) if 'val_categorical_accuracy' in history_df.columns else float(history_df['val_accuracy'].max())
        min_val_loss = float(history_df['val_loss'].min())

        summary = {
            "model_index": model_index,
            "model_name": folder_name,
            "model_key": model_key,
            "total_epochs_trained": len(history_df),
            "best_validation_accuracy": best_val_acc,
            "minimum_validation_loss": min_val_loss,
            "optimizer": optimizer_name,
            "initial_learning_rate": lr,
            "fine_tune_learning_rate": fine_tune_lr,
            "best_model_path": best_model_path,
            "history_csv_path": history_csv_path
        }

        summary_json_path = os.path.join(experiment_dir, "experiment_summary.json")
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"[SUMMARY SAVED] Experiment summary saved to: {summary_json_path}")
