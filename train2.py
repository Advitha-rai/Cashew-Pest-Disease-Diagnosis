"""
Cashew Pest and Disease Diagnosis System
Standalone Isolated Training Engine for Model #7 (MobileNetV3Large ONLY)

File: train2.py
Purpose: Complete isolation of Model #7 training to prevent any runtime module drift
         or full-backbone unfreezing.
"""

import os
import sys
import json
import inspect
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.utils import set_seed, get_logger, get_optimal_batch_size
from src.dataset import create_reproducible_splits, build_tf_data_pipelines
import src.models
from src.models import build_keras_model
from src.loss import get_loss_function

logger = get_logger("IsolatedModel7Engine")


# ---------------------------------------------------------
# Helper: Optimizer Factory
# ---------------------------------------------------------
def get_optimizer(optimizer_name: str = "adam", learning_rate: float = 1e-4) -> keras.optimizers.Optimizer:
    name_lower = optimizer_name.lower()
    if name_lower == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    elif name_lower == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif name_lower == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


# ---------------------------------------------------------
# Helper: Curves Generator
# ---------------------------------------------------------
def plot_and_save_training_curves(history_df: pd.DataFrame, experiment_dir: str):
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


# ---------------------------------------------------------
# Main Isolated Model #7 Training Routine
# ---------------------------------------------------------
def run_isolated_training(
    epochs: int = Config.EPOCHS,
    warmup_epochs: int = Config.WARMUP_EPOCHS,
    lr: float = Config.LEARNING_RATE,
    fine_tune_lr: float = Config.FINE_TUNE_LEARNING_RATE,
    optimizer_name: str = Config.OPTIMIZER,
    loss_name: str = "categorical_crossentropy",
    patience: int = Config.PATIENCE
):
    # SECTION 3: MODEL CONFIGURATION
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]

    if model_key != "mobilenet_v3_large":
        raise RuntimeError(f"CRITICAL ERROR: train2.py is hard-coded for Model #7 (mobilenet_v3_large). Got '{model_key}'")

    print("\n============================================================")
    print("[ISOLATED MODEL #7 TRAINING]")
    print("============================")
    print(f"Model index: {model_index}")
    print(f"Model name: {folder_name}")
    print(f"Model key: {model_key}")
    print("============================================================\n")

    # SECTION 4: FRESH PROCESS / RUNTIME DIAGNOSTICS
    script_file = os.path.abspath(__file__)
    models_file = os.path.abspath(src.models.__file__)
    config_file = os.path.abspath(src.config.__file__)

    print(f"__file__:                 {__file__}")
    print(f"os.path.abspath(__file__): {script_file}")
    print(f"src.models.__file__:       {models_file}")
    print(f"src.config.__file__:       {config_file}")
    print("[ISOLATED RUNTIME] train2.py active")
    isolated_engine_module = "src" + "." + "train"
    print(f"[ISOLATED RUNTIME] {isolated_engine_module} is NOT imported\n")

    if isolated_engine_module in sys.modules:
        raise RuntimeError(f"CRITICAL ERROR: {isolated_engine_module} was imported! train2.py must run independently without {isolated_engine_module}.")

    # SECTION 12: FORBIDDEN MESSAGE & PATTERN CHECK INSIDE TRAIN2.PY SOURCE
    with open(script_file, "r", encoding="utf-8") as f:
        self_source = f.read()

    forbidden_target_unfreeze = "unfreeze" + "_" + "model" + "_" + "backbone" + "("
    forbidden_target_base = "base" + "_" + "model" + ".trainable" + " = " + "True"
    forbidden_target_backbone = "back" + "bone" + ".trainable" + " = " + "True"
    forbidden_patterns = [
        "[FINE-TUNING] Unfroze all backbone layers for full fine-tuning.",
        forbidden_target_base,
        forbidden_target_backbone,
        forbidden_target_unfreeze
    ]

    for pattern in forbidden_patterns:
        if pattern in self_source:
            raise RuntimeError(f"CRITICAL FORBIDDEN PATTERN CHECK FAILED: Found forbidden string '{pattern}' inside train2.py!")

    # SECTION 1: EXPERIMENT DIRECTORY ISOLATION
    base_dir = Config.get_base_dir()
    experiment_dir = os.path.join(base_dir, "Experiments", "07_MobileNetV3Large_Isolated")
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Isolated Experiment Directory: {experiment_dir}")

    # SECTION 5: DATASET PIPELINE SETUP
    set_seed(Config.SEED)
    split_info = create_reproducible_splits(seed=Config.SEED)
    batch_size = get_optimal_batch_size()
    tf_pipelines = build_tf_data_pipelines(split_info, batch_size=batch_size)

    train_ds = tf_pipelines["train"]
    val_ds = tf_pipelines["val"]
    num_classes = len(split_info["class_names"])
    class_weights_dict = {i: float(w) for i, w in enumerate(split_info["class_weights"])}

    logger.info(f"Loaded Class Weights for Isolated Training: {class_weights_dict}")

    # Callbacks & Artifact paths
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

    # SECTION 6: BUILD MODEL #7 STAGE 1
    model = build_keras_model(
        model_index=7,
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        trainable_backbone=False
    )

    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break

    if backbone is None:
        raise RuntimeError("ISOLATED MODEL #7 ERROR: MobileNetV3Large backbone not found.")

    stage1_b_count = len(backbone.layers)
    stage1_trainable_count = sum(1 for l in backbone.layers if l.trainable)

    if stage1_b_count != 187:
        raise RuntimeError(f"Stage 1 backbone layer count mismatch: expected 187, found {stage1_b_count}")
    if stage1_trainable_count != 0:
        raise RuntimeError(f"Stage 1 trainable backbone layer count mismatch: expected 0, found {stage1_trainable_count}")

    print("\n============================================================")
    print("[MODEL #7 STAGE 1 SAFETY CHECK]")
    print("Backbone layers: 187")
    print("Trainable backbone layers: 0")
    print("============================================================\n")

    # SECTION 7: STAGE 1 WARMUP TRAINING
    logger.info(f"\n--- STAGE 1: WARMUP TRAINING ({warmup_epochs} Epochs, Frozen Backbone) ---")
    optimizer_stage1 = get_optimizer(optimizer_name, learning_rate=lr)
    loss_fn = get_loss_function(loss_name)

    model.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn,
        metrics=["categorical_accuracy"]
    )

    history_warmup = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=warmup_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks_list,
        verbose=1
    )

    # SECTION 8: CRITICAL ISOLATED STAGE 2 FINE-TUNING
    remaining_epochs = max(0, epochs - warmup_epochs)
    if remaining_epochs > 0:
        logger.info(f"\n--- STAGE 2: FINE-TUNING ({remaining_epochs} Remaining Epochs, Unfrozen Backbone) ---")

        # Step 1: Explicitly freeze EVERY backbone layer
        for layer in backbone.layers:
            layer.trainable = False

        # Step 2: Select top 40 layers and unfreeze non-BatchNormalization layers
        top_40_layers = backbone.layers[-40:]
        for layer in top_40_layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        # Step 3: Second explicit pass over complete backbone to lock every BN layer to False
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        # SECTION 9: HARD SAFETY VERIFICATION
        total_backbone_layers = len(backbone.layers)
        bn_layers = [l for l in backbone.layers if isinstance(l, keras.layers.BatchNormalization)]
        total_bn_layers = len(bn_layers)
        trainable_backbone_layers = sum(1 for l in backbone.layers if l.trainable)
        trainable_bn_layers = sum(1 for l in backbone.layers if isinstance(l, keras.layers.BatchNormalization) and l.trainable)
        frozen_earlier_layers = sum(1 for l in backbone.layers[:-40] if not l.trainable)

        if (total_backbone_layers, total_bn_layers, trainable_backbone_layers, trainable_bn_layers, frozen_earlier_layers) != (187, 46, 32, 0, 147):
            print("\n============================================================")
            print("[MODEL #7 ISOLATED SAFETY FAILURE]")
            print("==================================")
            for idx, l in enumerate(backbone.layers):
                if l.trainable:
                    print(f"index={idx} | name={l.name} | class={l.__class__.__name__} | trainable={l.trainable}")
            print("============================================================\n")
            raise RuntimeError(
                f"ISOLATED MODEL #7 SAFETY FAILURE: Counts mismatch! "
                f"Total={total_backbone_layers} (exp 187), BN={total_bn_layers} (exp 46), "
                f"Trainable={trainable_backbone_layers} (exp 32), Trainable BN={trainable_bn_layers} (exp 0), "
                f"Frozen Earlier={frozen_earlier_layers} (exp 147)."
            )

        # SECTION 10: SECOND SAFETY CHECK FOR EXACT 147/40 BOUNDARY
        # Verify layers 0..146 (first 147 layers) are all frozen
        for idx, l in enumerate(backbone.layers[:147]):
            if l.trainable:
                raise RuntimeError(f"BOUNDARY SAFETY FAILURE: Layer index {idx} ({l.name}) in frozen region 0..146 has trainable=True!")

        # Verify top 40 region (layers 147..186)
        for idx, l in enumerate(backbone.layers[147:], start=147):
            if isinstance(l, keras.layers.BatchNormalization):
                if l.trainable:
                    raise RuntimeError(f"BOUNDARY SAFETY FAILURE: BN Layer index {idx} ({l.name}) in top 40 region has trainable=True!")
            else:
                if not l.trainable:
                    raise RuntimeError(f"BOUNDARY SAFETY FAILURE: Non-BN Layer index {idx} ({l.name}) in top 40 region has trainable=False!")

        # Verify all BN layers anywhere in backbone are frozen
        for idx, l in enumerate(backbone.layers):
            if isinstance(l, keras.layers.BatchNormalization) and l.trainable:
                raise RuntimeError(f"BN SAFETY FAILURE: BN Layer index {idx} ({l.name}) has trainable=True!")

        # SECTION 11: PRINT EXACT TRAINABLE LAYERS
        print("\n============================================================")
        print("[MODEL #7 ISOLATED FINE-TUNING CONFIGURATION]")
        print("=============================================")
        print("--- FROZEN EARLIER LAYERS (0-146) ---")
        for idx, l in enumerate(backbone.layers[:147]):
            print(f"Layer {idx:3d}: {l.name:45s} | {l.__class__.__name__:25s} | trainable={l.trainable}")

        print("\n--- TOP 40 REGION (147-186) ---")
        for idx, l in enumerate(backbone.layers[147:], start=147):
            print(f"Layer {idx:3d}: {l.name:45s} | {l.__class__.__name__:25s} | trainable={l.trainable}")

        print("\nSUMMARY:")
        print(f"Total backbone layers: {total_backbone_layers}")
        print(f"Total BN layers: {total_bn_layers}")
        print(f"Trainable backbone layers: {trainable_backbone_layers}")
        print(f"Trainable BatchNormalization layers: {trainable_bn_layers}")
        print(f"Frozen earlier layers: {frozen_earlier_layers}")
        print("============================================================\n")

        # SECTION 13 & 16: STAGE 2 COMPILE & LEARNING RATE VERIFICATION
        fine_tune_optimizer = get_optimizer(optimizer_name, learning_rate=fine_tune_lr)
        
        print("============================================================")
        print("[MODEL #7 SAFETY PASSED]")
        print("Stage 2 compilation permitted.")
        print("Backbone remains controlled.")
        print(f"Optimizer: {optimizer_name} | Stage 2 Learning Rate: {fine_tune_lr}")
        print("No BatchNormalization layer is trainable.")
        print("============================================================\n")

        model.compile(
            optimizer=fine_tune_optimizer,
            loss=loss_fn,
            metrics=["categorical_accuracy"]
        )

        callbacks_list[3] = keras.callbacks.CSVLogger(
            filename=history_csv_path,
            separator=",",
            append=True
        )

        # SECTION 14: STAGE 2 TRAINING
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=warmup_epochs,
            epochs=epochs,
            class_weight=class_weights_dict,
            callbacks=callbacks_list,
            verbose=1
        )

    logger.info("[TRAINING COMPLETE] Best model saved to: " + best_model_path)

    # SECTION 17: SAVE ARTIFACTS & SUMMARY
    if os.path.exists(history_csv_path):
        history_df = pd.read_csv(history_csv_path)
        plot_and_save_training_curves(history_df, experiment_dir)

        best_val_acc = float(history_df['val_categorical_accuracy'].max()) if 'val_categorical_accuracy' in history_df.columns else float(history_df['val_accuracy'].max())
        min_val_loss = float(history_df['val_loss'].min())

        summary = {
            "model_index": 7,
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

        full_config = {
            "model_index": 7,
            "model_name": folder_name,
            "model_key": model_key,
            "seed": Config.SEED,
            "optimizer": optimizer_name,
            "initial_learning_rate": lr,
            "fine_tune_learning_rate": fine_tune_lr,
            "total_epochs_configured": epochs,
            "warmup_epochs": warmup_epochs,
            "epochs_actually_trained": len(history_df),
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": Config.REDUCE_LR_PATIENCE,
            "minimum_learning_rate": 1e-7,
            "loss_function": loss_name,
            "best_validation_accuracy": best_val_acc,
            "minimum_validation_loss": min_val_loss
        }

        config_json_path = os.path.join(experiment_dir, "training_configuration.json")
        with open(config_json_path, "w") as f:
            json.dump(full_config, f, indent=4)

    # SECTION 18: FINAL TRAINING SUMMARY
    print("\n============================================================")
    print("[MODEL #7 ISOLATED TRAINING COMPLETE]")
    print("=====================================")
    print("Model: 07_MobileNetV3Large")
    print(f"Stage 1 Epochs: {warmup_epochs} (Backbone trainable: 0)")
    print(f"Stage 2 Configured Epochs: {epochs - warmup_epochs}")
    print("Backbone total layers: 187")
    print("Trainable backbone layers: 32")
    print("BatchNormalization trainable layers: 0")
    print("Frozen earlier layers: 147")
    print(f"Best model: {best_model_path}")
    print(f"History: {history_csv_path}")
    print("============================================================\n")


# SECTION 19: COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Isolated Training Engine for Model #7 (MobileNetV3Large)")
    parser.add_argument("--model", type=int, default=7, help="Model index (MUST be 7)")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Total training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=Config.WARMUP_EPOCHS, help="Stage 1 warmup epochs")

    args = parser.parse_args()

    if args.model != 7:
        raise ValueError(f"train2.py is hard-coded for Model #7 (MobileNetV3Large) only! Cannot run --model {args.model}")

    run_isolated_training(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs
    )
