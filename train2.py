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
# Smoke Test Engine (1 Batch Forward/Loss/Backward Pass)
# ---------------------------------------------------------
def run_smoke_test(model: keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, loss_fn: keras.losses.Loss, optimizer: keras.optimizers.Optimizer, experiment_dir: str):
    print("\n============================================================")
    print("[RUNNING MODEL #7 SMOKE TEST (1 Batch Pass)]")
    print("============================================================")

    for images, labels in train_ds.take(1):
        with tf.GradientTape() as tape:
            preds = model(images, training=True)
            loss = loss_fn(labels, preds)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"  [PASS] Forward Pass Output Shape: {preds.shape}")
        print(f"  [PASS] Batch Loss Value: {float(loss):.4f}")
        print(f"  [PASS] Gradient Computation: {len(grads)} gradients updated")

    for val_images, val_labels in val_ds.take(1):
        val_preds = model(val_images, training=False)
        val_loss = loss_fn(val_labels, val_preds)
        print(f"  [PASS] Validation Pass Output Shape: {val_preds.shape}")
        print(f"  [PASS] Validation Loss Value: {float(val_loss):.4f}")

    smoke_ckpt_path = os.path.join(experiment_dir, "smoke_checkpoint.keras")
    model.save(smoke_ckpt_path)
    if not os.path.exists(smoke_ckpt_path):
        raise RuntimeError("SMOKE TEST FAILURE: Could not save smoke checkpoint!")
    print(f"  [PASS] Smoke Checkpoint Saved: {smoke_ckpt_path}")
    print("============================================================\n")


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
    patience: int = Config.PATIENCE,
    smoke_test_only: bool = False
):
    # SECTION 3: MODEL CONFIGURATION
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]

    if model_key != "mobilenet_v3_large":
        raise RuntimeError(f"CRITICAL ERROR: train2.py is hard-coded for Model #7 (mobilenet_v3_large). Got '{model_key}'")

    print("\n============================================================")
    print("[ISOLATED MODEL #7 TRAINING ENGINE]")
    print("==================================")
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

    # FORBIDDEN PATTERN CHECK INSIDE TRAIN2.PY SOURCE
    with open(script_file, "r", encoding="utf-8") as f:
        self_source = f.read()

    forbidden_target_msg = "[" + "FINE-TUNING" + "] " + "Unfroze all backbone layers " + "for full fine-tuning."
    forbidden_target_unfreeze = "unfreeze" + "_" + "model" + "_" + "backbone" + "("
    forbidden_target_base = "base" + "_" + "model" + ".trainable" + " = " + "True"
    forbidden_target_backbone = "back" + "bone" + ".trainable" + " = " + "True"
    forbidden_patterns = [
        forbidden_target_msg,
        forbidden_target_base,
        forbidden_target_backbone,
        forbidden_target_unfreeze
    ]

    for pattern in forbidden_patterns:
        if pattern in self_source:
            raise RuntimeError(f"CRITICAL FORBIDDEN PATTERN CHECK FAILED: Found forbidden string '{pattern}' inside train2.py!")

    # EXPERIMENT DIRECTORY ISOLATION
    base_dir = Config.get_base_dir()
    experiment_dir = os.path.join(base_dir, "Experiments", "07_MobileNetV3Large_Isolated")
    os.makedirs(experiment_dir, exist_ok=True)
    logger.info(f"Isolated Experiment Directory: {experiment_dir}")

    # DATASET PIPELINE SETUP
    set_seed(Config.SEED)
    split_info = create_reproducible_splits(seed=Config.SEED)

    total_samples = len(split_info["train_paths"]) + len(split_info["val_paths"]) + len(split_info["test_paths"])
    train_count = len(split_info["train_paths"])
    val_count = len(split_info["val_paths"])
    test_count = len(split_info["test_paths"])

    # Synthetic dataset safety check: Real dataset has ~5,734 images.
    if total_samples < 500:
        err_msg = (
            f"CRITICAL DATASET FAILURE: Only {total_samples} samples detected (synthetic fallback triggered)! "
            f"Real cashew dataset expected (~5,734 images). Aborting training."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    batch_size = get_optimal_batch_size()
    tf_pipelines = build_tf_data_pipelines(split_info, batch_size=batch_size)

    train_ds = tf_pipelines["train"]
    val_ds = tf_pipelines["val"]
    num_classes = len(split_info["class_names"])
    class_weights_dict = {i: float(w) for i, w in enumerate(split_info["class_weights"])}

    # BUILD MODEL #7 STAGE 1
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

    trainable_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    total_params = sum(np.prod(v.shape) for v in model.variables)

    # ---------------------------------------------------------
    # MANDATORY PRE-FLIGHT CHECKS
    # ---------------------------------------------------------
    print("\n============================================================")
    print("[PRE-FLIGHT CHECK 1: MODEL 7 CHECK]")
    print("===================================")
    print(f"Model Name:           {folder_name}")
    print(f"Input Shape:          {model.input_shape}")
    print(f"Output Shape:         {model.output_shape}")
    print(f"Number of Classes:    {num_classes}")
    print(f"Loss Function:       {loss_name}")
    print(f"Optimizer:            {optimizer_name}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters:     {total_params:,}")

    print("\n[PRE-FLIGHT CHECK 2: DATA CHECK]")
    print("=================================")
    print(f"Total Dataset Samples: {total_samples}")
    print(f"Training Sample Count: {train_count}")
    print(f"Validation Count:      {val_count}")
    print(f"Testing Count:         {test_count}")
    print(f"Image Input Shape:     (224, 224, 3)")
    print(f"Label Format:          One-Hot Encoded ({num_classes},)")
    print("Class Distribution:")
    for cls_name, count in split_info["class_counts"].items():
        print(f"  - {cls_name}: {count}")

    print("\n[PRE-FLIGHT CHECK 3: PATH CHECK]")
    print("=================================")
    print(f"Project Root:         {base_dir}")
    print(f"Raw Dataset Path:     {Config.get_raw_dir()}")
    print(f"Train Split CSV:      {os.path.join(Config.get_preprocessed_dir(), 'train_split.csv')}")
    print(f"Val Split CSV:        {os.path.join(Config.get_preprocessed_dir(), 'val_split.csv')}")
    print(f"Checkpoint Dir:       {experiment_dir}")
    print(f"Artifact Dir:         {experiment_dir}")

    gpus = tf.config.list_physical_devices('GPU')
    gpu_status = f"{len(gpus)} GPU(s) Available: {[g.name for g in gpus]}" if gpus else "CPU Only (No GPU detected)"

    print("\n[PRE-FLIGHT CHECK 4: REPRODUCIBILITY CHECK]")
    print("===========================================")
    print(f"Random Seed:          {Config.SEED}")
    print(f"TensorFlow Version:   {tf.__version__}")
    print(f"Hardware Compute:     {gpu_status}")
    print("============================================================\n")

    # Assert basic pre-flight requirements
    if model.output_shape[-1] != num_classes:
        raise RuntimeError(f"PRE-FLIGHT FAILURE: Output shape {model.output_shape} incompatible with class count {num_classes}")
    if total_samples < 500:
        raise RuntimeError("PRE-FLIGHT FAILURE: Dataset sample count too low!")

    # ---------------------------------------------------------
    # SMOKE TEST EXECUTION
    # ---------------------------------------------------------
    optimizer_stage1 = get_optimizer(optimizer_name, learning_rate=lr)
    loss_fn = get_loss_function(loss_name)

    model.compile(
        optimizer=optimizer_stage1,
        loss=loss_fn,
        metrics=["categorical_accuracy"]
    )

    run_smoke_test(model, train_ds, val_ds, loss_fn, optimizer_stage1, experiment_dir)

    if smoke_test_only:
        print("[SMOKE TEST ONLY MODE COMPLETED SUCCESSFULLY] Stopping before full training as requested.")
        return

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

    # SECTION 7: STAGE 1 WARMUP TRAINING
    logger.info(f"\n--- STAGE 1: WARMUP TRAINING ({warmup_epochs} Epochs, Frozen Backbone) ---")
    print(f"[MODEL 7][TRAINING] Starting Stage 1 Warmup ({warmup_epochs} Epochs)...")

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
        print(f"[MODEL 7][TRAINING] Starting Stage 2 Controlled Fine-Tuning ({remaining_epochs} Epochs)...")

        # Step 1: Freeze EVERY backbone layer
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

        # HARD SAFETY VERIFICATION
        total_backbone_layers = len(backbone.layers)
        bn_layers = [l for l in backbone.layers if isinstance(l, keras.layers.BatchNormalization)]
        total_bn_layers = len(bn_layers)
        trainable_backbone_layers = sum(1 for l in backbone.layers if l.trainable)
        trainable_bn_layers = sum(1 for l in backbone.layers if isinstance(l, keras.layers.BatchNormalization) and l.trainable)
        frozen_earlier_layers = sum(1 for l in backbone.layers[:-40] if not l.trainable)

        if (total_backbone_layers, total_bn_layers, trainable_backbone_layers, trainable_bn_layers, frozen_earlier_layers) != (187, 46, 32, 0, 147):
            raise RuntimeError(
                f"ISOLATED MODEL #7 SAFETY FAILURE: Counts mismatch! "
                f"Total={total_backbone_layers} (exp 187), BN={total_bn_layers} (exp 46), "
                f"Trainable={trainable_backbone_layers} (exp 32), Trainable BN={trainable_bn_layers} (exp 0), "
                f"Frozen Earlier={frozen_earlier_layers} (exp 147)."
            )

        fine_tune_optimizer = get_optimizer(optimizer_name, learning_rate=fine_tune_lr)

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

    # SAVE ARTIFACTS & SUMMARY
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

    # ---------------------------------------------------------
    # POST-TRAINING INFERENCE & RELOAD VALIDATION
    # ---------------------------------------------------------
    print("\n============================================================")
    print("[POST-TRAINING ARTIFACT & INFERENCE VALIDATION]")
    print("===============================================")
    if not os.path.exists(best_model_path):
        raise RuntimeError(f"POST-TRAINING FAILURE: Saved model checkpoint not found at {best_model_path}")
    print(f"  [PASS] Best Checkpoint Exists: {best_model_path}")

    reloaded_model = keras.models.load_model(best_model_path, custom_objects={loss_name: loss_fn})
    print(f"  [PASS] Model Successfully Reloaded from Disk")

    for test_images, _ in val_ds.take(1):
        sample_img = test_images[:1]
        sample_pred = reloaded_model.predict(sample_img, verbose=0)
        prob_sum = float(np.sum(sample_pred))
        pred_class = int(np.argmax(sample_pred[0]))
        print(f"  [PASS] Inference Output Shape: {sample_pred.shape}")
        print(f"  [PASS] Inference Probabilities Sum: {prob_sum:.4f}")
        print(f"  [PASS] Predicted Class Index: {pred_class} ({split_info['class_names'][pred_class]})")

    print("============================================================\n")


# COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Isolated Training Engine for Model #7 (MobileNetV3Large)")
    parser.add_argument("--model", type=int, default=7, help="Model index (MUST be 7)")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Total training epochs (default: 50)")
    parser.add_argument("--warmup-epochs", type=int, default=Config.WARMUP_EPOCHS, help="Stage 1 warmup epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--fine-tune-lr", type=float, default=Config.FINE_TUNE_LEARNING_RATE, help="Fine-tuning learning rate (default: 1e-5)")
    parser.add_argument("--optimizer", type=str, default=Config.OPTIMIZER, choices=["adam", "adamw", "sgd"], help="Optimizer selection (default: adam)")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy", choices=["categorical_crossentropy", "focal_loss"], help="Loss function selection")
    parser.add_argument("--patience", type=int, default=Config.PATIENCE, help="Early stopping patience (default: 10)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch pre-flight smoke test only and exit")

    args = parser.parse_args()

    if args.model != 7:
        raise ValueError(f"train2.py is hard-coded for Model #7 (MobileNetV3Large) only! Cannot run --model {args.model}")

    run_isolated_training(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        optimizer_name=args.optimizer,
        loss_name=args.loss,
        patience=args.patience,
        smoke_test_only=args.smoke_test
    )
