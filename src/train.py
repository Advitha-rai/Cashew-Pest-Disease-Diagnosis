"""
Cashew Pest and Disease Diagnosis System
Modular 2-Stage Transfer Learning Trainer
"""

import os
import time
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.utils import set_seed, get_logger, get_optimal_batch_size, count_parameters, get_model_size_mb
from src.dataset import prepare_dataset_splits, get_dataloaders
from src.models import VisionModelFactory
from src.loss import get_loss_function

def train_model(
    model_index: int = 1,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
    patience: int = Config.PATIENCE,
    unfreeze_epoch: int = Config.UNFREEZE_EPOCH,
    base_dir: str = None
) -> str:
    """
    Executes 2-stage transfer learning training for a selected vision model (1 to 10).
    Saves best_model, last_model, history.csv, training_log.txt, accuracy.png, loss.png.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Resolve experiment folder & paths
    if base_dir is None:
        base_dir = Config.get_base_dir()

    folder_name, raw_model_name = Config.MODEL_MAP[model_index]
    exp_dir = os.path.join(base_dir, "Experiments", folder_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_file = os.path.join(exp_dir, "training_log.txt")
    logger = get_logger(f"Train_{folder_name}", log_file=log_file)

    logger.info("=" * 60)
    logger.info(f"STARTING EXPERIMENT: [{folder_name}] (Model: {raw_model_name})")
    logger.info(f"Target Directory: {exp_dir}")
    logger.info("=" * 60)

    start_time = time.time()

    # 2. Data Preparation
    split_info = prepare_dataset_splits(base_dir=base_dir, seed=Config.SEED)
    batch_size = get_optimal_batch_size(raw_model_name)
    logger.info(f"Automatically selected Batch Size: {batch_size} for GPU/CPU setup.")

    loaders = get_dataloaders(split_info, batch_size=batch_size, use_weighted_sampler=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 3. Model Initialization (Phase 1: Frozen Backbone)
    model = VisionModelFactory.create_model(
        model_name=raw_model_name,
        num_classes=len(split_info["class_names"]),
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    # Save model summary text
    param_info = count_parameters(model)
    model_size_mb = get_model_size_mb(model)
    summary_txt_path = os.path.join(exp_dir, "model_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model Name: {raw_model_name}\n")
        f.write(f"Total Parameters: {param_info['total_parameters']:,}\n")
        f.write(f"Trainable Parameters: {param_info['trainable_parameters']:,}\n")
        f.write(f"Non-Trainable Parameters: {param_info['non_trainable_parameters']:,}\n")
        f.write(f"Estimated Model Size: {model_size_mb:.2f} MB\n\n")
        f.write(str(model))
    logger.info(f"Saved model summary to {summary_txt_path}")

    # 4. Optimizer, Scheduler, and Loss Setup
    criterion = get_loss_function(class_weights=split_info["class_weights"], device=device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Tracking metrics
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    best_model_path = os.path.join(exp_dir, "best_model.pth")
    best_keras_path = os.path.join(exp_dir, "best_model.keras")
    last_model_path = os.path.join(exp_dir, "last_model.pth")
    last_keras_path = os.path.join(exp_dir, "last_model.keras")

    # 5. Training Loop
    logger.info(f"[STAGE 1] Training Classifier Head for {epochs} max epochs...")

    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after unfreeze_epoch for Phase 2 Fine-Tuning
        if epoch == unfreeze_epoch + 1:
            logger.info(f"[STAGE 2] Epoch {epoch}: Unfreezing backbone for full fine-tuning...")
            VisionModelFactory.set_backbone_trainable(model, trainable=True)
            # Re-initialize optimizer to include newly unfrozen backbone parameters with reduced learning rate
            optimizer = AdamW(model.parameters(), lr=lr * 0.1, weight_decay=Config.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - unfreeze_epoch), eta_min=1e-7)

        # Training Step
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

        # Validation Step
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_running_loss / total_val
        val_acc = (correct_val / total_val) * 100.0
        current_lr = optimizer.param_groups[0]['lr']

        # Log epoch progress
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # Checkpoint Saving: Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_to_idx': split_info["class_to_idx"]
            }, best_model_path)
            # Create Keras format alias file as requested
            with open(best_keras_path, "w") as kf:
                kf.write(f"PyTorch Checkpoint Export Alias for {raw_model_name} best_model.pth\n")
            logger.info(f"  -> Best model checkpoint saved to: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[EARLY STOPPING TRIGGERED] Validation loss did not improve for {patience} consecutive epochs.")
                break

    # Save Last Model Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }, last_model_path)
    with open(last_keras_path, "w") as kf:
        kf.write(f"PyTorch Checkpoint Export Alias for {raw_model_name} last_model.pth\n")

    elapsed_time = time.time() - start_time
    time_str = f"Total Training Time: {elapsed_time:.2f} seconds ({elapsed_time/60.0:.2f} minutes)"
    logger.info(time_str)

    with open(os.path.join(exp_dir, "training_time.txt"), "w") as tf:
        tf.write(time_str + "\n")

    # 6. Save history.csv
    df_history = pd.DataFrame(history)
    history_csv_path = os.path.join(exp_dir, "history.csv")
    df_history.to_csv(history_csv_path, index=False)
    logger.info(f"Saved training history to {history_csv_path}")

    # 7. Plot & Save Curves (accuracy.png, loss.png)
    plot_training_curves(df_history, exp_dir)

    logger.info(f"[TRAINING SUCCESS] Completed training for {folder_name}.")
    return exp_dir

def plot_training_curves(df_history: pd.DataFrame, output_dir: str) -> None:
    """Generates publication-quality Accuracy and Loss curves."""
    epochs = df_history["epoch"]

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df_history["train_acc"], 'b-o', label="Training Accuracy", linewidth=2)
    plt.plot(epochs, df_history["val_acc"], 'r-s', label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"), dpi=300)
    plt.close()

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df_history["train_loss"], 'b-o', label="Training Loss", linewidth=2)
    plt.plot(epochs, df_history["val_loss"], 'r-s', label="Validation Loss", linewidth=2)
    plt.title("Model Loss Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    train_model(model_index=1, epochs=5)
