"""
Cashew Pest and Disease Diagnosis System
Unseen Test Set Evaluator & Publication Metrics Suite
"""

import os
import time
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

from src.config import Config
from src.utils import set_seed, get_logger, get_model_size_mb, get_gpu_memory_mb, count_parameters
from src.dataset import prepare_dataset_splits, get_dataloaders
from src.models import VisionModelFactory

logger = get_logger("TestEvaluator")

def evaluate_model(
    model_index: int = 1,
    base_dir: str = None,
    checkpoint_name: str = "best_model.pth"
) -> Dict:
    """
    Evaluates a trained model ONLY on the unseen 15% test split.
    Generates: metrics.csv, predictions.csv, confusion_matrix.png,
    classification_report.txt, roc_curve.png, precision_recall_curve.png.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    if base_dir is None:
        base_dir = Config.get_base_dir()

    folder_name, raw_model_name = Config.MODEL_MAP[model_index]
    exp_dir = os.path.join(base_dir, "Experiments", folder_name)
    best_model_path = os.path.join(exp_dir, checkpoint_name)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found at '{best_model_path}'. Train model first.")

    logger.info("=" * 60)
    logger.info(f"EVALUATING MODEL ON UNSEEN TEST SET: [{folder_name}]")
    logger.info(f"Checkpoint Path: {best_model_path}")
    logger.info("=" * 60)

    # 1. Load Dataset Splits & Test Loader
    split_info = prepare_dataset_splits(base_dir=base_dir, seed=Config.SEED)
    class_names = split_info["class_names"]
    num_classes = len(class_names)
    
    loaders = get_dataloaders(split_info, batch_size=1, use_weighted_sampler=False)
    test_loader = loaders["test"]
    test_paths = split_info["test_paths"]

    # 2. Load Model Checkpoint
    checkpoint = torch.load(best_model_path, map_location=device)
    model = VisionModelFactory.create_model(
        model_name=raw_model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False
    ).to(device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Model Size & Memory Metrics
    model_size_mb = get_model_size_mb(model)
    gpu_mem_mb = get_gpu_memory_mb()
    total_params = count_parameters(model)["total_parameters"]

    # 4. Perform Inference on Unseen Test Images
    all_targets = []
    all_preds = []
    all_probs = []
    inference_times = []
    predictions_log = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            target = labels.item()

            t0 = time.perf_counter()
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            inference_times.append(latency_ms)

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            is_correct = (pred_class == target)

            all_targets.append(target)
            all_preds.append(pred_class)
            all_probs.append(probs)

            predictions_log.append({
                "Image_Path": test_paths[idx],
                "Ground_Truth": class_names[target],
                "Predicted_Class": class_names[pred_class],
                "Confidence": round(confidence, 4),
                "Is_Correct": is_correct
            })

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    avg_inference_ms = float(np.mean(inference_times))

    # 5. Compute Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)

    logger.info(f"[TEST RESULTS - {folder_name}]")
    logger.info(f"  - Test Accuracy:        {acc * 100.0:.2f}%")
    logger.info(f"  - Precision (Macro):    {prec_macro:.4f}")
    logger.info(f"  - Recall (Macro):       {rec_macro:.4f}")
    logger.info(f"  - F1-Score (Macro):     {f1_macro:.4f}")
    logger.info(f"  - Avg Inference Time:   {avg_inference_ms:.2f} ms / image")
    logger.info(f"  - Model Size:           {model_size_mb:.2f} MB")

    # 6. Save predictions.csv
    df_preds = pd.DataFrame(predictions_log)
    df_preds.to_csv(os.path.join(exp_dir, "predictions.csv"), index=False)

    # 7. Save metrics.csv
    metrics_summary = {
        "Model_Name": [raw_model_name],
        "Folder_Name": [folder_name],
        "Test_Accuracy": [acc],
        "Precision_Macro": [prec_macro],
        "Recall_Macro": [rec_macro],
        "F1_Score_Macro": [f1_macro],
        "Precision_Weighted": [prec_weight],
        "Recall_Weighted": [rec_weight],
        "F1_Score_Weighted": [f1_weight],
        "Avg_Inference_Time_ms": [avg_inference_ms],
        "Model_Size_MB": [model_size_mb],
        "Total_Parameters": [total_params],
        "GPU_Memory_MB": [gpu_mem_mb]
    }
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # 8. Save classification_report.txt
    cls_report = classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0)
    report_path = os.path.join(exp_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(f"Classification Report - {folder_name} ({raw_model_name})\n")
        rf.write("=" * 60 + "\n\n")
        rf.write(cls_report)

    # 9. Generate & Save Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, os.path.join(exp_dir, "confusion_matrix.png"), title=f"Confusion Matrix ({folder_name})")

    # 10. Generate & Save ROC Curve
    plot_roc_curve(all_targets, all_probs, class_names, os.path.join(exp_dir, "roc_curve.png"))

    # 11. Generate & Save Precision-Recall Curve
    plot_precision_recall_curve(all_targets, all_probs, class_names, os.path.join(exp_dir, "precision_recall_curve.png"))

    logger.info(f"[EVALUATION SUCCESS] All test artifacts saved to {exp_dir}")
    return metrics_summary

# ---------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: str, title: str) -> None:
    """Plots and saves normalized confusion matrix heatmap."""
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Scale'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_roc_curve(targets: np.ndarray, probs: np.ndarray, classes: List[str], output_path: str) -> None:
    """Generates One-vs-Rest ROC curves with AUC for each class."""
    n_classes = len(classes)
    y_bin = label_binarize(targets, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7))
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        if np.sum(y_bin[:, i]) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors(i), lw=2, label=f"{classes[i]} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_precision_recall_curve(targets: np.ndarray, probs: np.ndarray, classes: List[str], output_path: str) -> None:
    """Generates Multi-Class Precision-Recall curves with Average Precision (AP)."""
    n_classes = len(classes)
    y_bin = label_binarize(targets, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 7))
    colors = plt.cm.get_cmap('tab10', n_classes)

    for i in range(n_classes):
        if np.sum(y_bin[:, i]) == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        plt.plot(recall, precision, color=colors(i), lw=2, label=f"{classes[i]} (AP = {ap:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-Class Precision-Recall (PR) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_model(model_index=1)
