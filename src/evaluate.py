"""
Cashew Pest and Disease Diagnosis System
Phase 4: Comprehensive Model Evaluation & Benchmarking Engine (TensorFlow / Keras)

Computes research-grade classification metrics, confusion matrices, ROC/PR curves,
calibration errors (ECE), confidence analysis, error profiling, latency timing,
misclassified image exports, and cross-model comparative rankings.

Includes Independent Dataset Loading:
Directly loads Preprocessed/test_split.csv when available to keep evaluation 100%
decoupled from training, with automatic fallback to create_reproducible_splits().

Includes Confidence Thresholding Policy (80% confidence requirement):
Prevents random predictions by outputting 'Prediction Uncertain. Please upload a clearer image'
whenever prediction confidence drops below Config.CONFIDENCE_THRESHOLD (0.80).
"""

import os
import sys
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef, roc_curve, auc, precision_recall_curve,
    average_precision_score
)

from src.config import Config
from src.utils import set_seed, get_logger
from src.dataset import create_reproducible_splits

# Configure dedicated evaluation loggers
eval_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("EvaluationEngine", eval_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)


# ---------------------------------------------------------
# 1. Independent Test Dataset Loader
# ---------------------------------------------------------
def load_evaluation_test_dataset() -> Dict:
    """
    Loads test dataset directly from Preprocessed/test_split.csv if present,
    making evaluation 100% independent from training execution.
    If test_split.csv is missing, falls back to create_reproducible_splits().
    """
    preprocessed_dir = Config.get_preprocessed_dir()
    test_csv_path = os.path.join(preprocessed_dir, "test_split.csv")

    if os.path.exists(test_csv_path):
        logger.info(f"Loading pre-existing test split CSV from: {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)
        
        test_paths = test_df["file_path"].tolist()

        if "class_name" in test_df.columns:
            unique_classes = sorted(test_df["class_name"].unique().tolist())
            class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            idx_to_class = {i: c for i, c in enumerate(unique_classes)}
            
            if "label" in test_df.columns:
                test_labels_idx = test_df["label"].values
            else:
                test_labels_idx = np.array([class_to_idx[c] for c in test_df["class_name"]])
        else:
            unique_classes = Config.DEFAULT_CLASSES
            class_to_idx = {c: i for i, c in enumerate(unique_classes)}
            idx_to_class = {i: c for i, c in enumerate(unique_classes)}
            test_labels_idx = test_df["label"].values

        return {
            "test_paths": test_paths,
            "test_labels": test_labels_idx,
            "class_names": unique_classes,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class
        }
    else:
        logger.warning(f"Test split CSV not found at '{test_csv_path}'. Running fallback split generator...")
        split_info = create_reproducible_splits(seed=Config.SEED)
        return {
            "test_paths": split_info["test_paths"],
            "test_labels": np.array(split_info["test_labels"]),
            "class_names": split_info["class_names"],
            "class_to_idx": split_info["class_to_idx"],
            "idx_to_class": split_info["idx_to_class"]
        }


# ---------------------------------------------------------
# 2. Single Image Inference Engine with Uncertainty Protection
# ---------------------------------------------------------
def predict_single_image(model: keras.Model, image_path: str, class_names: List[str]) -> Dict:
    """
    Performs inference on a single image with confidence thresholding protection.
    If prediction confidence < Config.CONFIDENCE_THRESHOLD (80%), prevents random guessing
    and returns 'Prediction Uncertain. Please upload a clearer image'.
    """
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [Config.IMG_HEIGHT, Config.IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    img_batch = tf.expand_dims(img, axis=0)

    probs = model(img_batch, training=False).numpy()[0]
    top_class_idx = int(np.argmax(probs))
    confidence = float(probs[top_class_idx])
    top_class_name = class_names[top_class_idx]

    is_uncertain = confidence < Config.CONFIDENCE_THRESHOLD

    if is_uncertain:
        display_message = Config.UNCERTAIN_PREDICTION_MESSAGE
        prediction_result = "Prediction Uncertain"
    else:
        display_message = f"Diagnosed: {top_class_name} (Confidence: {confidence * 100:.1f}%)"
        prediction_result = top_class_name

    return {
        "top_class": top_class_name,
        "confidence": round(confidence, 4),
        "is_uncertain": is_uncertain,
        "prediction_result": prediction_result,
        "display_message": display_message,
        "all_probabilities": {cls: round(float(probs[i]), 4) for i, cls in enumerate(class_names)}
    }


# ---------------------------------------------------------
# 3. Expected Calibration Error (ECE) Calculator
# ---------------------------------------------------------
def compute_expected_calibration_error(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics for Reliability Diagrams.
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin.astype(float))
        bin_sizes.append(np.sum(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return float(ece), np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


# ---------------------------------------------------------
# 4. Plotting Engine (Publication-Quality Visualizations)
# ---------------------------------------------------------
def plot_and_save_confusion_matrices(cm_raw: np.ndarray, cm_norm: np.ndarray, class_names: List[str], save_dir: str):
    """Generates high-resolution raw and normalized confusion matrix heatmaps."""
    # 1. Raw Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title('Cashew Diagnosis - Confusion Matrix (Raw Counts)', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    raw_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(raw_path, dpi=300)
    plt.close()

    # 2. Normalized Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title('Cashew Diagnosis - Normalized Confusion Matrix', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    norm_path = os.path.join(save_dir, "confusion_matrix_normalized.png")
    plt.savefig(norm_path, dpi=300)
    plt.close()


def plot_and_save_roc_curves(y_true_onehot: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_dir: str) -> Dict[str, float]:
    """Generates One-vs-Rest (OvR), Micro-average, and Macro-average ROC curves."""
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
        roc_auc[class_names[i]] = float(auc(fpr[i], tpr[i]))

    # Compute micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_prob.ravel())
    roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(auc(fpr["macro"], tpr["macro"]))

    # Plot ROC Curves
    plt.figure(figsize=(9, 7))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})', color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})', color='navy', linestyle=':', linewidth=3)

    colors = ['#2b5c8f', '#e07a5f', '#81b29a', '#f4a261', '#9b59b6', '#34495e']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc[class_names[i]]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title('Multi-Class One-vs-Rest ROC Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    return roc_auc


def plot_and_save_precision_recall_curves(y_true_onehot: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_dir: str):
    """Generates Per-Class and Micro-average Precision-Recall curves."""
    n_classes = len(class_names)
    precision = dict()
    recall = dict()

    plt.figure(figsize=(9, 7))

    # Micro-average Precision-Recall
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_onehot.ravel(), y_prob.ravel())
    micro_ap = average_precision_score(y_true_onehot, y_prob, average="micro")
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=3, linestyle=':', label=f'Micro-average PR (AP = {micro_ap:.4f})')

    colors = ['#2b5c8f', '#e07a5f', '#81b29a', '#f4a261', '#9b59b6', '#34495e']
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR {class_names[i]} (AP = {ap:.4f})')

    plt.xlabel('Recall', fontsize=11, fontweight='bold')
    plt.ylabel('Precision', fontsize=11, fontweight='bold')
    plt.title('Multi-Class Precision-Recall Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    pr_path = os.path.join(save_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()


def plot_and_save_confidence_distribution(confidences: np.ndarray, correctness: np.ndarray, save_dir: str):
    """Generates confidence distribution histogram comparing correct vs incorrect predictions."""
    plt.figure(figsize=(9, 5))
    correct_confs = confidences[correctness]
    incorrect_confs = confidences[~correctness]

    plt.hist(correct_confs, bins=20, alpha=0.7, color='#81b29a', label=f'Correct Predictions (n={len(correct_confs)})', edgecolor='black')
    if len(incorrect_confs) > 0:
        plt.hist(incorrect_confs, bins=20, alpha=0.7, color='#e07a5f', label=f'Incorrect Predictions (n={len(incorrect_confs)})', edgecolor='black')

    # Draw vertical line for the 80% confidence threshold requirement
    plt.axvline(x=Config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Uncertainty Threshold ({Config.CONFIDENCE_THRESHOLD*100:.0f}%)')

    plt.xlabel('Prediction Confidence Score', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Test Samples', fontsize=11, fontweight='bold')
    plt.title('Model Prediction Confidence & Uncertainty Threshold Histogram', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    conf_path = os.path.join(save_dir, "confidence_distribution.png")
    plt.savefig(conf_path, dpi=300)
    plt.close()


# ---------------------------------------------------------
# 5. Model Profiling & Memory Stats
# ---------------------------------------------------------
def profile_model_information(model: keras.Model, model_file_path: str) -> Dict:
    """Calculates total/trainable parameters, file size (MB), and GPU/CPU device info."""
    total_params = int(model.count_params())
    trainable_params = int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_params = total_params - trainable_params

    file_size_mb = 0.0
    if os.path.exists(model_file_path):
        file_size_mb = float(os.path.getsize(model_file_path) / (1024 * 1024))

    gpus = tf.config.list_physical_devices('GPU')
    device_used = gpus[0].name if gpus else "CPU"

    return {
        "model_file_size_mb": round(file_size_mb, 2),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "hardware_device_used": device_used
    }


# ---------------------------------------------------------
# 6. Main Single Model Evaluation Engine
# ---------------------------------------------------------
def evaluate_single_model(model_index: int) -> Dict:
    """
    Evaluates a single model from Experiments/<Model_Name>/best_model.keras on the Phase 2 test set.
    Includes confidence thresholding (80%) to flag uncertain predictions.
    """
    set_seed(Config.SEED)
    folder_name, model_key = Config.MODEL_MAP[model_index]
    experiment_dir = Config.get_experiment_dir(model_index)
    model_path = os.path.join(experiment_dir, "best_model.keras")

    logger.info(f"\n=======================================================================")
    logger.info(f"  EVALUATING MODEL #{model_index}: {folder_name}")
    logger.info(f"=======================================================================")

    if not os.path.exists(model_path):
        err_msg = f"Model checkpoint file not found at: {model_path}. Train the model before evaluating."
        exc_logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # 1. Load Trained Keras Model
    logger.info(f"Loading Keras model checkpoint from: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.warning(f"Standard load failed ({e}). Loading with custom_objects wrapper...")
        model = keras.models.load_model(model_path, compile=False, safe_mode=False)

    model.compile(loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    # 2. Load Phase 2 Test Dataset Split (Directly from Preprocessed/test_split.csv)
    test_data = load_evaluation_test_dataset()
    test_paths = test_data["test_paths"]
    test_labels_idx = np.array(test_data["test_labels"])
    class_names = test_data["class_names"]
    idx_to_class = test_data["idx_to_class"]
    num_classes = len(class_names)

    y_true_onehot = tf.one_hot(test_labels_idx, depth=num_classes).numpy()

    # 3. Perform Batched Inference & Measure Latency
    logger.info(f"Running inference on {len(test_paths)} test images...")
    y_prob_list = []
    
    start_time = time.time()
    for img_p in test_paths:
        img_bytes = tf.io.read_file(img_p)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [Config.IMG_HEIGHT, Config.IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0
        img_batch = tf.expand_dims(img, axis=0)

        preds = model(img_batch, training=False)
        y_prob_list.append(preds.numpy()[0])
    
    total_inference_time = time.time() - start_time
    avg_latency_ms = (total_inference_time / len(test_paths)) * 1000.0
    throughput_fps = len(test_paths) / total_inference_time

    y_prob = np.array(y_prob_list)
    y_pred_idx = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred_idx == test_labels_idx)

    # Uncertainty evaluation check (Confidence < 80%)
    is_uncertain_mask = confidences < Config.CONFIDENCE_THRESHOLD
    num_uncertain = int(np.sum(is_uncertain_mask))
    uncertain_ratio = float(num_uncertain / len(test_paths))

    system_responses = [
        f"Diagnosed: {idx_to_class[pred_idx]} (Confidence: {conf*100:.1f}%)" if conf >= Config.CONFIDENCE_THRESHOLD
        else Config.UNCERTAIN_PREDICTION_MESSAGE
        for pred_idx, conf in zip(y_pred_idx, confidences)
    ]

    # 4. Compute Standard Classification Metrics
    acc = accuracy_score(test_labels_idx, y_pred_idx)
    bal_acc = balanced_accuracy_score(test_labels_idx, y_pred_idx)
    kappa = cohen_kappa_score(test_labels_idx, y_pred_idx)
    mcc = matthews_corrcoef(test_labels_idx, y_pred_idx)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_labels_idx, y_pred_idx, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels_idx, y_pred_idx, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(test_labels_idx, y_pred_idx, average=None, zero_division=0)

    ece, bin_accs, bin_confs, bin_sizes = compute_expected_calibration_error(test_labels_idx, y_prob)

    # 5. Export Prediction Details CSV with Uncertainty Status
    predictions_df = pd.DataFrame({
        "image_path": test_paths,
        "true_label": [idx_to_class[i] for i in test_labels_idx],
        "top_predicted_label": [idx_to_class[i] for i in y_pred_idx],
        "confidence_score": np.round(confidences, 4),
        "is_uncertain": is_uncertain_mask,
        "user_display_message": system_responses,
        "correctness": correctness
    })
    pred_csv_path = os.path.join(experiment_dir, "predictions.csv")
    predictions_df.to_csv(pred_csv_path, index=False)

    # 6. Export Confusion Matrices
    cm_raw = confusion_matrix(test_labels_idx, y_pred_idx)
    cm_norm = confusion_matrix(test_labels_idx, y_pred_idx, normalize='true')
    plot_and_save_confusion_matrices(cm_raw, cm_norm, class_names, experiment_dir)

    cm_df = pd.DataFrame(cm_raw, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(experiment_dir, "confusion_matrix.csv"))

    # 7. Export Classification Reports
    clr_dict = classification_report(test_labels_idx, y_pred_idx, target_names=class_names, output_dict=True, zero_division=0)
    clr_txt = classification_report(test_labels_idx, y_pred_idx, target_names=class_names, zero_division=0)

    with open(os.path.join(experiment_dir, "classification_report.txt"), "w") as f:
        f.write(clr_txt)

    with open(os.path.join(experiment_dir, "classification_report.json"), "w") as f:
        json.dump(clr_dict, f, indent=4)

    clr_df = pd.DataFrame(clr_dict).transpose()
    clr_df.to_csv(os.path.join(experiment_dir, "classification_report.csv"))

    # 8. Export ROC & Precision-Recall Curves
    roc_auc_scores = plot_and_save_roc_curves(y_true_onehot, y_prob, class_names, experiment_dir)
    pd.DataFrame(list(roc_auc_scores.items()), columns=["Class/Average", "ROC_AUC"]).to_csv(os.path.join(experiment_dir, "roc_auc_scores.csv"), index=False)
    plot_and_save_precision_recall_curves(y_true_onehot, y_prob, class_names, experiment_dir)

    # 9. Export Confidence Histogram
    plot_and_save_confidence_distribution(confidences, correctness, experiment_dir)

    # 10. Error Analysis & Misclassified Image Thumbnails
    misclassified_dir = os.path.join(experiment_dir, "misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)

    error_indices = np.where(~correctness)[0]
    error_list = []

    for err_idx in error_indices:
        src_path = test_paths[err_idx]
        true_cls = idx_to_class[test_labels_idx[err_idx]]
        pred_cls = idx_to_class[y_pred_idx[err_idx]]
        conf = confidences[err_idx]

        img_name = os.path.basename(src_path)
        dest_filename = f"{true_cls}_{pred_cls}_{img_name}"
        dest_path = os.path.join(misclassified_dir, dest_filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)

        error_list.append({
            "image_name": img_name,
            "true_class": true_cls,
            "predicted_class": pred_cls,
            "confidence": conf,
            "is_uncertain": bool(conf < Config.CONFIDENCE_THRESHOLD),
            "saved_thumbnail": dest_path
        })

    pd.DataFrame(error_list).to_csv(os.path.join(experiment_dir, "error_analysis.csv"), index=False)

    # 11. Performance & Model Info JSON Exports
    perf_data = {
        "total_test_images": len(test_paths),
        "total_inference_time_seconds": round(total_inference_time, 4),
        "avg_inference_time_per_image_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2)
    }
    with open(os.path.join(experiment_dir, "performance.json"), "w") as f:
        json.dump(perf_data, f, indent=4)

    model_info = profile_model_information(model, model_path)
    with open(os.path.join(experiment_dir, "model_information.json"), "w") as f:
        json.dump(model_info, f, indent=4)

    # 12. Full Evaluation Summary JSON
    summary_data = {
        "model_index": model_index,
        "model_name": folder_name,
        "test_accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "matthews_corrcoef": round(float(mcc), 4),
        "expected_calibration_error": round(float(ece), 4),
        "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
        "uncertain_prediction_count": num_uncertain,
        "uncertain_prediction_percentage": round(uncertain_ratio * 100.0, 2),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "inference_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
        "model_size_mb": model_info["model_file_size_mb"],
        "roc_auc_scores": roc_auc_scores
    }

    with open(os.path.join(experiment_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    logger.info(f"[EVALUATION SUCCESS] Model #{model_index} ({folder_name}) evaluated successfully!")
    logger.info(f"  - Test Accuracy:          {acc * 100:.2f}%")
    logger.info(f"  - Macro F1-Score:         {f1_macro:.4f}")
    logger.info(f"  - MCC Score:              {mcc:.4f}")
    logger.info(f"  - ECE Calibration Error:  {ece:.4f}")
    logger.info(f"  - Low-Confidence Flagged: {num_uncertain} samples ({uncertain_ratio*100:.1f}%) < {Config.CONFIDENCE_THRESHOLD*100:.0f}%")
    logger.info(f"  - Latency:                {avg_latency_ms:.2f} ms/image ({throughput_fps:.1f} FPS)")

    return summary_data


# ---------------------------------------------------------
# 7. Global Cross-Model Comparison & Ranking Engine
# ---------------------------------------------------------
def evaluate_all_models() -> pd.DataFrame:
    """
    Evaluates all trained models, generates comparison tables/Excel files,
    creates comparative bar graphs, and ranks models automatically inside Experiments/Comparison/.
    """
    comparison_dir = Config.get_comparison_dir()
    logger.info(f"\n=======================================================================")
    logger.info(f"  RUNNING GLOBAL CROSS-MODEL COMPARISON & RANKING ENGINE")
    logger.info(f"=======================================================================")

    results = []
    for model_index in range(1, len(Config.MODEL_MAP) + 1):
        try:
            res = evaluate_single_model(model_index)
            results.append(res)
        except Exception as e:
            msg = f"Skipping Model #{model_index} evaluation due to error: {str(e)}"
            logger.warning(msg)
            exc_logger.error(msg)

    if not results:
        logger.error("[COMPARISON ERROR] No trained models found to evaluate.")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Calculate Global Ranking Score based on Accuracy, F1-macro, MCC, Latency, and Size
    df['rank_score'] = (
        df['test_accuracy'] * 0.35 +
        df['f1_macro'] * 0.35 +
        df['matthews_corrcoef'] * 0.20 -
        (df['inference_latency_ms'] / df['inference_latency_ms'].max()) * 0.05 -
        (df['model_size_mb'] / df['model_size_mb'].max()) * 0.05
    )
    df = df.sort_values(by='rank_score', ascending=False).reset_index(drop=True)
    df['overall_rank'] = range(1, len(df) + 1)

    # Save Comparison Tables
    csv_path = os.path.join(comparison_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False)

    try:
        excel_path = os.path.join(comparison_dir, "comparison_metrics.xlsx")
        df.to_excel(excel_path, index=False)
    except Exception:
        pass

    ranking_df = df[['overall_rank', 'model_name', 'test_accuracy', 'f1_macro', 'matthews_corrcoef', 'inference_latency_ms', 'model_size_mb', 'uncertain_prediction_percentage']]
    ranking_df.to_csv(os.path.join(comparison_dir, "model_ranking.csv"), index=False)

    with open(os.path.join(comparison_dir, "model_ranking.json"), "w") as f:
        json.dump(ranking_df.to_dict(orient="records"), f, indent=4)

    # Generate Publication-Quality Comparative Bar Charts
    metrics_to_plot = [
        ("test_accuracy", "Test Accuracy", "accuracy_comparison.png", "#2b5c8f"),
        ("f1_macro", "Macro F1-Score", "f1_score_comparison.png", "#81b29a"),
        ("matthews_corrcoef", "Matthews Correlation Coefficient (MCC)", "mcc_comparison.png", "#e07a5f"),
        ("inference_latency_ms", "Inference Latency (ms/image)", "inference_latency_comparison.png", "#f4a261"),
        ("model_size_mb", "Model Checkpoint Size (MB)", "model_size_comparison.png", "#9b59b6")
    ]

    for col, title, filename, color in metrics_to_plot:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x="model_name", y=col, color=color)
        plt.title(f'Cross-Model Benchmarking - {title}', fontsize=14, fontweight='bold')
        plt.xlabel('Model Architecture', fontsize=11, fontweight='bold')
        plt.ylabel(title, fontsize=11, fontweight='bold')
        plt.xticks(rotation=25)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, filename), dpi=300)
        plt.close()

    logger.info(f"\n[GLOBAL COMPARISON COMPLETE] All models evaluated & ranked successfully!")
    logger.info(f"Top Performing Model: {df.iloc[0]['model_name']} (Accuracy: {df.iloc[0]['test_accuracy']*100:.2f}%, F1: {df.iloc[0]['f1_macro']:.4f})")
    logger.info(f"Comparison report saved to: {comparison_dir}")

    return df
