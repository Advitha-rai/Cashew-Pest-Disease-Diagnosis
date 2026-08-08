"""
Cashew Pest and Disease Diagnosis System
Phase 5: Production-Quality Soft-Voting Ensemble Engine (TensorFlow / Keras)

Combines predictions from VGG16, DenseNet121, and ConvNeXtTiny using probability-level
soft voting with validation-based constrained weight search and full evaluation reporting.
Includes fast PIL-based parallel image validation, 80% confidence thresholding, and Complete Dataset
descriptive classification reporting across Train + Validation + Test splits.
"""

import os
import sys
import time
import json
import shutil
import logging
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
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
from src.evaluate import (
    parse_inference_image, build_inference_dataset, compute_expected_calibration_error,
    plot_and_save_confusion_matrices, plot_and_save_roc_curves,
    plot_and_save_precision_recall_curves, plot_and_save_confidence_distribution,
    profile_model_information
)

# Configure dedicated ensemble loggers
ensemble_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("EnsembleEngine", ensemble_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)


# ---------------------------------------------------------
# 1. Fast Parallel Image Validation & Domain Safety Profiler
# ---------------------------------------------------------
def validate_image_file_fast(image_path: str) -> Tuple[bool, str]:
    """
    Fast, lightweight image validator using PIL/Pillow.
    Checks file existence, readability, non-zero byte size, valid extensions,
    image decodability, minimum dimensions, and RGB channel conversion.
    Eliminates expensive per-image TensorFlow graph and reduce_std() calls.
    """
    if not os.path.exists(image_path):
        return False, f"File does not exist: {image_path}"

    if os.path.getsize(image_path) == 0:
        return False, "File is zero bytes."

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
    if not image_path.lower().endswith(valid_extensions):
        return False, f"Unsupported file extension. Expected one of {valid_extensions}."

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width < 10 or height < 10:
                return False, f"Image dimensions too small: {width}x{height}."
            img.verify()

        # Re-open after verify() to test RGB conversion
        with Image.open(image_path) as img:
            img.convert("RGB")

        return True, "Valid image"

    except Exception as e:
        return False, f"Failed to decode image data with PIL ({str(e)})."


def validate_image_file(image_path: str) -> Tuple[bool, str]:
    """Alias for fast PIL-based image validation."""
    return validate_image_file_fast(image_path)


# ---------------------------------------------------------
# 2. Ensemble Model Loader
# ---------------------------------------------------------
def load_ensemble_models() -> Tuple[Dict[str, keras.Model], List[str]]:
    """
    Loads the 3 selected model checkpoints (VGG16, DenseNet121, ConvNeXtTiny) safely.
    Returns dictionary of loaded models and list of model keys.
    """
    selected_indices = Config.ENSEMBLE_MODEL_INDICES
    models = {}
    model_keys = []

    logger.info(f"Loading Phase 5 Ensemble Sub-Models (Indices: {selected_indices})...")

    for idx in selected_indices:
        folder_name, model_key = Config.MODEL_MAP[idx]
        checkpoint_path = os.path.join(Config.get_base_dir(), "Experiments", folder_name, "best_model.keras")

        if not os.path.exists(checkpoint_path):
            err_msg = f"Ensemble model checkpoint missing at: {checkpoint_path}. Train the model before building ensemble."
            exc_logger.error(err_msg)
            raise FileNotFoundError(err_msg)

        logger.info(f"Loading checkpoint for '{folder_name}' from: {checkpoint_path}")
        try:
            model = keras.models.load_model(checkpoint_path, compile=False)
        except Exception as e:
            logger.warning(f"Standard load failed ({e}). Retrying with safe_mode=False...")
            model = keras.models.load_model(checkpoint_path, compile=False, safe_mode=False)

        models[folder_name] = model
        model_keys.append(folder_name)

    logger.info(f"[ENSEMBLE LOAD SUCCESS] Successfully loaded {len(models)} models: {list(models.keys())}")
    return models, model_keys


# ---------------------------------------------------------
# 3. Dataset Loader for Validation and Test Splits
# ---------------------------------------------------------
def load_split_dataset_csv(split_name: str) -> Dict:
    """
    Loads split dataset directly from Preprocessed/<split_name>_split.csv.
    """
    preprocessed_dir = Config.get_preprocessed_dir()
    csv_path = os.path.join(preprocessed_dir, f"{split_name}_split.csv")

    if not os.path.exists(csv_path):
        err_msg = f"Required dataset split file not found at: {csv_path}"
        exc_logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info(f"Loading '{split_name}' split dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    paths = df["file_path"].tolist()

    if "class_name" in df.columns:
        unique_classes = sorted(df["class_name"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        idx_to_class = {i: c for i, c in enumerate(unique_classes)}
        
        if "label" in df.columns:
            labels_idx = df["label"].values
        else:
            labels_idx = np.array([class_to_idx[c] for c in df["class_name"]])
    else:
        unique_classes = Config.DEFAULT_CLASSES
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        idx_to_class = {i: c for i, c in enumerate(unique_classes)}
        labels_idx = df["label"].values

    return {
        "paths": paths,
        "labels": labels_idx,
        "class_names": unique_classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }


# ---------------------------------------------------------
# 4. Validation-Based Ensemble Weight Optimization Engine
# ---------------------------------------------------------
def validate_and_search_ensemble_weights() -> Dict:
    """
    Loads the validation split (val_split.csv), runs batched inference for all 3 models,
    evaluates baseline equal weights (1/3, 1/3, 1/3), and performs a grid search for optimal
    non-negative ensemble weights summing to 1. Primary criterion = Validation Macro F1.
    Saves ensemble_weights.json and validation_results.csv/json.
    """
    set_seed(Config.SEED)
    ensemble_dir = Config.get_ensemble_dir()
    
    logger.info(f"\n=======================================================================")
    logger.info(f"  PHASE 5: VALIDATION-BASED ENSEMBLE WEIGHT SELECTION")
    logger.info(f"=======================================================================")

    # 1. Load Validation Dataset
    val_data = load_split_dataset_csv("val")
    val_paths = val_data["paths"]
    val_labels = val_data["labels"]
    class_names = val_data["class_names"]

    logger.info(f"Loaded {len(val_paths)} validation samples across classes: {class_names}")

    # 2. Load Ensemble Models
    models, model_names = load_ensemble_models()
    batch_size = Config.BATCH_SIZE

    # 3. Generate Batched Predictions for each model on Validation Set (Fresh Dataset per model)
    val_probs = {}
    for name in model_names:
        logger.info(f"Generating validation predictions for sub-model '{name}'...")
        print(f"\nGenerating validation predictions for sub-model '{name}'...")
        val_ds = build_inference_dataset(val_paths, batch_size=batch_size)
        probs = models[name].predict(val_ds, verbose=1)
        val_probs[name] = np.array(probs)

    # 4. Evaluate Baseline Equal Weights (1/3, 1/3, 1/3)
    equal_w = {name: 1.0 / len(model_names) for name in model_names}
    equal_p = sum(equal_w[name] * val_probs[name] for name in model_names)
    equal_preds = np.argmax(equal_p, axis=1)

    equal_acc = accuracy_score(val_labels, equal_preds)
    equal_p_mac, equal_r_mac, equal_f1_mac, _ = precision_recall_fscore_support(val_labels, equal_preds, average='macro', zero_division=0)
    equal_mcc = matthews_corrcoef(val_labels, equal_preds)

    logger.info(f"Baseline Equal-Weight Ensemble (Validation) -> Accuracy: {equal_acc*100:.2f}%, Macro F1: {equal_f1_mac:.4f}, MCC: {equal_mcc:.4f}")

    # 5. Constrained Grid Search over Non-Negative Weights (Step 0.05, Sum = 1.0)
    best_f1 = -1.0
    best_acc = -1.0
    best_weights = equal_w.copy()
    search_records = []

    step = 0.05
    steps = int(round(1.0 / step)) + 1
    grid_vals = [round(i * step, 2) for i in range(steps)]

    for w1 in grid_vals:
        for w2 in grid_vals:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < 0:
                continue
            if not np.isclose(w1 + w2 + w3, 1.0):
                continue

            weights_dict = {
                model_names[0]: w1,
                model_names[1]: w2,
                model_names[2]: w3
            }

            combo_p = w1 * val_probs[model_names[0]] + w2 * val_probs[model_names[1]] + w3 * val_probs[model_names[2]]
            combo_preds = np.argmax(combo_p, axis=1)

            acc = accuracy_score(val_labels, combo_preds)
            _, _, f1_mac, _ = precision_recall_fscore_support(val_labels, combo_preds, average='macro', zero_division=0)
            mcc = matthews_corrcoef(val_labels, combo_preds)

            search_records.append({
                "w_VGG16": w1,
                "w_DenseNet121": w2,
                "w_ConvNeXtTiny": w3,
                "val_accuracy": round(acc, 4),
                "val_macro_f1": round(f1_mac, 4),
                "val_mcc": round(mcc, 4)
            })

            # Primary criterion: Macro F1, Secondary criterion: Accuracy
            if (f1_mac > best_f1) or (np.isclose(f1_mac, best_f1) and acc > best_acc):
                best_f1 = f1_mac
                best_acc = acc
                best_weights = weights_dict

    logger.info(f"\n[WEIGHT SEARCH COMPLETE]")
    logger.info(f"Optimal Ensemble Weights selected on Validation Set:")
    for name, w in best_weights.items():
        logger.info(f"  - {name}: {w:.2f}")
    logger.info(f"Validation Metric Achievements -> Macro F1: {best_f1:.4f}, Accuracy: {best_acc*100:.2f}%")

    # 6. Save Weight & Validation Artifacts
    weights_json_path = os.path.join(ensemble_dir, "ensemble_weights.json")
    weights_export = {
        "selected_weights": {name: float(w) for name, w in best_weights.items()},
        "baseline_equal_weights": {name: float(w) for name, w in equal_w.items()},
        "selection_dataset": "validation",
        "selection_metric": "macro_f1",
        "validation_samples": len(val_paths),
        "validation_baseline_macro_f1": round(float(equal_f1_mac), 4),
        "validation_optimal_macro_f1": round(float(best_f1), 4),
        "validation_baseline_accuracy": round(float(equal_acc), 4),
        "validation_optimal_accuracy": round(float(best_acc), 4)
    }
    with open(weights_json_path, "w") as f:
        json.dump(weights_export, f, indent=4)

    pd.DataFrame(search_records).to_csv(os.path.join(ensemble_dir, "validation_results.csv"), index=False)
    with open(os.path.join(ensemble_dir, "validation_results.json"), "w") as f:
        json.dump(weights_export, f, indent=4)

    return weights_export


# ---------------------------------------------------------
# 5. Final Ensemble Test Evaluation & Benchmarking Engine
# ---------------------------------------------------------
def evaluate_ensemble_test_set() -> Dict:
    """
    Evaluates the validation-optimized soft-voting ensemble on the untouched test set (test_split.csv).
    Exports full metrics, confusion matrices, ROC/PR curves, per-class counts, model comparisons,
    misclassified image thumbnails, and summary artifacts to Experiments/Ensemble/.
    """
    set_seed(Config.SEED)
    ensemble_dir = Config.get_ensemble_dir()

    logger.info(f"\n=======================================================================")
    logger.info(f"  PHASE 5: FINAL ENSEMBLE TEST SET EVALUATION")
    logger.info(f"=======================================================================")

    # 1. Load Ensemble Weights
    weights_json_path = os.path.join(ensemble_dir, "ensemble_weights.json")
    if not os.path.exists(weights_json_path):
        logger.warning(f"Ensemble weights file missing at {weights_json_path}. Running validation search first...")
        weights_info = validate_and_search_ensemble_weights()
    else:
        with open(weights_json_path, "r") as f:
            weights_info = json.load(f)

    weights = weights_info["selected_weights"]
    equal_weights = weights_info["baseline_equal_weights"]
    logger.info(f"Loaded Ensemble Weights for Test Evaluation: {weights}")

    # 2. Load Test Dataset Split
    test_data = load_split_dataset_csv("test")
    test_paths = test_data["paths"]
    test_labels = np.array(test_data["labels"])
    class_names = test_data["class_names"]
    idx_to_class = test_data["idx_to_class"]
    num_classes = len(class_names)

    y_true_onehot = tf.one_hot(test_labels, depth=num_classes).numpy()

    # 3. Load Models and Run Batched Inference on Test Set (Fresh Dataset per model)
    models, model_names = load_ensemble_models()
    batch_size = Config.BATCH_SIZE

    test_probs = {}
    individual_summaries = {}
    
    start_time = time.time()
    for name in model_names:
        logger.info(f"Generating test predictions for sub-model '{name}'...")
        print(f"\nGenerating test predictions for sub-model '{name}'...")
        sub_start = time.time()
        test_ds = build_inference_dataset(test_paths, batch_size=batch_size)
        probs = models[name].predict(test_ds, verbose=1)
        sub_time = time.time() - sub_start
        test_probs[name] = np.array(probs)

        # Individual model metrics on test set
        sub_preds = np.argmax(test_probs[name], axis=1)
        sub_acc = accuracy_score(test_labels, sub_preds)
        _, _, sub_f1, _ = precision_recall_fscore_support(test_labels, sub_preds, average='macro', zero_division=0)
        sub_mcc = matthews_corrcoef(test_labels, sub_preds)
        sub_unc = float(np.mean(np.max(test_probs[name], axis=1) < Config.CONFIDENCE_THRESHOLD) * 100.0)

        # Profile size
        checkpoint_path = os.path.join(Config.get_base_dir(), "Experiments", name, "best_model.keras")
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024) if os.path.exists(checkpoint_path) else 0.0

        individual_summaries[name] = {
            "accuracy": round(float(sub_acc), 4),
            "macro_f1": round(float(sub_f1), 4),
            "mcc": round(float(sub_mcc), 4),
            "uncertain_pct": round(sub_unc, 2),
            "latency_ms": round((sub_time / len(test_paths)) * 1000.0, 2),
            "model_size_mb": round(size_mb, 2)
        }

    total_inference_time = time.time() - start_time
    avg_latency_ms = (total_inference_time / len(test_paths)) * 1000.0
    throughput_fps = len(test_paths) / total_inference_time

    # 4. Compute Equal-Weight Baseline Test Predictions
    equal_p = sum(equal_weights[name] * test_probs[name] for name in model_names)
    equal_preds = np.argmax(equal_p, axis=1)
    equal_acc = accuracy_score(test_labels, equal_preds)
    _, _, equal_f1, _ = precision_recall_fscore_support(test_labels, equal_preds, average='macro', zero_division=0)
    equal_mcc = matthews_corrcoef(test_labels, equal_preds)
    equal_unc = float(np.mean(np.max(equal_p, axis=1) < Config.CONFIDENCE_THRESHOLD) * 100.0)

    # 5. Compute Final Validation-Weighted Soft-Voting Ensemble Probabilities
    ensemble_probs = sum(weights[name] * test_probs[name] for name in model_names)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    confidences = np.max(ensemble_probs, axis=1)
    correctness = (ensemble_preds == test_labels)

    # Apply 80% Uncertainty Threshold
    is_uncertain_mask = confidences < Config.CONFIDENCE_THRESHOLD
    num_uncertain = int(np.sum(is_uncertain_mask))
    uncertain_ratio = float(num_uncertain / len(test_paths))

    system_responses = [
        f"Diagnosed: {idx_to_class[pred_idx]} (Confidence: {conf*100:.1f}%)" if conf >= Config.CONFIDENCE_THRESHOLD
        else Config.UNCERTAIN_PREDICTION_MESSAGE
        for pred_idx, conf in zip(ensemble_preds, confidences)
    ]

    # 6. Compute Standard Classification Metrics for Ensemble
    acc = accuracy_score(test_labels, ensemble_preds)
    bal_acc = balanced_accuracy_score(test_labels, ensemble_preds)
    kappa = cohen_kappa_score(test_labels, ensemble_preds)
    mcc = matthews_corrcoef(test_labels, ensemble_preds)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(test_labels, ensemble_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(test_labels, ensemble_preds, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(test_labels, ensemble_preds, average=None, zero_division=0)

    ece, bin_accs, bin_confs, bin_sizes = compute_expected_calibration_error(test_labels, ensemble_probs)

    # 7. Compute Per-Class Correct / Total Analysis
    per_class_rows = []
    for i, cls_name in enumerate(class_names):
        cls_mask = (test_labels == i)
        total_cls = int(np.sum(cls_mask))
        correct_cls = int(np.sum(correctness & cls_mask))
        acc_cls = (correct_cls / total_cls) * 100.0 if total_cls > 0 else 0.0

        per_class_rows.append({
            "class_name": cls_name,
            "correct_samples": correct_cls,
            "total_samples": total_cls,
            "accuracy_pct": round(acc_cls, 2),
            "formatted_result": f"{cls_name}: {correct_cls}/{total_cls} = {acc_cls:.2f}%",
            "precision": round(float(precision_per_class[i]), 4),
            "recall": round(float(recall_per_class[i]), 4),
            "f1_score": round(float(f1_per_class[i]), 4)
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.to_csv(os.path.join(ensemble_dir, "ensemble_per_class_results.csv"), index=False)

    logger.info(f"\n--- ENSEMBLE PER-CLASS BREAKDOWN ---")
    for r in per_class_rows:
        logger.info(f"  {r['formatted_result']}")

    # 8. Export Prediction Details CSV
    predictions_df = pd.DataFrame({
        "image_path": test_paths,
        "true_label": [idx_to_class[i] for i in test_labels],
        "top_predicted_label": [idx_to_class[i] for i in ensemble_preds],
        "ensemble_confidence": np.round(confidences, 4),
        "is_uncertain": is_uncertain_mask,
        "user_display_message": system_responses,
        "correctness": correctness,
        "VGG16_prob": [np.round(p, 4).tolist() for p in test_probs["03_VGG16"]],
        "DenseNet121_prob": [np.round(p, 4).tolist() for p in test_probs["05_DenseNet121"]],
        "ConvNeXtTiny_prob": [np.round(p, 4).tolist() for p in test_probs["08_ConvNeXtTiny"]],
        "ensemble_prob": [np.round(p, 4).tolist() for p in ensemble_probs]
    })
    predictions_df.to_csv(os.path.join(ensemble_dir, "test_predictions.csv"), index=False)

    # 9. Export Confusion Matrices & Plots
    cm_raw = confusion_matrix(test_labels, ensemble_preds)
    cm_norm = confusion_matrix(test_labels, ensemble_preds, normalize='true')
    plot_and_save_confusion_matrices(cm_raw, cm_norm, class_names, ensemble_dir)

    # Rename saved confusion matrix filenames to match Phase 5 prefix requirement
    shutil.copy2(os.path.join(ensemble_dir, "confusion_matrix.png"), os.path.join(ensemble_dir, "test_confusion_matrix.png"))
    shutil.copy2(os.path.join(ensemble_dir, "confusion_matrix_normalized.png"), os.path.join(ensemble_dir, "test_confusion_matrix_normalized.png"))
    pd.DataFrame(cm_raw, index=class_names, columns=class_names).to_csv(os.path.join(ensemble_dir, "test_confusion_matrix.csv"))

    # 10. Export Classification Reports
    clr_dict = classification_report(test_labels, ensemble_preds, target_names=class_names, output_dict=True, zero_division=0)
    clr_txt = classification_report(test_labels, ensemble_preds, target_names=class_names, zero_division=0)

    with open(os.path.join(ensemble_dir, "test_classification_report.txt"), "w") as f:
        f.write(clr_txt)

    with open(os.path.join(ensemble_dir, "test_classification_report.json"), "w") as f:
        json.dump(clr_dict, f, indent=4)

    pd.DataFrame(clr_dict).transpose().to_csv(os.path.join(ensemble_dir, "test_classification_report.csv"))

    # 11. Export Curves & Plots
    roc_auc_scores = plot_and_save_roc_curves(y_true_onehot, ensemble_probs, class_names, ensemble_dir)
    shutil.copy2(os.path.join(ensemble_dir, "roc_curve.png"), os.path.join(ensemble_dir, "test_roc_curve.png"))
    pd.DataFrame(list(roc_auc_scores.items()), columns=["Class/Average", "ROC_AUC"]).to_csv(os.path.join(ensemble_dir, "test_roc_auc_scores.csv"), index=False)

    plot_and_save_precision_recall_curves(y_true_onehot, ensemble_probs, class_names, ensemble_dir)
    shutil.copy2(os.path.join(ensemble_dir, "precision_recall_curve.png"), os.path.join(ensemble_dir, "test_precision_recall_curve.png"))

    plot_and_save_confidence_distribution(confidences, correctness, ensemble_dir)
    shutil.copy2(os.path.join(ensemble_dir, "confidence_distribution.png"), os.path.join(ensemble_dir, "test_confidence_distribution.png"))

    # 12. Error Analysis & Misclassified Thumbnails
    misclassified_dir = os.path.join(ensemble_dir, "misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)

    error_indices = np.where(~correctness)[0]
    error_list = []
    for err_idx in error_indices:
        src_path = test_paths[err_idx]
        true_cls = idx_to_class[test_labels[err_idx]]
        pred_cls = idx_to_class[ensemble_preds[err_idx]]
        conf = confidences[err_idx]

        img_name = os.path.basename(src_path)
        dest_path = os.path.join(misclassified_dir, f"{true_cls}_{pred_cls}_{img_name}")
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
    pd.DataFrame(error_list).to_csv(os.path.join(ensemble_dir, "error_analysis.csv"), index=False)

    # 13. Performance JSON & Full Summary JSON
    total_model_size_mb = sum(ind["model_size_mb"] for ind in individual_summaries.values())
    perf_data = {
        "total_test_images": len(test_paths),
        "total_inference_time_seconds": round(total_inference_time, 4),
        "avg_inference_time_per_image_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
        "total_ensemble_model_size_mb": round(total_model_size_mb, 2)
    }
    with open(os.path.join(ensemble_dir, "ensemble_performance.json"), "w") as f:
        json.dump(perf_data, f, indent=4)

    summary_data = {
        "pipeline_phase": "Phase 5 - Soft Voting Ensemble",
        "ensemble_weights": weights,
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
        "per_class_results": {r["class_name"]: r["formatted_result"] for r in per_class_rows},
        "inference_latency_ms": round(avg_latency_ms, 2),
        "throughput_fps": round(throughput_fps, 2),
        "total_model_size_mb": round(total_model_size_mb, 2),
        "roc_auc_scores": roc_auc_scores
    }
    with open(os.path.join(ensemble_dir, "ensemble_evaluation_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # 14. Cross-Model Comparison Table Against Individual Models
    comparison_rows = [
        {
            "Model / Architecture": "03_VGG16",
            "Accuracy (%)": round(individual_summaries["03_VGG16"]["accuracy"] * 100, 2),
            "Macro F1": individual_summaries["03_VGG16"]["macro_f1"],
            "MCC": individual_summaries["03_VGG16"]["mcc"],
            "Uncertainty (%)": individual_summaries["03_VGG16"]["uncertain_pct"],
            "Latency (ms)": individual_summaries["03_VGG16"]["latency_ms"],
            "Model Size (MB)": individual_summaries["03_VGG16"]["model_size_mb"]
        },
        {
            "Model / Architecture": "05_DenseNet121",
            "Accuracy (%)": round(individual_summaries["05_DenseNet121"]["accuracy"] * 100, 2),
            "Macro F1": individual_summaries["05_DenseNet121"]["macro_f1"],
            "MCC": individual_summaries["05_DenseNet121"]["mcc"],
            "Uncertainty (%)": individual_summaries["05_DenseNet121"]["uncertain_pct"],
            "Latency (ms)": individual_summaries["05_DenseNet121"]["latency_ms"],
            "Model Size (MB)": individual_summaries["05_DenseNet121"]["model_size_mb"]
        },
        {
            "Model / Architecture": "08_ConvNeXtTiny",
            "Accuracy (%)": round(individual_summaries["08_ConvNeXtTiny"]["accuracy"] * 100, 2),
            "Macro F1": individual_summaries["08_ConvNeXtTiny"]["macro_f1"],
            "MCC": individual_summaries["08_ConvNeXtTiny"]["mcc"],
            "Uncertainty (%)": individual_summaries["08_ConvNeXtTiny"]["uncertain_pct"],
            "Latency (ms)": individual_summaries["08_ConvNeXtTiny"]["latency_ms"],
            "Model Size (MB)": individual_summaries["08_ConvNeXtTiny"]["model_size_mb"]
        },
        {
            "Model / Architecture": "Equal-weight Ensemble",
            "Accuracy (%)": round(equal_acc * 100, 2),
            "Macro F1": round(float(equal_f1), 4),
            "MCC": round(float(equal_mcc), 4),
            "Uncertainty (%)": round(equal_unc, 2),
            "Latency (ms)": round(avg_latency_ms, 2),
            "Model Size (MB)": round(total_model_size_mb, 2)
        },
        {
            "Model / Architecture": "Validation-Weighted Ensemble (Optimal)",
            "Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(float(f1_macro), 4),
            "MCC": round(float(mcc), 4),
            "Uncertainty (%)": round(uncertain_ratio * 100.0, 2),
            "Latency (ms)": round(avg_latency_ms, 2),
            "Model Size (MB)": round(total_model_size_mb, 2)
        }
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(ensemble_dir, "model_comparison.csv"), index=False)
    try:
        comp_df.to_excel(os.path.join(ensemble_dir, "model_comparison.xlsx"), index=False)
    except Exception:
        pass

    logger.info(f"\n=======================================================================")
    logger.info(f"  FINAL TEST ENSEMBLE EVALUATION RESULTS")
    logger.info(f"=======================================================================")
    logger.info(f"  - Test Accuracy:          {acc * 100:.2f}%")
    logger.info(f"  - Macro F1-Score:         {f1_macro:.4f}")
    logger.info(f"  - MCC Score:              {mcc:.4f}")
    logger.info(f"  - ECE Calibration Error:  {ece:.4f}")
    logger.info(f"  - Low-Confidence Flagged: {num_uncertain} samples ({uncertain_ratio*100:.1f}%) < 80%")

    best_single_name = "03_VGG16"
    best_single_acc = individual_summaries[best_single_name]["accuracy"]
    if acc > best_single_acc:
        logger.info(f"  [SUMMARY] Validation-Weighted Ensemble IMPROVED performance over best single model ({best_single_name}: {best_single_acc*100:.2f}% -> Ensemble: {acc*100:.2f}%).")
    else:
        logger.info(f"  [SUMMARY] Validation-Weighted Ensemble achieved competitive performance ({acc*100:.2f}% vs {best_single_name}: {best_single_acc*100:.2f}%).")

    return summary_data


# ---------------------------------------------------------
# 6. Single Image Ensemble Inference API
# ---------------------------------------------------------
def predict_ensemble_single_image(image_path: str, models: Optional[Dict[str, keras.Model]] = None, weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Performs end-to-end soft-voting ensemble inference on a single image.
    Includes input validation protection and 80% confidence uncertainty thresholding.
    """
    class_names = Config.DEFAULT_CLASSES

    # 1. Input Image Validation
    is_valid, val_msg = validate_image_file(image_path)
    if not is_valid:
        return {
            "status": "INVALID_IMAGE",
            "predicted_class": None,
            "confidence_score": 0.0,
            "is_uncertain": True,
            "user_display_message": Config.INVALID_IMAGE_MESSAGE,
            "error_details": val_msg
        }

    # 2. Load Models and Weights if not supplied
    if models is None:
        models, model_names = load_ensemble_models()
    else:
        model_names = list(models.keys())

    if weights is None:
        ensemble_dir = Config.get_ensemble_dir()
        weights_path = os.path.join(ensemble_dir, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights = json.load(f)["selected_weights"]
        else:
            weights = {name: 1.0 / len(model_names) for name in model_names}

    # 3. Preprocess Image
    img = parse_inference_image(image_path)
    img_batch = tf.expand_dims(img, axis=0)

    # 4. Generate Predictions from each model
    sub_probs = {}
    for name in model_names:
        preds = models[name](img_batch, training=False).numpy()[0]
        sub_probs[name] = preds

    # 5. Calculate Soft-Voting Combined Ensemble Probabilities
    ensemble_prob = sum(weights[name] * sub_probs[name] for name in model_names)
    top_class_idx = int(np.argmax(ensemble_prob))
    confidence = float(ensemble_prob[top_class_idx])
    top_class_name = class_names[top_class_idx]

    # 6. Apply 80% Uncertainty Threshold
    is_uncertain = confidence < Config.CONFIDENCE_THRESHOLD

    if is_uncertain:
        user_display_message = Config.UNCERTAIN_PREDICTION_MESSAGE
        predicted_class_result = "Prediction Uncertain"
    else:
        user_display_message = f"Diagnosed: {top_class_name} (Confidence: {confidence * 100:.1f}%)"
        predicted_class_result = top_class_name

    return {
        "status": "SUCCESS",
        "predicted_class": predicted_class_result,
        "confidence_score": round(confidence, 4),
        "is_uncertain": is_uncertain,
        "user_display_message": user_display_message,
        "model_probabilities": {name: [round(float(p), 4) for p in sub_probs[name]] for name in model_names},
        "ensemble_probabilities": [round(float(p), 4) for p in ensemble_prob]
    }


# ---------------------------------------------------------
# 7. Complete Dataset Descriptive Classification Reporting
# ---------------------------------------------------------
def evaluate_full_dataset() -> Dict:
    """
    Combines train_split.csv, val_split.csv, and test_split.csv into a unified dataset,
    runs vectorised batched ensemble inference on every valid unique image, and exports
    complete dataset reports, confusion matrices, per-class correct/total breakdowns,
    split-wise comparisons, misclassified thumbnails, and invalid image logs to:
    Experiments/Ensemble/Full_Dataset_Classification/
    Uses fast parallel ThreadPoolExecutor image validation to avoid Colab freezing.
    """
    pipeline_start_time = time.time()
    set_seed(Config.SEED)
    output_dir = Config.get_full_dataset_classification_dir()
    ensemble_dir = Config.get_ensemble_dir()

    logger.info(f"\n=======================================================================")
    logger.info(f"  PHASE 5: COMPLETE DATASET CLASSIFICATION REPORT")
    logger.info(f"=======================================================================")

    # 1. Verify and Load Train, Validation, and Test CSV Files
    preprocessed_dir = Config.get_preprocessed_dir()
    splits_to_load = ["train", "val", "test"]
    split_dfs = []

    train_count = 0
    val_count = 0
    test_count = 0

    for s_name in splits_to_load:
        c_path = os.path.join(preprocessed_dir, f"{s_name}_split.csv")
        if not os.path.exists(c_path):
            err_msg = f"Cannot run complete dataset evaluation. Missing split file: {c_path}"
            exc_logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        
        df = pd.read_csv(c_path)
        split_display = "Train" if s_name == "train" else ("Validation" if s_name == "val" else "Test")
        df["split"] = split_display

        if s_name == "train":
            train_count = len(df)
        elif s_name == "val":
            val_count = len(df)
        elif s_name == "test":
            test_count = len(df)

        split_dfs.append(df)

    # 2. Concatenate and Deduplicate File Paths
    combined_df = pd.concat(split_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    total_unique_images = len(combined_df)

    logger.info(f"Dataset Concatenation Summary -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
    logger.info(f"Total Unique Images after deduplication: {total_unique_images}")

    # 3. Fast Parallel Image Validation & Invalid Logging (ThreadPoolExecutor)
    val_start_time = time.time()
    logger.info(f"Validating {total_unique_images} dataset images using parallel ThreadPoolExecutor...")
    print(f"\nValidating {total_unique_images} dataset images using parallel ThreadPoolExecutor...")

    all_paths = combined_df["file_path"].tolist()
    valid_mask = [False] * total_unique_images
    invalid_logs = []

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(validate_image_file_fast, img_p) for img_p in all_paths]
        for idx, future in enumerate(futures):
            is_val, reason = future.result()
            valid_mask[idx] = is_val
            if not is_val:
                invalid_logs.append(f"[INVALID IMAGE #{idx+1}] Path: '{all_paths[idx]}' | Reason: {reason}")
            
            # Progress logging every 500 images
            if (idx + 1) % 500 == 0 or (idx + 1) == total_unique_images:
                print(f"[VALIDATION] Checked {idx + 1} / {total_unique_images} images")
                logger.info(f"[VALIDATION] Checked {idx + 1} / {total_unique_images} images")

    val_elapsed_time = time.time() - val_start_time
    valid_mask = np.array(valid_mask)
    valid_df = combined_df[valid_mask].reset_index(drop=True)
    invalid_count = int(np.sum(~valid_mask))
    valid_count = len(valid_df)

    print(f"Image Validation Complete -> Valid: {valid_count}, Invalid: {invalid_count} (Took {val_elapsed_time:.2f}s)")
    logger.info(f"[VALIDATION TIME] Image Validation Complete -> Valid: {valid_count}, Invalid: {invalid_count} in {val_elapsed_time:.2f} seconds")

    # Write invalid_images.log
    invalid_log_path = os.path.join(output_dir, "invalid_images.log")
    with open(invalid_log_path, "w") as f:
        f.write(f"COMPLETE DATASET IMAGE VALIDATION REPORT\n")
        f.write(f"Total Input Images : {total_unique_images}\n")
        f.write(f"Valid Images       : {valid_count}\n")
        f.write(f"Invalid Images     : {invalid_count}\n")
        f.write(f"Validation Time    : {val_elapsed_time:.2f} seconds\n")
        f.write(f"=" * 60 + "\n\n")
        if invalid_logs:
            for log_entry in invalid_logs:
                f.write(log_entry + "\n")
        else:
            f.write("No invalid images detected. All dataset images passed validation.\n")

    if valid_count == 0:
        err_msg = "Zero valid images found in complete dataset."
        exc_logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Determine class mappings
    if "class_name" in valid_df.columns:
        class_names = sorted(valid_df["class_name"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        idx_to_class = {i: c for i, c in enumerate(class_names)}
        
        if "label" in valid_df.columns:
            valid_labels = valid_df["label"].values
        else:
            valid_labels = np.array([class_to_idx[c] for c in valid_df["class_name"]])
    else:
        class_names = Config.DEFAULT_CLASSES
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        idx_to_class = {i: c for i, c in enumerate(class_names)}
        valid_labels = valid_df["label"].values

    valid_paths = valid_df["file_path"].tolist()
    valid_splits = valid_df["split"].tolist()
    num_classes = len(class_names)
    y_true_onehot = tf.one_hot(valid_labels, depth=num_classes).numpy()

    # 4. Load Models & Finalized Ensemble Weights
    models, model_names = load_ensemble_models()
    weights_path = os.path.join(ensemble_dir, "ensemble_weights.json")
    if not os.path.exists(weights_path):
        logger.warning("Ensemble weights missing. Running validation search first...")
        weights_info = validate_and_search_ensemble_weights()
    else:
        with open(weights_path, "r") as f:
            weights_info = json.load(f)

    weights = weights_info["selected_weights"]
    logger.info(f"Loaded finalized ensemble weights for complete dataset classification: {weights}")

    # 5. Batched Vectorised Inference on Complete Dataset (Fresh dataset per model)
    batch_size = Config.BATCH_SIZE
    full_probs = {}
    inf_start_time = time.time()

    for name in model_names:
        logger.info(f"Running complete dataset batched inference for sub-model '{name}'...")
        print(f"\nRunning complete dataset batched inference for sub-model '{name}'...")
        model_ds = build_inference_dataset(valid_paths, batch_size=batch_size)
        probs = models[name].predict(model_ds, verbose=1)
        full_probs[name] = np.array(probs)

    inf_elapsed_time = time.time() - inf_start_time
    total_pipeline_time = time.time() - pipeline_start_time

    logger.info(f"[MODEL INFERENCE TIME] Batched inference completed across all 3 models in {inf_elapsed_time:.2f} seconds")
    logger.info(f"[TOTAL PIPELINE TIME] Complete Dataset Classification finished in {total_pipeline_time:.2f} seconds")

    avg_latency_ms = (inf_elapsed_time / valid_count) * 1000.0
    throughput_fps = valid_count / inf_elapsed_time

    # 6. Soft-Voting Combination & Confidence Thresholding
    ensemble_probs = sum(weights[name] * full_probs[name] for name in model_names)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    confidences = np.max(ensemble_probs, axis=1)
    correctness = (ensemble_preds == valid_labels)

    is_uncertain_mask = confidences < Config.CONFIDENCE_THRESHOLD
    num_uncertain = int(np.sum(is_uncertain_mask))
    uncertain_ratio = float(num_uncertain / valid_count)

    system_responses = [
        f"Diagnosed: {idx_to_class[pred_idx]} (Confidence: {conf*100:.1f}%)" if conf >= Config.CONFIDENCE_THRESHOLD
        else Config.UNCERTAIN_PREDICTION_MESSAGE
        for pred_idx, conf in zip(ensemble_preds, confidences)
    ]

    # 7. Compute Classification Metrics
    total_correct = int(np.sum(correctness))
    total_incorrect = valid_count - total_correct
    overall_accuracy = (total_correct / valid_count) * 100.0

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(valid_labels, ensemble_preds, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(valid_labels, ensemble_preds, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(valid_labels, ensemble_preds, average=None, zero_division=0)

    # 8. Export full_dataset_predictions.csv
    pred_df = pd.DataFrame({
        "split": valid_splits,
        "image_path": valid_paths,
        "actual_class": [idx_to_class[i] for i in valid_labels],
        "predicted_class": [idx_to_class[i] for i in ensemble_preds],
        "confidence_score": np.round(confidences, 4),
        "is_uncertain": is_uncertain_mask,
        "is_correct": correctness,
        "user_display_message": system_responses
    })
    pred_df.to_csv(os.path.join(output_dir, "full_dataset_predictions.csv"), index=False)

    # 9. Export full_dataset_per_class_results.csv
    per_class_rows = []
    for i, cls_name in enumerate(class_names):
        cls_mask = (valid_labels == i)
        total_cls = int(np.sum(cls_mask))
        correct_cls = int(np.sum(correctness & cls_mask))
        incorrect_cls = total_cls - correct_cls
        acc_cls = (correct_cls / total_cls) * 100.0 if total_cls > 0 else 0.0
        
        unc_cls = int(np.sum(is_uncertain_mask & cls_mask))
        unc_pct_cls = (unc_cls / total_cls) * 100.0 if total_cls > 0 else 0.0

        per_class_rows.append({
            "Actual Class": cls_name,
            "Correct": correct_cls,
            "Incorrect": incorrect_cls,
            "Total": total_cls,
            "Accuracy": round(acc_cls, 2),
            "Uncertain Count": unc_cls,
            "Uncertain Percentage": round(unc_pct_cls, 2),
            "formatted_result": f"{cls_name:<13}: Correct: {correct_cls} / {total_cls} = {acc_cls:.2f}%"
        })

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.drop(columns=["formatted_result"]).to_csv(os.path.join(output_dir, "full_dataset_per_class_results.csv"), index=False)

    # 10. Export Separate Split Results CSV (full_dataset_split_results.csv)
    split_results_rows = []
    for split_label in ["Train", "Validation", "Test"]:
        s_mask = (valid_df["split"] == split_label).values
        s_total = int(np.sum(s_mask))
        if s_total > 0:
            s_corr = int(np.sum(correctness & s_mask))
            s_inc = s_total - s_corr
            s_acc = (s_corr / s_total) * 100.0
            s_unc = int(np.sum(is_uncertain_mask & s_mask))
            s_unc_pct = (s_unc / s_total) * 100.0
        else:
            s_corr = s_inc = s_acc = s_unc = s_unc_pct = 0.0

        split_results_rows.append({
            "split": split_label,
            "total_images": s_total,
            "correct": s_corr,
            "incorrect": s_inc,
            "accuracy": round(s_acc, 2),
            "uncertain_count": s_unc,
            "uncertain_percentage": round(s_unc_pct, 2)
        })

    # Complete dataset summary row
    split_results_rows.append({
        "split": "Complete Dataset",
        "total_images": valid_count,
        "correct": total_correct,
        "incorrect": total_incorrect,
        "accuracy": round(overall_accuracy, 2),
        "uncertain_count": num_uncertain,
        "uncertain_percentage": round(uncertain_ratio * 100.0, 2)
    })

    split_results_df = pd.DataFrame(split_results_rows)
    split_results_df.to_csv(os.path.join(output_dir, "full_dataset_split_results.csv"), index=False)

    # 11. Export full_dataset_summary.json
    summary_data = {
        "pipeline_phase": "Phase 5 - Complete Dataset Descriptive Classification",
        "total_input_images": total_unique_images,
        "valid_images": valid_count,
        "invalid_images": invalid_count,
        "train_count": train_count,
        "validation_count": val_count,
        "test_count": test_count,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "overall_accuracy": round(overall_accuracy, 2),
        "total_uncertain": num_uncertain,
        "uncertain_percentage": round(uncertain_ratio * 100.0, 2),
        "timing_stats": {
            "validation_time_seconds": round(val_elapsed_time, 2),
            "inference_time_seconds": round(inf_elapsed_time, 2),
            "total_pipeline_time_seconds": round(total_pipeline_time, 2)
        },
        "per_class_results": {r["Actual Class"]: f"{r['Correct']} / {r['Total']} = {r['Accuracy']:.2f}%" for r in per_class_rows},
        "split_wise_accuracies": {r["split"]: f"{r['accuracy']:.2f}%" for r in split_results_rows}
    }

    with open(os.path.join(output_dir, "full_dataset_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # 12. Confusion Matrices
    cm_raw = confusion_matrix(valid_labels, ensemble_preds)
    cm_norm = confusion_matrix(valid_labels, ensemble_preds, normalize='true')
    plot_and_save_confusion_matrices(cm_raw, cm_norm, class_names, output_dir)

    shutil.copy2(os.path.join(output_dir, "confusion_matrix.png"), os.path.join(output_dir, "full_dataset_confusion_matrix.png"))
    shutil.copy2(os.path.join(output_dir, "confusion_matrix_normalized.png"), os.path.join(output_dir, "full_dataset_confusion_matrix_normalized.png"))
    pd.DataFrame(cm_raw, index=class_names, columns=class_names).to_csv(os.path.join(output_dir, "full_dataset_confusion_matrix.csv"))

    # 13. Classification Reports
    clr_dict = classification_report(valid_labels, ensemble_preds, target_names=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(output_dir, "full_dataset_classification_report.json"), "w") as f:
        json.dump(clr_dict, f, indent=4)

    pd.DataFrame(clr_dict).transpose().to_csv(os.path.join(output_dir, "full_dataset_classification_report.csv"))

    # 14. Misclassified Thumbnails
    misclassified_dir = os.path.join(output_dir, "full_dataset_misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)

    error_indices = np.where(~correctness)[0]
    for err_idx in error_indices:
        src_path = valid_paths[err_idx]
        true_cls = idx_to_class[valid_labels[err_idx]]
        pred_cls = idx_to_class[ensemble_preds[err_idx]]

        img_name = os.path.basename(src_path)
        dest_path = os.path.join(misclassified_dir, f"{true_cls}_{pred_cls}_{img_name}")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)

    # 15. Formatted Console Output
    console_summary = f"""
==================================================
COMPLETE DATASET CLASSIFICATION
==================================================

Train images       : {train_count}
Validation images  : {val_count}
Test images        : {test_count}
Total unique images: {total_unique_images}

Valid images       : {valid_count}
Invalid images     : {invalid_count}

Correct predictions: {total_correct}
Incorrect          : {total_incorrect}

Overall accuracy   : {overall_accuracy:.2f}%
"""
    for r in per_class_rows:
        console_summary += f"\n{r['Actual Class']}:\nCorrect: {r['Correct']} / {r['Total']}\nAccuracy: {r['Accuracy']:.2f}%\n"

    console_summary += f"""
==================================================
SPLIT-WISE ACCURACY BREAKDOWN
==================================================
"""
    for s_row in split_results_rows:
        console_summary += f"{s_row['split']:<18}: {s_row['correct']} / {s_row['total_images']} = {s_row['accuracy']:.2f}%\n"

    console_summary += f"""
[TIMING STATS]
  - Validation Time    : {val_elapsed_time:.2f}s
  - Inference Time     : {inf_elapsed_time:.2f}s
  - Total Pipeline Time: {total_pipeline_time:.2f}s

==================================================
IMPORTANT METHODOLOGICAL REQUIREMENT:

Clearly label this result as:
"Complete Dataset Descriptive Classification"
because Train images were used during model training.

Do NOT call complete-dataset accuracy the unbiased test accuracy.
The official unbiased performance remains the untouched Test Set evaluation.
==================================================
"""
    print(console_summary)
    logger.info(console_summary)

    return summary_data
