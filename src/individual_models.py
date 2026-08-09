"""
Cashew Pest and Disease Diagnosis System
Phase 6: Individual Model Complete Dataset Classification Engine (TensorFlow / Keras)

Evaluates all 8 trained vision models (MobileNetV2 through ConvNeXtTiny) independently on the complete
5,734-image dataset (Train + Validation + Test) with fast parallel PIL image validation, vectorised
batched inference, per-class breakdown, split-wise comparisons, and global 8-model benchmark reporting.
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
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, balanced_accuracy_score, cohen_kappa_score,
    matthews_corrcoef
)

from src.config import Config
from src.utils import set_seed, get_logger
from src.evaluate import (
    parse_inference_image, build_inference_dataset, compute_expected_calibration_error,
    plot_and_save_confusion_matrices, profile_model_information
)
from src.ensemble import validate_image_file_fast

# Configure dedicated Phase 6 loggers
phase6_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("Phase6IndividualModels", phase6_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)


# ---------------------------------------------------------
# 1. Dataset Loader & Parallel Safety Validator
# ---------------------------------------------------------
def load_full_dataset_dataframe() -> Tuple[pd.DataFrame, Dict, float]:
    """
    Concatenates train_split.csv, val_split.csv, and test_split.csv from Preprocessed/,
    deduplicates file paths, and validates all images in parallel using ThreadPoolExecutor.
    Returns (valid_df, dataset_stats, val_elapsed_time).
    """
    preprocessed_dir = Config.get_preprocessed_dir()
    splits_to_load = ["train", "val", "test"]
    split_dfs = []

    train_count = 0
    val_count = 0
    test_count = 0

    for s_name in splits_to_load:
        c_path = os.path.join(preprocessed_dir, f"{s_name}_split.csv")
        if not os.path.exists(c_path):
            err_msg = f"Cannot run Phase 6 complete dataset evaluation. Missing split file: {c_path}"
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

    combined_df = pd.concat(split_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["file_path"]).reset_index(drop=True)
    total_unique_images = len(combined_df)

    logger.info(f"[PHASE 6 DATASET] Train: {train_count}, Val: {val_count}, Test: {test_count} | Total Unique: {total_unique_images}")

    # Fast parallel validation using ThreadPoolExecutor
    val_start_time = time.time()
    logger.info(f"Validating {total_unique_images} dataset images using parallel ThreadPoolExecutor...")

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
            
            if (idx + 1) % 500 == 0 or (idx + 1) == total_unique_images:
                logger.info(f"[VALIDATION] Checked {idx + 1} / {total_unique_images} images")

    val_elapsed_time = time.time() - val_start_time
    valid_mask = np.array(valid_mask)
    valid_df = combined_df[valid_mask].reset_index(drop=True)
    invalid_count = int(np.sum(~valid_mask))
    valid_count = len(valid_df)

    # Save invalid_images.log to root Phase 6 output directory
    root_output_dir = Config.get_individual_models_full_dataset_dir()
    invalid_log_path = os.path.join(root_output_dir, "invalid_images.log")
    with open(invalid_log_path, "w") as f:
        f.write(f"PHASE 6 COMPLETE DATASET IMAGE VALIDATION REPORT\n")
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

    logger.info(f"Image Validation Complete -> Valid: {valid_count}, Invalid: {invalid_count} (Took {val_elapsed_time:.2f}s)")

    dataset_stats = {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "total_unique_images": total_unique_images,
        "valid_count": valid_count,
        "invalid_count": invalid_count
    }

    return valid_df, dataset_stats, val_elapsed_time


# ---------------------------------------------------------
# 2. Evaluate Single Model on Complete Dataset
# ---------------------------------------------------------
def evaluate_single_model_full_dataset(model_index: int, valid_df: Optional[pd.DataFrame] = None, dataset_stats: Optional[Dict] = None) -> Dict:
    """
    Evaluates a single model from MODEL_MAP[model_index] on the complete 5,734-image dataset.
    Exports full predictions CSV, per-class breakdown, split-wise comparison, summary JSON,
    confusion matrices, classification reports, and misclassified image thumbnails.
    """
    set_seed(Config.SEED)
    folder_name, model_key = Config.MODEL_MAP[model_index]
    model_output_dir = Config.get_individual_models_full_dataset_dir(model_index)
    checkpoint_path = os.path.join(Config.get_base_dir(), "Experiments", folder_name, "best_model.keras")

    print(f"\n==================================================")
    print(f"MODEL {model_index:02d}: {folder_name}")
    print(f"==================================================")

    if not os.path.exists(checkpoint_path):
        err_msg = f"Model checkpoint missing at: {checkpoint_path}. Train the model before running Phase 6."
        exc_logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    # 1. Load Dataset if not supplied
    val_elapsed_time = 0.0
    if valid_df is None or dataset_stats is None:
        valid_df, dataset_stats, val_elapsed_time = load_full_dataset_dataframe()

    valid_paths = valid_df["file_path"].tolist()
    valid_splits = valid_df["split"].tolist()
    valid_count = len(valid_df)

    # Determine class mapping
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

    num_classes = len(class_names)

    # 2. Load Model
    logger.info(f"Loading Keras checkpoint for '{folder_name}' from: {checkpoint_path}")
    try:
        model = keras.models.load_model(checkpoint_path, compile=False)
    except Exception as e:
        logger.warning(f"Standard load failed ({e}). Loading with safe_mode=False...")
        model = keras.models.load_model(checkpoint_path, compile=False, safe_mode=False)

    # 3. Vectorised Batched Inference using Fresh Dataset
    batch_size = Config.BATCH_SIZE
    print(f"\nRunning inference on {valid_count} images for {folder_name} (Batch Size = {batch_size})...")
    logger.info(f"Running inference on {valid_count} images for {folder_name}...")

    inf_start_time = time.time()
    model_ds = build_inference_dataset(valid_paths, batch_size=batch_size)
    y_prob = model.predict(model_ds, verbose=1)
    inf_elapsed_time = time.time() - inf_start_time

    print("Inference complete.")

    y_prob = np.array(y_prob)
    y_pred_idx = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    correctness = (y_pred_idx == valid_labels)

    # Apply 80% Uncertainty Threshold
    is_uncertain_mask = confidences < Config.CONFIDENCE_THRESHOLD
    num_uncertain = int(np.sum(is_uncertain_mask))
    uncertain_ratio = float(num_uncertain / valid_count)

    system_responses = [
        f"Diagnosed: {idx_to_class[pred_idx]} (Confidence: {conf*100:.1f}%)" if conf >= Config.CONFIDENCE_THRESHOLD
        else Config.UNCERTAIN_PREDICTION_MESSAGE
        for pred_idx, conf in zip(y_pred_idx, confidences)
    ]

    # 4. Compute Overall Metrics
    total_correct = int(np.sum(correctness))
    total_incorrect = valid_count - total_correct
    overall_accuracy = (total_correct / valid_count) * 100.0

    print(f"Correct: {total_correct} / {valid_count}")
    print(f"Accuracy: {overall_accuracy:.2f}%\n")

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(valid_labels, y_pred_idx, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(valid_labels, y_pred_idx, average='weighted', zero_division=0)
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(valid_labels, y_pred_idx, average=None, zero_division=0)

    # 5. Export full_dataset_predictions.csv
    pred_df = pd.DataFrame({
        "split": valid_splits,
        "image_path": valid_paths,
        "actual_class": [idx_to_class[i] for i in valid_labels],
        "predicted_class": [idx_to_class[i] for i in y_pred_idx],
        "confidence_score": np.round(confidences, 4),
        "is_uncertain": is_uncertain_mask,
        "is_correct": correctness,
        "user_display_message": system_responses
    })
    pred_df.to_csv(os.path.join(model_output_dir, "full_dataset_predictions.csv"), index=False)

    # 6. Export full_dataset_per_class_results.csv & Format Console Breakdown
    per_class_rows = []
    per_class_dict = {}
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
            "formatted_result": f"{cls_name}:\nCorrect: {correct_cls} / {total_cls}\nAccuracy: {acc_cls:.2f}%\n"
        })

        per_class_dict[cls_name] = {
            "correct": correct_cls,
            "total": total_cls,
            "accuracy": round(acc_cls, 2)
        }

    per_class_df = pd.DataFrame(per_class_rows)
    per_class_df.drop(columns=["formatted_result"]).to_csv(os.path.join(model_output_dir, "full_dataset_per_class_results.csv"), index=False)

    for r in per_class_rows:
        print(r["formatted_result"])

    # 7. Export Separate Split Results CSV (full_dataset_split_results.csv)
    split_results_rows = []
    split_accuracies_dict = {}

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

        split_accuracies_dict[split_label] = round(s_acc, 2)

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
    split_accuracies_dict["Complete Dataset"] = round(overall_accuracy, 2)

    split_results_df = pd.DataFrame(split_results_rows)
    split_results_df.to_csv(os.path.join(model_output_dir, "full_dataset_split_results.csv"), index=False)

    # 8. Export full_dataset_summary.json
    model_info = profile_model_information(model, checkpoint_path)
    summary_data = {
        "model_index": model_index,
        "model_name": folder_name,
        "pipeline_phase": "Phase 6 - Individual Model Complete Dataset Classification",
        "dataset_counts": dataset_stats,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "overall_accuracy": round(overall_accuracy, 2),
        "total_uncertain": num_uncertain,
        "uncertain_percentage": round(uncertain_ratio * 100.0, 2),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "inference_time_seconds": round(inf_elapsed_time, 2),
        "per_class_results": per_class_dict,
        "split_wise_accuracies": split_accuracies_dict,
        "model_info": model_info,
        "methodological_notice": "Complete Dataset Descriptive Classification. Official unbiased performance remains the untouched Phase 4 Test set evaluation."
    }

    with open(os.path.join(model_output_dir, "full_dataset_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # 9. Confusion Matrices
    cm_raw = confusion_matrix(valid_labels, y_pred_idx)
    cm_norm = confusion_matrix(valid_labels, y_pred_idx, normalize='true')
    plot_and_save_confusion_matrices(cm_raw, cm_norm, class_names, model_output_dir)

    shutil.copy2(os.path.join(model_output_dir, "confusion_matrix.png"), os.path.join(model_output_dir, "full_dataset_confusion_matrix.png"))
    shutil.copy2(os.path.join(model_output_dir, "confusion_matrix_normalized.png"), os.path.join(model_output_dir, "full_dataset_confusion_matrix_normalized.png"))
    pd.DataFrame(cm_raw, index=class_names, columns=class_names).to_csv(os.path.join(model_output_dir, "full_dataset_confusion_matrix.csv"))

    # 10. Classification Reports
    clr_dict = classification_report(valid_labels, y_pred_idx, target_names=class_names, output_dict=True, zero_division=0)
    with open(os.path.join(model_output_dir, "full_dataset_classification_report.json"), "w") as f:
        json.dump(clr_dict, f, indent=4)

    pd.DataFrame(clr_dict).transpose().to_csv(os.path.join(model_output_dir, "full_dataset_classification_report.csv"))

    # 11. Misclassified Thumbnails
    misclassified_dir = os.path.join(model_output_dir, "full_dataset_misclassified_images")
    os.makedirs(misclassified_dir, exist_ok=True)

    error_indices = np.where(~correctness)[0]
    for err_idx in error_indices:
        src_path = valid_paths[err_idx]
        true_cls = idx_to_class[valid_labels[err_idx]]
        pred_cls = idx_to_class[y_pred_idx[err_idx]]

        img_name = os.path.basename(src_path)
        dest_path = os.path.join(misclassified_dir, f"{true_cls}_{pred_cls}_{img_name}")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)

    print("==================================================")
    print("IMPORTANT METHODOLOGICAL REQUIREMENT:")
    print('Clearly label this result as: "Complete Dataset Descriptive Classification"')
    print("because Train images were used during model training.")
    print("Do NOT call complete-dataset accuracy the unbiased test accuracy.")
    print("The official unbiased performance remains the untouched Test Set evaluation.")
    print("==================================================\n")

    return summary_data


# ---------------------------------------------------------
# 3. Global 8-Model Comparison & Benchmarking Engine
# ---------------------------------------------------------
def evaluate_all_models_full_dataset() -> pd.DataFrame:
    """
    Runs complete dataset classification for all 8 vision architectures, builds the global
    8_model_complete_dataset_comparison table (CSV/XLSX), generates comparison bar charts,
    and prints formatted benchmark progress.
    """
    phase6_start_time = time.time()
    root_output_dir = Config.get_individual_models_full_dataset_dir()

    logger.info(f"\n=======================================================================")
    logger.info(f"  PHASE 6: ALL 8 INDIVIDUAL MODELS COMPLETE DATASET CLASSIFICATION")
    logger.info(f"=======================================================================")

    # 1. Load Dataset once
    valid_df, dataset_stats, val_elapsed_time = load_full_dataset_dataframe()
    valid_count = dataset_stats["valid_count"]

    all_summaries = []
    comparison_rows = []

    # 2. Evaluate all 8 models
    for model_index in range(1, len(Config.MODEL_MAP) + 1):
        folder_name, _ = Config.MODEL_MAP[model_index]
        try:
            summary = evaluate_single_model_full_dataset(model_index, valid_df=valid_df, dataset_stats=dataset_stats)
            all_summaries.append(summary)

            per_cls = summary["per_class_results"]
            splits_acc = summary["split_wise_accuracies"]

            row = {
                "Model": folder_name,
                "Overall Correct": summary["total_correct"],
                "Overall Total": valid_count,
                "Overall Accuracy (%)": summary["overall_accuracy"],
                "Train Accuracy (%)": splits_acc["Train"],
                "Validation Accuracy (%)": splits_acc["Validation"],
                "Test Accuracy (%)": splits_acc["Test"],
                "Complete Dataset Accuracy (%)": splits_acc["Complete Dataset"]
            }

            for cls_name, cls_result in per_cls.items():
                safe_name = cls_name.replace(" ", "_")
                row[f"{safe_name} Correct"] = cls_result["correct"]
                row[f"{safe_name} Total"] = cls_result["total"]
                row[f"{safe_name} Accuracy (%)"] = cls_result["accuracy"]

            comparison_rows.append(row)
        except Exception as e:
            msg = f"Skipping Model #{model_index} ({folder_name}) due to error: {str(e)}"
            logger.warning(msg)
            exc_logger.error(msg)

    if not comparison_rows:
        logger.error("[PHASE 6 ERROR] No models evaluated successfully.")
        return pd.DataFrame()

    comp_df = pd.DataFrame(comparison_rows)

    # 3. Export Global Comparison CSV & XLSX
    csv_path = os.path.join(root_output_dir, "8_model_complete_dataset_comparison.csv")
    comp_df.to_csv(csv_path, index=False)

    try:
        excel_path = os.path.join(root_output_dir, "8_model_complete_dataset_comparison.xlsx")
        comp_df.to_excel(excel_path, index=False)
    except Exception:
        pass

    # 4. Generate Publication-Quality Comparison Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_df, x="Model", y="Complete Dataset Accuracy (%)", palette="Blues_d")
    plt.title("Phase 6: Complete Dataset Descriptive Accuracy Across All 8 Vision Models", fontsize=14, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=11, fontweight='bold')
    plt.ylabel("Complete Dataset Accuracy (%)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=25)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(root_output_dir, "8_model_accuracy_comparison.png"), dpi=300)
    plt.close()

    total_phase6_time = time.time() - phase6_start_time

    # 5. Formatted Summary Console Report
    print(f"\n=======================================================================")
    print(f"  PHASE 6 GLOBAL 8-MODEL COMPLETE DATASET COMPARISON SUMMARY")
    print(f"=======================================================================")
    summary_table = comp_df[["Model", "Train Accuracy (%)", "Validation Accuracy (%)", "Test Accuracy (%)", "Complete Dataset Accuracy (%)"]]
    print(summary_table.to_string(index=False))

    print(f"\n[PHASE 6 TIMING STATS]")
    print(f"  - Validation Time    : {val_elapsed_time:.2f}s")
    print(f"  - Total Phase 6 Time : {total_phase6_time:.2f}s")
    print(f"Reports saved to: {root_output_dir}\n")

    logger.info(f"Phase 6 Complete Dataset Classification finished in {total_phase6_time:.2f} seconds.")
    return comp_df
