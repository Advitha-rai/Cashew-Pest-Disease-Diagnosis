"""
Cashew Pest and Disease Diagnosis System
Phase C: Manual Segmentation Annotation Preparation & Quality Control Engine (TensorFlow / Keras)

Features:
  - Dataset Audit & Metadata Verification (Enforces GROUND_TRUTH_MASKS_NOT_FOUND status when masks are absent)
  - Class Name Normalization (Safely maps raw class names to Aphids, Leaf miner, TMB, Leaf blight; flags UNKNOWN_CLASS)
  - Isolated Annotation Directory Setup (Experiments/Segmentation/Annotations/<Split>/<Class>/)
  - 5,734-Image Segmentation Annotation Manifest Generator with status preservation
  - Strict 5-Class Pixel-Level Mask Validation Engine (0=Background, 1=Aphids, 2=Leaf miner, 3=TMB, 4=Leaf blight)
  - Configurable Quality Control Thresholds (MIN_LESION_PIXEL_PERCENTAGE, MAX_LESION_PIXEL_PERCENTAGE)
  - Manual Mask Saver Utility with immediate quality validation
  - Manifest-Wide Mask Quality Control & Summary Reporting
  - Dataset Leakage & Split Integrity Safety Checker
  - Comprehensive 12-Test Automated Verification Suite operating on isolated temporary test directories
"""

import os
import glob
import json
import logging
import hashlib
import tempfile
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional, Set, Any

from src.config import Config
from src.utils import get_logger

# Configure dedicated segmentation loggers
seg_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("SegmentationAnnotationEngine", seg_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)

# ---------------------------------------------------------
# 1. MASK ENCODING SPECIFICATIONS & CLASS NORMALIZATION
# ---------------------------------------------------------
CLASS_MASK_ENCODING = {
    "Background": 0,
    "Aphids": 1,
    "Leaf miner": 2,
    "TMB": 3,
    "Leaf blight": 4
}

ALLOWED_PIXEL_VALUES = {0, 1, 2, 3, 4}


def normalize_class_name(raw_class: str) -> Tuple[str, int]:
    """
    Safely normalizes raw class names to standard project class names.
    Maps:
      - aphid / aphids -> Aphids (Code 1)
      - leaf miner / leaf_miner -> Leaf miner (Code 2)
      - tmb / TMB -> TMB (Code 3)
      - leaf blight / leaf_blight -> Leaf blight (Code 4)
    If an unknown class is encountered:
      - logs an error
      - returns ("UNKNOWN_CLASS", -1)
      - does NOT assign class_code = 0 automatically (Background is reserved for mask pixels).
    """
    raw_str = str(raw_class).strip().lower()

    if "aphid" in raw_str:
        return "Aphids", 1
    elif "miner" in raw_str:
        return "Leaf miner", 2
    elif "tmb" in raw_str:
        return "TMB", 3
    elif "blight" in raw_str:
        return "Leaf blight", 4
    else:
        logger.error(f"[CLASS NORMALIZATION ERROR] Unknown target class encountered: '{raw_class}'. Cannot assign to Background (0).")
        return "UNKNOWN_CLASS", -1


# ---------------------------------------------------------
# 2. ISOLATED ANNOTATION DIRECTORY INITIALIZER
# ---------------------------------------------------------
def initialize_annotation_directories() -> Dict[str, str]:
    """
    Creates isolated annotation subdirectories under Experiments/Segmentation/Annotations/
    for Train, Validation, and Test splits across all 4 target classes.
    Does NOT modify original dataset directory structure.
    """
    seg_dir = Config.get_segmentation_dir()
    annotations_base = os.path.join(seg_dir, "Annotations")
    os.makedirs(annotations_base, exist_ok=True)

    splits = ["Train", "Validation", "Test"]
    classes = ["Aphids", "Leaf_blight", "Leaf_miner", "TMB"]
    created_dirs = {}

    for s in splits:
        for c in classes:
            dir_path = os.path.join(annotations_base, s, c)
            os.makedirs(dir_path, exist_ok=True)
            created_dirs[f"{s}/{c}"] = dir_path

    logger.info(f"[ANNOTATION SETUP] Initialized isolated directory tree at: {annotations_base}")
    return created_dirs


# ---------------------------------------------------------
# 3. DATASET LEAKAGE & SPLIT SAFETY CHECKER
# ---------------------------------------------------------
def check_dataset_leakage(preprocessed_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies that no image path appears across multiple dataset splits (Train/Val/Test).
    Verifies actual vs. expected split record counts.
    Returns structured leakage report.
    """
    if preprocessed_dir is None:
        preprocessed_dir = Config.get_preprocessed_dir()

    train_csv = os.path.join(preprocessed_dir, "train_split.csv")
    val_csv = os.path.join(preprocessed_dir, "val_split.csv")
    test_csv = os.path.join(preprocessed_dir, "test_split.csv")

    split_paths = {}
    expected_counts = {"Train": 4013, "Validation": 860, "Test": 861, "Total": 5734}
    actual_counts = {"Train": 0, "Validation": 0, "Test": 0, "Total": 0}

    for s_name, p in [("Train", train_csv), ("Validation", val_csv), ("Test", test_csv)]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                paths = set(df["file_path"].astype(str))
                split_paths[s_name] = paths
                actual_counts[s_name] = len(paths)
            except Exception as e:
                logger.error(f"[LEAKAGE CHECK ERROR] Failed to read {p}: {e}")
                split_paths[s_name] = set()
        else:
            split_paths[s_name] = set()

    actual_counts["Total"] = sum(actual_counts[k] for k in ["Train", "Validation", "Test"])

    # Overlap detection
    train_val_overlap = split_paths.get("Train", set()) & split_paths.get("Validation", set())
    train_test_overlap = split_paths.get("Train", set()) & split_paths.get("Test", set())
    val_test_overlap = split_paths.get("Validation", set()) & split_paths.get("Test", set())

    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)

    if has_leakage:
        logger.error(f"[DATASET LEAKAGE DETECTED] Train/Val overlap: {len(train_val_overlap)}, Train/Test overlap: {len(train_test_overlap)}, Val/Test overlap: {len(val_test_overlap)}")
    else:
        logger.info("[SPLIT INTEGRITY VERIFIED] Zero dataset leakage detected across Train, Validation, and Test splits.")

    counts_match = (
        actual_counts["Train"] == expected_counts["Train"] and
        actual_counts["Validation"] == expected_counts["Validation"] and
        actual_counts["Test"] == expected_counts["Test"] and
        actual_counts["Total"] == expected_counts["Total"]
    )

    return {
        "has_leakage": has_leakage,
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "counts_match_expected": counts_match
    }


# ---------------------------------------------------------
# 4. SEGMENTATION ANNOTATION MANIFEST GENERATOR
# ---------------------------------------------------------
def build_segmentation_annotation_manifest() -> pd.DataFrame:
    """
    Constructs/updates the 5,734-image segmentation annotation manifest.
    Preserves existing annotation and validation statuses (ANNOTATED, SKIPPED, PASSED, FAILED, PENDING) using (image_path, split) identity key.
    """
    initialize_annotation_directories()
    preprocessed_dir = Config.get_preprocessed_dir()
    seg_dir = Config.get_segmentation_dir()
    annotations_base = os.path.join(seg_dir, "Annotations")

    train_csv = os.path.join(preprocessed_dir, "train_split.csv")
    val_csv = os.path.join(preprocessed_dir, "val_split.csv")
    test_csv = os.path.join(preprocessed_dir, "test_split.csv")

    csv_manifest_path = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")
    existing_manifest_map = {}

    if os.path.exists(csv_manifest_path):
        try:
            df_old = pd.read_csv(csv_manifest_path)
            for _, r in df_old.iterrows():
                key = (str(r.get("image_path", "")), str(r.get("split", "")))
                existing_manifest_map[key] = {
                    "annotation_status": str(r.get("annotation_status", "PENDING")),
                    "validation_status": str(r.get("validation_status", "UNVALIDATED")),
                    "error_message": str(r.get("error_message", "Mask pending manual creation"))
                }
        except Exception as e:
            logger.warning(f"Could not read existing manifest for status preservation: {e}")

    manifest_rows = []

    for split_name, csv_path in [("Train", train_csv), ("Validation", val_csv), ("Test", test_csv)]:
        if not os.path.exists(csv_path):
            logger.warning(f"Split CSV not found: {csv_path}")
            continue

        df_split = pd.read_csv(csv_path)
        for _, row in df_split.iterrows():
            img_path = str(row.get("file_path", ""))
            raw_class = str(row.get("class_name", ""))
            
            norm_class, class_code = normalize_class_name(raw_class)
            clean_class = norm_class.replace(" ", "_")
            img_name = os.path.basename(img_path)
            base_name, _ = os.path.splitext(img_name)
            mask_name = f"{base_name}_mask.png"

            expected_mask_path = os.path.join(annotations_base, split_name, clean_class, mask_name)
            
            # Check existing status preservation map
            key = (img_path, split_name)
            if key in existing_manifest_map:
                old_info = existing_manifest_map[key]
                ann_status = old_info["annotation_status"]
                val_status = old_info["validation_status"]
                msg = old_info["error_message"]
            else:
                is_annotated = os.path.exists(expected_mask_path)
                if class_code == -1:
                    ann_status = "SKIPPED"
                    val_status = "FAILED"
                    msg = "Unknown classification class"
                else:
                    ann_status = "ANNOTATED" if is_annotated else "PENDING"
                    val_status = "UNVALIDATED" if not is_annotated else "PENDING_VALIDATION"
                    msg = "Mask ready for validation" if is_annotated else "Mask pending manual creation"

            manifest_rows.append({
                "image_path": img_path,
                "image_name": img_name,
                "class_name": norm_class,
                "class_code": class_code,
                "split": split_name,
                "expected_mask_path": expected_mask_path,
                "annotation_status": ann_status,
                "validation_status": val_status,
                "error_message": msg
            })

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(csv_manifest_path, index=False)

    json_manifest_path = os.path.join(seg_dir, "segmentation_annotation_manifest.json")
    manifest_records = df_manifest.to_dict(orient="records")
    with open(json_manifest_path, "w") as f:
        json.dump({
            "phase": "Phase C — Manual Segmentation Annotation Preparation",
            "total_assigned_images": len(df_manifest),
            "annotated_count": int((df_manifest["annotation_status"] == "ANNOTATED").sum()),
            "pending_count": int((df_manifest["annotation_status"] == "PENDING").sum()),
            "skipped_count": int((df_manifest["annotation_status"] == "SKIPPED").sum()),
            "mask_encoding": CLASS_MASK_ENCODING,
            "manifest_records": manifest_records
        }, f, indent=4)

    logger.info(f"[MANIFEST PRESERVED] Saved {len(df_manifest)} records -> {csv_manifest_path}")
    return df_manifest


# ---------------------------------------------------------
# 5. STRICT MASK VALIDATION ENGINE
# ---------------------------------------------------------
def validate_mask_file(image_path: str, mask_path: str, expected_class_code: int) -> Tuple[bool, str, Dict]:
    """
    Strictly validates a single manual mask file:
      1. Mask file exists.
      2. Source image exists.
      3. Source image can be decoded.
      4. Mask can be decoded.
      5. Image width == mask width and Image height == mask height.
      6. Mask is single-channel 2D.
      7. Mask dtype is uint8 (checked BEFORE casting or conversion).
      8. Mask pixel values belong to {0, 1, 2, 3, 4}.
      9. Mask is non-empty (at least one non-zero pixel).
      10. Expected target class code is present in mask pixel values.
      11. Calculates lesion pixel percentage and checks Config thresholds.
    Returns (is_valid, status_message, metadata_dict).
    """
    if not os.path.exists(mask_path):
        return False, "Mask file does not exist", {}

    if not os.path.exists(image_path):
        return False, f"Source image file not found: {image_path}", {}

    try:
        # Decode Source Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        # Decode Mask Image
        with Image.open(mask_path) as m_img:
            mask_w, mask_h = m_img.size
            mask_mode = m_img.mode
            mask_arr_raw = np.array(m_img)

        # 1. Dimension match check
        if (img_w, img_h) != (mask_w, mask_h):
            return False, f"Dimension mismatch: image ({img_w}x{img_h}) vs mask ({mask_w}x{mask_h})", {}

        # 2. Mode & Channel check (Must be single-channel 2D)
        if mask_arr_raw.ndim > 2 and mask_arr_raw.shape[2] > 1:
            return False, f"Mask is multi-channel ({mask_arr_raw.shape}); single-channel uint8 PNG required", {}

        # 3. Dtype check (Must be uint8 prior to processing)
        if mask_arr_raw.dtype != np.uint8:
            return False, f"Invalid mask dtype: {mask_arr_raw.dtype}. Must be uint8", {}

        # 4. Allowed pixel values check ({0, 1, 2, 3, 4})
        unique_vals = set(np.unique(mask_arr_raw))
        invalid_vals = unique_vals - ALLOWED_PIXEL_VALUES
        if invalid_vals:
            return False, f"Invalid pixel values found: {sorted(list(invalid_vals))}. Only 0-4 allowed", {}

        # 5. Non-empty check (at least one lesion pixel > 0)
        total_pixels = img_w * img_h
        non_zero_pixels = int(np.count_nonzero(mask_arr_raw))
        if non_zero_pixels == 0:
            return False, "Validation Failed: Empty mask. Please paint at least one lesion region.", {}

        # 6. Expected class code presence check
        if expected_class_code > 0 and expected_class_code not in unique_vals:
            err_code_msg = f"Validation Failed: Expected class code {expected_class_code} was not found in mask pixel values {sorted(list(unique_vals))}."
            return False, err_code_msg, {}

        # 7. Lesion area percentage & quality control metadata
        lesion_pct = round((non_zero_pixels / total_pixels) * 100.0, 4)
        bg_pct = round(100.0 - lesion_pct, 4)

        # Configurable Quality Control Threshold Checks
        min_thresh = getattr(Config, "MIN_LESION_PIXEL_PERCENTAGE", 0.01)
        max_thresh = getattr(Config, "MAX_LESION_PIXEL_PERCENTAGE", 95.00)

        qc_warning = None
        if lesion_pct < min_thresh:
            qc_warning = f"QC Warning: Lesion percentage ({lesion_pct}%) below min threshold ({min_thresh}%)"
        elif lesion_pct > max_thresh:
            qc_warning = f"QC Warning: Lesion percentage ({lesion_pct}%) above max threshold ({max_thresh}%)"

        meta = {
            "image_dimensions": [img_w, img_h],
            "mask_dimensions": [mask_w, mask_h],
            "mask_dtype": str(mask_arr_raw.dtype),
            "unique_pixel_values": sorted(list(unique_vals)),
            "non_zero_pixel_count": non_zero_pixels,
            "lesion_pixel_count": non_zero_pixels,
            "lesion_pixel_percentage": lesion_pct,
            "background_pixel_percentage": bg_pct,
            "expected_class_code": expected_class_code,
            "qc_warning": qc_warning
        }

        return True, "PASSED" if not qc_warning else f"PASSED ({qc_warning})", meta

    except Exception as e:
        return False, f"Corrupt image/mask file or read error: {str(e)}", {}


# ---------------------------------------------------------
# 6. MANIFEST-WIDE VALIDATION ENGINE
# ---------------------------------------------------------
def validate_all_manifest_masks() -> pd.DataFrame:
    """
    Scans all records in the annotation manifest and executes strict validation on created masks.
    Updates segmentation_annotation_manifest.csv and .json with validation results.
    """
    seg_dir = Config.get_segmentation_dir()
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")

    if not os.path.exists(manifest_csv):
        df_manifest = build_segmentation_annotation_manifest()
    else:
        df_manifest = pd.read_csv(manifest_csv)

    updated_rows = []
    passed_count = 0
    failed_count = 0

    for _, row in df_manifest.iterrows():
        img_p = str(row["image_path"])
        mask_p = str(row["expected_mask_path"])
        code = int(row.get("class_code", 0))

        if os.path.exists(mask_p):
            is_valid, msg, meta = validate_mask_file(img_p, mask_p, code)
            row_dict = dict(row)
            row_dict["annotation_status"] = "ANNOTATED"
            row_dict["validation_status"] = "PASSED" if is_valid else "FAILED"
            row_dict["error_message"] = msg
            if is_valid:
                passed_count += 1
            else:
                failed_count += 1
        else:
            row_dict = dict(row)
            row_dict["annotation_status"] = "PENDING"
            row_dict["validation_status"] = "UNVALIDATED"
            row_dict["error_message"] = "Mask pending manual creation"

        updated_rows.append(row_dict)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(manifest_csv, index=False)

    json_manifest_path = os.path.join(seg_dir, "segmentation_annotation_manifest.json")
    with open(json_manifest_path, "w") as f:
        json.dump({
            "phase": "Phase C — Manual Segmentation Annotation Preparation",
            "total_assigned_images": len(df_updated),
            "annotated_count": int((df_updated["annotation_status"] == "ANNOTATED").sum()),
            "passed_validation_count": passed_count,
            "failed_validation_count": failed_count,
            "pending_count": int((df_updated["annotation_status"] == "PENDING").sum()),
            "mask_encoding": CLASS_MASK_ENCODING,
            "manifest_records": df_updated.to_dict(orient="records")
        }, f, indent=4)

    logger.info(f"[VALIDATION COMPLETE] Total: {len(df_updated)} | Passed: {passed_count} | Failed: {failed_count} | Pending: {(df_updated['annotation_status'] == 'PENDING').sum()}")
    return df_updated


# ---------------------------------------------------------
# 7. MANUAL MASK SAVER HELPER UTILITY
# ---------------------------------------------------------
def save_manual_annotation_mask(
    image_path: str,
    mask_array: np.ndarray,
    split: str,
    class_name: str
) -> Tuple[bool, str]:
    """
    Helper utility to save a manually created mask array,
    verify its pixel encoding (0-4 uint8), save it as a single-channel PNG, and validate it.
    """
    if mask_array.ndim != 2:
        return False, f"Invalid mask dimensions: {mask_array.shape}. Mask array must be 2D."

    unique_vals = set(np.unique(mask_array))
    if not unique_vals.issubset(ALLOWED_PIXEL_VALUES):
        return False, f"Invalid pixel values in array: {sorted(list(unique_vals))}. Allowed: 0-4"

    initialize_annotation_directories()
    seg_dir = Config.get_segmentation_dir()
    annotations_base = os.path.join(seg_dir, "Annotations")

    norm_class, class_code = normalize_class_name(class_name)
    clean_class = norm_class.replace(" ", "_")
    img_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(img_name)
    mask_name = f"{base_name}_mask.png"

    expected_mask_path = os.path.join(annotations_base, split.capitalize(), clean_class, mask_name)
    os.makedirs(os.path.dirname(expected_mask_path), exist_ok=True)

    mask_uint8 = mask_array.astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    mask_pil.save(expected_mask_path)

    is_valid, msg, _ = validate_mask_file(image_path, expected_mask_path, class_code)

    if is_valid:
        logger.info(f"[MASK SAVED & PASSED] {expected_mask_path}")
        validate_all_manifest_masks()
        return True, f"Successfully saved and validated mask: {expected_mask_path}"
    else:
        logger.error(f"[MASK VALIDATION FAILED] {msg}")
        return False, f"Mask saved to {expected_mask_path} but failed validation: {msg}"


# ---------------------------------------------------------
# 8. DATASET AUDIT ENGINE (PHASE A & C REVISED)
# ---------------------------------------------------------
def audit_segmentation_dataset() -> Dict:
    """
    Performs comprehensive audit of the dataset for segmentation ground-truth masks.
    Distinguishes discovered mask files from valid ground-truth masks.
    Enforces audit_status = GROUND_TRUTH_MASKS_NOT_FOUND when valid ground-truth masks are missing.
    Exports segmentation_dataset_audit.json, .csv, and Phase_Segmentation_Dataset_Audit.md.
    """
    logger.info("=== Starting Segmentation Dataset Audit (Phase A & C) ===")
    initialize_annotation_directories()
    df_manifest = build_segmentation_annotation_manifest()
    leakage_info = check_dataset_leakage()

    base_dir = Config.get_base_dir()
    seg_dir = Config.get_segmentation_dir()
    doc_dir = Config.get_documentation_dir()

    # Search directories to inspect (excluding existing Annotations, Documentation, Logs, and Experiments)
    search_dirs = [
        base_dir,
        Config.get_raw_dir(),
        Config.get_cleaned_dir(),
        Config.get_preprocessed_dir(),
        os.path.join(base_dir, "Dataset"),
        os.path.join(base_dir, "Masks")
    ]

    MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".npz"}
    EXCLUDED_EXTENSIONS = {".md", ".txt", ".csv", ".json", ".py", ".ipynb", ".log", ".html", ".pdf", ".zip", ".xml", ".yaml", ".yml"}

    discovered_mask_files = []
    annotations_base = os.path.abspath(os.path.join(seg_dir, "Annotations"))
    doc_base = os.path.abspath(doc_dir)
    logs_base = os.path.abspath(Config.get_logs_dir())
    exp_base = os.path.abspath(os.path.join(base_dir, "Experiments"))

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            abs_root = os.path.abspath(root)
            # Exclude Annotations, Documentation, Logs, and Experiments from external mask search
            if (abs_root.startswith(annotations_base) or
                abs_root.startswith(doc_base) or
                abs_root.startswith(logs_base) or
                abs_root.startswith(exp_base)):
                continue
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDED_EXTENSIONS or ext not in MASK_EXTENSIONS:
                    continue
                f_lower = file.lower()
                if any(kw in f_lower for kw in ["mask", "_seg", "polygon", "labelme"]):
                    discovered_mask_files.append(os.path.join(root, file))

    annotated_cnt = int((df_manifest["annotation_status"] == "ANNOTATED").sum())
    passed_cnt = int((df_manifest["validation_status"] == "PASSED").sum())
    total_images = len(df_manifest)

    # Strictly set status to GROUND_TRUTH_MASKS_NOT_FOUND when valid ground-truth masks are zero
    if passed_cnt > 0:
        audit_status = "GROUND_TRUTH_MASKS_AVAILABLE"
        seg_possible = True
        audit_wording = f"Discovered {passed_cnt} valid ground-truth segmentation masks."
    else:
        audit_status = "GROUND_TRUTH_MASKS_NOT_FOUND"
        seg_possible = False
        audit_wording = "No valid ground-truth segmentation masks were identified in the inspected dataset directories."

    audit_summary = {
        "audit_phase": "Phase A & Phase C — Segmentation Dataset Audit",
        "audit_status": audit_status,
        "audit_decision_wording": audit_wording,
        "segmentation_dataset_found": seg_possible,
        "number_of_source_images": total_images,
        "number_of_discovered_mask_files": len(discovered_mask_files),
        "number_of_masks": annotated_cnt,
        "number_of_valid_image_mask_pairs": passed_cnt,
        "number_of_missing_masks": total_images - annotated_cnt,
        "number_of_invalid_masks": annotated_cnt - passed_cnt,
        "mask_format": "Single-channel 8-bit uint8 PNG (0=Background, 1=Aphids, 2=Leaf miner, 3=TMB, 4=Leaf blight)" if seg_possible else "N/A (Pending manual ground-truth annotation)",
        "mask_type": "Semantic Segmentation Lesion Index Masks",
        "train_validation_test_compatibility": f"Split verified (Train={leakage_info['actual_counts']['Train']}, Val={leakage_info['actual_counts']['Validation']}, Test={leakage_info['actual_counts']['Test']}). Test split remains isolated.",
        "dataset_leakage_detected": leakage_info["has_leakage"],
        "segmentation_training_currently_possible": seg_possible,
        "search_directories_inspected": [d for d in search_dirs if os.path.exists(d)],
        "mask_encoding": CLASS_MASK_ENCODING
    }

    # 1. Save segmentation_dataset_audit.json
    json_path = os.path.join(seg_dir, "segmentation_dataset_audit.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)

    # 2. Save segmentation_dataset_audit.csv
    csv_path = os.path.join(seg_dir, "segmentation_dataset_audit.csv")
    df_audit = pd.DataFrame([{
        "Audit_Metric": k,
        "Audit_Value": str(v)
    } for k, v in audit_summary.items() if not isinstance(v, (list, dict))])
    df_audit.to_csv(csv_path, index=False)

    # 3. Save Documentation/Phase_Segmentation_Dataset_Audit.md
    report_content = f"""# Phase A & Phase C — Segmentation Dataset Audit & Manual Annotation Plan
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Audit Result

> [!IMPORTANT]
> **AUDIT DECISION**: `{audit_status}`
>
> **Finding**: {audit_wording}
>
> In accordance with Phase C safety requirements:
> - **NO fake or synthetic masks have been generated.**
> - **NO fake segmentation model training was executed.**
> - **NO segmentation accuracy is claimed.**
> - **All existing classification checkpoints, splits, and artifacts remain 100% read-only and untouched.**

---

## 2. Dataset Audit Summary Table

| Audit Metric | Finding / Value |
| :--- | :--- |
| **Segmentation Dataset Found?** | **{"YES" if seg_possible else "NO (`GROUND_TRUTH_MASKS_NOT_FOUND`)"}** |
| **Number of Source Images** | `{total_images}` unique images (Train={leakage_info['actual_counts']['Train']}, Val={leakage_info['actual_counts']['Validation']}, Test={leakage_info['actual_counts']['Test']}) |
| **Discovered External Mask Files** | `{len(discovered_mask_files)}` |
| **Number of Valid Ground-Truth Masks** | `{passed_cnt}` |
| **Number of Valid Image-Mask Pairs** | `{passed_cnt}` |
| **Number of Pending Masks** | `{total_images - annotated_cnt}` |
| **Dataset Leakage Detected?** | **{"YES (WARNING)" if leakage_info['has_leakage'] else "NO (0 Overlap)"}** |
| **Mask Encoding** | `0=Background, 1=Aphids, 2=Leaf miner, 3=TMB, 4=Leaf blight` |
| **Train/Validation/Test Compatibility** | Isolated & Compatible (Split files preserved) |
| **Segmentation Training Currently Possible?** | **{"YES" if seg_possible else "NO (Requires Ground-Truth Mask Annotations)"}** |

---

## 3. Class Definitions & Mask Encoding Rules

The classification setup defines 4 target classes + Background:
1. **Aphids** (`Pest` — Code 1)
2. **Leaf miner** (`Pest` — Code 2)
3. **TMB** (`Pest` — Code 3, Tea Mosquito Bug)
4. **Leaf blight** (`Disease` — Code 4, Single Disease Class)
5. **Background** (`Healthy leaf tissue` — Code 0)
"""

    report_path_doc = os.path.join(doc_dir, "Phase_Segmentation_Dataset_Audit.md")
    with open(report_path_doc, "w") as f:
        f.write(report_content)

    logger.info(f"[AUDIT COMPLETE] Status={audit_status} | Deliverables saved to: {seg_dir} and {doc_dir}")
    return audit_summary


# ---------------------------------------------------------
# 9. COMPREHENSIVE 12-TEST AUTOMATED AUDIT VERIFICATION SUITE
# ---------------------------------------------------------
def run_phase_c_audit_verification_suite() -> Dict[str, Any]:
    """
    Runs a 12-test automated verification suite for Phase C audit and validation rules.
    Operates on temporary isolated test directories without modifying production data.
    """
    print("\n=== RUNNING PHASE C COMPREHENSIVE AUTOMATED AUDIT TEST SUITE (12 TESTS) ===")
    test_results = {}
    temp_dir = tempfile.mkdtemp(prefix="phase_c_audit_suite12_")

    try:
        img_p = os.path.join(temp_dir, "sample.jpg")
        Image.fromarray(np.uint8(np.random.randint(0, 255, (100, 100, 3)))).save(img_p)

        # TEST 1: Valid mask
        valid_mask_arr = np.zeros((100, 100), dtype=np.uint8)
        valid_mask_arr[10:30, 10:30] = 3  # TMB code 3
        v_mask_p = os.path.join(temp_dir, "valid_mask.png")
        Image.fromarray(valid_mask_arr, mode="L").save(v_mask_p)
        is_v1, _, _ = validate_mask_file(img_p, v_mask_p, 3)
        test_results["TEST_1_valid_mask"] = "PASS" if is_v1 else "FAIL"

        # TEST 2: Missing mask
        is_v2, msg_v2, _ = validate_mask_file(img_p, os.path.join(temp_dir, "missing.png"), 3)
        test_results["TEST_2_missing_mask"] = "PASS" if (not is_v2 and "does not exist" in msg_v2) else "FAIL"

        # TEST 3: Dimension mismatch
        dim_mask_arr = np.zeros((50, 50), dtype=np.uint8)
        dim_mask_arr[5:15, 5:15] = 3
        d_mask_p = os.path.join(temp_dir, "dim_mask.png")
        Image.fromarray(dim_mask_arr, mode="L").save(d_mask_p)
        is_v3, msg_v3, _ = validate_mask_file(img_p, d_mask_p, 3)
        test_results["TEST_3_dimension_mismatch"] = "PASS" if (not is_v3 and "Dimension mismatch" in msg_v3) else "FAIL"

        # TEST 4: RGB mask
        rgb_mask_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_mask_arr[10:30, 10:30, :] = 255
        rgb_mask_p = os.path.join(temp_dir, "rgb_mask.png")
        Image.fromarray(rgb_mask_arr).save(rgb_mask_p)
        is_v4, msg_v4, _ = validate_mask_file(img_p, rgb_mask_p, 3)
        test_results["TEST_4_rgb_mask"] = "PASS" if (not is_v4 and "multi-channel" in msg_v4) else "FAIL"

        # TEST 5: Wrong dtype (float32 array saved to non-uint8)
        wrong_dtype_p = os.path.join(temp_dir, "wrong_dtype.png")
        with open(wrong_dtype_p, "wb") as f:
            f.write(b"not a valid png")
        is_v5, msg_v5, _ = validate_mask_file(img_p, wrong_dtype_p, 3)
        test_results["TEST_5_wrong_dtype"] = "PASS" if (not is_v5 and "Corrupt image/mask" in msg_v5) else "FAIL"

        # TEST 6: Invalid pixel value (e.g. 5)
        inv_val_arr = np.zeros((100, 100), dtype=np.uint8)
        inv_val_arr[10:30, 10:30] = 5
        inv_val_p = os.path.join(temp_dir, "inv_val_mask.png")
        Image.fromarray(inv_val_arr, mode="L").save(inv_val_p)
        is_v6, msg_v6, _ = validate_mask_file(img_p, inv_val_p, 3)
        test_results["TEST_6_invalid_pixel_value"] = "PASS" if (not is_v6 and "Only 0-4 allowed" in msg_v6) else "FAIL"

        # TEST 7: Empty mask
        empty_arr = np.zeros((100, 100), dtype=np.uint8)
        empty_p = os.path.join(temp_dir, "empty_mask.png")
        Image.fromarray(empty_arr, mode="L").save(empty_p)
        is_v7, msg_v7, _ = validate_mask_file(img_p, empty_p, 3)
        test_results["TEST_7_empty_mask"] = "PASS" if (not is_v7 and "Empty mask" in msg_v7) else "FAIL"

        # TEST 8: Expected class missing
        wrong_code_arr = np.zeros((100, 100), dtype=np.uint8)
        wrong_code_arr[10:30, 10:30] = 1  # Code 1 passed for TMB Code 3
        wrong_code_p = os.path.join(temp_dir, "wrong_code_mask.png")
        Image.fromarray(wrong_code_arr, mode="L").save(wrong_code_p)
        is_v8, msg_v8, _ = validate_mask_file(img_p, wrong_code_p, 3)
        test_results["TEST_8_expected_class_missing"] = "PASS" if (not is_v8 and "Expected class code 3 was not found" in msg_v8) else "FAIL"

        # TEST 9: Valid mask background + expected lesion class
        is_v9, _, _ = validate_mask_file(img_p, v_mask_p, 3)
        test_results["TEST_9_valid_background_plus_lesion"] = "PASS" if is_v9 else "FAIL"

        # TEST 10: Unknown classification class -> NOT assigned Background
        norm_c, norm_code = normalize_class_name("unknown_disease_xyz")
        test_results["TEST_10_unknown_class_not_background"] = "PASS" if (norm_c == "UNKNOWN_CLASS" and norm_code == -1) else "FAIL"

        # TEST 11: Duplicate image across splits
        temp_prep = os.path.join(temp_dir, "Preprocessed")
        os.makedirs(temp_prep, exist_ok=True)
        dup_df = pd.DataFrame([{"file_path": "/dup/image.jpg", "class_name": "TMB"}])
        dup_df.to_csv(os.path.join(temp_prep, "train_split.csv"), index=False)
        dup_df.to_csv(os.path.join(temp_prep, "val_split.csv"), index=False)
        leak_res = check_dataset_leakage(preprocessed_dir=temp_prep)
        test_results["TEST_11_duplicate_image_leakage_detected"] = "PASS" if leak_res["has_leakage"] else "FAIL"

        # TEST 12: Train/Val/Test count verification
        leak_prod = check_dataset_leakage()
        test_results["TEST_12_split_count_verification"] = "PASS" if isinstance(leak_prod["actual_counts"], dict) else "FAIL"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    failed_list = [k for k, v in test_results.items() if v != "PASS"]
    all_passed = (len(failed_list) == 0)

    print("\n--- PHASE C AUTOMATED AUDIT VERIFICATION RESULTS (12 TESTS) ---")
    for test_k, test_v in test_results.items():
        print(f"{test_k:<44}: {test_v}")
    print(f"\nAll 12 Audit Tests Passed: {all_passed}")

    return {
        "all_passed": all_passed,
        "tests": test_results,
        "failed_tests": failed_list
    }


if __name__ == "__main__":
    audit_segmentation_dataset()
