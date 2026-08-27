"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — True Read-Only Segmentation Annotation Audit
Framework: Python / PIL / NumPy / Pandas
"""

import os
import sys
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# Dynamic Environment & Path Discovery
# ---------------------------------------------------------
if Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
    DRIVE_ROOT = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
elif Path("/content/Cashew-Pest-Disease-Diagnosis").exists():
    DRIVE_ROOT = Path("/content/Cashew-Pest-Disease-Diagnosis")
else:
    DRIVE_ROOT = Path.cwd()

SEGMENTATION_ROOT = DRIVE_ROOT / "Experiments" / "Segmentation"
ACTIVE_ANNOTATION_ROOT = SEGMENTATION_ROOT / "Annotations" / "Train"
BACKUP_ROOT = SEGMENTATION_ROOT / "backup_previous"
TRAIN_CSV = DRIVE_ROOT / "Preprocessed" / "train_split.csv"

# Canonical Class Codes Mapping
CLASS_MAPPING = {
    "aphids": 1,
    "leaf_miner": 2,
    "leaf miner": 2,
    "leaf_blight": 3,
    "leaf blight": 3,
    "tmb": 4,
}

CLASS_DISPLAY_NAMES = {
    1: "Aphids",
    2: "Leaf_miner",
    3: "Leaf_blight",
    4: "TMB",
}


def normalize_class_key(cname: str) -> str:
    """Normalizes class string for robust dictionary matching."""
    s = str(cname).strip().lower().replace("-", "_")
    if "aphid" in s:
        return "aphids"
    elif "miner" in s:
        return "leaf_miner"
    elif "blight" in s:
        return "leaf_blight"
    elif "tmb" in s or "mosquito" in s:
        return "tmb"
    return s


def get_class_code(cname: str) -> int:
    key = normalize_class_key(cname)
    return CLASS_MAPPING.get(key, -1)


def run_true_annotation_audit():
    print("============================================================")
    print("TRUE SEGMENTATION ANNOTATION AUDIT (STRICTLY READ-ONLY)")
    print("============================================================")
    print(f"Project Root         : {DRIVE_ROOT}")
    print(f"Active Annotation Dir: {ACTIVE_ANNOTATION_ROOT}")
    print(f"Backup Dir           : {BACKUP_ROOT}")
    print(f"Train Split CSV      : {TRAIN_CSV}")
    print("============================================================\n")

    # ---------------------------------------------------------
    # STEP 1: Search Active Annotation Directory ONLY
    # ---------------------------------------------------------
    active_mask_files: List[Path] = []
    if ACTIVE_ANNOTATION_ROOT.exists():
        for ext in ("*_mask.png", "*_mask.PNG", "*.png", "*.PNG"):
            active_mask_files.extend(list(ACTIVE_ANNOTATION_ROOT.glob(f"**/{ext}")))
    
    # Filter out any non-mask PNGs if necessary, making sure no backup paths are included
    active_mask_files = sorted(list(set([
        p for p in active_mask_files
        if "backup_previous" not in str(p) and p.is_file()
    ])))

    # Separate Backup Masks Count
    backup_mask_files: List[Path] = []
    if BACKUP_ROOT.exists():
        for ext in ("*.png", "*.PNG"):
            backup_mask_files.extend(list(BACKUP_ROOT.glob(f"**/{ext}")))
    backup_mask_files = sorted(list(set(backup_mask_files)))

    # ---------------------------------------------------------
    # STEP 2: Mask Validation on Active Masks
    # ---------------------------------------------------------
    valid_active_masks: List[Dict[str, Any]] = []
    invalid_active_masks: List[Dict[str, Any]] = []
    
    validation_checks = {
        "size_224": True,
        "mode_L": True,
        "dtype_uint8": True,
        "allowed_codes": True,
        "non_empty": True,
    }

    for mask_path in active_mask_files:
        # Determine class from parent folder name
        parent_dir_name = mask_path.parent.name
        exp_code = get_class_code(parent_dir_name)
        norm_class_name = CLASS_DISPLAY_NAMES.get(exp_code, parent_dir_name)

        # Extract image stem key (e.g. IMG_7924_mask.png -> img_7924)
        stem = mask_path.name
        if stem.lower().endswith("_mask.png"):
            img_stem_key = stem[:-9].lower()
        elif stem.lower().endswith(".png"):
            img_stem_key = stem[:-4].lower()
        else:
            img_stem_key = mask_path.stem.lower()

        record = {
            "mask_path": mask_path,
            "filename": mask_path.name,
            "class_name": norm_class_name,
            "class_code": exp_code,
            "img_stem_key": img_stem_key,
        }

        try:
            with Image.open(mask_path) as pil_img:
                size_tuple = pil_img.size
                mode_str = pil_img.mode
                arr = np.asarray(pil_img)
                dtype_str = str(arr.dtype)
                unique_vals = set(np.unique(arr))

            fg_count = int(np.count_nonzero(arr))
            
            check_geom = (size_tuple == (224, 224))
            check_mode = (mode_str == "L")
            check_dtype = (dtype_str == "uint8")
            check_codes = unique_vals.issubset({0, 1, 2, 3, 4})
            check_non_empty = (fg_count > 0)
            check_exp_code = (exp_code in unique_vals) if exp_code > 0 else True

            if not check_geom: validation_checks["size_224"] = False
            if not check_mode: validation_checks["mode_L"] = False
            if not check_dtype: validation_checks["dtype_uint8"] = False
            if not check_codes: validation_checks["allowed_codes"] = False
            if not check_non_empty: validation_checks["non_empty"] = False

            record["size"] = size_tuple
            record["mode"] = mode_str
            record["dtype"] = dtype_str
            record["unique_values"] = sorted(list(unique_vals))
            record["fg_count"] = fg_count

            if check_geom and check_mode and check_dtype and check_codes and check_non_empty and check_exp_code:
                valid_active_masks.append(record)
            else:
                reasons = []
                if not check_geom: reasons.append(f"Size mismatch {size_tuple}")
                if not check_mode: reasons.append(f"Mode mismatch {mode_str}")
                if not check_dtype: reasons.append(f"Dtype mismatch {dtype_str}")
                if not check_codes: reasons.append(f"Invalid codes {unique_vals}")
                if not check_non_empty: reasons.append("Empty (0 fg pixels)")
                if not check_exp_code: reasons.append(f"Missing expected code {exp_code}")
                record["reasons"] = reasons
                invalid_active_masks.append(record)

        except Exception as exc:
            record["reasons"] = [f"Exception loading image: {exc}"]
            invalid_active_masks.append(record)

    # ---------------------------------------------------------
    # STEP 3 & STEP 4: Load Train Split CSV & Match Masks
    # ---------------------------------------------------------
    train_df = None
    train_records_by_stem: Dict[str, Dict[str, Any]] = {}
    train_class_counts: Dict[str, int] = {
        "Aphids": 0,
        "Leaf_miner": 0,
        "Leaf_blight": 0,
        "TMB": 0,
    }

    if TRAIN_CSV.exists():
        try:
            train_df = pd.read_csv(TRAIN_CSV)
            print(f"Train Split CSV loaded. Columns: {list(train_df.columns)}")

            # Identify filename/path column
            path_col = None
            for c in ("file_path", "image_path", "filename", "image_name", "image"):
                if c in train_df.columns:
                    path_col = c
                    break

            # Identify class column
            class_col = None
            for c in ("class_name", "class", "label", "category"):
                if c in train_df.columns:
                    class_col = c
                    break

            if path_col and class_col:
                for _, row in train_df.iterrows():
                    full_path = str(row[path_col])
                    raw_cls = str(row[class_col])
                    code = get_class_code(raw_cls)
                    display_cls = CLASS_DISPLAY_NAMES.get(code, raw_cls)

                    stem_key = Path(full_path).stem.lower()
                    img_filename = Path(full_path).name

                    train_records_by_stem[stem_key] = {
                        "filename": img_filename,
                        "class_name": display_cls,
                        "class_code": code,
                    }
                    if display_cls in train_class_counts:
                        train_class_counts[display_cls] += 1
        except Exception as exc:
            print(f"Warning: Error reading Train Split CSV ({exc})")

    total_train_images = len(train_records_by_stem) if train_records_by_stem else 4013

    # ---------------------------------------------------------
    # STEP 7 & STEP 9: Unique Active Annotations & Duplicate Handling
    # ---------------------------------------------------------
    unique_annotated_stems: Set[str] = set()
    duplicate_active_masks: List[Dict[str, Any]] = []
    matched_valid_masks: List[Dict[str, Any]] = []
    unmatched_valid_masks: List[Dict[str, Any]] = []

    per_class_valid_annotated: Dict[str, Set[str]] = {
        "Aphids": set(),
        "Leaf_miner": set(),
        "Leaf_blight": set(),
        "TMB": set(),
    }

    for rec in valid_active_masks:
        stem_key = rec["img_stem_key"]
        cname = rec["class_name"]

        if stem_key in unique_annotated_stems:
            duplicate_active_masks.append(rec)
        else:
            unique_annotated_stems.add(stem_key)
            if stem_key in train_records_by_stem:
                matched_valid_masks.append(rec)
            else:
                unmatched_valid_masks.append(rec)

            if cname in per_class_valid_annotated:
                per_class_valid_annotated[cname].add(stem_key)

    unique_annotated_count = len(unique_annotated_stems)
    pending_training_count = max(0, total_train_images - unique_annotated_count)

    # Missing annotations list (up to 50)
    missing_stems = set(train_records_by_stem.keys()) - unique_annotated_stems
    missing_filenames = [train_records_by_stem[s]["filename"] for s in sorted(list(missing_stems))]

    # ---------------------------------------------------------
    # STEP 8: Detailed Audit Output Report
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRUE ANNOTATION AUDIT")
    print("=" * 60)
    print(f"Active masks found           : {len(active_mask_files)}")
    print(f"Valid active masks           : {len(valid_active_masks)}")
    print(f"Invalid active masks         : {len(invalid_active_masks)}")
    print(f"Backup masks (Ignored)       : {len(backup_mask_files)}")
    print(f"Duplicate active masks       : {len(duplicate_active_masks)}")
    print(f"Unique annotated train images: {unique_annotated_count}")
    print(f"Total training images        : {total_train_images}")
    print(f"Pending training images      : {pending_training_count}")

    print("\nPER CLASS ANNOTATION PROGRESS")
    print("-" * 60)
    print(f"{'Class Name':<15s} | {'Total Train':<12s} | {'Annotated':<10s} | {'Pending':<10s}")
    print("-" * 60)
    for cname in ("Aphids", "Leaf_miner", "Leaf_blight", "TMB"):
        tot = train_class_counts.get(cname, 0)
        ann = len(per_class_valid_annotated.get(cname, set()))
        pnd = max(0, tot - ann) if tot > 0 else 0
        print(f"{cname:<15s} | {tot:<12d} | {ann:<10d} | {pnd:<10d}")

    print("\nMASK VALIDATION CHECKS (Active Directory)")
    print("-" * 60)
    print(f"  224x224 Geometry    : {'PASS' if validation_checks['size_224'] else 'FAIL'}")
    print(f"  Mode L (Grayscale)  : {'PASS' if validation_checks['mode_L'] else 'FAIL'}")
    print(f"  dtype uint8         : {'PASS' if validation_checks['dtype_uint8'] else 'FAIL'}")
    print(f"  Allowed Class Codes : {'PASS' if validation_checks['allowed_codes'] else 'FAIL'}")
    print(f"  Non-Empty (fg > 0)  : {'PASS' if validation_checks['non_empty'] else 'FAIL'}")

    print("\nDUPLICATE ACTIVE MASKS")
    print("-" * 60)
    if duplicate_active_masks:
        for dup in duplicate_active_masks:
            print(f"  ⚠️ Duplicate: {dup['filename']} ({dup['class_name']})")
    else:
        print("  None (0 duplicate active masks)")

    print("\nINVALID ACTIVE MASKS")
    print("-" * 60)
    if invalid_active_masks:
        for inv in invalid_active_masks:
            print(f"  ❌ Invalid: {inv['filename']} | Reasons: {', '.join(inv['reasons'])}")
    else:
        print("  None (0 invalid active masks)")

    print("\nBACKUP MASKS (Stored under backup_previous/)")
    print("-" * 60)
    print(f"  Count: {len(backup_mask_files)} (Reported separately, NOT counted as active annotations)")

    print("\nMISSING ANNOTATIONS (Sample of first 50 pending Train images)")
    print("-" * 60)
    if missing_filenames:
        for fname in missing_filenames[:50]:
            print(f"  - {fname}")
        if len(missing_filenames) > 50:
            print(f"  ... and {len(missing_filenames) - 50} more pending images.")
    else:
        print("  None! All training images are annotated.")

    print("\n============================================================")
    print(f"FINAL AUDIT RESULT:")
    print(f"TRUE NUMBER OF MANUALLY ANNOTATED IMAGES: {unique_annotated_count}")
    print("============================================================\n")

    return {
        "unique_annotated_count": unique_annotated_count,
        "valid_active_masks": len(valid_active_masks),
        "invalid_active_masks": len(invalid_active_masks),
        "backup_masks": len(backup_mask_files),
        "total_train_images": total_train_images,
        "pending_training_count": pending_training_count,
        "per_class_annotated": {k: len(v) for k, v in per_class_valid_annotated.items()},
    }


if __name__ == "__main__":
    run_true_annotation_audit()
