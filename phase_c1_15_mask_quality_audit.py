"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — Read-Only Comprehensive Segmentation Mask Quality Audit
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
ACTIVE_MASK_DIR = SEGMENTATION_ROOT / "Annotations" / "Train"
BACKUP_ROOT = SEGMENTATION_ROOT / "backup_previous"
TRAIN_CSV = DRIVE_ROOT / "Preprocessed" / "train_split.csv"
CLEANED_DATASET_DIR = DRIVE_ROOT / "Dataset" / "Cleaned"

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


def run_mask_quality_audit():
    print("============================================================")
    print("SEGMENTATION MASK QUALITY AUDIT")
    print("STRICTLY READ-ONLY")
    print("============================================================")
    print(f"Project Root          : {DRIVE_ROOT}")
    print(f"Active Mask Directory : {ACTIVE_MASK_DIR}")
    print(f"Train Split CSV       : {TRAIN_CSV}")
    print("============================================================\n")

    # 1. Load Train Split CSV read-only for Image-Mask Pairing
    train_records_by_stem: Dict[str, Dict[str, Any]] = {}
    if TRAIN_CSV.exists():
        try:
            df_train = pd.read_csv(TRAIN_CSV)
            path_col = None
            for c in ("file_path", "image_path", "filename", "image_name", "image"):
                if c in df_train.columns:
                    path_col = c
                    break

            class_col = None
            for c in ("class_name", "class", "label", "category"):
                if c in df_train.columns:
                    class_col = c
                    break

            if path_col and class_col:
                for _, row in df_train.iterrows():
                    full_p = str(row[path_col])
                    raw_c = str(row[class_col])
                    code = get_class_code(raw_c)
                    disp_c = CLASS_DISPLAY_NAMES.get(code, raw_c)

                    stem_key = Path(full_p).stem.lower()
                    
                    # Resolve physical image path on disk
                    phys_p = Path(full_p)
                    if not phys_p.exists():
                        phys_p = CLEANED_DATASET_DIR / raw_c / Path(full_p).name
                        if not phys_p.exists():
                            phys_p = CLEANED_DATASET_DIR / disp_c / Path(full_p).name

                    train_records_by_stem[stem_key] = {
                        "full_path": str(phys_p),
                        "exists": phys_p.exists(),
                        "class_name": disp_c,
                        "class_code": code,
                    }
        except Exception as exc:
            print(f"Warning: Could not read Train Split CSV: {exc}")

    # 2. Search Active Annotations/Train directory ONLY
    active_mask_files: List[Path] = []
    if ACTIVE_MASK_DIR.exists():
        for ext in ("*_mask.png", "*_mask.PNG", "*.png", "*.PNG"):
            active_mask_files.extend(list(ACTIVE_MASK_DIR.glob(f"**/{ext}")))

    # Filter strictly for active masks, excluding backup_previous
    active_mask_files = sorted(list(set([
        p for p in active_mask_files
        if "backup_previous" not in str(p) and p.is_file()
    ])))

    # 3. Duplicate Detection on Active Mask Stems
    stem_to_mask_paths: Dict[str, List[Path]] = {}
    for mpath in active_mask_files:
        stem_name = mpath.name
        if stem_name.lower().endswith("_mask.png"):
            s_key = stem_name[:-9].lower()
        elif stem_name.lower().endswith(".png"):
            s_key = stem_name[:-4].lower()
        else:
            s_key = mpath.stem.lower()

        if s_key not in stem_to_mask_paths:
            stem_to_mask_paths[s_key] = []
        stem_to_mask_paths[s_key].append(mpath)

    duplicate_stems = {s: paths for s, paths in stem_to_mask_paths.items() if len(paths) > 1}

    # Tracking Structures
    per_class_stats: Dict[str, Dict[str, int]] = {
        "Aphids": {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0},
        "Leaf_miner": {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0},
        "Leaf_blight": {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0},
        "TMB": {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0},
    }

    total_active_masks = len(active_mask_files)
    valid_masks_count = 0
    invalid_masks_count = 0
    empty_masks_count = 0
    unexpected_value_masks_count = 0
    missing_code_masks_count = 0
    pairing_errors_count = 0

    all_fg_counts: List[int] = []
    all_fg_percentages: List[float] = []
    itemized_audit_results: List[Dict[str, Any]] = []

    print(f"Found {total_active_masks} active masks in {ACTIVE_MASK_DIR}\n")

    # 4. Detailed Audit Loop per Active Mask File
    for mpath in active_mask_files:
        parent_folder = mpath.parent.name
        exp_code = get_class_code(parent_folder)
        cls_disp = CLASS_DISPLAY_NAMES.get(exp_code, parent_folder)

        if cls_disp not in per_class_stats:
            per_class_stats[cls_disp] = {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0}
        
        per_class_stats[cls_disp]["total"] += 1

        stem_name = mpath.name
        if stem_name.lower().endswith("_mask.png"):
            s_key = stem_name[:-9].lower()
        elif stem_name.lower().endswith(".png"):
            s_key = stem_name[:-4].lower()
        else:
            s_key = mpath.stem.lower()

        # Image-Mask Pairing Check
        pairing_info = train_records_by_stem.get(s_key, None)
        pairing_valid = (pairing_info is not None and pairing_info.get("exists", False))
        if not pairing_valid:
            pairing_errors_count += 1
            per_class_stats[cls_disp]["pairing_errors"] += 1

        # Check A: File Readability
        try:
            with Image.open(mpath) as pil_img:
                size_tuple = pil_img.size
                mode_str = pil_img.mode
                arr = np.asarray(pil_img)
                dtype_str = str(arr.dtype)
                unique_vals = set(np.unique(arr))
                corrupted = False
        except Exception as exc:
            corrupted = True
            size_tuple = (0, 0)
            mode_str = "ERR"
            dtype_str = "ERR"
            unique_vals = set()
            arr = np.array([])

        fg_count = int(np.count_nonzero(arr > 0)) if not corrupted else 0
        total_px = 224 * 224
        fg_pct = round((fg_count / total_px) * 100.0, 4) if not corrupted else 0.0

        if not corrupted:
            all_fg_counts.append(fg_count)
            all_fg_percentages.append(fg_pct)

        # Checks B-H
        check_geom = (size_tuple == (224, 224))
        check_mode = (mode_str == "L")
        check_dtype = (dtype_str == "uint8")
        check_allowed = unique_vals.issubset({0, 1, 2, 3, 4})
        check_non_empty = (fg_count > 0)
        check_code_present = (exp_code in unique_vals) if exp_code > 0 else True

        flags = []
        if corrupted:
            flags.append("CORRUPTED_FILE")
        if not check_geom:
            flags.append(f"INCORRECT_GEOMETRY_{size_tuple}")
        if not check_mode:
            flags.append(f"INCORRECT_MODE_{mode_str}")
        if not check_dtype:
            flags.append(f"INCORRECT_DTYPE_{dtype_str}")
        if not check_allowed:
            flags.append(f"UNEXPECTED_PIXEL_VALUES_{sorted(list(unique_vals))}")
            unexpected_value_masks_count += 1
        if not check_non_empty:
            flags.append("EMPTY_MASK_0_FG")
            empty_masks_count += 1
            per_class_stats[cls_disp]["empty"] += 1
        elif fg_pct < 0.1:
            flags.append(f"WARNING_EXTREMELY_SMALL_FG_{fg_pct:.3f}%")
        elif fg_pct > 75.0:
            flags.append(f"WARNING_SUSPICIOUSLY_LARGE_FG_{fg_pct:.2f}%")

        if not check_code_present:
            flags.append(f"MISSING_EXPECTED_CLASS_CODE_{exp_code}")
            missing_code_masks_count += 1
            per_class_stats[cls_disp]["missing_code"] += 1

        if not pairing_valid:
            flags.append("PAIRING_ERROR_SOURCE_IMAGE_MISSING")

        is_mask_valid = (
            not corrupted and check_geom and check_mode and check_dtype
            and check_allowed and check_non_empty and check_code_present and pairing_valid
        )

        if is_mask_valid:
            valid_masks_count += 1
            per_class_stats[cls_disp]["valid"] += 1
        else:
            invalid_masks_count += 1
            per_class_stats[cls_disp]["invalid"] += 1

        itemized_audit_results.append({
            "mask_path": str(mpath),
            "filename": mpath.name,
            "class_name": cls_disp,
            "expected_code": exp_code,
            "size": size_tuple,
            "mode": mode_str,
            "dtype": dtype_str,
            "unique_values": sorted(list(unique_vals)),
            "fg_count": fg_count,
            "fg_pct": fg_pct,
            "pairing_valid": pairing_valid,
            "is_valid": is_mask_valid,
            "flags": flags,
        })

    # 5. Output Per-Class Summary Table (K)
    print("============================================================")
    print("PER-CLASS MASK QUALITY SUMMARY")
    print("============================================================")
    print(f"{'Class':<15s} | {'Total Masks':<12s} | {'Valid':<8s} | {'Invalid':<8s} | {'Empty':<8s} | {'Missing Code':<12s} | {'Pairing Errors':<14s}")
    print("-" * 88)
    for cname in ("Aphids", "Leaf_miner", "Leaf_blight", "TMB"):
        st = per_class_stats.get(cname, {"total": 0, "valid": 0, "invalid": 0, "empty": 0, "missing_code": 0, "pairing_errors": 0})
        print(f"{cname:<15s} | {st['total']:<12d} | {st['valid']:<8d} | {st['invalid']:<8d} | {st['empty']:<8d} | {st['missing_code']:<12d} | {st['pairing_errors']:<14d}")
    print("=" * 88 + "\n")

    # 6. Calculate Foreground % Statistics
    if all_fg_percentages:
        min_fg_pct = float(np.min(all_fg_percentages))
        max_fg_pct = float(np.max(all_fg_percentages))
        mean_fg_pct = float(np.mean(all_fg_percentages))
        median_fg_pct = float(np.median(all_fg_percentages))
        total_fg_pixels = int(np.sum(all_fg_counts))
    else:
        min_fg_pct = max_fg_pct = mean_fg_pct = median_fg_pct = 0.0
        total_fg_pixels = 0

    num_duplicate_stems = len(duplicate_stems)

    # 7. Output Overall Summary Table (L)
    print("============================================================")
    print("OVERALL AUDIT SUMMARY")
    print("============================================================")
    print(f"Total Active Masks         : {total_active_masks}")
    print(f"Valid Masks                : {valid_masks_count}")
    print(f"Invalid Masks              : {invalid_masks_count}")
    print(f"Empty Masks                : {empty_masks_count}")
    print(f"Unexpected-Value Masks     : {unexpected_value_masks_count}")
    print(f"Missing-Class-Code Masks   : {missing_code_masks_count}")
    print(f"Pairing Errors             : {pairing_errors_count}")
    print(f"Duplicate Active Masks     : {num_duplicate_stems}")
    print(f"Total Foreground Pixels    : {total_fg_pixels:,}")
    print(f"Minimum Foreground %       : {min_fg_pct:.4f}%")
    print(f"Maximum Foreground %       : {max_fg_pct:.4f}%")
    print(f"Mean Foreground %          : {mean_fg_pct:.4f}%")
    print(f"Median Foreground %        : {median_fg_pct:.4f}%")
    print("============================================================\n")

    # 8. Report Flags or Suspicious Items
    flagged_items = [it for it in itemized_audit_results if not it["is_valid"] or any("WARNING" in f for f in it["flags"])]
    if flagged_items:
        print("--- FLAGGED / INVALID MASKS DETAILED REPORT ---")
        for it in flagged_items:
            status_str = "INVALID" if not it["is_valid"] else "WARNING"
            print(f"  [{status_str}] File: {it['filename']} ({it['class_name']}) | Flags: {', '.join(it['flags'])}")
        print()

    # 9. Quality Decision Logic (M)
    has_hard_errors = (
        invalid_masks_count > 0
        or empty_masks_count > 0
        or unexpected_value_masks_count > 0
        or missing_code_masks_count > 0
        or pairing_errors_count > 0
        or num_duplicate_stems > 0
    )
    has_warnings = any("WARNING" in f for it in itemized_audit_results for f in it["flags"])

    if not has_hard_errors and not has_warnings:
        quality_decision = "PASS"
        is_ready = "YES"
    elif not has_hard_errors and has_warnings:
        quality_decision = "PASS WITH WARNINGS"
        is_ready = "YES"
    else:
        quality_decision = "FAIL"
        is_ready = "NO"

    print("============================================================")
    print(f"FINAL QUALITY DECISION : {quality_decision}")
    print("============================================================")
    print(f"READY FOR SEGMENTATION TRAINING: {is_ready}")
    print("============================================================\n")

    return {
        "total_active_masks": total_active_masks,
        "valid_masks": valid_masks_count,
        "invalid_masks": invalid_masks_count,
        "empty_masks": empty_masks_count,
        "decision": quality_decision,
        "ready": is_ready,
    }


if __name__ == "__main__":
    run_mask_quality_audit()
