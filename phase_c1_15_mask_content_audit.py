"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — Read-Only Physical Mask Content & Quality Auditor
Framework: Python / PIL / NumPy / Pandas
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# Dynamic Environment & Path Discovery
# ---------------------------------------------------------
if Path("/content/Cashew-Pest-Disease-Diagnosis").exists():
    REPO_ROOT = Path("/content/Cashew-Pest-Disease-Diagnosis")
elif Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
    REPO_ROOT = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
else:
    REPO_ROOT = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
    DRIVE_ROOT = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
else:
    DRIVE_ROOT = REPO_ROOT

from src.segmentation.config import (
    CLASS_CODES,
    ALLOWED_MASK_VALUES,
    CANONICAL_MANIFEST,
    ANNOTATIONS_DIR,
    normalize_class_name,
)
from src.segmentation.manifest import load_manifest


def run_mask_content_audit():
    print("============================================================")
    print("PHASE C.1.15 — MASK CONTENT AUDIT (STRICTLY READ-ONLY)")
    print("============================================================")
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Drive Root      : {DRIVE_ROOT}")

    # 1. Resolve Canonical Manifest Path
    manifest_candidates = [
        DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv",
        REPO_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv",
        CANONICAL_MANIFEST,
    ]
    manifest_path = None
    for cand in manifest_candidates:
        if cand.exists():
            manifest_path = cand
            break

    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Canonical manifest not found. Checked: {manifest_candidates}")

    print(f"Canonical Manifest Path: {manifest_path}")

    # Load Manifest
    df_man = load_manifest(manifest_path)
    total_manifest_rows = len(df_man)

    # Filter Annotated Records
    df_annotated = df_man[
        (df_man["annotation_status"] == "ANNOTATED")
        & (df_man["validation_status"] == "PASSED")
    ]
    annotated_records_count = len(df_annotated)

    # Search directory for all physical masks in Annotations/
    ann_base_candidates = [
        DRIVE_ROOT / "Experiments" / "Segmentation" / "Annotations",
        REPO_ROOT / "Experiments" / "Segmentation" / "Annotations",
        ANNOTATIONS_DIR,
    ]
    ann_base = None
    for cand in ann_base_candidates:
        if cand.exists():
            ann_base = cand
            break

    # Discover all physical mask files on disk
    all_physical_masks_on_disk: List[Path] = []
    if ann_base and ann_base.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            all_physical_masks_on_disk.extend(list(ann_base.glob(f"**/{ext}")))

    print(f"Total Manifest Rows    : {total_manifest_rows}")
    print(f"Annotated Records      : {annotated_records_count}")
    print(f"Physical Masks On Disk : {len(all_physical_masks_on_disk)}")
    print("============================================================\n")

    # Metrics Tracking
    physical_masks_found = 0
    missing_masks_count = 0
    valid_non_blank_masks = 0
    blank_masks_count = 0
    invalid_class_code_count = 0
    invalid_geometry_count = 0
    invalid_format_count = 0
    overall_valid_masks = 0

    blank_mask_filenames: List[str] = []

    class_stats: Dict[str, Dict[str, int]] = {
        "Aphids": {"annotated": 0, "found": 0, "non_blank": 0, "blank": 0, "invalid": 0},
        "Leaf_Miner": {"annotated": 0, "found": 0, "non_blank": 0, "blank": 0, "invalid": 0},
        "Leaf_Blight": {"annotated": 0, "found": 0, "non_blank": 0, "blank": 0, "invalid": 0},
        "TMB": {"annotated": 0, "found": 0, "non_blank": 0, "blank": 0, "invalid": 0},
    }

    # Helper to resolve physical mask path
    def resolve_mask_file(exp_path_str: str, img_name: str, class_norm: str, split_str: str) -> Optional[Path]:
        if exp_path_str:
            p = Path(exp_path_str)
            if p.exists():
                return p
        # Check standard annotation tree
        stem = Path(img_name).stem
        mask_name = f"{stem}_mask.png"

        folder_aliases = [class_norm, class_norm.replace("_", " ")]
        for ann_dir in ann_base_candidates:
            if ann_dir and ann_dir.exists():
                for fol in folder_aliases:
                    cand = ann_dir / split_str / fol / mask_name
                    if cand.exists():
                        return cand
                    cand = ann_dir / "Train" / fol / mask_name
                    if cand.exists():
                        return cand
                    cand = ann_dir / "Validation" / fol / mask_name
                    if cand.exists():
                        return cand
        return None

    # Perform detailed audit on each annotated record
    if annotated_records_count == 0 and len(all_physical_masks_on_disk) == 0:
        print("No annotated records or physical masks found to audit.")
    else:
        # Build set of records to audit
        audit_items = []

        # 1. Add annotated records from manifest
        for _, row in df_annotated.iterrows():
            img_name = str(row.get("image_name", ""))
            raw_cls = str(row.get("class_name", ""))
            norm_cls = normalize_class_name(raw_cls)
            split_str = str(row.get("split", "Train"))
            exp_mask_str = str(row.get("expected_mask_path", ""))
            exp_code = CLASS_CODES.get(norm_cls, -1)

            if norm_cls in class_stats:
                class_stats[norm_cls]["annotated"] += 1

            mask_path = resolve_mask_file(exp_mask_str, img_name, norm_cls, split_str)
            audit_items.append({
                "image_name": img_name,
                "class_name": norm_cls,
                "class_code": exp_code,
                "split": split_str,
                "mask_path": mask_path,
                "expected_path_str": exp_mask_str or f"{stem if 'stem' in locals() else img_name}_mask.png",
                "source": "MANIFEST",
            })

        # 2. Add any physical masks found on disk not covered by annotated manifest records
        audited_mask_paths = {item["mask_path"] for item in audit_items if item["mask_path"] is not None}
        for pm in all_physical_masks_on_disk:
            if pm not in audited_mask_paths:
                # Infer class from parent folder name
                parent_folder = pm.parent.name
                norm_cls = normalize_class_name(parent_folder)
                exp_code = CLASS_CODES.get(norm_cls, -1)
                img_stem = pm.name.replace("_mask.png", "").replace(".png", "")
                audit_items.append({
                    "image_name": f"{img_stem}.jpg",
                    "class_name": norm_cls,
                    "class_code": exp_code,
                    "split": pm.parent.parent.name if pm.parent.parent.name in ("Train", "Validation") else "Train",
                    "mask_path": pm,
                    "expected_path_str": str(pm),
                    "source": "PHYSICAL_DISK",
                })

        for idx, item in enumerate(audit_items, 1):
            img_name = item["image_name"]
            norm_cls = item["class_name"]
            exp_code = item["class_code"]
            mask_path = item["mask_path"]

            print(f"------------------------------------------------------------")
            print(f"[{idx}/{len(audit_items)}] Record Audit")
            print(f"Image            : {img_name}")
            print(f"Class            : {norm_cls}")
            print(f"Class Code       : {exp_code}")
            print(f"Mask Path        : {mask_path if mask_path else item['expected_path_str']}")

            if mask_path is None or not mask_path.exists():
                missing_masks_count += 1
                if norm_cls in class_stats:
                    class_stats[norm_cls]["invalid"] += 1
                print(f"Exists           : NO")
                print(f"Blank            : N/A")
                print(f"Class-Code       : FAIL — FILE MISSING")
                print(f"Overall          : FAIL — PHYSICAL MASK MISSING")
                continue

            physical_masks_found += 1
            if norm_cls in class_stats:
                class_stats[norm_cls]["found"] += 1

            print(f"Exists           : YES")

            try:
                with Image.open(mask_path) as pil_img:
                    mode_str = pil_img.mode
                    size_tuple = pil_img.size
                    arr = np.asarray(pil_img)
                    dtype_str = str(arr.dtype)
                    unique_vals = sorted(list(np.unique(arr)))

                total_pixels = arr.size
                bg_count = int(np.sum(arr == 0))
                fg_count = int(np.sum(arr > 0))
                fg_pct = (fg_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

                is_blank = (fg_count == 0)
                allowed_vals = {0, exp_code}
                u_set = set(unique_vals)

                is_geom_valid = (size_tuple == (224, 224))
                is_mode_valid = (mode_str == "L")
                is_dtype_valid = (dtype_str == "uint8")
                is_code_valid = u_set.issubset(allowed_vals)

                print(f"Mode             : {mode_str}")
                print(f"Size             : {size_tuple}")
                print(f"Dtype            : {dtype_str}")
                print(f"Unique Values    : {unique_vals}")
                print(f"Background       : {bg_count}")
                print(f"Foreground       : {fg_count}")
                print(f"Foreground %     : {fg_pct:.2f}%")
                print(f"Blank            : {'YES' if is_blank else 'NO'}")
                print(f"Class-Code       : {'PASS' if is_code_valid else 'FAIL'}")

                overall_pass = True
                reasons = []

                if not is_mode_valid:
                    overall_pass = False
                    invalid_format_count += 1
                    reasons.append(f"Invalid mode ({mode_str} != L)")

                if not is_geom_valid:
                    overall_pass = False
                    invalid_geometry_count += 1
                    reasons.append(f"Invalid geometry ({size_tuple} != (224, 224))")

                if not is_code_valid:
                    overall_pass = False
                    invalid_class_code_count += 1
                    reasons.append(f"Unexpected values {unique_vals} vs allowed {allowed_vals}")

                if is_blank:
                    overall_pass = False
                    blank_masks_count += 1
                    blank_mask_filenames.append(mask_path.name)
                    if norm_cls in class_stats:
                        class_stats[norm_cls]["blank"] += 1
                    reasons.append("BLANK MASK (0 foreground pixels)")
                else:
                    valid_non_blank_masks += 1
                    if norm_cls in class_stats:
                        class_stats[norm_cls]["non_blank"] += 1

                if overall_pass:
                    overall_valid_masks += 1
                    print(f"Overall          : PASS")
                else:
                    if norm_cls in class_stats and not is_blank:
                        class_stats[norm_cls]["invalid"] += 1
                    print(f"Overall          : FAIL — {', '.join(reasons)}")

            except Exception as exc:
                invalid_format_count += 1
                if norm_cls in class_stats:
                    class_stats[norm_cls]["invalid"] += 1
                print(f"Overall          : FAIL — EXCEPTION ({exc})")

        print("------------------------------------------------------------\n")

    # Final Overall Summary
    print("============================================================")
    print("PHASE C.1.15 — MASK CONTENT AUDIT SUMMARY")
    print("============================================================")
    print(f"Total Annotated Records    : {annotated_records_count}")
    print(f"Physical Masks Found       : {physical_masks_found}")
    print(f"Missing Masks              : {missing_masks_count}")
    print(f"Valid Non-Blank Masks      : {valid_non_blank_masks}")
    print(f"Blank Masks                : {blank_masks_count}")
    print(f"Invalid Class-Code Masks   : {invalid_class_code_count}")
    print(f"Invalid Geometry           : {invalid_geometry_count}")
    print(f"Invalid Format             : {invalid_format_count}")
    print(f"Overall Valid Masks        : {overall_valid_masks}")
    print("============================================================")

    if blank_masks_count > 0:
        print("\n--- LIST OF BLANK MASKS (REQUIRE RE-ANNOTATION) ---")
        for fn in blank_mask_filenames:
            print(f"  ❌ {fn}")

    # Per-Class Summary Table
    print("\n============================================================")
    print("PER-CLASS MASK CONTENT AUDIT SUMMARY")
    print("============================================================")
    print(f"{'Class Name':<15s} | {'Annotated':<10s} | {'Found':<8s} | {'Non-Blank':<10s} | {'Blank':<8s} | {'Invalid':<8s}")
    print("-" * 72)
    for cname in ("Aphids", "Leaf_Miner", "Leaf_Blight", "TMB"):
        st = class_stats.get(cname, {"annotated": 0, "found": 0, "non_blank": 0, "blank": 0, "invalid": 0})
        print(f"{cname:<15s} | {st['annotated']:<10d} | {st['found']:<8d} | {st['non_blank']:<10d} | {st['blank']:<8d} | {st['invalid']:<8d}")
    print("============================================================\n")


if __name__ == "__main__":
    run_mask_content_audit()
