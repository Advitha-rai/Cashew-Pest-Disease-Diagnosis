"""
Cashew Pest and Disease Diagnosis System
Phase C.1.13 — Periodic Annotation Monitor Script (READ-ONLY)
Framework: TensorFlow / Keras (Cross-Platform & Colab / Drive Compatible)
"""

import os
import sys
from pathlib import Path
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
    CANONICAL_MANIFEST,
    ANNOTATIONS_DIR,
    READ_ONLY_SPLIT,
    ANNOTATABLE_SPLITS,
    normalize_class_name,
)
from src.segmentation.manifest import load_manifest, get_annotation_progress_report


def monitor_annotation_progress():
    print("==================================================")
    print("PHASE C.1.13 — ANNOTATION MONITOR (READ-ONLY)")
    print("==================================================")
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Drive Storage   : {DRIVE_ROOT}")

    manifest_path = DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    df_man = load_manifest(manifest_path)
    prog = get_annotation_progress_report(manifest_path)

    total_eligible = prog["total_eligible_images"]
    ann_cnt = prog["annotated_count"]
    skip_cnt = prog["skipped_count"]
    pend_cnt = prog["pending_count"]
    passed_cnt = prog["passed_validation_count"]
    failed_cnt = prog["failed_validation_count"]
    test_cnt = prog["test_images_isolated"]

    print(f"\nLive Annotation Status:")
    print(f"  Total Eligible : {total_eligible}")
    print(f"  Annotated      : {ann_cnt}")
    print(f"  Skipped        : {skip_cnt}")
    print(f"  Pending        : {pend_cnt}")
    print(f"  Passed Valid.  : {passed_cnt}")
    print(f"  Failed Valid.  : {failed_cnt}")
    print(f"  Test (Isolated): {test_cnt} [READ ONLY]")
    print(f"  Progress       : {prog['progress_percentage']}%")

    errors = []

    # 1. Accounting Verification
    accounting_valid = (total_eligible == (ann_cnt + skip_cnt + pend_cnt))
    if not accounting_valid:
        errors.append(f"Accounting mismatch: {total_eligible} != {ann_cnt} + {skip_cnt} + {pend_cnt}")

    # 2. Validation Subordination
    val_valid = (passed_cnt <= ann_cnt)
    if not val_valid:
        errors.append(f"Validation anomaly: passed ({passed_cnt}) > annotated ({ann_cnt})")

    # 3. Test Isolation
    test_valid = (test_cnt in [0, 861])
    if not test_valid:
        errors.append(f"Test count anomaly: {test_cnt} != 861")

    # 4. Audit Existing Physical Masks
    possible_ann_dirs = [
        DRIVE_ROOT / "Experiments" / "Segmentation" / "Annotations",
        REPO_ROOT / "Experiments" / "Segmentation" / "Annotations",
        ANNOTATIONS_DIR,
    ]

    invalid_masks = []
    total_physical_masks = 0

    for ad in possible_ann_dirs:
        if ad.exists():
            for m_p in ad.glob("**/*.png"):
                total_physical_masks += 1
                try:
                    norm_c = normalize_class_name(m_p.parent.name)
                    exp_c = CLASS_CODES.get(norm_c, -1)
                    with Image.open(m_p) as img:
                        arr = np.asarray(img)
                        u_vals = set(np.unique(arr))
                        if not u_vals.issubset({0, exp_c}):
                            invalid_masks.append((str(m_p), norm_c, u_vals, exp_c))
                except Exception as e:
                    invalid_masks.append((str(m_p), "Unknown", str(e), -1))

    print(f"\nMask Files Inspected: {total_physical_masks}")
    print(f"Invalid Masks Found : {len(invalid_masks)}")

    if len(invalid_masks) > 0:
        for im in invalid_masks[:5]:
            print(f"  - WARNING on {im[0]}: values={im[2]}, expected subset of {{0, {im[3]}}}")
        errors.append(f"Found {len(invalid_masks)} invalid mask files.")

    all_pass = accounting_valid and val_valid and test_valid and (len(invalid_masks) == 0) and (len(errors) == 0)

    print("\n==================================================")
    print(f"PHASE C.1.13 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("==================================================")


if __name__ == "__main__":
    monitor_annotation_progress()
