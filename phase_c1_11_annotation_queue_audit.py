"""
Cashew Pest and Disease Diagnosis System
Phase C.1.11 — Annotation Queue Audit Script (READ-ONLY)
Framework: TensorFlow / Keras (Cross-Platform & Colab / Drive Compatible)
"""

import os
import sys
from pathlib import Path
import pandas as pd

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
    ANNOTATABLE_SPLITS,
    READ_ONLY_SPLIT,
    CANONICAL_MANIFEST,
    normalize_class_name,
)
from src.segmentation.manifest import load_manifest, get_next_pending_image, get_annotation_progress_report


def audit_annotation_queue():
    print("==================================================")
    print("PHASE C.1.11 — ANNOTATION QUEUE AUDIT (READ-ONLY)")
    print("==================================================")
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Drive Storage   : {DRIVE_ROOT}")

    manifest_path = DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    df_man = load_manifest(manifest_path)
    prog = get_annotation_progress_report(manifest_path)

    # 1. Split Isolation & Eligible Filtering
    df_eligible = df_man[df_man["split"].isin(ANNOTATABLE_SPLITS)]
    df_pending = df_eligible[df_eligible["annotation_status"] == "PENDING"]
    df_annotated = df_eligible[df_eligible["annotation_status"] == "ANNOTATED"]
    df_skipped = df_eligible[df_eligible["annotation_status"] == "SKIPPED"]

    # 2. Verify no Test images in eligible/pending queues
    test_in_pending = (df_pending["split"] == READ_ONLY_SPLIT).sum()
    test_in_eligible = (df_eligible["split"] == READ_ONLY_SPLIT).sum()

    queue_secure = (test_in_pending == 0 and test_in_eligible == 0)

    # 3. Next Pending Item
    next_item = get_next_pending_image(split="Train", manifest_csv=manifest_path)
    if next_item is None:
        next_item = get_next_pending_image(split="Validation", manifest_csv=manifest_path)

    # 4. Queue Validation
    queue_valid = True
    errors = []

    if test_in_pending > 0:
        queue_valid = False
        errors.append(f"Security error: {test_in_pending} Test images found in pending queue!")

    # Check for duplicate pending image paths
    dup_pending = df_pending["image_path"].duplicated().sum()
    if dup_pending > 0:
        queue_valid = False
        errors.append(f"Found {dup_pending} duplicate image paths in pending queue.")

    print(f"\nQueue Progress Summary:")
    print(f"  Total Eligible : {len(df_eligible)}")
    print(f"  Annotated      : {len(df_annotated)}")
    print(f"  Skipped        : {len(df_skipped)}")
    print(f"  Pending        : {len(df_pending)}")
    print(f"  Progress       : {prog['progress_percentage']}%")

    if next_item:
        print(f"\nNext Pending Annotation Item:")
        print(f"  Image Name     : {next_item.get('image_name')}")
        print(f"  Split          : {next_item.get('split')}")
        print(f"  Class Name     : {next_item.get('class_name')}")
        print(f"  Class Code     : {next_item.get('class_code')}")
        print(f"  Image Path     : {next_item.get('image_path')}")
        print(f"  Expected Mask  : {next_item.get('expected_mask_path')}")
    else:
        print("\nNext Pending Item: None (All eligible images are annotated or skipped)")

    print(f"\nTest Split Isolation : {'STRICTLY_READ_ONLY (PASS)' if queue_secure else 'FAILED'}")

    all_pass = queue_valid and queue_secure and len(errors) == 0

    print("\n==================================================")
    print(f"PHASE C.1.11 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("==================================================")


if __name__ == "__main__":
    audit_annotation_queue()
