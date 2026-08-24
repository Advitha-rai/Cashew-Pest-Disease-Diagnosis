"""
Cashew Pest and Disease Diagnosis System
Phase C.1.10 — Annotation Environment Initialization Script
Framework: TensorFlow / Keras (Cross-Platform & Colab / Drive Compatible)
"""

import os
import sys
from pathlib import Path

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
    SEGMENTATION_DIR,
)
from src.segmentation.callbacks import register_colab_callbacks
from src.segmentation.manifest import load_manifest, get_annotation_progress_report
from src.segmentation.pipeline import initialize_segmentation


def setup_annotation_environment():
    print("==================================================")
    print("PHASE C.1.10 — ANNOTATION ENVIRONMENT INITIALIZATION")
    print("==================================================")
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Storage Root    : {DRIVE_ROOT}")

    results = {}
    errors = []

    # 1. Initialize empty working directories safely
    try:
        ann_base = DRIVE_ROOT / "Experiments" / "Segmentation" / "Annotations"
        for split in ["Train", "Validation"]:
            for cls_name in ["Aphids", "Leaf_Miner", "Leaf_Blight", "TMB"]:
                folder = ann_base / split / cls_name
                folder.mkdir(parents=True, exist_ok=True)
        results["Working directories initialized"] = "PASS"
        print("[1/4] Isolated Annotation Directories .... PASS")
    except Exception as exc:
        results["Working directories initialized"] = "FAIL"
        errors.append(f"Directory creation failed: {exc}")

    # 2. Verify callback environment
    cb_status = register_colab_callbacks()
    cb_ok = cb_status.get("success", False)
    results["Callback environment"] = "PASS" if cb_ok else "FAIL"
    print(f"[2/4] Callback Registration ........... {'PASS' if cb_ok else 'FAIL'} ({cb_status.get('status')})")

    # 3. Verify manifest readiness
    manifest_path = DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    try:
        df_man = load_manifest(manifest_path)
        prog = get_annotation_progress_report(manifest_path)
        results["Manifest environment"] = "PASS"
        print(f"[3/4] Manifest Verification ........... PASS (Total={len(df_man)}, Eligible={prog['total_eligible_images']})")
    except Exception as exc:
        results["Manifest environment"] = "FAIL"
        errors.append(f"Manifest load failed: {exc}")

    # 4. Safe UI initialization readiness (without launching session)
    try:
        from src.segmentation.ui import build_annotation_html
        from src.segmentation.pipeline import launch_segmentation_tool
        results["UI readiness"] = "PASS"
        print("[4/4] UI Pipeline Readiness ........... PASS")
    except Exception as exc:
        results["UI readiness"] = "FAIL"
        errors.append(f"UI module check failed: {exc}")

    all_pass = all(v == "PASS" for v in results.values()) and len(errors) == 0

    print("\n==================================================")
    print(f"PHASE C.1.10 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("==================================================")


if __name__ == "__main__":
    setup_annotation_environment()
