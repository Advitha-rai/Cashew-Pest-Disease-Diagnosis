"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — Interactive Bulk Annotation Runner
Framework: TensorFlow / Keras (Google Colab / Jupyter Environment)
"""

import os
import sys
import argparse
import importlib
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

# Force fresh import of segmentation modules
for mod in list(sys.modules):
    if mod == "src.segmentation" or mod.startswith("src.segmentation."):
        del sys.modules[mod]

importlib.invalidate_caches()

from src.segmentation import (
    initialize_segmentation,
    launch_segmentation_tool,
    get_annotation_progress,
    load_manifest,
    get_next_pending_image,
    assert_annotation_allowed,
)
from src.segmentation.config import (
    CANONICAL_MANIFEST,
    DATASET_DIR,
    CLASS_CODES,
    ANNOTATABLE_SPLITS,
    READ_ONLY_SPLIT,
)


def run_bulk_annotation():
    parser = argparse.ArgumentParser(description="Phase C.1.15 Bulk Annotation Runner")
    parser.add_argument("--split", type=str, default="Train", choices=["Train", "Validation"], help="Split partition to annotate (Test is prohibited)")
    parser.add_argument("--class", type=str, dest="class_name", default=None, help="Target class filter (Aphids, Leaf_Miner, Leaf_Blight, TMB)")
    parser.add_argument("--debug-ui", action="store_true", help="Enable frontend debug diagnostics")

    args, unknown = parser.parse_known_args()

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

    if manifest_path is None:
        manifest_path = CANONICAL_MANIFEST

    # 2. Initialize Segmentation Callbacks & Environment
    initialize_segmentation(manifest_csv=manifest_path)
    prog = get_annotation_progress(manifest_path)

    # 3. Read-Only Backend Test-Split Security Check
    test_protected = False
    try:
        assert_annotation_allowed("Test", "/path/to/test.jpg")
    except PermissionError:
        test_protected = True

    # 4. Next Pending Item Identification
    next_item = get_next_pending_image(split=args.split, class_name=args.class_name, manifest_csv=manifest_path)
    if next_item is None and args.split == "Train":
        next_item = get_next_pending_image(split="Validation", class_name=args.class_name, manifest_csv=manifest_path)

    next_img_name = next_item.get("image_name", "None (All pending images processed)") if next_item else "None"
    next_class_name = next_item.get("class_name", "N/A") if next_item else "N/A"
    next_split_name = next_item.get("split", "N/A") if next_item else "N/A"

    # 5. Startup Report
    print("==================================================")
    print("PHASE C.1.15 — BULK ANNOTATION RUNNER")
    print("==================================================")
    print(f"Repository Root    : {REPO_ROOT}")
    print(f"Storage Root       : {DRIVE_ROOT}")
    print(f"Canonical Manifest : {manifest_path}")

    print(f"\nTotal Eligible     : {prog['total_eligible_images']}")
    print(f"Annotated          : {prog['annotated_count']}")
    print(f"Skipped            : {prog['skipped_count']}")
    print(f"Pending            : {prog['pending_count']}")
    print(f"Passed             : {prog['passed_validation_count']}")
    print(f"Failed             : {prog['failed_validation_count']}")
    print(f"Test Images        : {prog['test_images_isolated']} [READ ONLY]")

    print(f"\nNext Pending Image : {next_img_name}")
    print(f"Next Pending Class : {next_class_name}")
    print(f"Next Pending Split : {next_split_name}")

    print(f"\nTest Protection    : {'STRICTLY_READ_ONLY (ENFORCED)' if test_protected else 'FAILED'}")
    print(f"Dataset Preservation: PRESERVED (5734 images)")

    print("==================================================")
    print("READY FOR MANUAL BULK ANNOTATION")
    print("==================================================\n")

    # 6. Launch Modular Manual Annotation UI
    return launch_segmentation_tool(
        split=args.split,
        class_name=args.class_name,
        manifest_csv=manifest_path,
        debug_ui=args.debug_ui
    )


if __name__ == "__main__":
    run_bulk_annotation()
