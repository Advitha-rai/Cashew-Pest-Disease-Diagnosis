"""
Cashew Pest and Disease Diagnosis System
Phase C.1.12 — Manual Bulk Annotation Interactive Interface Launcher
Framework: TensorFlow / Keras (Google Colab / Jupyter Dual-Canvas UI)
"""

import os
import sys
import argparse
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

from src.segmentation import (
    initialize_segmentation,
    launch_segmentation_tool,
    get_annotation_progress,
)
from src.segmentation.config import CANONICAL_MANIFEST


def main():
    parser = argparse.ArgumentParser(description="Launch Cashew Manual Segmentation Annotation Interface")
    parser.add_argument("--split", type=str, default="Train", choices=["Train", "Validation"], help="Split partition to annotate (Test is prohibited)")
    parser.add_argument("--class", type=str, dest="class_name", default=None, help="Target class filter (Aphids, Leaf_Miner, Leaf_Blight, TMB)")
    parser.add_argument("--debug-ui", action="store_true", help="Enable frontend debug messages")

    args = parser.parse_args()

    print("==================================================")
    print("PHASE C.1.12 — CASHEW MANUAL SEGMENTATION UI")
    print("==================================================")
    print("Initializing environment and registering Colab callbacks...")

    manifest_path = DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv"
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    init_res = initialize_segmentation(manifest_csv=manifest_path)

    print(f"\nLaunching manual annotation canvas for Split='{args.split}', Class='{args.class_name}'...")
    print("Instructions:")
    print("  1. Paint lesion area with the brush (or erase).")
    print("  2. Click '💾 Save Mask & Next' to validate, atomically save uint8 PNG mask, and load next image.")
    print("  3. Click '⏭️ Skip for Review' if the image is ambiguous or unsuitable.")
    print("  4. Test split is strictly isolated and read-only.")
    print("==================================================\n")

    return launch_segmentation_tool(
        split=args.split,
        class_name=args.class_name,
        manifest_csv=manifest_path,
        debug_ui=args.debug_ui
    )


if __name__ == "__main__":
    main()
