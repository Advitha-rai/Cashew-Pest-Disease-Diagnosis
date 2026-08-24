"""
Cashew Pest and Disease Diagnosis System
Phase C: Manual Segmentation Annotation CLI Launcher (TensorFlow / Keras)

Usage Examples:
  # 1. Initialize segmentation environment:
  python segmentation.py --init

  # 2. Build or refresh annotation manifest:
  python segmentation.py --manifest

  # 3. Display annotation progress report:
  python segmentation.py --progress

  # 4. Launch interactive annotation interface:
  python segmentation.py --annotate --split Train --class Aphids
"""

import argparse
import sys
from pathlib import Path

from src.segmentation import (
    initialize_segmentation,
    launch_segmentation_tool,
    get_annotation_progress,
    build_segmentation_manifest,
    validate_mask_file,
    load_manifest,
    SegmentationConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Cashew Pest & Disease Manual Segmentation Tool CLI")

    parser.add_argument("--init", action="store_true", help="Initialize segmentation directories and Colab callbacks")
    parser.add_argument("--manifest", action="store_true", help="Build or refresh 5,734-image segmentation annotation manifest")
    parser.add_argument("--progress", "--summary", action="store_true", dest="progress", help="Display annotation progress report")
    parser.add_argument("--annotate", action="store_true", help="Launch interactive annotation interface")
    parser.add_argument("--split", type=str, default="Train", choices=["Train", "Validation"], help="Split partition to annotate")
    parser.add_argument("--class", type=str, dest="class_name", default=None, help="Class filter (Aphids, Leaf_Miner, Leaf_Blight, TMB)")
    parser.add_argument("--debug-ui", action="store_true", help="Enable verbose frontend UI diagnostics")

    args = parser.parse_args()

    # 1. Initialize
    if args.init:
        initialize_segmentation()
        return

    # 2. Manifest
    if args.manifest:
        print("[CLI] Building 5,734-Image Segmentation Manifest...")
        df = build_segmentation_manifest(force_rebuild=True)
        print(f"Manifest generated with {len(df)} total images -> {SegmentationConfig.get_manifest_path()}")
        return

    # 3. Progress
    if args.progress:
        rep = get_annotation_progress()
        print("\n==================================================")
        print("SEGMENTATION ANNOTATION PROGRESS")
        print("==================================================")
        print(f"Total Eligible Pool : {rep['total_eligible_images']} (Train={rep['train_images']}, Val={rep['validation_images']})")
        print(f"Isolated Test Set   : {rep['test_images_isolated']} [READ ONLY]")
        print(f"Annotated           : {rep['annotated_count']}")
        print(f"Passed Validation   : {rep['passed_validation_count']}")
        print(f"Skipped             : {rep['skipped_count']}")
        print(f"Pending             : {rep['pending_count']}")
        print(f"Progress            : {rep['progress_percentage']}%")
        print("==================================================\n")
        return

    # 4. Annotate
    if args.annotate:
        launch_segmentation_tool(
            split=args.split,
            class_name=args.class_name,
            debug_ui=args.debug_ui
        )
        return

    # Default action: show progress and usage
    initialize_segmentation()


if __name__ == "__main__":
    main()
