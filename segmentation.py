"""
Cashew Pest and Disease Diagnosis System
Phase C: Manual Segmentation Annotation Preparation CLI (TensorFlow / Keras)

Usage Examples:
  # 1. Run segmentation dataset audit (Phase A/C):
  python segmentation.py --audit

  # 2. Build or refresh 5,734-image annotation manifest:
  python segmentation.py --manifest

  # 3. Perform strict validation on all created manual masks:
  python segmentation.py --validate

  # 4. Display annotation status summary:
  python segmentation.py --summary
"""

import argparse
import sys
import json
import os
import importlib
from src.config import Config
import src.segmentation as segmentation_module
import src.segmentation_tool as segmentation_tool_module

# Dynamically reload modules to prevent stale memory imports in Google Colab / Jupyter
importlib.reload(segmentation_module)
importlib.reload(segmentation_tool_module)

from src.segmentation import (
    audit_segmentation_dataset,
    build_segmentation_annotation_manifest,
    validate_all_manifest_masks
)
from src.segmentation_tool import (
    launch_colab_annotation_interface,
    get_annotation_progress_report,
    run_minimal_colab_callback_test,
    run_phase_c1_end_to_end_test
)

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Segmentation Annotation CLI")
    
    parser.add_argument("--audit", action="store_true", help="Perform segmentation dataset audit & enforce GROUND_TRUTH_MASKS_NOT_FOUND status")
    parser.add_argument("--manifest", action="store_true", help="Build/refresh 5,734-image segmentation annotation manifest")
    parser.add_argument("--validate", action="store_true", help="Run strict quality-control validation on all created manual masks")
    parser.add_argument("--summary", action="store_true", help="Display annotation status summary (Assigned, Annotated, Pending, Validated)")
    parser.add_argument("--annotate", action="store_true", help="Launch interactive Colab/Jupyter manual annotation interface")
    parser.add_argument("--progress", action="store_true", help="Display detailed annotation progress report across eligible Train/Val pool")
    parser.add_argument("--split", type=str, default=None, choices=["Train", "Validation"], help="Filter annotation pool by split (Test split is strictly excluded)")
    parser.add_argument("--class", type=str, dest="class_name", default=None, choices=["Aphids", "Leaf miner", "TMB", "Leaf blight"], help="Filter annotation pool by target class")
    parser.add_argument("--test-callback", action="store_true", help="Run minimal Google Colab callback test button")
    parser.add_argument("--test-tool", action="store_true", help="Run Phase C.1 end-to-end verification test suite")

    args = parser.parse_args()

    # 1. Audit Option
    if args.audit:
        print("[CLI] Running Segmentation Dataset Audit Engine (Phase A & C)...")
        res = audit_segmentation_dataset()
        print("\n--- SEGMENTATION DATASET AUDIT COMPLETE ---")
        print(f"Audit Decision             : {res['audit_status']}")
        print(f"Source Images              : {res['number_of_source_images']}")
        print(f"Ground-Truth Masks Found   : {res['number_of_masks']}")
        print(f"Valid Image-Mask Pairs     : {res['number_of_valid_image_mask_pairs']}")
        print(f"Missing Masks              : {res['number_of_missing_masks']}")
        print(f"Segmentation Training Mode : False (Requires ground-truth mask annotations)\n")
        return

    # 2. Manifest Option
    if args.manifest:
        print("[CLI] Building 5,734-Image Segmentation Annotation Manifest...")
        df_manifest = build_segmentation_annotation_manifest()
        print("\n--- SEGMENTATION ANNOTATION MANIFEST GENERATED ---")
        print(f"Total Assigned Images      : {len(df_manifest)}")
        print(f"Annotated Masks            : {(df_manifest['annotation_status'] == 'ANNOTATED').sum()}")
        print(f"Pending Masks              : {(df_manifest['annotation_status'] == 'PENDING').sum()}")
        print(f"Manifest CSV               : Experiments/Segmentation/segmentation_annotation_manifest.csv\n")
        return

    # 3. Validate Option
    if args.validate:
        print("[CLI] Executing Strict Validation Engine on Created Masks...")
        df_updated = validate_all_manifest_masks()
        passed = (df_updated['validation_status'] == 'PASSED').sum()
        failed = (df_updated['validation_status'] == 'FAILED').sum()
        pending = (df_updated['annotation_status'] == 'PENDING').sum()
        print("\n--- MASK QUALITY CONTROL VALIDATION COMPLETE ---")
        print(f"Total Assigned Images      : {len(df_updated)}")
        print(f"Passed Validation          : {passed}")
        print(f"Failed Validation          : {failed}")
        print(f"Pending Annotations        : {pending}\n")
        return

    # 4. Progress Option
    if args.progress or args.summary:
        print("[CLI] Displaying Segmentation Annotation Summary Progress...")
        rep = get_annotation_progress_report()
        print("\n--- SEGMENTATION ANNOTATION PROGRESS REPORT ---")
        print(f"Test Isolation Status      : {rep['test_set_isolation_status']}")
        print(f"Total Eligible Images      : {rep['total_eligible_images']} (Train=4013, Val=860)")
        print(f"Isolated Test Images       : {rep['test_images_isolated']} (Read-Only)")
        print(f"Annotated Masks            : {rep['annotated_count']}")
        print(f"Passed Validation          : {rep['passed_validation_count']}")
        print(f"Pending Annotations        : {rep['pending_count']}")
        print(f"Progress Percentage        : {rep['progress_percentage']}%\n")
        return

    # 5. Minimal Callback Test Option
    if args.test_callback:
        print("[CLI] Launching Minimal Google Colab Callback Communication Test...")
        run_minimal_colab_callback_test()
        return

    # 6. End-to-End Test Option
    if args.test_tool:
        print("[CLI] Running Phase C.1 End-to-End Verification Test Suite...")
        run_phase_c1_end_to_end_test()
        return

    # 7. Annotate Option
    if args.annotate:
        print(f"[CLI] Launching Interactive Colab Annotation Tool (Split={args.split}, Class={args.class_name})...")
        launch_colab_annotation_interface(split=args.split, class_name=args.class_name)
        return

    # Default action: Run dry-run annotation tool setup
    print("[CLI] Executing default segmentation annotation tool setup...")
    rep = get_annotation_progress_report()
    print(f"\nEligible Train/Val Images : {rep['total_eligible_images']}")
    print(f"Pending Annotations       : {rep['pending_count']}")
    print(f"To launch annotation tool, use: python segmentation.py --annotate\n")

if __name__ == "__main__":
    main()


