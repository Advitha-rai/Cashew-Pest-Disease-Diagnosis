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
from src.config import Config
from src.segmentation import (
    audit_segmentation_dataset,
    build_segmentation_annotation_manifest,
    validate_all_manifest_masks
)

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Segmentation Annotation CLI")
    
    parser.add_argument("--audit", action="store_true", help="Perform segmentation dataset audit & enforce GROUND_TRUTH_MASKS_NOT_FOUND status")
    parser.add_argument("--manifest", action="store_true", help="Build/refresh 5,734-image segmentation annotation manifest")
    parser.add_argument("--validate", action="store_true", help="Run strict quality-control validation on all created manual masks")
    parser.add_argument("--summary", action="store_true", help="Display annotation status summary (Assigned, Annotated, Pending, Validated)")

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

    # 4. Summary Option
    if args.summary:
        print("[CLI] Displaying Segmentation Annotation Summary Status...")
        res = audit_segmentation_dataset()
        print("\n--- SEGMENTATION ANNOTATION SUMMARY ---")
        print(f"Audit Decision             : {res['audit_status']}")
        print(f"Assigned Images            : {res['number_of_source_images']} (Train=4013, Val=860, Test=861)")
        print(f"Annotated Masks            : {res['number_of_masks']}")
        print(f"Pending Annotations        : {res['number_of_missing_masks']}")
        print(f"Valid Image-Mask Pairs     : {res['number_of_valid_image_mask_pairs']}\n")
        return

    # Default action: Run audit and manifest build setup
    print("[CLI] Executing default segmentation annotation preparation setup...")
    res = audit_segmentation_dataset()
    print("\nAnnotation preparation setup complete. Use python segmentation.py --audit, --manifest, or --validate.")

if __name__ == "__main__":
    main()
