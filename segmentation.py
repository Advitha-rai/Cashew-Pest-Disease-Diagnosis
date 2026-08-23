"""
Cashew Pest and Disease Diagnosis System
Segmentation Dataset Audit CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run complete segmentation dataset audit (Phase A):
  python segmentation.py --audit

  # 2. Run default audit execution:
  python segmentation.py
"""

import argparse
import sys
import json
from src.segmentation import audit_segmentation_dataset

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Segmentation Dataset Audit CLI")
    
    parser.add_argument(
        "--audit", action="store_true",
        help="Perform complete segmentation dataset audit, search for ground-truth masks, and generate audit reports"
    )

    args = parser.parse_args()

    print("[CLI] Running Segmentation Dataset Audit Engine (Phase A)...")
    res = audit_segmentation_dataset()
    print("\n--- SEGMENTATION DATASET AUDIT COMPLETE ---")
    print(f"Audit Decision             : {res['audit_status']}")
    print(f"Source Images              : {res['number_of_source_images']}")
    print(f"Ground-Truth Masks Found   : {res['number_of_masks']}")
    print(f"Valid Image-Mask Pairs     : {res['number_of_valid_image_mask_pairs']}")
    print(f"Missing Masks              : {res['number_of_missing_masks']}")
    print(f"Segmentation Training Mode : False (Requires ground-truth mask annotations)\n")

if __name__ == "__main__":
    main()
