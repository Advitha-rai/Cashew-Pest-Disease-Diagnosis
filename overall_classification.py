"""
Cashew Pest and Disease Diagnosis System
Phase 10: Overall Pest & Disease Classification Summary CLI Entrypoint (TensorFlow / Keras / OpenPyXL)

Usage Examples:
  # 1. Run complete Phase 10 Overall Pest & Disease Summary report generation:
  python overall_classification.py --all

  # 2. Run default summary execution:
  python overall_classification.py
"""

import argparse
import sys
import json
from src.config import Config
from src.overall_classification import run_phase10_overall_classification_pipeline

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Phase 10 Overall Pest & Disease Summary Engine")
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete Phase 10 Overall Pest & Disease classification summary and export Excel workbook"
    )

    args = parser.parse_args()

    print("[CLI] Running Phase 10 Overall Pest & Disease Classification Summary Engine...")
    summary = run_phase10_overall_classification_pipeline()
    print("\n--- PHASE 10 OVERALL CLASSIFICATION PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()
