"""
Cashew Pest and Disease Diagnosis System
Phase 9: Grad-CAM Localization Validation and Model Trustworthiness Audit CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run complete Phase 9 Explainability Validation & Model Trustworthiness Audit:
  python explainability_validation.py --all

  # 2. Run default validation execution:
  python explainability_validation.py
"""

import argparse
import sys
import json
from src.config import Config
from src.explainability_validation import run_phase9_validation_pipeline

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Phase 9 Explainability Validation Engine")
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete Phase 9 Grad-CAM localization validation, inter-model agreement, error risk audit, and visualization grids"
    )

    args = parser.parse_args()

    print("[CLI] Running Phase 9 Explainability Validation & Model Trustworthiness Audit...")
    summary = run_phase9_validation_pipeline()
    print("\n--- PHASE 9 EXPLAINABILITY VALIDATION COMPLETE ---")

if __name__ == "__main__":
    main()
