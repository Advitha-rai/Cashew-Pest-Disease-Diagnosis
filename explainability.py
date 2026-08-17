"""
Cashew Pest and Disease Diagnosis System
Phase 8: Explainability, Grad-CAM, and Model Integration Preparation CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run complete Phase 8 Explainability & Grad-CAM pipeline:
  python explainability.py --all

  # 2. Run default explainability execution:
  python explainability.py
"""

import argparse
import sys
import json
from src.config import Config
from src.explainability import run_phase8_explainability_pipeline

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Phase 8 Explainability Engine")
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete Phase 8 Grad-CAM, ensemble explainability fusion, and misclassification error analysis"
    )

    args = parser.parse_args()

    print("[CLI] Running Phase 8 Explainability and Grad-CAM Pipeline...")
    summary = run_phase8_explainability_pipeline()
    print("\n--- PHASE 8 EXPLAINABILITY PIPELINE COMPLETE ---")

if __name__ == "__main__":
    main()
