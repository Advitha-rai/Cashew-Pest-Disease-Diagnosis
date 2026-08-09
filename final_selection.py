"""
Cashew Pest and Disease Diagnosis System
Phase 7: Final Model Selection and Deployment Readiness CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run complete Phase 7 final model selection and deployment readiness analysis:
  python final_selection.py

  # 2. Run analysis with explicit CLI flag:
  python final_selection.py --analyze
"""

import argparse
import sys
import json
from src.config import Config
from src.final_selection import run_final_model_selection

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Final Model Selection Engine")
    
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run Phase 7 final model selection, multi-criteria ranking, and deployment readiness analysis"
    )

    args = parser.parse_args()

    print("[CLI] Running Phase 7 Final Model Selection and Deployment Readiness Analysis...")
    result = run_final_model_selection()
    print("\n--- PHASE 7 SELECTION COMPLETE ---")

if __name__ == "__main__":
    main()
