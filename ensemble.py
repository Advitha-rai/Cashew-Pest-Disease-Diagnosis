"""
Cashew Pest and Disease Diagnosis System
Phase 5: Main Ensemble CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run validation inference and optimal weight selection:
  python ensemble.py --validate

  # 2. Run final ensemble evaluation on untouched test set:
  python ensemble.py --test

  # 3. Run complete Phase 5 pipeline (Validation + Weight Search + Test Evaluation + Comparison):
  python ensemble.py --all

  # 4. Perform single image ensemble prediction:
  python ensemble.py --predict path/to/sample_leaf.jpg

  # 5. Run complete dataset descriptive classification (Train + Validation + Test):
  python ensemble.py --full-dataset
"""

import argparse
import sys
import json
from src.config import Config
from src.ensemble import (
    validate_and_search_ensemble_weights,
    evaluate_ensemble_test_set,
    predict_ensemble_single_image,
    evaluate_full_dataset
)

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Soft-Voting Ensemble Engine")
    
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation inference and optimal ensemble weight search on val_split.csv"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run final ensemble evaluation on test_split.csv using validation-selected weights"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete Phase 5 pipeline (Validation + Weight Search + Test Evaluation + Cross-Model Comparison)"
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Path to an image file for single-image ensemble prediction with invalid image & uncertainty protection"
    )
    parser.add_argument(
        "--full-dataset", action="store_true",
        help="Run descriptive classification on the complete dataset (Train + Validation + Test)"
    )

    args = parser.parse_args()

    if args.predict:
        print(f"[CLI] Running ensemble prediction on single image: {args.predict}")
        result = predict_ensemble_single_image(args.predict)
        print("\n--- ENSEMBLE PREDICTION RESULT ---")
        print(json.dumps(result, indent=4))
    elif args.full_dataset:
        print("[CLI] Running complete dataset descriptive classification...")
        evaluate_full_dataset()
    elif args.validate:
        print("[CLI] Running validation weight selection...")
        validate_and_search_ensemble_weights()
    elif args.test:
        print("[CLI] Running test evaluation on finalized ensemble...")
        evaluate_ensemble_test_set()
    elif args.all:
        print("[CLI] Running complete Phase 5 ensemble pipeline...")
        validate_and_search_ensemble_weights()
        evaluate_ensemble_test_set()
    else:
        print("[NOTICE] No action specified. Running complete Phase 5 ensemble pipeline...")
        validate_and_search_ensemble_weights()
        evaluate_ensemble_test_set()

if __name__ == "__main__":
    main()
