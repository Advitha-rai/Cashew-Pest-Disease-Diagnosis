"""
Cashew Pest and Disease Diagnosis System
Hyperparameter Audit Repair & Evidence Reconstruction CLI Entrypoint (TensorFlow / Keras / OpenPyXL)

Usage Examples:
  # 1. Run complete hyperparameter audit repair and evidence reconstruction:
  python hyperparameter_audit_repair.py --all

  # 2. Run default audit repair execution:
  python hyperparameter_audit_repair.py
"""

import argparse
import sys
import json
from src.config import Config
from src.hyperparameter_audit_repair import audit_and_repair_hyperparameters

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Hyperparameter Audit Repair Engine")
    
    parser.add_argument(
        "--all", action="store_true",
        help="Run complete hyperparameter audit repair, evidence trace reconstruction, and multi-sheet Excel generation"
    )

    args = parser.parse_args()

    print("[CLI] Running Hyperparameter Audit Repair & Evidence Reconstruction Engine...")
    summary = audit_and_repair_hyperparameters()
    print("\n--- HYPERPARAMETER AUDIT REPAIR COMPLETE ---")

if __name__ == "__main__":
    main()
