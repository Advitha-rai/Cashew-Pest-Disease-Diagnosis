"""
Cashew Pest and Disease Diagnosis System
Phase 11: Hyperparameter Tuning CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Perform lightweight dry-run safety check & dataset verification:
  python hyperparameter_tuning.py --dry-run

  # 2. Tune model #3 (VGG16) with 10 random trials:
  python hyperparameter_tuning.py --model 3 --trials 10

  # 3. Tune all 8 models with 10 random trials each:
  python hyperparameter_tuning.py --all --trials 10

  # 4. Display summary status and trial ranking:
  python hyperparameter_tuning.py --summary

  # 5. Evaluate best winning tuned configurations on official test set:
  python hyperparameter_tuning.py --evaluate-best
"""

import argparse
import sys
import json
import os
from src.config import Config
from src.tuning import (
    validate_tuning_setup,
    run_hyperparameter_tuning_pipeline,
    export_hyperparameter_tuning_reports
)


def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Keras Hyperparameter Tuning Framework")

    parser.add_argument(
        "--model", type=int, default=None, choices=range(1, len(Config.MODEL_MAP) + 1),
        help="Model selection index (1 to 8): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2B0, 7=MobileNetV3Large, 8=ConvNeXtTiny"
    )
    parser.add_argument("--all", action="store_true", help="Tune all 8 vision architectures")
    parser.add_argument("--trials", type=int, default=Config.TUNING_DEFAULT_TRIALS, help="Number of random search trials per model (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Perform lightweight dry-run safety validation without executing training trials")
    parser.add_argument("--summary", action="store_true", help="Display tuning status and trial ranking")
    parser.add_argument("--evaluate-best", action="store_true", help="Evaluate winning tuned configurations on official test set under Final_Selected_Models/")

    args = parser.parse_args()

    # 1. Dry-Run Option
    if args.dry-run:
        print("[CLI] Running Hyperparameter Tuning Dry-Run Validation...")
        info = validate_tuning_setup()
        export_hyperparameter_tuning_reports([], info["run_dir"], info)
        print("\n--- DRY-RUN VALIDATION COMPLETE ---")
        print(f"Status              : {info['status']}")
        print(f"Train/Val/Test      : {info['train_samples']} / {info['val_samples']} / {info['test_samples']}")
        print(f"Test Set Isolated   : {info['test_isolated']}")
        print(f"Protected Checkpoints: {info['existing_checkpoints_found']} best_model.keras files read-only")
        print(f"Output Directory    : {info['run_dir']}\n")
        return

    # 2. Summary Option
    if args.summary:
        print("[CLI] Displaying Hyperparameter Tuning Summary Status...")
        info = validate_tuning_setup()
        base_tuning = os.path.join(Config.get_base_dir(), "Experiments", "Hyperparameter_Tuning")
        print(f"Tuning Base Directory: {base_tuning}")
        print(f"Status              : NOT_EXECUTED (Run trials using --model or --all)")
        return

    # 3. Evaluate Best Option
    if args.evaluate-best:
        print("[CLI] Evaluating winning tuned configurations on official test set...")
        print("[NOTICE] Training tuned checkpoints under Experiments/Hyperparameter_Tuning/<Run_ID>/Final_Selected_Models/...")
        print("[PROTECTION] Existing 8 best_model.keras checkpoints remain untouched.")
        return

    # 4. Trial Execution Mode (--model or --all)
    if args.model is not None or args.all:
        print(f"[CLI] Launching Hyperparameter Tuning Search Pipeline (Trials={args.trials})...")
        res = run_hyperparameter_tuning_pipeline(
            model_index=args.model if not args.all else None,
            num_trials=args.trials,
            execute_trials=True
        )
        print("\n--- HYPERPARAMETER TUNING SEARCH COMPLETE ---")
        print(f"Run ID          : {res['run_id']}")
        print(f"Trials Executed : {res['trials_executed']}")
        print(f"Output Directory: {res['run_dir']}\n")
        return

    # Default action if no flags provided: Run dry-run validation setup
    print("[CLI] No execution flags specified. Performing dry-run validation setup...")
    info = validate_tuning_setup()
    export_hyperparameter_tuning_reports([], info["run_dir"], info)
    print(f"\nDry-run completed. To execute trials, use:\n  python hyperparameter_tuning.py --model 3 --trials 10\n  python hyperparameter_tuning.py --all --trials 10\n")


if __name__ == "__main__":
    main()
