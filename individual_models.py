"""
Cashew Pest and Disease Diagnosis System
Phase 6: Individual Model Complete Dataset Classification CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # 1. Run complete dataset classification across ALL 8 vision architectures:
  python individual_models.py --full-dataset

  # 2. Run complete dataset classification for a single model by index (1 to 8):
  python individual_models.py --model 1

  # 3. Run complete dataset classification across ALL 8 models (alias for --full-dataset):
  python individual_models.py --all
"""

import argparse
import sys
import json
from src.config import Config
from src.individual_models import (
    evaluate_single_model_full_dataset,
    evaluate_all_models_full_dataset
)

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Individual Models Complete Dataset Evaluator")
    
    parser.add_argument(
        "--model", type=int, default=None, choices=range(1, len(Config.MODEL_MAP) + 1),
        help="Model selection index (1 to 8): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2B0, 7=MobileNetV3Large, 8=ConvNeXtTiny"
    )
    parser.add_argument(
        "--full-dataset", action="store_true",
        help="Run descriptive classification on the complete 5,734-image dataset across all 8 vision models"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all 8 vision models on the complete dataset (alias for --full-dataset)"
    )

    args = parser.parse_args()

    if args.model is not None:
        folder_name, _ = Config.MODEL_MAP[args.model]
        print(f"[CLI] Running complete dataset classification for Model #{args.model} ({folder_name})...")
        evaluate_single_model_full_dataset(args.model)
    elif args.full_dataset or args.all:
        print("[CLI] Running complete dataset classification across ALL 8 vision models...")
        evaluate_all_models_full_dataset()
    else:
        print("[NOTICE] No model specified. Defaulting to evaluating all 8 vision models on the complete dataset...")
        evaluate_all_models_full_dataset()

if __name__ == "__main__":
    main()
