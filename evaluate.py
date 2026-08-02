"""
Cashew Pest and Disease Diagnosis System
Phase 4: Main Model Evaluation CLI Entrypoint (TensorFlow / Keras)

Usage Examples:
  # Evaluate a single model by index (1 to 8):
  python evaluate.py --model 1

  # Evaluate all 8 vision models and generate global cross-model comparison report & rankings:
  python evaluate.py --all
"""

import argparse
import sys
from src.config import Config
from src.evaluate import evaluate_single_model, evaluate_all_models

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Diagnosis Model Evaluator")
    
    parser.add_argument(
        "--model", type=int, default=None, choices=range(1, len(Config.MODEL_MAP) + 1),
        help="Model selection index (1 to 8): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2B0, 7=MobileNetV3Large, 8=ConvNeXtTiny"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Evaluate all trained models and generate cross-model comparative ranking reports in Experiments/Comparison/"
    )

    args = parser.parse_args()

    if args.all:
        print("[CLI] Running global evaluation and benchmarking across all trained models...")
        evaluate_all_models()
    elif args.model is not None:
        print(f"[CLI] Evaluating model index #{args.model} ({Config.MODEL_MAP[args.model][0]})...")
        evaluate_single_model(args.model)
    else:
        print("[NOTICE] No model specified. Defaulting to evaluating all models...")
        evaluate_all_models()

if __name__ == "__main__":
    main()
