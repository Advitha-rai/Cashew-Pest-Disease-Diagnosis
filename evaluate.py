"""
Cashew Pest and Disease Diagnosis System
Main Evaluation CLI Entrypoint
Usage:
  python evaluate.py --model 1
"""

import argparse
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Model Evaluator")
    parser.add_argument("--model", type=int, default=1, choices=range(1, 11),
                        help="Model index (1 to 10): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2, 7=MobileNetV3, 8=ConvNeXt, 9=Swin, 10=DINOv2")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint filename to evaluate")
    
    args = parser.parse_args()
    evaluate_model(model_index=args.model, checkpoint_name=args.checkpoint)

if __name__ == "__main__":
    main()
