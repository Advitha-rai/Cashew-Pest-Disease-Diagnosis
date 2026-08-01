"""
Cashew Pest and Disease Diagnosis System
Main Training CLI Entrypoint
Usage:
  python train.py --model 1 --epochs 50 --lr 0.0001
"""

import argparse
from src.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Diagnosis Model Trainer")
    parser.add_argument("--model", type=int, default=1, choices=range(1, 11),
                        help="Model selection index (1 to 10): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2, 7=MobileNetV3, 8=ConvNeXt, 9=Swin, 10=DINOv2")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    args = parser.parse_args()
    train_model(model_index=args.model, epochs=args.epochs, lr=args.lr, patience=args.patience)

if __name__ == "__main__":
    main()
