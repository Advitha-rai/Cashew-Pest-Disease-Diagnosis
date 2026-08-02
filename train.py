"""
Cashew Pest and Disease Diagnosis System
Phase 3: Main Training CLI Entrypoint (TensorFlow / Keras)

Usage:
  python train.py --model 1 --epochs 50 --lr 0.0001 --optimizer adam
"""

import argparse
from src.train import train_model
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Cashew Pest and Disease Diagnosis Keras Model Trainer")
    
    parser.add_argument(
        "--model", type=int, default=1, choices=range(1, len(Config.MODEL_MAP) + 1),
        help="Model selection index (1 to 8): 1=MobileNetV2, 2=ResNet50, 3=VGG16, 4=InceptionV3, 5=DenseNet121, 6=EfficientNetV2B0, 7=MobileNetV3Large, 8=ConvNeXtTiny"
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS, help="Maximum training epochs (default: 50)")
    parser.add_argument("--warmup-epochs", type=int, default=Config.WARMUP_EPOCHS, help="Warmup epochs with frozen backbone (default: 5)")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE, help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--fine-tune-lr", type=float, default=Config.FINE_TUNE_LEARNING_RATE, help="Fine-tuning learning rate (default: 0.00001)")
    parser.add_argument("--optimizer", type=str, default=Config.OPTIMIZER, choices=["adam", "adamw", "sgd"], help="Optimizer selection (default: adam)")
    parser.add_argument("--loss", type=str, default="categorical_crossentropy", choices=["categorical_crossentropy", "focal_loss"], help="Loss function selection")
    parser.add_argument("--patience", type=int, default=Config.PATIENCE, help="Early stopping patience (default: 10)")
    
    args = parser.parse_args()
    
    train_model(
        model_index=args.model,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        fine_tune_lr=args.fine_tune_lr,
        optimizer_name=args.optimizer,
        loss_name=args.loss,
        patience=args.patience
    )

if __name__ == "__main__":
    main()
