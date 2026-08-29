"""
Cashew Pest and Disease Diagnosis System
Model #7 Pre-Training Safety Verification Script
"""

import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

import tensorflow as tf
from tensorflow import keras
from src.config import Config
from src.models import build_keras_model, unfreeze_model_backbone

def run_safety_check():
    print("[SAFETY CHECK] Initializing Model #7 safety verification...")
    
    # 1. Model index = 7, Key = mobilenet_v3_large
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]
    assert model_index == 7, f"Expected model index 7, got {model_index}"
    assert model_key == "mobilenet_v3_large", f"Expected key mobilenet_v3_large, got {model_key}"

    # 2. Build Stage 1 Model (Frozen Backbone)
    model = build_keras_model(
        model_index=model_index,
        num_classes=4,
        input_shape=(224, 224, 3),
        trainable_backbone=False
    )
    
    # Verify model output shape
    assert model.output_shape == (None, 4), f"Expected output shape (None, 4), got {model.output_shape}"

    # Identify backbone
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break
    assert base_model is not None, "Could not find nested base backbone model!"
    
    # 3. Verify total backbone layers and total BatchNorm layers
    total_backbone_layers = len(base_model.layers)
    assert total_backbone_layers == 187, f"Expected 187 backbone layers, got {total_backbone_layers}"
    
    bn_layers = [
        layer for layer in base_model.layers 
        if isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in layer.__class__.__name__
        or "BatchNorm" in layer.__class__.__name__
    ]
    total_bn_layers = len(bn_layers)
    assert total_bn_layers == 46, f"Expected 46 BatchNorm layers, got {total_bn_layers}"

    # 4. Verify Stage 1 trainable backbone layers == 0
    stage1_trainable_backbone_layers = sum(1 for layer in base_model.layers if layer.trainable)
    assert stage1_trainable_backbone_layers == 0, f"Stage 1 trainable backbone layers must be 0, got {stage1_trainable_backbone_layers}"

    # 5. Apply controlled Stage 2 unfreezing
    model = unfreeze_model_backbone(model, model_name_key=model_key)

    # 6. Verify Stage 2 properties
    stage2_trainable_backbone_layers = sum(1 for layer in base_model.layers if layer.trainable)
    trainable_bn_layers = sum(
        1 for layer in base_model.layers 
        if layer.trainable and (
            isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
            or "BatchNormalization" in layer.__class__.__name__
            or "BatchNorm" in layer.__class__.__name__
        )
    )

    assert stage2_trainable_backbone_layers > 0, "Stage 2 trainable backbone layers must be > 0"
    assert stage2_trainable_backbone_layers <= 40, f"Stage 2 trainable backbone layers must be <= 40, got {stage2_trainable_backbone_layers}"
    assert trainable_bn_layers == 0, f"Trainable BatchNorm layers must equal 0, got {trainable_bn_layers}"

    # 7. Verify layers 0 through 146 are all frozen
    for idx in range(147):
        assert not base_model.layers[idx].trainable, f"Layer {idx} ({base_model.layers[idx].name}) should be frozen, but trainable=True!"

    # 8. Print Safety Check Summary
    print("\n============================================================")
    print("MODEL #7 SAFETY CHECK: PASS")
    print("============================================================")
    print("\nMobileNetV3Large backbone:")
    print(f"{total_backbone_layers} layers\n")
    print("BatchNormalization layers:")
    print(f"{total_bn_layers}\n")
    print("Stage 1 trainable backbone layers:")
    print(f"{stage1_trainable_backbone_layers}\n")
    print("Controlled Stage 2 trainable backbone layers:")
    print(f"{stage2_trainable_backbone_layers}\n")
    print("Trainable BatchNormalization layers:")
    print(f"{trainable_bn_layers}\n")
    print("Earlier layers 0–146:")
    print("FROZEN\n")
    print("Output classes:")
    print("4\n")
    print("READY FOR RETRAINING: YES")
    print("============================================================\n")

if __name__ == "__main__":
    run_safety_check()
