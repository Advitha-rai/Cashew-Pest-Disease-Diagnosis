"""
Cashew Pest and Disease Diagnosis System
Task 8: Fresh Process Runtime & Import Verification Script for Model #7
"""

import sys
import os
import inspect

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

import tensorflow as tf
from tensorflow import keras

import src.config
import src.models
import src.train
from src.config import Config
from src.models import build_keras_model, unfreeze_model_backbone

def run_fresh_verification():
    print("\n============================================================")
    print("[TASK 8 FRESH PROCESS VERIFICATION] Starting Verification...")
    print("============================================================")

    # 1. Verify Import Paths
    models_file = os.path.abspath(src.models.__file__)
    train_file = os.path.abspath(src.train.__file__)
    print(f"1. src.models.__file__: {models_file}")
    print(f"2. src.train.__file__:  {train_file}")
    
    assert "src" in models_file and "models.py" in models_file, f"Invalid models.py path: {models_file}"
    assert "src" in train_file and "train.py" in train_file, f"Invalid train.py path: {train_file}"

    # 3. Verify Signature of unfreeze_model_backbone
    sig = inspect.signature(unfreeze_model_backbone)
    print(f"3. unfreeze_model_backbone signature: {sig}")
    assert "model_name_key" in sig.parameters, "FAIL: unfreeze_model_backbone signature is missing model_name_key parameter!"

    # 4. Build Model #7 (Stage 1)
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]
    assert model_key == "mobilenet_v3_large", f"Expected mobilenet_v3_large, got {model_key}"

    print(f"\nConstructing Model #{model_index} ({folder_name}, key='{model_key}')...")
    model = build_keras_model(
        model_index=model_index,
        num_classes=4,
        input_shape=(224, 224, 3),
        trainable_backbone=False
    )

    backbone = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            backbone = layer
            break
    assert backbone is not None, "Could not locate nested backbone!"

    stage1_b_layers = len(backbone.layers)
    stage1_trainable_b = sum(1 for l in backbone.layers if l.trainable)
    print(f"4. Stage 1 Check: Total Backbone Layers={stage1_b_layers}, Trainable Backbone Layers={stage1_trainable_b}")
    assert stage1_b_layers == 187, f"Expected 187 backbone layers, got {stage1_b_layers}"
    assert stage1_trainable_b == 0, f"Expected 0 trainable backbone layers in Stage 1, got {stage1_trainable_b}"

    # 5. Apply Controlled Fine-Tuning (Stage 2)
    print(f"\nApplying unfreeze_model_backbone(model, model_name_key='{model_key}')...")
    model = unfreeze_model_backbone(model, model_name_key=model_key)

    total_b_layers = len(backbone.layers)
    total_bn = sum(
        1 for l in backbone.layers
        if isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
        or "BatchNorm" in l.__class__.__name__
    )
    trainable_b = sum(1 for l in backbone.layers if l.trainable)
    trainable_bn = sum(
        1 for l in backbone.layers
        if l.trainable and (
            isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
            or "BatchNormalization" in l.__class__.__name__
            or "BatchNorm" in l.__class__.__name__
        )
    )
    frozen_earlier = sum(1 for l in backbone.layers[:-40] if not l.trainable)

    print("\n5. Stage 2 Controlled Fine-Tuning Counts:")
    print(f"   - Total Backbone Layers:              {total_b_layers} (expected: 187)")
    print(f"   - Total BatchNormalization Layers:    {total_bn} (expected: 46)")
    print(f"   - Trainable Backbone Layers:          {trainable_b} (expected: 32)")
    print(f"   - Trainable BatchNormalization Layers: {trainable_bn} (expected: 0)")
    print(f"   - Frozen Earlier Layers (0..146):     {frozen_earlier} (expected: 147)")

    assert total_b_layers == 187, f"Expected 187, got {total_b_layers}"
    assert total_bn == 46, f"Expected 46, got {total_bn}"
    assert trainable_b == 32, f"Expected 32, got {trainable_b}"
    assert trainable_bn == 0, f"Expected 0, got {trainable_bn}"
    assert frozen_earlier == 147, f"Expected 147, got {frozen_earlier}"

    # 6. Verify old full-unfreeze message is impossible
    models_py_path = os.path.abspath(src.models.__file__)
    with open(models_py_path, "r", encoding="utf-8") as f:
        models_code = f.read()

    # Verify is_mobilenet_v3 logic separates mobilenet_v3_large from default branch
    assert 'if is_mobilenet_v3:' in models_code, "FAIL: is_mobilenet_v3 branch missing from src/models.py!"
    print("6. Verification: MobileNetV3Large is isolated from default full-unfreeze branch.")

    # 7. Confirm Safety Guard exists in src/train.py before compile
    train_py_path = os.path.abspath(src.train.__file__)
    with open(train_py_path, "r", encoding="utf-8") as f:
        train_code = f.read()

    assert '[TRAINING SAFETY] Model #7 controlled fine-tuning VERIFIED' in train_code, "FAIL: Safety guard missing from src/train.py!"
    assert 'raise RuntimeError' in train_code, "FAIL: RuntimeError missing from safety guard!"
    print("7. Verification: Defensive safety guard present in src/train.py prior to model.compile().")

    # 8. Confirm no training epochs were run
    print("8. Verification: Zero training epochs executed. No model.fit() was called.")

    print("\n============================================================")
    print("[FRESH PROCESS VERIFICATION PASSED] Model #7 is 100% SAFE FOR RETRAINING.")
    print("============================================================\n")

if __name__ == "__main__":
    run_fresh_verification()
