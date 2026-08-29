"""
Cashew Pest and Disease Diagnosis System
Task 10: Final Fresh-Process Runtime & Import Verification Script for Model #7
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

def run_final_verification():
    print("\n============================================================")
    print("FINAL FRESH-PROCESS RUNTIME & IMPORT VERIFICATION")
    print("============================================================")

    # 1. Module File Paths
    train_file = os.path.abspath(src.train.__file__)
    models_file = os.path.abspath(src.models.__file__)
    print(f"src.train  -> {train_file}")
    print(f"src.models -> {models_file}")

    assert "src" in train_file and "train.py" in train_file, f"Unexpected train.py path: {train_file}"
    assert "src" in models_file and "models.py" in models_file, f"Unexpected models.py path: {models_file}"

    # 2. Function Signature Check
    sig = inspect.signature(unfreeze_model_backbone)
    print(f"\nunfreeze_model_backbone signature: {sig}")
    assert "model_name_key" in sig.parameters, "FAIL: Signature missing model_name_key parameter!"

    # 3. Model #7 Stage 1 Counts
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]
    assert model_key == "mobilenet_v3_large", f"Expected mobilenet_v3_large, got {model_key}"

    print(f"\nBuilding Model #{model_index} ({folder_name}) Stage 1...")
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
    assert backbone is not None, "Nested backbone not found!"

    stage1_b_layers = len(backbone.layers)
    stage1_trainable_b = sum(1 for l in backbone.layers if l.trainable)

    print("\nStage 1:")
    print(f"Backbone = {stage1_b_layers}")
    print(f"Trainable backbone = {stage1_trainable_b}")

    assert stage1_b_layers == 187, f"Expected Stage 1 backbone=187, got {stage1_b_layers}"
    assert stage1_trainable_b == 0, f"Expected Stage 1 trainable=0, got {stage1_trainable_b}"

    # 4. Model #7 Stage 2 Counts
    print(f"\nApplying unfreeze_model_backbone(model, model_name_key='{model_key}')...")
    model = unfreeze_model_backbone(model, model_name_key=model_key)

    stage2_b_layers = len(backbone.layers)
    stage2_bn = sum(
        1 for l in backbone.layers
        if isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
        or "BatchNorm" in l.__class__.__name__
    )
    stage2_trainable_b = sum(1 for l in backbone.layers if l.trainable)
    stage2_trainable_bn = sum(
        1 for l in backbone.layers
        if l.trainable and (
            isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
            or "BatchNormalization" in l.__class__.__name__
            or "BatchNorm" in l.__class__.__name__
        )
    )
    stage2_frozen_earlier = sum(1 for l in backbone.layers[:-40] if not l.trainable)

    print("\nStage 2:")
    print(f"Backbone = {stage2_b_layers}")
    print(f"BN = {stage2_bn}")
    print(f"Trainable backbone = {stage2_trainable_b}")
    print(f"Trainable BN = {stage2_trainable_bn}")
    print(f"Frozen earlier = {stage2_frozen_earlier}")

    assert stage2_b_layers == 187, f"Expected Stage 2 backbone=187, got {stage2_b_layers}"
    assert stage2_bn == 46, f"Expected Stage 2 BN=46, got {stage2_bn}"
    assert stage2_trainable_b == 32, f"Expected Stage 2 trainable backbone=32, got {stage2_trainable_b}"
    assert stage2_trainable_bn == 0, f"Expected Stage 2 trainable BN=0, got {stage2_trainable_bn}"
    assert stage2_frozen_earlier == 147, f"Expected Stage 2 frozen earlier=147, got {stage2_frozen_earlier}"

    # 5. Top-Level Launcher Check
    launcher_path = os.path.join(os.getcwd(), "train.py")
    with open(launcher_path, "r", encoding="utf-8") as f:
        launcher_code = f.read()
    assert "from src.train import train_model" in launcher_code, "Launcher does not import train_model from src.train!"
    assert "def train_model(" not in launcher_code, "Launcher contains duplicated training engine code!"
    print(f"\nTop-Level Launcher ({launcher_path}): Verified (imports train_model from src.train).")

    # 6. Confirm no training epochs were run
    print("\nZero training epochs executed. model.fit() was NOT called.")

    print("\n============================================================")
    print("VERIFICATION COMPLETED SUCCESSFULLY: Model #7 is 100% Ready.")
    print("============================================================\n")

if __name__ == "__main__":
    run_final_verification()
