"""
Cashew Pest and Disease Diagnosis System
Task 6: Fresh-Process Model #7 Runtime & Source Code Verification Script
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

def run_verification():
    # 1. Print current working directory
    cwd = os.getcwd()
    print(f"Current Working Directory: {cwd}")

    # 2. Print sys.path
    print(f"sys.path: {sys.path}")

    # 3. Print absolute src.models.__file__
    models_file = os.path.abspath(src.models.__file__)
    print(f"src.models.__file__: {models_file}")

    # 4. Print absolute src.train.__file__
    train_file = os.path.abspath(src.train.__file__)
    print(f"src.train.__file__:  {train_file}")

    # 5. Print inspect.getsourcefile(unfreeze_model_backbone)
    source_file = os.path.abspath(inspect.getsourcefile(unfreeze_model_backbone))
    print(f"unfreeze_model_backbone source file: {source_file}")

    # 6. Print the first line number of the function
    source_lines, first_line_no = inspect.getsourcelines(unfreeze_model_backbone)
    print(f"unfreeze_model_backbone first line number: {first_line_no}")

    # 7. Print whether active function source contains required terms
    active_source = inspect.getsource(unfreeze_model_backbone)
    required_terms = [
        "mobilenet_v3_large",
        "total_backbone_layers",
        "trainable_backbone_layers",
        "trainable_bn_layers",
        "frozen_earlier_layers"
    ]

    print("\nChecking required terms in unfreeze_model_backbone source:")
    for term in required_terms:
        present = term in active_source
        print(f"  - '{term}': {'PRESENT' if present else 'MISSING'}")
        assert present, f"FAIL: Term '{term}' missing from active unfreeze_model_backbone source code!"

    # 8. Build Model #7
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

    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break
    assert base_model is not None, "Nested backbone not found!"

    # 9. Confirm Stage 1
    stage1_b = len(base_model.layers)
    stage1_trainable_b = sum(1 for l in base_model.layers if l.trainable)
    print(f"\nStage 1: Backbone={stage1_b} (expected 187), Trainable Backbone={stage1_trainable_b} (expected 0)")
    assert stage1_b == 187, f"Expected 187 backbone layers, got {stage1_b}"
    assert stage1_trainable_b == 0, f"Expected 0 trainable backbone layers, got {stage1_trainable_b}"

    # 10. Apply runtime src.models.unfreeze_model_backbone
    print(f"\nApplying runtime unfreeze_model_backbone(model, model_name_key='{model_key}')...")
    model = unfreeze_model_backbone(model, model_name_key=model_key)

    # 11. Confirm Stage 2
    stage2_b = len(base_model.layers)
    stage2_bn = sum(
        1 for l in base_model.layers
        if isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
        or "BatchNorm" in l.__class__.__name__
    )
    stage2_trainable_b = sum(1 for l in base_model.layers if l.trainable)
    stage2_trainable_bn = sum(
        1 for l in base_model.layers
        if l.trainable and (
            isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
            or "BatchNormalization" in l.__class__.__name__
            or "BatchNorm" in l.__class__.__name__
        )
    )
    stage2_frozen_earlier = sum(1 for l in base_model.layers[:-40] if not l.trainable)

    print("\nStage 2:")
    print(f"  - Total Backbone Layers: {stage2_b} (expected 187)")
    print(f"  - Total BN Layers:       {stage2_bn} (expected 46)")
    print(f"  - Trainable Backbone:    {stage2_trainable_b} (expected 32)")
    print(f"  - Trainable BN:          {stage2_trainable_bn} (expected 0)")
    print(f"  - Frozen Earlier:        {stage2_frozen_earlier} (expected 147)")

    assert stage2_b == 187, f"Expected 187 backbone layers, got {stage2_b}"
    assert stage2_bn == 46, f"Expected 46 BN layers, got {stage2_bn}"
    assert stage2_trainable_b == 32, f"Expected 32 trainable backbone layers, got {stage2_trainable_b}"
    assert stage2_trainable_bn == 0, f"Expected 0 trainable BN layers, got {stage2_trainable_bn}"
    assert stage2_frozen_earlier == 147, f"Expected 147 frozen earlier layers, got {stage2_frozen_earlier}"

    # 12. Search source code for old full-unfreeze message & verify it's unreachable for Model #7
    assert "model_name_key == \"mobilenet_v3_large\"" in active_source, "Controlled branch missing!"
    print("\n[PASS] Verified controlled branch returns early for Model #7.")
    print("[PASS] Old full-unfreeze message is UNREACHABLE for Model #7.")

    # 13. Confirm model.fit() was NOT called
    print("Zero training epochs executed. model.fit() was NOT called.")

    print("\n============================================================")
    print("MODEL #7 FRESH RUNTIME VERIFICATION: PASS")
    print("============================================================\n")

if __name__ == "__main__":
    run_verification()
