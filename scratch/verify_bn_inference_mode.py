"""
Cashew Pest and Disease Diagnosis System
Fresh-Process Verification & BN Moving-Statistics Test for Model #7
"""

import sys
import os
import inspect
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

import tensorflow as tf
from tensorflow import keras

import src.config
import src.models
import src.train
from src.config import Config
from src.models import build_keras_model, unfreeze_model_backbone

def run_bn_verification():
    print("\n============================================================")
    print("FRESH-PROCESS MODEL #7 BN MOVING-STATISTICS VERIFICATION")
    print("============================================================")

    # 1. Build Model #7 Stage 1
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

    stage1_b_layers = len(base_model.layers)
    stage1_trainable_b = sum(1 for l in base_model.layers if l.trainable)
    stage1_bn_count = sum(
        1 for l in base_model.layers
        if isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
        or "BatchNorm" in l.__class__.__name__
    )

    print("\nStage 1 Target Check:")
    print(f"  - Total Backbone Layers: {stage1_b_layers} (expected: 187)")
    print(f"  - Trainable Backbone Layers: {stage1_trainable_b} (expected: 0)")
    print(f"  - Total BN Layers: {stage1_bn_count} (expected: 46)")

    assert stage1_b_layers == 187, f"Expected Stage 1 backbone=187, got {stage1_b_layers}"
    assert stage1_trainable_b == 0, f"Expected Stage 1 trainable=0, got {stage1_trainable_b}"
    assert stage1_bn_count == 46, f"Expected Stage 1 BN=46, got {stage1_bn_count}"

    # 2. Stage 2 Controlled Fine-Tuning
    print(f"\nApplying unfreeze_model_backbone(model, model_name_key='{model_key}')...")
    model = unfreeze_model_backbone(model, model_name_key=model_key)

    stage2_b_layers = len(base_model.layers)
    stage2_bn_count = sum(
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

    print("\nStage 2 Target Check:")
    print(f"  - Total Backbone Layers: {stage2_b_layers} (expected: 187)")
    print(f"  - Total BN Layers: {stage2_bn_count} (expected: 46)")
    print(f"  - Trainable Backbone Layers: {stage2_trainable_b} (expected: 32)")
    print(f"  - Trainable BN Layers: {stage2_trainable_bn} (expected: 0)")
    print(f"  - Frozen Earlier Layers (0..146): {stage2_frozen_earlier} (expected: 147)")

    assert stage2_b_layers == 187, f"Expected Stage 2 backbone=187, got {stage2_b_layers}"
    assert stage2_bn_count == 46, f"Expected Stage 2 BN=46, got {stage2_bn_count}"
    assert stage2_trainable_b == 32, f"Expected Stage 2 trainable backbone=32, got {stage2_trainable_b}"
    assert stage2_trainable_bn == 0, f"Expected Stage 2 trainable BN=0, got {stage2_trainable_bn}"
    assert stage2_frozen_earlier == 147, f"Expected Stage 2 frozen earlier=147, got {stage2_frozen_earlier}"

    # 3. BN Moving Statistics Test
    print("\nExecuting BN Moving-Statistics Stability Test during Stage 2 Training Forward Pass...")
    bn_stats_before = {}
    for layer in base_model.layers:
        if isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization)) or "BatchNormalization" in layer.__class__.__name__ or "BatchNorm" in layer.__class__.__name__:
            bn_stats_before[layer.name] = (
                layer.moving_mean.numpy().copy(),
                layer.moving_variance.numpy().copy()
            )

    dummy_input = tf.random.normal((8, 224, 224, 3))
    
    with tf.GradientTape() as tape:
        preds = model(dummy_input, training=True)
        loss = tf.reduce_mean(preds)

    grads = tape.gradient(loss, model.trainable_variables)

    # Check BN stats drift
    bn_drift_count = 0
    for name, (mean_before, var_before) in bn_stats_before.items():
        layer = base_model.get_layer(name)
        mean_after = layer.moving_mean.numpy()
        var_after = layer.moving_variance.numpy()

        diff_mean = np.max(np.abs(mean_before - mean_after))
        diff_var = np.max(np.abs(var_before - var_after))

        if diff_mean > 0.0 or diff_var > 0.0:
            bn_drift_count += 1
            print(f"[FAIL] BN Layer '{name}' drift detected: max_diff_mean={diff_mean}, max_diff_var={diff_var}")

    print(f"Backbone BN layers with updated moving statistics: {bn_drift_count} / {len(bn_stats_before)}")
    assert bn_drift_count == 0, f"FAIL: {bn_drift_count} BatchNorm layers updated moving statistics!"

    # 4. Confirm Trainable Variables for 32 Non-BN Layers
    trainable_var_tensors = model.trainable_variables
    print(f"\nTrainable variable weight tensors in model: {len(trainable_var_tensors)}")
    assert len(trainable_var_tensors) > 0, "FAIL: Model has 0 trainable variable tensors!"

    # 5. Confirm no training was started
    print("\nZero training epochs executed. model.fit() was NOT called.")

    print("\n============================================================")
    print("[BN MOVING-STATISTICS TEST PASSED] MobileNetV3Large BN is 100% Stable!")
    print("============================================================\n")

if __name__ == "__main__":
    run_bn_verification()
