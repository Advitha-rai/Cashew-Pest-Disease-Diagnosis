"""
Test script to inspect BN moving statistics before and after a training=True forward pass.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.models import build_keras_model, unfreeze_model_backbone
from src.config import Config

def test_bn_moving_stats():
    print("\n============================================================")
    print("[BN MOVING STATS TEST] Initializing Model #7 Test...")
    print("============================================================")

    model = build_keras_model(model_index=7, num_classes=4, input_shape=(224, 224, 3), trainable_backbone=False)
    model = unfreeze_model_backbone(model, model_name_key="mobilenet_v3_large")

    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    # Record initial BN moving statistics for all BN layers in base_model
    bn_stats_before = {}
    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization)) or "BatchNormalization" in layer.__class__.__name__ or "BatchNorm" in layer.__class__.__name__:
            mean_val = layer.moving_mean.numpy().copy()
            var_val = layer.moving_variance.numpy().copy()
            bn_stats_before[layer.name] = (mean_val, var_val)

    print(f"Recorded moving statistics for {len(bn_stats_before)} backbone BatchNorm layers.")

    # Record head BN layer stats
    head_bn_layer = None
    for layer in model.layers:
        if layer.name == "head_batch_norm":
            head_bn_layer = layer
            break

    if head_bn_layer:
        head_mean_before = head_bn_layer.moving_mean.numpy().copy()
        head_var_before = head_bn_layer.moving_variance.numpy().copy()

    # Perform forward pass with training=True
    dummy_input = np.random.randn(8, 224, 224, 3).astype(np.float32)
    
    # Run training forward pass and gradient calculation
    with tf.GradientTape() as tape:
        preds = model(dummy_input, training=True)
        loss = tf.reduce_mean(preds)
    
    grads = tape.gradient(loss, model.trainable_variables)
    
    # Check Backbone BN moving statistics after forward pass
    backbone_bn_changed_count = 0
    for name, (mean_before, var_before) in bn_stats_before.items():
        layer = base_model.get_layer(name)
        mean_after = layer.moving_mean.numpy()
        var_after = layer.moving_variance.numpy()
        
        diff_mean = np.max(np.abs(mean_before - mean_after))
        diff_var = np.max(np.abs(var_before - var_after))
        
        if diff_mean > 0 or diff_var > 0:
            backbone_bn_changed_count += 1
            print(f"[BN DRIFT] Layer '{name}' changed! max_diff_mean={diff_mean}, max_diff_var={diff_var}")

    print(f"\nBackbone BN layers with changed moving statistics: {backbone_bn_changed_count} / {len(bn_stats_before)}")
    
    # Check head BN layer stats
    if head_bn_layer:
        head_mean_after = head_bn_layer.moving_mean.numpy()
        head_var_after = head_bn_layer.moving_variance.numpy()
        head_diff_mean = np.max(np.abs(head_mean_before - head_mean_after))
        head_diff_var = np.max(np.abs(head_var_before - head_var_after))
        print(f"Head BN layer ('head_batch_norm') moving statistics diff: mean_diff={head_diff_mean}, var_diff={head_diff_var}")

    # Check trainable variables
    trainable_var_names = [v.name for v in model.trainable_variables]
    print(f"\nTotal trainable variable tensors: {len(trainable_var_names)}")
    bn_trainable_vars = [name for name in trainable_var_names if "batch_normalization" in name.lower() or "bn" in name.lower() or "gamma" in name.lower() or "beta" in name.lower()]
    print(f"Trainable BN variable tensors in model: {bn_trainable_vars}")

    return backbone_bn_changed_count

if __name__ == "__main__":
    test_bn_moving_stats()
