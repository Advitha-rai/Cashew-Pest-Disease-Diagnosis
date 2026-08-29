"""
Cashew Pest and Disease Diagnosis System
Fresh-Process Verification Script for Model #7 Runtime Import & Fine-Tuning Execution
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

def run_runtime_verification():
    print("\n============================================================")
    print("[FRESH PROCESS RUNTIME VERIFICATION] Testing Model #7 Entrypoint...")
    print("============================================================")

    # 1. Check Module Files & Launcher Resolution
    train_file = os.path.abspath(src.train.__file__)
    models_file = os.path.abspath(src.models.__file__)
    launcher_file = os.path.abspath(os.path.join(os.getcwd(), "train.py"))

    print(f"1. src.train.__file__:  {train_file}")
    print(f"2. src.models.__file__: {models_file}")
    print(f"3. Top-level Launcher:  {launcher_file}")

    assert "src" in train_file and "train.py" in train_file, f"Invalid train.py path: {train_file}"
    assert "src" in models_file and "models.py" in models_file, f"Invalid models.py path: {models_file}"

    # Verify top-level launcher code imports src.train
    with open(launcher_file, "r", encoding="utf-8") as f:
        launcher_code = f.read()
    assert "from src.train import train_model" in launcher_code, "Launcher does not import train_model from src.train!"
    print(" [PASS] Top-level launcher imports train_model from src.train.")

    # 2. Check unfreeze_model_backbone Source Code
    source_file = inspect.getsourcefile(unfreeze_model_backbone)
    source_code = inspect.getsource(unfreeze_model_backbone)
    sig = inspect.signature(unfreeze_model_backbone)

    print(f"\n4. unfreeze_model_backbone Source File: {source_file}")
    print(f"5. unfreeze_model_backbone Signature:   {sig}")

    assert source_file is not None and "models.py" in source_file, f"Invalid unfreeze source file: {source_file}"
    assert "model_name_key" in sig.parameters, "Signature missing model_name_key parameter!"
    assert "MobileNetV3Large controlled fine-tuning" in source_code, "Source code does NOT contain controlled fine-tuning implementation!"
    print(" [PASS] Source code contains controlled MobileNetV3Large fine-tuning.")

    # 3. Model #7 Stage 1 Verification
    model_index = 7
    folder_name, model_key = Config.MODEL_MAP[model_index]
    assert model_key == "mobilenet_v3_large", f"Expected mobilenet_v3_large, got {model_key}"

    print(f"\nConstructing Model #{model_index} ({folder_name}) Stage 1...")
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
    assert base_model is not None, "Could not find base model backbone!"

    stage1_b = len(base_model.layers)
    stage1_trainable_b = sum(1 for l in base_model.layers if l.trainable)
    stage1_bn = sum(
        1 for l in base_model.layers
        if isinstance(l, (tf.keras.layers.BatchNormalization, keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
        or "BatchNorm" in l.__class__.__name__
    )

    print(f"Stage 1: Backbone={stage1_b} (exp 187), Trainable Backbone={stage1_trainable_b} (exp 0), BN={stage1_bn} (exp 46)")
    assert stage1_b == 187, f"Expected 187, got {stage1_b}"
    assert stage1_trainable_b == 0, f"Expected 0, got {stage1_trainable_b}"
    assert stage1_bn == 46, f"Expected 46, got {stage1_bn}"

    # 4. Model #7 Stage 2 Unfreeze Execution & Message Capture
    print("\nExecuting Stage 2 unfreeze_model_backbone(model, model_name_key='mobilenet_v3_large')...")
    
    # Redirect stdout to capture prints
    import io
    from contextlib import redirect_stdout
    f_out = io.StringIO()
    with redirect_stdout(f_out):
        model = unfreeze_model_backbone(model, model_name_key=model_key)
    output_text = f_out.getvalue()
    print(output_text)

    # Verify old full-unfreeze message NEVER appears and controlled message DOES appear
    assert "[FINE-TUNING] MobileNetV3Large controlled fine-tuning" in output_text, "Controlled fine-tuning log line missing!"
    assert "[FINE-TUNING] Unfroze all backbone layers for full fine-tuning." not in output_text, "FAIL: Old full-unfreeze message was printed!"
    print(" [PASS] Message '[FINE-TUNING] MobileNetV3Large controlled fine-tuning' printed.")
    print(" [PASS] Message '[FINE-TUNING] Unfroze all backbone layers for full fine-tuning.' NOT printed.")

    # 5. Model #7 Stage 2 Layer Counts Verification
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

    print("\nStage 2 Layer Counts:")
    print(f"  - Total Backbone Layers: {stage2_b} (expected: 187)")
    print(f"  - Total BN Layers:       {stage2_bn} (expected: 46)")
    print(f"  - Trainable Backbone:    {stage2_trainable_b} (expected: 32)")
    print(f"  - Trainable BN:          {stage2_trainable_bn} (expected: 0)")
    print(f"  - Frozen Earlier:        {stage2_frozen_earlier} (expected: 147)")

    assert stage2_b == 187, f"Expected 187, got {stage2_b}"
    assert stage2_bn == 46, f"Expected 46, got {stage2_bn}"
    assert stage2_trainable_b == 32, f"Expected 32, got {stage2_trainable_b}"
    assert stage2_trainable_bn == 0, f"Expected 0, got {stage2_trainable_bn}"
    assert stage2_frozen_earlier == 147, f"Expected 147, got {stage2_frozen_earlier}"

    # 6. Confirm no training epochs were run
    print("\nZero training epochs executed. model.fit() was NOT called.")

    print("\n============================================================")
    print("RUN-TIME DIAGNOSTICS & IMPORT VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("============================================================\n")

if __name__ == "__main__":
    run_runtime_verification()
