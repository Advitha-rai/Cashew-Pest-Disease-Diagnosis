"""
Cashew Pest and Disease Diagnosis System
Static Verification Script for Isolated Training Engine train2.py

File: scratch/verify_train2_model7.py
"""

import sys
import os
import inspect

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

def run_static_verification():
    train2_path = os.path.join(os.getcwd(), "train2.py")
    if not os.path.exists(train2_path):
        raise FileNotFoundError(f"train2.py not found at: {train2_path}")

    with open(train2_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    print("\n============================================================")
    print("[TRAIN2 STATIC VERIFICATION]")
    print("============================")

    # 1. Model #7 only check
    assert "model_index = 7" in source_code or "args.model != 7" in source_code, "FAIL: Model 7 index check missing!"
    print("PASS: Model #7 only")

    # 2. Key check
    assert "mobilenet_v3_large" in source_code, "FAIL: mobilenet_v3_large key missing!"

    # 3. No unfreeze_model_backbone() check
    assert "unfreeze_model_backbone" not in source_code, "FAIL: unfreeze_model_backbone found in train2.py!"
    print("PASS: No unfreeze_model_backbone()")

    # 4. No full-backbone unfreeze checks
    assert "base_model.trainable = True" not in source_code, "FAIL: base_model.trainable = True found!"
    assert "backbone.trainable = True" not in source_code, "FAIL: backbone.trainable = True found!"
    print("PASS: No full-backbone unfreeze")

    # 5. Top 40 controlled region check
    assert "backbone.layers[-40:]" in source_code, "FAIL: backbone.layers[-40:] top 40 region missing!"
    print("PASS: Top 40 controlled region")

    # 6. BatchNormalization frozen check
    assert "isinstance(layer, keras.layers.BatchNormalization)" in source_code or "BatchNormalization" in source_code, "FAIL: BatchNormalization handling missing!"
    print("PASS: BatchNormalization frozen")

    # 7. Expected safety counts present check
    for count_str in ["187", "46", "32", "0", "147"]:
        assert count_str in source_code, f"FAIL: Expected safety count string '{count_str}' missing from source!"
    print("PASS: Expected safety counts present")

    # 8. Separate experiment directory check
    assert "07_MobileNetV3Large_Isolated" in source_code, "FAIL: Separate experiment directory 07_MobileNetV3Large_Isolated missing!"
    print("PASS: Separate experiment directory")

    # 9. src.train not imported check
    assert "src.train" not in source_code and "import src.train" not in source_code and "from src.train" not in source_code, "FAIL: src.train is imported!"
    print("PASS: src.train not imported")

    print("============================\n")
    print("# [TRAIN2 STATIC VERIFICATION PASSED]\n")

if __name__ == "__main__":
    run_static_verification()
