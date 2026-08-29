"""
Static Verification Script for src/train.py Model #7 Safety Guard
"""

import os
import sys

def verify_train_py():
    train_py_path = os.path.join(os.getcwd(), "src", "train.py")
    assert os.path.exists(train_py_path), f"File not found: {train_py_path}"
    
    with open(train_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("[STATIC VERIFICATION] Checking src/train.py...")

    # 1. Check model_name_key=model_key exists in Stage 2 call
    assert "model_name_key=model_key" in content, "FAIL: model_name_key=model_key not found in Stage 2 call!"
    print(" [PASS] Explicit model_name_key=model_key present in Stage 2 call.")

    # 2. Check safety guard block exists
    assert 'if model_key == "mobilenet_v3_large":' in content, "FAIL: Safety guard block for mobilenet_v3_large not found!"
    print(" [PASS] Model #7 safety guard block present.")

    # 3. Check expected values 187, 46, 32, 0, and 147 are present in checks
    assert "187" in content and "46" in content and "32" in content and "147" in content, "FAIL: Expected values (187, 46, 32, 147) missing from safety guard checks!"
    print(" [PASS] Expected values (187, 46, 32, 0, 147) present in safety guard checks.")

    # 4. Check exception type is RuntimeError
    assert "raise RuntimeError" in content, "FAIL: raise RuntimeError missing from safety guard!"
    print(" [PASS] RuntimeError exception handling verified.")

    # 5. Check no Model #7 path calls generic full-backbone unfreezing
    assert "model = unfreeze_model_backbone(model)" not in content, "FAIL: Found un-parameterized unfreeze_model_backbone call!"
    print(" [PASS] No un-parameterized full-backbone unfreezing call for Model #7.")

    print("\n============================================================")
    print("STATIC VERIFICATION: ALL CHECKS PASSED SUCCESSFULLY")
    print("============================================================")

if __name__ == "__main__":
    verify_train_py()
