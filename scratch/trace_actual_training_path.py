"""
Cashew Pest and Disease Diagnosis System
Definitive Trace Script for Actual Training Entry Point & Unfreeze Function Source
"""

import sys
import os
import inspect

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

import src.config
import src.models
import src.train
from src.train import train_model
from src.models import unfreeze_model_backbone

def run_trace():
    print("\n============================================================")
    print("[DEFINITIVE EXECUTION TRACE]")
    print("============================================================")

    root_script = os.path.abspath(os.path.join(os.getcwd(), "train.py"))
    train_source = os.path.abspath(inspect.getsourcefile(train_model))
    unfreeze_source = os.path.abspath(inspect.getsourcefile(unfreeze_model_backbone))
    unfreeze_module = unfreeze_model_backbone.__module__
    unfreeze_sig = inspect.signature(unfreeze_model_backbone)

    print(f"1. ROOT SCRIPT PATH:                       {root_script}")
    print(f"2. ACTUAL train_model SOURCE:               {train_source}")
    print(f"3. ACTUAL unfreeze_model_backbone SOURCE:   {unfreeze_source}")
    print(f"4. ACTUAL unfreeze_model_backbone MODULE:   {unfreeze_module}")
    print(f"5. ACTUAL unfreeze_model_backbone SIGNATURE:{unfreeze_sig}")
    print("============================================================\n")

    # Search entire workspace for any occurrences of the old full-unfreeze string
    print("Searching repository for old full-unfreeze message string...")
    old_msg = "Unfroze all backbone layers for full fine-tuning."
    matches = []

    for root, dirs, files in os.walk(os.getcwd()):
        if ".git" in root or "__pycache__" in root or ".gemini" in root:
            continue
        for file in files:
            if file.endswith(".py") or file.endswith(".bak"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    for line_idx, line in enumerate(lines, 1):
                        if old_msg in line:
                            matches.append((os.path.relpath(file_path, os.getcwd()), line_idx, line.strip()))
                except Exception:
                    pass

    print(f"Found {len(matches)} occurrences of old full-unfreeze message:")
    for rel_path, line_no, content in matches:
        print(f"  - {rel_path}:{line_no} -> {content}")

    print("\n============================================================")
    print("TRACE COMPLETED: Real entrypoint points to src/train.py & src/models.py")
    print("============================================================\n")

if __name__ == "__main__":
    run_trace()
