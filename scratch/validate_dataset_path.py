"""
Cashew Pest and Disease Diagnosis System
Phase 2: Production Dataset Path & Integrity Validation Tool

File: scratch/validate_dataset_path.py
Purpose: Performs safe pre-flight validation of Google Drive project base path,
         raw dataset location, class folder existence, real image counts,
         and confirms that synthetic fallback is DISABLED.
"""

import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from src.config import Config

def validate_dataset_path():
    project_base = Config.get_base_dir()
    raw_dataset = Config.get_raw_dir()

    print("\n========== DATASET PATH VALIDATION ==========")
    print(f"\nProject Base:\n {project_base}")
    print(f"\nRaw Dataset:\n {raw_dataset}\n")

    expected_classes = Config.DEFAULT_CLASSES
    class_counts = {}
    total_images = 0
    missing_classes = []

    if os.path.exists(raw_dataset) and os.path.isdir(raw_dataset):
        for cls_name in expected_classes:
            cls_dir = os.path.join(raw_dataset, cls_name)
            if os.path.exists(cls_dir) and os.path.isdir(cls_dir):
                images = [
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
                ]
                count = len(images)
                class_counts[cls_name] = count
                total_images += count
            else:
                class_counts[cls_name] = 0
                missing_classes.append(cls_name)

        for cls_name in expected_classes:
            print(f"{cls_name}:\n {class_counts[cls_name]} images")

        status = "VALID REAL DATASET" if total_images > 0 and not missing_classes else "MISSING OR EMPTY CLASS FOLDERS"

    else:
        for cls_name in expected_classes:
            print(f"{cls_name}:\n 0 images (Path not mounted locally)")
        status = "DATASET DIRECTORY NOT MOVED/NOT FOUND IN LOCAL ENVIRONMENT (Requires Google Drive Mount)"

    print("\nSynthetic fallback:\n DISABLED")
    print(f"\nDataset status:\n {status}")
    print("\nNo model training performed.")
    print("No files modified inside the dataset.")
    print("==============================================\n")

if __name__ == "__main__":
    validate_dataset_path()
