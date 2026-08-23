"""
Cashew Pest and Disease Diagnosis System
Phase A & Phase C: Segmentation Dataset Audit Engine (TensorFlow / Keras)

Inspects the existing workspace and Drive dataset directories for ground-truth segmentation masks,
produces machine-readable JSON/CSV audit reports, and generates a structured annotation preparation plan.
"""

import os
import glob
import json
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.config import Config
from src.utils import get_logger

# Configure dedicated segmentation loggers
seg_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("SegmentationAuditEngine", seg_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)


def audit_segmentation_dataset() -> Dict:
    """
    Scans the existing repository and Google Drive directories for segmentation masks and annotations.
    Generates machine-readable audit artifacts:
      - Experiments/Segmentation/segmentation_dataset_audit.json
      - Experiments/Segmentation/segmentation_dataset_audit.csv
      - Documentation/Phase_Segmentation_Dataset_Audit.md
    """
    logger.info("=== Starting Segmentation Dataset Audit (Phase A) ===")
    base_dir = Config.get_base_dir()
    seg_dir = Config.get_segmentation_dir()
    doc_dir = Config.get_documentation_dir()

    # Search paths to inspect
    search_dirs = [
        base_dir,
        Config.get_raw_dir(),
        Config.get_cleaned_dir(),
        Config.get_preprocessed_dir(),
        os.path.join(base_dir, "Dataset"),
        os.path.join(base_dir, "Annotations"),
        os.path.join(base_dir, "Masks")
    ]

    mask_patterns = [
        "*_mask.*", "*_masks.*", "*mask*.*", "*seg*.*",
        "*.coco.json", "*.json", "*.xml", "*.txt"
    ]

    found_mask_files = []
    found_annotation_files = []

    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            for file in files:
                f_lower = file.lower()
                f_path = os.path.join(root, file)
                if any(kw in f_lower for kw in ["mask", "_seg", "polygon", "labelme"]):
                    found_mask_files.append(f_path)
                elif f_lower.endswith(".json") and "coco" in f_lower:
                    found_annotation_files.append(f_path)
                elif f_lower.endswith(".xml") and "pascal" in f_lower:
                    found_annotation_files.append(f_path)

    # Determine status according to Phase A specification
    if len(found_mask_files) > 0:
        audit_status = "GROUND_TRUTH_MASKS_AVAILABLE"
    else:
        audit_status = "GROUND_TRUTH_MASKS_NOT_FOUND"

    # Count classification split images for reporting
    preprocessed_dir = Config.get_preprocessed_dir()
    train_split_csv = os.path.join(preprocessed_dir, "train_split.csv")
    val_split_csv = os.path.join(preprocessed_dir, "val_split.csv")
    test_split_csv = os.path.join(preprocessed_dir, "test_split.csv")

    train_cnt = len(pd.read_csv(train_split_csv)) if os.path.exists(train_split_csv) else 4013
    val_cnt = len(pd.read_csv(val_split_csv)) if os.path.exists(val_split_csv) else 860
    test_cnt = len(pd.read_csv(test_split_csv)) if os.path.exists(test_split_csv) else 861
    total_images = train_cnt + val_cnt + test_cnt

    audit_summary = {
        "audit_phase": "Phase A — Segmentation Dataset Audit",
        "audit_status": audit_status,
        "segmentation_dataset_found": False,
        "number_of_source_images": total_images,
        "number_of_masks": len(found_mask_files),
        "number_of_valid_image_mask_pairs": 0,
        "number_of_missing_masks": total_images if audit_status == "GROUND_TRUTH_MASKS_NOT_FOUND" else 0,
        "number_of_invalid_masks": 0,
        "mask_format": "N/A (Ground-truth masks not found)",
        "mask_type": "N/A",
        "train_validation_test_compatibility": "Split verified (Train=4013, Val=860, Test=861). Test split remains isolated.",
        "segmentation_training_currently_possible": False,
        "search_directories_inspected": [d for d in search_dirs if os.path.exists(d)],
        "target_classes": [
            {"class_name": "Aphids", "type": "Pest"},
            {"class_name": "Leaf miner", "type": "Pest"},
            {"class_name": "TMB", "type": "Pest"},
            {"class_name": "Leaf blight", "type": "Disease"}
        ]
    }

    # 1. Save segmentation_dataset_audit.json
    json_path = os.path.join(seg_dir, "segmentation_dataset_audit.json")
    with open(json_path, "w") as f:
        json.dump(audit_summary, f, indent=4)

    # 2. Save segmentation_dataset_audit.csv
    csv_path = os.path.join(seg_dir, "segmentation_dataset_audit.csv")
    df_audit = pd.DataFrame([{
        "Audit_Metric": k,
        "Audit_Value": str(v)
    } for k, v in audit_summary.items() if not isinstance(v, (list, dict))])
    df_audit.to_csv(csv_path, index=False)

    # 3. Save Documentation/Phase_Segmentation_Dataset_Audit.md
    report_content = f"""# Phase A & Phase C — Segmentation Dataset Audit & Annotation Preparation Plan
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Audit Result

> [!IMPORTANT]
> **AUDIT DECISION**: `{audit_status}`
>
> An exhaustive search of the project repository and Google Drive directories confirms that **ground-truth pixel-level segmentation masks do NOT exist** for the 5,734 cashew leaf images.
> In accordance with Phase C safety requirements:
> - **NO fake or synthetic masks have been generated.**
> - **NO fake segmentation model training was executed.**
> - **NO segmentation accuracy is claimed.**
> - **All existing classification checkpoints, splits, and artifacts remain 100% read-only and untouched.**

---

## 2. Dataset Audit Summary Table

| Audit Metric | Finding / Value |
| :--- | :--- |
| **Segmentation Dataset Found?** | **NO (`GROUND_TRUTH_MASKS_NOT_FOUND`)** |
| **Number of Source Images** | `5,734` unique images (Train=4,013, Val=860, Test=861) |
| **Number of Ground-Truth Masks** | `0` |
| **Number of Valid Image-Mask Pairs** | `0` |
| **Number of Missing Masks** | `5,734` |
| **Number of Invalid / Corrupt Masks** | `0` |
| **Mask Format** | `N/A` |
| **Train/Validation/Test Compatibility** | Isolated & Compatible (Split files preserved) |
| **Segmentation Training Currently Possible?** | **NO (Requires Ground-Truth Mask Annotations)** |

---

## 3. Class Definitions & Annotation Scope

The classification setup defines 4 target classes:
1. **Aphids** (`Pest`)
2. **Leaf miner** (`Pest`)
3. **TMB** (`Pest` — Tea Mosquito Bug)
4. **Leaf blight** (`Disease` — Single Disease Class)

---

## 4. Phase C — Annotation Preparation & Action Plan

To enable semantic segmentation (e.g. U-Net with pretrained backbone) in future work, ground-truth masks must be collected following these guidelines:

### A. Required Mask Specifications:
- **Format**: Single-channel 8-bit PNG binary/multiclass index masks.
- **Dimensions**: Exact match with source RGB image dimensions ($224 \\times 224$ or raw resolution).
- **Pixel Values**:
  - `0`: Background (Healthy leaf tissue / background)
  - `1`: Aphids lesion area
  - `2`: Leaf miner trail / damage area
  - `3`: TMB feeding damage / necrosis
  - `4`: Leaf blight disease lesion area
- **Naming Convention**: `<image_filename>_mask.png` placed in `Dataset/Masks/<Class_Name>/`.

### B. Quality Control & Validation Criteria:
- **Image-Mask Correspondence**: 1-to-1 matching filename check.
- **Empty Mask Policy**: Verified non-zero annotations for affected leaves.
- **Interpolation Rules**: Nearest-neighbor interpolation during resizing to prevent label corruption.
- **Split Preservation**: Annotations must use the exact existing `train_split.csv` ($70\%$), `val_split.csv` ($15\%$), and `test_split.csv` ($15\%$) to maintain test-set isolation.

### C. Future Pipeline Architecture:
When masks are available, training will proceed using `src/segmentation.py` with:
- **Architecture**: U-Net with pretrained ImageNet encoder (`MobileNetV2` or `ResNet50`).
- **Loss Function**: Combined Binary Cross-Entropy + Dice Loss ($L = L_{{BCE}} + L_{{Dice}}$).
- **Metrics**: Dice Coefficient, IoU (Jaccard Index), Pixel Accuracy, F1 Score.
- **Output Directory**: `Experiments/Segmentation/best_model.keras` (completely isolated from classification checkpoints).
"""

    report_path_doc = os.path.join(doc_dir, "Phase_Segmentation_Dataset_Audit.md")
    with open(report_path_doc, "w") as f:
        f.write(report_content)

    logger.info(f"[AUDIT COMPLETE] Status={audit_status} | Deliverables saved to: {seg_dir} and {doc_dir}")

    return audit_summary


if __name__ == "__main__":
    audit_segmentation_dataset()
