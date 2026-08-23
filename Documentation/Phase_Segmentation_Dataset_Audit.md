# Phase A & Phase C — Segmentation Dataset Audit & Annotation Preparation Plan
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Audit Result

> [!IMPORTANT]
> **AUDIT DECISION**: `GROUND_TRUTH_MASKS_NOT_FOUND`
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
- **Dimensions**: Exact match with source RGB image dimensions ($224 \times 224$ or raw resolution).
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
- **Loss Function**: Combined Binary Cross-Entropy + Dice Loss ($L = L_{BCE} + L_{Dice}$).
- **Metrics**: Dice Coefficient, IoU (Jaccard Index), Pixel Accuracy, F1 Score.
- **Output Directory**: `Experiments/Segmentation/best_model.keras` (completely isolated from classification checkpoints).
