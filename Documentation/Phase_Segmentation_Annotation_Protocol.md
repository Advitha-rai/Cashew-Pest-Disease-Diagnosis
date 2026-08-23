# Phase C — Manual Segmentation Annotation Protocol & Quality-Control Guide
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Purpose & Scope

This document establishes the official **Manual Segmentation Annotation Protocol** for generating research-quality pixel-level ground-truth masks for the 5,734 cashew leaf images.

In accordance with strict project safety rules:
- **NO fake, synthetic, or pseudo ground-truth masks are allowed.**
- **NO segmentation model training is permitted until valid ground-truth masks are annotated.**
- **Existing classification checkpoints, split files, and artifacts remain 100% read-only and untouched.**

---

## 2. 5-Class Pixel Mask Encoding Specifications

All masks must be exported as **single-channel 8-bit uint8 PNG images** (`mode='L'`) matching the exact pixel dimensions of their corresponding source image.

| Pixel Value | Class / Region Name | Category | Description |
| :---: | :--- | :--- | :--- |
| **`0`** | **Background** | Healthy Leaf / Background | Healthy cashew leaf tissue, soil, background objects |
| **`1`** | **Aphids** | `Pest` | Aphid clusters, honeydew excretion, curling damage |
| **`2`** | **Leaf miner** | `Pest` | Serpentine mines, silvery trails, larval tunnels |
| **`3`** | **TMB** | `Pest` | Tea Mosquito Bug feeding lesions, necrotic spots |
| **`4`** | **Leaf blight** | `Disease` | Fungal blight lesions, brown necrotic patches |

> [!IMPORTANT]
> - `Leaf blight` (Code 4) is the **ONLY DISEASE CLASS**.
> - `TMB` (Code 3) is a **PEST**. Do NOT assign code 4 to TMB lesions.

---

## 3. Annotation Directory Structure

All annotated mask files must be placed in their isolated split/class folder:

```
Experiments/Segmentation/
    ├── Annotations/
    │   ├── Train/
    │   │   ├── Aphids/
    │   │   ├── Leaf_blight/
    │   │   ├── Leaf_miner/
    │   │   └── TMB/
    │   ├── Validation/
    │   │   ├── Aphids/
    │   │   ├── Leaf_blight/
    │   │   ├── Leaf_miner/
    │   │   └── TMB/
    │   └── Test/
    │       ├── Aphids/
    │       ├── Leaf_blight/
    │       ├── Leaf_miner/
    │       └── TMB/
    ├── segmentation_annotation_manifest.csv
    ├── segmentation_annotation_manifest.json
    ├── segmentation_dataset_audit.json
    └── segmentation_dataset_audit.csv
```

### Deterministic Mask File Naming Rule:
If source image is `Aphids_farm_001.jpg`, expected mask filename is:
```
Experiments/Segmentation/Annotations/Train/Aphids/Aphids_farm_001_mask.png
```

---

## 4. Quality Control & Mask Validation Protocol

Every manually created mask is subjected to 6 automated validation checks via `validate_mask_file()`:

1. **Existence Check**: File must exist at its expected path.
2. **Dimension Matching**: Mask width and height must exactly match source RGB image dimensions.
3. **Format Check**: Must be a single-channel 8-bit image (`uint8`).
4. **Allowed Pixel Values**: Unique pixel values must belong exclusively to `{0, 1, 2, 3, 4}`.
5. **Non-Empty Check**: Mask must contain at least 1 non-zero pixel indicating the target lesion region.
6. **Class Alignment Check**: Non-zero pixels should match the expected target class code.

---

## 5. Google Colab / Local Interactive Manual Annotation Helper

To annotate an image interactively in Google Colab or Python script:

```python
from src.segmentation import save_manual_annotation_mask
import numpy as np

# Load source image dimensions
# Draw or create mask array shape (height, width) with pixel values 0-4
mask_array = np.zeros((224, 224), dtype=np.uint8)

# Set lesion pixels (e.g. Aphids = 1)
mask_array[50:100, 60:110] = 1

# Save & validate mask
success, message = save_manual_annotation_mask(
    image_path="Dataset/Raw/Aphids/Aphids_farm_001.jpg",
    mask_array=mask_array,
    split="Train",
    class_name="Aphids"
)
print(message)
```

---

## 6. Manifest & Validation Commands

```bash
# 1. Inspect dataset audit status:
python segmentation.py --audit

# 2. Build or refresh 5,734-image annotation manifest:
python segmentation.py --manifest

# 3. Validate created masks:
python segmentation.py --validate

# 4. View summary status:
python segmentation.py --summary
```
