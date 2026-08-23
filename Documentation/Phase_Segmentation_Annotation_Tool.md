# Phase C.1 — Interactive Manual Segmentation Annotation Tool Guide
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Purpose & Scope

This document details the **Phase C.1 Interactive Manual Segmentation Annotation Tool** designed specifically for Google Colab and Jupyter Notebook environments.

Per strict project rules:
- **Test set images (861) are strictly EXCLUDED** from the default annotation pool to maintain 100% isolation for final evaluation.
- **NO automatic, synthetic, or pseudo ground-truth masks are generated or used.**
- **NO segmentation model training is executed.**
- **Existing classification models, splits, and checkpoints remain 100% read-only and untouched.**

---

## 2. Eligible Annotation Pool & Test Set Protection

| Dataset Split | Sample Count | Annotation Eligibility Status |
| :--- | :---: | :--- |
| **Train** | `4,013` | **ELIGIBLE** (Primary Annotation Pool) |
| **Validation** | `860` | **ELIGIBLE** (Validation Annotation Pool) |
| **Test** | `861` | **ISOLATED & EXCLUDED (READ-ONLY)** |
| **Total Pool** | `4,873` | **Eligible Manual Annotation Targets** |

---

## 3. 5-Class Pixel Mask Encoding & Canvas Color Legend

All created masks are saved as single-channel 8-bit `uint8` PNG files (`mode='L'`) using nearest-neighbor interpolation to preserve discrete class label indices:

| Pixel Value | Class Name | Category | Canvas Color | Region Annotation Rule |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | **Background** | Healthy Leaf / Background | Transparent / Black | Healthy leaf tissue, background soil, stem |
| **`1`** | **Aphids** | `Pest` | Red (`#FF0000`) | Visible aphid clusters & honeydew damage |
| **`2`** | **Leaf miner** | `Pest` | Green (`#00FF00`) | Visible serpentine mines & tunnels |
| **`3`** | **TMB** | `Pest` | Blue (`#0000FF`) | Visible Tea Mosquito Bug feeding necrosis |
| **`4`** | **Leaf blight** | `Disease` | Yellow (`#FFFF00`) | Visible fungal blight necrotic lesions |

---

## 4. UI Controls & Annotation Rules

### Widget UI Controls:
- **Brush Size Slider**: 1px to 50px line width control.
- **Paint Lesion Button**: Activates active class color brush.
- **Eraser Button**: Erases painted region (resets pixel to background `0`).
- **Undo Button**: Undoes up to 20 previous drawing strokes.
- **Clear Canvas Button**: Resets current image canvas.
- **Skip for Review Button**: Marks image `SKIPPED` in manifest for human review without creating an invented mask.
- **Save Mask & Next Button**: Resizes mask using nearest-neighbor interpolation, validates uint8 0–4 format, updates manifest to `ANNOTATED` / `PASSED`, and loads next pending image.

### Research Annotation Rules:
1. **Annotate Visibly Affected Lesions Only**: Do NOT color the entire leaf simply because the image belongs to a disease/pest class.
2. **Multiple Lesions**: If multiple disconnected lesion spots exist on the same leaf, annotate all visible affected spots.
3. **Healthy Leaf Tissue**: Remains class `0` (Background).
4. **Ambiguous Images**: If the lesion cannot be reliably identified, click **Skip for Review**. Do NOT invent a mask.

---

## 5. Google Colab Launch Cell Commands

To launch the interactive annotation tool in a Google Colab notebook cell:

```python
# 1. Launch annotation tool for next pending Train image:
from src.segmentation_tool import launch_colab_annotation_interface
launch_colab_annotation_interface(split="Train")

# 2. Filter by specific class (e.g. Aphids):
launch_colab_annotation_interface(split="Train", class_name="Aphids")

# 3. View annotation progress report:
from src.segmentation_tool import get_annotation_progress_report
rep = get_annotation_progress_report()
print(rep)
```

---

## 6. Continuous Progress Persistence & Resume Mechanism

- **Automatic Manifest Update**: Every saved or skipped image immediately updates `Experiments/Segmentation/segmentation_annotation_manifest.csv` and `.json`.
- **Colab Restart Recovery**: If your Google Colab runtime disconnects or restarts, re-running `launch_colab_annotation_interface()` automatically resumes at the exact next pending image without losing any completed annotations.
