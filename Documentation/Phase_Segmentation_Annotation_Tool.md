# Phase C.1 — Interactive Manual Segmentation Annotation Tool Guide (Repaired Engine)
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Repair Declaration

This document details the **repaired Phase C.1 Interactive Manual Segmentation Annotation Tool** for Google Colab and Jupyter Notebook environments.

> [!IMPORTANT]
> **REPAIR FIX SUMMARY**:
> The previous HTML5 canvas interface was lacking bi-directional JavaScript-to-Python execution callbacks.
> The engine has been upgraded with **Google Colab Kernel Callbacks (`google.colab.output.register_callback`)**:
> - Clicking **💾 Save Mask & Next** extracts off-screen single-channel class-ID mask data, transmits it to Python via `notebook.save_mask`, applies **nearest-neighbor spatial interpolation**, saves the uint8 0–4 PNG mask, executes `validate_mask_file()`, updates `annotation_status` to `ANNOTATED` and `validation_status` to `PASSED`, persists progress continuously to `segmentation_annotation_manifest.csv/.json`, and automatically loads the next pending image.
> - Clicking **⏭️ Skip for Review** calls `notebook.skip_image`, marks `annotation_status` as `SKIPPED`, updates manifest, and loads the next pending image.

---

## 2. Test Set Protection & Eligible Annotation Pool

| Dataset Split | Sample Count | Annotation Eligibility Status |
| :--- | :---: | :--- |
| **Train** | `4,013` | **ELIGIBLE** (Primary Annotation Pool) |
| **Validation** | `860` | **ELIGIBLE** (Validation Annotation Pool) |
| **Test** | `861` | **ISOLATED & EXCLUDED (READ-ONLY)** |
| **Total Pool** | `4,873` | **Eligible Manual Annotation Targets** |

> [!CAUTION]
> **TEST SET ISOLATION**:
> All 861 Test split images (`Preprocessed/test_split.csv`) are strictly **blocked and excluded** from the annotation tool. Attempting to select `Test` split raises an explicit isolation protection exception.

---

## 3. 5-Class Pixel Mask Encoding & Canvas Legend

Masks are saved as single-channel 8-bit `uint8` PNG files (`mode='L'`) containing discrete class indices `{0, 1, 2, 3, 4}`:

| Pixel Value | Class Name | Category | Canvas Overlay Color | Annotation Rule |
| :---: | :--- | :--- | :--- | :--- |
| **`0`** | **Background** | Healthy Leaf / Background | Transparent / Black | Healthy leaf tissue, background soil, stem |
| **`1`** | **Aphids** | `Pest` | Red (`#FF0000`) | Visible aphid clusters & honeydew damage |
| **`2`** | **Leaf miner** | `Pest` | Green (`#00FF00`) | Visible serpentine mines & tunnels |
| **`3`** | **TMB** | `Pest` | Blue (`#0000FF`) | Visible Tea Mosquito Bug feeding necrosis |
| **`4`** | **Leaf blight** | `Disease` | Yellow (`#FFFF00`) | Visible fungal blight necrotic lesions |

---

## 4. Google Colab Launch Commands & Callbacks

```python
# 1. Launch interactive annotation interface for Train pool:
from src.segmentation_tool import launch_colab_annotation_interface

launch_colab_annotation_interface(split="Train")

# 2. Filter by specific target class (e.g. Aphids):
launch_colab_annotation_interface(split="Train", class_name="Aphids")

# 3. View annotation progress report:
from src.segmentation_tool import get_annotation_progress_report
rep = get_annotation_progress_report()
print(rep)
```

---

## 5. Phase C.1 Fix Verification Report

```
PHASE C.1 FIX VERIFICATION
--------------------------
Test-set isolation      : PASS
861 test images excluded: PASS
Mask creation           : PASS
Mask saving             : PASS
Annotation status update: PASS
Validation update       : PASS
Skip functionality      : PASS
Next-image functionality: PASS
Progress update         : PASS
```
