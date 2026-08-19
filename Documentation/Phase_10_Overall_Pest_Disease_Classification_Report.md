# Phase 10 — Overall Pest & Disease Classification Summary Report
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras / OpenPyXL*

---

## 1. Executive Summary & Purpose

Phase 10 provides an executive-level **Overall Pest & Disease Classification Summary** across all 8 individual vision models (**01_MobileNetV2** through **08_ConvNeXtTiny**).
It aggregates the three pest classes (**Aphids**, **Leaf miner**, **TMB**) into a single **"Overall 3 Pests"** metric using pooled sample counts, while keeping the single disease class (**Leaf blight**) separate.

> [!NOTE]
> **Reporting / Aggregation Only**: No model retraining or new model inference was executed during this phase. All per-class values were extracted directly from authoritative Phase 6 complete-dataset classification artifacts.

---

## 2. Category Definitions

- **PESTS (3 Classes)**:
  1. `Aphids`
  2. `Leaf miner`
  3. `TMB`
- **DISEASE (1 Class)**:
  4. `Leaf blight`

---

## 3. Overall 3 Pests Pooled Calculation Formula

The "Overall 3 Pests" column pools the correct predictions and sample totals across all three pest classes:

$$\text{Pest Correct} = \text{Aphids Correct} + \text{Leaf miner Correct} + \text{TMB Correct}$$

$$\text{Pest Total} = \text{Aphids Total} + \text{Leaf miner Total} + \text{TMB Total}$$

$$\text{Pest Accuracy (\%)} = \left( \frac{\text{Pest Correct}}{\text{Pest Total}} \right) \times 100$$

- **Cell Result Format**: Three-line format with Excel line breaks (`wrap_text=True`):
  ```
  Correct: 2550
  Total: 2700
  Accuracy: 94.44%
  ```
- *Strict Rule*: Uses pooled sample counts, **NOT** an unweighted average of individual accuracies.


---

## 4. Models Included (8 Individual Models)

1. `01_MobileNetV2`
2. `02_ResNet50`
3. `03_VGG16`
4. `04_InceptionV3`
5. `05_DenseNet121`
6. `06_EfficientNetV2B0`
7. `07_MobileNetV3Large`
8. `08_ConvNeXtTiny`

*(Phase 5 Soft-Voting Ensemble is excluded per specification as this report is specifically for individual model benchmark comparison).*

---

## 5. Required Column Structure (6 Columns Only)

`Model` | `Aphids` | `Leaf miner` | `TMB` | `Overall 3 Pests` | `Leaf blight`

---

## 6. Excel Formatting Features (`openpyxl`)

- **File Path**: `Experiments/Overall_Pest_Disease_Classification/overall_pest_disease_classification.xlsx`
- **CSV Backup**: `Experiments/Overall_Pest_Disease_Classification/overall_pest_disease_classification.csv`
- **Header Row**: Bold font with soft blue background fill (`#D9E1F2`).
- **Panes**: Frozen top row (`A2`).
- **Grid Lines**: Enabled across sheet.
- **Alignment**: Centered text for result cells, left-aligned for model names.
- **Column Widths**: Auto-adjusted with padding.

---

## 7. Safety & Integrity Verifications

- [x] **Zero Model Retraining**: No weights or checkpoints were modified.
- [x] **Zero New Inference**: Extracted from existing Phase 6 classification artifacts.
- [x] **Artifact Preservation**: Phase 4–9 artifacts remain completely untouched.
- [x] **Pest vs Disease Mapping**: `Aphids`, `Leaf miner`, `TMB` designated as pests; `Leaf blight` designated as disease.

---

## 8. CLI Execution Command

To execute Phase 10 report generation in Google Colab:

```bash
python overall_classification.py --all
```
