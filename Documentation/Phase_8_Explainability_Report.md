# Phase 8 — Explainability, Grad-CAM, and Final Model Integration Preparation Report
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Objective

Phase 8 implements the **Explainability and Final Model Integration Preparation Engine** for the Cashew Pest and Disease Diagnosis System. It validates visual decision rationale for the Phase 5 Soft-Voting Ensemble sub-models (**`03_VGG16`**, **`05_DenseNet121`**, **`08_ConvNeXtTiny`**), computes ensemble-level visual explainability fusion, analyzes qualitative misclassification errors, documents the 80% confidence uncertainty policy, and prepares the backend API integration contract for the Laravel/PHP web application.

---

## 2. Target Model Architecture & Automatic Layer Selection

Grad-CAM target 4D feature map layers were automatically identified for each of the three ensemble sub-models using `find_target_conv_layer()`:

| Model Index & Name | Selected Target Conv Layer | Feature Map Output Shape | Layer Location Type |
| :--- | :--- | :--- | :--- |
| **`03_VGG16`** | `block5_conv3` | `(1, 14, 14, 512)` | Nested Base Model |
| **`05_DenseNet121`** | `conv5_block16_concat` | `(1, 7, 7, 1024)` | Nested Base Model |
| **`08_ConvNeXtTiny`** | `stage3_block2_conv2` | `(1, 7, 7, 768)` | Nested Base Model |

---

## 3. Grad-CAM & Ensemble Explainability Methodology

### 3.1. Individual Model Grad-CAM Formulation
For a target class $c$ and layer output $A^k$, class activation maps are calculated via:

$$w_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k w_k^c A^k \right)$$

### 3.2. Soft-Voting Ensemble Explainability Fusion
Ensemble explainability fuses normalized 224×224 spatial heatmaps from all three models using Phase 5 validation-selected soft-voting weights ($w_{\text{VGG16}}, w_{\text{DenseNet121}}, w_{\text{ConvNeXtTiny}}$):

$$\text{CAM}_{\text{ensemble}} = w_{\text{VGG16}} \cdot \text{CAM}_{\text{VGG16}} + w_{\text{DenseNet121}} \cdot \text{CAM}_{\text{DenseNet121}} + w_{\text{ConvNeXtTiny}} \cdot \text{CAM}_{\text{ConvNeXtTiny}}$$

$$\text{CAM}_{\text{ensemble\_norm}} = \frac{\text{CAM}_{\text{ensemble}}}{\max(\text{CAM}_{\text{ensemble}})}$$

---

## 4. Phase 7 Individual-Model Class-Wise Extraction Resolution

> [!NOTE]
> **Bug Fix Resolution**:
> Phase 7 previously loaded `test_classification_report.json` which did not exist for Phase 4 individual model checkpoints (which output `classification_report.json`).
> `src/final_selection.py` has been updated to inspect `classification_report.json`, `confusion_matrix.csv`, and `test_predictions.csv` directly from Phase 4 artifacts.
> Per-class test recall values for `Aphids`, `Leaf blight`, `Leaf miner`, and `TMB` are now populated directly from the untouched 861-image Test Set.

---

## 5. Confidence Threshold & Safety Policy

- **Threshold**: `CONFIDENCE_THRESHOLD = 0.80` (80% confidence requirement).
- **Rule**: If max ensemble probability $< 0.80$:
  - `is_uncertain = True`
  - `user_display_message = "Prediction Uncertain. Please upload a clearer image."`
- **Validation**: Prevents low-quality, blurry, out-of-distribution, or non-cashew leaf images from generating arbitrary predictions.

---

## 6. Frontend API Integration Contract (Laravel / PHP Web App)

The backend endpoint replaces the old 15-class `CashewAPI.py` contract with the new 4-class Soft-Voting Ensemble contract:

### `POST /predict` Payload Schema

```json
{
  "status": "success",
  "prediction": "Leaf blight",
  "confidence": 0.9425,
  "is_uncertain": false,
  "user_display_message": "Diagnosed: Leaf blight (Confidence: 94.3%)",
  "models": {
    "VGG16": {
      "prediction": "Leaf blight",
      "confidence": 0.9120,
      "probabilities": [0.01, 0.9120, 0.05, 0.028]
    },
    "DenseNet121": {
      "prediction": "Leaf blight",
      "confidence": 0.9610,
      "probabilities": [0.005, 0.9610, 0.02, 0.014]
    },
    "ConvNeXtTiny": {
      "prediction": "Leaf blight",
      "confidence": 0.9540,
      "probabilities": [0.006, 0.9540, 0.03, 0.01]
    }
  },
  "ensemble_probabilities": [0.007, 0.9425, 0.033, 0.0175],
  "explainability": {
    "gradcam_overlay_url": "/storage/explainability/Leaf_blight_sample_001_overlay.jpg"
  }
}
```

For low-confidence or unreadable images ($< 0.80$):

```json
{
  "status": "uncertain",
  "prediction": null,
  "confidence": 0.6210,
  "is_uncertain": true,
  "user_display_message": "Prediction Uncertain. Please upload a clearer image."
}
```

---

## 7. Output Directory Structure (`Experiments/Explainability/`)

```
Experiments/
└── Explainability/
    ├── GradCAM/
    │   ├── 03_VGG16/                                 # VGG16 heatmaps & overlays
    │   ├── 05_DenseNet121/                           # DenseNet121 heatmaps & overlays
    │   └── 08_ConvNeXtTiny/                          # ConvNeXtTiny heatmaps & overlays
    │
    ├── Ensemble/
    │   ├── ensemble_explanations/                    # Fused ensemble CAM overlays
    │   └── ensemble_prediction_analysis.csv         # Ensemble per-sample prediction analysis
    │
    ├── misclassification_analysis.csv                # Qualitative error analysis log
    ├── explainability_summary.json                   # Structured JSON summary report
    ├── gradcam_layer_selection.json                  # Selected feature map layer registry
    └── Phase_8_Explainability_Report.md             # Full Markdown Report
```

---

## 8. CLI Execution Command

To execute Phase 8 Explainability & Grad-CAM pipeline:

```bash
python explainability.py --all
```
