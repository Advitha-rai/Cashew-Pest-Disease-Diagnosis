# Phase 5 — Production-Quality Soft-Voting Ensemble Documentation
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Model Selection Rationale

Phase 5 implements a **production-quality, soft-voting ensemble prediction engine** combining the top three vision architectures evaluated during Phase 4:

1. **`03_VGG16`**: Outstanding single-model baseline (Test Accuracy: **90.82%**, Macro F1: **86.33%**, MCC: **84.96%**).
2. **`08_ConvNeXtTiny`**: Modern vision backbone capturing fine-grained spatial representations (Test Accuracy: **88.15%**, Macro F1: **84.36%**, MCC: **80.80%**).
3. **`05_DenseNet121`**: Dense feature reuse architecture providing strong feature diversity (Test Accuracy: **86.06%**, Macro F1: **82.21%**, MCC: **77.75%**).

### Why Soft Voting Over Hard Voting?
- **Hard Voting (Majority Vote)**: Discards model confidence scores and assigns equal weight to discrete predictions, leading to ties and loss of probabilistic calibration.
- **Probability Soft Voting**: Averages continuous probability vectors output by each model:
  $$P_{\text{ensemble}} = \sum_{i=1}^3 w_i \cdot P_i \quad \text{where } w_i \ge 0 \text{ and } \sum w_i = 1.0$$
  Soft voting preserves uncertainty, enhances calibration, and leverages high-confidence predictions from superior models.

---

## 2. Validation-Based Weight Optimization Methodology

To prevent data leakage, **ensemble weights are searched strictly on the 15% validation split (`val_split.csv`)**. The test set (`test_split.csv`) remains completely unseen until final evaluation.

### Search Procedure
1. **Baseline Equal Weights**: $w_{\text{VGG16}} = 1/3, w_{\text{DenseNet121}} = 1/3, w_{\text{ConvNeXtTiny}} = 1/3$.
2. **Constrained Grid Search**:
   - Grid over $(w_1, w_2, w_3)$ with step $0.05$, $w_i \ge 0$, $\sum w_i = 1.0$.
   - Primary Selection Criterion: **Validation Macro F1-Score**.
   - Secondary Selection Criterion: **Validation Accuracy**.
3. **Export**: Selected weights are saved to `Experiments/Ensemble/ensemble_weights.json`.

---

## 3. Confidence Thresholding & Invalid Image Protection

### 3.1. Uncertainty Policy (80% Confidence Requirement)
- **Rule**: `CONFIDENCE_THRESHOLD = 0.80`.
- If max ensemble probability $< 0.80$:
  - `is_uncertain = True`
  - `user_display_message = "Prediction Uncertain. Please upload a clearer image"`
- Prevents random guessing on ambiguous, low-quality, or blurry farm images.

### 3.2. Invalid Image & Input Safety Validation
Single-image inference inputs are checked prior to model execution:
1. **File Checks**: Existence, non-zero file size, valid extension (`.jpg`, `.png`, `.webp`, `.tiff`).
2. **Decoding & Format Checks**: Multi-channel RGB decoding integrity, resolution $\ge 10 \times 10$.
3. **Pixel Uniformity Check**: Rejects blank, zero-variance, or corrupted constant pixel data.
4. **Controlled Error Response**:
   If validation fails, returns:
   > `"Invalid image. Please upload a valid cashew leaf image."`

---

## 4. CLI Execution & Reproducibility Commands

```bash
# 1. Run validation inference and optimal weight selection:
python ensemble.py --validate

# 2. Run final ensemble evaluation on untouched test set:
python ensemble.py --test

# 3. Run complete Phase 5 pipeline (Validation + Weight Selection + Test Evaluation + Comparison):
python ensemble.py --all

# 4. Perform single image ensemble prediction:
python ensemble.py --predict path/to/sample_leaf.jpg
```

---

## 5. Output Directory Structure (`Experiments/Ensemble/`)

```
Experiments/
└── Ensemble/
    ├── ensemble_weights.json                 # Selected model weights & validation metrics
    ├── validation_results.csv                # Grid search weight optimization records
    ├── validation_results.json               # Structured validation summary
    ├── test_predictions.csv                  # Individual test predictions with confidence & uncertainty status
    ├── test_classification_report.csv        # Tabular classification report
    ├── test_classification_report.json       # JSON classification report
    ├── test_classification_report.txt        # Text classification report
    ├── test_confusion_matrix.png             # Raw confusion matrix heatmap
    ├── test_confusion_matrix_normalized.png  # Normalized confusion matrix heatmap
    ├── test_confusion_matrix.csv             # Confusion matrix counts
    ├── test_roc_curve.png                    # Multi-Class OvR ROC curves
    ├── test_roc_auc_scores.csv               # Per-class ROC-AUC scores
    ├── test_precision_recall_curve.png       # Precision-Recall curves
    ├── test_confidence_distribution.png      # Confidence histogram with 80% threshold line
    ├── ensemble_per_class_results.csv        # Per-class correct/total counts and percentages
    ├── ensemble_performance.json             # Inference latency (ms) and FPS throughput
    ├── ensemble_evaluation_summary.json      # Full evaluation summary JSON
    ├── model_comparison.csv                  # Comparative benchmark table against individual models
    ├── model_comparison.xlsx                 # Excel comparative benchmark table
    └── misclassified_images/                 # Saved thumbnails of misclassified test images
```

---

## 6. Limitations & Technical Notes

1. **Softmax Calibration Limitation**:
   High softmax confidence does not guarantee out-of-distribution (OOD) safety for non-cashew images. The system mitigates this risk through input validation checks, pixel variance profiling, and the 80% uncertainty threshold.
2. **Model Weight Preservation**:
   Sub-model checkpoints (`best_model.keras` in `03_VGG16`, `05_DenseNet121`, and `08_ConvNeXtTiny`) are read-only and were **not modified or retrained**.
