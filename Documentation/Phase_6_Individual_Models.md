# Phase 6 — Individual Model Complete Dataset Classification Documentation
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Purpose

Phase 6 performs descriptive classification on the **complete 5,734-image dataset (`Train` + `Validation` + `Test`)** for **each of the 8 trained vision architectures**:

1. `01_MobileNetV2`
2. `02_ResNet50`
3. `03_VGG16`
4. `04_InceptionV3`
5. `05_DenseNet121`
6. `06_EfficientNetV2B0`
7. `07_MobileNetV3Large`
8. `08_ConvNeXtTiny`

### Methodological Disclaimer (Descriptive vs. Unbiased Evaluation)
> [!IMPORTANT]
> **Descriptive Classification Notice**:
> Complete dataset classification evaluates how each model classifies all available dataset images (4,013 Train + 860 Validation + 861 Test = 5,734 total unique images).
> - **DO NOT** confuse complete dataset accuracy with unbiased test accuracy.
> - Because training set samples were previously seen by the models during Phase 3, complete dataset accuracy measures overall descriptive performance.
> - The official, unbiased evaluation of model generalization remains the **untouched 861-image Phase 4 Test Set evaluation**.

---

## 2. Technical Pipeline & Safety Architecture

1. **Dataset Concatenation & Deduplication**:
   - Concatenates `Preprocessed/train_split.csv`, `Preprocessed/val_split.csv`, and `Preprocessed/test_split.csv`.
   - Deduplicates file paths while retaining origin split tags (`Train`, `Validation`, `Test`).
2. **Fast Parallel PIL Image Validation**:
   - Uses `validate_image_file_fast()` with `ThreadPoolExecutor(max_workers=16)` to check file decodability, byte size, resolution, and RGB conversion in parallel.
   - Invalid files are logged to `invalid_images.log` and excluded from prediction metrics.
3. **Vectorised Batched Inference**:
   - Recreates a fresh `tf.data.Dataset` (`224×224` RGB, `batch_size=32`, `prefetch(AUTOTUNE)`) for each model.
   - Executes vectorised batched prediction via `model.predict(ds, verbose=1)`.
4. **Confidence Thresholding (80%)**:
   - Preserves `CONFIDENCE_THRESHOLD = 0.80`. If top softmax prediction confidence $< 0.80$, sets `is_uncertain = True` and outputs `"Prediction Uncertain. Please upload a clearer image"`.

---

## 3. CLI Execution & Reproducibility Commands

```bash
# 1. Run complete dataset classification across ALL 8 vision models:
python individual_models.py --full-dataset

# 2. Run complete dataset classification for a single model by index (1 to 8):
python individual_models.py --model 1

# 3. Run complete dataset classification across ALL 8 models (alias for --full-dataset):
python individual_models.py --all
```

---

## 4. Output Directory Structure

```
Experiments/
└── Individual_Models/
    └── Full_Dataset_Classification/
        ├── invalid_images.log                            # Audit log of invalid/corrupted files
        ├── 8_model_complete_dataset_comparison.csv       # Comparative table across all 8 models
        ├── 8_model_complete_dataset_comparison.xlsx      # Excel comparative table
        ├── 8_model_accuracy_comparison.png               # High-resolution comparison bar chart
        ├── 01_MobileNetV2/
        ├── 02_ResNet50/
        ├── 03_VGG16/
        ├── 04_InceptionV3/
        ├── 05_DenseNet121/
        ├── 06_EfficientNetV2B0/
        ├── 07_MobileNetV3Large/
        └── 08_ConvNeXtTiny/
            ├── full_dataset_predictions.csv              # Per-sample predictions across 5,734 images
            ├── full_dataset_per_class_results.csv        # Per-class correct/total counts and percentages
            ├── full_dataset_split_results.csv            # Accuracy for Train, Val, Test, and Complete Dataset
            ├── full_dataset_summary.json                 # Structured JSON summary report
            ├── full_dataset_classification_report.csv    # Tabular classification report
            ├── full_dataset_classification_report.json   # JSON classification report
            ├── full_dataset_confusion_matrix.png         # Raw confusion matrix heatmap
            ├── full_dataset_confusion_matrix_normalized.png # Normalized confusion matrix heatmap
            ├── full_dataset_confusion_matrix.csv         # Raw confusion matrix counts
            └── full_dataset_misclassified_images/        # Thumbnails of misclassified images
```

---

## 5. Model Checkpoint & Baseline Preservation

- Sub-model checkpoints (`best_model.keras` in `01` through `08`) remain read-only and were **not modified or retrained**.
- Phase 4 test evaluation metrics and Phase 5 ensemble artifacts remain **completely untouched**.
