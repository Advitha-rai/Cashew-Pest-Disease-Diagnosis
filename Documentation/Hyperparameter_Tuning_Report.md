# Phase 11 — Hyperparameter Tuning Recommendation & Results Report
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Safety Declaration

> [!IMPORTANT]
> **PREVIOUS TRAINING vs HYPERPARAMETER TUNING**:
> Previous model training used fixed default hyperparameters and was not a formal hyperparameter-tuning experiment.
> This Phase 11 tuning pipeline conducts a controlled, reproducible hyperparameter search across 10 hyperparameters using **Train (70%) + Validation (15%)** data only.
>
> **TEST SET ISOLATION & CHECKPOINT PROTECTION**:
> - The official **861-image test set** (`Preprocessed/test_split.csv`) remained **100% isolated** from hyperparameter selection.
> - All existing `best_model.keras` checkpoints across `Experiments/01_MobileNetV2/` .. `Experiments/08_ConvNeXtTiny/` remain **100% read-only and untouched**.

---

## 2. Execution Status

- **Tuning Run ID**: `Run_20260820_214600`
- **Execution Status**: `NOT_EXECUTED` (Code and dry-run setup complete; execute search using CLI)
- **Total Trials Configured**: `10` per model
- **Search Method**: `Random Search (Seed=42)`
- **Tuning Max Epochs**: `15`

---

## 3. Class Definitions & Classification Methodology

- **Pest 1**: `Aphids`
- **Pest 2**: `Leaf miner`
- **Pest 3**: `TMB`
- **Disease**: `Leaf blight`

$$\text{Overall 3 Pests Accuracy (\%)} = \frac{\text{Aphids}_{\text{corr}} + \text{Leaf miner}_{\text{corr}} + \text{TMB}_{\text{corr}}}{\text{Aphids}_{\text{tot}} + \text{Leaf miner}_{\text{tot}} + \text{TMB}_{\text{tot}}} \times 100$$

*Strict Rule*: Uses pooled sample counts, **NOT** an unweighted average of individual accuracies. `Leaf blight` (Disease) is strictly excluded from `Overall 3 Pests`.

---

## 4. Deliverables Generated (10-Sheet Excel Workbook)

- **10-Sheet Excel Workbook**: `Experiments/Hyperparameter_Tuning/Run_20260820_214600/Hyperparameter_Tuning_Final.xlsx`
  - Sheet 1: `"Tuning Summary"`
  - Sheet 2: `"All Trials"`
  - Sheet 3: `"Best Configurations"`
  - Sheet 4: `"Search Space"`
  - Sheet 5: `"Dataset Verification"`
  - Sheet 6: `"Evidence"`
  - Sheet 7: `"Classification Comparison"` (6 Columns: `Model | Pest 1 – Aphids | Pest 2 – Leaf miner | Pest 3 – TMB | Overall 3 Pests | Disease – Leaf blight`)
  - Sheet 8: `"Classification Details"` (10 Columns: `Model | Class | Category | Actual Samples | Correct Predictions | Incorrect Predictions | Accuracy | Precision | Recall | F1 Score`)
  - Sheet 9: `"Overall 3 Pests"` (11 Columns: `Model | Aphids Actual | Aphids Correct | Leaf Miner Actual | Leaf Miner Correct | TMB Actual | TMB Correct | Total Pest Images | Total Correct Pest Predictions | Total Incorrect Pest Predictions | Overall 3 Pests Accuracy`)
  - Sheet 10: `"Model Comparison"` (6 Columns: `Model | Aphids Accuracy | Leaf Miner Accuracy | TMB Accuracy | Overall 3 Pests Accuracy | Leaf Blight Accuracy`)
- **Master Trials CSV**: `Experiments/Hyperparameter_Tuning/Run_20260820_214600/hyperparameter_trials.csv`
- **Winning Config JSON**: `Experiments/Hyperparameter_Tuning/Run_20260820_214600/best_hyperparameters.json`
- **Evidence JSON**: `Experiments/Hyperparameter_Tuning/Run_20260820_214600/Hyperparameter_Tuning_Evidence.json`

---

## 5. Recommended Google Colab Execution Command

To execute 10 tuning trials for model #3 (VGG16):

```bash
python hyperparameter_tuning.py --model 3 --trials 10
```

To execute 10 tuning trials for all 8 models:

```bash
python hyperparameter_tuning.py --all --trials 10
```
