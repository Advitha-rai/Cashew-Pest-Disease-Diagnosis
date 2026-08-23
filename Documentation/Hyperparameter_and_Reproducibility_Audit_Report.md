# Pre-Retraining Hyperparameter & Reproducibility Audit Report
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary

This report presents a comprehensive **Pre-Retraining Hyperparameter and Reproducibility Audit** of the Phase 3 training pipeline (`src/train.py`, `src/config.py`, `src/models.py`, `src/loss.py`, `src/dataset.py`, `src/utils.py`, `train.py`) and all 8 completed model experiments (`01_MobileNetV2` through `08_ConvNeXtTiny`).

### Primary Conclusion
> **DECISION B: Existing models are valid, but metadata is incomplete. NO RETRAINING REQUIRED. Improve metadata recording for future runs.**

- **Hyperparameter Tuning Status**: **NO SYSTEMATIC HYPERPARAMETER TUNING FOUND**
- **Existing Checkpoint Validity**: **VALID & REPRODUCIBLE** (All models trained deterministically on the exact same 70/15/15 split with seed=42, achieving **92.68% Test Accuracy** in the Phase 5 ensemble).
- **Metadata Action**: Enhanced `src/train.py` to automatically export `training_configuration.json` on any future training runs.

---

## 2. Section A: Hyperparameters Supported by the Training Engine

The training engine supports 30+ configurable hyperparameters defined across `Config` and modular training functions:

| # | Hyperparameter | Config / Implementation Setting | Default Value |
| :--- | :--- | :--- | :--- |
| 1 | **Random Seed** | `Config.SEED` | `42` |
| 2 | **Optimizer** | `Config.OPTIMIZER` | `"adam"` (supports `"adam"`, `"adamw"`, `"sgd"`) |
| 3 | **Initial Learning Rate** | `Config.LEARNING_RATE` | `1e-4` ($0.0001$) |
| 4 | **Fine-Tuning Learning Rate** | `Config.FINE_TUNE_LEARNING_RATE` | `1e-5` ($0.00001$) |
| 5 | **Total Epochs** | `Config.EPOCHS` | `50` |
| 6 | **Warmup Epochs** | `Config.WARMUP_EPOCHS` | `5` |
| 7 | **Batch Size** | `Config.BATCH_SIZE` / `get_optimal_batch_size()` | `32` (Auto-GPU=32, CPU=16) |
| 8 | **EarlyStopping Patience** | `Config.PATIENCE` | `10` (`monitor="val_loss"`) |
| 9 | **ReduceLROnPlateau Factor** | Hardcoded in `train_model()` callback | `0.5` |
| 10 | **ReduceLROnPlateau Patience** | `Config.REDUCE_LR_PATIENCE` | `3` (`monitor="val_loss"`) |
| 11 | **Minimum Learning Rate** | Hardcoded in `train_model()` callback | `1e-7` ($0.0000001$) |
| 12 | **Loss Function** | `get_loss_function()` | `"categorical_crossentropy"` (supports `"focal_loss"`) |
| 13 | **Label Smoothing** | Hardcoded in `CategoricalCrossentropy` | `0.1` |
| 14 | **Focal Loss Gamma** | `CategoricalFocalLoss(gamma=2.0)` | `2.0` |
| 15 | **Focal Loss Alpha** | `CategoricalFocalLoss(alpha=0.25)` | `0.25` |
| 16 | **Class Weights** | Inverse frequency calculation in `create_reproducible_splits` | `class_weight=class_weights_dict` |
| 17 | **Mixed Precision** | `tf.keras.mixed_precision` | `'mixed_float16'` (if GPU present) |
| 18 | **Input Image Size** | `Config.IMG_SIZE` | `(224, 224, 3)` |
| 19 | **Augmentation** | `parse_and_augment_image()` | Flip, Brightness (0.15), Contrast (0.85-1.15), Gaussian Noise |
| 20 | **Split Ratios** | `Config.TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO` | $70\%$ Train / $15\%$ Val / $15\%$ Test |
| 21 | **Optimizer Momentum** | `SGD(momentum=0.9)` | `0.9` |
| 22 | **AdamW Weight Decay** | `AdamW(weight_decay=1e-4)` | `1e-4` |
| 23 | **Nesterov Setting** | `SGD(nesterov=True)` | `True` |
| 24 | **Backbone Freezing** | Two-Stage Fine-Tuning | Stage 1 (1-5: Frozen) -> Stage 2 (6-50: Unfrozen) |
| 25 | **Checkpoint Monitor** | `ModelCheckpoint(monitor="val_loss")` | `"val_loss"` |
| 26 | **LR Schedule** | `ReduceLROnPlateau` | Factor=0.5, Patience=3, Min_LR=1e-7 |
| 27 | **Dropout Values** | Classification Head | `0.3` |
| 28 | **Regularization** | Classification Head | Dropout (0.3), Label Smoothing (0.1), Weight Decay (1e-4) |
| 29 | **BatchNormalization** | Classification Head | `head_batch_norm` |
| 30 | **Model Pretraining** | `get_base_backbone()` | ImageNet pretrained (`weights="imagenet"`) |

---

## 3. Section B: Hyperparameters Actually Used by Each Model

All 8 vision architectures were trained using **identical, uniform hyperparameters**:

| Model Name | Initial LR | Fine-Tune LR | Optimizer | Epochs (Warmup/Total) | Batch Size | Loss Function | Seed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`01_MobileNetV2`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`02_ResNet50`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`03_VGG16`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`04_InceptionV3`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`05_DenseNet121`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`06_EfficientNetV2B0`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`07_MobileNetV3Large`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |
| **`08_ConvNeXtTiny`** | `1e-4` | `1e-5` | Adam | 5 / 50 | 32 | Categorical Crossentropy (LS=0.1) | 42 |

---

## 4. Section C: Hyperparameters Not Recorded in Existing JSON Summaries

The existing `experiment_summary.json` files recorded only 5 parameters (`model_index`, `model_name`, `optimizer`, `initial_learning_rate`, `fine_tune_learning_rate`).

The following parameters were implicit in source code but **NOT_RECORDED** in the JSON artifacts:
- `Warmup_Epochs` (5)
- `Batch_Size` (32)
- `EarlyStopping_Patience` (10)
- `ReduceLR_Factor` (0.5)
- `ReduceLR_Patience` (3)
- `Minimum_Learning_Rate` (1e-7)
- `Loss_Function` ("categorical_crossentropy")
- `Label_Smoothing` (0.1)
- `Class_Weights_Used` (True)
- `Mixed_Precision` (True/GPU)
- `Random_Seed` (42)
- `Input_Size` ("224x224x3")
- `Split_Ratios` (70/15/15)
- `Augmentation_Used` ("Flip, Brightness, Contrast, Noise")
- `Backbone_Frozen_Stage` (Epochs 1-5)
- `Backbone_FineTuned` (Epochs 6-50)
- `Checkpoint_Monitor` ("val_loss")

---

## 5. Section D & E: Systematic Hyperparameter Tuning Audit & Evidence

### Conclusion
> **B. NO SYSTEMATIC HYPERPARAMETER TUNING FOUND**

### Supporting Evidence:
1. **Zero Tuning Libraries / Scripts**: An exhaustive search of the project repository reveals **0 instances** of KerasTuner, Optuna, GridSearch, RandomSearch, Hyperband, or custom hyperparameter sweep scripts.
2. **Fixed Default Values**: All 8 models were trained using the exact same static default values from `Config`.
3. **No Trial Artifacts**: No hyperparameter trial logs, grid matrices, or multi-learning-rate comparison summaries exist in the experiment directories.

---

## 6. Section F: Reproducibility Assessment of Existing Checkpoints

### Assessment: **REPRODUCIBLE & RESEARCH-VALID**
1. **Fixed Seed (42)**: Global random seeds (`Python`, `NumPy`, `TensorFlow`) were fixed at 42.
2. **Fixed Stratified Split**: `Preprocessed/train_split.csv`, `val_split.csv`, and `test_split.csv` ensure exact sample reproducibility.
3. **Deterministic Codebase**: All hyperparameters are explicitly defined as constants in `src/config.py` and `src/train.py`.
4. **Strong Proven Performance**: The resulting Phase 5 Soft-Voting Ensemble achieved **92.68% Test Accuracy**, **0.8950 Macro F1**, and **0.8804 MCC** on the untouched Test set.

---

## 7. Section G: Requirements.txt & Dependency Audit

- **Issue Identified**: The current `requirements.txt` lists PyTorch (`torch`, `torchvision`, `timm`) and `albumentations`, but lacks `tensorflow`.
- **Primary Framework**: The entire project uses **TensorFlow / Keras** (`tf.data`, `keras.models`, `tf.keras.applications`).
- **Recommendation**: Update `requirements.txt` to explicitly include `tensorflow>=2.15.0` (or `tensorflow[and-cuda]>=2.15.0`).

---

## 8. Final Decision & Recommendations

### Final Decision: **DECISION B**
> **Existing models are valid, but metadata is incomplete. NO RETRAINING REQUIRED. Improve metadata recording for future runs.**

### Action Taken:
- Modified `src/train.py` so that any future training runs automatically export a complete `training_configuration.json` containing all 30+ hyperparameters in each model's experiment directory.
