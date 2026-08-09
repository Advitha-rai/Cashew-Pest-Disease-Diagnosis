# Phase 7 — Final Model Selection and Deployment Readiness Documentation
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Phase 7 Objective & Scope

Phase 7 performs the **final model selection and deployment readiness analysis** for the Cashew Pest and Disease Diagnosis System. It aggregates all quantitative evaluation results from Phase 4 (Individual Test Evaluation), Phase 5 (Soft-Voting Ensemble), and Phase 6 (Complete Dataset Descriptive Classification) to establish:

1. Which individual vision model achieved top performance on the untouched 861-image Test Set.
2. How the **Phase 5 Soft-Voting Ensemble** (`03_VGG16`, `05_DenseNet121`, `08_ConvNeXtTiny`) compares against the top individual model.
3. Class-specific strengths, weaknesses, and common misclassification confusion pairs.
4. Practical deployment trade-offs (inference latency vs. model size vs. RAM footprint).
5. The finalized model/system recommended for production Flask API deployment.

---

## 2. Dataset Methodology & Strict Methodological Boundaries

- **Total Dataset Size**: 5,734 unique cashew leaf images across 4 target classes (`Aphids`, `Leaf blight`, `Leaf miner`, `TMB`).
- **Reproducible Splits**: 70% Train (4,013 images) / 15% Validation (860 images) / 15% Test (861 images).

> [!IMPORTANT]
> **Methodological Rule**:
> - **Official Unbiased Evaluation**: Derived strictly from the **untouched Phase 4 861-image Test Set** and Phase 5 Test Set evaluation.
> - **Descriptive Supporting Metrics**: Derived from Phase 6 complete dataset classification (5,734 images) and included strictly as descriptive metadata because training samples were previously seen by the models.

---

## 3. Transparent Ranking Methodology & Selection Criteria

Individual models are evaluated and ranked using a transparent weighted composite score:

$$\text{Composite Score} = (0.40 \times \text{Test Macro F1}) + (0.30 \times \text{Test Accuracy}) + (0.20 \times \text{Test MCC}) + (0.10 \times \text{Min Class Accuracy})$$

### Selection Criteria Weights
1. **Test Macro F1-Score (40%)**: Primary metric ensuring balanced performance across all 4 classes without majority-class bias.
2. **Test Accuracy (30%)**: Overall classification accuracy on unseen test samples.
3. **Test MCC (Matthews Correlation Coefficient) (20%)**: Multi-class correlation robustness metric.
4. **Minimum Class Accuracy (10%)**: Worst-performing class safeguard preventing failure on rare disease types.

---

## 4. Models & Systems Compared

| System / Model Name | System Category | Data Source |
| :--- | :--- | :--- |
| `01_MobileNetV2` | Lightweight Mobile Architecture | Phase 4 Checkpoint |
| `02_ResNet50` | Residual Network Backbone | Phase 4 Checkpoint |
| `03_VGG16` | Deep Convolutional Backbone | Phase 4 Checkpoint |
| `04_InceptionV3` | Multi-Scale Factorized Architecture | Phase 4 Checkpoint |
| `05_DenseNet121` | Dense Feature Reuse Network | Phase 4 Checkpoint |
| `06_EfficientNetV2B0` | Neural Architecture Search Backbone | Phase 4 Checkpoint |
| `07_MobileNetV3Large` | Hardware-Aware Mobile Architecture | Phase 4 Checkpoint |
| `08_ConvNeXtTiny` | Modernized Pure-Conv Vision Backbone | Phase 4 Checkpoint |
| `Phase 5 Ensemble` | Probability Soft-Voting Ensemble | Phase 5 Grid-Searched Weights |

---

## 5. Output Artifacts Hierarchy (`Experiments/Final_Model_Selection/`)

```
Experiments/
└── Final_Model_Selection/
    ├── final_model_comparison.csv                # Complete unified benchmark table
    ├── final_model_comparison.xlsx               # Excel version of unified table
    ├── final_model_ranking.csv                   # Ranked individual models with composite scores
    ├── final_model_ranking.json                  # Structured JSON ranking report
    ├── final_model_selection_summary.json        # Executive summary JSON
    ├── class_wise_comparison.csv                 # Per-class accuracy breakdown across all systems
    ├── deployment_readiness_report.json          # Production deployment readiness checklist & decision
    ├── final_model_comparison.png                # Multi-panel visualization dashboard
    ├── test_accuracy_comparison.png              # Test Accuracy bar chart
    ├── test_macro_f1_comparison.png             # Test Macro F1 bar chart
    ├── test_mcc_comparison.png                   # Test MCC bar chart
    ├── inference_latency_comparison.png          # Inference Latency comparison bar chart
    └── Phase_7_Final_Model_Selection.md         # Full Markdown Report
```

---

## 6. CLI Execution Command

To execute Phase 7 analysis and generate all comparison artifacts:

```bash
python final_selection.py
```
