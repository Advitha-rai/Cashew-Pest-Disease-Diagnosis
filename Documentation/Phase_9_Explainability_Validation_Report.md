# Phase 9 — Grad-CAM Localization Validation & Model Trustworthiness Audit Report
**Cashew Pest and Disease Diagnosis System**
*Framework: TensorFlow / Keras*

---

## 1. Executive Summary & Audit Purpose

Phase 9 performs an **objective Grad-CAM Localization Validation and Model Trustworthiness Audit** for the Cashew Pest & Disease Diagnosis System. While Phase 5 established strong test prediction performance (**92.68% Test Accuracy**, **0.8950 Macro F1**, **0.8804 MCC**) and Phase 8 successfully generated Grad-CAM heatmaps, Phase 9 evaluates whether model visual attention is concentrated on genuine biological pest/disease symptom regions or influenced by background features, leaf margins, or image borders.

---

## 2. Dataset Methodology & Weakly Supervised Disclaimer

- **Test Set**: Untouched 861-image Test set (`Preprocessed/test_split.csv`).
- **Target Classes**: `['Aphids', 'Leaf blight', 'Leaf miner', 'TMB']`.
- **Selected Production System**: Phase 5 Soft-Voting Ensemble (`03_VGG16`, `05_DenseNet121`, `08_ConvNeXtTiny`).

> [!IMPORTANT]
> **Methodological Disclaimer**:
> Ground-truth lesion/pest segmentation masks are currently unavailable in the dataset. Therefore, automated spatial metrics (centroids, border percentages, mask IoUs) are **weakly-supervised spatial diagnostics** and must NOT be presented as quantitative biological ground truth.

---

## 3. Spatial Diagnostic Engine & Multi-Threshold Analysis

Each normalized Grad-CAM heatmap ($224 \times 224$) is analyzed across four binarization thresholds ($T \in \{0.20, 0.40, 0.60, 0.80\}$):

1. **Active Pixel Area Ratio (%)**: Percentage of image area with attention $\ge T$.
2. **Attention Centroid & Center Distance**: Spatial center-of-mass $(x_c, y_c)$ and distance from image center $(112, 112)$.
3. **Border vs. Central Concentration**:
   - Outer Border Margin: Outer 15% margin of image boundaries.
   - Central Region: Inner 70% region.
4. **Primary Spatial Diagnostic Flags**:
   - `BORDER_CONCENTRATED`: Border attention $> 35\%$.
   - `EXCESSIVELY_DIFFUSE`: $40\%$ threshold mask area ratio $> 50\%$.
   - `ISOLATED_HOTSPOT`: $60\%$ threshold mask area ratio $< 2\%$.
   - `PLANT_CENTERED`: Centroid distance $< 35$ pixels and border attention $< 15\%$.

---

## 4. Class-Specific Biological Expectations

| Target Class | Qualitative Biological Expectation |
| :--- | :--- |
| **`Aphids`** | Attention should overlap visible aphid clusters or feeding-damage regions. |
| **`Leaf blight`** | Attention should overlap diseased, blighted, or discolored foliar lesion areas. |
| **`Leaf miner`** | Attention should overlap serpentine miner trails, feeding tracks, or damaged tissue. |
| **`TMB`** | Attention should overlap Tea Mosquito Bug lesions, necrotic feeding spots, or pests. |

---

## 5. Pairwise Inter-Model Attention Agreement

Spatial agreement between `03_VGG16`, `05_DenseNet121`, `08_ConvNeXtTiny`, and the `Ensemble` is computed using:
- **Pearson Correlation ($r$)**
- **Cosine Similarity ($\cos \theta$)**
- **Mask IoU at 40% Threshold**
- **Centroid Distance (pixels)**

---

## 6. High-Confidence Wrong Prediction Risk Audit

Incorrect predictions are categorized by risk level:
- **CRITICAL**: Ensemble Confidence $\ge 90\%$, prediction incorrect, border attention $> 35\%$ or model divergence.
- **HIGH**: Ensemble Confidence $\ge 80\%$, prediction incorrect, background attention.
- **MEDIUM**: Ensemble Confidence $< 80\%$, prediction incorrect.
- **LOW**: Prediction correct.

---

## 7. Production Readiness Decision

> [!TIP]
> **Production Decision: OPTION B**
> - **Prediction Readiness**: **READY FOR DEPLOYMENT** (92.68% Test Accuracy, 0.8950 Macro F1).
> - **Explainability Readiness**: Strong prediction performance, but visual explainability requires ongoing qualitative monitoring until pixel-level segmentation masks are available for formal validation.

---

## 8. Output Artifacts Hierarchy (`Experiments/Explainability_Validation/`)

```
Experiments/
└── Explainability_Validation/
    ├── localization_validation_summary.csv       # Per-sample summary audit log
    ├── sample_localization_audit.csv             # Full spatial metrics breakdown
    ├── high_confidence_wrong_predictions.csv     # Ranked risk audit table
    ├── attention_statistics.csv                  # Aggregated spatial stats per model
    ├── model_attention_similarity.csv           # Pairwise inter-model similarity matrix
    ├── ensemble_attention_validation.csv         # Ensemble heatmap validation metrics
    ├── target_layer_validation.csv               # Target conv layer audit report
    ├── explainability_trust_scores.csv           # Diagnostic trust scores
    ├── production_explainability_readiness.json  # Final decision JSON
    ├── Phase_9_Explainability_Validation_Report.md # Full Markdown Report
    ├── Visualizations/
    │   ├── localization_validation_grid/         # 6-panel composite grids
    │   ├── high_confidence_errors/               # High-risk misclassification grids
    │   ├── model_attention_comparison/           # Model comparison heatmaps
    │   ├── ensemble_attention_comparison/        # Ensemble heatmaps
    │   └── suspicious_localizations/             # Flagged border/background heatmaps
    └── Metadata/
        └── validation_configuration.json         # Pipeline configuration log
```

---

## 9. CLI Execution Command

To execute Phase 9 Explainability Validation & Audit in Google Colab:

```bash
python explainability_validation.py --all
```
