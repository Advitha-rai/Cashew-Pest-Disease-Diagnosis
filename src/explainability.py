"""
Cashew Pest and Disease Diagnosis System
Phase 8: Explainability, Grad-CAM, and Model Integration Preparation Engine (TensorFlow / Keras)

Provides reliable Grad-CAM / Grad-CAM++ explanations for the Phase 5 ensemble models (03_VGG16,
05_DenseNet121, 08_ConvNeXtTiny), ensemble-level explainability fusion, qualitative misclassification
analysis, low-confidence uncertainty policy verification, and API contract preparation.
"""

import os
import sys
import time
import json
import shutil
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
from tensorflow import keras

from src.config import Config
from src.utils import set_seed, get_logger
from src.evaluate import parse_inference_image, build_inference_dataset
from src.ensemble import load_ensemble_models, validate_image_file_fast

# Configure dedicated Phase 8 loggers
phase8_log_path = os.path.join(Config.get_logs_dir(), "evaluation.log")
exception_log_path = os.path.join(Config.get_logs_dir(), "exceptions.log")

logger = get_logger("Phase8Explainability", phase8_log_path)
exc_logger = get_logger("ExceptionEngine", exception_log_path)


# ---------------------------------------------------------
# 1. Automatic Layer Selection Engine
# ---------------------------------------------------------
PREFERRED_CONV_LAYERS = {
    "03_VGG16": "block5_conv3",
    "05_DenseNet121": "conv5_block16_concat",
    "08_ConvNeXtTiny": "stage3_block2_conv2"
}

def get_target_layer_and_container(model: keras.Model, layer_name: Optional[str] = None) -> Tuple[Optional[keras.layers.Layer], Optional[keras.Model], str]:
    """
    Locates the target layer and its containing model container (nested base_model vs outer model).
    Returns (target_layer_object, container_model, location_type_str).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base_model = layer
            break

    # 1. Search nested base_model if present
    if base_model is not None:
        if layer_name:
            for l in base_model.layers:
                if l.name == layer_name:
                    return l, base_model, "base_model"
        
        # Fallback search inside base_model for rank-4 conv layer
        for l in reversed(base_model.layers):
            try:
                out_s = l.output_shape
                if isinstance(out_s, list):
                    out_s = out_s[0]
                if len(out_s) == 4 and out_s[1] is not None and out_s[1] > 1:
                    return l, base_model, "base_model"
            except Exception:
                continue

    # 2. Search outer model
    if layer_name:
        for l in model.layers:
            if l.name == layer_name:
                return l, model, "outer"

    for l in reversed(model.layers):
        try:
            out_s = l.output_shape
            if isinstance(out_s, list):
                out_s = out_s[0]
            if len(out_s) == 4 and out_s[1] is not None and out_s[1] > 1:
                return l, model, "outer"
        except Exception:
            continue

    return None, None, "unknown"


def find_target_conv_layer(model: keras.Model, model_name: Optional[str] = None) -> Tuple[Optional[str], str, Optional[Tuple]]:
    """
    Automatically identifies the target final 4D convolutional feature map layer for Grad-CAM.
    Prefers explicit architecture layers (block5_conv3, conv5_block16_concat, stage3_block2_conv2)
    with rank-4 fallback. Returns (layer_name, location_type, output_shape).
    """
    pref_layer = PREFERRED_CONV_LAYERS.get(model_name) if model_name else None
    target_layer, container, loc_type = get_target_layer_and_container(model, pref_layer)

    if target_layer is not None:
        try:
            out_s = target_layer.output_shape
            if isinstance(out_s, list):
                out_s = out_s[0]
            return target_layer.name, loc_type, out_s
        except Exception:
            return target_layer.name, loc_type, None

    return None, "unknown", None


# ---------------------------------------------------------
# 2. Grad-CAM Computation Engine
# ---------------------------------------------------------
def generate_gradcam_heatmap(
    model: keras.Model,
    img_tensor: tf.Tensor,
    class_index: int,
    layer_name: Optional[str] = None
) -> Tuple[np.ndarray, str]:
    """
    Generates a Grad-CAM heatmap for a target class index on a given input image tensor.
    Supports both outer functional models and nested backbone Keras models safely in Keras 3.
    Returns (heatmap, selected_layer_name).
    """
    target_layer, container, location_type = get_target_layer_and_container(model, layer_name)

    if target_layer is None:
        err_msg = f"Could not locate target conv layer '{layer_name}' in model."
        exc_logger.error(err_msg)
        raise ValueError(err_msg)

    actual_layer_name = target_layer.name

    if location_type == "base_model" and container is not None:
        base_model = container
        
        # In Keras 3, resolve single input tensor safely to avoid multi-input list mismatch
        if hasattr(base_model, "input") and not isinstance(base_model.input, (list, tuple)) and not str(type(base_model.input)).endswith("list'>"):
            base_input = base_model.input
        elif hasattr(base_model, "inputs") and len(base_model.inputs) > 0:
            base_input = base_model.inputs[0]
        else:
            base_input = base_model.input

        base_sub_model = keras.Model(
            inputs=base_input,
            outputs=[target_layer.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_tensor, tf.float32)
            tape.watch(img_tensor)

            conv_output, base_out = base_sub_model(img_tensor, training=False)
            tape.watch(conv_output)

            # Pass base_out through remaining classification head layers of outer model
            x = base_out
            for head_layer in model.layers:
                if head_layer != base_model and not isinstance(head_layer, keras.layers.InputLayer) and head_layer.name != "input_image":
                    x = head_layer(x, training=False)

            preds = x
            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_output)

    else:
        # Outer model layer
        if hasattr(model, "input") and not isinstance(model.input, (list, tuple)) and not str(type(model.input)).endswith("list'>"):
            model_input = model.input
        elif hasattr(model, "inputs") and len(model.inputs) > 0:
            model_input = model.inputs[0]
        else:
            model_input = model.input

        grad_model = keras.Model(
            inputs=model_input,
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_tensor, tf.float32)
            tape.watch(img_tensor)

            conv_output, preds = grad_model(img_tensor, training=False)
            tape.watch(conv_output)

            loss = preds[:, class_index]

        grads = tape.gradient(loss, conv_output)

    if grads is None:
        conv_out_val = conv_output[0].numpy()
        heatmap = np.mean(conv_out_val, axis=-1)
    else:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output_val = conv_output[0]
        heatmap = conv_output_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

    # Apply ReLU and normalize between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap, actual_layer_name



def superimpose_heatmap_overlay(
    img_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet"
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Superimposes a 2D Grad-CAM heatmap onto an original image.
    Returns (orig_pil, heatmap_pil, overlay_pil).
    """
    orig_img = Image.open(img_path).convert("RGB")
    orig_img_resized = orig_img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT), Image.BILINEAR)
    orig_arr = np.array(orig_img_resized)

    # Resize heatmap to match image dimensions
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((Config.IMG_WIDTH, Config.IMG_HEIGHT), Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # Convert heatmap to RGB using colormap (compatible with all Matplotlib versions)
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        cmap = cm.get_cmap(colormap)

    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # Drop alpha
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Compute weighted overlay
    overlay_arr = np.uint8(alpha * colored_heatmap + (1.0 - alpha) * orig_arr)

    heatmap_rgb_pil = Image.fromarray(colored_heatmap)
    overlay_pil = Image.fromarray(overlay_arr)

    return orig_img_resized, heatmap_rgb_pil, overlay_pil



# ---------------------------------------------------------
# 3. Batch Sample Selector & Grad-CAM Generator
# ---------------------------------------------------------
def select_explainability_test_samples() -> Tuple[pd.DataFrame, Dict]:
    """
    Selects representative test samples (3 correct + 2 incorrect per class) from test_split.csv.
    """
    preprocessed_dir = Config.get_preprocessed_dir()
    test_csv_path = os.path.join(preprocessed_dir, "test_split.csv")
    
    if not os.path.exists(test_csv_path):
        err_msg = f"Test split dataset missing at {test_csv_path}."
        exc_logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    df_test = pd.read_csv(test_csv_path)
    
    if "class_name" in df_test.columns:
        class_names = sorted(df_test["class_name"].unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        idx_to_class = {i: c for i, c in enumerate(class_names)}
        labels_idx = df_test["label"].values if "label" in df_test.columns else np.array([class_to_idx[c] for c in df_test["class_name"]])
    else:
        class_names = Config.DEFAULT_CLASSES
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        idx_to_class = {i: c for i, c in enumerate(class_names)}
        labels_idx = df_test["label"].values

    df_test["label_idx"] = labels_idx
    df_test["class_clean"] = [idx_to_class[i] for i in labels_idx]

    return df_test, {"class_names": class_names, "class_to_idx": class_to_idx, "idx_to_class": idx_to_class}


# ---------------------------------------------------------
# 4. Main Phase 8 Explainability Pipeline Execution
# ---------------------------------------------------------
def run_phase8_explainability_pipeline() -> Dict:
    """
    Main Phase 8 pipeline entrypoint.
    Generates Grad-CAM visualizations for 03_VGG16, 05_DenseNet121, 08_ConvNeXtTiny,
    fuses Soft-Voting Ensemble CAMs, analyzes misclassification errors, and exports all reports.
    """
    pipeline_start_time = time.time()
    set_seed(Config.SEED)

    root_explainability_dir = Config.get_explainability_dir()
    gradcam_root = Config.get_explainability_dir("GradCAM")
    ensemble_gradcam_dir = Config.get_explainability_dir(os.path.join("Ensemble", "ensemble_explanations"))

    logger.info(f"\n=======================================================================")
    logger.info(f"  PHASE 8: EXPLAINABILITY & GRAD-CAM PIPELINE")
    logger.info(f"=======================================================================")

    # 1. Load Models & Ensemble Weights
    models, model_names = load_ensemble_models()
    weights_path = os.path.join(Config.get_ensemble_dir(), "ensemble_weights.json")

    if not os.path.exists(weights_path):
        weights = {name: 1.0 / len(model_names) for name in model_names}
    else:
        with open(weights_path, "r") as f:
            weights = json.load(f)["selected_weights"]

    # 2. Layer Selection & Registration
    layer_selection = {}
    for name in model_names:
        layer_name, loc_type, out_shape = find_target_conv_layer(models[name], model_name=name)
        layer_selection[name] = {
            "target_conv_layer": layer_name,
            "location_type": loc_type,
            "output_shape": [str(s) for s in out_shape] if out_shape else []
        }
        logger.info(f"Grad-CAM Layer Selection for '{name}': layer='{layer_name}' (type={loc_type})")


    # Save layer selection JSON
    with open(os.path.join(root_explainability_dir, "gradcam_layer_selection.json"), "w") as f:
        json.dump(layer_selection, f, indent=4)

    # 3. Load Test Dataset & Samples
    df_test, mappings = select_explainability_test_samples()
    class_names = mappings["class_names"]
    idx_to_class = mappings["idx_to_class"]

    # 4. Generate Batched Predictions to Identify Correct vs Incorrect Samples
    test_paths = df_test["file_path"].tolist()
    test_labels = df_test["label_idx"].values

    sub_probs = {}
    for name in model_names:
        test_ds = build_inference_dataset(test_paths, batch_size=Config.BATCH_SIZE)
        sub_probs[name] = models[name].predict(test_ds, verbose=0)

    ensemble_probs = sum(weights[name] * sub_probs[name] for name in model_names)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_confs = np.max(ensemble_probs, axis=1)
    correctness = (ensemble_preds == test_labels)

    df_test["ens_pred"] = ensemble_preds
    df_test["ens_conf"] = ensemble_confs
    df_test["is_correct"] = correctness

    # 5. Select Representative Samples (Max 5 per class: 3 correct, 2 incorrect)
    selected_indices = []
    for c_idx, c_name in enumerate(class_names):
        cls_df = df_test[df_test["label_idx"] == c_idx]
        
        corr_idx = cls_df[cls_df["is_correct"]].index.tolist()[:3]
        incorr_idx = cls_df[~cls_df["is_correct"]].index.tolist()[:2]
        
        selected_indices.extend(corr_idx + incorr_idx)

    selected_indices = sorted(list(set(selected_indices)))
    logger.info(f"Selected {len(selected_indices)} representative test samples for Grad-CAM generation.")

    # 6. Generate Grad-CAM for Individual Models & Ensemble Fusion
    generated_images_count = 0
    ensemble_records = []

    for sample_rank, sample_idx in enumerate(selected_indices):
        img_p = df_test.loc[sample_idx, "file_path"]
        true_lbl = df_test.loc[sample_idx, "label_idx"]
        true_name = idx_to_class[true_lbl]
        
        if not os.path.exists(img_p):
            continue

        img_tensor = parse_inference_image(img_p)
        img_tensor_batch = tf.expand_dims(img_tensor, axis=0)

        # Generate Heatmaps for each model
        cams = {}
        sub_preds_sample = {}
        sub_confs_sample = {}

        for name in model_names:
            p_vec = sub_probs[name][sample_idx]
            pred_cls_idx = int(np.argmax(p_vec))
            sub_preds_sample[name] = idx_to_class[pred_cls_idx]
            sub_confs_sample[name] = float(p_vec[pred_cls_idx])

            # Generate heatmap for predicted class
            hmap, target_l = generate_gradcam_heatmap(models[name], img_tensor_batch, pred_cls_idx, layer_selection[name]["target_conv_layer"])
            cams[name] = hmap

            # Overlay generation
            orig_pil, hmap_pil, overlay_pil = superimpose_heatmap_overlay(img_p, hmap)

            img_base_name = os.path.splitext(os.path.basename(img_p))[0]
            status_str = "correct" if sub_preds_sample[name] == true_name else "misclassified"
            
            # Save to model-specific GradCAM folder
            model_gradcam_dir = os.path.join(gradcam_root, name)
            os.makedirs(model_gradcam_dir, exist_ok=True)
            
            # Also save to Experiments/<MODEL>/GradCAM/
            exp_model_gradcam = os.path.join(Config.get_base_dir(), "Experiments", name, "GradCAM")
            os.makedirs(exp_model_gradcam, exist_ok=True)

            clean_true_name = true_name.replace(" ", "_")
            file_stem = f"{clean_true_name}_sample_{sample_rank+1:03d}_{status_str}"
            
            orig_pil.save(os.path.join(model_gradcam_dir, f"{file_stem}_original.jpg"))
            hmap_pil.save(os.path.join(model_gradcam_dir, f"{file_stem}_heatmap.jpg"))
            overlay_pil.save(os.path.join(model_gradcam_dir, f"{file_stem}_overlay.jpg"))

            orig_pil.save(os.path.join(exp_model_gradcam, f"{file_stem}_original.jpg"))
            overlay_pil.save(os.path.join(exp_model_gradcam, f"{file_stem}_overlay.jpg"))

            generated_images_count += 3


        # Compute Ensemble Weighted Average CAM
        ens_pred_idx = int(df_test.loc[sample_idx, "ens_pred"])
        ens_pred_name = idx_to_class[ens_pred_idx]
        ens_conf_val = float(df_test.loc[sample_idx, "ens_conf"])

        # Resize each model's CAM to 224x224 and fuse with weights
        fused_cam = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.float32)
        for name in model_names:
            h_pil = Image.fromarray(np.uint8(255 * cams[name])).resize((Config.IMG_WIDTH, Config.IMG_HEIGHT), Image.BILINEAR)
            h_norm = np.array(h_pil) / 255.0
            fused_cam += weights[name] * h_norm

        max_fused = np.max(fused_cam)
        if max_fused > 0:
            fused_cam = fused_cam / max_fused

        orig_pil, ens_hmap_pil, ens_overlay_pil = superimpose_heatmap_overlay(img_p, fused_cam)

        ens_status_str = "correct" if ens_pred_name == true_name else "misclassified"
        ens_stem = f"Ensemble_{clean_true_name}_sample_{sample_rank+1:03d}_{ens_status_str}"
        
        orig_pil.save(os.path.join(ensemble_gradcam_dir, f"{ens_stem}_original.jpg"))
        ens_hmap_pil.save(os.path.join(ensemble_gradcam_dir, f"{ens_stem}_heatmap.jpg"))
        ens_overlay_pil.save(os.path.join(ensemble_gradcam_dir, f"{ens_stem}_overlay.jpg"))
        
        generated_images_count += 3


        ensemble_records.append({
            "sample_index": sample_idx,
            "image_name": os.path.basename(img_p),
            "true_class": true_name,
            "ensemble_prediction": ens_pred_name,
            "ensemble_confidence": round(ens_conf_val, 4),
            "is_uncertain": bool(ens_conf_val < Config.CONFIDENCE_THRESHOLD),
            "is_correct": bool(ens_pred_name == true_name),
            "VGG16_prediction": sub_preds_sample["03_VGG16"],
            "VGG16_confidence": round(sub_confs_sample["03_VGG16"], 4),
            "DenseNet121_prediction": sub_preds_sample["05_DenseNet121"],
            "DenseNet121_confidence": round(sub_confs_sample["05_DenseNet121"], 4),
            "ConvNeXtTiny_prediction": sub_preds_sample["08_ConvNeXtTiny"],
            "ConvNeXtTiny_confidence": round(sub_confs_sample["08_ConvNeXtTiny"], 4),
            "ensemble_overlay_path": os.path.join(ensemble_gradcam_dir, f"{ens_stem}_overlay.jpg")
        })

    # Export ensemble_prediction_analysis.csv
    ens_df = pd.DataFrame(ensemble_records)
    ens_df.to_csv(os.path.join(Config.get_explainability_dir("Ensemble"), "ensemble_prediction_analysis.csv"), index=False)

    # 7. Misclassification Error Analysis
    misclassified_df = df_test[~df_test["is_correct"]].copy()
    misclassified_records = []
    for _, row in misclassified_df.iterrows():
        s_idx = row.name
        img_p = row["file_path"]
        true_name = idx_to_class[row["label_idx"]]
        ens_p_name = idx_to_class[row["ens_pred"]]

        misclassified_records.append({
            "image_name": os.path.basename(img_p),
            "true_class": true_name,
            "predicted_class": ens_p_name,
            "ensemble_confidence": round(float(row["ens_conf"]), 4),
            "VGG16_prediction": idx_to_class[int(np.argmax(sub_probs["03_VGG16"][s_idx]))],
            "DenseNet121_prediction": idx_to_class[int(np.argmax(sub_probs["05_DenseNet121"][s_idx]))],
            "ConvNeXtTiny_prediction": idx_to_class[int(np.argmax(sub_probs["08_ConvNeXtTiny"][s_idx]))]
        })
    
    misc_df = pd.DataFrame(misclassified_records)
    misc_df.to_csv(os.path.join(root_explainability_dir, "misclassification_analysis.csv"), index=False)

    total_pipeline_time = time.time() - pipeline_start_time

    # 8. Export Summary JSON
    summary_data = {
        "pipeline_phase": "Phase 8 - Explainability & Grad-CAM Visualization",
        "selected_ensemble_models": list(model_names),
        "gradcam_layers": layer_selection,
        "total_test_samples_evaluated": len(selected_indices),
        "total_explainability_images_generated": generated_images_count,
        "total_misclassifications_analyzed": len(misclassified_records),
        "ensemble_combination_method": "Weighted spatial average of 224x224 normalized CAM heatmaps using validation-selected soft-voting weights",
        "uncertainty_policy": f"Predictions with confidence < {Config.CONFIDENCE_THRESHOLD*100:.0f}% output '{Config.UNCERTAIN_PREDICTION_MESSAGE}'",
        "pipeline_execution_time_seconds": round(total_pipeline_time, 2)
    }

    with open(os.path.join(root_explainability_dir, "explainability_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # 9. Formatted Console Summary
    print(f"\n=======================================================================")
    print(f"  PHASE 8 EXPLAINABILITY & GRAD-CAM SUMMARY")
    print(f"=======================================================================")
    print(f"Selected Conv Layers:")
    for m_n, info in layer_selection.items():
        print(f"  - {m_n:<15}: Layer = '{info['target_conv_layer']}' (Type: {info['location_type']})")
    print(f"\nGenerated Visualizations: {generated_images_count} image files across {len(selected_indices)} representative test samples.")
    print(f"Misclassification Analysis: Analyzed {len(misclassified_records)} incorrect predictions.")
    print(f"Execution Time: {total_pipeline_time:.2f} seconds.")
    print(f"Outputs saved to: {root_explainability_dir}\n")

    return summary_data
