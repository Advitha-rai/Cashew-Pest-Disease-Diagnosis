"""
Cashew Pest and Disease Diagnosis System
Phase C.2: Segmentation Model Evaluator & Visualization Generator
Framework: TensorFlow / Keras
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .data_loader import SegmentationDatasetLoader


class SegmentationEvaluator:
    """Evaluates trained U-Net, computes metrics, and generates visual predictions & training curves."""

    def __init__(
        self,
        model: Any,
        data_loader: SegmentationDatasetLoader,
        experiment_dir: Path,
    ):
        self.model = model
        self.data_loader = data_loader
        self.experiment_dir = experiment_dir
        self.viz_dir = self.experiment_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_metrics(
        self,
        records: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Calculates quantitative metrics across validation records."""
        if len(records) == 0:
            return {
                "sample_count": 0,
                "mean_dice": 0.0,
                "mean_iou": 0.0,
                "mean_accuracy": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
            }

        dices = []
        ious = []
        accuracies = []
        precisions = []
        recalls = []

        smooth = 1e-6

        for rec in records:
            img_arr, gt_mask = self.data_loader.load_image_and_binary_mask(
                rec["image_path"],
                rec["mask_path"],
                rec["class_code"],
            )

            pred_raw = self.model.predict(np.expand_dims(img_arr, axis=0), verbose=0)[0]
            pred_bin = (pred_raw >= threshold).astype(np.float32)

            gt_f = gt_mask.flatten()
            pred_f = pred_bin.flatten()

            intersection = np.sum(gt_f * pred_f)
            total = np.sum(gt_f) + np.sum(pred_f)
            union = total - intersection

            dice = (2.0 * intersection + smooth) / (total + smooth)
            iou = (intersection + smooth) / (union + smooth)
            acc = np.mean(gt_f == pred_f)

            tp = intersection
            fp = np.sum((pred_f == 1.0) & (gt_f == 0.0))
            fn = np.sum((pred_f == 0.0) & (gt_f == 1.0))

            prec = (tp + smooth) / (tp + fp + smooth)
            rec = (tp + smooth) / (tp + fn + smooth)

            dices.append(float(dice))
            ious.append(float(iou))
            accuracies.append(float(acc))
            precisions.append(float(prec))
            recalls.append(float(rec))

        metrics_summary = {
            "sample_count": len(records),
            "mean_dice": float(np.mean(dices)),
            "mean_iou": float(np.mean(ious)),
            "mean_accuracy": float(np.mean(accuracies)),
            "mean_precision": float(np.mean(precisions)),
            "mean_recall": float(np.mean(recalls)),
        }

        with open(self.experiment_dir / "evaluation_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=4)

        return metrics_summary

    def generate_visualizations(
        self,
        records: List[Dict[str, Any]],
        max_samples: int = 10,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Generates 4-panel visual comparison figures:
        1. Original RGB Image
        2. Ground-Truth Binary Mask
        3. Predicted Binary Mask
        4. Original + Predicted Mask Overlay
        """
        if plt is None or len(records) == 0:
            return []

        saved_plots = []
        samples_to_plot = records[: min(max_samples, len(records))]

        for idx, rec in enumerate(samples_to_plot):
            img_arr, gt_mask = self.data_loader.load_image_and_binary_mask(
                rec["image_path"],
                rec["mask_path"],
                rec["class_code"],
            )

            pred_raw = self.model.predict(np.expand_dims(img_arr, axis=0), verbose=0)[0]
            pred_bin = (pred_raw >= threshold).astype(np.float32)

            # Create RGB Overlay: red lesion overlay on original image
            overlay = img_arr.copy()
            pred_mask_2d = pred_bin[:, :, 0]
            overlay[pred_mask_2d == 1.0, 0] = np.clip(overlay[pred_mask_2d == 1.0, 0] * 0.5 + 0.5, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            fig.suptitle(
                f"Sample {idx+1}: {rec['image_name']} | Class: {rec['class_name']} ({rec['split']})",
                fontsize=13,
                fontweight="bold",
            )

            axes[0].imshow(img_arr)
            axes[0].set_title("1. Original Image (224x224)")
            axes[0].axis("off")

            axes[1].imshow(gt_mask[:, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[1].set_title(f"2. Ground Truth ({rec['class_name']})")
            axes[1].axis("off")

            axes[2].imshow(pred_bin[:, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("3. Predicted Mask (U-Net)")
            axes[2].axis("off")

            axes[3].imshow(overlay)
            axes[3].set_title("4. Predicted Overlay")
            axes[3].axis("off")

            plt.tight_layout()
            stem = Path(rec["image_name"]).stem
            out_file = self.viz_dir / f"pred_{idx+1:02d}_{stem}.png"
            fig.savefig(str(out_file), dpi=150, bbox_inches="tight")

            # Also save primary prediction_visualizations.png
            if idx == 0:
                primary_viz = self.experiment_dir / "prediction_visualizations.png"
                fig.savefig(str(primary_viz), dpi=150, bbox_inches="tight")

            plt.close(fig)
            saved_plots.append(str(out_file))

        print(f"[EVALUATOR] Saved {len(saved_plots)} visualization panels to: {self.viz_dir}")
        return saved_plots

    def plot_training_curves(self, history: Dict[str, List[float]]) -> Optional[str]:
        """Plots and saves loss, Dice, IoU, and accuracy curves over training epochs."""
        if plt is None or not history:
            return None

        epochs = range(1, len(history.get("loss", [])) + 1)
        if len(epochs) == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("U-Net Segmentation Training & Validation Curves", fontsize=15, fontweight="bold")

        # 1. Loss
        ax = axes[0, 0]
        if "loss" in history:
            ax.plot(epochs, history["loss"], label="Train Loss", color="#1F77B4", lw=2)
        if "val_loss" in history:
            ax.plot(epochs, history["val_loss"], label="Val Loss", color="#FF7F0E", lw=2, ls="--")
        ax.set_title("BCE-Dice Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Dice Coefficient
        ax = axes[0, 1]
        if "dice_coef" in history:
            ax.plot(epochs, history["dice_coef"], label="Train Dice", color="#2CA02C", lw=2)
        if "val_dice_coef" in history:
            ax.plot(epochs, history["val_dice_coef"], label="Val Dice", color="#D62728", lw=2, ls="--")
        ax.set_title("Dice Coefficient")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. IoU (Jaccard Index)
        ax = axes[1, 0]
        if "iou_metric" in history:
            ax.plot(epochs, history["iou_metric"], label="Train IoU", color="#9467BD", lw=2)
        if "val_iou_metric" in history:
            ax.plot(epochs, history["val_iou_metric"], label="Val IoU", color="#8C564B", lw=2, ls="--")
        ax.set_title("Intersection over Union (IoU)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IoU")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 4. Binary Accuracy
        ax = axes[1, 1]
        if "binary_accuracy" in history:
            ax.plot(epochs, history["binary_accuracy"], label="Train Accuracy", color="#17BECF", lw=2)
        if "val_binary_accuracy" in history:
            ax.plot(epochs, history["val_binary_accuracy"], label="Val Accuracy", color="#BCBD22", lw=2, ls="--")
        ax.set_title("Binary Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        curves_path = self.experiment_dir / "training_curves.png"
        fig.savefig(str(curves_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVALUATOR] Saved training curves to: {curves_path}")
        return str(curves_path)
