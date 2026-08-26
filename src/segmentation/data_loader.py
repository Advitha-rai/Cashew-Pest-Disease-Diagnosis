"""
Cashew Pest and Disease Diagnosis System
Phase C.2: Segmentation Dataset Loader & Data Pipeline
Framework: TensorFlow / Keras
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None

from .config import (
    CLASS_CODES,
    ALLOWED_MASK_VALUES,
    ANNOTATABLE_SPLITS,
    READ_ONLY_SPLIT,
    CANONICAL_MANIFEST,
    ANNOTATIONS_DIR,
    DATASET_DIR,
    normalize_class_name,
    get_class_code,
)
from .manifest import load_manifest


class SegmentationDatasetLoader:
    """
    Loads, audits, and parses validated manual segmentation annotations
    into binary foreground/background training pipelines.
    """

    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 4,
        storage_root: Optional[Path] = None,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.storage_root = storage_root or self._discover_storage_root()

        if manifest_path is None:
            candidates = [
                self.storage_root / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv",
                Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project/Experiments/Segmentation/segmentation_annotation_manifest.csv"),
                Path("/content/Cashew-Pest-Disease-Diagnosis/Experiments/Segmentation/segmentation_annotation_manifest.csv"),
                CANONICAL_MANIFEST,
            ]
            for c in candidates:
                if c.exists():
                    manifest_path = c
                    break
            if manifest_path is None:
                manifest_path = CANONICAL_MANIFEST

        self.manifest_path = manifest_path
        self.audit_summary: Dict[str, Any] = {}

    def _discover_storage_root(self) -> Path:
        if Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
            return Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
        elif Path("/content/Cashew-Pest-Disease-Diagnosis").exists():
            return Path("/content/Cashew-Pest-Disease-Diagnosis")
        return Path.cwd()

    def audit_and_load_records(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Loads all validated Train and Validation records from the canonical manifest.
        Excludes Test split completely and verifies mask existence & pixel validity.
        """
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Canonical manifest not found at: {self.manifest_path}")

        df = load_manifest(self.manifest_path)
        total_rows = len(df)

        # Split distributions
        train_total = int((df["split"] == "Train").sum()) if "split" in df.columns else 0
        val_total = int((df["split"] == "Validation").sum()) if "split" in df.columns else 0
        test_total = int((df["split"] == "Test").sum()) if "split" in df.columns else 0
        eligible_total = train_total + val_total

        # Filter strictly for validated annotations in eligible splits
        df_eligible = df[df["split"].isin(ANNOTATABLE_SPLITS)]
        df_annotated = df_eligible[
            (df_eligible["annotation_status"] == "ANNOTATED")
            & (df_eligible["validation_status"] == "PASSED")
        ]

        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []
        invalid_masks: List[Dict[str, Any]] = []
        missing_masks: List[Dict[str, Any]] = []

        ann_base = self.storage_root / "Experiments" / "Segmentation" / "Annotations"
        ds_base = self.storage_root / "Dataset" / "Cleaned"

        for _, row in df_annotated.iterrows():
            img_name = str(row.get("image_name", ""))
            raw_cls = str(row.get("class_name", ""))
            split_name = str(row.get("split", "Train"))
            exp_mask_str = str(row.get("expected_mask_path", ""))

            norm_cls = normalize_class_name(raw_cls)
            expected_code = CLASS_CODES.get(norm_cls, -1)

            # Resolve physical image path
            img_path = Path(str(row.get("image_path", "")))
            if not img_path.exists():
                img_path = ds_base / norm_cls / img_name
                if not img_path.exists():
                    img_path = Path.cwd() / "Dataset" / "Cleaned" / norm_cls / img_name

            # Resolve physical mask path
            mask_path = Path(exp_mask_str)
            if not mask_path.exists():
                mask_path = ann_base / split_name / norm_cls / Path(exp_mask_str).name
                if not mask_path.exists():
                    mask_path = ANNOTATIONS_DIR / split_name / norm_cls / Path(exp_mask_str).name

            if not mask_path.exists():
                missing_masks.append({
                    "image_name": img_name,
                    "class_name": norm_cls,
                    "split": split_name,
                    "expected_mask_path": str(mask_path),
                })
                continue

            # Verify mask format & unique values
            try:
                with Image.open(mask_path) as pil_mask:
                    mask_arr = np.asarray(pil_mask)
                    u_vals = set(np.unique(mask_arr))

                if not u_vals.issubset({0, expected_code}):
                    invalid_masks.append({
                        "image_name": img_name,
                        "class_name": norm_cls,
                        "mask_path": str(mask_path),
                        "unique_values": list(u_vals),
                        "expected_code": expected_code,
                    })
                    continue

                record = {
                    "image_name": img_name,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "class_name": norm_cls,
                    "class_code": expected_code,
                    "split": split_name,
                }

                if split_name == "Train":
                    train_records.append(record)
                elif split_name == "Validation":
                    val_records.append(record)

            except Exception as exc:
                invalid_masks.append({
                    "image_name": img_name,
                    "class_name": norm_cls,
                    "mask_path": str(mask_path),
                    "error": str(exc),
                })

        self.audit_summary = {
            "total_manifest_rows": total_rows,
            "train_rows": train_total,
            "val_rows": val_total,
            "test_rows_isolated": test_total,
            "eligible_rows": eligible_total,
            "annotated_passed_total": len(df_annotated),
            "validated_train_samples": len(train_records),
            "validated_val_samples": len(val_records),
            "missing_masks_count": len(missing_masks),
            "invalid_masks_count": len(invalid_masks),
            "test_split_protected": True,
        }

        return train_records, val_records

    def load_image_and_binary_mask(
        self,
        image_path: str,
        mask_path: str,
        class_code: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads image and mask from disk, normalizes image to [0, 1] RGB float32,
        and converts mask into single-channel binary foreground float32 array (0.0 or 1.0).
        """
        # 1. Load Image
        if Path(image_path).exists():
            with Image.open(image_path) as pil_img:
                img_rgb = pil_img.convert("RGB").resize(self.image_size)
                img_arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
        else:
            img_arr = np.zeros((*self.image_size, 3), dtype=np.float32)

        # 2. Load Mask & Convert to Binary Foreground
        with Image.open(mask_path) as pil_mask:
            mask_l = pil_mask.convert("L").resize(self.image_size, resample=Image.NEAREST)
            raw_mask = np.asarray(mask_l, dtype=np.uint8)

        # Foreground is 1.0 where pixel equals target class code (or > 0)
        bin_mask = (raw_mask == class_code).astype(np.float32)
        bin_mask = np.expand_dims(bin_mask, axis=-1)  # Shape: (224, 224, 1)

        return img_arr, bin_mask

    def create_tf_dataset(
        self,
        records: List[Dict[str, Any]],
        is_training: bool = True,
    ):
        """Builds a tf.data.Dataset pipeline from validated records."""
        if tf is None:
            raise ImportError("TensorFlow is required to build tf.data.Dataset pipelines.")

        if len(records) == 0:
            # Return empty dataset matching schema
            return tf.data.Dataset.from_tensor_slices((
                tf.zeros((0, *self.image_size, 3), dtype=tf.float32),
                tf.zeros((0, *self.image_size, 1), dtype=tf.float32),
            ))

        def _generator():
            for rec in records:
                img_arr, bin_mask = self.load_image_and_binary_mask(
                    rec["image_path"],
                    rec["mask_path"],
                    rec["class_code"],
                )
                yield img_arr, bin_mask

        output_signature = (
            tf.TensorSpec(shape=(*self.image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*self.image_size, 1), dtype=tf.float32),
        )

        dataset = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)

        if is_training:
            dataset = dataset.shuffle(buffer_size=max(10, len(records)))

        actual_batch = min(self.batch_size, len(records)) if len(records) > 0 else self.batch_size
        dataset = dataset.batch(actual_batch).prefetch(tf.data.AUTOTUNE)
        return dataset
