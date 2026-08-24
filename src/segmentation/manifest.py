"""
Cashew Pest and Disease Diagnosis System
Phase C: Deterministic Annotation Manifest & Status Accounting Engine
Framework: TensorFlow / Keras
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from PIL import Image

from .config import SegmentationConfig, normalize_class_name, get_class_code


def load_manifest(manifest_csv: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads the canonical segmentation annotation manifest DataFrame."""
    path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    if not path.exists():
        return build_segmentation_manifest(manifest_csv=path)

    df = pd.read_csv(path)
    required_cols = [
        "image_path", "image_name", "class_name", "class_code", "split",
        "expected_mask_path", "annotation_status", "validation_status", "error_message"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "PENDING" if "status" in col else ("UNVALIDATED" if col == "validation_status" else "")

    return df


def save_manifest_atomically(df: pd.DataFrame, manifest_csv: Optional[str | Path] = None) -> bool:
    """Saves DataFrame atomically via temporary file replacement."""
    target_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_file = target_path.with_suffix(".tmp")
    try:
        df.to_csv(temp_file, index=False)
        os.replace(temp_file, target_path)
        return True
    except Exception as e:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise IOError(f"Atomic manifest save failed for {target_path}: {e}")


def build_segmentation_manifest(
    force_rebuild: bool = False,
    manifest_csv: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Constructs/updates the 5,734-image segmentation annotation manifest from classification splits.
    Preserves existing ANNOTATED, SKIPPED, PASSED statuses across runs.
    """
    target_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    annotations_base = SegmentationConfig.get_annotations_dir()

    existing_status_map = {}
    if target_path.exists() and not force_rebuild:
        try:
            df_old = pd.read_csv(target_path)
            for _, r in df_old.iterrows():
                key = (str(r.get("image_path", "")), str(r.get("split", "")))
                existing_status_map[key] = {
                    "annotation_status": str(r.get("annotation_status", "PENDING")),
                    "validation_status": str(r.get("validation_status", "UNVALIDATED")),
                    "error_message": str(r.get("error_message", "Pending manual annotation")),
                    "mask_sha256": str(r.get("mask_sha256", "")),
                }
        except Exception:
            existing_status_map = {}

    preprocessed_dir = SegmentationConfig.get_preprocessed_dir()
    splits_to_scan = [
        ("Train", preprocessed_dir / "train_split.csv"),
        ("Validation", preprocessed_dir / "val_split.csv"),
        ("Test", preprocessed_dir / "test_split.csv"),
    ]

    manifest_rows: List[Dict[str, Any]] = []

    for split_name, csv_path in splits_to_scan:
        if not csv_path.exists():
            continue

        df_split = pd.read_csv(csv_path)
        img_col = "file_path" if "file_path" in df_split.columns else "image_path"
        cls_col = "class_name" if "class_name" in df_split.columns else "label"

        for _, row in df_split.iterrows():
            raw_img_path = str(row[img_col])
            raw_cls = str(row[cls_col])

            try:
                norm_cls = normalize_class_name(raw_cls)
                code = get_class_code(norm_cls)
            except Exception:
                norm_cls = "Unknown"
                code = -1

            img_name = Path(raw_img_path).name
            stem = Path(raw_img_path).stem
            mask_name = f"{stem}_mask.png"
            expected_mask_path = annotations_base / split_name / norm_cls / mask_name

            # Initialize isolated directory
            expected_mask_path.parent.mkdir(parents=True, exist_ok=True)

            key = (raw_img_path, split_name)
            if key in existing_status_map:
                saved = existing_status_map[key]
                ann_status = saved["annotation_status"]
                val_status = saved["validation_status"]
                err_msg = saved["error_message"]
                m_hash = saved["mask_sha256"]
            else:
                is_mask_on_disk = expected_mask_path.exists()
                ann_status = "ANNOTATED" if is_mask_on_disk else "PENDING"
                val_status = "PENDING_VALIDATION" if is_mask_on_disk else "UNVALIDATED"
                err_msg = "Mask ready for validation" if is_mask_on_disk else "Pending manual annotation"
                m_hash = ""

            manifest_rows.append({
                "image_path": raw_img_path,
                "image_name": img_name,
                "class_name": norm_cls,
                "class_code": code,
                "split": split_name,
                "width": SegmentationConfig.IMG_WIDTH,
                "height": SegmentationConfig.IMG_HEIGHT,
                "expected_mask_path": str(expected_mask_path),
                "annotation_status": ann_status,
                "validation_status": val_status,
                "error_message": err_msg,
                "mask_sha256": m_hash
            })

    df_manifest = pd.DataFrame(manifest_rows)
    save_manifest_atomically(df_manifest, target_path)
    return df_manifest


def find_manifest_match_index(
    df: pd.DataFrame,
    image_path: str | Path,
) -> Optional[int]:
    """Finds index of matching manifest row using robust path normalization."""
    req_norm = os.path.normcase(os.path.abspath(str(image_path)))
    req_name = Path(image_path).name

    # 1. Exact absolute path match
    for idx, row in df.iterrows():
        if os.path.normcase(os.path.abspath(str(row["image_path"]))) == req_norm:
            return int(idx)

    # 2. Filename match
    for idx, row in df.iterrows():
        if Path(str(row["image_path"])).name == req_name:
            return int(idx)

    return None


def get_next_pending_image(
    split: Optional[str] = "Train",
    class_name: Optional[str] = None,
    manifest_csv: Optional[str | Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Returns next unannotated PENDING image record from eligible Train/Validation pool.
    Never returns Test images.
    """
    df = load_manifest(manifest_csv)

    # Exclude Test split unconditionally
    df_eligible = df[df["split"].isin(SegmentationConfig.ANNOTATABLE_SPLITS)].copy()

    # Filter pending
    df_pending = df_eligible[df_eligible["annotation_status"] == "PENDING"].copy()

    if split:
        split_clean = str(split).strip().capitalize()
        if split_clean in SegmentationConfig.ANNOTATABLE_SPLITS:
            df_split_filtered = df_pending[df_pending["split"] == split_clean]
            if len(df_split_filtered) > 0:
                df_pending = df_split_filtered

    if class_name:
        try:
            norm_cls = normalize_class_name(class_name)
            df_cls_filtered = df_pending[df_pending["class_name"] == norm_cls]
            if len(df_cls_filtered) > 0:
                df_pending = df_cls_filtered
        except Exception:
            pass

    if len(df_pending) == 0:
        return None

    row = df_pending.iloc[0].to_dict()
    return row


def get_annotation_progress_report(manifest_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Calculates detailed segmentation progress and verifies split accounting consistency:
      eligible = pending + annotated + skipped
      passed <= annotated
    """
    df = load_manifest(manifest_csv)

    total_images = len(df)
    train_total = int((df["split"] == "Train").sum())
    val_total = int((df["split"] == "Validation").sum())
    test_total = int((df["split"] == "Test").sum())
    eligible_total = train_total + val_total

    df_eligible = df[df["split"].isin(SegmentationConfig.ANNOTATABLE_SPLITS)]

    annotated_cnt = int((df_eligible["annotation_status"] == "ANNOTATED").sum())
    skipped_cnt = int((df_eligible["annotation_status"] == "SKIPPED").sum())
    pending_cnt = int((df_eligible["annotation_status"] == "PENDING").sum())
    passed_cnt = int((df_eligible["validation_status"] == "PASSED").sum())
    failed_cnt = int((df_eligible["validation_status"] == "FAILED").sum())

    progress_pct = round((annotated_cnt / eligible_total) * 100.0, 2) if eligible_total > 0 else 0.0

    return {
        "total_source_images": total_images,
        "train_images": train_total,
        "validation_images": val_total,
        "test_images_isolated": test_total,
        "total_eligible_images": eligible_total,
        "annotated_count": annotated_cnt,
        "skipped_count": skipped_cnt,
        "pending_count": pending_cnt,
        "passed_validation_count": passed_cnt,
        "failed_validation_count": failed_cnt,
        "progress_percentage": progress_pct,
        "accounting_valid": (eligible_total == (annotated_cnt + skipped_cnt + pending_cnt)),
        "validation_consistent": (passed_cnt <= annotated_cnt),
        "test_set_isolation_status": "STRICTLY_READ_ONLY",
    }
