"""
Cashew Pest and Disease Diagnosis System
Phase C: Strict Mask Validation & Dataset Protection Engine
Framework: TensorFlow / Keras
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, Set, List
import numpy as np
import pandas as pd
from PIL import Image

from .config import SegmentationConfig, normalize_class_name, get_class_code


def compute_file_hash(file_path: str | Path) -> Optional[str]:
    """Computes SHA-256 hash of a file for integrity verification."""
    path = Path(file_path)
    if not path.exists():
        return None

    sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return None


def _get_test_paths(test_csv_path: str | Path) -> Set[str]:
    """Extracts normalized absolute file paths belonging to the isolated Test set."""
    path = Path(test_csv_path)
    if not path.exists():
        return set()

    try:
        df = pd.read_csv(path)
        col = "file_path" if "file_path" in df.columns else ("image_path" if "image_path" in df.columns else None)
        if not col:
            return set()
        return {
            os.path.normcase(os.path.abspath(str(p)))
            for p in df[col].dropna()
        }
    except Exception:
        return set()


def assert_annotation_allowed(
    split: str,
    image_path: str | Path,
    test_csv_path: Optional[str | Path] = None,
) -> None:
    """
    Strict backend security enforcement:
    Rejects Test split images from manual annotation, painting, saving, or skipping.
    Raises PermissionError if annotation is attempted on the isolated Test partition.
    """
    split_clean = str(split).strip().capitalize()

    if split_clean in SegmentationConfig.PROTECTED_SPLITS:
        raise PermissionError(
            f"Test split is strictly read-only. Annotation is prohibited for split '{split}'."
        )

    if split_clean not in SegmentationConfig.ANNOTATABLE_SPLITS:
        raise PermissionError(
            f"Annotation is not allowed for split '{split}'. Must be Train or Validation."
        )

    # Check path-level inclusion in Test CSV
    if test_csv_path is None:
        default_test_csv = SegmentationConfig.get_preprocessed_dir() / "test_split.csv"
        if default_test_csv.exists():
            test_csv_path = default_test_csv

    if test_csv_path is not None:
        test_paths = _get_test_paths(test_csv_path)
        norm_img_path = os.path.normcase(os.path.abspath(str(image_path)))
        if norm_img_path in test_paths:
            raise PermissionError(
                f"Image '{image_path}' belongs to the isolated Test set. Annotation is prohibited."
            )


def validate_mask_array(
    image_path: str | Path,
    mask_array: np.ndarray,
    expected_class_code: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Strictly validates a 2D mask NumPy array against source image and project rules.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        return False, f"Source image does not exist: {image_path}", {"error_code": "IMAGE_NOT_FOUND"}

    try:
        with Image.open(img_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        return False, f"Could not read source image: {e}", {"error_code": "IMAGE_READ_ERROR"}

    if not isinstance(mask_array, np.ndarray):
        return False, "Mask must be a NumPy array", {"error_code": "INVALID_TYPE"}

    if mask_array.ndim != 2:
        return False, f"Mask must be 2D. Received shape: {mask_array.shape}", {"error_code": "MULTI_CHANNEL_MASK"}

    if mask_array.dtype != np.uint8:
        return False, f"Mask dtype must be uint8. Received: {mask_array.dtype}", {"error_code": "INVALID_DTYPE"}

    mask_h, mask_w = mask_array.shape
    if (mask_w, mask_h) != (img_w, img_h):
        return False, f"Dimension mismatch: image ({img_w}x{img_h}) vs mask ({mask_w}x{mask_h})", {
            "error_code": "DIMENSION_MISMATCH",
            "image_dimensions": [img_w, img_h],
            "mask_dimensions": [mask_w, mask_h]
        }

    unique_vals = set(np.unique(mask_array))
    invalid_vals = unique_vals - SegmentationConfig.ALLOWED_MASK_VALUES
    if invalid_vals:
        return False, f"Invalid pixel values: {sorted(list(invalid_vals))}. Only 0-4 allowed.", {
            "error_code": "INVALID_PIXEL_VALUES",
            "unique_values": sorted(list(unique_vals))
        }

    foreground_pixels = int(np.count_nonzero(mask_array))
    if foreground_pixels == 0:
        return False, "Validation failed: Mask is completely empty (0 foreground pixels).", {
            "error_code": "EMPTY_MASK"
        }

    expected_class_code = int(expected_class_code)
    if expected_class_code > 0 and expected_class_code not in unique_vals:
        return False, f"Expected class code {expected_class_code} missing from mask values {sorted(list(unique_vals))}.", {
            "error_code": "MISSING_EXPECTED_CLASS",
            "expected_class_code": expected_class_code,
            "unique_values": sorted(list(unique_vals))
        }

    total_pixels = img_w * img_h
    lesion_pct = round((foreground_pixels / total_pixels) * 100.0, 4)

    meta = {
        "image_dimensions": [img_w, img_h],
        "mask_dimensions": [mask_w, mask_h],
        "mask_dtype": "uint8",
        "unique_pixel_values": sorted(list(unique_vals)),
        "foreground_pixels": foreground_pixels,
        "lesion_pixel_percentage": lesion_pct,
        "expected_class_code": expected_class_code,
    }

    return True, "PASSED", meta


def validate_mask_file(
    image_path: str | Path,
    mask_path: str | Path,
    expected_class_code: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Reopens a saved mask file from disk and validates integrity, mode, and pixels.
    """
    m_path = Path(mask_path)
    if not m_path.exists():
        return False, f"Mask file does not exist: {mask_path}", {"error_code": "FILE_NOT_FOUND"}

    try:
        with Image.open(m_path) as m_img:
            m_img.load()
            if m_img.mode == "RGBA":
                rgba = np.array(m_img)
                # Map alpha and red channel to discrete uint8
                mask_2d = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint8)
                valid_fg = (rgba[:, :, 3] > 0) & (rgba[:, :, 0] > 0)
                mask_2d[valid_fg] = rgba[:, :, 0][valid_fg]
                arr = mask_2d
            else:
                arr = np.asarray(m_img.convert("L"), dtype=np.uint8)

        return validate_mask_array(image_path, arr, expected_class_code)

    except Exception as e:
        return False, f"Corrupted mask image or read error: {e}", {"error_code": "CORRUPTED_FILE"}
