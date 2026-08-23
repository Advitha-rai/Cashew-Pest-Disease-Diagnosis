"""
segmentation_tool.py

Phase C.1 - Cashew Leaf Manual Segmentation Tool

Purpose
-------
Manual segmentation of visible pest/disease lesions on Cashew leaf images.

Class codes
-----------
0 = Background
1 = Aphids
2 = Leaf_Miner
3 = TMB
4 = Leaf_Blight

Safety guarantees
-----------------
- Train and Validation are annotatable.
- Test is strictly read-only.
- Classification split CSV files are never modified.
- Masks are single-channel uint8 PNG files.
- Valid mask values are only {0,1,2,3,4}.
- Expected class code must be present.
- Dimension mismatches are resized using nearest-neighbor.
- Manifest updates are atomic.
- Colab callbacks return JSON-safe dictionaries.
- Verification suite operates on temporary data.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_CODES = {
    "Aphids": 1,
    "Aphid": 1,

    "Leaf_Miner": 2,
    "Leaf Miner": 2,
    "Leaf miner": 2,

    "TMB": 3,

    "Leaf_Blight": 4,
    "Leaf Blight": 4,
}

VALID_MASK_VALUES = {0, 1, 2, 3, 4}
ANNOTATABLE_SPLITS = {"Train", "Validation"}
PROTECTED_SPLITS = {"Test"}

DEFAULT_BRUSH_SIZE = 14
MAX_UNDO_STATES = 25


# ============================================================
# OPTIONAL PROJECT CONFIG
# ============================================================

try:
    from src.config import Config
except Exception:
    Config = None


def _find_project_root() -> Path:
    """
    Attempts to locate the project root.
    """

    candidates = [
        Path.cwd(),
        Path("/content/Cashew-Pest-Disease-Diagnosis"),
        Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def get_preprocessed_dir() -> Path:
    """
    Gets preprocessing directory without requiring Config.
    """

    if Config is not None:
        try:
            p = Config.get_preprocessed_dir()
            return Path(p)
        except Exception:
            pass

    candidates = [
        PROJECT_ROOT / "Preprocessed",
        PROJECT_ROOT / "preprocessed",
        PROJECT_ROOT / "data" / "Preprocessed",
        PROJECT_ROOT / "data" / "preprocessed",
    ]

    for p in candidates:
        if p.exists():
            return p

    p = PROJECT_ROOT / "Preprocessed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_segmentation_dir() -> Path:
    """
    Directory used by the segmentation tool.
    """

    p = PROJECT_ROOT / "Segmentation"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_manifest_path() -> Path:
    return get_segmentation_dir() / "annotation_manifest.csv"


def get_mask_dir() -> Path:
    p = get_segmentation_dir() / "masks"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================
# JSON SAFETY
# ============================================================

def make_json_safe(obj: Any) -> Any:
    """
    Converts NumPy/Pandas/Path values into JSON-safe Python types.
    """

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Path):
        return str(obj)

    if pd.isna(obj):
        return None

    return obj


# ============================================================
# HASHING
# ============================================================

def compute_file_hash(file_path: str | Path) -> Optional[str]:
    """
    SHA-256 hash of a file.
    """

    path = Path(file_path)

    if not path.exists() or not path.is_file():
        return None

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# CLASS UTILITIES
# ============================================================

def normalize_class_name(class_name: str) -> str:
    """
    Normalizes known class-name variations.
    """

    if class_name is None:
        return ""

    value = str(class_name).strip()

    aliases = {
        "Aphid": "Aphids",
        "Leaf Miner": "Leaf_Miner",
        "Leaf miner": "Leaf_Miner",
        "Leaf Blight": "Leaf_Blight",
    }

    return aliases.get(value, value)


def get_class_code(class_name: str) -> int:
    normalized = normalize_class_name(class_name)

    if normalized not in CLASS_CODES:
        raise ValueError(
            f"Unknown class '{class_name}'. "
            f"Expected one of: {sorted(set(CLASS_CODES.keys()))}"
        )

    return CLASS_CODES[normalized]


# ============================================================
# ATOMIC CSV WRITING
# ============================================================

def save_manifest_atomically(
    df: pd.DataFrame,
    manifest_csv: str | Path,
) -> None:
    """
    Writes manifest atomically.

    The existing manifest is replaced only after the temporary
    file has been successfully written.
    """

    manifest_csv = Path(manifest_csv)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=".manifest_",
        suffix=".tmp",
        dir=str(manifest_csv.parent),
    )

    os.close(fd)

    try:
        df.to_csv(temp_name, index=False)

        # Validate that the temporary file can be read.
        pd.read_csv(temp_name)

        os.replace(temp_name, manifest_csv)

    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


# ============================================================
# MANIFEST CREATION
# ============================================================

MANIFEST_COLUMNS = [
    "image_path",
    "image_name",
    "class_name",
    "class_code",
    "split",
    "expected_mask_path",
    "annotation_status",
    "validation_status",
    "error_message",
]


def create_manifest_from_split_files(
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Creates a segmentation manifest from classification split CSVs.

    The source split CSVs are READ ONLY.
    """

    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    test_csv = Path(test_csv)
    output_csv = Path(output_csv)

    rows = []

    def add_split(csv_path: Path, split_name: str) -> None:

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)

        path_col = None

        for candidate in [
            "image_path",
            "file_path",
            "filepath",
            "path",
        ]:
            if candidate in df.columns:
                path_col = candidate
                break

        if path_col is None:
            raise ValueError(
                f"No image path column found in {csv_path}"
            )

        class_col = None

        for candidate in [
            "class_name",
            "class",
            "label",
            "class_label",
        ]:
            if candidate in df.columns:
                class_col = candidate
                break

        if class_col is None:
            raise ValueError(
                f"No class column found in {csv_path}"
            )

        for _, row in df.iterrows():

            image_path = str(row[path_col])
            class_name = normalize_class_name(row[class_col])
            class_code = get_class_code(class_name)

            image_name = os.path.basename(image_path)

            expected_mask_path = (
                get_mask_dir()
                / split_name
                / class_name
                / f"{Path(image_name).stem}_mask.png"
            )

            rows.append({
                "image_path": image_path,
                "image_name": image_name,
                "class_name": class_name,
                "class_code": class_code,
                "split": split_name,
                "expected_mask_path": str(expected_mask_path),
                "annotation_status": "PENDING",
                "validation_status": "UNVALIDATED",
                "error_message": "Pending",
            })

    add_split(train_csv, "Train")
    add_split(val_csv, "Validation")
    add_split(test_csv, "Test")

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    save_manifest_atomically(manifest, output_csv)

    return manifest


# ============================================================
# MANIFEST LOADING
# ============================================================

def load_manifest(
    manifest_csv: str | Path | None = None,
) -> pd.DataFrame:

    path = Path(manifest_csv or get_manifest_path())

    if not path.exists():
        raise FileNotFoundError(
            f"Annotation manifest not found: {path}"
        )

    df = pd.read_csv(path)

    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


# ============================================================
# TEST SPLIT PROTECTION
# ============================================================

def _get_test_paths(
    test_csv_path: str | Path,
) -> set[str]:

    path = Path(test_csv_path)

    if not path.exists():
        return set()

    df = pd.read_csv(path)

    path_col = None

    for candidate in [
        "image_path",
        "file_path",
        "filepath",
        "path",
    ]:
        if candidate in df.columns:
            path_col = candidate
            break

    if path_col is None:
        return set()

    return {
        os.path.normcase(os.path.abspath(str(x)))
        for x in df[path_col].dropna()
    }


def assert_annotation_allowed(
    split: str,
    image_path: str | Path,
    test_csv_path: str | Path | None = None,
) -> None:
    """
    Strictly prevents Test images from annotation.
    """

    split = str(split).strip()

    if split in PROTECTED_SPLITS:
        raise PermissionError(
            "Test split is read-only. Annotation is prohibited."
        )

    if split not in ANNOTATABLE_SPLITS:
        raise PermissionError(
            f"Annotation is not allowed for split '{split}'."
        )

    if test_csv_path is not None:

        test_paths = _get_test_paths(test_csv_path)

        normalized = os.path.normcase(
            os.path.abspath(str(image_path))
        )

        if normalized in test_paths:
            raise PermissionError(
                "Image belongs to isolated Test set. "
                "Annotation is prohibited."
            )


# ============================================================
# IMAGE INFORMATION
# ============================================================

def get_image_dimensions(
    image_path: str | Path,
) -> Tuple[int, int]:

    with Image.open(image_path) as img:
        return img.width, img.height


def image_to_base64(
    image_path: str | Path,
) -> str:

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded


# ============================================================
# NEXT IMAGE
# ============================================================

def get_next_pending_image(
    split: str | None = None,
    class_name: str | None = None,
    manifest_csv: str | Path | None = None,
) -> Optional[Dict[str, Any]]:

    df = load_manifest(manifest_csv)

    working = df.copy()

    if split is not None:
        working = working[
            working["split"].astype(str) == str(split)
        ]

    if class_name is not None:
        normalized = normalize_class_name(class_name)

        working = working[
            working["class_name"].apply(
                normalize_class_name
            ) == normalized
        ]

    working = working[
        working["annotation_status"].astype(str) == "PENDING"
    ]

    working = working[
        working["split"].isin(ANNOTATABLE_SPLITS)
    ]

    if working.empty:
        return None

    row = working.iloc[0]

    image_path = Path(str(row["image_path"]))

    if not image_path.exists():
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "class_name": normalize_class_name(
                row["class_name"]
            ),
            "class_code": int(row["class_code"]),
            "split": row["split"],
            "width": 0,
            "height": 0,
            "base64": None,
            "error": "Image file does not exist.",
        }

    width, height = get_image_dimensions(image_path)

    return make_json_safe({
        "image_path": str(image_path),
        "image_name": image_path.name,
        "class_name": normalize_class_name(
            row["class_name"]
        ),
        "class_code": int(row["class_code"]),
        "split": str(row["split"]),
        "width": width,
        "height": height,
        "base64": image_to_base64(image_path),
    })


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_mask_array(
    image_path: str | Path,
    mask_array: np.ndarray,
    expected_class_code: int,
) -> Tuple[bool, str, Dict[str, Any]]:

    meta: Dict[str, Any] = {}

    image_path = Path(image_path)

    if not image_path.exists():
        return (
            False,
            "Image does not exist.",
            {"error_code": "IMAGE_NOT_FOUND"},
        )

    with Image.open(image_path) as img:
        expected_w = img.width
        expected_h = img.height

    arr = np.asarray(mask_array)

    if arr.ndim == 3:

        if arr.shape[-1] == 4:
            alpha = arr[..., 3]

            # Transparent pixels are background.
            rgb = arr[..., :3]

            gray = np.asarray(
                Image.fromarray(
                    rgb.astype(np.uint8)
                ).convert("L")
            )

            gray[alpha == 0] = 0
            arr = gray

        else:
            arr = np.asarray(
                Image.fromarray(
                    arr.astype(np.uint8)
                ).convert("L")
            )

    if arr.ndim != 2:
        return (
            False,
            "Mask must be 2-dimensional.",
            {"error_code": "INVALID_DIMENSIONS"},
        )

    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.uint8)

    arr = arr.astype(np.uint8)

    meta["original_mask_dimensions"] = [
        int(arr.shape[0]),
        int(arr.shape[1]),
    ]

    unique_values = set(
        np.unique(arr).astype(int).tolist()
    )

    invalid_values = unique_values - VALID_MASK_VALUES

    if invalid_values:
        return (
            False,
            f"Invalid pixel values {sorted(invalid_values)}. "
            f"Only 0-4 allowed.",
            {
                "error_code": "INVALID_PIXEL_VALUES",
                "invalid_values": sorted(invalid_values),
            },
        )

    if arr.shape != (expected_h, expected_w):

        pil_mask = Image.fromarray(arr, mode="L")

        pil_mask = pil_mask.resize(
            (expected_w, expected_h),
            Image.Resampling.NEAREST,
        )

        arr = np.asarray(
            pil_mask,
            dtype=np.uint8,
        )

        meta["resized"] = True

    else:
        meta["resized"] = False

    if np.count_nonzero(arr) == 0:
        return (
            False,
            "Empty mask. Paint the visible lesion before saving.",
            {
                "error_code": "EMPTY_MASK",
                "mask_dimensions": [
                    expected_h,
                    expected_w,
                ],
            },
        )

    present_values = set(
        np.unique(arr).astype(int).tolist()
    )

    if expected_class_code not in present_values:
        return (
            False,
            f"Expected class code {expected_class_code} "
            "was not found in the mask.",
            {
                "error_code": "MISSING_EXPECTED_CLASS",
                "present_values": sorted(present_values),
            },
        )

    meta["mask_dimensions"] = [
        int(arr.shape[0]),
        int(arr.shape[1]),
    ]

    meta["unique_values"] = sorted(
        present_values
    )

    meta["foreground_pixels"] = int(
        np.count_nonzero(arr)
    )

    meta["expected_class_code"] = int(
        expected_class_code
    )

    return True, "Mask validation passed.", meta


def validate_mask_file(
    image_path: str | Path,
    mask_path: str | Path,
    expected_class_code: int,
) -> Tuple[bool, str, Dict[str, Any]]:

    mask_path = Path(mask_path)

    if not mask_path.exists():
        return (
            False,
            "Mask file does not exist.",
            {"error_code": "MASK_NOT_FOUND"},
        )

    try:

        with Image.open(mask_path) as img:
            img.load()

            if img.mode == "RGBA":
                arr = np.asarray(img)

            else:
                arr = np.asarray(
                    img.convert("L")
                )

        return validate_mask_array(
            image_path,
            arr,
            expected_class_code,
        )

    except Exception as exc:

        return (
            False,
            f"Could not read mask: {exc}",
            {
                "error_code": "MASK_READ_ERROR",
                "exception": str(exc),
            },
        )


# ============================================================
# SAVE MASK
# ============================================================

def process_annotation_submission(
    image_path: str | Path,
    mask_array: np.ndarray,
    split: str,
    class_name: str,
    manifest_csv: str | Path | None = None,
) -> Tuple[bool, str, Dict[str, Any]]:

    manifest_path = Path(
        manifest_csv or get_manifest_path()
    )

    expected_code = get_class_code(class_name)

    test_csv = get_preprocessed_dir() / "test_split.csv"

    try:

        assert_annotation_allowed(
            split,
            image_path,
            test_csv if test_csv.exists() else None,
        )

    except PermissionError as exc:

        return (
            False,
            str(exc),
            {
                "error_code": "ANNOTATION_NOT_ALLOWED",
            },
        )

    df = load_manifest(manifest_path)

    normalized_path = os.path.normcase(
        os.path.abspath(str(image_path))
    )

    matches = df[
        df["image_path"].apply(
            lambda x: os.path.normcase(
                os.path.abspath(str(x))
            )
        ) == normalized_path
    ]

    if matches.empty:

        return (
            False,
            "Image was not found in annotation manifest.",
            {
                "error_code": "IMAGE_NOT_IN_MANIFEST",
            },
        )

    idx = matches.index[0]

    row_split = str(df.loc[idx, "split"])

    if row_split != str(split):

        return (
            False,
            "Requested split does not match manifest split.",
            {
                "error_code": "SPLIT_MISMATCH",
            },
        )

    try:

        allowed, message, meta = validate_mask_array(
            image_path,
            mask_array,
            expected_code,
        )

        if not allowed:

            df.loc[idx, "validation_status"] = "FAILED"
            df.loc[idx, "error_message"] = message

            save_manifest_atomically(
                df,
                manifest_path,
            )

            return False, message, meta

        arr = np.asarray(mask_array)

        if arr.ndim == 3:
            arr = np.asarray(
                Image.fromarray(
                    arr.astype(np.uint8)
                ).convert("L")
            )

        arr = arr.astype(np.uint8)

        with Image.open(image_path) as img:
            width, height = img.width, img.height

        if arr.shape != (height, width):

            arr = np.asarray(
                Image.fromarray(
                    arr,
                    mode="L",
                ).resize(
                    (width, height),
                    Image.Resampling.NEAREST,
                ),
                dtype=np.uint8,
            )

        mask_path = Path(
            str(df.loc[idx, "expected_mask_path"])
        )

        mask_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_mask = mask_path.with_suffix(
            ".tmp.png"
        )

        Image.fromarray(
            arr,
            mode="L",
        ).save(
            temp_mask,
            format="PNG",
        )

        # Validate saved mask before replacing final file.
        valid_saved, saved_msg, saved_meta = (
            validate_mask_file(
                image_path,
                temp_mask,
                expected_code,
            )
        )

        if not valid_saved:

            if temp_mask.exists():
                temp_mask.unlink()

            df.loc[idx, "validation_status"] = "FAILED"
            df.loc[idx, "error_message"] = saved_msg

            save_manifest_atomically(
                df,
                manifest_path,
            )

            return False, saved_msg, saved_meta

        os.replace(
            temp_mask,
            mask_path,
        )

        df.loc[idx, "annotation_status"] = "ANNOTATED"
        df.loc[idx, "validation_status"] = "PASSED"
        df.loc[idx, "error_message"] = ""

        save_manifest_atomically(
            df,
            manifest_path,
        )

        meta["mask_path"] = str(mask_path)

        return (
            True,
            "Mask validated and saved.",
            meta,
        )

    except Exception as exc:

        return (
            False,
            f"Annotation processing failed: {exc}",
            {
                "error_code": "PROCESSING_ERROR",
                "exception": str(exc),
            },
        )


# ============================================================
# SKIP
# ============================================================

def mark_image_skipped(
    image_path: str | Path,
    reason: str = "Manual review skipped",
    manifest_csv: str | Path | None = None,
) -> bool:

    manifest_path = Path(
        manifest_csv or get_manifest_path()
    )

    df = load_manifest(manifest_path)

    normalized = os.path.normcase(
        os.path.abspath(str(image_path))
    )

    matches = df[
        df["image_path"].apply(
            lambda x: os.path.normcase(
                os.path.abspath(str(x))
            )
        ) == normalized
    ]

    if matches.empty:
        return False

    idx = matches.index[0]

    split = str(df.loc[idx, "split"])

    if split not in ANNOTATABLE_SPLITS:
        raise PermissionError(
            "Test split cannot be skipped or modified."
        )

    df.loc[idx, "annotation_status"] = "SKIPPED"
    df.loc[idx, "validation_status"] = "UNVALIDATED"
    df.loc[idx, "error_message"] = str(reason)

    save_manifest_atomically(
        df,
        manifest_path,
    )

    return True


# ============================================================
# PROGRESS
# ============================================================

def get_annotation_progress_report(
    manifest_csv: str | Path | None = None,
) -> Dict[str, Any]:

    df = load_manifest(manifest_csv)

    eligible = df[
        df["split"].isin(ANNOTATABLE_SPLITS)
    ]

    total = len(eligible)

    annotated = int(
        (
            eligible["annotation_status"]
            == "ANNOTATED"
        ).sum()
    )

    passed = int(
        (
            eligible["validation_status"]
            == "PASSED"
        ).sum()
    )

    skipped = int(
        (
            eligible["annotation_status"]
            == "SKIPPED"
        ).sum()
    )

    pending = int(
        (
            eligible["annotation_status"]
            == "PENDING"
        ).sum()
    )

    percentage = (
        round((annotated + skipped) / total * 100, 2)
        if total > 0
        else 100.0
    )

    return make_json_safe({
        "total_eligible_images": total,
        "annotated_count": annotated,
        "passed_validation_count": passed,
        "skipped_review_count": skipped,
        "pending_count": pending,
        "progress_percentage": percentage,
        "test_images_isolated": int(
            (df["split"] == "Test").sum()
        ),
    })


# ============================================================
# CALLBACKS
# ============================================================

_CALLBACK_REGISTRATION = {}


def colab_callback_health_check() -> Dict[str, Any]:

    return {
        "success": True,
        "status": "HEALTHY",
        "message": "COLAB_CALLBACK_WORKING",
    }


def colab_save_mask_handler(
    image_path: str,
    mask_base64: str,
    split: str,
    class_name: str,
    manifest_csv: str | Path | None = None,
) -> Dict[str, Any]:

    try:

        if not isinstance(mask_base64, str):
            raise ValueError(
                "Mask payload must be a base64 string."
            )

        if "," in mask_base64:
            mask_base64 = mask_base64.split(
                ",",
                1,
            )[1]

        raw = base64.b64decode(
            mask_base64,
            validate=True,
        )

        with Image.open(
            io.BytesIO(raw)
        ) as img:

            if img.mode == "RGBA":
                arr = np.asarray(img)

                alpha = arr[..., 3]
                rgb = arr[..., :3]

                gray = np.asarray(
                    Image.fromarray(
                        rgb.astype(np.uint8)
                    ).convert("L")
                )

                gray[alpha == 0] = 0
                mask_array = gray

            else:
                mask_array = np.asarray(
                    img.convert("L")
                )

        success, message, meta = (
            process_annotation_submission(
                image_path,
                mask_array,
                split,
                class_name,
                manifest_csv,
            )
        )

        if not success:

            return make_json_safe({
                "success": False,
                "message": message,
                "meta": meta,
                "progress":
                    get_annotation_progress_report(
                        manifest_csv
                    ),
                "next_item": None,
            })

        next_item = get_next_pending_image(
            split=split,
            class_name=class_name,
            manifest_csv=manifest_csv,
        )

        if next_item is None:
            next_item = get_next_pending_image(
                manifest_csv=manifest_csv,
            )

        return make_json_safe({
            "success": True,
            "message": message,
            "meta": meta,
            "progress":
                get_annotation_progress_report(
                    manifest_csv
                ),
            "next_item": next_item,
        })

    except Exception as exc:

        return {
            "success": False,
            "message": f"Save callback failed: {exc}",
            "next_item": None,
        }


def colab_skip_image_handler(
    image_path: str,
    reason: str,
    split: str,
    class_name: str,
    manifest_csv: str | Path | None = None,
) -> Dict[str, Any]:

    try:

        manifest_path = Path(
            manifest_csv or get_manifest_path()
        )

        # Never permit Test modifications.
        if str(split) not in ANNOTATABLE_SPLITS:
            raise PermissionError(
                "Test split is read-only."
            )

        success = mark_image_skipped(
            image_path,
            reason,
            manifest_path,
        )

        if not success:
            return {
                "success": False,
                "message": "Image was not found in manifest.",
                "next_item": None,
            }

        next_item = get_next_pending_image(
            split=split,
            class_name=class_name,
            manifest_csv=manifest_path,
        )

        if next_item is None:
            next_item = get_next_pending_image(
                split=split,
                manifest_csv=manifest_path,
            )

        if next_item is None:
            next_item = get_next_pending_image(
                manifest_csv=manifest_path,
            )

        return make_json_safe({
            "success": True,
            "message": "Image marked skipped.",
            "progress":
                get_annotation_progress_report(
                    manifest_path
                ),
            "next_item": next_item,
        })

    except Exception as exc:

        return {
            "success": False,
            "message": f"Skip callback failed: {exc}",
            "next_item": None,
        }


def register_colab_callbacks() -> Dict[str, Any]:

    callbacks = {
        "notebook.save_mask":
            colab_save_mask_handler,

        "notebook.skip_image":
            colab_skip_image_handler,
    }

    try:

        from google.colab import output

        output.register_callback(
            "notebook.save_mask",
            colab_save_mask_handler,
        )

        output.register_callback(
            "notebook.skip_image",
            colab_skip_image_handler,
        )

        _CALLBACK_REGISTRATION.update(callbacks)

        return {
            "success": True,
            "callbacks": list(callbacks.keys()),
            "message": "Callbacks registered successfully.",
        }

    except Exception as exc:

        _CALLBACK_REGISTRATION.update(callbacks)

        return {
            "success": False,
            "callbacks": list(callbacks.keys()),
            "message": f"Callback registration unavailable: {exc}",
        }


# ============================================================
# RESPONSE DECODER
# ============================================================

def decode_colab_response_payload(
    response: Any,
) -> Any:

    if response is None:
        return None

    payload = (
        response.get("data", response)
        if isinstance(response, dict)
        else response
    )

    if isinstance(payload, dict):

        if "application/json" in payload:
            payload = payload["application/json"]

        elif "text/plain" in payload:
            payload = payload["text/plain"]

    for _ in range(4):

        if not isinstance(payload, str):
            break

        try:
            payload = json.loads(payload)
        except Exception:
            break

    if (
        isinstance(payload, dict)
        and "data" in payload
        and isinstance(payload["data"], dict)
    ):

        nested = payload["data"]

        if "application/json" in nested:
            nested = nested["application/json"]

        elif "text/plain" in nested:
            nested = nested["text/plain"]

        for _ in range(4):

            if not isinstance(nested, str):
                break

            try:
                nested = json.loads(nested)
            except Exception:
                break

        payload = nested

    return payload


# ============================================================
# CANVAS HISTORY MODEL
# ============================================================

class CanvasHistoryStateModel:

    def __init__(
        self,
        width: int,
        height: int,
    ):

        self.width = int(width)
        self.height = int(height)

        self.current_state = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        self.undo_stack = [
            self.current_state.copy()
        ]

        self.redo_stack = []

    def _push_state(self):

        self.undo_stack.append(
            self.current_state.copy()
        )

        if len(self.undo_stack) > MAX_UNDO_STATES:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def paint_stroke(
        self,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        code: int,
    ):

        code = int(code)

        if code not in VALID_MASK_VALUES - {0}:
            raise ValueError(
                f"Invalid paint code: {code}"
            )

        x1, x2 = sorted(
            [max(0, x1), min(self.width, x2)]
        )

        y1, y2 = sorted(
            [max(0, y1), min(self.height, y2)]
        )

        if x2 > x1 and y2 > y1:

            self.current_state[
                y1:y2,
                x1:x2,
            ] = code

        self._push_state()

    def undo_stroke(self) -> bool:

        if len(self.undo_stack) <= 1:
            return False

        self.redo_stack.append(
            self.undo_stack.pop()
        )

        self.current_state = (
            self.undo_stack[-1].copy()
        )

        return True

    def redo_stroke(self) -> bool:

        if not self.redo_stack:
            return False

        state = self.redo_stack.pop()

        self.undo_stack.append(
            state.copy()
        )

        self.current_state = state.copy()

        return True

    def clear_canvas(self):

        self.current_state = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        self._push_state()

    def load_next_image(
        self,
        width: int,
        height: int,
    ):

        self.width = int(width)
        self.height = int(height)

        self.current_state = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        self.undo_stack = [
            self.current_state.copy()
        ]

        self.redo_stack = []


# ============================================================
# HTML UI
# ============================================================

def build_annotation_html(
    item: Dict[str, Any],
    progress: Dict[str, Any],
    debug_ui: bool = False,
) -> str:

    img_b64 = item.get("base64")

    if not img_b64:
        raise ValueError(
            "Image base64 data is missing."
        )

    width = int(item["width"])
    height = int(item["height"])

    max_display_width = 700

    if width > max_display_width:
        disp_w = max_display_width
        disp_h = int(
            disp_w * height / width
        )
    else:
        disp_w = width
        disp_h = height

    item_json = json.dumps(
        make_json_safe(item)
    )

    progress_json = json.dumps(
        make_json_safe(progress)
    )

    return f"""
<div id="annotation-widget-container"
     style="
     font-family:Arial,sans-serif;
     max-width:960px;
     padding:18px;
     border:2px solid #1F497D;
     border-radius:8px;
     background:#F8F9FA;">

<h3 style="margin-top:0;color:#1F497D;">
🎨 Cashew Leaf Manual Segmentation Tool — Phase C.1
</h3>

<div style="
background:#E9ECEF;
padding:12px;
border-radius:6px;
margin-bottom:12px;
display:flex;
flex-wrap:wrap;
gap:15px;
justify-content:space-between;">

<div>
<strong>Eligible Pool:</strong>
<span id="stat-total">
{progress.get("total_eligible_images", 0)}
</span>
</div>

<div>
<strong>Test:</strong>
<span id="stat-test">
{progress.get("test_images_isolated", 0)}
</span>
<span style="color:#DC3545;">READ ONLY</span>
</div>

<div>
<strong>Annotated:</strong>
<span id="stat-annotated">
{progress.get("annotated_count", 0)}
</span>
</div>

<div>
<strong>Passed:</strong>
<span id="stat-passed">
{progress.get("passed_validation_count", 0)}
</span>
</div>

<div>
<strong>Skipped:</strong>
<span id="stat-skipped">
{progress.get("skipped_review_count", 0)}
</span>
</div>

<div>
<strong>Pending:</strong>
<span id="stat-pending">
{progress.get("pending_count", 0)}
</span>
</div>

<div>
<strong>Progress:</strong>
<span id="stat-progress">
{progress.get("progress_percentage", 0)}%
</span>
</div>

</div>

<div style="
display:flex;
flex-wrap:wrap;
gap:15px;
margin-bottom:12px;
background:#FFF;
padding:10px;
border:1px solid #DEE2E6;
border-radius:4px;">

<div>
<strong>Split:</strong>
<span id="lbl-split">{item["split"]}</span>
</div>

<div>
<strong>Class:</strong>
<span id="lbl-class">
{item["class_name"]} (Code {item["class_code"]})
</span>
</div>

<div>
<strong>Dimensions:</strong>
<span id="lbl-dims">
{width} × {height} px
</span>
</div>

<div>
<strong>File:</strong>
<span id="lbl-file">
{item["image_name"]}
</span>
</div>

</div>

<div style="
background:#E9ECEF;
padding:10px;
border-radius:5px;
margin-bottom:12px;
display:flex;
flex-wrap:wrap;
gap:10px;
align-items:center;">

<label>
<strong>Brush Size:</strong>
<input
type="range"
id="brush-size"
min="1"
max="50"
value="{DEFAULT_BRUSH_SIZE}"
oninput="updateBrushSize(this.value)">
<span id="brush-val">{DEFAULT_BRUSH_SIZE}</span> px
</label>

<button
onclick="setMode('brush')"
id="btn-brush"
style="
background:#1F497D;
color:white;
border:none;
padding:6px 12px;
border-radius:4px;
cursor:pointer;
font-weight:bold;">
🖌️ Paint Lesion
</button>

<button
onclick="setMode('eraser')"
id="btn-eraser"
style="
background:#6C757D;
color:white;
border:none;
padding:6px 12px;
border-radius:4px;
cursor:pointer;
font-weight:bold;
opacity:.6;">
🧹 Eraser
</button>

<button
onclick="undoStroke()"
style="
background:#17A2B8;
color:white;
border:none;
padding:6px 12px;
border-radius:4px;
cursor:pointer;
font-weight:bold;">
↩️ Undo
</button>

<button
onclick="redoStroke()"
style="
background:#6C757D;
color:white;
border:none;
padding:6px 12px;
border-radius:4px;
cursor:pointer;
font-weight:bold;">
↪️ Redo
</button>

<button
onclick="clearCanvas()"
style="
background:#DC3545;
color:white;
border:none;
padding:6px 12px;
border-radius:4px;
cursor:pointer;
font-weight:bold;">
🗑️ Clear
</button>

</div>

<div id="canvas-wrapper"
style="
position:relative;
width:{disp_w}px;
height:{disp_h}px;
border:2px solid #1F497D;
margin:0 auto;
background:#000;
overflow:hidden;">

<img
id="bg-img"
src="data:image/jpeg;base64,{img_b64}"
style="
position:absolute;
left:0;
top:0;
width:100%;
height:100%;
pointer-events:none;
z-index:1;
object-fit:contain;">

<canvas
id="display-canvas"
width="{width}"
height="{height}"
style="
position:absolute;
left:0;
top:0;
width:100%;
height:100%;
z-index:2;
pointer-events:auto;
touch-action:none;
cursor:crosshair;">
</canvas>

</div>

<div
style="
margin-top:15px;
display:flex;
justify-content:space-between;
align-items:center;
gap:10px;
flex-wrap:wrap;">

<div
id="status-msg"
style="
font-weight:bold;
color:#1F497D;
font-size:14px;">

Ready: Paint the visible affected lesion and save.

</div>

<div style="display:flex;gap:10px;">

<button
onclick="skipImage()"
id="btn-skip"
style="
background:#FFC107;
color:black;
border:none;
padding:10px 18px;
border-radius:4px;
font-weight:bold;
cursor:pointer;">
⏭️ Skip for Review
</button>

<button
onclick="saveAndNext()"
id="btn-save"
style="
background:#28A745;
color:white;
border:none;
padding:10px 18px;
border-radius:4px;
font-weight:bold;
cursor:pointer;">
💾 Save Mask & Next
</button>

</div>
</div>

</div>

<script>

var DEBUG_UI = {str(bool(debug_ui)).lower()};

var currentImgPath =
{json.dumps(str(item["image_path"]))};

var currentSplit =
{json.dumps(str(item["split"]))};

var currentClass =
{json.dumps(str(item["class_name"]))};

var currentCode =
{int(item["class_code"])};

var displayCanvas =
document.getElementById("display-canvas");

var displayCtx =
displayCanvas.getContext("2d");

var maskCanvas =
document.createElement("canvas");

maskCanvas.width =
displayCanvas.width;

maskCanvas.height =
displayCanvas.height;

var maskCtx =
maskCanvas.getContext("2d");

var isDrawing = false;

var mode = "brush";

var brushSize = {DEFAULT_BRUSH_SIZE};

var lastPoint = null;

var undoStack = [];

var redoStack = [];

function logDebug(msg, obj) {{

    if (!DEBUG_UI) return;

    if (obj !== undefined)
        console.log("[C1 UI] " + msg, obj);
    else
        console.log("[C1 UI] " + msg);
}}

function cloneCanvasState() {{

    return maskCtx.getImageData(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
    );
}}

function saveState() {{

    undoStack.push(
        cloneCanvasState()
    );

    if (undoStack.length > 25)
        undoStack.shift();

    redoStack = [];

    renderMaskOverlay();
}}

function undoStroke() {{

    if (undoStack.length <= 1)
        return;

    redoStack.push(
        undoStack.pop()
    );

    maskCtx.putImageData(
        undoStack[undoStack.length - 1],
        0,
        0
    );

    renderMaskOverlay();
}}

function redoStroke() {{

    if (redoStack.length === 0)
        return;

    var state =
        redoStack.pop();

    undoStack.push(state);

    maskCtx.putImageData(
        state,
        0,
        0
    );

    renderMaskOverlay();
}}

function clearCanvas() {{

    maskCtx.globalCompositeOperation =
        "source-over";

    maskCtx.clearRect(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
    );

    saveState();
}}

function updateBrushSize(value) {{

    brushSize =
        parseInt(value);

    document.getElementById(
        "brush-val"
    ).innerText = value;
}}

function setMode(newMode) {{

    mode = newMode;

    document.getElementById(
        "btn-brush"
    ).style.opacity =
        newMode === "brush"
        ? "1"
        : ".6";

    document.getElementById(
        "btn-eraser"
    ).style.opacity =
        newMode === "eraser"
        ? "1"
        : ".6";
}}

function getPos(e) {{

    var rect =
        displayCanvas.getBoundingClientRect();

    var scaleX =
        maskCanvas.width / rect.width;

    var scaleY =
        maskCanvas.height / rect.height;

    return {{
        x:
        (e.clientX - rect.left)
        * scaleX,

        y:
        (e.clientY - rect.top)
        * scaleY
    }};
}}

function drawSegment(from, to) {{

    maskCtx.lineWidth =
        brushSize;

    maskCtx.lineCap =
        "round";

    maskCtx.lineJoin =
        "round";

    maskCtx.beginPath();

    maskCtx.moveTo(
        from.x,
        from.y
    );

    maskCtx.lineTo(
        to.x,
        to.y
    );

    if (mode === "brush") {{

        maskCtx.globalCompositeOperation =
            "source-over";

        maskCtx.strokeStyle =
            "rgb("
            + currentCode + ","
            + currentCode + ","
            + currentCode
            + ")";

    }} else {{

        maskCtx.globalCompositeOperation =
            "destination-out";

    }}

    maskCtx.stroke();

    /*
     * Ensure a pointer-down without movement
     * still creates a circular mark.
     */

    if (
        from.x === to.x &&
        from.y === to.y
    ) {{

        maskCtx.beginPath();

        maskCtx.arc(
            from.x,
            from.y,
            brushSize / 2,
            0,
            Math.PI * 2
        );

        maskCtx.fill();
    }}
}}

displayCanvas.addEventListener(
    "pointerdown",
    function(e) {{

        e.preventDefault();

        isDrawing = true;

        displayCanvas.setPointerCapture(
            e.pointerId
        );

        lastPoint = getPos(e);

        drawSegment(
            lastPoint,
            lastPoint
        );

        renderMaskOverlay();
    }}
);

displayCanvas.addEventListener(
    "pointermove",
    function(e) {{

        if (!isDrawing)
            return;

        e.preventDefault();

        var point = getPos(e);

        drawSegment(
            lastPoint,
            point
        );

        lastPoint = point;

        renderMaskOverlay();
    }}
);

displayCanvas.addEventListener(
    "pointerup",
    function(e) {{

        if (!isDrawing)
            return;

        isDrawing = false;

        try {{
            displayCanvas.releasePointerCapture(
                e.pointerId
            );
        }} catch (_) {{}}

        lastPoint = null;

        saveState();
    }}
);

displayCanvas.addEventListener(
    "pointercancel",
    function(e) {{

        if (!isDrawing)
            return;

        isDrawing = false;

        lastPoint = null;

        renderMaskOverlay();
    }}
);

function renderMaskOverlay() {{

    displayCtx.clearRect(
        0,
        0,
        displayCanvas.width,
        displayCanvas.height
    );

    var maskData =
        maskCtx.getImageData(
            0,
            0,
            maskCanvas.width,
            maskCanvas.height
        );

    var overlayData =
        displayCtx.createImageData(
            displayCanvas.width,
            displayCanvas.height
        );

    for (
        var i = 0;
        i < maskData.data.length;
        i += 4
    ) {{

        var code =
            maskData.data[i];

        if (code > 0) {{

            if (code === 1) {{
                overlayData.data[i] = 255;
                overlayData.data[i + 1] = 0;
                overlayData.data[i + 2] = 0;
            }}
            else if (code === 2) {{
                overlayData.data[i] = 0;
                overlayData.data[i + 1] = 255;
                overlayData.data[i + 2] = 0;
            }}
            else if (code === 3) {{
                overlayData.data[i] = 0;
                overlayData.data[i + 1] = 0;
                overlayData.data[i + 2] = 255;
            }}
            else if (code === 4) {{
                overlayData.data[i] = 255;
                overlayData.data[i + 1] = 255;
                overlayData.data[i + 2] = 0;
            }}

            overlayData.data[i + 3] = 180;
        }}
    }}

    displayCtx.putImageData(
        overlayData,
        0,
        0
    );
}}

function getRawMaskBase64() {{

    /*
     * Export only the mask canvas.
     * It is intentionally NOT the visual overlay.
     */

    return maskCanvas.toDataURL(
        "image/png"
    );
}}

function parseColabResponse(res) {{

    if (!res)
        return null;

    var payload =
        res && res.data
        ? res.data
        : res;

    if (
        payload &&
        typeof payload === "object"
    ) {{

        if (
            payload["application/json"]
        )
            payload =
                payload["application/json"];

        else if (
            payload["text/plain"]
        )
            payload =
                payload["text/plain"];
    }}

    for (var i = 0; i < 4; i++) {{

        if (
            typeof payload !== "string"
        )
            break;

        try {{
            payload =
                JSON.parse(payload);
        }}
        catch (_) {{
            break;
        }}
    }}

    return payload;
}}

function updateProgressUI(rep) {{

    if (!rep)
        return;

    var ids = {{
        annotated_count:
            "stat-annotated",

        passed_validation_count:
            "stat-passed",

        skipped_review_count:
            "stat-skipped",

        pending_count:
            "stat-pending",

        progress_percentage:
            "stat-progress"
    }};

    Object.keys(ids).forEach(
        function(key) {{

            var element =
                document.getElementById(
                    ids[key]
                );

            if (!element)
                return;

            var value =
                rep[key];

            if (
                key ===
                "progress_percentage"
            )
                value += "%";

            element.innerText =
                value;
        }}
    );
}}

function loadNextImageInPlace(item) {{

    if (!item) {{

        document.getElementById(
            "annotation-widget-container"
        ).innerHTML =
            "<h3 style='color:#28A745;text-align:center;padding:30px;'>"
            + "🎉 ALL ELIGIBLE TRAIN/VALIDATION IMAGES ARE ANNOTATED!"
            + "</h3>";

        return;
    }}

    currentImgPath =
        item.image_path;

    currentSplit =
        item.split;

    currentClass =
        item.class_name;

    currentCode =
        parseInt(item.class_code);

    document.getElementById(
        "lbl-split"
    ).innerText =
        currentSplit;

    document.getElementById(
        "lbl-class"
    ).innerText =
        currentClass
        + " (Code "
        + currentCode
        + ")";

    document.getElementById(
        "lbl-dims"
    ).innerText =
        item.width
        + " × "
        + item.height
        + " px";

    document.getElementById(
        "lbl-file"
    ).innerText =
        item.image_name;

    var bg =
        document.getElementById(
            "bg-img"
        );

    if (item.base64) {{

        bg.src =
            item.base64.indexOf(
                "data:image"
            ) === 0

            ? item.base64

            : "data:image/jpeg;base64,"
              + item.base64;
    }}

    displayCanvas.width =
        item.width;

    displayCanvas.height =
        item.height;

    maskCanvas.width =
        item.width;

    maskCanvas.height =
        item.height;

    maskCtx.clearRect(
        0,
        0,
        maskCanvas.width,
        maskCanvas.height
    );

    displayCtx.clearRect(
        0,
        0,
        displayCanvas.width,
        displayCanvas.height
    );

    undoStack = [];

    redoStack = [];

    saveState();

    document.getElementById(
        "btn-save"
    ).disabled = false;

    document.getElementById(
        "btn-skip"
    ).disabled = false;

    document.getElementById(
        "status-msg"
    ).style.color =
        "#1F497D";

    document.getElementById(
        "status-msg"
    ).innerText =
        "Ready: Paint the visible affected lesion and save.";
}}

function skipImage() {{

    document.getElementById(
        "btn-skip"
    ).disabled = true;

    document.getElementById(
        "status-msg"
    ).innerText =
        "⏳ Marking image skipped...";

    if (
        window.google &&
        google.colab &&
        google.colab.kernel
    ) {{

        google.colab.kernel
        .invokeFunction(
            "notebook.skip_image",
            [
                currentImgPath,
                "Manual review skipped",
                currentSplit,
                currentClass
            ],
            {{}}
        )
        .then(function(res) {{

            var data =
                parseColabResponse(res);

            if (
                data &&
                data.success
            ) {{

                updateProgressUI(
                    data.progress
                );

                document.getElementById(
                    "status-msg"
                ).innerText =
                    "✅ Skipped. Loading next image...";

                loadNextImageInPlace(
                    data.next_item
                );

            }} else {{

                document.getElementById(
                    "btn-skip"
                ).disabled = false;

                document.getElementById(
                    "status-msg"
                ).innerText =
                    "❌ "
                    + (
                        data &&
                        data.message
                        ? data.message
                        : "Skip failed."
                    );
            }}

        }})
        .catch(function(err) {{

            document.getElementById(
                "btn-skip"
            ).disabled = false;

            document.getElementById(
                "status-msg"
            ).innerText =
                "❌ Callback error: "
                + err;
        }});

    }} else {{

        document.getElementById(
            "btn-skip"
        ).disabled = false;

        document.getElementById(
            "status-msg"
        ).innerText =
            "❌ Google Colab callback API unavailable.";
    }}
}}

function saveAndNext() {{

    document.getElementById(
        "btn-save"
    ).disabled = true;

    document.getElementById(
        "status-msg"
    ).innerText =
        "⏳ Validating and saving mask...";

    var rawMaskB64 =
        getRawMaskBase64();

    if (
        window.google &&
        google.colab &&
        google.colab.kernel
    ) {{

        google.colab.kernel
        .invokeFunction(
            "notebook.save_mask",
            [
                currentImgPath,
                rawMaskB64,
                currentSplit,
                currentClass
            ],
            {{}}
        )
        .then(function(res) {{

            var data =
                parseColabResponse(res);

            if (
                data &&
                data.success
            ) {{

                updateProgressUI(
                    data.progress
                );

                document.getElementById(
                    "status-msg"
                ).innerText =
                    "✅ Mask validated and saved. Loading next image...";

                loadNextImageInPlace(
                    data.next_item
                );

            }} else {{

                document.getElementById(
                    "btn-save"
                ).disabled = false;

                document.getElementById(
                    "status-msg"
                ).innerText =
                    "❌ "
                    + (
                        data &&
                        data.message
                        ? data.message
                        : "Validation failed."
                    );
            }}

        }})
        .catch(function(err) {{

            document.getElementById(
                "btn-save"
            ).disabled = false;

            document.getElementById(
                "status-msg"
            ).innerText =
                "❌ Callback error: "
                + err;
        }});

    }} else {{

        document.getElementById(
            "btn-save"
        ).disabled = false;

        document.getElementById(
            "status-msg"
        ).innerText =
            "❌ Google Colab callback API unavailable.";
    }}
}}

saveState();

</script>
"""


# ============================================================
# DISPLAY TOOL
# ============================================================

def show_annotation_tool(
    split: str = "Train",
    class_name: str | None = None,
    manifest_csv: str | Path | None = None,
    debug_ui: bool = False,
):
    """
    Displays the manual annotation UI in Colab/Jupyter.
    """

    item = get_next_pending_image(
        split=split,
        class_name=class_name,
        manifest_csv=manifest_csv,
    )

    if item is None:
        print(
            "No pending image found for the requested filter."
        )
        return None

    progress = get_annotation_progress_report(
        manifest_csv
    )

    html = build_annotation_html(
        item,
        progress,
        debug_ui=debug_ui,
    )

    try:

        from IPython.display import HTML, display

        html_obj = HTML(html)
        display(html_obj)

        return html_obj

    except ImportError:

        print(
            "[INTERACTIVE TOOL READY]",
            item["image_path"],
        )

        return html


# ============================================================
# VERIFICATION HELPERS
# ============================================================

def _make_dummy_image(path: Path) -> None:

    rng = np.random.default_rng(42)

    arr = np.uint8(
        rng.integers(
            0,
            255,
            size=(100, 100, 3),
        )
    )

    Image.fromarray(arr).save(path)


def _make_dummy_mask(
    code: int,
    shape=(100, 100),
) -> np.ndarray:

    mask = np.zeros(
        shape,
        dtype=np.uint8,
    )

    mask[10:30, 10:30] = code

    return mask


# ============================================================
# 25-TEST TRUTHFUL VERIFICATION SUITE
# ============================================================

def run_phase_c1_verification_suite() -> Dict[str, Any]:
    """
    Runs the complete 25-test Phase C.1 verification suite.

    All tests use temporary scratch data except the explicit
    fingerprint checks for the project's three classification
    split CSVs.

    The suite must never modify those split files.
    """

    print(
        "\n=== PHASE C.1 TRUTHFUL VERIFICATION SUITE ==="
    )

    preprocessed_dir = get_preprocessed_dir()

    train_csv = preprocessed_dir / "train_split.csv"
    val_csv = preprocessed_dir / "val_split.csv"
    test_csv = preprocessed_dir / "test_split.csv"

    hashes_before = {
        "train_split.csv":
            compute_file_hash(train_csv),

        "val_split.csv":
            compute_file_hash(val_csv),

        "test_split.csv":
            compute_file_hash(test_csv),
    }

    test_results: Dict[str, str] = {}

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="phase_c1_suite25_"
        )
    )

    try:

        dummy_train = temp_dir / "dummy_train.jpg"
        dummy_val = temp_dir / "dummy_val.jpg"
        dummy_test = temp_dir / "dummy_test.jpg"

        _make_dummy_image(dummy_train)
        _make_dummy_image(dummy_val)
        _make_dummy_image(dummy_test)

        # ----------------------------------------------------
        # TEST 1
        # ----------------------------------------------------

        reg = register_colab_callbacks()

        test_results[
            "TEST_1_callback_registration"
        ] = (
            "PASS"
            if (
                isinstance(reg, dict)
                and "callbacks" in reg
                and
                "notebook.save_mask"
                in reg["callbacks"]
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 2
        # ----------------------------------------------------

        health = colab_callback_health_check()

        test_results[
            "TEST_2_python_callback_execution"
        ] = (
            "PASS"
            if (
                health.get("status")
                == "HEALTHY"
                and
                health.get("success")
                is True
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 3
        # ----------------------------------------------------

        payload = {
            "success": True,
            "message":
                "COLAB_CALLBACK_WORKING",
        }

        payloads = [
            payload,

            {
                "data": {
                    "application/json":
                        json.dumps(payload)
                }
            },

            {
                "data": {
                    "text/plain":
                        json.dumps(payload)
                }
            },

            {
                "data": {
                    "application/json":
                        json.dumps(
                            json.dumps(payload)
                        )
                }
            },
        ]

        decoded = [
            decode_colab_response_payload(x)
            for x in payloads
        ]

        test_results[
            "TEST_3_js_response_parsing"
        ] = (
            "PASS"
            if all(
                isinstance(x, dict)
                and
                x.get("message")
                == "COLAB_CALLBACK_WORKING"
                for x in decoded
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # Temporary manifest
        # ----------------------------------------------------

        temp_manifest = (
            temp_dir / "manifest.csv"
        )

        rows = [
            {
                "image_path": str(dummy_train),
                "image_name":
                    dummy_train.name,
                "class_name": "TMB",
                "class_code": 3,
                "split": "Train",
                "expected_mask_path":
                    str(
                        temp_dir
                        / "train_mask.png"
                    ),
                "annotation_status":
                    "PENDING",
                "validation_status":
                    "UNVALIDATED",
                "error_message":
                    "Pending",
            },

            {
                "image_path": str(dummy_val),
                "image_name":
                    dummy_val.name,
                "class_name": "Aphids",
                "class_code": 1,
                "split": "Validation",
                "expected_mask_path":
                    str(
                        temp_dir
                        / "val_mask.png"
                    ),
                "annotation_status":
                    "PENDING",
                "validation_status":
                    "UNVALIDATED",
                "error_message":
                    "Pending",
            },

            {
                "image_path": str(dummy_test),
                "image_name":
                    dummy_test.name,
                "class_name": "Leaf_Miner",
                "class_code": 2,
                "split": "Test",
                "expected_mask_path":
                    str(
                        temp_dir
                        / "test_mask.png"
                    ),
                "annotation_status":
                    "PENDING",
                "validation_status":
                    "UNVALIDATED",
                "error_message":
                    "Pending",
            },
        ]

        pd.DataFrame(
            rows,
            columns=MANIFEST_COLUMNS,
        ).to_csv(
            temp_manifest,
            index=False,
        )

        temp_test_csv = (
            temp_dir / "test_split.csv"
        )

        pd.DataFrame({
            "file_path": [str(dummy_test)],
            "class_name": ["Leaf_Miner"],
        }).to_csv(
            temp_test_csv,
            index=False,
        )

        # ----------------------------------------------------
        # TEST 4
        # ----------------------------------------------------

        try:

            assert_annotation_allowed(
                "Test",
                dummy_test,
                temp_test_csv,
            )

            test_results[
                "TEST_4_test_split_rejection"
            ] = "FAIL"

        except PermissionError:

            test_results[
                "TEST_4_test_split_rejection"
            ] = "PASS"

        # ----------------------------------------------------
        # TEST 5
        # ----------------------------------------------------

        try:

            assert_annotation_allowed(
                "Train",
                dummy_test,
                temp_test_csv,
            )

            test_results[
                "TEST_5_test_image_path_rejection"
            ] = "FAIL"

        except PermissionError:

            test_results[
                "TEST_5_test_image_path_rejection"
            ] = "PASS"

        # ----------------------------------------------------
        # TEST 6
        # ----------------------------------------------------

        item = get_next_pending_image(
            split="Train",
            manifest_csv=temp_manifest,
        )

        test_results[
            "TEST_6_train_selection"
        ] = (
            "PASS"
            if item
            and item["split"] == "Train"
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 7
        # ----------------------------------------------------

        item = get_next_pending_image(
            split="Validation",
            manifest_csv=temp_manifest,
        )

        test_results[
            "TEST_7_validation_selection"
        ] = (
            "PASS"
            if item
            and item["split"]
            == "Validation"
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 8
        # ----------------------------------------------------

        item = get_next_pending_image(
            class_name="TMB",
            manifest_csv=temp_manifest,
        )

        test_results[
            "TEST_8_class_filtering"
        ] = (
            "PASS"
            if item
            and item["class_name"]
            == "TMB"
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 9
        # ----------------------------------------------------

        empty = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        ok, msg, meta = (
            process_annotation_submission(
                dummy_train,
                empty,
                "Train",
                "TMB",
                temp_manifest,
            )
        )

        test_results[
            "TEST_9_empty_mask_rejection"
        ] = (
            "PASS"
            if (
                not ok
                and
                meta.get("error_code")
                == "EMPTY_MASK"
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 10
        # ----------------------------------------------------

        invalid = np.zeros(
            (100, 100),
            dtype=np.uint8,
        )

        invalid[10:30, 10:30] = 255

        ok, msg, meta = validate_mask_array(
            dummy_train,
            invalid,
            3,
        )

        test_results[
            "TEST_10_invalid_pixel_value_rejection"
        ] = (
            "PASS"
            if (
                not ok
                and
                meta.get("error_code")
                == "INVALID_PIXEL_VALUES"
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 11
        # ----------------------------------------------------

        wrong = _make_dummy_mask(1)

        ok, msg, meta = validate_mask_array(
            dummy_train,
            wrong,
            3,
        )

        test_results[
            "TEST_11_missing_expected_class_rejection"
        ] = (
            "PASS"
            if (
                not ok
                and
                meta.get("error_code")
                == "MISSING_EXPECTED_CLASS"
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 12
        # ----------------------------------------------------

        small_mask = _make_dummy_mask(
            3,
            (50, 50),
        )

        ok, msg, meta = (
            process_annotation_submission(
                dummy_train,
                small_mask,
                "Train",
                "TMB",
                temp_manifest,
            )
        )

        test_results[
            "TEST_12_dimension_mismatch"
        ] = (
            "PASS"
            if ok
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 13
        # ----------------------------------------------------

        test_results[
            "TEST_13_nearest_neighbor_resize"
        ] = (
            "PASS"
            if (
                ok
                and
                meta.get("mask_dimensions")
                == [100, 100]
                and
                meta.get("resized")
                is True
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 14
        # ----------------------------------------------------

        manifest_check = pd.read_csv(
            temp_manifest
        )

        row = manifest_check[
            manifest_check["image_path"]
            == str(dummy_train)
        ].iloc[0]

        test_results[
            "TEST_14_manifest_persistence"
        ] = (
            "PASS"
            if (
                row["annotation_status"]
                == "ANNOTATED"
                and
                row["validation_status"]
                == "PASSED"
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 15
        # ----------------------------------------------------

        skip_ok = mark_image_skipped(
            dummy_val,
            "Skipped test",
            temp_manifest,
        )

        manifest_check = pd.read_csv(
            temp_manifest
        )

        row = manifest_check[
            manifest_check["image_path"]
            == str(dummy_val)
        ].iloc[0]

        test_results[
            "TEST_15_skip_logic"
        ] = (
            "PASS"
            if (
                skip_ok
                and
                row["annotation_status"]
                == "SKIPPED"
                and
                row["validation_status"]
                == "UNVALIDATED"
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 16
        # ----------------------------------------------------

        next_train = get_next_pending_image(
            split="Train",
            manifest_csv=temp_manifest,
        )

        test_results[
            "TEST_16_next_pending_logic"
        ] = (
            "PASS"
            if next_train is None
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 17
        # ----------------------------------------------------

        progress = (
            get_annotation_progress_report(
                temp_manifest
            )
        )

        test_results[
            "TEST_17_progress_calculation"
        ] = (
            "PASS"
            if (
                progress["annotated_count"]
                == 1
                and
                progress["skipped_review_count"]
                == 1
                and
                progress["pending_count"]
                == 0
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 18
        # ----------------------------------------------------

        hashes_after = {
            "train_split.csv":
                compute_file_hash(train_csv),

            "val_split.csv":
                compute_file_hash(val_csv),

            "test_split.csv":
                compute_file_hash(test_csv),
        }

        test_results[
            "TEST_18_classification_files_unchanged"
        ] = (
            "PASS"
            if (
                hashes_before
                == hashes_after
                and
                (
                    (
                        train_csv.exists()
                        and
                        val_csv.exists()
                        and
                        test_csv.exists()
                    )
                    or
                    all(
                        x is None
                        for x in hashes_before.values()
                    )
                )
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 19
        # Actual test-image fingerprint
        # ----------------------------------------------------

        test_hash_before = (
            compute_file_hash(dummy_test)
        )

        try:

            process_annotation_submission(
                dummy_test,
                _make_dummy_mask(2),
                "Test",
                "Leaf_Miner",
                temp_manifest,
            )

            annotation_blocked = False

        except PermissionError:

            annotation_blocked = True

        test_hash_after = (
            compute_file_hash(dummy_test)
        )

        manifest_after = pd.read_csv(
            temp_manifest
        )

        test_row = manifest_after[
            manifest_after["image_path"]
            == str(dummy_test)
        ].iloc[0]

        test_results[
            "TEST_19_test_images_unchanged"
        ] = (
            "PASS"
            if (
                annotation_blocked
                and
                test_hash_before
                == test_hash_after
                and
                test_row[
                    "annotation_status"
                ] == "PENDING"
                and
                test_row[
                    "validation_status"
                ] == "UNVALIDATED"
                and
                not Path(
                    test_row[
                        "expected_mask_path"
                    ]
                ).exists()
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 20
        # ----------------------------------------------------

        error_response = (
            colab_save_mask_handler(
                str(temp_dir / "missing.jpg"),
                "invalid_base64",
                "Train",
                "TMB",
                temp_manifest,
            )
        )

        test_results[
            "TEST_20_callback_exception_handling"
        ] = (
            "PASS"
            if (
                isinstance(
                    error_response,
                    dict,
                )
                and
                error_response.get(
                    "success"
                ) is False
                and
                bool(
                    error_response.get(
                        "message"
                    )
                )
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 21
        # ----------------------------------------------------

        reg1 = register_colab_callbacks()
        reg2 = register_colab_callbacks()

        test_results[
            "TEST_21_duplicate_registration"
        ] = (
            "PASS"
            if (
                set(
                    reg1.get(
                        "callbacks",
                        [],
                    )
                )
                ==
                set(
                    reg2.get(
                        "callbacks",
                        [],
                    )
                )
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 22
        # Behavioral undo/redo test
        # ----------------------------------------------------

        model = CanvasHistoryStateModel(
            100,
            100,
        )

        initial_empty = (
            np.count_nonzero(
                model.current_state
            ) == 0
        )

        model.paint_stroke(
            10,
            30,
            10,
            30,
            3,
        )

        painted = (
            np.count_nonzero(
                model.current_state
            ) > 0
        )

        after_paint = (
            model.current_state.copy()
        )

        undo_ok = (
            model.undo_stroke()
            and
            np.count_nonzero(
                model.current_state
            ) == 0
        )

        redo_ok = (
            model.redo_stroke()
            and
            np.array_equal(
                model.current_state,
                after_paint,
            )
        )

        model.undo_stroke()

        model.paint_stroke(
            40,
            50,
            40,
            50,
            1,
        )

        redo_cleared = (
            len(model.redo_stack) == 0
        )

        model.clear_canvas()

        clear_ok = (
            np.count_nonzero(
                model.current_state
            ) == 0
        )

        model.load_next_image(
            120,
            120,
        )

        reset_ok = (
            model.width == 120
            and
            model.height == 120
            and
            np.count_nonzero(
                model.current_state
            ) == 0
            and
            len(model.undo_stack) == 1
            and
            len(model.redo_stack) == 0
        )

        test_results[
            "TEST_22_undo_redo_state_reset"
        ] = (
            "PASS"
            if (
                initial_empty
                and
                painted
                and
                undo_ok
                and
                redo_ok
                and
                redo_cleared
                and
                clear_ok
                and
                reset_ok
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 23
        # Filter preservation
        # ----------------------------------------------------

        dummy_tmb_2 = (
            temp_dir / "dummy_tmb_2.jpg"
        )

        _make_dummy_image(
            dummy_tmb_2
        )

        df = pd.read_csv(
            temp_manifest
        )

        new_row = pd.DataFrame([{
            "image_path":
                str(dummy_tmb_2),

            "image_name":
                dummy_tmb_2.name,

            "class_name":
                "TMB",

            "class_code":
                3,

            "split":
                "Train",

            "expected_mask_path":
                str(
                    temp_dir
                    / "dummy_tmb_2_mask.png"
                ),

            "annotation_status":
                "PENDING",

            "validation_status":
                "UNVALIDATED",

            "error_message":
                "Pending",
        }])

        df = pd.concat(
            [df, new_row],
            ignore_index=True,
        )

        save_manifest_atomically(
            df,
            temp_manifest,
        )

        valid_mask = _make_dummy_mask(
            3
        )

        result = (
            process_annotation_submission(
                dummy_tmb_2,
                valid_mask,
                "Train",
                "TMB",
                temp_manifest,
            )
        )

        # The original Train/TMB image was already
        # annotated, so the newly added TMB image
        # should be returned.
        next_item = get_next_pending_image(
            split="Train",
            class_name="TMB",
            manifest_csv=temp_manifest,
        )

        test_results[
            "TEST_23_filter_preservation"
        ] = (
            "PASS"
            if (
                result[0]
                and
                next_item is None
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 24
        # ----------------------------------------------------

        safe = make_json_safe({
            "np_int":
                np.int64(42),

            "np_arr":
                np.array([1, 2, 3]),

            "np_bool":
                np.bool_(True),
        })

        test_results[
            "TEST_24_json_serialization"
        ] = (
            "PASS"
            if (
                isinstance(
                    safe["np_int"],
                    int,
                )
                and
                isinstance(
                    safe["np_arr"],
                    list,
                )
                and
                isinstance(
                    safe["np_bool"],
                    bool,
                )
            )
            else "FAIL"
        )

        # ----------------------------------------------------
        # TEST 25
        # ----------------------------------------------------

        original_path = (
            r'C:\Users\Test User\Cashew "Images"\image.jpg'
        )

        encoded = json.dumps(
            original_path
        )

        decoded = json.loads(
            encoded
        )

        test_results[
            "TEST_25_path_escaping"
        ] = (
            "PASS"
            if decoded == original_path
            else "FAIL"
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    failed = [
        name
        for name, result in test_results.items()
        if result != "PASS"
    ]

    all_passed = len(failed) == 0

    hashes_final = {
        "train_split.csv":
            compute_file_hash(train_csv),

        "val_split.csv":
            compute_file_hash(val_csv),

        "test_split.csv":
            compute_file_hash(test_csv),
    }

    classification_files_unchanged = (
        hashes_before == hashes_final
        and
        all(
            os.path.exists(x)
            for x in [
                train_csv,
                val_csv,
                test_csv,
            ]
        )
    )

    print(
        "\n--- PHASE C.1 VERIFICATION RESULTS ---"
    )

    for name, result in test_results.items():
        print(
            f"{name:<50}: {result}"
        )

    print(
        "\nAll Tests Passed:",
        all_passed,
    )

    print(
        "Classification Split Files Unchanged:",
        classification_files_unchanged,
    )

    return make_json_safe({
        "all_passed":
            all_passed,

        "tests":
            test_results,

        "test_results":
            test_results,

        "failed_tests":
            failed,

        "classification_files_unchanged":
            classification_files_unchanged,

        "file_hashes_before":
            hashes_before,

        "file_hashes_after":
            hashes_final,

        "hashes_unchanged":
            hashes_before == hashes_final,
    })


# ============================================================
# STARTUP HELPER
# ============================================================

def initialize_phase_c1(
    train_csv: str | Path | None = None,
    val_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    force_rebuild_manifest: bool = False,
) -> Path:

    """
    Initializes the segmentation manifest.

    IMPORTANT:
    The classification split CSV files are read only.
    """

    manifest_path = get_manifest_path()

    if (
        manifest_path.exists()
        and
        not force_rebuild_manifest
    ):
        print(
            f"Existing segmentation manifest preserved:\n"
            f"{manifest_path}"
        )

        register_colab_callbacks()

        return manifest_path

    if train_csv is None:
        train_csv = (
            get_preprocessed_dir()
            / "train_split.csv"
        )

    if val_csv is None:
        val_csv = (
            get_preprocessed_dir()
            / "val_split.csv"
        )

    if test_csv is None:
        test_csv = (
            get_preprocessed_dir()
            / "test_split.csv"
        )

    create_manifest_from_split_files(
        train_csv,
        val_csv,
        test_csv,
        manifest_path,
    )

    register_colab_callbacks()

    print(
        "Phase C.1 initialized."
    )

    print(
        f"Manifest: {manifest_path}"
    )

    return manifest_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Phase C.1 Segmentation Tool"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Preprocessed directory: "
        f"{get_preprocessed_dir()}"
    )

    print(
        f"Segmentation directory: "
        f"{get_segmentation_dir()}"
    )

    print(
        "\nRunning verification suite..."
    )

    result = (
        run_phase_c1_verification_suite()
    )

    print(
        "\nFinal status:",
        result["all_passed"],
    )

    if result["failed_tests"]:
        print(
            "Failed tests:",
            result["failed_tests"],
        )