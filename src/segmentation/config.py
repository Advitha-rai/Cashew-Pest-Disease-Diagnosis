"""
Cashew Pest and Disease Diagnosis System
Phase C: Canonical Segmentation Configuration & Constants
Framework: TensorFlow / Keras
"""

import os
from pathlib import Path
from typing import Dict, Set, Tuple, Optional


# ---------------------------------------------------------
# 1. Image Specifications
# ---------------------------------------------------------
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
IMAGE_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS: int = 3
MASK_DTYPE: str = "uint8"
MASK_MODE: str = "L"  # Grayscale 8-bit single channel


# ---------------------------------------------------------
# 2. Canonical Class Codes & Encodings
# ---------------------------------------------------------
CLASS_CODES: Dict[str, int] = {
    "Background": 0,
    "Aphids": 1,
    "Leaf_Miner": 2,
    "Leaf_Blight": 3,
    "TMB": 4,
}

ALLOWED_MASK_VALUES: Set[int] = {0, 1, 2, 3, 4}


# ---------------------------------------------------------
# 3. Split Security Policy
# ---------------------------------------------------------
ANNOTATABLE_SPLITS: Tuple[str, ...] = ("Train", "Validation")
READ_ONLY_SPLIT: str = "Test"
PROTECTED_SPLITS: Tuple[str, ...] = ("Test",)

EXPECTED_COUNTS: Dict[str, int] = {
    "Train": 4013,
    "Validation": 860,
    "Test": 861,
    "Total": 5734,
    "Eligible": 4873,
}


# ---------------------------------------------------------
# 4. UI Visualization Colors (Overlay Only)
# ---------------------------------------------------------
UI_OVERLAY_COLORS: Dict[int, str] = {
    0: "rgba(0, 0, 0, 0)",          # Transparent background
    1: "rgba(220, 53, 69, 0.7)",    # Aphids -> Red
    2: "rgba(40, 167, 69, 0.7)",    # Leaf_Miner -> Green
    3: "rgba(0, 123, 255, 0.7)",    # Leaf_Blight -> Blue
    4: "rgba(255, 193, 7, 0.7)",    # TMB -> Yellow
}


# ---------------------------------------------------------
# 5. Dynamic Project Path Resolvers
# ---------------------------------------------------------
def get_project_root() -> Path:
    """Finds the root repository or drive workspace."""
    drive_path = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
    if drive_path.exists():
        return drive_path

    colab_repo = Path("/content/Cashew-Pest-Disease-Diagnosis")
    if colab_repo.exists():
        return colab_repo

    return Path.cwd()


def get_dataset_dir() -> Path:
    """Returns path to cleaned dataset directory."""
    candidates = [
        Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project/Dataset/Cleaned"),
        Path("/content/Cashew-Pest-Disease-Diagnosis/Dataset/Cleaned"),
        Path.cwd() / "Dataset" / "Cleaned",
        Path.cwd() / "Dataset",
    ]
    for c in candidates:
        if c.exists():
            return c
    return get_project_root() / "Dataset" / "Cleaned"


def get_preprocessed_dir() -> Path:
    """Returns path to preprocessed splits directory."""
    candidates = [
        Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project/Preprocessed"),
        Path("/content/Cashew-Pest-Disease-Diagnosis/Preprocessed"),
        Path.cwd() / "Preprocessed",
    ]
    for c in candidates:
        if c.exists():
            return c
    path = get_project_root() / "Preprocessed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_segmentation_dir() -> Path:
    """Returns path to segmentation experiments directory."""
    candidates = [
        Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project/Experiments/Segmentation"),
        Path("/content/Cashew-Pest-Disease-Diagnosis/Experiments/Segmentation"),
        Path.cwd() / "Experiments" / "Segmentation",
    ]
    for c in candidates:
        if c.exists():
            return c
    path = get_project_root() / "Experiments" / "Segmentation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_manifest_path() -> Path:
    """Returns single canonical manifest CSV path."""
    return get_segmentation_dir() / "segmentation_annotation_manifest.csv"


def get_annotations_dir() -> Path:
    """Returns isolated annotation masks directory."""
    path = get_segmentation_dir() / "Annotations"
    path.mkdir(parents=True, exist_ok=True)
    return path


PROJECT_ROOT: Path = get_project_root()
DATASET_DIR: Path = get_dataset_dir()
PREPROCESSED_DIR: Path = get_preprocessed_dir()
SEGMENTATION_DIR: Path = get_segmentation_dir()
CANONICAL_MANIFEST: Path = get_manifest_path()
ANNOTATIONS_DIR: Path = get_annotations_dir()


# ---------------------------------------------------------
# 6. Backward-Compatible Config Class
# ---------------------------------------------------------
class SegmentationConfig:
    """Class wrapper exposing all canonical constants and path methods."""
    IMG_HEIGHT = IMG_HEIGHT
    IMG_WIDTH = IMG_WIDTH
    IMG_SIZE = IMAGE_SIZE
    IMAGE_SIZE = IMAGE_SIZE
    CHANNELS = CHANNELS
    MASK_DTYPE = MASK_DTYPE
    MASK_MODE = MASK_MODE

    CLASS_CODES = CLASS_CODES
    ALLOWED_MASK_VALUES = ALLOWED_MASK_VALUES
    ANNOTATABLE_SPLITS = set(ANNOTATABLE_SPLITS)
    READ_ONLY_SPLIT = READ_ONLY_SPLIT
    PROTECTED_SPLITS = set(PROTECTED_SPLITS)
    EXPECTED_COUNTS = EXPECTED_COUNTS
    UI_OVERLAY_COLORS = UI_OVERLAY_COLORS

    PROJECT_ROOT = PROJECT_ROOT
    DATASET_DIR = DATASET_DIR
    PREPROCESSED_DIR = PREPROCESSED_DIR
    SEGMENTATION_DIR = SEGMENTATION_DIR
    CANONICAL_MANIFEST = CANONICAL_MANIFEST
    ANNOTATIONS_DIR = ANNOTATIONS_DIR

    @classmethod
    def get_project_root(cls) -> Path:
        return get_project_root()

    @classmethod
    def get_dataset_dir(cls) -> Path:
        return get_dataset_dir()

    @classmethod
    def get_preprocessed_dir(cls) -> Path:
        return get_preprocessed_dir()

    @classmethod
    def get_segmentation_dir(cls) -> Path:
        return get_segmentation_dir()

    @classmethod
    def get_manifest_path(cls) -> Path:
        return get_manifest_path()

    @classmethod
    def get_annotations_dir(cls) -> Path:
        return get_annotations_dir()


# ---------------------------------------------------------
# 7. Robust Class Normalization
# ---------------------------------------------------------
def normalize_class_name(raw_class: str) -> str:
    """
    Normalizes arbitrary class name string representations (with spaces,
    underscores, casing variations) to canonical project class names:
      - Aphids -> Aphids
      - Leaf miner / leaf_miner / LEAF MINER / Leaf Miner -> Leaf_Miner
      - Leaf Blight / leaf_blight / LEAF BLIGHT -> Leaf_Blight
      - tmb / TMB / tea mosquito bug / Tea Mosquito Bug -> TMB
      - background -> Background
    """
    raw = str(raw_class).strip().lower().replace("-", "_").replace(" ", "_")

    if "aphid" in raw:
        return "Aphids"
    elif "miner" in raw:
        return "Leaf_Miner"
    elif "blight" in raw:
        return "Leaf_Blight"
    elif "tmb" in raw or "mosquito" in raw:
        return "TMB"
    elif "bg" in raw or "back" in raw or "healthy" in raw:
        return "Background"
    else:
        raise ValueError(
            f"Unknown class name: '{raw_class}'. "
            f"Expected one of: {sorted(list(CLASS_CODES.keys()))}"
        )


def get_class_code(class_name: str) -> int:
    """Returns canonical numeric code (0-4) for a given class name."""
    canonical = normalize_class_name(class_name)
    return CLASS_CODES[canonical]


__all__ = [
    "IMAGE_SIZE",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "CHANNELS",
    "MASK_DTYPE",
    "MASK_MODE",
    "CLASS_CODES",
    "ALLOWED_MASK_VALUES",
    "ANNOTATABLE_SPLITS",
    "READ_ONLY_SPLIT",
    "PROTECTED_SPLITS",
    "EXPECTED_COUNTS",
    "UI_OVERLAY_COLORS",
    "PROJECT_ROOT",
    "DATASET_DIR",
    "PREPROCESSED_DIR",
    "SEGMENTATION_DIR",
    "CANONICAL_MANIFEST",
    "ANNOTATIONS_DIR",
    "SegmentationConfig",
    "normalize_class_name",
    "get_class_code",
    "get_project_root",
    "get_dataset_dir",
    "get_preprocessed_dir",
    "get_segmentation_dir",
    "get_manifest_path",
    "get_annotations_dir",
]
