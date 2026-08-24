"""
Cashew Pest and Disease Diagnosis System
Phase C: Canonical Segmentation Configuration & Constants
Framework: TensorFlow / Keras
"""

import os
from pathlib import Path
from typing import Dict, Set, Tuple, Optional


class SegmentationConfig:
    """
    Single source of truth for all segmentation configuration, class mappings,
    file paths, and split security policies.
    """
    # ---------------------------------------------------------
    # Image Specifications
    # ---------------------------------------------------------
    IMG_HEIGHT: int = 224
    IMG_WIDTH: int = 224
    IMG_SIZE: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH)
    CHANNELS: int = 3
    MASK_DTYPE: str = "uint8"
    MASK_MODE: str = "L"  # Grayscale 8-bit single channel

    # ---------------------------------------------------------
    # Canonical Class Codes & Encodings
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
    # Split Security Policy
    # ---------------------------------------------------------
    ANNOTATABLE_SPLITS: Set[str] = {"Train", "Validation"}
    PROTECTED_SPLITS: Set[str] = {"Test"}

    # Expected Dataset Partition Counts
    EXPECTED_COUNTS: Dict[str, int] = {
        "Train": 4013,
        "Validation": 860,
        "Test": 861,
        "Total": 5734,
        "Eligible": 4873,
    }

    # ---------------------------------------------------------
    # UI Visualization Colors (Overlay Only)
    # ---------------------------------------------------------
    UI_OVERLAY_COLORS: Dict[int, str] = {
        0: "rgba(0, 0, 0, 0)",          # Transparent background
        1: "rgba(220, 53, 69, 0.7)",    # Aphids -> Red
        2: "rgba(40, 167, 69, 0.7)",    # Leaf_Miner -> Green
        3: "rgba(0, 123, 255, 0.7)",    # Leaf_Blight -> Blue
        4: "rgba(255, 193, 7, 0.7)",    # TMB -> Yellow
    }

    # ---------------------------------------------------------
    # Dynamic Path Resolvers
    # ---------------------------------------------------------
    @classmethod
    def get_project_root(cls) -> Path:
        """Finds the root repository or drive workspace."""
        drive_path = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
        if drive_path.exists():
            return drive_path

        colab_repo = Path("/content/Cashew-Pest-Disease-Diagnosis")
        if colab_repo.exists():
            return colab_repo

        return Path.cwd()

    @classmethod
    def get_dataset_dir(cls) -> Path:
        """Returns path to cleaned dataset directory."""
        root = cls.get_project_root()
        candidate = root / "Dataset" / "Cleaned"
        if candidate.exists():
            return candidate
        return root / "Dataset"

    @classmethod
    def get_preprocessed_dir(cls) -> Path:
        """Returns path to preprocessed splits directory."""
        path = cls.get_project_root() / "Preprocessed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_segmentation_dir(cls) -> Path:
        """Returns path to segmentation experiments directory."""
        path = cls.get_project_root() / "Experiments" / "Segmentation"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def get_manifest_path(cls) -> Path:
        """Returns single canonical manifest CSV path."""
        return cls.get_segmentation_dir() / "segmentation_annotation_manifest.csv"

    @classmethod
    def get_annotations_dir(cls) -> Path:
        """Returns isolated annotation masks directory."""
        path = cls.get_segmentation_dir() / "Annotations"
        path.mkdir(parents=True, exist_ok=True)
        return path


def normalize_class_name(raw_class: str) -> str:
    """
    Normalizes arbitrary class name string representations (with spaces,
    underscores, casing variations) to canonical project class names.
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
            f"Expected one of: {sorted(list(SegmentationConfig.CLASS_CODES.keys()))}"
        )


def get_class_code(class_name: str) -> int:
    """Returns canonical numeric code (0-4) for a given class name."""
    canonical = normalize_class_name(class_name)
    return SegmentationConfig.CLASS_CODES[canonical]
