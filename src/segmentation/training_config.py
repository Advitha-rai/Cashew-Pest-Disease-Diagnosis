"""
Cashew Pest and Disease Diagnosis System
Phase C.2: Segmentation Training Configuration
Framework: TensorFlow / Keras
"""

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from datetime import datetime


@dataclass
class SegmentationTrainingConfig:
    """Configuration parameters for U-Net lesion segmentation training."""

    # Model architecture
    image_height: int = 224
    image_width: int = 224
    num_channels: int = 3
    num_classes: int = 1  # Binary lesion segmentation (foreground vs background)
    encoder_filters: Tuple[int, ...] = (32, 64, 128, 256)
    bottleneck_filters: int = 512
    dropout_rate: float = 0.2

    # Training hyperparameters
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 30
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    smooth: float = 1e-6

    # Callbacks
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6

    # Data policy
    min_samples_warning_threshold: int = 20

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.image_height, self.image_width, self.num_channels)

    @property
    def output_shape(self) -> Tuple[int, int, int]:
        return (self.image_height, self.image_width, self.num_classes)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_experiment_dir(base_dir: Optional[Path] = None, run_name: Optional[str] = None) -> Path:
    """
    Creates a new timestamped experiment directory under Google Drive or local storage.
    Example: Experiments/Segmentation/Experiments/UNet/run_YYYYMMDD_HHMMSS/
    """
    if base_dir is None:
        if Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
            base_dir = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
        elif Path("/content/Cashew-Pest-Disease-Diagnosis").exists():
            base_dir = Path("/content/Cashew-Pest-Disease-Diagnosis")
        else:
            base_dir = Path.cwd()

    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    exp_dir = base_dir / "Experiments" / "Segmentation" / "Experiments" / "UNet" / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
