"""
Cashew Pest and Disease Diagnosis System
Global Configuration & Experiment Parameters
"""

import os
import torch
from pathlib import Path

class Config:
    # ---------------------------------------------------------
    # System & Seed Settings
    # ---------------------------------------------------------
    SEED = 42
    PROJECT_NAME = "Cashew_Pest_Disease_Project"
    
    # ---------------------------------------------------------
    # Supported Vision Models (1-10)
    # ---------------------------------------------------------
    MODEL_MAP = {
        1: ("01_MobileNetV2", "mobilenet_v2"),
        2: ("02_ResNet50", "resnet50"),
        3: ("03_VGG16", "vgg16"),
        4: ("04_InceptionV3", "inception_v3"),
        5: ("05_DenseNet121", "densenet121"),
        6: ("06_EfficientNetV2", "efficientnet_v2_s"),
        7: ("07_MobileNetV3", "mobilenet_v3_large"),
        8: ("08_ConvNeXt", "convnext_tiny"),
        9: ("09_SwinTransformer", "swin_t"),
        10: ("10_DINOv2", "dinov2_vits14")
    }

    # ---------------------------------------------------------
    # Target Classes (Inferred dynamically from raw folder if present)
    # ---------------------------------------------------------
    DEFAULT_CLASSES = [
        "Anthracnose",
        "Gummosis",
        "Healthy",
        "Leaf_Miner",
        "Powdery_Mildew",
        "Stem_Borer",
        "Tea_Mosquito_Bug"
    ]
    
    # ---------------------------------------------------------
    # Dataset Split Ratios
    # ---------------------------------------------------------
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience
    UNFREEZE_EPOCH = 5  # Unfreeze backbone after 5 epochs
    CONFIDENCE_THRESHOLD = 0.80  # 80% confidence requirement

    # ---------------------------------------------------------
    # Device Resolution
    # ---------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_base_dir(cls) -> str:
        """Dynamically resolves root path (Google Drive vs Local workspace)."""
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            return os.path.join(drive_path, cls.PROJECT_NAME)
        return os.path.join(os.getcwd(), cls.PROJECT_NAME)

    @classmethod
    def get_experiment_dir(cls, model_index: int) -> str:
        """Returns the isolated experiment folder path for a selected model (1-10)."""
        if model_index not in cls.MODEL_MAP:
            raise ValueError(f"Invalid model index {model_index}. Must be between 1 and 10.")
        folder_name, _ = cls.MODEL_MAP[model_index]
        return os.path.join(cls.get_base_dir(), "Experiments", folder_name)

    @classmethod
    def get_dataset_dir(cls, sub_split: str = "Raw") -> str:
        """Returns dataset sub-directory path."""
        return os.path.join(cls.get_base_dir(), "Dataset", sub_split)

    @classmethod
    def get_ensemble_dir(cls) -> str:
        """Returns ensemble experiment directory."""
        return os.path.join(cls.get_base_dir(), "Experiments", "Ensemble")

    @classmethod
    def get_deployment_dir(cls) -> str:
        """Returns deployment package directory."""
        return os.path.join(cls.get_base_dir(), "Deployment")
