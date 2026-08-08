"""
Cashew Pest and Disease Diagnosis System
Phase 5: Global Configuration Parameters
Framework: TensorFlow / Keras
"""

import os

class Config:
    # ---------------------------------------------------------
    # System & Reproducibility Settings
    # ---------------------------------------------------------
    SEED = 42
    PROJECT_NAME = "Cashew_Pest_Disease_Project"
    
    # ---------------------------------------------------------
    # Dataset & Preprocessing Specifications
    # ---------------------------------------------------------
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
    CHANNELS = 3  # RGB preservation
    
    # ---------------------------------------------------------
    # Supported TensorFlow / Keras Vision Architectures (1-8)
    # ---------------------------------------------------------
    MODEL_MAP = {
        1: ("01_MobileNetV2", "mobilenet_v2"),
        2: ("02_ResNet50", "resnet50"),
        3: ("03_VGG16", "vgg16"),
        4: ("04_InceptionV3", "inception_v3"),
        5: ("05_DenseNet121", "densenet121"),
        6: ("06_EfficientNetV2B0", "efficientnet_v2_b0"),
        7: ("07_MobileNetV3Large", "mobilenet_v3_large"),
        8: ("08_ConvNeXtTiny", "convnext_tiny")
    }

    # ---------------------------------------------------------
    # Phase 5 Ensemble Selected Sub-Models
    # ---------------------------------------------------------
    ENSEMBLE_MODEL_INDICES = [3, 5, 8]  # 03_VGG16, 05_DenseNet121, 08_ConvNeXtTiny
    ENSEMBLE_MODEL_NAMES = ["VGG16", "DenseNet121", "ConvNeXtTiny"]

    # ---------------------------------------------------------
    # Target Classes (Dynamic folder auto-detection fallback)
    # ---------------------------------------------------------
    DEFAULT_CLASSES = [
        "Aphids",
        "Leaf_Blight",
        "Leaf_Miner",
        "TMB"
    ]
    
    # ---------------------------------------------------------
    # Reproducible Split Ratios (70% Train / 15% Val / 15% Test)
    # ---------------------------------------------------------
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    WARMUP_EPOCHS = 5  # Initial epochs with frozen backbone
    LEARNING_RATE = 1e-4
    FINE_TUNE_LEARNING_RATE = 1e-5
    OPTIMIZER = "adam"  # Options: adam, adamw, sgd
    PATIENCE = 10
    REDUCE_LR_PATIENCE = 3

    # ---------------------------------------------------------
    # Inference Confidence Threshold & Uncertainty Policy
    # ---------------------------------------------------------
    CONFIDENCE_THRESHOLD = 0.80  # 80% confidence requirement to avoid random guessing
    UNCERTAIN_PREDICTION_MESSAGE = "Prediction Uncertain. Please upload a clearer image"
    INVALID_IMAGE_MESSAGE = "Invalid image. Please upload a valid cashew leaf image."

    # ---------------------------------------------------------
    # Dynamic Google Drive Path Resolvers
    # ---------------------------------------------------------
    @classmethod
    def get_base_dir(cls) -> str:
        """Resolves root path dynamically (Google Drive vs Local workspace)."""
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            return os.path.join(drive_path, cls.PROJECT_NAME)
        return os.path.join(os.getcwd(), cls.PROJECT_NAME)

    @classmethod
    def get_raw_dir(cls) -> str:
        """Returns path to raw dataset: /content/drive/MyDrive/Cashew_Pest_Disease_Project/Dataset/Raw/"""
        return os.path.join(cls.get_base_dir(), "Dataset", "Raw")

    @classmethod
    def get_cleaned_dir(cls) -> str:
        """Returns path to cleaned dataset."""
        return os.path.join(cls.get_base_dir(), "Dataset", "Cleaned")

    @classmethod
    def get_preprocessed_dir(cls) -> str:
        """Returns path to preprocessed outputs in Google Drive."""
        path = os.path.join(cls.get_base_dir(), "Preprocessed")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_logs_dir(cls) -> str:
        """Returns path to logs directory in Google Drive."""
        path = os.path.join(cls.get_base_dir(), "Logs")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_documentation_dir(cls) -> str:
        """Returns path to documentation directory in Google Drive."""
        path = os.path.join(cls.get_base_dir(), "Documentation")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_experiment_dir(cls, model_index: int) -> str:
        """Returns isolated experiment directory for a selected model: Experiments/<Model_Name>/"""
        if model_index not in cls.MODEL_MAP:
            raise ValueError(f"Invalid model index {model_index}. Choice must be between 1 and {len(cls.MODEL_MAP)}.")
        folder_name, _ = cls.MODEL_MAP[model_index]
        path = os.path.join(cls.get_base_dir(), "Experiments", folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_comparison_dir(cls) -> str:
        """Returns comparison directory for global model benchmarking: Experiments/Comparison/"""
        path = os.path.join(cls.get_base_dir(), "Experiments", "Comparison")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_ensemble_dir(cls) -> str:
        """Returns output directory for Phase 5 ensemble results: Experiments/Ensemble/"""
        path = os.path.join(cls.get_base_dir(), "Experiments", "Ensemble")
        os.makedirs(path, exist_ok=True)
        return path

    @classmethod
    def get_full_dataset_classification_dir(cls) -> str:
        """Returns directory for complete dataset classification report: Experiments/Ensemble/Full_Dataset_Classification/"""
        path = os.path.join(cls.get_ensemble_dir(), "Full_Dataset_Classification")
        os.makedirs(path, exist_ok=True)
        return path
