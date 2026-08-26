"""
Cashew Pest and Disease Diagnosis System
Phase C: Manual Cashew Leaf Segmentation Package
Framework: TensorFlow / Keras

Public API:
    - initialize_segmentation()
    - launch_segmentation_tool()
    - get_annotation_progress()
    - process_annotation_submission()
    - skip_annotation()
"""

from .config import (
    IMAGE_SIZE,
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    MASK_DTYPE,
    MASK_MODE,
    CLASS_CODES,
    ALLOWED_MASK_VALUES,
    ANNOTATABLE_SPLITS,
    READ_ONLY_SPLIT,
    PROTECTED_SPLITS,
    EXPECTED_COUNTS,
    UI_OVERLAY_COLORS,
    PROJECT_ROOT,
    DATASET_DIR,
    PREPROCESSED_DIR,
    SEGMENTATION_DIR,
    CANONICAL_MANIFEST,
    ANNOTATIONS_DIR,
    SegmentationConfig,
    normalize_class_name,
    get_class_code,
)
from .history import CanvasHistoryStateModel
from .validation import (
    validate_mask_file,
    validate_mask_array,
    assert_annotation_allowed,
    compute_file_hash,
)
from .manifest import (
    load_manifest,
    save_manifest_atomically,
    build_segmentation_manifest,
    get_next_pending_image,
    get_annotation_progress_report,
)
from .callbacks import (
    register_colab_callbacks,
    decode_colab_response_payload,
    make_json_safe,
    colab_callback_health_check,
)
from .ui import (
    build_annotation_html,
    image_to_base64,
)
from .pipeline import (
    initialize_segmentation,
    launch_segmentation_tool,
    process_annotation_submission,
    skip_annotation,
    get_annotation_progress,
    colab_save_mask_handler,
    colab_skip_image_handler,
)

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
    "CanvasHistoryStateModel",
    "validate_mask_file",
    "validate_mask_array",
    "assert_annotation_allowed",
    "compute_file_hash",
    "load_manifest",
    "save_manifest_atomically",
    "build_segmentation_manifest",
    "get_next_pending_image",
    "get_annotation_progress_report",
    "register_colab_callbacks",
    "decode_colab_response_payload",
    "make_json_safe",
    "colab_callback_health_check",
    "build_annotation_html",
    "image_to_base64",
    "initialize_segmentation",
    "launch_segmentation_tool",
    "process_annotation_submission",
    "skip_annotation",
    "get_annotation_progress",
    "colab_save_mask_handler",
    "colab_skip_image_handler",
    "SegmentationTrainingConfig",
    "get_experiment_dir",
    "SegmentationDatasetLoader",
    "build_unet",
    "dice_coef",
    "dice_loss",
    "iou_metric",
    "bce_dice_loss",
    "SegmentationTrainer",
    "SegmentationEvaluator",
    "SegmentationExperimentManager",
]

# Phase C.2 Training API
from .training_config import SegmentationTrainingConfig, get_experiment_dir
from .data_loader import SegmentationDatasetLoader
from .unet_model import build_unet, dice_coef, dice_loss, iou_metric, bce_dice_loss
from .trainer import SegmentationTrainer
from .evaluator import SegmentationEvaluator
from .experiment import SegmentationExperimentManager
