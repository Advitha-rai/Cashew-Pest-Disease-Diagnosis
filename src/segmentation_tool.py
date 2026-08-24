"""
Cashew Pest and Disease Diagnosis System
Phase C.1: Compatibility Shim for src.segmentation_tool
Delegates to the canonical modular src.segmentation package.
"""

from src.segmentation.config import (
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
    get_project_root,
    get_dataset_dir,
    get_preprocessed_dir,
    get_segmentation_dir,
    get_manifest_path,
    get_annotations_dir,
)
from src.segmentation.history import CanvasHistoryStateModel
from src.segmentation.validation import (
    validate_mask_file,
    validate_mask_array,
    assert_annotation_allowed,
    compute_file_hash,
)
from src.segmentation.manifest import (
    load_manifest,
    save_manifest_atomically,
    build_segmentation_manifest,
    get_next_pending_image,
    get_annotation_progress_report,
)
from src.segmentation.callbacks import (
    register_colab_callbacks,
    decode_colab_response_payload,
    make_json_safe,
    colab_callback_health_check,
)
from src.segmentation.ui import (
    build_annotation_html,
    image_to_base64,
)
from src.segmentation.pipeline import (
    initialize_segmentation,
    launch_segmentation_tool,
    process_annotation_submission,
    skip_annotation,
    get_annotation_progress,
    colab_save_mask_handler,
    colab_skip_image_handler,
)

# Compatibility Aliases for Legacy Function Names
launch_colab_annotation_interface = launch_segmentation_tool
show_annotation_tool = launch_segmentation_tool
initialize_phase_c1 = initialize_segmentation
run_minimal_colab_callback_test = colab_callback_health_check

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
    "launch_colab_annotation_interface",
    "show_annotation_tool",
    "process_annotation_submission",
    "skip_annotation",
    "get_annotation_progress",
    "colab_save_mask_handler",
    "colab_skip_image_handler",
    "initialize_phase_c1",
    "run_minimal_colab_callback_test",
]