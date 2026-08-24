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
]
