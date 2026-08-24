"""
Cashew Pest and Disease Diagnosis System
Phase C: Compatibility Shim for src.segmentation
Delegates to the canonical modular src.segmentation package.
"""

from src.segmentation.config import (
    SegmentationConfig,
    normalize_class_name,
    get_class_code,
)
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
build_segmentation_annotation_manifest = build_segmentation_manifest
launch_colab_annotation_interface = launch_segmentation_tool
show_annotation_tool = launch_segmentation_tool
validate_all_manifest_masks = lambda manifest_csv=None: load_manifest(manifest_csv)

__all__ = [
    "SegmentationConfig",
    "normalize_class_name",
    "get_class_code",
    "validate_mask_file",
    "validate_mask_array",
    "assert_annotation_allowed",
    "compute_file_hash",
    "load_manifest",
    "save_manifest_atomically",
    "build_segmentation_manifest",
    "build_segmentation_annotation_manifest",
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
]
