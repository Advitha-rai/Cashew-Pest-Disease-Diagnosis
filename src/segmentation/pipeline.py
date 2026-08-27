"""
Cashew Pest and Disease Diagnosis System
Phase C: Complete Segmentation Annotation Pipeline & Execution Engine
Framework: TensorFlow / Keras
"""

import io
import os
import base64
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from PIL import Image

from .config import SegmentationConfig, normalize_class_name, get_class_code
from .manifest import (
    load_manifest,
    save_manifest_atomically,
    find_manifest_match_index,
    get_next_pending_image,
    get_annotation_progress_report,
    build_segmentation_manifest,
)
from .validation import (
    assert_annotation_allowed,
    validate_mask_array,
    validate_mask_file,
    compute_file_hash,
)
from .callbacks import register_colab_callbacks, make_json_safe
from .ui import build_annotation_html, image_to_base64


def process_annotation_submission(
    image_path: str | Path,
    mask_input: Union[str, np.ndarray],
    split: str,
    class_name: str,
    manifest_csv: Optional[str | Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Core annotation submission workflow:
      1. Backend Test Split Security Check
      2. Base64 Decode -> Grayscale Mask Conversion
      3. Array Validation
      4. Atomic Temporary PNG Write
      5. Reopen & Saved File Validation
      6. Atomic Final File Replacement
      7. Atomic Manifest Status Update
      8. Progress Recalculation & Next Item Selection
    """
    img_path = Path(image_path)
    split_clean = str(split).strip().capitalize()
    norm_class = normalize_class_name(class_name)
    code = get_class_code(norm_class)

    # 1. Strict backend test split protection
    try:
        assert_annotation_allowed(split_clean, img_path)
    except PermissionError as exc:
        return False, str(exc), {"error_code": "ANNOTATION_NOT_ALLOWED"}

    # 2. Decode mask input
    if isinstance(mask_input, str):
        try:
            raw_b64 = mask_input.split(",")[1] if "," in mask_input else mask_input
            mask_bytes = base64.b64decode(raw_b64)
            with Image.open(io.BytesIO(mask_bytes)) as pil_mask:
                pil_mask.load()
                if pil_mask.mode == "RGBA":
                    rgba = np.array(pil_mask)
                    mask_arr = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint8)
                    valid_fg = (rgba[:, :, 3] > 0) & (rgba[:, :, 0] > 0)
                    mask_arr[valid_fg] = rgba[:, :, 0][valid_fg]
                else:
                    mask_arr = np.asarray(pil_mask.convert("L"), dtype=np.uint8)
        except Exception as e:
            return False, f"Failed to decode base64 mask: {e}", {"error_code": "DECODE_ERROR"}
    elif isinstance(mask_input, np.ndarray):
        mask_arr = mask_input.astype(np.uint8)
    else:
        return False, "Invalid mask input type.", {"error_code": "INVALID_INPUT"}

    # Diagnostic print / verification
    non_zero_pixels = int(np.count_nonzero(mask_arr))
    unique_vals = sorted([int(v) for v in np.unique(mask_arr)])
    print(f"[BACKEND DECODE DEBUG] Image: {img_path.name} | Shape: {mask_arr.shape} | Dtype: {mask_arr.dtype} | Min: {int(np.min(mask_arr))} | Max: {int(np.max(mask_arr))} | NonZero: {non_zero_pixels} | Unique: {unique_vals}")

    # 3. Validate array
    is_valid, val_msg, val_meta = validate_mask_array(img_path, mask_arr, code)
    if not is_valid:
        return False, val_msg, val_meta

    # 4. Manifest lookup and target paths
    manifest_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    df = load_manifest(manifest_path)
    match_idx = find_manifest_match_index(df, img_path)

    if match_idx is not None:
        target_mask_path = Path(df.at[match_idx, "expected_mask_path"])
    else:
        target_mask_path = (
            SegmentationConfig.get_annotations_dir()
            / split_clean
            / norm_class
            / f"{img_path.stem}_mask.png"
        )

    target_mask_path.parent.mkdir(parents=True, exist_ok=True)
    temp_mask_path = target_mask_path.with_suffix(".tmp.png")

    # 5. Atomic temporary PNG write & revalidation
    try:
        pil_out = Image.fromarray(mask_arr, mode="L")
        pil_out.save(temp_mask_path)

        file_valid, file_msg, file_meta = validate_mask_file(img_path, temp_mask_path, code)
        if not file_valid:
            if temp_mask_path.exists():
                temp_mask_path.unlink()
            return False, f"Saved temporary mask validation failed: {file_msg}", file_meta

        # Atomic replace
        os.replace(temp_mask_path, target_mask_path)
        mask_hash = compute_file_hash(target_mask_path)

    except Exception as e:
        if temp_mask_path.exists():
            try:
                temp_mask_path.unlink()
            except Exception:
                pass
        return False, f"Atomic mask write error: {e}", {"error_code": "FILE_WRITE_ERROR"}

    # 6. Update manifest atomically
    if match_idx is not None:
        df.at[match_idx, "annotation_status"] = "ANNOTATED"
        df.at[match_idx, "validation_status"] = "PASSED"
        df.at[match_idx, "error_message"] = "Annotation validated and saved."
        df.at[match_idx, "mask_sha256"] = str(mask_hash or "")
        save_manifest_atomically(df, manifest_path)

    # 7. Progress report & next item
    progress = get_annotation_progress_report(manifest_path)
    next_item = get_next_pending_image(split=split_clean, class_name=norm_class, manifest_csv=manifest_path)
    if next_item:
        try:
            next_item["base64"] = image_to_base64(next_item["image_path"])
        except Exception:
            pass

    return True, f"Annotation successfully validated and saved. (Unique: {unique_vals}, Foreground: {non_zero_pixels}px)", {
        "mask_path": str(target_mask_path),
        "mask_sha256": mask_hash,
        "progress": progress,
        "next_item": next_item,
        "unique_values": unique_vals,
        "foreground_pixels": non_zero_pixels,
        "class_code": code,
    }


def skip_annotation(
    image_path: str | Path,
    reason: str = "Manual review skipped",
    split: str = "Train",
    class_name: Optional[str] = None,
    manifest_csv: Optional[str | Path] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Marks an image as SKIPPED in the manifest without creating a mask or altering Test split.
    """
    img_path = Path(image_path)
    split_clean = str(split).strip().capitalize()

    try:
        assert_annotation_allowed(split_clean, img_path)
    except PermissionError as exc:
        return False, str(exc), {"error_code": "ANNOTATION_NOT_ALLOWED"}

    manifest_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    df = load_manifest(manifest_path)
    match_idx = find_manifest_match_index(df, img_path)

    if match_idx is None:
        return False, f"Image not found in manifest: {image_path}", {"error_code": "NOT_IN_MANIFEST"}

    df.at[match_idx, "annotation_status"] = "SKIPPED"
    df.at[match_idx, "validation_status"] = "UNVALIDATED"
    df.at[match_idx, "error_message"] = str(reason)
    save_manifest_atomically(df, manifest_path)

    progress = get_annotation_progress_report(manifest_path)
    norm_cls = normalize_class_name(class_name) if class_name else None
    next_item = get_next_pending_image(split=split_clean, class_name=norm_cls, manifest_csv=manifest_path)
    if next_item:
        try:
            next_item["base64"] = image_to_base64(next_item["image_path"])
        except Exception:
            pass

    return True, f"Image successfully marked skipped: {reason}", {
        "progress": progress,
        "next_item": next_item
    }


# ---------------------------------------------------------
# Callback Dispatch Handlers
# ---------------------------------------------------------
def colab_save_mask_handler(
    image_path: str,
    mask_base64: str,
    split: str,
    class_name: str,
    manifest_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Google Colab IPC handler for notebook.save_mask."""
    success, msg, data = process_annotation_submission(
        image_path=image_path,
        mask_input=mask_base64,
        split=split,
        class_name=class_name,
        manifest_csv=manifest_csv
    )
    return make_json_safe({
        "success": success,
        "message": msg,
        "progress": data.get("progress", {}),
        "next_item": data.get("next_item"),
        "unique_values": data.get("unique_values", []),
        "foreground_pixels": data.get("foreground_pixels", 0),
        "class_code": data.get("class_code", 0),
    })


def colab_skip_image_handler(
    image_path: str,
    reason: str,
    split: str,
    class_name: str,
    manifest_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Google Colab IPC handler for notebook.skip_image."""
    success, msg, data = skip_annotation(
        image_path=image_path,
        reason=reason,
        split=split,
        class_name=class_name,
        manifest_csv=manifest_csv
    )
    return make_json_safe({
        "success": success,
        "message": msg,
        "progress": data.get("progress", {}),
        "next_item": data.get("next_item")
    })


# ---------------------------------------------------------
# Public API Workflows
# ---------------------------------------------------------
def initialize_segmentation(manifest_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    """
    Single unified startup function for Jupyter/Colab:
      1. Verifies configuration and paths.
      2. Registers Google Colab callbacks idempotently.
      3. Loads/validates canonical manifest.
      4. Reports overall annotation readiness.
    """
    manifest_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    df = load_manifest(manifest_path)
    cb_status = register_colab_callbacks(
        save_handler=colab_save_mask_handler,
        skip_handler=colab_skip_image_handler
    )
    progress = get_annotation_progress_report(manifest_path)

    print("==================================================")
    print("CASHEW SEGMENTATION SYSTEM INITIALIZED")
    print("==================================================")
    print(f"Manifest Path       : {manifest_path}")
    print(f"Total Source Images : {progress['total_source_images']}")
    print(f"Eligible Pool       : {progress['total_eligible_images']} (Train={progress['train_images']}, Val={progress['validation_images']})")
    print(f"Isolated Test Set   : {progress['test_images_isolated']} [READ ONLY]")
    print(f"Annotated           : {progress['annotated_count']}")
    print(f"Passed Validation   : {progress['passed_validation_count']}")
    print(f"Skipped             : {progress['skipped_count']}")
    print(f"Pending             : {progress['pending_count']}")
    print(f"Progress            : {progress['progress_percentage']}%")
    print(f"Callbacks           : {cb_status.get('status')}")
    print("==================================================")

    return {
        "status": "READY",
        "callbacks": cb_status,
        "progress": progress,
        "manifest_path": str(manifest_path)
    }


def launch_segmentation_tool(
    split: str = "Train",
    class_name: Optional[str] = None,
    manifest_csv: Optional[str | Path] = None,
    debug_ui: bool = False,
):
    """
    Single primary public entry point to launch the interactive manual segmentation tool.
    """
    manifest_path = Path(manifest_csv) if manifest_csv else SegmentationConfig.get_manifest_path()
    register_colab_callbacks(
        save_handler=colab_save_mask_handler,
        skip_handler=colab_skip_image_handler
    )

    item = get_next_pending_image(split=split, class_name=class_name, manifest_csv=manifest_path)
    if item is None:
        print(f"🎉 No pending images found for split='{split}', class='{class_name}'.")
        return None

    try:
        item["base64"] = image_to_base64(item["image_path"])
    except Exception as e:
        print(f"Error loading image '{item.get('image_path')}': {e}")
        return None

    progress = get_annotation_progress_report(manifest_path)
    html_code = build_annotation_html(item, progress, manifest_csv=manifest_path, debug_ui=debug_ui)

    try:
        from IPython.display import HTML, display
        html_obj = HTML(html_code)
        display(html_obj)
        return html_obj
    except ImportError:
        print(f"[LOCAL RUNTIME] Loaded pending image: {item['image_path']}")
        return html_code


def get_annotation_progress(manifest_csv: Optional[str | Path] = None) -> Dict[str, Any]:
    """Public helper returning current progress dictionary."""
    return get_annotation_progress_report(manifest_csv)
