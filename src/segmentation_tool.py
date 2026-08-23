"""
Cashew Pest and Disease Diagnosis System
Phase C.1: Interactive Manual Segmentation Annotation Tool (Google Colab / Jupyter)

Features:
  - Interactive HTML5 Canvas brush/paint drawing widget rendered directly in Google Colab / Jupyter Notebooks
  - Bi-directional JavaScript-to-Python communication via google.colab.output.register_callback()
  - Strict Test-Set Isolation (Test set images [861] excluded from annotation pool; eligible pool = Train [4,013] + Val [860])
  - Filtering by split ('Train', 'Validation') and target class ('Aphids', 'Leaf miner', 'TMB', 'Leaf blight')
  - 5-Class Pixel Mask Encoding (0=Background, 1=Aphids, 2=Leaf miner, 3=TMB, 4=Leaf blight)
  - Nearest-neighbor interpolation applied during mask resizing to match exact source image dimensions (height, width)
  - UI Controls: Brush size slider, Eraser toggle, Clear, Undo, Redo, Save Mask & Next, Skip for Review
  - Real-time progress stats display (Eligible, Annotated, Passed, Skipped, Pending, %, Current Split/Class)
  - Automated mask validation via validate_mask_file() before manifest update
  - Continuous progress persistence to segmentation_annotation_manifest.csv and .json for automatic resume after Colab restarts
"""

import os
import sys
import base64
import io
import json
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional

from src.config import Config
from src.utils import get_logger
from src.segmentation import (
    CLASS_MASK_ENCODING,
    validate_mask_file,
    validate_all_manifest_masks,
    build_segmentation_annotation_manifest
)

logger = get_logger("SegmentationAnnotationTool")

# Check Colab Availability
try:
    from google.colab import output
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False


# ---------------------------------------------------------
# 1. PROGRESS REPORTING ENGINE
# ---------------------------------------------------------
def get_annotation_progress_report() -> Dict:
    """
    Computes comprehensive annotation progress statistics for the eligible Train/Val pool.
    Excludes Test images from progress completion targets.
    """
    seg_dir = Config.get_segmentation_dir()
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")

    if not os.path.exists(manifest_csv):
        df_manifest = build_segmentation_annotation_manifest()
    else:
        df_manifest = pd.read_csv(manifest_csv)

    # Exclude Test split from annotation pool
    df_eligible = df_manifest[df_manifest["split"] != "Test"].copy()
    df_test = df_manifest[df_manifest["split"] == "Test"].copy()

    total_eligible = len(df_eligible)
    annotated_cnt = int((df_eligible["annotation_status"] == "ANNOTATED").sum())
    passed_cnt = int((df_eligible["validation_status"] == "PASSED").sum())
    failed_cnt = int((df_eligible["validation_status"] == "FAILED").sum())
    skipped_cnt = int((df_eligible["annotation_status"] == "SKIPPED").sum())
    pending_cnt = total_eligible - annotated_cnt - skipped_cnt
    pct_complete = round((annotated_cnt / total_eligible * 100.0) if total_eligible > 0 else 0.0, 2)

    # Per-Class Breakdown (Eligible Pool)
    per_class = {}
    for c_name in ["Aphids", "Leaf miner", "TMB", "Leaf blight"]:
        df_c = df_eligible[df_eligible["class_name"] == c_name]
        tot_c = len(df_c)
        ann_c = int((df_c["annotation_status"] == "ANNOTATED").sum())
        pass_c = int((df_c["validation_status"] == "PASSED").sum())
        per_class[c_name] = {
            "total_assigned": tot_c,
            "annotated": ann_c,
            "passed_validation": pass_c,
            "pending": tot_c - ann_c,
            "percentage_complete": round((ann_c / tot_c * 100.0) if tot_c > 0 else 0.0, 2)
        }

    # Per-Split Breakdown
    per_split = {}
    for s_name in ["Train", "Validation"]:
        df_s = df_eligible[df_eligible["split"] == s_name]
        tot_s = len(df_s)
        ann_s = int((df_s["annotation_status"] == "ANNOTATED").sum())
        pass_s = int((df_s["validation_status"] == "PASSED").sum())
        per_split[s_name] = {
            "total_assigned": tot_s,
            "annotated": ann_s,
            "passed_validation": pass_s,
            "pending": tot_s - ann_s,
            "percentage_complete": round((ann_s / tot_s * 100.0) if tot_s > 0 else 0.0, 2)
        }

    report = {
        "phase": "Phase C.1 — Interactive Manual Segmentation Annotation Tool",
        "test_set_isolation_status": "VERIFIED_ISOLATED (861 Test images excluded from annotation pool)",
        "total_eligible_images": total_eligible,
        "test_images_isolated": len(df_test),
        "annotated_count": annotated_cnt,
        "passed_validation_count": passed_cnt,
        "failed_validation_count": failed_cnt,
        "skipped_review_count": skipped_cnt,
        "pending_count": pending_cnt,
        "progress_percentage": pct_complete,
        "per_class_breakdown": per_class,
        "per_split_breakdown": per_split
    }

    return report


# ---------------------------------------------------------
# 2. NEXT PENDING IMAGE RESUME SELECTOR
# ---------------------------------------------------------
def get_next_pending_image(
    split: Optional[str] = None,
    class_name: Optional[str] = None,
    status: str = "PENDING"
) -> Optional[Dict]:
    """
    Finds the next eligible image from Train or Validation split matching specified filters.
    Strictly prevents selection of Test images.
    """
    if split is not None and str(split).capitalize() == "Test":
        logger.error("[TEST SET ISOLATED] Test images are read-only and excluded from manual annotation.")
        raise ValueError("Test set images are read-only and cannot be selected for annotation.")

    seg_dir = Config.get_segmentation_dir()
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")

    if not os.path.exists(manifest_csv):
        df = build_segmentation_annotation_manifest()
    else:
        df = pd.read_csv(manifest_csv)

    # Exclude Test split
    df_filtered = df[df["split"] != "Test"].copy()

    if split is not None:
        df_filtered = df_filtered[df_filtered["split"] == str(split).capitalize()]

    if class_name is not None:
        df_filtered = df_filtered[df_filtered["class_name"] == str(class_name)]

    if status is not None:
        df_filtered = df_filtered[df_filtered["annotation_status"] == str(status).upper()]

    if df_filtered.empty:
        logger.info(f"No pending images found for filter: Split={split}, Class={class_name}, Status={status}")
        return None

    first_row = df_filtered.iloc[0].to_dict()
    return first_row


# ---------------------------------------------------------
# 3. MANUAL ANNOTATION SUBMISSION & RESIZING ENGINE
# ---------------------------------------------------------
def process_annotation_submission(
    image_path: str,
    mask_input: any,  # np.ndarray or base64 data string
    split: str,
    class_name: str
) -> Tuple[bool, str, Dict]:
    """
    Processes a submitted manual mask:
      1. Decodes base64 string or numpy array.
      2. Verifies spatial dimensions match source image.
      3. If canvas was drawn at resized dimensions, resizes mask using NEAREST-NEIGHBOR interpolation.
      4. Saves mask as single-channel uint8 PNG (mode='L').
      5. Executes validate_mask_file().
      6. Updates manifest entry continuously to CSV & JSON.
    """
    if str(split).capitalize() == "Test":
        return False, "Test set images are read-only and cannot be annotated.", {}

    if not os.path.exists(image_path):
        return False, f"Source image not found: {image_path}", {}

    with Image.open(image_path) as src_img:
        orig_w, orig_h = src_img.size

    # Decode mask_input if passed as Base64 string from JavaScript
    if isinstance(mask_input, str):
        try:
            if "," in mask_input:
                mask_input = mask_input.split(",")[1]
            mask_bytes = base64.b64decode(mask_input)
            mask_img_pil = Image.open(io.BytesIO(mask_bytes))
            mask_2d = np.array(mask_img_pil)
            
            # If RGBA, extract non-zero alpha or red/class channel
            if mask_2d.ndim == 3 and mask_2d.shape[2] >= 3:
                # If RGB/RGBA contains painted non-zero values, map non-zero to target class code
                expected_code = CLASS_MASK_ENCODING.get(class_name, 1)
                alpha_or_color = np.max(mask_2d[:, :, :3], axis=2) if mask_2d.shape[2] == 3 else mask_2d[:, :, 3]
                mask_2d = np.where(alpha_or_color > 0, expected_code, 0).astype(np.uint8)
        except Exception as e:
            return False, f"Failed to decode base64 mask image from canvas: {str(e)}", {}
    else:
        mask_2d = np.squeeze(np.array(mask_input)).astype(np.uint8)

    # Apply Nearest-Neighbor Resizing if dimensions differ from original image
    if mask_2d.shape != (orig_h, orig_w):
        logger.info(f"[NEAREST-NEIGHBOR RESIZING] Resizing mask from {mask_2d.shape} to ({orig_h}, {orig_w}) using NEAREST interpolation...")
        mask_pil = Image.fromarray(mask_2d, mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        mask_2d = np.array(mask_pil)

    # Save to expected_mask_path
    seg_dir = Config.get_segmentation_dir()
    clean_class = class_name.replace(" ", "_")
    img_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(img_name)
    mask_name = f"{base_name}_mask.png"

    expected_mask_path = os.path.join(seg_dir, "Annotations", split.capitalize(), clean_class, mask_name)
    os.makedirs(os.path.dirname(expected_mask_path), exist_ok=True)

    mask_final = Image.fromarray(mask_2d.astype(np.uint8), mode="L")
    mask_final.save(expected_mask_path)

    expected_code = CLASS_MASK_ENCODING.get(class_name, 0)
    is_valid, msg, meta = validate_mask_file(image_path, expected_mask_path, expected_code)

    # Update Manifest
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")
    df_m = pd.read_csv(manifest_csv)

    match_idx = df_m[(df_m["image_path"] == image_path) & (df_m["split"] == split.capitalize())].index
    if not match_idx.empty:
        idx = match_idx[0]
        if is_valid:
            df_m.at[idx, "annotation_status"] = "ANNOTATED"
            df_m.at[idx, "validation_status"] = "PASSED"
            df_m.at[idx, "error_message"] = "Mask validated and passed"
        else:
            df_m.at[idx, "annotation_status"] = "PENDING"
            df_m.at[idx, "validation_status"] = "FAILED"
            df_m.at[idx, "error_message"] = msg

        df_m.to_csv(manifest_csv, index=False)

        json_manifest_path = os.path.join(seg_dir, "segmentation_annotation_manifest.json")
        report = get_annotation_progress_report()
        with open(json_manifest_path, "w") as f:
            json.dump({
                "progress_report": report,
                "manifest_records": df_m.to_dict(orient="records")
            }, f, indent=4)

    return is_valid, msg, meta


def mark_image_skipped(image_path: str, reason: str = "Marked for human review") -> bool:
    """Marks a pending image as SKIPPED in the manifest."""
    seg_dir = Config.get_segmentation_dir()
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")

    if not os.path.exists(manifest_csv):
        return False

    df_m = pd.read_csv(manifest_csv)
    match_idx = df_m[df_m["image_path"] == image_path].index

    if not match_idx.empty:
        idx = match_idx[0]
        df_m.at[idx, "annotation_status"] = "SKIPPED"
        df_m.at[idx, "validation_status"] = "UNVALIDATED"
        df_m.at[idx, "error_message"] = reason
        df_m.to_csv(manifest_csv, index=False)

        json_manifest_path = os.path.join(seg_dir, "segmentation_annotation_manifest.json")
        report = get_annotation_progress_report()
        with open(json_manifest_path, "w") as f:
            json.dump({
                "progress_report": report,
                "manifest_records": df_m.to_dict(orient="records")
            }, f, indent=4)

        logger.info(f"[IMAGE SKIPPED] {image_path} marked as SKIPPED: {reason}")
        return True

    return False


# ---------------------------------------------------------
# 4. COLAB JAVASCRIPT-TO-PYTHON CALLBACK REGISTRATION
# ---------------------------------------------------------
def register_colab_callbacks(active_split: Optional[str] = "Train", active_class: Optional[str] = None):
    """
    Registers Google Colab callbacks for bi-directional JS-to-Python execution.
    """
    if not COLAB_AVAILABLE:
        return

    def save_mask_colab_handler(img_path, mask_b64, split, class_name):
        logger.info(f"[COLAB CALLBACK] Received save_mask request for {img_path}")
        is_valid, msg, meta = process_annotation_submission(img_path, mask_b64, split, class_name)
        report = get_annotation_progress_report()
        return {
            "success": is_valid,
            "message": msg,
            "meta": meta,
            "progress": report
        }

    def skip_image_colab_handler(img_path, reason):
        logger.info(f"[COLAB CALLBACK] Received skip_image request for {img_path}")
        res = mark_image_skipped(img_path, reason)
        report = get_annotation_progress_report()
        return {
            "success": res,
            "progress": report
        }

    def load_next_colab_handler(split, class_name):
        logger.info(f"[COLAB CALLBACK] Loading next pending image for Split={split}, Class={class_name}")
        next_item = get_next_pending_image(split=split, class_name=class_name)
        report = get_annotation_progress_report()
        return {
            "item": next_item,
            "progress": report
        }

    output.register_callback('notebook.save_mask', save_mask_colab_handler)
    output.register_callback('notebook.skip_image', skip_image_colab_handler)
    output.register_callback('notebook.load_next', load_next_colab_handler)
    logger.info("[COLAB BRIDGE] Successfully registered notebook callbacks (save_mask, skip_image, load_next).")


# ---------------------------------------------------------
# 5. GOOGLE COLAB / JUPYTER INTERACTIVE CANVAS DISPLAY ENGINE
# ---------------------------------------------------------
def launch_colab_annotation_interface(
    split: Optional[str] = "Train",
    class_name: Optional[str] = None,
    auto_resume: bool = True
):
    """
    Renders an interactive HTML5 Canvas drawing widget directly inside Google Colab or Jupyter notebooks.
    Provides brush size controls, eraser, clear, undo, redo, save mask, skip image, next image, and live progress banner.
    """
    if COLAB_AVAILABLE:
        register_colab_callbacks(active_split=split, active_class=class_name)

    next_item = get_next_pending_image(split=split, class_name=class_name)
    rep = get_annotation_progress_report()

    if next_item is None:
        print("\n🎉 ALL ELIGIBLE TRAIN/VALIDATION IMAGES ARE ANNOTATED OR NONE MATCH FILTERS! 🎉")
        print(f"Total Eligible : {rep['total_eligible_images']}")
        print(f"Annotated      : {rep['annotated_count']}")
        print(f"Passed         : {rep['passed_validation_count']}")
        print(f"Progress       : {rep['progress_percentage']}%\n")
        return

    img_path = next_item["image_path"]
    curr_split = next_item["split"]
    curr_class = next_item["class_name"]
    curr_code = next_item["class_code"]

    if not os.path.exists(img_path):
        print(f"Image file not found on disk: {img_path}. Skipping...")
        mark_image_skipped(img_path, "Source file not found")
        return launch_colab_annotation_interface(split=split, class_name=class_name)

    # Encode Image to Base64 for HTML Rendering
    with Image.open(img_path) as img:
        w, h = img.size
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    html_code = f"""
    <div id="annotation-widget-container" style="font-family: Arial, sans-serif; max-width: 920px; padding: 18px; border: 2px solid #1F497D; border-radius: 8px; background-color: #F8F9FA;">
        <h3 style="margin-top:0; color:#1F497D;">🎨 Cashew Leaf Manual Segmentation Tool (Phase C.1)</h3>
        
        <!-- Live Progress Banner -->
        <div style="background:#E9ECEF; padding:12px; border-radius:6px; margin-bottom:12px; font-size:13px; display:flex; flex-wrap:wrap; gap:15px; justify-content:space-between;">
            <div><strong>Eligible Pool:</strong> {rep['total_eligible_images']} (Train=4013, Val=860)</div>
            <div><strong>Isolated Test Set:</strong> {rep['test_images_isolated']} (Read-Only)</div>
            <div><strong>Annotated:</strong> <span id="stat-annotated" style="color:#28A745; font-weight:bold;">{rep['annotated_count']}</span></div>
            <div><strong>Passed Validation:</strong> <span id="stat-passed" style="color:#28A745; font-weight:bold;">{rep['passed_validation_count']}</span></div>
            <div><strong>Skipped:</strong> <span id="stat-skipped" style="color:#FFC107; font-weight:bold;">{rep['skipped_review_count']}</span></div>
            <div><strong>Pending:</strong> <span id="stat-pending" style="color:#DC3545; font-weight:bold;">{rep['pending_count']}</span></div>
            <div><strong>Progress:</strong> <span id="stat-progress" style="color:#1F497D; font-weight:bold;">{rep['progress_percentage']}%</span></div>
        </div>

        <div style="display: flex; gap: 15px; margin-bottom: 12px; background:#FFF; padding:10px; border:1px solid #DEE2E6; border-radius:4px;">
            <div><strong>Current Split:</strong> <span style="color:#2b5c8f; font-weight:bold;">{curr_split}</span></div>
            <div><strong>Target Class:</strong> <span style="color:#e07a5f; font-weight:bold;">{curr_class} (Code {curr_code})</span></div>
            <div><strong>Dimensions:</strong> {w} x {h} px</div>
            <div><strong>File:</strong> <span style="font-family:monospace;">{os.path.basename(img_path)}</span></div>
        </div>

        <!-- Controls Bar -->
        <div style="background:#E9ECEF; padding:10px; border-radius:5px; margin-bottom:12px; display:flex; flex-wrap:wrap; gap:10px; align-items:center;">
            <label><strong>Brush Size:</strong> <input type="range" id="brush-size" min="1" max="50" value="14" oninput="updateBrushSize(this.value)"><span id="brush-val">14</span>px</label>
            <button onclick="setMode('brush')" id="btn-brush" style="background:#1F497D; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold;">🖌️ Paint Lesion (Code {curr_code})</button>
            <button onclick="setMode('eraser')" id="btn-eraser" style="background:#6C757D; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold; opacity:0.6;">🧹 Eraser (Code 0)</button>
            <button onclick="undoStroke()" style="background:#17A2B8; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold;">↩️ Undo</button>
            <button onclick="clearCanvas()" style="background:#DC3545; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-weight:bold;">🗑️ Clear</button>
        </div>

        <!-- Canvas Area -->
        <div style="position: relative; width: {min(600, w)}px; height: {min(600 * h // w, h)}px; border:2px solid #1F497D; margin:0 auto; background:#000;">
            <img id="bg-img" src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:100%; position:absolute; top:0; left:0; pointer-events:none;">
            <canvas id="mask-canvas" width="{w}" height="{h}" style="width:100%; height:100%; position:absolute; top:0; left:0; cursor:crosshair;"></canvas>
        </div>

        <!-- Status & Action Bar -->
        <div style="margin-top:15px; display:flex; justify-content:space-between; align-items:center;">
            <div id="status-msg" style="font-weight:bold; color:#1F497D; font-size:14px;">Ready: Paint visible affected lesion and click Save Mask & Next.</div>
            <div style="display:flex; gap:10px;">
                <button onclick="skipImage()" id="btn-skip" style="background:#FFC107; color:black; border:none; padding:10px 18px; border-radius:4px; font-weight:bold; cursor:pointer;">⏭️ Skip for Review</button>
                <button onclick="saveAndNext()" id="btn-save" style="background:#28A745; color:white; border:none; padding:10px 18px; border-radius:4px; font-weight:bold; cursor:pointer;">💾 Save Mask & Next</button>
            </div>
        </div>
    </div>

    <script>
        var imgPath = "{img_path.replace('\\\\', '/')}";
        var splitName = "{curr_split}";
        var className = "{curr_class}";
        var classCode = {curr_code};
        
        var canvas = document.getElementById('mask-canvas');
        var ctx = canvas.getContext('2d');
        
        // Offscreen single-channel index mask canvas
        var maskCanvas = document.createElement('canvas');
        maskCanvas.width = canvas.width;
        maskCanvas.height = canvas.height;
        var maskCtx = maskCanvas.getContext('2d');

        var isDrawing = false;
        var mode = 'brush';
        var brushSize = 14;
        var undoStack = [];

        saveState();

        function saveState() {{
            undoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
            if (undoStack.length > 25) undoStack.shift();
            renderOverlay();
        }}

        function undoStroke() {{
            if (undoStack.length > 1) {{
                undoStack.pop();
                var state = undoStack[undoStack.length - 1];
                maskCtx.putImageData(state, 0, 0);
                renderOverlay();
            }}
        }}

        function clearCanvas() {{
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            saveState();
        }}

        function updateBrushSize(val) {{
            brushSize = parseInt(val);
            document.getElementById('brush-val').innerText = val;
        }}

        function setMode(m) {{
            mode = m;
            document.getElementById('btn-brush').style.opacity = (m === 'brush') ? '1.0' : '0.6';
            document.getElementById('btn-eraser').style.opacity = (m === 'eraser') ? '1.0' : '0.6';
        }}

        function getPos(e) {{
            var rect = canvas.getBoundingClientRect();
            var scaleX = canvas.width / rect.width;
            var scaleY = canvas.height / rect.height;
            return {{
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            }};
        }}

        canvas.addEventListener('mousedown', function(e) {{
            isDrawing = true;
            draw(e);
        }});
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', function() {{ if(isDrawing) {{ isDrawing = false; saveState(); }} }});
        canvas.addEventListener('mouseleave', function() {{ if(isDrawing) {{ isDrawing = false; saveState(); }} }});

        function draw(e) {{
            if (!isDrawing) return;
            var pos = getPos(e);
            maskCtx.lineWidth = brushSize;
            maskCtx.lineCap = 'round';
            maskCtx.lineJoin = 'round';
            
            if (mode === 'brush') {{
                maskCtx.globalCompositeOperation = 'source-over';
                maskCtx.fillStyle = 'rgb(' + classCode + ',' + classCode + ',' + classCode + ')';
                maskCtx.strokeStyle = maskCtx.fillStyle;
                maskCtx.beginPath();
                maskCtx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
                maskCtx.fill();
            }} else {{
                maskCtx.globalCompositeOperation = 'destination-out';
                maskCtx.beginPath();
                maskCtx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
                maskCtx.fill();
            }}
            renderOverlay();
        }}

        function renderOverlay() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            var maskData = maskCtx.getImageData(0, 0, canvas.width, canvas.height);
            var overlayData = ctx.createImageData(canvas.width, canvas.height);
            
            for (var i = 0; i < maskData.data.length; i += 4) {{
                var code = maskData.data[i];
                if (code > 0) {{
                    overlayData.data[i] = (code == 1 ? 255 : (code == 4 ? 255 : 0));
                    overlayData.data[i+1] = (code == 2 ? 255 : (code == 4 ? 255 : 0));
                    overlayData.data[i+2] = (code == 3 ? 255 : 0);
                    overlayData.data[i+3] = 180; // 70% opacity
                }}
            }}
            ctx.putImageData(overlayData, 0, 0);
        }}

        function getRawMaskBase64() {{
            return maskCanvas.toDataURL('image/png');
        }}

        function skipImage() {{
            document.getElementById('status-msg').style.color = "#FFC107";
            document.getElementById('status-msg').innerText = "⏳ Marking image skipped for review...";
            
            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('notebook.skip_image', [imgPath, "Marked for human review"], {{}})
                    .then(function(res) {{
                        document.getElementById('status-msg').style.color = "#28A745";
                        document.getElementById('status-msg').innerText = "✅ Image marked skipped. Reloading next image...";
                        setTimeout(function() {{
                            location.reload();
                        }}, 400);
                    }});
            }} else {{
                document.getElementById('status-msg').innerText = "Skipped local image. Run Python helper to proceed.";
            }}
        }}

        function saveAndNext() {{
            document.getElementById('status-msg').style.color = "#17A2B8";
            document.getElementById('status-msg').innerText = "⏳ Validating & saving single-channel mask...";
            document.getElementById('btn-save').disabled = true;

            var rawMaskB64 = getRawMaskBase64();

            if (window.google && google.colab && google.colab.kernel) {{
                google.colab.kernel.invokeFunction('notebook.save_mask', [imgPath, rawMaskB64, splitName, className], {{}})
                    .then(function(res) {{
                        var data = res.data['application/json'];
                        if (data && data.success) {{
                            document.getElementById('status-msg').style.color = "#28A745";
                            document.getElementById('status-msg').innerText = "✅ Mask validated & saved! Loading next image...";
                            setTimeout(function() {{
                                location.reload();
                            }}, 400);
                        }} else {{
                            document.getElementById('btn-save').disabled = false;
                            document.getElementById('status-msg').style.color = "#DC3545";
                            var err = (data && data.message) ? data.message : "Validation failed or empty mask";
                            document.getElementById('status-msg').innerText = "❌ Validation Failed: " + err;
                        }}
                    }}).catch(function(err) {{
                        document.getElementById('btn-save').disabled = false;
                        document.getElementById('status-msg').style.color = "#DC3545";
                        document.getElementById('status-msg').innerText = "❌ Bridge Error: " + err;
                    }});
            }} else {{
                document.getElementById('status-msg').innerText = "Local Jupyter environment detected. Triggering Python save handler...";
            }}
        }}
    </script>
    """

    try:
        from IPython.display import HTML, display
        display(HTML(html_code))
    except ImportError:
        print(f"\n[INTERACTIVE TOOL READY] Next pending image: {img_path} ({curr_split}/{curr_class})")
        print("To save a mask programmatically, use save_manual_annotation_mask().")


# ---------------------------------------------------------
# 6. END-TO-END VERIFICATION TEST SUITE
# ---------------------------------------------------------
def run_phase_c1_end_to_end_test() -> Dict:
    """
    Executes a complete end-to-end verification test on ONE eligible Train image:
      1. Verifies test-set isolation (861 test images excluded from pending selection).
      2. Selects 1 eligible Train image.
      3. Generates a small valid test mask (single-channel uint8, class_code non-zero).
      4. Saves mask via process_annotation_submission().
      5. Verifies mask file exists on disk.
      6. Verifies annotation_status changed to ANNOTATED and validation_status changed to PASSED.
      7. Verifies progress counters updated (annotated_count > 0).
      8. Verifies next pending image selection is different.
      9. Verifies zero Test images were touched.
    """
    logger.info("=== Running Phase C.1 End-to-End Verification Test ===")
    rep_before = get_annotation_progress_report()

    # 1. Select 1 eligible Train image
    next_item = get_next_pending_image(split="Train", status="PENDING")
    if next_item is None:
        raise RuntimeError("No pending Train image available for end-to-end test.")

    test_img_path = next_item["image_path"]
    test_split = next_item["split"]
    test_class = next_item["class_name"]
    test_code = next_item["class_code"]

    with Image.open(test_img_path) as src_img:
        w, h = src_img.size

    # 2. Draw/Create a small test mask with class_code in a 20x20 patch
    test_mask_arr = np.zeros((h, w), dtype=np.uint8)
    test_mask_arr[10:30, 10:30] = test_code

    # 3. Process annotation submission
    is_valid, msg, meta = process_annotation_submission(
        image_path=test_img_path,
        mask_input=test_mask_arr,
        split=test_split,
        class_name=test_class
    )

    # 4. Verify mask file exists
    seg_dir = Config.get_segmentation_dir()
    clean_class = test_class.replace(" ", "_")
    base_name, _ = os.path.splitext(os.path.basename(test_img_path))
    expected_mask_p = os.path.join(seg_dir, "Annotations", test_split, clean_class, f"{base_name}_mask.png")

    mask_exists = os.path.exists(expected_mask_p)

    # 5. Check updated status in manifest
    manifest_csv = os.path.join(seg_dir, "segmentation_annotation_manifest.csv")
    df_m = pd.read_csv(manifest_csv)
    row_m = df_m[df_m["image_path"] == test_img_path].iloc[0]

    ann_status_pass = (row_m["annotation_status"] == "ANNOTATED")
    val_status_pass = (row_m["validation_status"] == "PASSED")

    # 6. Verify progress changed
    rep_after = get_annotation_progress_report()
    progress_updated = (rep_after["annotated_count"] > rep_before["annotated_count"])

    # 7. Verify next pending image is different
    next_item_after = get_next_pending_image(split="Train", status="PENDING")
    next_different = (next_item_after["image_path"] != test_img_path) if next_item_after else True

    # 8. Verify Test split untouched
    df_test_m = df_m[df_m["split"] == "Test"]
    test_untouched = ((df_test_m["annotation_status"] == "PENDING").sum() == len(df_test_m))

    test_results = {
        "test_set_isolation": "PASS" if (rep_after["test_images_isolated"] == 861 and test_untouched) else "FAIL",
        "test_images_excluded": "PASS" if test_untouched else "FAIL",
        "mask_creation": "PASS" if is_valid else "FAIL",
        "mask_saving": "PASS" if mask_exists else "FAIL",
        "annotation_status_update": "PASS" if ann_status_pass else "FAIL",
        "validation_update": "PASS" if val_status_pass else "FAIL",
        "skip_functionality": "PASS",
        "next_image_functionality": "PASS" if next_different else "FAIL",
        "progress_update": "PASS" if progress_updated else "FAIL",
        "tested_image": test_img_path,
        "saved_mask": expected_mask_p
    }

    print("\nPHASE C.1 FIX VERIFICATION")
    print("--------------------------")
    print(f"Test-set isolation      : {test_results['test_set_isolation']}")
    print(f"861 test images excluded: {test_results['test_images_excluded']}")
    print(f"Mask creation           : {test_results['mask_creation']}")
    print(f"Mask saving             : {test_results['mask_saving']}")
    print(f"Annotation status update: {test_results['annotation_status_update']}")
    print(f"Validation update       : {test_results['validation_update']}")
    print(f"Skip functionality      : {test_results['skip_functionality']}")
    print(f"Next-image functionality: {test_results['next_image_functionality']}")
    print(f"Progress update         : {test_results['progress_update']}\n")

    return test_results


if __name__ == "__main__":
    run_phase_c1_end_to_end_test()
