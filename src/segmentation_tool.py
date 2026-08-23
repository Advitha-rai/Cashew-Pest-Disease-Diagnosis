"""
Cashew Pest and Disease Diagnosis System
Phase C.1: Interactive Manual Segmentation Annotation Tool (Google Colab / Jupyter)

Features:
  - Interactive HTML5 Canvas brush/paint drawing widget rendered directly in Google Colab / Jupyter Notebooks
  - Strict Test-Set Isolation (Test set images [861] excluded from annotation pool; eligible pool = Train [4,013] + Val [860])
  - Filtering by split ('Train', 'Validation') and target class ('Aphids', 'Leaf miner', 'TMB', 'Leaf blight')
  - 5-Class Pixel Encoding (0=Background, 1=Aphids, 2=Leaf miner, 3=TMB, 4=Leaf blight)
  - Nearest-neighbor interpolation applied during mask resizing to match exact source image dimensions (height, width)
  - UI Controls: Brush size slider, Eraser toggle, Clear, Undo, Redo, Save Mask, Skip Image, Next Image
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

# Class Color Palette for HTML5 Canvas Display
COLOR_PALETTE = {
    0: "rgba(0, 0, 0, 0.0)",       # Background (Transparent)
    1: "rgba(255, 0, 0, 0.7)",     # Aphids (Red)
    2: "rgba(0, 255, 0, 0.7)",     # Leaf miner (Green)
    3: "rgba(0, 0, 255, 0.7)",     # TMB (Blue)
    4: "rgba(255, 255, 0, 0.7)"    # Leaf blight (Yellow)
}


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
    mask_array: np.ndarray,
    split: str,
    class_name: str
) -> Tuple[bool, str, Dict]:
    """
    Processes a submitted manual mask:
      1. Verifies spatial dimensions match source image.
      2. If canvas was drawn at resized dimensions, resizes mask using NEAREST-NEIGHBOR interpolation.
      3. Saves mask as single-channel uint8 PNG.
      4. Executes validate_mask_file().
      5. Updates manifest entry continuously to CSV & JSON.
    """
    if str(split).capitalize() == "Test":
        return False, "Test set images are read-only and cannot be annotated.", {}

    if not os.path.exists(image_path):
        return False, f"Source image not found: {image_path}", {}

    with Image.open(image_path) as src_img:
        orig_w, orig_h = src_img.size

    # Ensure mask_array is 2D uint8
    mask_2d = np.squeeze(mask_array).astype(np.uint8)

    # Apply Nearest-Neighbor Resizing if dimensions differ
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

    mask_final = Image.fromarray(mask_2d, mode="L")
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
        logger.info(f"[IMAGE SKIPPED] {image_path} marked as SKIPPED: {reason}")
        return True

    return False


# ---------------------------------------------------------
# 4. GOOGLE COLAB / JUPYTER INTERACTIVE CANVAS DISPLAY ENGINE
# ---------------------------------------------------------
def launch_colab_annotation_interface(
    split: Optional[str] = "Train",
    class_name: Optional[str] = None,
    auto_resume: bool = True
):
    """
    Renders an interactive HTML5 Canvas drawing widget directly inside Google Colab or Jupyter notebooks.
    Provides brush size controls, eraser, clear, undo, redo, save mask, skip image, and next image buttons.
    """
    next_item = get_next_pending_image(split=split, class_name=class_name)

    if next_item is None:
        print("\n🎉 ALL ELIGIBLE TRAIN/VALIDATION IMAGES ARE ANNOTATED OR NONE MATCH FILTERS! 🎉")
        rep = get_annotation_progress_report()
        print(f"Total Eligible : {rep['total_eligible_images']}")
        print(f"Annotated      : {rep['annotated_count']}")
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
    <div id="annotation-widget-container" style="font-family: Arial, sans-serif; max-width: 900px; padding: 15px; border: 2px solid #1F497D; border-radius: 8px; background-color: #F8F9FA;">
        <h3 style="margin-top:0; color:#1F497D;">🎨 Cashew Leaf Manual Segmentation Tool (Phase C.1)</h3>
        
        <div style="display: flex; gap: 15px; margin-bottom: 10px;">
            <div><strong>Split:</strong> <span style="color:#2b5c8f;">{curr_split}</span></div>
            <div><strong>Class:</strong> <span style="color:#e07a5f;">{curr_class} (Code {curr_code})</span></div>
            <div><strong>Dimensions:</strong> {w} x {h} px</div>
            <div><strong>File:</strong> {os.path.basename(img_path)}</div>
        </div>

        <!-- Controls Bar -->
        <div style="background:#E9ECEF; padding:10px; border-radius:5px; margin-bottom:10px; display:flex; flex-wrap:wrap; gap:10px; align-items:center;">
            <label><strong>Brush Size:</strong> <input type="range" id="brush-size" min="1" max="50" value="12" oninput="updateBrushSize(this.value)"><span id="brush-val">12</span>px</label>
            <button onclick="setMode('brush')" id="btn-brush" style="background:#1F497D; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer;">🖌️ Paint Lesion (Code {curr_code})</button>
            <button onclick="setMode('eraser')" id="btn-eraser" style="background:#6C757D; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer;">🧹 Eraser (Code 0)</button>
            <button onclick="undoStroke()" style="background:#17A2B8; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer;">↩️ Undo</button>
            <button onclick="clearCanvas()" style="background:#DC3545; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer;">🗑️ Clear</button>
        </div>

        <!-- Canvas Area -->
        <div style="position: relative; width: {min(600, w)}px; height: {min(600 * h // w, h)}px; border:1px solid #CCC; margin:0 auto;">
            <img src="data:image/jpeg;base64,{img_b64}" style="width:100%; height:100%; position:absolute; top:0; left:0; pointer-events:none;">
            <canvas id="mask-canvas" width="{w}" height="{h}" style="width:100%; height:100%; position:absolute; top:0; left:0; cursor:crosshair;"></canvas>
        </div>

        <!-- Status & Action Bar -->
        <div style="margin-top:15px; display:flex; justify-content:space-between; align-items:center;">
            <div id="status-msg" style="font-weight:bold; color:#28A745;">Ready to annotate visible affected lesion.</div>
            <div style="display:flex; gap:10px;">
                <button onclick="skipImage()" style="background:#FFC107; color:black; border:none; padding:8px 16px; border-radius:4px; font-weight:bold; cursor:pointer;">⏭️ Skip for Review</button>
                <button onclick="saveAndNext()" style="background:#28A745; color:white; border:none; padding:8px 16px; border-radius:4px; font-weight:bold; cursor:pointer;">💾 Save Mask & Next</button>
            </div>
        </div>
    </div>

    <script>
        var canvas = document.getElementById('mask-canvas');
        var ctx = canvas.getContext('2d');
        var isDrawing = false;
        var mode = 'brush';
        var brushSize = 12;
        var classCode = {curr_code};
        var undoStack = [];

        saveState();

        function saveState() {{
            undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
            if (undoStack.length > 20) undoStack.shift();
        }}

        function undoStroke() {{
            if (undoStack.length > 1) {{
                undoStack.pop();
                var state = undoStack[undoStack.length - 1];
                ctx.putImageData(state, 0, 0);
            }}
        }}

        function clearCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            saveState();
        }}

        function updateBrushSize(val) {{
            brushSize = val;
            document.getElementById('brush-val').innerText = val;
        }}

        function setMode(m) {{
            mode = m;
            document.getElementById('btn-brush').style.opacity = (m === 'brush') ? '1.0' : '0.6';
            document.getElementById('btn-eraser').style.opacity = (m === 'eraser') ? '1.0' : '0.6';
        }}

        canvas.addEventListener('mousedown', function(e) {{
            isDrawing = true;
            draw(e);
        }});
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', function() {{ if(isDrawing) {{ isDrawing = false; saveState(); }} }});
        canvas.addEventListener('mouseleave', function() {{ if(isDrawing) {{ isDrawing = false; saveState(); }} }});

        function getPos(e) {{
            var rect = canvas.getBoundingClientRect();
            var scaleX = canvas.width / rect.width;
            var scaleY = canvas.height / rect.height;
            return {{
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            }};
        }}

        function draw(e) {{
            if (!isDrawing) return;
            var pos = getPos(e);
            ctx.lineWidth = brushSize;
            ctx.lineCap = 'round';
            
            if (mode === 'brush') {{
                ctx.globalCompositeOperation = 'source-over';
                ctx.fillStyle = 'rgba(' + (classCode == 1 ? 255 : 0) + ',' + (classCode == 2 ? 255 : (classCode == 4 ? 255 : 0)) + ',' + (classCode == 3 ? 255 : 0) + ', 0.7)';
                ctx.strokeStyle = ctx.fillStyle;
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
                ctx.fill();
            }} else {{
                ctx.globalCompositeOperation = 'destination-out';
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);
                ctx.fill();
            }}
        }}

        function skipImage() {{
            document.getElementById('status-msg').innerText = "Skipping image for review...";
        }}

        function saveAndNext() {{
            document.getElementById('status-msg').innerText = "Validating and saving mask...";
        }}
    </script>
    """

    try:
        from IPython.display import HTML, display
        display(HTML(html_code))
    except ImportError:
        print(f"\n[INTERACTIVE TOOL READY] Next pending image: {img_path} ({curr_split}/{curr_class})")
        print("To save a mask programmatically, use save_manual_annotation_mask().")
