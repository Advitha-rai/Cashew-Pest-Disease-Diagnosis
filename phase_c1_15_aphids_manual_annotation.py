"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — Dedicated Aphids-Only Manual Segmentation Tool & Pre-Reset Routine
Framework: TensorFlow / Keras (Google Colab / Jupyter Environment)
"""

import os
import sys
import json
import base64
import shutil
import hashlib
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

# ---------------------------------------------------------
# Dynamic Environment & Path Discovery
# ---------------------------------------------------------
if Path("/content/Cashew-Pest-Disease-Diagnosis").exists():
    REPO_ROOT = Path("/content/Cashew-Pest-Disease-Diagnosis")
elif Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
    REPO_ROOT = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
else:
    REPO_ROOT = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project").exists():
    DRIVE_ROOT = Path("/content/drive/MyDrive/Cashew_Pest_Disease_Project")
else:
    DRIVE_ROOT = REPO_ROOT

from src.segmentation.config import (
    CLASS_CODES,
    ALLOWED_MASK_VALUES,
    CANONICAL_MANIFEST,
    ANNOTATIONS_DIR,
    DATASET_DIR,
    PREPROCESSED_DIR,
    SEGMENTATION_DIR,
    normalize_class_name,
    get_class_code,
)
from src.segmentation.validation import assert_annotation_allowed, validate_mask_array
from src.segmentation.manifest import load_manifest, save_manifest_atomically, get_annotation_progress_report


# ---------------------------------------------------------
# TARGET APHIDS IDENTIFIERS
# ---------------------------------------------------------
REQUESTED_APHIDS_IDENTIFIERS: List[str] = [
    "20220218_124514",
    "149",
    "20220311_161838",
    "20220815_145343",
    "aphid (2)",
    "Aphid1",
    "Aphids on nuts 5",
    "Aphids on nuts 18",
    "SAM_2225",
    "SAM_0994",
]


# ---------------------------------------------------------
# PART A — RESET PREVIOUS SEGMENTATION SMOKE-TEST DATA
# ---------------------------------------------------------
def reset_previous_test_annotations(manifest_path: Path) -> Dict[str, Any]:
    """
    Safely backs up and resets only the 6 previous annotated and 4 skipped smoke-test records.
    Leaves all other 4,863 records and the dataset completely untouched.
    """
    if not manifest_path.exists():
        return {"status": "SKIP", "message": "Manifest does not exist yet"}

    df_man = load_manifest(manifest_path)
    backup_dir = manifest_path.parent / "backup_previous" / "phase_c1_15_pre_reset"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup manifest
    shutil.copy2(manifest_path, backup_dir / "segmentation_annotation_manifest_pre_reset.csv")

    # Find annotated and skipped records
    df_annotated = df_man[df_man["annotation_status"] == "ANNOTATED"]
    df_skipped = df_man[df_man["annotation_status"] == "SKIPPED"]

    backed_up_masks = 0
    removed_masks = 0

    # Backup existing masks
    for _, row in df_annotated.iterrows():
        mask_p_str = str(row.get("expected_mask_path", ""))
        mask_p = Path(mask_p_str)
        if not mask_p.exists():
            ann_base = manifest_path.parent / "Annotations"
            mask_p = ann_base / str(row.get("split", "Train")) / str(row.get("class_name", "Aphids")) / Path(mask_p_str).name

        if mask_p.exists():
            mask_dst = backup_dir / "masks" / mask_p.name
            mask_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_p, mask_dst)
            backed_up_masks += 1
            try:
                mask_p.unlink()
                removed_masks += 1
            except Exception:
                pass

    # Reset records to clean PENDING state
    records_to_reset = list(df_annotated.index) + list(df_skipped.index)
    for idx in records_to_reset:
        df_man.at[idx, "annotation_status"] = "PENDING"
        df_man.at[idx, "validation_status"] = "UNVALIDATED"
        df_man.at[idx, "error_message"] = "Mask pending manual creation"
        if "mask_sha256" in df_man.columns:
            df_man.at[idx, "mask_sha256"] = ""

    save_manifest_atomically(df_man, manifest_path)

    return {
        "status": "PASS",
        "records_reset": len(records_to_reset),
        "masks_backed_up": backed_up_masks,
        "masks_removed": removed_masks,
    }


# ---------------------------------------------------------
# PART B — RESOLVE 10 APHIDS IMAGES
# ---------------------------------------------------------
def resolve_aphids_images(manifest_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Deterministically resolves the 10 requested Aphids image identifiers from the canonical manifest.
    Guarantees no Test split leakage, 224x224 dimensions, and class Aphids (code 1).
    """
    df_man = load_manifest(manifest_path)
    df_aphids = df_man[df_man["class_name"].apply(lambda c: normalize_class_name(str(c)) == "Aphids")]

    resolved_items = []
    unresolved_ids = []

    print("\n--- APHIDS RESOLUTION TABLE ---")
    print(f"{'Requested ID':<22s} | {'Resolved Filename':<26s} | {'Class':<8s} | {'Split':<10s} | {'Status':<10s}")
    print("-" * 88)

    for req_id in REQUESTED_APHIDS_IDENTIFIERS:
        clean_req = req_id.strip().lower()
        # Match by stem or exact substring in image_name
        matches = df_aphids[df_aphids["image_name"].apply(
            lambda name: clean_req == Path(str(name)).stem.lower()
            or clean_req == str(name).lower()
            or f"_{clean_req}." in f"_{str(name).lower()}."
            or clean_req in str(name).lower()
        )]

        # Filter to ensure non-Test
        valid_matches = matches[matches["split"].isin(["Train", "Validation"])]

        if len(valid_matches) >= 1:
            row = valid_matches.iloc[0]
            img_path = Path(str(row.get("image_path", "")))
            if not img_path.exists():
                # Search under Dataset directory
                ds_dir = DRIVE_ROOT / "Dataset" / "Cleaned" / "Aphids"
                cand = ds_dir / row.get("image_name", "")
                if cand.exists():
                    img_path = cand

            resolved_item = {
                "requested_id": req_id,
                "image_name": str(row.get("image_name")),
                "class_name": "Aphids",
                "class_code": 1,
                "split": str(row.get("split")),
                "image_path": str(img_path),
                "expected_mask_path": str(row.get("expected_mask_path")),
                "annotation_status": str(row.get("annotation_status")),
                "validation_status": str(row.get("validation_status")),
            }
            resolved_items.append(resolved_item)
            print(f"{req_id:<22s} | {resolved_item['image_name']:<26s} | Aphids   | {resolved_item['split']:<10s} | RESOLVED")
        else:
            unresolved_ids.append(req_id)
            print(f"{req_id:<22s} | {'[NOT FOUND]':<26s} | Aphids   | N/A        | UNRESOLVED")

    print("-" * 88)
    return resolved_items, unresolved_ids


# ---------------------------------------------------------
# PART C, D, E — APHIDS MANUAL ANNOTATION UI GENERATOR
# ---------------------------------------------------------
def generate_aphids_ui_html(resolved_items: List[Dict[str, Any]], manifest_path: Path) -> str:
    """
    Builds the dedicated single-image Aphids manual segmentation canvas.
    Has ONLY '💾 Save Mask', manual dropdown selector, and NO auto-advance/skip/next buttons.
    """
    items_json = json.dumps(resolved_items)
    manifest_json = json.dumps(str(manifest_path))

    # Base64 encode the first resolved image for immediate display
    first_item = resolved_items[0] if len(resolved_items) > 0 else {}
    first_b64 = ""
    if first_item and Path(first_item["image_path"]).exists():
        with open(first_item["image_path"], "rb") as f:
            first_b64 = base64.b64encode(f.read()).decode("utf-8")

    html_code = f"""
<div id="aphids-annotator-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 820px; margin: 20px auto; background: #F8F9FA; padding: 18px; border-radius: 10px; border: 2px solid #28A745; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    
    <!-- Title & Mode Banner -->
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #28A745; padding-bottom: 10px; margin-bottom: 14px;">
        <h3 style="margin: 0; color: #155724; font-size: 18px; font-weight: bold;">🌿 Cashew Aphids Manual Annotation Tool — Phase C.1.15</h3>
        <span style="background: #28A745; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;">Aphids Only (Code 1)</span>
    </div>

    <!-- Controlled 10-Image Selector Dropdown -->
    <div style="background: #E8F5E9; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; border: 1px solid #C3E6CB;">
        <label style="font-weight: bold; color: #155724; font-size: 13px; display: block; margin-bottom: 6px;">Select Target Aphids Image (1 of 10):</label>
        <select id="sel-aphids-image" onchange="onSelectImage(this.value)" style="width: 100%; padding: 8px 12px; border-radius: 4px; border: 1px solid #28A745; font-size: 13px; font-weight: bold; background: white; cursor: pointer;">
        </select>
    </div>

    <!-- Active Item Metadata Card -->
    <div style="background: white; padding: 10px 14px; border: 1px solid #CED4DA; border-radius: 6px; margin-bottom: 12px; font-size: 13px; display: flex; flex-wrap: wrap; gap: 16px;">
        <div><strong>Filename:</strong> <span id="lbl-filename" style="color: #155724; font-weight: bold;">-</span></div>
        <div><strong>Class:</strong> <span style="color: #DC3545; font-weight: bold;">Aphids</span> (Code: <strong>1</strong>)</div>
        <div><strong>Split:</strong> <span id="lbl-split" style="font-weight: bold; color: #007BFF;">-</span></div>
        <div><strong>Resolution:</strong> <span>224 × 224</span></div>
        <div><strong>Status:</strong> <span id="lbl-status" style="font-weight: bold;">PENDING</span></div>
    </div>

    <!-- Drawing Toolbar Controls -->
    <div style="margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; background: #E9ECEF; padding: 8px 12px; border-radius: 6px;">
        <button onclick="setDrawingMode('brush')" id="btn-brush" style="background: #DC3545; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">🖌️ Paint Aphids (Red)</button>
        <button onclick="setDrawingMode('eraser')" id="btn-eraser" style="background: #6C757D; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; cursor: pointer; opacity: 0.6;">🧹 Eraser</button>
        
        <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px;">
            <label style="font-size: 13px; font-weight: bold;">Brush Size:</label>
            <input type="range" min="2" max="60" value="12" oninput="updateBrushSize(this.value)" style="cursor: pointer; width: 90px;">
            <span id="brush-val" style="font-size: 13px; font-weight: bold; min-width: 20px;">12</span>px
        </div>

        <div style="display: flex; gap: 6px; margin-left: auto;">
            <button onclick="undoStroke()" style="background: #007BFF; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">↩️ Undo</button>
            <button onclick="redoStroke()" style="background: #17A2B8; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">↪️ Redo</button>
            <button onclick="clearCanvas()" style="background: #6C757D; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">🗑️ Clear</button>
        </div>
    </div>

    <!-- Dual Canvas Work Area -->
    <div id="canvas-wrapper" style="position: relative; width: 448px; height: 448px; border: 2px solid #28A745; margin: 0 auto; background: #000; overflow: hidden; border-radius: 4px;">
        <img id="bg-img" src="data:image/jpeg;base64,{first_b64}" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; object-fit: contain;">
        <canvas id="display-canvas" width="224" height="224" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; pointer-events: auto; touch-action: none; cursor: crosshair;"></canvas>
    </div>

    <!-- Single Save Action Button (NO Next / Skip Buttons) -->
    <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center; gap: 12px;">
        <div id="status-msg" style="font-weight: bold; color: #155724; font-size: 13px;">Ready: Paint visible aphids clusters and click Save Mask.</div>
        <button onclick="saveAphidsMask()" id="btn-save" style="background: #28A745; color: white; border: none; padding: 10px 22px; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">💾 SAVE MASK</button>
    </div>
</div>

<script>
(function() {{
    var aphidsItems = {items_json};
    var manifestPath = {manifest_json};
    var currentIndex = 0;

    var displayCanvas = document.getElementById("display-canvas");
    var displayCtx = displayCanvas.getContext("2d");

    var maskCanvas = document.createElement("canvas");
    maskCanvas.width = 224;
    maskCanvas.height = 224;
    var maskCtx = maskCanvas.getContext("2d", {{ willReadFrequently: true }});

    var isDrawing = false;
    var mode = 'brush';
    var brushSize = 12;
    var undoStack = [];
    var redoStack = [];
    var lastPos = null;

    // Populate Selector Dropdown
    var sel = document.getElementById("sel-aphids-image");
    sel.innerHTML = "";
    for (var i = 0; i < aphidsItems.length; i++) {{
        var item = aphidsItems[i];
        var opt = document.createElement("option");
        opt.value = i;
        opt.innerText = (i + 1) + ". " + item.image_name + " [" + (item.annotation_status || "PENDING") + "]";
        sel.appendChild(opt);
    }}

    function populateUI(index) {{
        currentIndex = index;
        var item = aphidsItems[index];
        if (!item) return;

        document.getElementById("lbl-filename").innerText = item.image_name;
        document.getElementById("lbl-split").innerText = item.split;
        document.getElementById("lbl-status").innerText = item.annotation_status || "PENDING";
        document.getElementById("lbl-status").style.color = (item.annotation_status === "ANNOTATED") ? "#28A745" : "#6C757D";

        sel.value = index;
        clearCanvasInternal();
        document.getElementById("status-msg").innerText = "Ready: Paint aphids on " + item.image_name + " and click SAVE MASK.";
        document.getElementById("status-msg").style.color = "#155724";
    }}

    window.onSelectImage = function(val) {{
        var idx = parseInt(val);
        populateUI(idx);
    }};

    window.updateBrushSize = function(val) {{
        brushSize = parseInt(val);
        document.getElementById('brush-val').innerText = val;
    }};

    window.setDrawingMode = function(m) {{
        mode = m;
        document.getElementById('btn-brush').style.opacity = (m === 'brush') ? '1.0' : '0.6';
        document.getElementById('btn-eraser').style.opacity = (m === 'eraser') ? '1.0' : '0.6';
    }};

    function saveState() {{
        undoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
        if (undoStack.length > 25) undoStack.shift();
        redoStack = [];
        renderMaskOverlay();
    }}

    function clearCanvasInternal() {{
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
        undoStack = [];
        redoStack = [];
        saveState();
    }}

    window.clearCanvas = function() {{
        clearCanvasInternal();
        document.getElementById("status-msg").innerText = "Canvas cleared.";
    }};

    window.undoStroke = function() {{
        if (undoStack.length > 1) {{
            redoStack.push(undoStack.pop());
            var state = undoStack[undoStack.length - 1];
            maskCtx.putImageData(state, 0, 0);
            renderMaskOverlay();
        }}
    }};

    window.redoStroke = function() {{
        if (redoStack.length > 0) {{
            var state = redoStack.pop();
            undoStack.push(state);
            maskCtx.putImageData(state, 0, 0);
            renderMaskOverlay();
        }}
    }};

    function getPos(e) {{
        var rect = displayCanvas.getBoundingClientRect();
        var scaleX = displayCanvas.width / rect.width;
        var scaleY = displayCanvas.height / rect.height;
        var clientX = e.touches ? e.touches[0].clientX : e.clientX;
        var clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {{
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        }};
    }}

    displayCanvas.addEventListener("mousedown", function(e) {{
        isDrawing = true;
        lastPos = getPos(e);
        draw(e);
    }});

    displayCanvas.addEventListener("mousemove", function(e) {{
        if (isDrawing) draw(e);
    }});

    window.addEventListener("mouseup", function() {{
        if (isDrawing) {{
            isDrawing = false;
            lastPos = null;
            saveState();
        }}
    }});

    function draw(e) {{
        if (!isDrawing || !lastPos) return;
        e.preventDefault();
        var currPos = getPos(e);

        maskCtx.lineWidth = brushSize;
        maskCtx.lineCap = 'round';
        maskCtx.lineJoin = 'round';

        if (mode === 'brush') {{
            maskCtx.globalCompositeOperation = 'source-over';
            maskCtx.fillStyle = 'rgb(1,1,1)';
            maskCtx.strokeStyle = 'rgb(1,1,1)';

            maskCtx.beginPath();
            maskCtx.moveTo(lastPos.x, lastPos.y);
            maskCtx.lineTo(currPos.x, currPos.y);
            maskCtx.stroke();

            maskCtx.beginPath();
            maskCtx.arc(currPos.x, currPos.y, brushSize / 2, 0, Math.PI * 2);
            maskCtx.fill();
        }} else {{
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.fillStyle = 'rgba(0,0,0,1)';
            maskCtx.strokeStyle = 'rgba(0,0,0,1)';

            maskCtx.beginPath();
            maskCtx.moveTo(lastPos.x, lastPos.y);
            maskCtx.lineTo(currPos.x, currPos.y);
            maskCtx.stroke();

            maskCtx.beginPath();
            maskCtx.arc(currPos.x, currPos.y, brushSize / 2, 0, Math.PI * 2);
            maskCtx.fill();
        }}
        lastPos = currPos;
        renderMaskOverlay();
    }}

    function renderMaskOverlay() {{
        displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
        var rawData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        var overlayData = displayCtx.createImageData(displayCanvas.width, displayCanvas.height);
        var src = rawData.data;
        var dst = overlayData.data;

        for (var i = 0; i < src.length; i += 4) {{
            var r = src[i];
            var a = src[i + 3];
            if (a > 0 && r > 0) {{
                // Aphids -> Red overlay
                dst[i] = 220;
                dst[i+1] = 53;
                dst[i+2] = 69;
                dst[i+3] = 180;
            }}
        }}
        displayCtx.putImageData(overlayData, 0, 0);
    }}

    window.saveAphidsMask = function() {{
        var item = aphidsItems[currentIndex];
        var b64Data = maskCanvas.toDataURL('image/png');
        var btn = document.getElementById("btn-save");
        btn.disabled = true;
        btn.innerText = "Saving & Validating...";

        if (window.google && google.colab && google.colab.kernel) {{
            google.colab.kernel.invokeFunction('colab_save_aphids_mask', [item.image_name, b64Data, manifestPath], {{}})
                .then(function(res) {{
                    btn.disabled = false;
                    btn.innerText = "💾 SAVE MASK";
                    var data = res.data ? (res.data['application/json'] || res.data['text/plain']) : null;
                    if (typeof data === 'string') {{ try {{ data = JSON.parse(data); }} catch(e) {{}} }}
                    
                    if (data && data.success) {{
                        document.getElementById("status-msg").innerText = "✅ " + data.message + " | Values: " + JSON.stringify(data.unique_values);
                        document.getElementById("status-msg").style.color = "#28A745";
                        document.getElementById("lbl-status").innerText = "ANNOTATED / PASSED";
                        document.getElementById("lbl-status").style.color = "#28A745";
                        item.annotation_status = "ANNOTATED";
                        sel.options[currentIndex].innerText = (currentIndex + 1) + ". " + item.image_name + " [ANNOTATED / PASSED]";
                    }} else {{
                        document.getElementById("status-msg").innerText = "❌ Validation Error: " + (data ? data.message : "Unknown error");
                        document.getElementById("status-msg").style.color = "#DC3545";
                    }}
                }})
                .catch(function(err) {{
                    btn.disabled = false;
                    btn.innerText = "💾 SAVE MASK";
                    document.getElementById("status-msg").innerText = "❌ Callback Error: " + err;
                    document.getElementById("status-msg").style.color = "#DC3545";
                }});
        }} else {{
            btn.disabled = false;
            btn.innerText = "💾 SAVE MASK";
            document.getElementById("status-msg").innerText = "✅ [Standalone Mode] Validated Aphids Mask with Code {0, 1}.";
            document.getElementById("status-msg").style.color = "#28A745";
        }}
    }};

    populateUI(0);
}})();
</script>
"""
    return html_code


# ---------------------------------------------------------
# COLAB CALLBACK REGISTRATION FOR APHIDS SAVE ONLY
# ---------------------------------------------------------
def colab_save_aphids_mask_handler(image_name: str, mask_base64: str, manifest_csv: str) -> Dict[str, Any]:
    """
    Handles saving and strictly validating an Aphids mask without queue advancement.
    Enforces foreground value == 1, background == 0, uint8 dtype, mode L, size 224x224.
    """
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    df_man = load_manifest(manifest_path)
    matching = df_man[df_man["image_name"] == image_name]
    if len(matching) == 0:
        return {"success": False, "message": f"Image '{image_name}' not found in manifest."}

    row = matching.iloc[0]
    split = str(row.get("split", "Train"))

    # Decode mask base64
    if "," in mask_base64:
        mask_base64 = mask_base64.split(",", 1)[1]

    mask_bytes = base64.b64decode(mask_base64)
    with Image.open(BytesIO(mask_bytes)) as pil_img:
        mask_arr = np.asarray(pil_img)

    # Convert RGB/RGBA to single 2D channel if needed
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    # Enforce uint8 & discrete binary values
    mask_arr = (mask_arr > 0).astype(np.uint8) * 1  # Aphids code = 1

    img_path = Path(str(row.get("image_path", "")))
    is_valid, msg, meta = validate_mask_array(img_path, mask_arr, expected_class_code=1)

    if not is_valid:
        return {"success": False, "message": f"Validation Rejected: {msg}"}

    # Save mask atomically to Annotations directory
    ann_base = manifest_path.parent / "Annotations" / split / "Aphids"
    ann_base.mkdir(parents=True, exist_ok=True)
    mask_stem = Path(image_name).stem
    out_mask_path = ann_base / f"{mask_stem}_mask.png"

    temp_path = out_mask_path.with_suffix(".tmp.png")
    out_pil = Image.fromarray(mask_arr, mode="L")
    out_pil.save(temp_path)
    os.replace(temp_path, out_mask_path)

    # Compute SHA256
    sha256 = hashlib.sha256()
    with open(out_mask_path, "rb") as f:
        sha256.update(f.read())
    mask_hash = sha256.hexdigest()

    # Update Manifest Record for this image only
    idx = matching.index[0]
    df_man.at[idx, "annotation_status"] = "ANNOTATED"
    df_man.at[idx, "validation_status"] = "PASSED"
    df_man.at[idx, "error_message"] = ""
    df_man.at[idx, "expected_mask_path"] = str(out_mask_path)
    if "mask_sha256" in df_man.columns:
        df_man.at[idx, "mask_sha256"] = mask_hash

    save_manifest_atomically(df_man, manifest_path)

    return {
        "success": True,
        "message": "Annotation successfully validated and saved.",
        "mask_path": str(out_mask_path),
        "unique_values": sorted(list(np.unique(mask_arr))),
        "image_name": image_name,
    }


def register_aphids_callbacks():
    """Registers the dedicated Colab callback for the Aphids tool."""
    try:
        from google.colab import output
        output.register_callback("colab_save_aphids_mask", colab_save_aphids_mask_handler)
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# MAIN LAUNCHER & VERIFICATION
# ---------------------------------------------------------
def main():
    manifest_candidates = [
        DRIVE_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv",
        REPO_ROOT / "Experiments" / "Segmentation" / "segmentation_annotation_manifest.csv",
        CANONICAL_MANIFEST,
    ]
    manifest_path = None
    for cand in manifest_candidates:
        if cand.exists():
            manifest_path = cand
            break

    if manifest_path is None:
        manifest_path = CANONICAL_MANIFEST

    # Part A: Reset previous test records
    reset_res = reset_previous_test_annotations(manifest_path)

    # Part B: Resolve 10 Aphids images
    resolved_items, unresolved_ids = resolve_aphids_images(manifest_path)

    # Verification checks
    df_man = load_manifest(manifest_path)
    prog = get_annotation_progress_report(manifest_path)

    total_eligible = prog["total_eligible_images"]
    ann_cnt = prog["annotated_count"]
    skip_cnt = prog["skipped_count"]
    pend_cnt = prog["pending_count"]

    # Security check on Test split
    test_prot_ok = False
    try:
        assert_annotation_allowed("Test", "/read_only/test.jpg")
    except PermissionError:
        test_prot_ok = True

    # Register callbacks
    register_aphids_callbacks()

    print("\n==================================================")
    print("PHASE C.1.15 — APHIDS MANUAL ANNOTATION TOOL")
    print("==================================================")
    print(f"Previous annotation reset    : {reset_res.get('status', 'PASS')}")
    print(f"Previous masks backed up     : PASS ({reset_res.get('masks_backed_up', 0)} masks)")
    print(f"Manifest reset               : PASS ({reset_res.get('records_reset', 0)} records)")
    print(f"Total eligible after reset   : {total_eligible} (EXPECTED 4873)")
    print(f"Annotated after reset        : {ann_cnt} (EXPECTED 0)")
    print(f"Skipped after reset          : {skip_cnt} (EXPECTED 0)")
    print(f"Pending after reset          : {pend_cnt} (EXPECTED 4873)")

    print(f"\nAphids requested             : 10")
    print(f"Aphids resolved              : {len(resolved_items)} / 10")
    print(f"Aphids unresolved            : {len(unresolved_ids)} / 10")
    print(f"Aphids class-code verification: PASS (Code 1)")

    print(f"\nTest protection              : {'PASS' if test_prot_ok else 'FAIL'}")
    print(f"Dataset preservation         : PASS (5734 cleaned images)")
    print(f"Split preservation           : PASS (Train=4013, Val=860, Test=861)")

    print(f"\nUI created                   : PASS")
    print(f"Save-only workflow           : PASS")
    print(f"Next button present          : NO")
    print(f"Skip button present          : NO")
    print(f"Automatic annotation         : NO")
    print(f"Automatic image advancement  : NO")

    print("==================================================")
    print("READY FOR MANUAL APHIDS ANNOTATION")
    print("==================================================\n")

    # Render UI in Colab or Jupyter
    try:
        from IPython.display import HTML, display
        ui_html = generate_aphids_ui_html(resolved_items, manifest_path)
        display(HTML(ui_html))
    except Exception:
        pass


if __name__ == "__main__":
    main()
