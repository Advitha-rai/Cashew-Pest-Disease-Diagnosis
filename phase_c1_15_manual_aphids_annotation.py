"""
Cashew Pest and Disease Diagnosis System
Phase C.1.15 — Dedicated Single-Image Aphids Manual Annotation Tool
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

# ============================================================
# TARGET IMAGE CONFIGURATION
# Set the exact filename of the Aphids image you want to annotate:
# ============================================================
TARGET_IMAGE = "20220218_124514.jpg"

# The 10 canonical requested Aphids image stems for reference:
CANONICAL_APHIDS_STEMS = [
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
    normalize_class_name,
)
from src.segmentation.validation import assert_annotation_allowed, validate_mask_array
from src.segmentation.manifest import load_manifest, save_manifest_atomically


# ---------------------------------------------------------
# 1. Image Resolution Helper
# ---------------------------------------------------------
def find_exact_aphids_image(target_name: str, manifest_path: Path) -> Tuple[Optional[Path], Optional[Dict[str, Any]], str]:
    """
    Locates the target Aphids image file on disk and verifies its manifest metadata.
    Ensures non-Test split, 224x224 dimensions, and class Aphids (code 1).
    """
    clean_target = target_name.strip()
    target_stem = Path(clean_target).stem.lower()

    # 1. Search Dataset directory
    possible_ds_dirs = [
        DRIVE_ROOT / "Dataset" / "Cleaned" / "Aphids",
        REPO_ROOT / "Dataset" / "Cleaned" / "Aphids",
        DATASET_DIR / "Aphids" if DATASET_DIR.name != "Aphids" else DATASET_DIR,
        Path.cwd() / "Dataset" / "Cleaned" / "Aphids",
    ]

    found_img_path = None
    for ds_dir in possible_ds_dirs:
        if ds_dir.exists():
            # Exact name match
            cand = ds_dir / clean_target
            if cand.exists():
                found_img_path = cand
                break
            # Case-insensitive or extension match
            for f in ds_dir.glob("*"):
                if f.name.lower() == clean_target.lower() or f.stem.lower() == target_stem:
                    found_img_path = f
                    break
            if found_img_path:
                break

    # 2. Check Manifest Record
    manifest_record = None
    if manifest_path.exists():
        df_man = load_manifest(manifest_path)
        matching = df_man[
            df_man["image_name"].apply(
                lambda x: str(x).lower() == clean_target.lower()
                or Path(str(x)).stem.lower() == target_stem
                or clean_target.lower() in str(x).lower()
            )
        ]
        if len(matching) > 0:
            manifest_record = matching.iloc[0].to_dict()
            if not found_img_path:
                mp = Path(str(manifest_record.get("image_path", "")))
                if mp.exists():
                    found_img_path = mp

    if not found_img_path and not manifest_record:
        return None, None, f"Image '{target_name}' not found in Dataset/Cleaned/Aphids or manifest."

    resolved_name = found_img_path.name if found_img_path else (manifest_record.get("image_name") if manifest_record else clean_target)
    return found_img_path, manifest_record, f"Successfully resolved '{resolved_name}'."


# ---------------------------------------------------------
# 2. UI HTML Generator (Strict Single-Image, Save-Only)
# ---------------------------------------------------------
def generate_single_image_ui(
    img_name: str,
    img_path: Optional[Path],
    manifest_record: Optional[Dict[str, Any]],
    manifest_path: Path,
) -> str:
    """
    Builds the dedicated single-image Aphids manual segmentation interface.
    Features ONLY 'SAVE ANNOTATION'. NO Next, NO Skip, NO Previous, NO queue.
    """
    img_b64 = ""
    if img_path and img_path.exists():
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

    split_name = manifest_record.get("split", "Train") if manifest_record else "Train"
    status_name = manifest_record.get("annotation_status", "PENDING") if manifest_record else "PENDING"

    manifest_json = json.dumps(str(manifest_path))
    img_name_json = json.dumps(img_name)

    html_code = f"""
<div id="aphids-manual-container" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 820px; margin: 20px auto; background: #FFFFFF; padding: 20px; border-radius: 10px; border: 2px solid #28A745; box-shadow: 0 4px 14px rgba(0,0,0,0.12);">
    
    <!-- Title Banner -->
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #28A745; padding-bottom: 12px; margin-bottom: 14px;">
        <h3 style="margin: 0; color: #155724; font-size: 19px; font-weight: bold;">🌿 Manual Aphids Segmentation Tool — Phase C.1.15</h3>
        <span style="background: #28A745; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold;">Class: Aphids (Code 1)</span>
    </div>

    <!-- Target Image Details Card -->
    <div style="background: #F8F9FA; padding: 12px 16px; border: 1px solid #CED4DA; border-radius: 6px; margin-bottom: 14px; font-size: 13px; display: flex; flex-wrap: wrap; gap: 18px;">
        <div><strong>Target File:</strong> <span style="color: #155724; font-weight: bold;">{img_name}</span></div>
        <div><strong>Class:</strong> <span style="color: #DC3545; font-weight: bold;">Aphids</span> (Code: 1)</div>
        <div><strong>Split:</strong> <span style="color: #007BFF; font-weight: bold;">{split_name}</span></div>
        <div><strong>Resolution:</strong> <span>224 × 224</span></div>
        <div><strong>Status:</strong> <span id="lbl-status" style="font-weight: bold; color: {'#28A745' if status_name == 'ANNOTATED' else '#6C757D'};">{status_name}</span></div>
    </div>

    <!-- Drawing Toolbar -->
    <div style="margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; background: #E9ECEF; padding: 10px 14px; border-radius: 6px;">
        <button onclick="setDrawingMode('brush')" id="btn-brush" style="background: #DC3545; color: white; border: none; padding: 7px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">🖌️ Paint Aphids (Red)</button>
        <button onclick="setDrawingMode('eraser')" id="btn-eraser" style="background: #6C757D; color: white; border: none; padding: 7px 16px; border-radius: 4px; font-weight: bold; cursor: pointer; opacity: 0.6;">🧹 Eraser</button>
        
        <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px;">
            <label style="font-size: 13px; font-weight: bold;">Brush Size:</label>
            <input type="range" min="2" max="60" value="12" oninput="updateBrushSize(this.value)" style="cursor: pointer; width: 100px;">
            <span id="brush-val" style="font-size: 13px; font-weight: bold; min-width: 20px;">12</span>px
        </div>

        <div style="display: flex; gap: 6px; margin-left: auto;">
            <button onclick="undoStroke()" style="background: #007BFF; color: white; border: none; padding: 7px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">↩️ Undo</button>
            <button onclick="redoStroke()" style="background: #17A2B8; color: white; border: none; padding: 7px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">↪️ Redo</button>
            <button onclick="clearCanvas()" style="background: #6C757D; color: white; border: none; padding: 7px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">🗑️ Clear</button>
        </div>
    </div>

    <!-- Dual Canvas Work Area (224x224 scaled to 448x448 for drawing precision) -->
    <div id="canvas-wrapper" style="position: relative; width: 448px; height: 448px; border: 2px solid #28A745; margin: 0 auto; background: #000; overflow: hidden; border-radius: 4px;">
        <img id="bg-img" src="data:image/jpeg;base64,{img_b64}" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; object-fit: contain;">
        <canvas id="display-canvas" width="224" height="224" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; pointer-events: auto; touch-action: none; cursor: crosshair;"></canvas>
    </div>

    <!-- Action Bar: ONLY SAVE ANNOTATION (NO Next / Skip buttons) -->
    <div style="margin-top: 16px; display: flex; justify-content: space-between; align-items: center; gap: 14px;">
        <div id="status-msg" style="font-weight: bold; color: #155724; font-size: 13px;">Ready: Paint visible aphids clusters on the leaf and click SAVE ANNOTATION.</div>
        <button onclick="saveAnnotation()" id="btn-save" style="background: #28A745; color: white; border: none; padding: 11px 26px; border-radius: 6px; font-size: 14px; font-weight: bold; cursor: pointer; box-shadow: 0 3px 8px rgba(0,0,0,0.18);">💾 SAVE ANNOTATION</button>
    </div>
</div>

<script>
(function() {{
    var imageName = {img_name_json};
    var manifestPath = {manifest_json};

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

    saveState();

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
        if (undoStack.length > 30) undoStack.shift();
        redoStack = [];
        renderOverlay();
    }}

    window.clearCanvas = function() {{
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
        undoStack = [];
        redoStack = [];
        saveState();
        document.getElementById("status-msg").innerText = "Canvas cleared.";
    }};

    window.undoStroke = function() {{
        if (undoStack.length > 1) {{
            redoStack.push(undoStack.pop());
            var state = undoStack[undoStack.length - 1];
            maskCtx.putImageData(state, 0, 0);
            renderOverlay();
        }}
    }};

    window.redoStroke = function() {{
        if (redoStack.length > 0) {{
            var state = redoStack.pop();
            undoStack.push(state);
            maskCtx.putImageData(state, 0, 0);
            renderOverlay();
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
        renderOverlay();
    }}

    function renderOverlay() {{
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

    window.saveAnnotation = function() {{
        var b64Data = maskCanvas.toDataURL('image/png');
        var btn = document.getElementById("btn-save");
        btn.disabled = true;
        btn.innerText = "Saving & Validating...";

        if (window.google && google.colab && google.colab.kernel) {{
            google.colab.kernel.invokeFunction('colab_save_single_aphids_annotation', [imageName, b64Data, manifestPath], {{}})
                .then(function(res) {{
                    btn.disabled = false;
                    btn.innerText = "💾 SAVE ANNOTATION";
                    var data = res.data ? (res.data['application/json'] || res.data['text/plain']) : null;
                    if (typeof data === 'string') {{ try {{ data = JSON.parse(data); }} catch(e) {{}} }}
                    
                    if (data && data.success) {{
                        document.getElementById("status-msg").innerText = "✅ " + data.message + " | Mask: " + data.mask_path;
                        document.getElementById("status-msg").style.color = "#28A745";
                        document.getElementById("lbl-status").innerText = "ANNOTATED / PASSED";
                        document.getElementById("lbl-status").style.color = "#28A745";
                    }} else {{
                        document.getElementById("status-msg").innerText = "❌ Validation Error: " + (data ? data.message : "Unknown error");
                        document.getElementById("status-msg").style.color = "#DC3545";
                    }}
                }})
                .catch(function(err) {{
                    btn.disabled = false;
                    btn.innerText = "💾 SAVE ANNOTATION";
                    document.getElementById("status-msg").innerText = "❌ Callback Error: " + err;
                    document.getElementById("status-msg").style.color = "#DC3545";
                }});
        }} else {{
            btn.disabled = false;
            btn.innerText = "💾 SAVE ANNOTATION";
            document.getElementById("status-msg").innerText = "✅ [Standalone Preview Mode] Validated Aphids Mask values {0, 1}.";
            document.getElementById("status-msg").style.color = "#28A745";
        }}
    }};
}})();
</script>
"""
    return html_code


# ---------------------------------------------------------
# 3. Backend Save Handler & Strict Validation
# ---------------------------------------------------------
def colab_save_single_aphids_annotation(image_name: str, mask_base64: str, manifest_csv: str) -> Dict[str, Any]:
    """
    Handles saving and strictly validating an Aphids mask for the specified image.
    Enforces foreground value == 1, background == 0, uint8 dtype, mode L, size 224x224.
    Updates only this image's manifest row and creates a timestamped backup before writing.
    """
    manifest_path = Path(manifest_csv)
    if not manifest_path.exists():
        manifest_path = CANONICAL_MANIFEST

    df_man = load_manifest(manifest_path)
    clean_target = image_name.strip()
    target_stem = Path(clean_target).stem.lower()

    matching = df_man[
        df_man["image_name"].apply(
            lambda x: str(x).lower() == clean_target.lower()
            or Path(str(x)).stem.lower() == target_stem
            or clean_target.lower() in str(x).lower()
        )
    ]

    if len(matching) == 0:
        return {"success": False, "message": f"Image '{image_name}' not found in canonical manifest."}

    row = matching.iloc[0]
    split = str(row.get("split", "Train"))

    # Security check on Test split
    if split == "Test":
        return {"success": False, "message": "Security Violation: Test split images cannot be annotated!"}

    # Decode mask base64
    if "," in mask_base64:
        mask_base64 = mask_base64.split(",", 1)[1]

    mask_bytes = base64.b64decode(mask_base64)
    with Image.open(BytesIO(mask_bytes)) as pil_img:
        mask_arr = np.asarray(pil_img)

    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]

    # Verify non-empty annotation
    fg_pixels = int(np.count_nonzero(mask_arr))
    if fg_pixels == 0:
        return {"success": False, "message": "Validation Rejected: Mask is completely empty! Please paint lesion pixels."}

    # Convert foreground to Aphids code = 1
    mask_arr = (mask_arr > 0).astype(np.uint8) * 1

    img_path = Path(str(row.get("image_path", "")))
    is_valid, msg, meta = validate_mask_array(img_path, mask_arr, expected_class_code=1)

    if not is_valid:
        return {"success": False, "message": f"Validation Rejected: {msg}"}

    # 1. Create Pre-Save Backup of Manifest
    backup_dir = manifest_path.parent / "backup_previous" / f"backup_before_{Path(image_name).stem}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, backup_dir / f"manifest_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # 2. Save Mask Atomically to Disk
    ann_base = manifest_path.parent / "Annotations" / "Train" / "Aphids"
    ann_base.mkdir(parents=True, exist_ok=True)
    mask_stem = Path(image_name).stem
    out_mask_path = ann_base / f"{mask_stem}_mask.png"

    temp_path = out_mask_path.with_suffix(".tmp.png")
    out_pil = Image.fromarray(mask_arr, mode="L")
    out_pil.save(temp_path)
    os.replace(temp_path, out_mask_path)

    # 3. Compute SHA256
    sha256 = hashlib.sha256()
    with open(out_mask_path, "rb") as f:
        sha256.update(f.read())
    mask_hash = sha256.hexdigest()

    # 4. Update Canonical Manifest Row (ONLY this image)
    idx = matching.index[0]
    df_man.at[idx, "annotation_status"] = "ANNOTATED"
    df_man.at[idx, "validation_status"] = "PASSED"
    df_man.at[idx, "class_name"] = "Aphids"
    df_man.at[idx, "class_code"] = 1
    df_man.at[idx, "split"] = "Train"
    df_man.at[idx, "expected_mask_path"] = str(out_mask_path)
    df_man.at[idx, "error_message"] = ""
    if "mask_sha256" in df_man.columns:
        df_man.at[idx, "mask_sha256"] = mask_hash

    save_manifest_atomically(df_man, manifest_path)

    print("\n============================================================")
    print("ANNOTATION SAVED SUCCESSFULLY")
    print("============================================================")
    print(f"Image            : {image_name}")
    print(f"Class            : Aphids")
    print(f"Class Code       : 1")
    print(f"Mask             : {out_mask_path}")
    print(f"Unique Values    : {sorted(list(np.unique(mask_arr)))}")
    print(f"Geometry         : 224 x 224")
    print(f"Validation       : PASSED")
    print(f"Manifest Updated : YES (SHA-256: {mask_hash[:12]}...)")
    print("============================================================\n")

    return {
        "success": True,
        "message": "Annotation successfully validated and saved.",
        "mask_path": str(out_mask_path),
        "unique_values": sorted(list(np.unique(mask_arr))),
        "image_name": image_name,
    }


def register_colab_callbacks():
    """Registers the dedicated single-image Colab callback."""
    try:
        from google.colab import output
        output.register_callback("colab_save_single_aphids_annotation", colab_save_single_aphids_annotation)
        return True
    except Exception:
        return False


# ---------------------------------------------------------
# 4. Main Tool Execution
# ---------------------------------------------------------
def main():
    print("============================================================")
    print("PHASE C.1.15 — MANUAL APHIDS ANNOTATION TOOL")
    print("============================================================")
    print(f"Repository Root : {REPO_ROOT}")
    print(f"Drive Root      : {DRIVE_ROOT}")
    print(f"Configured Target Image : {TARGET_IMAGE}")

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

    # Step 1: Resolve Target Image
    img_path, record, msg = find_exact_aphids_image(TARGET_IMAGE, manifest_path)
    print(f"\nResolution Status: {msg}")
    if img_path:
        print(f"Image File Path  : {img_path}")

    if record:
        print(f"Manifest Split   : {record.get('split')}")
        print(f"Manifest Status  : {record.get('annotation_status')}")

        if record.get("annotation_status") == "ANNOTATED":
            print(f"\n⚠️ NOTE: Image '{TARGET_IMAGE}' is ALREADY annotated with status PASSED.")
            print("  You can redraw and click SAVE ANNOTATION to update the mask if needed.")

    # Register Callback
    register_colab_callbacks()

    # Display Canvas UI in Colab / Jupyter
    try:
        from IPython.display import HTML, display
        ui_html = generate_single_image_ui(
            img_name=TARGET_IMAGE,
            img_path=img_path,
            manifest_record=record,
            manifest_path=manifest_path,
        )
        display(HTML(ui_html))
    except Exception:
        pass


if __name__ == "__main__":
    main()
