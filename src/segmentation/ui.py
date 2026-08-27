"""
Cashew Pest and Disease Diagnosis System
Phase C: HTML5 Dual-Canvas Interactive Manual Segmentation UI Generator
Framework: TensorFlow / Keras
"""

import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

from .config import SegmentationConfig, get_class_code
from .callbacks import make_json_safe


def image_to_base64(image_path: str | Path) -> str:
    """Converts a local image file to a base64 encoded string."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Source image not found: {image_path}")

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_annotation_html(
    item: Dict[str, Any],
    progress: Dict[str, Any],
    manifest_csv: Optional[str | Path] = None,
    debug_ui: bool = False,
) -> str:
    """
    Constructs the complete standalone HTML/CSS/JS dual-canvas manual segmentation interface.
    """
    img_path = Path(item["image_path"])
    img_b64 = item.get("base64")
    if not img_b64:
        img_b64 = image_to_base64(img_path)

    width = int(item.get("width", SegmentationConfig.IMG_WIDTH))
    height = int(item.get("height", SegmentationConfig.IMG_HEIGHT))

    # Calculate scaled display dimensions (max width 600px for comfortable display)
    max_disp_w = 600
    if width > max_disp_w:
        disp_w = max_disp_w
        disp_h = int(disp_w * height / width)
    else:
        disp_w = width
        disp_h = height

    item_json = json.dumps(make_json_safe(item))
    progress_json = json.dumps(make_json_safe(progress))
    manifest_json = json.dumps(str(manifest_csv or SegmentationConfig.get_manifest_path()))

    curr_class = str(item.get("class_name", "Aphids"))
    curr_code = int(item.get("class_code", get_class_code(curr_class)))
    curr_split = str(item.get("split", "Train"))

    return f"""
<div id="annotation-widget-container"
     style="font-family: Arial, sans-serif; max-width: 900px; padding: 18px; border: 2px solid #1F497D; border-radius: 8px; background: #F8F9FA;">

    <!-- Title & Header -->
    <h3 style="margin-top: 0; color: #1F497D; display: flex; align-items: center; justify-content: space-between;">
        <span>🎨 Cashew Leaf Manual Segmentation Tool</span>
        <span style="font-size: 13px; font-weight: normal; background: #28A745; color: white; padding: 3px 8px; border-radius: 4px;">Phase C</span>
    </h3>

    <!-- Progress Statistics Bar -->
    <div style="background: #E9ECEF; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; display: flex; flex-wrap: wrap; gap: 14px; justify-content: space-between; font-size: 13px;">
        <div><strong>Eligible:</strong> <span id="stat-total">{progress.get("total_eligible_images", 4873)}</span></div>
        <div><strong>Test:</strong> <span id="stat-test">{progress.get("test_images_isolated", 861)}</span> <span style="color: #DC3545; font-weight: bold;">[READ ONLY]</span></div>
        <div><strong>Annotated:</strong> <span id="stat-annotated" style="color: #28A745; font-weight: bold;">{progress.get("annotated_count", 0)}</span></div>
        <div><strong>Passed:</strong> <span id="stat-passed" style="color: #1F497D; font-weight: bold;">{progress.get("passed_validation_count", 0)}</span></div>
        <div><strong>Skipped:</strong> <span id="stat-skipped" style="color: #FFC107; font-weight: bold;">{progress.get("skipped_count", 0)}</span></div>
        <div><strong>Pending:</strong> <span id="stat-pending" style="color: #6C757D; font-weight: bold;">{progress.get("pending_count", 4873)}</span></div>
        <div><strong>Progress:</strong> <span id="stat-progress" style="font-weight: bold;">{progress.get("progress_percentage", 0.0)}%</span></div>
    </div>

    <!-- Active Item Information Bar -->
    <div style="background: white; padding: 10px 14px; border: 1px solid #CED4DA; border-radius: 6px; margin-bottom: 14px; font-size: 13px; display: flex; flex-wrap: wrap; gap: 14px;">
        <div><strong>Split:</strong> <span id="lbl-split" style="font-weight: bold; color: #1F497D;">{curr_split}</span></div>
        <div><strong>Target Class:</strong> <span id="lbl-class" style="font-weight: bold; color: #1F497D;">{curr_class}</span></div>
        <div><strong>Class Code:</strong> <span id="lbl-code" style="font-weight: bold; color: #28A745;">{curr_code}</span></div>
        <div><strong>Resolution:</strong> <span id="lbl-dim">{width} × {height}</span></div>
        <div><strong>File:</strong> <span id="lbl-file" style="word-break: break-all;">{img_path.name}</span></div>
    </div>

    <!-- Drawing Toolbar Controls -->
    <div style="margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; background: #E9ECEF; padding: 8px 12px; border-radius: 6px;">
        <button onclick="setMode('brush')" id="btn-brush" style="background: #28A745; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; cursor: pointer;">🖌️ Paint Lesion</button>
        <button onclick="setMode('eraser')" id="btn-eraser" style="background: #6C757D; color: white; border: none; padding: 6px 14px; border-radius: 4px; font-weight: bold; cursor: pointer; opacity: 0.7;">🧹 Eraser</button>
        
        <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px;">
            <label style="font-size: 13px; font-weight: bold;">Brush Size:</label>
            <input type="range" min="2" max="60" value="14" oninput="updateBrushSize(this.value)" style="cursor: pointer; width: 100px;">
            <span id="brush-val" style="font-size: 13px; font-weight: bold; min-width: 20px;">14</span>px
        </div>

        <div style="display: flex; gap: 6px; margin-left: auto;">
            <button onclick="undoStroke()" style="background: #007BFF; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">↩️ Undo</button>
            <button onclick="redoStroke()" style="background: #17A2B8; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">↪️ Redo</button>
            <button onclick="clearCanvas()" style="background: #DC3545; color: white; border: none; padding: 6px 12px; border-radius: 4px; font-weight: bold; cursor: pointer;">🗑️ Clear</button>
        </div>
    </div>

    <!-- Dual Canvas Work Area -->
    <div id="canvas-wrapper" style="position: relative; width: {disp_w}px; height: {disp_h}px; border: 2px solid #1F497D; margin: 0 auto; background: #000; overflow: hidden; border-radius: 4px;">
        <img id="bg-img" src="data:image/jpeg;base64,{img_b64}" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; object-fit: contain;">
        <canvas id="display-canvas" width="{width}" height="{height}" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; pointer-events: auto; touch-action: none; cursor: crosshair;"></canvas>
    </div>

    <!-- Action Bar (Skip & Save) -->
    <div style="margin-top: 14px; display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap;">
        <div id="status-msg" style="font-weight: bold; color: #1F497D; font-size: 13px;">Ready: Paint visible lesion areas and click Save Mask & Next.</div>
        <div style="display: flex; gap: 10px;">
            <button onclick="skipImage()" id="btn-skip" style="background: #FFC107; color: black; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; cursor: pointer;">⏭️ Skip for Review</button>
            <button onclick="saveAndNext()" id="btn-save" style="background: #28A745; color: white; border: none; padding: 8px 18px; border-radius: 4px; font-weight: bold; cursor: pointer;">💾 Save Mask & Next</button>
        </div>
    </div>
</div>

<script>
(function() {{
    var DEBUG_UI = {str(bool(debug_ui)).lower()};
    var currentImgPath = {json.dumps(str(item["image_path"]))};
    var currentSplit = {json.dumps(curr_split)};
    var currentClass = {json.dumps(curr_class)};
    var currentCode = {curr_code};
    var currentManifest = {manifest_json};

    var displayCanvas = document.getElementById("display-canvas");
    var displayCtx = displayCanvas.getContext("2d");

    var maskCanvas = document.createElement("canvas");
    maskCanvas.width = displayCanvas.width;
    maskCanvas.height = displayCanvas.height;
    var maskCtx = maskCanvas.getContext("2d", {{ willReadFrequently: true }});

    var isDrawing = false;
    var mode = 'brush';
    var brushSize = 14;
    var undoStack = [];
    var redoStack = [];
    var lastPos = null;

    var colorMap = {{
        1: "rgba(220, 53, 69, 0.7)",   // Aphids -> Red
        2: "rgba(40, 167, 69, 0.7)",   // Leaf_Miner -> Green
        3: "rgba(0, 123, 255, 0.7)",   // Leaf_Blight -> Blue
        4: "rgba(255, 193, 7, 0.7)"    // TMB -> Yellow
    }};

    saveState();

    window.updateBrushSize = function(val) {{
        brushSize = parseInt(val);
        document.getElementById('brush-val').innerText = val;
    }};

    window.setMode = function(m) {{
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

    window.clearCanvas = function() {{
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        saveState();
    }};

    function getPos(e) {{
        var rect = displayCanvas.getBoundingClientRect();
        var scaleX = maskCanvas.width / rect.width;
        var scaleY = maskCanvas.height / rect.height;
        var clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
        var clientY = e.clientY || (e.touches && e.touches[0] ? e.touches[0].clientY : 0);
        return {{
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        }};
    }}

    displayCanvas.addEventListener('pointerdown', function(e) {{
        isDrawing = true;
        displayCanvas.setPointerCapture(e.pointerId);
        lastPos = getPos(e);
        draw(e);
    }});

    displayCanvas.addEventListener('pointermove', function(e) {{
        if (isDrawing) draw(e);
    }});

    displayCanvas.addEventListener('pointerup', function(e) {{
        if (isDrawing) {{
            isDrawing = false;
            lastPos = null;
            saveState();
        }}
    }});

    displayCanvas.addEventListener('pointercancel', function(e) {{
        if (isDrawing) {{
            isDrawing = false;
            lastPos = null;
            saveState();
        }}
    }});

    displayCanvas.addEventListener('pointerleave', function(e) {{
        if (isDrawing && !displayCanvas.hasPointerCapture(e.pointerId)) {{
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
            maskCtx.fillStyle = 'rgb(' + currentCode + ',' + currentCode + ',' + currentCode + ')';
            maskCtx.strokeStyle = maskCtx.fillStyle;

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
                if (r === 1) {{ dst[i] = 220; dst[i+1] = 53; dst[i+2] = 69; dst[i+3] = 180; }}
                else if (r === 2) {{ dst[i] = 40; dst[i+1] = 167; dst[i+2] = 69; dst[i+3] = 180; }}
                else if (r === 3) {{ dst[i] = 0; dst[i+1] = 123; dst[i+2] = 255; dst[i+3] = 180; }}
                else if (r === 4) {{ dst[i] = 255; dst[i+1] = 193; dst[i+2] = 7; dst[i+3] = 180; }}
            }}
        }}
        displayCtx.putImageData(overlayData, 0, 0);
    }}

    function getRawMaskBase64() {{
        var data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
        var nonZero = 0;
        var maxValue = 0;

        for (var i = 0; i < data.length; i += 4) {{
            if (data[i] > 0 || data[i + 1] > 0 || data[i + 2] > 0 || data[i + 3] > 0) {{
                nonZero++;
            }}
            maxValue = Math.max(maxValue, data[i], data[i + 1], data[i + 2], data[i + 3]);
        }}

        console.log("MASK DEBUG:", {{
            width: maskCanvas.width,
            height: maskCanvas.height,
            nonZeroPixels: nonZero,
            maxValue: maxValue
        }});

        return maskCanvas.toDataURL('image/png');
    }}

    function parseColabResponse(res) {{
        if (!res) return null;
        if (res.data) {{
            if (res.data['application/json']) return res.data['application/json'];
            if (res.data['text/plain']) {{
                try {{ return JSON.parse(res.data['text/plain']); }} catch(e) {{ return {{ message: res.data['text/plain'] }}; }}
            }}
        }}
        if (typeof res === 'string') {{
            try {{ return JSON.parse(res); }} catch(e) {{ return {{ message: res }}; }}
        }}
        return res;
    }}

    function updateProgressUI(prog) {{
        if (!prog) return;
        var map = {{
            total_eligible_images: 'stat-total',
            annotated_count: 'stat-annotated',
            passed_validation_count: 'stat-passed',
            skipped_count: 'stat-skipped',
            pending_count: 'stat-pending',
            progress_percentage: 'stat-progress'
        }};
        Object.keys(map).forEach(function(k) {{
            var el = document.getElementById(map[k]);
            if (el && prog[k] !== undefined) {{
                el.innerText = (k === 'progress_percentage') ? prog[k] + '%' : prog[k];
            }}
        }});
    }}

    function loadNextImageInPlace(item) {{
        if (!item) {{
            document.getElementById('annotation-widget-container').innerHTML =
                "<div style='text-align: center; padding: 40px; background: #D4EDDA; border-radius: 8px;'>" +
                "<h2 style='color: #155724;'>🎉 All Eligible Images Are Annotated!</h2>" +
                "<p style='color: #155724;'>No further pending images found in the selected split/class.</p>" +
                "</div>";
            return;
        }}

        currentImgPath = item.image_path;
        currentSplit = item.split;
        currentClass = item.class_name;
        currentCode = parseInt(item.class_code);

        document.getElementById('lbl-split').innerText = currentSplit;
        document.getElementById('lbl-class').innerText = currentClass;
        document.getElementById('lbl-code').innerText = currentCode;
        document.getElementById('lbl-file').innerText = (item.image_path || '').split('/').pop().split('\\\\').pop();

        var w = parseInt(item.width || 224);
        var h = parseInt(item.height || 224);
        document.getElementById('lbl-dim').innerText = w + ' × ' + h;

        displayCanvas.width = w;
        displayCanvas.height = h;
        maskCanvas.width = w;
        maskCanvas.height = h;

        undoStack = [];
        redoStack = [];
        maskCtx.clearRect(0, 0, w, h);
        saveState();

        document.getElementById('bg-img').src = 'data:image/jpeg;base64,' + item.base64;
        document.getElementById('btn-save').disabled = false;
        document.getElementById('btn-skip').disabled = false;
        document.getElementById('status-msg').innerText = 'Ready: Paint visible lesion areas and click Save Mask & Next.';
        document.getElementById('status-msg').style.color = '#1F497D';
    }}

    window.saveAndNext = function() {{
        document.getElementById('btn-save').disabled = true;
        document.getElementById('status-msg').innerText = '⏳ Validating and saving mask...';

        var rawMaskB64 = getRawMaskBase64();

        if (window.google && google.colab && google.colab.kernel) {{
            google.colab.kernel.invokeFunction(
                'notebook.save_mask',
                [currentImgPath, rawMaskB64, currentSplit, currentClass, currentManifest],
                {{}}
            ).then(function(res) {{
                var data = parseColabResponse(res);
                if (data && data.success) {{
                    updateProgressUI(data.progress);
                    document.getElementById('status-msg').innerText = '✅ Mask validated and saved.';
                    loadNextImageInPlace(data.next_item);
                }} else {{
                    document.getElementById('btn-save').disabled = false;
                    document.getElementById('status-msg').innerText = '❌ ' + ((data && data.message) ? data.message : 'Save failed.');
                    document.getElementById('status-msg').style.color = '#DC3545';
                }}
            }}).catch(function(err) {{
                document.getElementById('btn-save').disabled = false;
                document.getElementById('status-msg').innerText = '❌ Callback error: ' + err;
                document.getElementById('status-msg').style.color = '#DC3545';
            }});
        }} else {{
            document.getElementById('btn-save').disabled = false;
            document.getElementById('status-msg').innerText = '⚠️ Running in local test mode (Google Colab kernel not active).';
        }}
    }};

    window.skipImage = function() {{
        document.getElementById('btn-skip').disabled = true;
        document.getElementById('status-msg').innerText = '⏳ Marking image skipped for review...';

        if (window.google && google.colab && google.colab.kernel) {{
            google.colab.kernel.invokeFunction(
                'notebook.skip_image',
                [currentImgPath, 'Manual review skipped', currentSplit, currentClass, currentManifest],
                {{}}
            ).then(function(res) {{
                var data = parseColabResponse(res);
                if (data && data.success) {{
                    updateProgressUI(data.progress);
                    document.getElementById('status-msg').innerText = '✅ Image marked skipped.';
                    loadNextImageInPlace(data.next_item);
                }} else {{
                    document.getElementById('btn-skip').disabled = false;
                    document.getElementById('status-msg').innerText = '❌ ' + ((data && data.message) ? data.message : 'Skip failed.');
                    document.getElementById('status-msg').style.color = '#DC3545';
                }}
            }}).catch(function(err) {{
                document.getElementById('btn-skip').disabled = false;
                document.getElementById('status-msg').innerText = '❌ Callback error: ' + err;
                document.getElementById('status-msg').style.color = '#DC3545';
            }});
        }} else {{
            document.getElementById('btn-skip').disabled = false;
            document.getElementById('status-msg').innerText = '⚠️ Running in local test mode (Google Colab kernel not active).';
        }}
    }};
}})();
</script>
"""
