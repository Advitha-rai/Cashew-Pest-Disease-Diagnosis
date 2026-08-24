"""
Cashew Pest and Disease Diagnosis System
Phase C: Google Colab Callback Registration & IPC Communication Engine
Framework: TensorFlow / Keras
"""

import json
from typing import Dict, Any, Optional
import numpy as np


def make_json_safe(obj: Any) -> Any:
    """Recursively converts NumPy and Path data types to JSON-serializable primitives."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]
    elif hasattr(obj, "__fspath__"):
        return str(obj)
    return obj


def decode_colab_response_payload(raw_response: Any) -> Dict[str, Any]:
    """Decodes nested responses returned by Google Colab kernel invokeFunction."""
    if isinstance(raw_response, dict):
        if "data" in raw_response:
            inner = raw_response["data"]
            if isinstance(inner, dict):
                if "application/json" in inner:
                    return decode_colab_response_payload(inner["application/json"])
                if "text/plain" in inner:
                    return decode_colab_response_payload(inner["text/plain"])
        return {str(k): make_json_safe(v) for k, v in raw_response.items()}

    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
            return decode_colab_response_payload(parsed)
        except Exception:
            return {"raw_text": raw_response}

    return {"value": make_json_safe(raw_response)}


def colab_callback_health_check() -> Dict[str, Any]:
    """Health check ping verifying JavaScript <-> Python IPC in Colab."""
    return make_json_safe({
        "status": "OK",
        "message": "Google Colab callback bridge is operational.",
        "success": True
    })


def register_colab_callbacks(
    save_handler=None,
    skip_handler=None,
) -> Dict[str, Any]:
    """
    Registers Google Colab kernel callbacks idempotently.
    Supports notebook.save_mask, notebook.skip_image, and test.callback.
    """
    try:
        from google.colab import output # type: ignore

        # Default to pipeline handlers if not explicitly provided
        if save_handler is None or skip_handler is None:
            from .pipeline import colab_save_mask_handler, colab_skip_image_handler
            save_handler = save_handler or colab_save_mask_handler
            skip_handler = skip_handler or colab_skip_image_handler

        output.register_callback("notebook.save_mask", save_handler)
        output.register_callback("notebook.skip_image", skip_handler)
        output.register_callback("test.callback", colab_callback_health_check)

        return {
            "status": "OK",
            "success": True,
            "callbacks": [
                "notebook.save_mask",
                "notebook.skip_image",
                "test.callback"
            ]
        }
    except ImportError:
        # Running outside Google Colab (e.g. local Python / automated test suite)
        return {
            "status": "LOCAL_ENVIRONMENT",
            "success": True,
            "callbacks": [
                "notebook.save_mask",
                "notebook.skip_image",
                "test.callback"
            ],
            "note": "google.colab not available; running in local headless mode"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "success": False,
            "error": str(e)
        }
