"""
Cashew Pest and Disease Diagnosis System
Phase C: Canvas History & Undo/Redo State Model
Framework: TensorFlow / Keras
"""

from typing import List, Optional
import numpy as np
from .config import SegmentationConfig


class CanvasHistoryStateModel:
    """
    Lightweight, memory-bounded, deep-copy safe canvas history model.
    Maintains undo/redo stacks for lesion mask annotation.
    """
    MAX_HISTORY: int = 25

    def __init__(self, width: int = 224, height: int = 224):
        self.width: int = int(width)
        self.height: int = int(height)

        self.current_state: np.ndarray = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )
        self.undo_stack: List[np.ndarray] = [self.current_state.copy()]
        self.redo_stack: List[np.ndarray] = []

    def _push_state(self) -> None:
        """Pushes current state onto undo stack and clears redo stack."""
        self.undo_stack.append(self.current_state.copy())
        if len(self.undo_stack) > self.MAX_HISTORY:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def paint_stroke(self, x1: int, x2: int, y1: int, y2: int, code: int) -> None:
        """Applies a rectangular bounding stroke or region and records state."""
        code = int(code)
        if code not in SegmentationConfig.ALLOWED_MASK_VALUES - {0}:
            raise ValueError(f"Invalid paint code: {code}. Allowed: 1-4")

        x_min, x_max = sorted([max(0, int(x1)), min(self.width, int(x2))])
        y_min, y_max = sorted([max(0, int(y1)), min(self.height, int(y2))])

        if x_max > x_min and y_max > y_min:
            self.current_state[y_min:y_max, x_min:x_max] = code

        self._push_state()

    def undo_stroke(self) -> bool:
        """Reverts to previous canvas state. Returns True if undone, False if at initial state."""
        if len(self.undo_stack) <= 1:
            return False

        popped = self.undo_stack.pop()
        self.redo_stack.append(popped)
        self.current_state = self.undo_stack[-1].copy()
        return True

    def redo_stroke(self) -> bool:
        """Restores previously undone canvas state. Returns True if redone, False if no redo states."""
        if not self.redo_stack:
            return False

        state = self.redo_stack.pop()
        self.undo_stack.append(state.copy())
        self.current_state = state.copy()
        return True

    def clear_canvas(self) -> None:
        """Resets current state to all zeros while recording an undo point."""
        self.current_state = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )
        self._push_state()

    def load_next_image(self, width: int = 224, height: int = 224) -> None:
        """Resets canvas dimensions and history for a new image."""
        self.width = int(width)
        self.height = int(height)
        self.current_state = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )
        self.undo_stack = [self.current_state.copy()]
        self.redo_stack.clear()
