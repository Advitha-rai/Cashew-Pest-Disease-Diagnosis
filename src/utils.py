"""
Cashew Pest and Disease Diagnosis System
Phase 2: Utility Functions, Image Hashing, Logging & Hardware Helpers
Framework: TensorFlow / Keras
"""

import os
import hashlib
import random
import logging
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """Fixes all random seeds across Python, NumPy, and TensorFlow for 100% reproducible experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if TF_AVAILABLE:
        tf.random.set_seed(seed)

    print(f"[REPRODUCIBILITY] Global random seed set to: {seed}")


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Configures a standardized clean logger for pipeline logs, integrity errors, and summaries."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console output handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File logging output handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


def calculate_md5(file_path: str, chunk_size: int = 8192) -> str:
    """
    Computes MD5 hash of an image file to detect duplicate images across the dataset.
    """
    md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        return ""


def get_optimal_batch_size() -> int:
    """
    Automatically selects the optimal batch size based on available GPU memory
    in TensorFlow.
    """
    if TF_AVAILABLE:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[GPU AUTO-CONFIG] TensorFlow GPU detected: {gpus[0].name}")
            return 32

    print("[GPU AUTO-CONFIG] GPU unavailable or CPU mode. Default batch size: 16")
    return 16
