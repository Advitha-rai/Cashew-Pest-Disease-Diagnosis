"""
Cashew Pest and Disease Diagnosis System
Utility Functions & Environment Helper Methods
"""

import os
import random
import logging
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Fixes all random seeds for 100% reproducible experiments."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[REPRODUCIBILITY] Global random seed set to: {seed}")

def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Configures a standardized clean logger for training logs and errors."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File logging output
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_gpu_memory_mb() -> float:
    """Returns currently allocated GPU memory in Megabytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0

def get_optimal_batch_size(model_name: str) -> int:
    """
    Automatically selects the optimal batch size based on available CUDA GPU memory
    and model complexity.
    """
    if not torch.cuda.is_available():
        print("[GPU AUTO-CONFIG] CUDA unavailable. Falling back to CPU batch size: 8")
        return 8
        
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[GPU AUTO-CONFIG] Detected GPU: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")
    
    # Large memory (>12GB VRAM - e.g. T4, V100, A100, RTX 3090/4090)
    if total_vram_gb >= 12.0:
        if any(m in model_name for m in ["swin", "dinov2", "convnext"]):
            return 32
        elif "vgg" in model_name or "inception" in model_name:
            return 32
        else:
            return 64
            
    # Medium memory (6GB - 12GB VRAM - e.g. GTX 1660, RTX 3060, Colab Free T4)
    elif total_vram_gb >= 6.0:
        if any(m in model_name for m in ["swin", "dinov2", "convnext"]):
            return 16
        elif "vgg" in model_name or "inception" in model_name:
            return 16
        else:
            return 32
            
    # Low memory (< 6GB VRAM)
    else:
        if any(m in model_name for m in ["swin", "dinov2", "convnext"]):
            return 8
        else:
            return 16

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calculates total model size in Megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return float(size_all_mb)

def count_parameters(model: torch.nn.Module) -> dict:
    """Counts total and trainable parameters of a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }
