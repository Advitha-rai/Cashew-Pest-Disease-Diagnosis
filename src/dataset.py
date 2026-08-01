"""
Cashew Pest and Disease Diagnosis System
Dataset Pipeline, Data Preprocessing, Integrity Verification & DataLoaders
"""

import os
import glob
import shutil
import logging
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import cv2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from src.config import Config
from src.utils import set_seed, get_logger

# Allow PIL to load slightly corrupt/truncated images safely during verification phase
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger("DatasetPipeline")

# ---------------------------------------------------------
# Albumentations Transformations Engine
# ---------------------------------------------------------
def get_transforms(img_size: int = 224) -> Tuple[A.Compose, A.Compose]:
    """
    Constructs realistic, domain-specific Albumentations transforms.
    
    Training Augmentations:
      - Horizontal Flip (p=0.5)
      - Small Rotation (±15°)
      - Random Brightness & Contrast (p=0.4)
      - Shift, Scale, & Zoom (p=0.3)
      - Subtle Gaussian Noise (p=0.2)
      
    Validation / Test Transforms:
      - Resize only + Normalization
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_test_transform

# ---------------------------------------------------------
# PyTorch Custom Dataset Class
# ---------------------------------------------------------
class CashewDataset(Dataset):
    """
    PyTorch Dataset wrapper for Cashew Pest and Disease images.
    Supports Albumentations transformations.
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform: Optional[A.Compose] = None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load RGB image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            # Fallback PIL reader if opencv fails
            pil_img = Image.open(img_path).convert('RGB')
            image = np.array(pil_img)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        return image, label

# ---------------------------------------------------------
# Data Cleaning & Integrity Verification
# ---------------------------------------------------------
def verify_and_clean_dataset(raw_dir: str, cleaned_dir: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Scans the raw dataset directory, verifies image integrity (PIL decode + OpenCV check),
    removes corrupt files, copies valid files to cleaned_dir, and returns valid file paths.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    valid_paths = []
    valid_labels = []
    corrupt_files = []
    class_counts = {}

    if not os.path.exists(raw_dir):
        logger.warning(f"Raw directory '{raw_dir}' does not exist. Creating empty structure.")
        os.makedirs(raw_dir, exist_ok=True)
        return [], [], {}

    classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    if not classes:
        logger.warning(f"No subfolders found in {raw_dir}. Assuming dataset classes from default Config.")
        classes = Config.DEFAULT_CLASSES
        for c in classes:
            os.makedirs(os.path.join(raw_dir, c), exist_ok=True)

    logger.info(f"Detected {len(classes)} target classes: {classes}")

    for class_name in classes:
        class_raw_path = os.path.join(raw_dir, class_name)
        class_clean_path = os.path.join(cleaned_dir, class_name)
        os.makedirs(class_clean_path, exist_ok=True)

        image_files = glob.glob(os.path.join(class_raw_path, "*.*"))
        valid_in_class = 0

        for img_path in image_files:
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue

            try:
                # Test PIL opening & verification
                with Image.open(img_path) as img:
                    img.verify()
                
                # Test OpenCV decoding
                cv_img = cv2.imread(img_path)
                if cv_img is None or cv_img.size == 0:
                    raise ValueError("OpenCV failed to decode image frame.")

                # Target cleaned file path
                dest_path = os.path.join(class_clean_path, os.path.basename(img_path))
                if not os.path.exists(dest_path):
                    shutil.copy2(img_path, dest_path)

                valid_paths.append(dest_path)
                valid_labels.append(class_name)
                valid_in_class += 1

            except Exception as e:
                logger.warning(f"Corrupt or unreadable image removed: {img_path} | Error: {str(e)}")
                corrupt_files.append(img_path)

        class_counts[class_name] = valid_in_class

    logger.info(f"[INTEGRITY VERIFICATION COMPLETE]")
    logger.info(f"  - Total Valid Images: {len(valid_paths)}")
    logger.info(f"  - Total Corrupt Images Removed: {len(corrupt_files)}")
    return valid_paths, valid_labels, class_counts

# ---------------------------------------------------------
# Stratified Train / Validation / Test Splitter
# ---------------------------------------------------------
def prepare_dataset_splits(
    base_dir: Optional[str] = None,
    seed: int = Config.SEED,
    img_size: int = 224
) -> Dict:
    """
    Executes automated dataset verification, 70/15/15 stratified split,
    calculates class weights for handling class imbalance, and sets up data paths.
    """
    set_seed(seed)
    
    if base_dir is None:
        base_dir = Config.get_base_dir()

    raw_dir = os.path.join(base_dir, "Dataset", "Raw")
    cleaned_dir = os.path.join(base_dir, "Dataset", "Cleaned")
    
    # 1. Clean & verify dataset
    file_paths, class_names_list, class_counts = verify_and_clean_dataset(raw_dir, cleaned_dir)

    # Fallback synthetic dummy dataset generator if raw directory is completely empty
    if len(file_paths) == 0:
        logger.warning("[DATASET WARNING] No images found in Dataset/Raw. Creating synthetic sample dataset for testing...")
        file_paths, class_names_list, class_counts = create_synthetic_cashew_dataset(cleaned_dir)

    unique_classes = sorted(list(set(class_names_list)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for i, c in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[c] for c in class_names_list]

    # 2. Stratified 70% Train, 30% Temp (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        numeric_labels,
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=numeric_labels,
        random_state=seed
    )

    # 3. Stratified 15% Validation, 15% Test (Split temp 50-50)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    logger.info("[DATASET SPLIT COMPLETE (Fixed Seed = 42)]")
    logger.info(f"  - Training Set (70%):   {len(train_paths)} samples")
    logger.info(f"  - Validation Set (15%): {len(val_paths)} samples")
    logger.info(f"  - Test Set (15%):       {len(test_paths)} samples")

    # 4. Calculate Class Weights for Imbalance Handling
    class_sample_counts = np.bincount(train_labels, minlength=len(unique_classes))
    total_samples = len(train_labels)
    class_weights = total_samples / (len(unique_classes) * np.maximum(class_sample_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Return dataset split dictionary
    return {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "test_paths": test_paths, "test_labels": test_labels,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": unique_classes,
        "class_weights": class_weights_tensor,
        "class_counts": class_counts
    }

# ---------------------------------------------------------
# PyTorch DataLoaders Factory
# ---------------------------------------------------------
def get_dataloaders(
    split_info: Dict,
    batch_size: int = 32,
    img_size: int = 224,
    use_weighted_sampler: bool = True,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for Train, Validation, and Test sets with optional weighted sampler.
    """
    train_transform, val_test_transform = get_transforms(img_size=img_size)

    train_dataset = CashewDataset(split_info["train_paths"], split_info["train_labels"], transform=train_transform)
    val_dataset = CashewDataset(split_info["val_paths"], split_info["val_labels"], transform=val_test_transform)
    test_dataset = CashewDataset(split_info["test_paths"], split_info["test_labels"], transform=val_test_transform)

    # Imbalance handling via WeightedRandomSampler if requested
    if use_weighted_sampler:
        targets = split_info["train_labels"]
        class_weights = split_info["class_weights"]
        sample_weights = [class_weights[t] for t in targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

# ---------------------------------------------------------
# Synthetic Dataset Generator for Testing / Demo Purposes
# ---------------------------------------------------------
def create_synthetic_cashew_dataset(dest_dir: str, num_per_class: int = 20) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Creates synthetic RGB images for initial pipeline testing if user raw dataset is missing."""
    os.makedirs(dest_dir, exist_ok=True)
    file_paths = []
    labels = []
    counts = {}

    for cls in Config.DEFAULT_CLASSES:
        cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        counts[cls] = num_per_class

        for i in range(num_per_class):
            img_name = f"{cls}_sample_{i+1:03d}.jpg"
            img_path = os.path.join(cls_dir, img_name)

            if not os.path.exists(img_path):
                # Generate synthetic colorful pattern image
                img_data = np.random.randint(50, 220, (224, 224, 3), dtype=np.uint8)
                # Add synthetic leaf-like green tint or pest spots
                img_data[:, :, 1] = np.clip(img_data[:, :, 1] + 40, 0, 255)
                cv2.imwrite(img_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

            file_paths.append(img_path)
            labels.append(cls)

    logger.info(f"[SYNTHETIC DATASET CREATED] {len(file_paths)} images across {len(Config.DEFAULT_CLASSES)} classes at: {dest_dir}")
    return file_paths, labels, counts
