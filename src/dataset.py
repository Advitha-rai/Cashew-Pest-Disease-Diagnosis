"""
Cashew Pest and Disease Diagnosis System
Phase 2: Complete TensorFlow / Keras Data Pipeline
Includes Integrity Verification, Duplicate Detection, Stratified Splitting,
Class Weight Calculation, tf.data Pipeline Construction, Visualizations & Drive Exports
"""

import os
import glob
import json
import shutil
from typing import Tuple, Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.config import Config
from src.utils import set_seed, get_logger, calculate_md5, get_optimal_batch_size

# Allow PIL to safely verify truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger("DatasetPipeline")


# ---------------------------------------------------------
# 1. Dataset Integrity Verification & Duplicate Detection
# ---------------------------------------------------------
def verify_and_clean_dataset(raw_dir: str, cleaned_dir: str) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Scans the raw dataset folder from Google Drive, automatically detects folder names as class labels,
    verifies image integrity (PIL + OpenCV decoding), detects duplicate images using MD5 hashing,
    logs corrupted and duplicate images directly to Google Drive Logs/, and returns valid file paths.
    """
    os.makedirs(cleaned_dir, exist_ok=True)
    logs_dir = Config.get_logs_dir()
    
    corrupt_log_path = os.path.join(logs_dir, "corrupted_images.log")
    duplicate_log_path = os.path.join(logs_dir, "duplicate_images.log")
    
    corrupt_logger = get_logger("CorruptCheck", corrupt_log_path)
    duplicate_logger = get_logger("DuplicateCheck", duplicate_log_path)

    valid_paths = []
    valid_labels = []
    corrupt_files = []
    duplicate_files = []
    seen_hashes = {}
    class_counts = {}

    if not os.path.exists(raw_dir):
        logger.warning(f"Raw directory '{raw_dir}' does not exist. Creating default directory structure.")
        os.makedirs(raw_dir, exist_ok=True)

    # Automatically detect folder names inside Raw directory as class labels
    detected_classes = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    if not detected_classes:
        logger.warning(f"No class subfolders found in {raw_dir}. Initializing default class folders: {Config.DEFAULT_CLASSES}")
        detected_classes = Config.DEFAULT_CLASSES
        for c in detected_classes:
            os.makedirs(os.path.join(raw_dir, c), exist_ok=True)

    logger.info(f"[DYNAMIC CLASS DETECTION] Detected {len(detected_classes)} target classes: {detected_classes}")

    for class_name in detected_classes:
        class_raw_path = os.path.join(raw_dir, class_name)
        class_clean_path = os.path.join(cleaned_dir, class_name)
        os.makedirs(class_clean_path, exist_ok=True)

        image_files = glob.glob(os.path.join(class_raw_path, "*.*"))
        valid_in_class = 0

        for img_path in image_files:
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                continue

            # Integrity Check: PIL open + OpenCV decode
            try:
                with Image.open(img_path) as img:
                    img.verify()

                cv_img = cv2.imread(img_path)
                if cv_img is None or cv_img.size == 0:
                    raise ValueError("OpenCV failed to decode image frame.")

            except Exception as e:
                msg = f"Corrupt image skipped: {img_path} | Reason: {str(e)}"
                corrupt_logger.warning(msg)
                corrupt_files.append(img_path)
                continue

            # Duplicate Check: MD5 Hashing
            img_hash = calculate_md5(img_path)
            if img_hash and img_hash in seen_hashes:
                original_file = seen_hashes[img_hash]
                msg = f"Duplicate image skipped: {img_path} (Duplicate of: {original_file})"
                duplicate_logger.info(msg)
                duplicate_files.append(img_path)
                continue
            elif img_hash:
                seen_hashes[img_hash] = img_path

            # Copy verified clean image to Cleaned directory
            dest_path = os.path.join(class_clean_path, os.path.basename(img_path))
            if not os.path.exists(dest_path):
                shutil.copy2(img_path, dest_path)

            valid_paths.append(dest_path)
            valid_labels.append(class_name)
            valid_in_class += 1

        class_counts[class_name] = valid_in_class

    logger.info("[INTEGRITY & DUPLICATE VERIFICATION COMPLETE]")
    logger.info(f"  - Total Valid Images:     {len(valid_paths)}")
    logger.info(f"  - Corrupt Images Logged:  {len(corrupt_files)} -> {corrupt_log_path}")
    logger.info(f"  - Duplicate Images Logged:{len(duplicate_files)} -> {duplicate_log_path}")
    
    return valid_paths, valid_labels, class_counts


# ---------------------------------------------------------
# Synthetic Dataset Generator for Testing / Demo Purposes
# ---------------------------------------------------------
def create_synthetic_cashew_dataset(dest_dir: str, num_per_class: int = 25) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Creates synthetic RGB images for testing the pipeline if Google Drive Raw dataset is empty."""
    os.makedirs(dest_dir, exist_ok=True)
    file_paths = []
    labels = []
    counts = {}

    for cls in Config.DEFAULT_CLASSES:
        cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        counts[cls] = num_per_class

        for i in range(num_per_class):
            img_name = f"{cls}_farm_{i+1:03d}.jpg"
            img_path = os.path.join(cls_dir, img_name)

            if not os.path.exists(img_path):
                img_data = np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)
                img_data[:, :, 1] = np.clip(img_data[:, :, 1] + 50, 0, 255)
                cv2.imwrite(img_path, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR))

            file_paths.append(img_path)
            labels.append(cls)

    logger.info(f"[SYNTHETIC DATASET CREATED] Generated {len(file_paths)} RGB images across {len(Config.DEFAULT_CLASSES)} classes at: {dest_dir}")
    return file_paths, labels, counts


# ---------------------------------------------------------
# 2. Reproducible Stratified Train / Val / Test Splitter
# ---------------------------------------------------------
def create_reproducible_splits(
    raw_dir: Optional[str] = None,
    seed: int = Config.SEED
) -> Dict:
    """
    Creates 70% Train, 15% Validation, and 15% Testing stratified splits with a fixed random seed (42).
    Saves split metadata (train_split.csv, val_split.csv, test_split.csv) and class_weights.json
    directly into Google Drive Preprocessed/ folder for 100% reproducibility across all models.
    """
    set_seed(seed)
    
    if raw_dir is None:
        raw_dir = Config.get_raw_dir()

    cleaned_dir = Config.get_cleaned_dir()
    preprocessed_dir = Config.get_preprocessed_dir()
    
    # Verify dataset integrity & duplicates
    file_paths, class_names_list, class_counts = verify_and_clean_dataset(raw_dir, cleaned_dir)

    # Fallback to synthetic dataset if Raw folder is empty
    if len(file_paths) == 0:
        logger.warning("[DATASET NOTICE] Raw folder is empty. Generating synthetic sample dataset for testing...")
        file_paths, class_names_list, class_counts = create_synthetic_cashew_dataset(cleaned_dir)

    unique_classes = sorted(list(set(class_names_list)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    idx_to_class = {i: c for i, c in enumerate(unique_classes)}
    numeric_labels = [class_to_idx[c] for c in class_names_list]

    # Stratified 70% Train, 30% Temp (Val + Test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths,
        numeric_labels,
        test_size=(Config.VAL_RATIO + Config.TEST_RATIO),
        stratify=numeric_labels,
        random_state=seed
    )

    # Stratified 15% Validation, 15% Test (Split temp 50-50)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,
        stratify=temp_labels,
        random_state=seed
    )

    logger.info(f"[STRATIFIED SPLIT COMPLETE] Seed={seed}")
    logger.info(f"  - Training Set (70%):   {len(train_paths)} samples")
    logger.info(f"  - Validation Set (15%): {len(val_paths)} samples")
    logger.info(f"  - Testing Set (15%):    {len(test_paths)} samples")

    # Save Split Information CSVs to Google Drive Preprocessed folder
    pd.DataFrame({"file_path": train_paths, "class_name": [idx_to_class[l] for l in train_labels], "label": train_labels}).to_csv(os.path.join(preprocessed_dir, "train_split.csv"), index=False)
    pd.DataFrame({"file_path": val_paths, "class_name": [idx_to_class[l] for l in val_labels], "label": val_labels}).to_csv(os.path.join(preprocessed_dir, "val_split.csv"), index=False)
    pd.DataFrame({"file_path": test_paths, "class_name": [idx_to_class[l] for l in test_labels], "label": test_labels}).to_csv(os.path.join(preprocessed_dir, "test_split.csv"), index=False)

    # Compute Class Weights for Imbalance Handling
    class_sample_counts = np.bincount(train_labels, minlength=len(unique_classes))
    total_train_samples = len(train_labels)
    class_weights = total_train_samples / (len(unique_classes) * np.maximum(class_sample_counts, 1))
    
    # Save Class Weights to JSON in Google Drive
    weights_dict = {idx_to_class[i]: float(w) for i, w in enumerate(class_weights)}
    with open(os.path.join(preprocessed_dir, "class_weights.json"), "w") as f:
        json.dump(weights_dict, f, indent=4)
        
    logger.info(f"[CLASS WEIGHTS SAVED] {weights_dict} -> {os.path.join(preprocessed_dir, 'class_weights.json')}")

    return {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "test_paths": test_paths, "test_labels": test_labels,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "class_names": unique_classes,
        "class_weights": class_weights,
        "weights_dict": weights_dict,
        "class_counts": class_counts
    }


# ---------------------------------------------------------
# 3. TensorFlow Data Pipeline Construction (tf.data)
# ---------------------------------------------------------
def parse_and_augment_image(file_path, label, augment=False, img_size=Config.IMG_SIZE, num_classes=4):
    """
    Parses RGB image file, resizes to (224, 224), normalizes pixels to [0, 1],
    applies realistic training augmentations, and converts label to one-hot vector (num_classes,).
    """
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [img_size[0], img_size[1]])
    
    if augment:
        # 1. Horizontal Flip
        img = tf.image.random_flip_left_right(img)
        # 2. Brightness Adjustment
        img = tf.image.random_brightness(img, max_delta=0.15)
        # 3. Contrast Adjustment
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
        # 4. Small Gaussian Noise
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=4.0)
        img = tf.clip_by_value(img + noise, 0.0, 255.0)

    # Pixel normalization to [0, 1] preserving RGB
    img = tf.cast(img, tf.float32) / 255.0

    # One-hot encode label to shape (num_classes,)
    one_hot_label = tf.one_hot(tf.cast(label, tf.int32), depth=num_classes)
    return img, one_hot_label


def build_tf_data_pipelines(
    split_info: Dict,
    batch_size: int = Config.BATCH_SIZE,
    img_size: Tuple[int, int] = Config.IMG_SIZE
) -> Dict:
    """
    Creates efficient TensorFlow tf.data.Dataset pipelines with:
    Shuffle, Batch, Prefetch (AUTOTUNE), Parallel loading (num_parallel_calls=AUTOTUNE),
    Caching, One-Hot Encoded Targets (batch_size, num_classes), and Realistic Training Augmentations.
    """
    if not TF_AVAILABLE:
        logger.warning("[TF NOTICE] TensorFlow is not installed in environment. Skipping tf.data building.")
        return None

    num_classes = len(split_info["class_names"])

    def create_dataset(file_paths, labels, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        if is_training:
            ds = ds.shuffle(buffer_size=len(file_paths), seed=Config.SEED)
            ds = ds.map(
                lambda x, y: parse_and_augment_image(x, y, augment=True, img_size=img_size, num_classes=num_classes),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                lambda x, y: parse_and_augment_image(x, y, augment=False, img_size=img_size, num_classes=num_classes),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    train_ds = create_dataset(split_info["train_paths"], split_info["train_labels"], is_training=True)
    val_ds = create_dataset(split_info["val_paths"], split_info["val_labels"], is_training=False)
    test_ds = create_dataset(split_info["test_paths"], split_info["test_labels"], is_training=False)

    logger.info(f"[TF.DATA PIPELINES CREATED] Batch Size={batch_size} | Image Resolution={img_size} | Num Classes={num_classes}")
    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds
    }


# ---------------------------------------------------------
# 4. Visualizations & Report Export Functions
# ---------------------------------------------------------
def visualize_and_save_dataset_samples(split_info: Dict, save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Generates sample image grids (Image + Class Name) for Train, Validation, and Testing sets,
    creates the Class Distribution bar chart, and exports summary statistics CSV to Google Drive.
    """
    if save_dir is None:
        save_dir = Config.get_documentation_dir()
    os.makedirs(save_dir, exist_ok=True)

    # 1. Class Distribution Bar Graph
    plt.figure(figsize=(10, 5))
    classes = split_info["class_names"]
    
    train_counts = [np.sum(np.array(split_info["train_labels"]) == i) for i in range(len(classes))]
    val_counts = [np.sum(np.array(split_info["val_labels"]) == i) for i in range(len(classes))]
    test_counts = [np.sum(np.array(split_info["test_labels"]) == i) for i in range(len(classes))]

    x = np.arange(len(classes))
    width = 0.25

    plt.bar(x - width, train_counts, width, label='Train (70%)', color='#2b5c8f')
    plt.bar(x, val_counts, width, label='Val (15%)', color='#e07a5f')
    plt.bar(x + width, test_counts, width, label='Test (15%)', color='#81b29a')

    plt.xlabel('Cashew Pest & Disease Class', fontsize=12, fontweight='bold')
    plt.ylabel('Number of RGB Images', fontsize=12, fontweight='bold')
    plt.title('Cashew Dataset Stratified Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(x, classes, rotation=15)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    dist_chart_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(dist_chart_path, dpi=300)
    plt.close()
    logger.info(f"[GRAPH SAVED] Class distribution chart saved to: {dist_chart_path}")

    # 2. Display & Save Sample Images (Train, Val, Test)
    for split_name, paths, labels in [
        ("Train", split_info["train_paths"], split_info["train_labels"]),
        ("Val", split_info["val_paths"], split_info["val_labels"]),
        ("Test", split_info["test_paths"], split_info["test_labels"])
    ]:
        num_samples = min(8, len(paths))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()

        for idx in range(num_samples):
            img_p = paths[idx]
            lbl = labels[idx]
            cls_name = split_info["idx_to_class"][lbl]

            cv_img = cv2.imread(img_p)
            if cv_img is not None:
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                rgb_img = cv2.resize(rgb_img, (224, 224))
            else:
                rgb_img = np.zeros((224, 224, 3), dtype=np.uint8)

            axes[idx].imshow(rgb_img)
            axes[idx].set_title(f"{cls_name}", fontsize=10, fontweight='bold')
            axes[idx].axis('off')

        for idx in range(num_samples, 8):
            axes[idx].axis('off')

        plt.suptitle(f"Cashew Dataset - {split_name} Sample Batch", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        sample_grid_path = os.path.join(save_dir, f"sample_{split_name.lower()}_batch.png")
        plt.savefig(sample_grid_path, dpi=300)
        plt.close()
        logger.info(f"[VISUALIZATION SAVED] {split_name} sample grid saved to: {sample_grid_path}")

    # 3. Export Summary Statistics CSV to Google Drive Preprocessed folder
    stats_data = []
    preprocessed_dir = Config.get_preprocessed_dir()
    for i, c_name in enumerate(classes):
        stats_data.append({
            "Class Name": c_name,
            "Total Clean Images": train_counts[i] + val_counts[i] + test_counts[i],
            "Train Samples (70%)": train_counts[i],
            "Val Samples (15%)": val_counts[i],
            "Test Samples (15%)": test_counts[i],
            "Calculated Class Weight": round(float(split_info["class_weights"][i]), 4)
        })

    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = os.path.join(preprocessed_dir, "dataset_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    logger.info(f"[STATISTICS SAVED] Dataset statistics saved to: {stats_csv_path}")

    return stats_df
