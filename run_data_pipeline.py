"""
Cashew Pest and Disease Diagnosis System
Phase 2: Main Data Pipeline & Preprocessing Execution Entrypoint
Framework: TensorFlow / Keras
"""

import os
import sys

from setup_drive_structure import setup_project_structure
from src.config import Config
from src.utils import set_seed, get_logger, get_optimal_batch_size
from src.dataset import (
    create_reproducible_splits,
    build_tf_data_pipelines,
    visualize_and_save_dataset_samples
)

logger = get_logger("Phase2_DataPipeline")

def main():
    print("=" * 75)
    print("   CASHEW PEST & DISEASE DIAGNOSIS - TENSORFLOW DATA PIPELINE (PHASE 2)  ")
    print("=" * 75)

    # 1. Initialize Google Drive / Project Directory Hierarchy
    project_root = setup_project_structure()
    logger.info(f"Project root resolved at: {project_root}")

    # 2. Fix global random seed for 100% reproducible splits
    set_seed(Config.SEED)

    # 3. Detect optimal batch size based on available GPU memory
    optimal_batch_size = get_optimal_batch_size()
    logger.info(f"Optimal batch size configured: {optimal_batch_size}")

    # 4. Execute dataset verification, duplicate detection, 70/15/15 split & class weighting
    logger.info("Executing dataset integrity check, MD5 duplicate check & 70/15/15 stratified split...")
    split_info = create_reproducible_splits(seed=Config.SEED)

    # 5. Build TensorFlow tf.data Pipelines (Shuffle, Batch, Prefetch, Parallel loading, Cache, Augmentations)
    logger.info("Constructing TensorFlow tf.data Pipelines...")
    tf_pipelines = build_tf_data_pipelines(split_info, batch_size=optimal_batch_size)

    # 6. Generate sample batch visualizations, class distribution plot & statistics CSV
    logger.info("Generating dataset visualizations and summary statistics...")
    stats_df = visualize_and_save_dataset_samples(split_info)

    # 7. Display Summary Statistics
    print("\n" + "=" * 75)
    print("                  COMPLETE DATASET STATISTICS TABLE                 ")
    print("=" * 75)
    print(stats_df.to_string(index=False))
    print("=" * 75)

    print("\n[GOOGLE DRIVE SAVED ARTIFACTS]")
    print(f"  ├── Logs Directory:          {Config.get_logs_dir()}")
    print(f"  │    ├── corrupted_images.log")
    print(f"  │    └── duplicate_images.log")
    print(f"  ├── Preprocessed Directory:  {Config.get_preprocessed_dir()}")
    print(f"  │    ├── train_split.csv")
    print(f"  │    ├── val_split.csv")
    print(f"  │    ├── test_split.csv")
    print(f"  │    ├── class_weights.json")
    print(f"  │    └── dataset_statistics.csv")
    print(f"  └── Documentation Directory: {Config.get_documentation_dir()}")
    print(f"       ├── class_distribution.png")
    print(f"       ├── sample_train_batch.png")
    print(f"       ├── sample_val_batch.png")
    print(f"       └── sample_test_batch.png")

    print("\n" + "=" * 75)
    print("  [SUCCESS] Phase 2 Data Pipeline Execution Completed!")
    print("  NO MODEL TRAINING HAS BEEN RUN.")
    print("  Awaiting user confirmation before proceeding to Phase 3 (Model Training).")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()
