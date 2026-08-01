"""
Cashew Pest and Disease Diagnosis System
Automatic Workspace & Google Drive Directory Hierarchy Generator
"""

import os
from pathlib import Path

# Google Drive / Project Directory Hierarchy Specification
PROJECT_ROOT_NAME = "Cashew_Pest_Disease_Project"

FOLDER_STRUCTURE = [
    "Dataset/Raw",
    "Dataset/Cleaned",
    "Dataset/Train",
    "Dataset/Validation",
    "Dataset/Test",
    "Dataset/Preprocessed",
    "Experiments/01_MobileNetV2",
    "Experiments/02_ResNet50",
    "Experiments/03_VGG16",
    "Experiments/04_InceptionV3",
    "Experiments/05_DenseNet121",
    "Experiments/06_EfficientNetV2",
    "Experiments/07_MobileNetV3",
    "Experiments/08_ConvNeXt",
    "Experiments/09_SwinTransformer",
    "Experiments/10_DINOv2",
    "Experiments/Comparison",
    "Experiments/Ensemble",
    "Saved_Models",
    "Training_Curves",
    "Confusion_Matrices",
    "Classification_Reports",
    "GradCAM",
    "Website",
    "Deployment",
    "Logs",
    "Final_Model",
    "Documentation"
]

def setup_project_structure(base_dir: str = None) -> str:
    """
    Creates the standardized project folder hierarchy automatically.
    
    Args:
        base_dir (str, optional): Base directory where Cashew_Pest_Disease_Project should be created.
                                 If None, detects if running in Google Drive or local environment.
                                 
    Returns:
        str: Absolute path to project root.
    """
    if base_dir is None:
        # Check if running inside Google Colab with Drive mounted
        drive_path = "/content/drive/MyDrive"
        if os.path.exists(drive_path):
            base_dir = drive_path
            print(f"[INFO] Google Drive detected at {drive_path}")
        else:
            base_dir = os.getcwd()
            print(f"[INFO] Local environment detected at {base_dir}")
            
    project_root = os.path.join(base_dir, PROJECT_ROOT_NAME)
    os.makedirs(project_root, exist_ok=True)
    
    created_count = 0
    for folder in FOLDER_STRUCTURE:
        folder_path = os.path.join(project_root, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            created_count += 1
            
    # Also create sub-directories inside each experiment folder
    exp_subfolders = ["GradCAM", "Logs"]
    for i in range(1, 11):
        model_dirs = [
            "01_MobileNetV2", "02_ResNet50", "03_VGG16", "04_InceptionV3",
            "05_DenseNet121", "06_EfficientNetV2", "07_MobileNetV3",
            "08_ConvNeXt", "09_SwinTransformer", "10_DINOv2"
        ]
        exp_dir = os.path.join(project_root, "Experiments", model_dirs[i-1])
        for sub in exp_subfolders:
            os.makedirs(os.path.join(exp_dir, sub), exist_ok=True)

    print(f"[SUCCESS] Workspace initialized at: {project_root}")
    print(f"[INFO] Directories created/verified: {len(FOLDER_STRUCTURE)} primary folders.")
    return project_root

if __name__ == "__main__":
    setup_project_structure()
