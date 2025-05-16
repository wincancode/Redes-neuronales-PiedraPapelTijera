import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Set your paths
source_dir = '../processed_dataset_contrast'  # Your main folder with rock/paper/scissors subfolders
output_dir = 'processed_dataset_contrast'  # Where to save the split datasets

# Create output directories
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

# Classes to process
classes = ['rock', 'paper', 'scissors']

for class_name in classes:
    # Get all image files for this class
    class_dir = os.path.join(source_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split into train (70%), temp (30%)
    train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
    
    # Split temp into val (50%) and test (50%) of the remaining 30% (so 15% each)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    # Create class subdirectories in each split
    for split, files in [('train', train_files), ('valid', val_files), ('test', test_files)]:
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)
        
        # Copy files to their new locations
        for f in files:
            src = os.path.join(class_dir, f)
            dst = os.path.join(output_dir, split, class_name, f)
            shutil.copy2(src, dst)
    
    print(f"Class {class_name}:")
    print(f"  Training samples: {len(train_files)}")
    print(f"  Validation samples: {len(val_files)}")
    print(f"  Test samples: {len(test_files)}")

print("\nDataset splitting complete!")