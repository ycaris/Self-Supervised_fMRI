import os
import json
import random

def split_data(data_folder, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    # Ensure the percentages sum up to 1
    assert train_pct + val_pct + test_pct == 1, "Percentages must sum up to 1"

    # List all files in the data folder
    files = os.listdir(data_folder)
    
    # Shuffle files to ensure random distribution
    random.shuffle(files)
    
    # Calculate split sizes
    total_files = len(files)
    train_size = int(total_files * train_pct)
    val_size = int(total_files * val_pct)
    
    # Split the files
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]
    
    # Save the splits to a JSON file
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    with open('/home/yz2337/project/multi_fmri/code/json_files/pretrained/data_splits.json', 'w') as f:
        json.dump(splits, f, indent=4)

# Set your data folder path
data_folder = '/home/yz2337/project/multi_fmri/data/ABIDE/numpy'
# Optionally adjust the percentages
split_data(data_folder, train_pct=0.7, val_pct=0.15, test_pct=0.15)
