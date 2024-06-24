import scipy.io
import numpy as np
import json
import os
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold

# Load the .mat files
mat_file = '/home/yz2337/project/multi_fmri/data/ACE_Wave1/ace_resting_timeseries_aal_dx.mat'
mat_file2 = '/home/yz2337/project/multi_fmri/data/ACE_Wave1/ace_resting_timeseries_aal_srs.mat'
data = scipy.io.loadmat(mat_file)
data2 = scipy.io.loadmat(mat_file2)

# Extract variables 'X' and 'dx'
X = data['X'].flatten()  # Assuming X is of shape (290, 1)
dx = data['dx'].flatten()  # Assuming dx is of shape (290, 1)
# dx = data2['srs_norm'].flatten()

# Create directory to save numpy files
output_dir = '/home/yz2337/project/multi_fmri/data/ACE_Wave1/numpy_norm'
os.makedirs(output_dir, exist_ok=True)

# Initialize KFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Filter out subjects with less than 64 entries
valid_indices = [i for i in range(len(X)) if X[i].shape[0] > 64]
X_valid = X[valid_indices]
dx_valid = dx[valid_indices]

# Split the data into 5 folds and save each fold separately
for fold, (train_index, test_index) in enumerate(skf.split(X_valid, dx_valid)):
    # for fold, (train_index, test_index) in enumerate(kf.split(X_valid), start=1):
    json_fold = {'train': [], 'val': [], 'test': []}
    fold_name = f'fold_{fold}'

    # Split train_index into train and validation sets
    train_idx, val_idx = train_test_split(
        train_index, test_size=0.1, random_state=42, stratify=dx_valid[train_index])
    # train_idx, val_idx = train_test_split(
    #     train_index, test_size=0.1, random_state=42)

    # Create JSON entries for training set
    for i in train_idx:
        filename = f'{valid_indices[i]+1:03}.npy'
        json_entry = {
            "id": f"{valid_indices[i]+1:03}",
            "group": dx_valid[i].item()
        }
        json_fold['train'].append(json_entry)

    # Create JSON entries for validation set
    for i in val_idx:
        filename = f'{valid_indices[i]+1:03}.npy'
        json_entry = {
            "id": f"{valid_indices[i]+1:03}",
            "group": dx_valid[i].item()
        }
        json_fold['val'].append(json_entry)

    # Create JSON entries for test set
    for i in test_index:
        filename = f'{valid_indices[i]+1:03}.npy'
        json_entry = {
            "id": f"{valid_indices[i]+1:03}",
            "group": dx_valid[i].item()
        }
        json_fold['test'].append(json_entry)

    # Save each fold's JSON file
    json_file_fold = f'/home/yz2337/project/multi_fmri/code/json_files/ace/cls/{fold_name}.json'
    with open(json_file_fold, 'w') as f:
        json.dump(json_fold, f, indent=4)

print("Files saved successfully.")
