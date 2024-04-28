import json
import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedKFold
import random

# Load your MATLAB file
mat = scipy.io.loadmat(
    '/home/yz2337/project/multi_fmri/data/ACE_Wave1/matlab/ACE_info_biopoint.mat')

# Assuming the MATLAB file has 'subject_ids' and 'labels'
# Adjust field names based on your MATLAB file
subject_ids = mat['ids'].squeeze()
labels = mat['group'].squeeze()

pair = list(zip(subject_ids, labels))
random.shuffle(pair)
subject_ids_shuffled, labels_shuffled = zip(* pair)
subject_ids = subject_ids_shuffled
labels = labels_shuffled


# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True)

# Create and save JSON files for each fold
for i, (train_index, test_index) in enumerate(skf.split(subject_ids, labels)):
    train_ids_labels = [{'id': subject_ids[idx][0], 'group': (
        int)(labels[idx])} for idx in train_index[:-20]]
    val_ids_labels = [{'id': subject_ids[idx][0], 'group': (
        int)(labels[idx])} for idx in train_index[-20:]]
    test_ids_labels = [{'id': subject_ids[idx][0],
                        'group': (int)(labels[idx])} for idx in test_index]

    fold_data = {
        'train': train_ids_labels,
        'val': val_ids_labels,
        'test': test_ids_labels
    }

    with open(f'/home/yz2337/project/multi_fmri/code/json_files/fmri_only/fold_{i+1}_points.json', 'w') as file:
        json.dump(fold_data, file, indent=4)

# # This script will create 10 JSON files named 'fold_1.json', 'fold_2.json', ..., 'fold_10.json'
