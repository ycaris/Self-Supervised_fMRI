import numpy as np
import os
from scipy.io import savemat


def normalize_columns(arr):
    """
    Normalize the columns of a 2D numpy array.
    Each column will have mean = 0 and std = 1 after normalization.
    """
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    normalized = (arr - mean) / std
    return normalized


def load_and_normalize_files(folder_path, save_path, save_mat_path):
    """
    Load all numpy files from the specified folder and normalize each 2D array column-wise.
    """
    normalized_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            # Load the numpy file
            data = np.load(file_path)
            normalized = normalize_columns(data)

        # save normalized data
            np.save(os.path.join(save_path, file_name), normalized)
            savemat(os.path.join(save_mat_path, file_name.split(
                '.npy')[0] + '.mat'), {'data': normalized})

    return normalized_data


# Example usage
folder_path = '/home/yz2337/project/multi_fmri/data/ABIDE/numpy'
save_path = '/home/yz2337/project/multi_fmri/data/ABIDE/numpy_norm'
save_mat_path = '/home/yz2337/project/multi_fmri/data/ABIDE/mat_norm'
normalized_arrays = load_and_normalize_files(
    folder_path, save_path, save_mat_path)

# You can process the normalized_arrays as needed
