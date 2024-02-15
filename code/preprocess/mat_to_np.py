import os
import numpy as np
from scipy.io import loadmat


mat_path1 = '/gpfs/gibbs/project/dvornek/yz2337/multi_fmri/data/ACE_Wave1/matlab/ACE_info_biopoint.mat'
mat_path2 = '/gpfs/gibbs/project/dvornek/yz2337/multi_fmri/data/ACE_Wave1/matlab/biopoint_aal.mat'

save_path = '/gpfs/gibbs/project/dvornek/yz2337/multi_fmri/data/biopoint'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# print an example subject
info = loadmat(mat_path1)
print(info['ids'][1][0][0])
points = loadmat(mat_path2)
print(points['X'][1][0].shape)


ids = info['ids']  # This should be an array of names
x = points['X']  # This should be an array of matrices

# Check if the lengths of subject names and matrices are equal
if len(ids) != len(x):
    raise ValueError("The number of subject names and matrices do not match")

# Save each matrix as a .npy file
for i, id in enumerate(ids):
    # Assuming subject_name is an array or list-like, use subject_name[0] to get the actual name
    filename = os.path.join(save_path, f"{id[0][0]}.npy")
    np.save(filename, x[i][0])

    if (x[i][0].shape != (154, 116)):  ## the shape of time series of fmri should be 154 time points, 116 features
        print(id)
        print(x[i][0].shape)
