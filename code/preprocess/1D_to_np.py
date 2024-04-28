import os
import numpy as np


txt_dir = '/home/yz2337/project/multi_fmri/data/ABIDE/download/cpac/filt_noglobal/rois_aal'

save_dir = '/home/yz2337/project/multi_fmri/data/ABIDE/numpy'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save each matrix as a .npy file
for sub in os.listdir(txt_dir):

    # load 1D txt file into numpy
    filename = sub.split('.1D')[0]+'.npy'
    filepath = os.path.join(txt_dir, sub)
    sub_np = np.loadtxt(filepath, skiprows=1)

    # print(sub_np[:,100])  # (176~ x 116), varying time

    # save numpy as new file
    np.save(os.path.join(save_dir, filename), sub_np)
