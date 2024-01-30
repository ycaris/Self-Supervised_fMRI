import os 
import numpy as np
from scipy.io import loadmat


file_path = ''
data = loadmat(file_path)
print(data['ids'])