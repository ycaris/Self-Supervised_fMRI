import pandas as pd
import json

# Load the CSV file
csv_file_path = '/home/yz2337/project/multi_fmri/data/ABIDE/Phenotypic_V1_0b_preprocessed1.csv'
df = pd.read_csv(csv_file_path)

# Extract the relevant columns
data = df[['FILE_ID', 'DX_GROUP', 'func_mean_fd']]
data = data[(data['FILE_ID'] != "no_filename") & (data['func_mean_fd'] < 0.2)]
data = data[['FILE_ID', 'DX_GROUP']]

data_pos = data[(data['DX_GROUP'] == 1)]
data_neg = data[(data['DX_GROUP'] == 2)]


data_pos.to_csv('/home/yz2337/project/multi_fmri/data/ABIDE/autism.csv',
                encoding='utf-8', index=False)
data_neg.to_csv('/home/yz2337/project/multi_fmri/data/ABIDE/control.csv',
                encoding='utf-8', index=False)
