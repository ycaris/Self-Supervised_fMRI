import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load the CSV file
csv_file_path = '/home/yz2337/project/multi_fmri/data/ABIDE/Phenotypic_V1_0b_preprocessed1.csv'
df = pd.read_csv(csv_file_path)

# Extract the relevant columns
data = df[['FILE_ID', 'DX_GROUP', 'func_mean_fd']]
data = data[(data['FILE_ID'] != "no_filename") & (data['func_mean_fd'] < 0.2)]
data = data[['FILE_ID', 'DX_GROUP']]

# Split the data into training, validation, and testing sets
# train_val, test = train_test_split(data, test_size=0.15)
# train, val = train_test_split(train_val, test_size=0.12)
train, val = train_test_split(data, test_size=0.1)

# Convert to desired format for JSON


def data_to_dict(data):
    return [{"id": row['FILE_ID']+'_rois_aal', "group": 0 if row['DX_GROUP'] == 2 else 1} for index, row in data.iterrows()]


# dataset_splits = {
#     "train": data_to_dict(train),
#     "val": data_to_dict(val),
#     "test": data_to_dict(test)
# }

dataset_splits = {
    "train": data_to_dict(train),
    "val": data_to_dict(val)
}

# Save the splits to a JSON file
# Update this path if necessary
json_file_path = '/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group.json'
with open(json_file_path, 'w') as json_file:
    json.dump(dataset_splits, json_file, indent=4)

print(f'Dataset splits saved to {json_file_path}')
