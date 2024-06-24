import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import json

# Load CSV files
positive_df = pd.read_csv(
    '/home/yz2337/project/multi_fmri/data/ABIDE/autism.csv')
negative_df = pd.read_csv(
    '/home/yz2337/project/multi_fmri/data/ABIDE/control.csv')

# Concatenate positive and negative samples into a single DataFrame
df = pd.concat([positive_df, negative_df])

# Define percentages for training splits and the constant percentages for validation and testing
training_percentages = [20, 40, 60, 80, 100]
testing_percentage = 20

# Calculate counts for validation and testing sets
total_samples = len(df)
testing_count = int(total_samples * testing_percentage / 100)
print(f'{testing_count} testing subjects')


# create pretrain json
def create_pretrain(train_df, val_df, fold):
    dataset_splits = {
        "train": data_to_dict(train_df),
        "val": data_to_dict(val_df)
    }
    json_file_path = f'/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group_fold{fold}.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(dataset_splits, json_file, indent=4)
    print(f'Dataset splits saved to {json_file_path}')


# Function to generate and save data splits to JSON files for percentage
def generate_and_save_splits(train_df, fold_idx, percentage, validation_set, testing_set, pretrain_fold_idx):
    # Calculate the count for the current percentage of training data
    train_df_count = len(train_df)
    training_count = int(train_df_count * percentage / 100)

    # choose the percentage of training data
    if percentage == 100:
        training_set = train_df
    else:
        training_set, _ = train_test_split(
            train_df, train_size=training_count, stratify=train_df['DX_GROUP'])

    # Create a dictionary for the current split
    dataset_splits = {
        "train": data_to_dict(training_set),
        "val": data_to_dict(validation_set),
        "test": data_to_dict(testing_set)
    }

    # Save the dictionary to a JSON file
    with open(f'/home/yz2337/project/multi_fmri/code/json_files/abide_percent_cv/pretrain_fold{pretrain_fold_idx}/{percentage}/fold_{fold_idx}.json', 'w') as f:
        json.dump(dataset_splits, f, indent=4)


def data_to_dict(data):
    return [{"id": row['FILE_ID']+'_rois_aal', "group": 0 if row['DX_GROUP'] == 2 else 1} for index, row in data.iterrows()]


# Single testing set

# Generate splits for validation and testing once, as they remain constant
# training_df, testing_df = train_test_split(
#     df, test_size=testing_count, stratify=df['DX_GROUP'])

# create pretrain dataset
# create_pretrain(training_df, testing_df)


# # Generate five-fold splits for each training percentage
# skf = StratifiedKFold(n_splits=5)

# for percentage in training_percentages:
#     for fold_idx, (train_index, val_index) in enumerate(skf.split(training_df, training_df['DX_GROUP']), start=1):
#         fold_train_df = training_df.iloc[train_index]
#         fold_val_df = training_df.iloc[val_index]
#         generate_and_save_splits(
#             fold_train_df, fold_idx, percentage, fold_val_df, testing_df)

# 5 fold pretraining and testing
skf_pretrain = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
for pretrain_fold_idx, (train_idx, test_idx) in enumerate(skf_pretrain.split(df, df['DX_GROUP']), start=1):
    training_df, testing_df = df.iloc[train_idx], df.iloc[test_idx]
    create_pretrain(training_df, testing_df, pretrain_fold_idx)

    # 5 fold fine-tuning process
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for percentage in training_percentages:
        for fold_idx, (train_index, val_index) in enumerate(skf.split(training_df, training_df['DX_GROUP']), start=1):
            fold_train_df = training_df.iloc[train_index]
            fold_val_df = training_df.iloc[val_index]
            generate_and_save_splits(
                fold_train_df, fold_idx, percentage, fold_val_df, testing_df, pretrain_fold_idx)
