import json
import os
import io
import sys 
import torch.utils.data.dataset

sys.path.append('/home/yz2337/project/multi_fmri/code/utils')
from utils.transform import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def collate_fn(batch):
    # Initialize containers for the batched data
    ids = []
    xs = []
    ys = []

    # Loop through each item in the batch
    for item in batch:
        # Assuming id is a scalar or string, no need to convert to tensor
        ids.append(item['id'])
        xs.append(item['x'])  # Assuming x is already a tensor
        ys.append(item['y'])  # Assuming y is already a tensor

    # Stack the numeric data into a single tensor for 'x' and 'y'
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)

    # 'ids' remains a list because they might not be tensor-like or have varying lengths
    # If ids are numeric and you want to convert them to a tensor, you can do so, but it's not common for ids

    return {'id': ids, 'x': xs, 'y': ys}


def get_loader(args):

    train_transforms = ToTensor()

    val_transforms = ToTensor()

    train_dataset = SRDataset(json_file=args.json_file, 
                              transforms=train_transforms,
                              mode='train', data_path=args.data_path)
    val_dataset = SRDataset(json_file=args.json_file, transforms=val_transforms,
                            mode='val', data_path=args.data_path)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)

    return train_loader, val_loader


def get_test_loader(args):

    test_transforms = ToTensor()

    test_dataset = SRDataset(json_file=args.json_file, transforms=test_transforms,
                             mode='test', data_path=args.data_path)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)

    return test_loader


class SRDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, json_file, transforms=None, mode='train', data_path=None):

        self.dataset = []
        self.transforms = transforms
        self.load_dataset(json_file, mode, data_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id, fmri, group = self.dataset[idx]
        fmri, group = self.transforms(fmri, group)
        return {'id': id, 'x': fmri, 'y': group}

    def load_dataset(self, json_file, mode, data_path):

        # load the files as list
        with open(json_file, 'r') as f:
            file_lists = json.load(f)
        if mode not in file_lists:
            raise ValueError("Mode must be 'train' or 'val' or 'test' ")

        # crop fmri into short sequence
        crop = RandomCrop(size=48)

        # add each subject into dataset
        for item in tqdm(file_lists[mode]):
            id = item.split('.npy')[0]
            fmri_path = os.path.join(data_path, item)
            fmri = np.load(fmri_path)
            
            group = 1 # dummy variable to pass in crop function


            for i in range(10):
                cropped_fmri = crop(fmri, group)
                # first 36 time period to predict next 12 time period
                self.dataset.append((id, cropped_fmri[:36, :], cropped_fmri[36:, :]))