import json
import os
import io
import sys
import torch.utils.data.dataset

from utils.transform import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random


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


def get_train_loader(args):

    # torch data loader
    # Set up data augmentation
    # train_transforms = Compose([
    #     # RandomCrop(args.CROP_IMG_SIZE, args.CONST.SCALE),
    #     # FlipRotate(),
    #     # BGR2RGB(),
    #     # RandomColorChannel(),
    #     # utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
    #     # utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    #     # utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
    #     ToTensor()
    # ])
    train_transforms = ToTensor()

    # val_transforms = Compose([
    #     # utils.data_transforms.BorderCrop(cfg.CONST.SCALE),
    #     BGR2RGB(),
    #     # utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    #     ToTensor()
    # ])

    train_dataset = SRDataset(json_file=args.json_file, transforms=train_transforms,
                              mode='train', data_path=args.data_path,
                              time_period=args.time_period)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.bs,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)

    return train_loader


def get_val_loader(args):

    val_transforms = ToTensor()
    val_dataset = SRDataset(json_file=args.json_file, transforms=val_transforms,
                            mode='val', data_path=args.data_path, time_period=args.time_period)

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)

    return val_loader


def get_test_loader(args):

    test_transforms = ToTensor()

    test_dataset = SRDataset(json_file=args.json_file, transforms=test_transforms,
                             mode='test', data_path=args.data_path, time_period=args.time_period)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=True)

    return test_loader


class SRDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, json_file, transforms=None, mode='train', data_path=None, time_period=48):

        self.dataset = []
        self.transforms = transforms
        self.load_dataset(json_file, mode, data_path, time_period)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id, fmri, group = self.dataset[idx]
        fmri, group = self.transforms(fmri, group)
        return {'id': id, 'x': fmri, 'y': group}

    def load_dataset(self, json_file, mode, data_path, time_period):

        # load the files as list
        with open(json_file, 'r') as f:
            file_lists = json.load(f)
        if mode not in file_lists:
            raise ValueError("Mode must be 'train' or 'val' or 'test' ")

        # crop fmri into short sequence
        # crop = RandomCrop(size=time_period+1)
        crop = RandomCrop(size=time_period)
        # futureMask = FutureMask(mask_size=16)
        randomMask = RandomMaskROI()

        for item in tqdm(file_lists[mode]):

            # id = item.split('.npy')[0]
            # fmri_path = os.path.join(data_path, item)
            id = item['id']
            fmri_path = os.path.join(data_path, f'{id}.npy')

            fmri = np.load(fmri_path)

            group = 1  # dummy variable to pass in crop function

            # 0-64 predict 1-65
            # if mode == 'train':
            #     x_shape = fmri.shape[0]
            #     # for i in range(int(x_shape/2)):
            #     for i in range(10):
            #         cropped_fmri = crop(fmri, group)
            #         self.dataset.append(
            #             (id, cropped_fmri[:time_period, :], cropped_fmri[1:time_period+1, :]))
            # else:
            #     for i in range(20):
            #         cropped_fmri = crop(fmri, group)
            #         self.dataset.append(
            #             (id, cropped_fmri[:time_period, :], cropped_fmri[1:time_period+1, :]))

            if mode == 'train':
                for i in range(10):
                    cropped_fmri = crop(fmri, group)
                    masked_fmri = randomMask(cropped_fmri)
                    # masked_fmri = futureMask(cropped_fmri)
                    self.dataset.append((id, masked_fmri, cropped_fmri))
            else:
                for i in range(20):
                    cropped_fmri = crop(fmri, group)
                    masked_fmri = randomMask(cropped_fmri)
                    # masked_fmri = futureMask(cropped_fmri)
                    self.dataset.append((id, masked_fmri, cropped_fmri))
