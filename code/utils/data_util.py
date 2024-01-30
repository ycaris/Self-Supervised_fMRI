import cv2
import json
import os
import io
import torch.utils.data.dataset
from transform import *
from torch.utils.data import DataLoader


def get_loader(args):
    
    # torch data loader
    # Set up data augmentation
    train_transforms = Compose([
        RandomCrop(args.CROP_IMG_SIZE, args.CONST.SCALE),
        FlipRotate(),
        BGR2RGB(),
        RandomColorChannel(),
        # utils.data_transforms.ColorJitter(cfg.DATA.COLOR_JITTER),
        # utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        # utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        ToTensor()
    ])

    val_transforms = Compose([
        # utils.data_transforms.BorderCrop(cfg.CONST.SCALE),
        BGR2RGB(),
        # utils.data_transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        ToTensor()
    ])
    
    train_dataset = SRDataset(args.train_file, train_transforms)
    val_dataset = SRDataset(args.val_file, val_transforms)
    
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.bs,
            num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            num_workers=args.num_workers, pin_memory=True, shuffle=False)
    
    return train_loader, val_loader


# def data_augmentation(image):
#     augmented_images_arrays, augmented_images_list = [], []
#     to_transform = [image, np.rot90(image, axes=(1, 2))]

#     for t in to_transform:
#         t_ud = t[:, ::-1, ...]
#         t_lr = t[:, :, ::-1, ...]
#         t_udlr = t_ud[:, :, ::-1, ...]

#         flips = [t_ud, t_lr, t_udlr]
#         augmented_images_arrays.extend(flips)

#     augmented_images_arrays.extend(to_transform)

#     for img in augmented_images_arrays:
#         img_unbatch = list(img)
#         augmented_images_list.extend(img_unbatch)

#     return augmented_images_list


# def create_patches(image, patch_size, step):
#     image = view_as_windows(image, patch_size, step)
#     h, w = image.shape[:2]
#     image = np.reshape(image, (h * w, patch_size[0], patch_size[1], patch_size[2]))

#     return image

class SRDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, json_file, transforms=None, mode='training'):
    
        self.dataset = []
        self.transforms = transforms
        self.load_dataset(json_file, mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name, img_lr, img_hr = self.dataset[idx]
        img_lr, img_hr = self.transforms(img_lr, img_hr)
        return img_name, img_lr, img_hr

    def load_dataset(self, json_file, mode):
        
        # load the files as list
        with open(json_file, 'r') as f:
            filepath_list = json.load(f)
        if mode not in filepath_list:
            raise ValueError("Mode must be 'training' or 'validation'")
        
        for filepath in tqdm(filepath_list[mode]):
            name = filepath.slipt(".nii")[0]
            lr_p = filepath['image']
            hr_p = filepath['label']
            lr = nib.load(lr_p).get_fdata()
            hr = nib.load(hr_p).get_fdata()
            
            self.dataset.append((name, lr, hr))
    
        
            
        