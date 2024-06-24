#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''


import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random
import numbers


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms[:-1]
        self.ToTensor = transforms[-1]

    def __call__(self, fmri, group):
        for t in self.transforms:
            fmri, group = t(fmri, group)

        fmri, group = self.ToTensor(fmri, group)

        return fmri, group


# class ColorJitter(object):
#     def __init__(self, color_adjust_para):
#         """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
#         """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
#         """saturation [max(0, 1 - saturation), 1 + saturation] or the given [min, max]"""
#         """hue [-hue, hue] 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5"""
#         '''Ajust brightness, contrast, saturation, hue'''
#         '''Input: PIL Image, Output: PIL Image'''
#         self.brightness, self.contrast, self.saturation, self.hue = color_adjust_para

#     def __call__(self, img_lr, img_hr):
#         if random.random() > 0.8:
#             return img_lr, img_hr
#         img_lr, img_hr = [Image.fromarray(np.uint8(img))
#                           for img in [img_lr, img_hr]]
#         if self.brightness > 0:
#             brightness_factor = np.random.uniform(
#                 max(0, 1 - self.brightness), 1 + self.brightness)
#             img_lr, img_hr = [F.adjust_brightness(
#                 img, brightness_factor) for img in [img_lr, img_hr]]

#         if self.contrast > 0:
#             contrast_factor = np.random.uniform(
#                 max(0, 1 - self.contrast), 1 + self.contrast)
#             img_lr, img_hr = [F.adjust_contrast(
#                 img, contrast_factor) for img in [img_lr, img_hr]]

#         if self.saturation > 0:
#             saturation_factor = np.random.uniform(
#                 max(0, 1 - self.saturation), 1 + self.saturation)
#             img_lr, img_hr = [F.adjust_saturation(
#                 img, saturation_factor) for img in [img_lr, img_hr]]

#         if self.hue > 0:
#             hue_factor = np.random.uniform(-self.hue, self.hue)
#             img_lr, img_hr = [F.adjust_hue(img, hue_factor)
#                               for img in [img_lr, img_hr]]

#         img_lr, img_hr = [np.asarray(img) for img in [img_lr, img_hr]]
#         img_lr, img_hr = [img.clip(0, 255) for img in [img_lr, img_hr]]

#         return img_lr, img_hr


# class RandomColorChannel(object):
#     def __call__(self, img_lr, img_hr):
#         if random.random() > 0.8:
#             return img_lr, img_hr
#         random_order = np.random.permutation(3)
#         img_lr, img_hr = [img[:, :, random_order] for img in [img_lr, img_hr]]

#         return img_lr, img_hr


# class BGR2RGB(object):
#     def __call__(self, img_lr, img_hr):
#         random_order = [2, 1, 0]
#         img_lr, img_hr = [img[:, :, random_order] for img in [img_lr, img_hr]]

#         return img_lr, img_hr


# class RandomGaussianNoise(object):
#     def __init__(self, gaussian_para):
#         self.sigma = gaussian_para

#     def __call__(self, img_lr, img_hr):
#         if random.random() > 0.8:
#             return img_lr, img_hr
#         noise_std = np.random.randint(1, self.sigma)

#         gaussian_noise = np.random.randn(*img_lr.shape)*noise_std
#         # only apply to lr images
#         img_lr = (img_lr + gaussian_noise).clip(0, 255)

#         return img_lr, img_hr


# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, img_lr, img_hr):
#         assert (all([isinstance(img, np.ndarray) for img in [img_lr, img_hr]]))
#         img_lr, img_hr = [img/self.std - self.mean for img in [img_lr, img_hr]]

#         return img_lr, img_hr


class FlipAxis(object):

    def __call__(self, fmri, group):
        fmri = np.transpose(1, 0)
        return fmri


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, fmri, group):

        time = np.shape(fmri)[0]
        t_start = random.randint(0, time - self.size)
        t_end = t_start + self.size
        fmri = fmri[t_start:t_end, :]
        return fmri


class FutureMask(object):
    def __init__(self, mask_size):
        self.mask_size = mask_size

    def __call__(self, fmri):

        masked_fmri = fmri.copy()
        start_index = np.shape(fmri)[0] - self.mask_size
        masked_fmri[start_index:, :] = 0
        return masked_fmri


class RandomMask(object):

    def __call__(self, fmri):

        p = random.random()
        mask_ratio = 0.25 if p < 0.5 else 0.5

        masked_fmri = fmri.copy()
        total_elements = fmri.shape[0] * fmri.shape[1]
        num_to_mask = int(np.floor(mask_ratio * total_elements))

        # generate random row and column indices
        rows, cols = fmri.shape
        mask_rows = np.random.choice(rows, num_to_mask, replace=True)
        mask_cols = np.random.choice(cols, num_to_mask, replace=True)

        # Apply the mask to the randomly selected indices
        for row, col in zip(mask_rows, mask_cols):
            masked_fmri[row, col] = 0

        return masked_fmri


class RandomMaskROI(object):
    def __call__(self, fmri):

        p = random.random()
        mask_ratio = 0.25 if p < 0.5 else 0.5

        masked_fmri = fmri.copy()
        num_to_mask = int(np.floor(mask_ratio * fmri.shape[1]))

        # generate random row and column indices
        rows, cols = fmri.shape
        mask_cols = np.random.choice(cols, num_to_mask, replace=True)

        # Apply the mask to the randomly selected indices
        masked_fmri[:, mask_cols] = 0

        return masked_fmri


class RandomMaskTime(object):
    def __call__(self, fmri):

        p = random.random()
        mask_ratio = 0.25 if p < 0.5 else 0.5

        masked_fmri = fmri.copy()
        num_to_mask = int(np.floor(mask_ratio * fmri.shape[0]))

        # generate random row and column indices
        rows, cols = fmri.shape
        mask_rows = np.random.choice(rows, num_to_mask, replace=True)

        # Apply the mask to the randomly selected indices
        masked_fmri[mask_rows, :] = 0

        return masked_fmri


class ToTensor(object):
    """Concert time series fmri points from numpy to torch tensor"""

    def __call__(self, fmri, group):
        assert (isinstance(fmri, np.ndarray))
        fmri = torch.tensor(fmri, dtype=torch.float32)
        group = torch.tensor(group, dtype=torch.float32)
        return fmri, group
