# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import nibabel as nb
import numpy as np
from networks.unetr import UNETR
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandScaleIntensityd,
    RandAdjustContrastd,
    ScaleIntensityRangePercentilesd,
    RandZoomd,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    load_decathlon_properties,
    partition_dataset,
    select_cross_validation_folds,
    SmartCacheDataset,
    Dataset,
    decollate_batch,
    DistributedSampler,
)
from monai.data import CacheDataset,SmartCacheDataset, DataLoader, Dataset, partition_dataset
from pdb import set_trace as bp

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRangePercentilesd(keys=["image"],lower=0, upper=100,b_min=0.0, b_max=1.0, clip=True),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.1,
                prob=0.3,
            ),
            RandAffined(
                keys=['image','label'],
                prob=0.1,
                shear_range=(0.05,0.05,0.05),
            ),   

            ToTensord(keys=["image", "label"]),         
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            ScaleIntensityRangePercentilesd(keys=["image"],lower=0, upper=100,b_min=0.0, b_max=1.0, clip=True),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dir ='/nfs/masi/zhouy26/22Summer/IHI'

    split_JSON = '/nfs/masi/zhouy26/22Summer/IHI/json/subscore/fold0.json'
    # split_JSON = './json/sides/left/fold0.json'
    # split_JSON = './json/sides/data_aug/right/fold0.json'
    jsonlist = split_JSON


    datalist = load_decathlon_datalist(jsonlist, False, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(jsonlist, False, "validation", base_dir=data_dir)
    
        
#     train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_num=24, num_workers=8)
#     train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
#     train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=8, sampler=train_sampler)
    
# 
#     val_ds = Dataset(data=val_files, transform=val_transforms)
#     val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, shuffle=False)
    
    # print(datalist)
    
    train_ds = Dataset(data=datalist, transform=train_transforms)

    # train_files = partition_dataset(data=datalist, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[dist.get_rank()]
    # train_ds = SmartCacheDataset(data=train_files, transform=train_transforms, cache_num=6, replace_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)

#     val_files = partition_dataset(data=val_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True)[dist.get_rank()]
    val_ds = Dataset(data=val_files, transform=val_transforms)
    # val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader