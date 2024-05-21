# pyright: reportPrivateImportUsage=false
import glob
import os
from monai.data import CacheDataset
import numpy as np
from typing import Union, Tuple
import random

from .transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
)

def load(data_dir: str, img_size: Tuple[int, ...]=(128,128,128), train_split: int = 24, show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, CacheDataset, int]:
    """
    Load dataset

    - Parameters:
        - data_dir: A `str` of data directory
        - img_size: An `int` of image size
        - show_verbose: A `bool` of flag to show loading progress
    - Returns: A `tuple` of training `DataLoader`, validation `DataLoader`, and the number of classes in `int`
    """
    # load images and labels
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "Images.Training", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "Labels.Training", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    # Assign Testing Set First (Last 24 patients)
    test_data_dicts = data_dicts[-train_split:] # Last 24 patients for testing
    # print(f'MSD-BraTS Test length: {len(test_data_dicts)}')

    train_data_dicts_temp = data_dicts[:-train_split] # First 460 patients for train/validation
    # print(f'Train/val MSD-BraTS length: {len(train_data_dicts_temp)}')

    # randomly shuffle the dataset
    seed = 0
    random.Random(seed).shuffle(train_data_dicts_temp)

    # Assign Train/Validation Sets (388 for training & 72 for validation)
    val_split = 72
    train_data_dicts, val_data_dicts = train_data_dicts_temp[:-val_split], train_data_dicts_temp[-val_split:]
    # print(f'MSD-BraTS Train length: {len(train_data_dicts)}')
    # print(f'MSD-BraTS Val length: {len(val_data_dicts)}')   
        
    ##############################################################################################################

    # Train transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    
    train_transforms += [
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size,
            pos=1,
            neg=0, # neg=0 & pos=1 to always pick a foreground voxel as center for random crop
            num_samples=2, # 4 used for nnUnet/ try 2 for UNETR 
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.50,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.50,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.50,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=1.0,
        ),
        ToTensord(keys=["image", "label"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore    
    
    ################################################################################################

    # Validation transforms
    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    
    val_transforms += [
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        ToTensord(keys=["image", "label"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore    

    ################################################################################################

    # Test transforms (same as Validation Transforms)
    test_transforms = val_transforms

    ################################################################################################

    # get datasets
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        num_workers=8,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )
    
    val_ds = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        num_workers=8,
        progress=show_verbose
    )

    test_ds = CacheDataset(
        data=test_data_dicts,
        transform=test_transforms,
        num_workers=8,
        progress=show_verbose
    )

    num_classes = 4

    return train_ds, val_ds, test_ds, num_classes  

