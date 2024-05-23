# pyright: reportPrivateImportUsage=false
from typing import Any, Union, Tuple, Optional
from pathlib import Path

import glob, numpy as np, os, torch
import monai.transforms as transforms
from monai.data import CacheDataset, Dataset
from torch.utils.data import Subset
import logging

from .transforms import SetModality#, StackModalities, GetSize, SetModality


def load_SingleVolume(
    train_image_dir: Path, 
    train_label_dir: Path, 
    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 
    validation_image_dir: Optional[Path] = None,
    validation_label_dir: Optional[Path] = None,
    
    roi_size: tuple[int] = [190, 190, 155],
    set_modality: Optional[Union[int, list[int]]] = None,

    show_verbose: bool = False,
    ndim: int = 3,
    search_key: str = '*.nii.gz',
    chached: bool = False,
    cache_num: Union[int, tuple[int]] = 1,
    num_workers: int = os.cpu_count(),
    logger: Optional[logging.Logger] = None
    ) -> Tuple[CacheDataset, CacheDataset]:
    if logger: logger.info(f'Loading images from single volume from source {train_image_dir}...')

    # index 0 is the image volume, index 1 is the ground truth
    keys = ['image', 'label']

    # Load the training images/labels in a dictionary
    train_images = sorted(glob.glob(os.path.join(train_image_dir, search_key)))
    train_labels = sorted(glob.glob(os.path.join(train_label_dir, search_key)))
    img_size = img_size if isinstance(img_size, tuple) else (img_size for _ in range(ndim))
    train_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(train_images, train_labels)]

    # If a validation data directory is defined, create a dictionary for them
    if (validation_image_dir and validation_label_dir):
        val_images = sorted(glob.glob(os.path.join(validation_image_dir, search_key)))
        val_labels = sorted(glob.glob(os.path.join(validation_label_dir, search_key)))
        val_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(val_images, val_labels)]

    # If no validation dir is defined, split the training one into two as defined by train_split
    else:
        if not isinstance(train_split, int): raise ValueError(f'If a validation image and label directory are not specified, use train_split to create a validation set from the trianing cohort wtih n=train_split cases')
        val_dict = train_dict[-train_split:]
        train_dict = train_dict[:-train_split]
    # Creating the sequence of transformations based off of transformation option dictionary
    # td = transform_options if isinstance(transform_options, TransformOptions) else TransformOptions.from_dict(transform_options) # Renaming the options to reduce size


    # MSD
    # roi_size = [190, 190, 155]
    
    # # UCSF
    # roi_size = [185, 212, 155] # Based off of the maximum dimensions of the labels
    # For validation
    validation_transform_list = [
        transforms.LoadImaged(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
        transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size),
    ]
    if set_modality is not None:
        validation_transform_list.append(SetModality(mode=set_modality, key='image'))

    # Specific for training
    training_transform_list = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=4),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0),
    ]
    # Adding the training transfomrations onto the loading
    training_transform_list = validation_transform_list + training_transform_list
    # Turning it into a tensor
    training_transform_list.append(transforms.ToTensord(keys=keys))#, dtype=torch.float))
    validation_transform_list.append(transforms.ToTensord(keys=keys))#, dtype=torch.float))
    
    train_transforms = transforms.Compose(training_transform_list)
    val_transforms = transforms.Compose(validation_transform_list)

    # print('Testing Transform Lists')
    # [print(f'\t{tsf}') for tsf in training_transform_list]
    # print(f'Validation Transform List')
    # [print(f'\t{tsf}') for tsf in validation_transform_list]

    cache_num = cache_num if isinstance(cache_num, tuple) else (cache_num, cache_num)

    if not chached:
        train_dataset = Dataset(
            data=train_dict,
            transform=train_transforms,
            # progress=True,
        )

        val_dataset = Dataset(
            data=val_dict,
            transform=val_transforms,
            # progress=True
        )

    else:
        train_dataset = CacheDataset(
            data=train_dict,
            transform=train_transforms,
            cache_num=cache_num[0],
            num_workers=num_workers,
            cache_rate=1.0,
            progress=show_verbose
        )
        val_dataset = CacheDataset(
            data=val_dict,
            transform=val_transforms,
            cache_num=cache_num[1],
            num_workers=num_workers,
            cache_rate=1.0,
        )
    return train_dataset, val_dataset
