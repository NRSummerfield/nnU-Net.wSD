# pyright: reportPrivateImportUsage=false
from typing import Any, Union, Tuple, Optional
from pathlib import Path

import glob, numpy as np, os, torch
import monai.transforms as transforms
from torch.utils.data import Subset
import logging

from math import radians


from monai.data import CacheDataset, Dataset
# from .data_manager import Dataset, CacheDataset
from .transforms import TransformOptions, load_transforms, SetModality, MapTransform, SpatialPadD, ScaleIntensityD

class skip(MapTransform):
    def __init__(self):
        ...
    def __call__(self, d):
        return d
    
class scale_to_one(MapTransform):
    def __init__(self, keys):
        self.k = keys if isinstance(keys, list) else [keys]

    def __call__(self, d):
        d = dict(d)
        for k in self.k:
            d[k] = (d[k] - d[k].min())  / (d[k].max() - d[k].min())
        return d

class print_keys(MapTransform):
    def __init__(self, ks:list[str]):
        self.ks = ks if isinstance(ks, list) else [ks]
    
    def __call__(self, d):
        d = dict(d)
        for k in self.ks:
            print(f'{k}: {d[k].shape}')
        return d

class to_nclass(MapTransform):
    def __init__(self, key:str, expected_n:int):
        self.key = key
        self.expected_n = expected_n

    def __call__(self, d):
        d = dict(d)
        print(np.unique(d[self.key]))
        if isinstance(d[self.key], np.ndarray):
            if list(np.unique(d[self.key])) == [i for i in range(self.expected_n)]:
                d[self.key] = np.where(d[self.key] >= self.expected_n, 0, d[self.key])
        print(np.unique(d[self.key]))
        return d


class add_channel(MapTransform):
    def __init__(self, ks:list[str]):
        self.ks = ks if isinstance(ks, list) else [ks]
    
    def __call__(self, d):
        d = dict(d)
        for k in self.ks:
            d[k] = d[k][None]
        return d

    
class round_label(MapTransform):
    def __init__(self, k:str):
        self.k = k
    
    def __call__(self, d):
        d = dict(d)
        if isinstance(d[self.k], torch.Tensor): d[self.k] = torch.round(d[self.k])
        if isinstance(d[self.k], np.ndarray): d[self.k] = np.round(d[self.k])
        else: raise TypeError(f"fuck you: {type(d[self.k])}")
        return d


################################################################
# loaders
################################################################

def load(
    train_image_dir: Path, 
    train_label_dir: Path, 

    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 

    validation_image_dir: Optional[Path] = None,
    validation_label_dir: Optional[Path] = None,
    
    roi_size: tuple[int] = [190, 190, 155],
    set_modality: Optional[Union[int, list[int]]] = None,
    num_samples: int = 1,

    add_channel_arg: bool = True,
    scale_intensity_arg: bool = False,
    zero_mean: bool = True,
    show_verbose: bool = False,
    ndim: int = 3,
    search_key: str = '*.nii.gz',
    chached: bool = False,
    cache_num: Union[int, tuple[int]] = 1,
    num_workers: int = os.cpu_count(),
    logger: Optional[logging.Logger] = None
    ) -> Tuple[CacheDataset, CacheDataset]:
    print(f'\nLoading classic augmentations\n')
    # index 0 is the image volume, index 1 is the ground truth
    keys = ['image', 'label']

    # Load the training images/labels in a dictionary
    if isinstance(train_image_dir, list): 
        train_images = train_image_dir
    else:
        train_images = sorted(glob.glob(os.path.join(train_image_dir, search_key)))

    if isinstance(train_label_dir, list): 
        train_labels = train_label_dir
    else:
        train_labels = sorted(glob.glob(os.path.join(train_label_dir, search_key)))

    img_size = img_size if isinstance(img_size, tuple) else (img_size for _ in range(ndim))
    train_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(train_images, train_labels)]

    # If a validation data directory is defined, create a dictionary for them
    if (validation_image_dir and validation_label_dir):
        if isinstance(validation_image_dir, list):
            val_images = validation_image_dir
        else:
            val_images = sorted(glob.glob(os.path.join(validation_image_dir, search_key)))
        
        if isinstance(validation_label_dir, list):
            val_labels = validation_label_dir
        else:
            val_labels = sorted(glob.glob(os.path.join(validation_label_dir, search_key)))

        val_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(val_images, val_labels)]

    # If no validation dir is defined, split the training one into two as defined by train_split
    else:
        if not isinstance(train_split, int): raise ValueError(f'If a validation image and label directory are not specified, use train_split to create a validation set from the trianing cohort wtih n=train_split cases')
        val_dict = train_dict[-train_split:]
        train_dict = train_dict[:-train_split]

    # print(train_dict, val_dict)

    [print(_dict) for _dict in train_dict]
    print()
    [print(_dict) for _dict in val_dict]

    # Creating the sequence of transformations
    print(f'Length of training data: {len(train_dict)}')
    print(f'Length of validaiton data: {len(val_dict)}')

    # For validation 
    validation_transform_list = [
        transforms.LoadImaged(keys=keys),
        add_channel(ks=keys) if add_channel_arg else skip(),
        scale_to_one(keys=keys[0]) if scale_intensity_arg else skip(),
        SpatialPadD(keys=keys, spatial_size=roi_size),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        ]

    # Optional Zero_mean normalization
    if zero_mean:
        validation_transform_list.append(transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True))

    # Center crop
    # validation_transform_list.append(transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size))
    
    # Optional set modality
    if set_modality is not None:
        validation_transform_list.append(SetModality(mode=set_modality, key='image'))

    _10 = radians(10)
    r10 = [-_10, _10]
    sp5 = [-2, 2]
    # Specific for training
    training_transform_list = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=num_samples),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
        # THIS ONE IS ADDED
        transforms.RandRotated(keys=keys, prob=1, range_x=r10, range_y=r10, range_z=r10, mode=['bilinear', 'nearest']),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0),
        # THIS ONE IS ADDED BUT DOESN"T CHANGE ANYTHING
        round_label(k=keys[1]), # ROUND THE LABEL TOMAKE SURE THEYRE INTS
        # print_keys(ks=keys)
        # transforms.RandGaussianSmoothd(keys='image', prob=1, sigma_x=sp5, sigma_y=sp5, sigma_z=sp5, ),
        # transforms.RandGaussianNoised(keys='image', prob=0.1, mean=0, std=0.1),
    ]
    training_transform_list = validation_transform_list + training_transform_list
    # Turning into tensors
    training_transform_list.append(transforms.ToTensord(keys=keys))
    validation_transform_list.append(transforms.ToTensord(keys=keys))
    
    train_transforms = transforms.Compose(training_transform_list)
    val_transforms = transforms.Compose(validation_transform_list)

    if logger:
        logger.info('Testing Transform Lists:')
        [logger.info(f'\t{tsf}') for tsf in training_transform_list]
        logger.info(f'Validation Transform List:')
        [logger.info(f'\t{tsf}') for tsf in validation_transform_list]


    cache_num = cache_num if isinstance(cache_num, tuple) else (cache_num, cache_num)

    if not chached:
        train_dataset = Dataset(
            data=train_dict,
            transform=train_transforms,
        )

        val_dataset = Dataset(
            data=val_dict,
            transform=val_transforms,
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

