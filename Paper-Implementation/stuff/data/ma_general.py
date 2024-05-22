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
from .transforms import TransformOptions, load_transforms, SetModality, MapTransform


class StackModalities(MapTransform):
    """
    Takes a monai style data dictionary of multiple images and stacks them in a specified order.
    """

    def __init__(self, keys: list[Any], out_key: Any, channel_stack: int=0, del_item: bool=False, data_shape: Union[torch.Size, Any]='shape') -> None:
        """
        keys: The list of keys in order to be stacked on top of each other. Use `None` as a placeholder for zeros
        out_key: The new key to represent the stacked images
        channel_stack: The index to stack the images on
        del_item: toggle to delete the orignial index from the dictionary to save space
        """
        self.keys = keys
        self.out_key = out_key
        self.channel_stack = channel_stack
        self.del_item = del_item
        if isinstance(data_shape, torch.Size): self.shape = data_shape
        self.shape = data_shape 
    
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # print(self.keys)
        d = dict(data)
        # Get the shape from the dictionary or input
        if None in self.keys:
            shape = self.shape if isinstance(self.shape, torch.Size) else d[self.shape]
        else: shape = None

        arrays = []
        for key in self.keys:
            if key is not None:
                _data = torch.tensor(d[key])
                if self.del_item: d.pop(key) # For memory saving
                arrays.append(_data)
            else:
                arrays.append(torch.zeros(shape)) 
        d[self.out_key] = torch.cat(arrays, dim=self.channel_stack)
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

def load_SingleVolume(
    train_image_dir: Path, 
    train_label_dir: Path, 
    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 
    validation_image_dir: Optional[Path] = None,
    validation_label_dir: Optional[Path] = None,
    
    roi_size: tuple[int] = [190, 190, 155],
    set_modality: Optional[Union[int, list[int]]] = None,

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
    # Creating the sequence of transformations
    print(f'Length of training data: {len(train_dict)}')
    print(f'Length of validaiton data: {len(val_dict)}')

    # For validation 
    validation_transform_list = [
        transforms.LoadImaged(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS")]

    # Optional Zero_mean normalization
    if zero_mean:
        validation_transform_list.append(transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True))

    # Center crop
    validation_transform_list.append(transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size))
    
    # Optional set modality
    if set_modality is not None:
        validation_transform_list.append(SetModality(mode=set_modality, key='image'))

    _10 = radians(10)
    r10 = [-_10, _10]
    sp5 = [-2, 2]
    # Specific for training
    training_transform_list = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=4),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
        # THIS ONE IS ADDED
        transforms.RandRotated(keys=keys, prob=1, range_x=r10, range_y=r10, range_z=r10, mode=['bilinear', 'nearest']),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0),
        # THIS ONE IS ADDED BUT DOESN"T CHANGE ANYTHING
        round_label(k=keys[1]), # ROUND THE LABEL TOMAKE SURE THEYRE INTS
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



def load_SingleVolume_advanced(
    train_image_dir: Path, 
    train_label_dir: Path, 
    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 
    validation_image_dir: Optional[Path] = None,
    validation_label_dir: Optional[Path] = None,
    
    roi_size: tuple[int] = [190, 190, 155],
    set_modality: Optional[Union[int, list[int]]] = None,

    zero_mean: bool = True,
    show_verbose: bool = False,
    ndim: int = 3,
    search_key: str = '*.nii.gz',
    chached: bool = False,
    cache_num: Union[int, tuple[int]] = 1,
    num_workers: int = os.cpu_count(),
    logger: Optional[logging.Logger] = None
    ) -> Tuple[CacheDataset, CacheDataset]:

    print('\nLoading with advanced augmentation\n')
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
    # Creating the sequence of transformations
    
    # For validation 
    validation_transform_list = [
        transforms.LoadImaged(keys=keys),
        transforms.Orientationd(keys=keys, axcodes="RAS")]

    # Optional Zero_mean normalization
    if zero_mean:
        validation_transform_list.append(transforms.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True))

    # Center crop
    validation_transform_list.append(transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size))
    
    # Optional set modality
    if set_modality is not None:
        validation_transform_list.append(SetModality(mode=set_modality, key='image'))

    # Specific for training
    _10 = radians(10)
    r10 = [-_10, _10]
    sp5 = [1, 1.5]
    training_transform_list = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=4),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
        transforms.RandRotated(keys=keys, prob=1, range_x=r10, range_y=r10, range_z=r10, mode=['bilinear', 'nearest']),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0),
        transforms.RandGaussianSmoothd(keys='image', prob=1, sigma_x=sp5, sigma_y=sp5, sigma_z=sp5, ),
        transforms.RandGaussianNoised(keys='image', prob=0.1, mean=0, std=0.1),

    ]
    training_transform_list = validation_transform_list + training_transform_list
    # Turning into tensors
    training_transform_list.append(transforms.ToTensord(keys=keys))
    validation_transform_list.append(transforms.ToTensord(keys=keys))
    
    train_transforms = transforms.Compose(training_transform_list)
    val_transforms = transforms.Compose(validation_transform_list)

    if logger:
        logger.info('\Testing Transform Lists:')
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





def load_forked(
    forked_keys: list,
    forked_paths: dict[Any, Path],
    train_image_dir: Path, 
    train_label_dir: Path, 
    img_size: Union[int, tuple[int, ...]], 
    train_split: Optional[int] = None, 
    validation_image_dir: Optional[Path] = None,
    validation_label_dir: Optional[Path] = None,
    show_verbose: bool = False,
    ndim: int = 3,
    search_key: str = '*.nii.gz',
    cache_num: Union[int, tuple[int]] = 1,
    num_workers: int = os.cpu_count(),
    roi_size: tuple[int] = [190, 190, 155],
    chached: bool = False,
    logger: Optional[logging.Logger] = None
    ) -> Tuple[CacheDataset, CacheDataset]:
    if logger: logger.info('Loading images from seperate locations into one volume...')
    

    img_size = img_size if isinstance(img_size, tuple) else (img_size for _ in range(ndim))

    # Load the training images in a dictionary of lists and labels as a list
    train_images = []
    for k in forked_paths.keys():
        train_images.append(sorted(glob.glob(os.path.join(train_image_dir, forked_paths[k], search_key))))
    train_labels = sorted(glob.glob(os.path.join(train_label_dir, search_key)))

    # Constructing the training cohort
    # List of dictionaries, each with a list of keys/files of variable sizes
    train_dict = []
    for *files, label in zip(*train_images, train_labels):
        _patient_dict = {'label': label}
        for key, file in zip(forked_keys, files):
            _patient_dict[key] = file
        train_dict.append(_patient_dict)


    # If a validation data directory is defined, create a dictionary for them
    if (validation_image_dir and validation_label_dir):
        val_images = []
        for k in forked_paths.keys():
            val_images.append(sorted(glob.glob(os.path.join(validation_image_dir, forked_paths[k], search_key))))
        val_labels = sorted(glob.glob(os.path.join(validation_label_dir, search_key)))
        val_dict = []
        for *files, label in zip(*val_images, val_labels):
            _patient_dict = {'label': label}
            for key, file in zip(forked_keys, files):
                _patient_dict[key] = file
            val_dict.append(_patient_dict)

    # If no validation dir is defined, split the training one into two as defined by train_split
    else:
        if not isinstance(train_split, int): raise ValueError(f'If a validation image and label directory are not specified, use train_split (int) to create a validation set from the trianing cohort wtih n=train_split cases')
        val_dict = train_dict[-train_split:]
        train_dict = train_dict[:-train_split]
    
    # Creating the sequence of transformations based off of transformation option dictionary
    # td = transform_options if isinstance(transform_options, TransformOptions) else TransformOptions.from_dict(transform_options) # Renaming the options to reduce size
    
    load_keys = ['label']
    load_keys.extend(forked_keys)
    
    keys = ['image', 'label']

    # For validation
    validation_transform_list = [
        transforms.LoadImaged(keys=load_keys),
        transforms.NormalizeIntensityd(keys=forked_keys, nonzero=True, channel_wise=True),
        StackModalities(keys=forked_keys, out_key='image'),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size),
    ]

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
    training_transform_list.append(transforms.ToTensord(keys=keys, dtype=torch.float))
    validation_transform_list.append(transforms.ToTensord(keys=keys, dtype=torch.float))
    
    train_transforms = transforms.Compose(training_transform_list)
    val_transforms = transforms.Compose(validation_transform_list)

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
