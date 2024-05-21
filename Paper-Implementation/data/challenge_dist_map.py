# pyright: reportPrivateImportUsage=false
import glob
import os
from monai.data import CacheDataset
import numpy as np
from typing import Union, Tuple
import random

from .transforms import (
    AddChanneld,
    Compose,
    ConvertLabel,
    LoadImaged,
    AsChannelFirstd,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    CenterSpatialCropd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

def loadBoun(data_dir: str, img_size: Tuple[int, ...]=(96,96,96), scale_intensity_ranged: bool = True, train_split: int = 4, show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, CacheDataset, int]:
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
        glob.glob(os.path.join(data_dir, "imagesTraining", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTraining", "*.nii.gz")))
    train_dist_maps = sorted(
        glob.glob(os.path.join(data_dir, "distmapsTraining", "*.nii.gz")))
    
    data_dicts = [
        {"image": image_name, "label": label_name, "dist_map": dist_map_name}
        for image_name, label_name, dist_map_name in zip(train_images, train_labels, train_dist_maps)
    ]
    
    # Assign Testing Set First (4 patients)
    test_data_dicts = data_dicts[:train_split] # First 4 patients for testing
    
    # Remove the 'dist_map' key from the test_dataset as distance maps are not required during testing (ONLY required during training)
    # initializing key 
    del_key = 'dist_map'  
    # Remove Key from Dictionary List
    for items in test_data_dicts:
        if del_key in items:
            del items[del_key]
    
    train_data_dicts_temp = data_dicts[train_split:] # Last 16 patients for train/validation

    # randomly shuffle the dataset
    seed = 0
    random.Random(seed).shuffle(train_data_dicts_temp)

    # Assign Train/Validation Sets (12 for training & 4 for validation)
    train_data_dicts, val_data_dicts = train_data_dicts_temp[train_split:], train_data_dicts_temp[:train_split] 

    # Remove the 'dist_map' key from the validation_dataset as distance maps are not required during validation (ONLY required during training)
    # initializing key 
    del_key = 'dist_map'  
    # Remove Key from Dictionary List
    for items in val_data_dicts:
        if del_key in items:
            del items[del_key]
    
    # train_data_dicts, val_data_dicts = data_dicts[train_split:], data_dicts[:train_split]

    ##############################################################################################################

    # Train transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label", "dist_map"]),
        AddChanneld(keys=["image", "label"]),
        AsChannelFirstd(keys=("dist_map")), # same as data_dict['dist_map'].permute(3, 0, 1, 2)
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ), # already done while generating dist maps
        Orientationd(keys=["image", "label", "dist_map"], axcodes="RAS"),
        # CenterSpatialCropd(keys=["image", "label", "dist_map"], roi_size=(128,128,128)),
    ]
    if scale_intensity_ranged:
        train_transforms += [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # for CT Only
        ]
    train_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        SpatialPadd(keys=["image", "label", "dist_map"], spatial_size=img_size, mode='reflect'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label", "dist_map"],
            label_key="label",
            spatial_size=img_size,  # 16*n
            pos=1,
            neg=0, # neg=0 & pos=1 to always pick a foreground voxel as center for random crop
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="label", select_fn=lambda x: x > 0),

        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label", "dist_map"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label", "dist_map"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label", "dist_map"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore

    ################################################################################################
    
    # Validation transforms WITHOUT distance maps
    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ), # already done while generating dist maps
        Orientationd(keys=["image", "label"], axcodes="RAS")
    ]
    
    if scale_intensity_ranged:
        val_transforms += [
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ] # for CT Only
    val_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        ToTensord(keys=["image", "label"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore

    ##########################################################################

    # Test transforms (same as Validation Transforms)
    test_transforms = val_transforms

    ##########################################################################

    # get datasets
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        cache_num=2,
        cache_rate=1.0,
        num_workers=8,
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

    num_classes = 8

    return train_ds, val_ds, test_ds, num_classes 

