# pyright: reportPrivateImportUsage=false
import glob
import os
from monai.data import CacheDataset
from typing import Tuple

from .transforms import (
    AddChanneld,
    Compose,
    ConvertLabel,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

def load(data_dir: str, img_size: Tuple[int, ...]=(96,96,96), scale_intensity_ranged: bool = True, train_split: int = 4, show_verbose: bool = False) -> Tuple[CacheDataset, CacheDataset, int]:
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
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
   
    ## Divide the 20 patients for 5-fold Cross-Validation

    # Patient 1-4 (First 4 patients) for validation - Fold 1
    # train_data_dicts, val_data_dicts = data_dicts[train_split:], data_dicts[:train_split] 

    # Patient 5-8 for validation - Fold 2
    # train_data_dicts, val_data_dicts = data_dicts[:train_split] + data_dicts[train_split+4:], data_dicts[train_split:train_split+4]

    # Patient 9-12 for validation - Fold 3
    # train_data_dicts, val_data_dicts = data_dicts[:train_split+4] + data_dicts[train_split+4*2:], data_dicts[train_split+4:train_split+4*2] 

    # Patient 13-16 for validation - Fold 4
    # train_data_dicts, val_data_dicts = data_dicts[:train_split+4*2] + data_dicts[train_split+4*3:], data_dicts[train_split+4*2:train_split+4*3] 

    # Patient 17-20 for validation - Fold 5
    train_data_dicts, val_data_dicts = data_dicts[:train_split+4*3], data_dicts[train_split+4*3:] 
    
    ##############################################################################################################

    # Train transforms
    train_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if scale_intensity_ranged:
        train_transforms += [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),  # for CT Only
        ]
    train_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        SpatialPadd(keys=["image", "label"], spatial_size=img_size, mode='reflect'),  # only pad when size < img_size
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=img_size, 
            pos=1,
            neg=0, # neg=0 & pos=1 to always pick a foreground voxel as center for random crop
            num_samples=2, 
            image_key="image",
            image_threshold=0,
        ),

        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
    train_transforms = Compose(train_transforms) # type: ignore    
    
    ################################################################################################

    # Validation transforms
    val_transforms: list[MapTransform] = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        ConvertLabel(keys='label'),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if scale_intensity_ranged: # for CT Only
        val_transforms += [
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        ]
    val_transforms += [
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),       
        ToTensord(keys=["image", "label"]),
    ]
    val_transforms = Compose(val_transforms) # type: ignore    

    ################################################################################################

    # get datasets
    train_ds = CacheDataset(
        data=train_data_dicts,
        transform=train_transforms,
        num_workers=4,
        cache_num=2,
        cache_rate=1.0,
        progress=show_verbose
    )
    
    val_ds = CacheDataset(
        data=val_data_dicts,
        transform=val_transforms,
        num_workers=4,
        progress=show_verbose
    )

    num_classes = 8

    return train_ds, val_ds, num_classes  

