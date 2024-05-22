# Code to load up viewray data

# Generic imports
import glob, os, logging, json
from typing import Any, Union, Tuple, Optional, Sequence
from math import radians

# Installed
import numpy as np, monai.transforms as transforms
from monai.data import CacheDataset, Dataset
from monai.data.meta_tensor import MetaTensor
import torch

class round_label(transforms.MapTransform):
    def __init__(self, k:str):
        self.k = k
    
    def __call__(self, d):
        d = dict(d)
        if isinstance(d[self.k], (torch.Tensor, MetaTensor)): d[self.k] = torch.round(d[self.k])
        elif isinstance(d[self.k], np.ndarray): d[self.k] = np.round(d[self.k])
        else: raise TypeError(f"Wrong input type: {type(d[self.k])}")
        return d

class ViewRay_JSON:
    def __init__(self, json_src:str, mst_src:str='.', fold:int = 1, sim_only:bool = False):
        self.sim_only = sim_only

        # Load the json description
        self.raw = json.load(open(json_src))
        if not self.raw['data_id'] == 'HFHS_ViewRay_Data': raise ValueError(f'JSON data id must be "HFHS_ViewRay_Data". Got {self.raw["data_id"]}.')

        # Get the path templates
        self.mst_img_path = os.path.join(mst_src, self.raw['image_path']).split('{pid}.{volume}')
        self.mst_lab_path = os.path.join(mst_src, self.raw['label_path']).split('{pid}.{volume}')

        # Sort the data into it's training and validation sets
        self.training_patient, self.validation_patient = [], []
        for set in self.raw['train']:
            if set['fold'] == str(fold):
                self.validation_patient.append(set)
            else:
                self.training_patient.append(set)
        self.testing_patient = self.raw['test']
        
        # construct a list of dictionaries that contain the paths to the relevent things
        self.training_dict = self.construct_src_list(self.training_patient)
        self.validation_dict = self.construct_src_list(self.validation_patient)
        self.testing_dict = self.construct_src_list(self.testing_patient)
        
    # Constructs a lsit of dictionaries for paths to image / label pairs
    def construct_src_list(self, src_dict):
        out_list = []
        for set in src_dict: # For each set in the dictionary
            for vol in set['volumes']: # For each volume listed
                entry = {
                    'image': f'{set["pid"]}.{vol}'.join(self.mst_img_path),
                    'label': f'{set["pid"]}.{vol}'.join(self.mst_lab_path),
                } # create a path entry for it

                # Append it depending on if sim_only is flagged
                if self.sim_only and vol=='SIM': out_list.append(entry)
                elif not self.sim_only: out_list.append(entry)
        return out_list

    @property
    def for_training(self):
        return self.training_dict, self.validation_dict
    
    @property
    def for_testing(self):
        return [], self.testing_dict


def load(
        # Data locations
        src: str,
        data_json: str,

        # Data selection
        fold: int = 1,
        sim_only: bool = False,
        testing_or_training:str = 'testing',

        # Options
        logger: Optional[Any] = None,
        cache_num: Sequence[int] = (10, 10),
        cached:bool = True,
        num_workers: int = 1,
        return_testing_paths: bool = False,

        # Augmentation setttings
        img_size: Sequence[int] = (96, 96, 96),
        num_samples: int = 4,
        roi_size: Union[Sequence[int], int] = (128, 128, 128)
    ):
    # QA of the inputs
    if not (testing_or_training == 'testing' or testing_or_training == 'training'):
        raise ValueError(f'Either "testing" or "training" must be given as argument for "testing_or_training". Got {testing_or_training}')
    if logger:
        logger.info(f'\nLoading data...\n\tLoading {testing_or_training} sets.')

    # Get the information from the JSON file
    info = ViewRay_JSON(json_src=data_json, mst_src=src, fold=fold, sim_only=sim_only)

    # Get the right data dicts depending on if its for training or testing
    to_augment_dict, to_preproc_dict = info.for_training if testing_or_training == 'training' else info.for_testing

    # Define the transformations for training
    keys = ["image", "label"]
    preproc_transforms = [
        transforms.LoadImaged(keys=keys, image_only=True),
        transforms.Orientationd(keys=keys, axcodes="RAS"),
        transforms.CenterSpatialCropd(keys=keys, roi_size=roi_size),
        transforms.NormalizeIntensityd(keys=keys[0], nonzero=True, channel_wise=True),
    ]
    range_10 = [-radians(10), radians(10)]
    augment_transforms = [
        transforms.RandCropByPosNegLabeld(keys=keys, image_key=keys[0], label_key=keys[1], neg=0, spatial_size=img_size, num_samples=num_samples),
        transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3),
        transforms.RandRotated(keys=keys, prob=1, range_x=range_10, range_y=range_10, range_z=range_10, mode=['bilinear', 'nearest']),
        transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=1.0),
        transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=1.0),
        round_label(k=keys[1])
    ]
    augment_transforms = preproc_transforms + augment_transforms
    preproc_transforms.append(transforms.ToTensord(keys=keys))
    augment_transforms.append(transforms.ToTensord(keys=keys))

    if logger:
        logger.info('Augmented Transform Lists:')
        [logger.info(f'\t{tsf}') for tsf in augment_transforms]
        logger.info(f'Preproc Transform List:')
        [logger.info(f'\t{tsf}') for tsf in preproc_transforms]

    preproc_transforms = transforms.Compose(preproc_transforms)
    augment_transforms = transforms.Compose(augment_transforms)

    if cached:
        augemnt_dataset = CacheDataset(
            data=to_augment_dict,
            transform=augment_transforms,
            cache_num=cache_num[0],
            num_workers=num_workers,
            cache_rate=1.0,
        )
        preproc_dataset = CacheDataset(
            data=to_preproc_dict,
            transform=preproc_transforms,
            cache_num=cache_num[1],
            num_workers=num_workers,
            cache_rate=1.0,
        )
    else:
        augemnt_dataset = Dataset(
            data=to_augment_dict,
            transform=augment_transforms
        )
        preproc_dataset = Dataset(
            data=to_preproc_dict,
            transform=preproc_transforms
        )

    if return_testing_paths: return augemnt_dataset, preproc_dataset, info.testing_dict
    return augemnt_dataset, preproc_dataset



if __name__ == '__main__':
    load(
        src = '/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data',
        data_json="/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data/cohort.json", 
        fold = 1,
        sim_only=True,
        testing_or_training='training',   
    )

    print('Done')