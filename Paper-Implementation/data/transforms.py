# pyright: reportPrivateImportUsage=false
from monai.transforms import AsDiscrete, AddChanneld, AsChannelFirstd, Compose, LoadImaged, LoadImage, SaveImage, SaveImaged, MapTransform, NormalizeIntensityd, Orientationd, CenterSpatialCropd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, RandShiftIntensityd, Spacingd, ScaleIntensityRanged, SpatialPadd, ToTensord
from monai.transforms import  CropForegroundd, MapTransform

import numpy as np
import torch

# convert the label pixel value 
class ConvertLabel(MapTransform):
    
    #[0. 205. 420. 500. 550. 600. 820. 850.]

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key][d[key]==205] = 1
            d[key][d[key]==420] = 2
            d[key][d[key]==500] = 3
            d[key][d[key]==550] = 4
            d[key][d[key]==600] = 5
            d[key][d[key]==820] = 6
            d[key][d[key]==850] = 7
            d[key][d[key]==421] = 0 ## have a pixel with value 421, which should be a mistake
        
        return d

class Skip:
    def __init__(self) -> None:
        """
        Simple class that just acts as a placeholder for toggle options
        """
        self.PlaceHolder = True

    def __call__(self, data:dict) -> dict:
        return data

class SetModality(MapTransform):
    def __init__(self, mode:int, key:str) -> None:
        """
        Takes a monai style data dictionary and highlights a single channel. Used for multichannel images where only one is wanted
        mode : the index of the channel to be looked at
        key : the dictionary key that defines the array of interest
        """
        self.mode = mode
        self.key = key
    
    def __call__(self, data:dict) -> dict:
        d = dict(data)
        d[self.key] = d[self.key][self.mode, ...][None, ...]
        return d

class NormToOne(MapTransform):
    def __init__(self, key:str) -> None:
        """
        Takes a monai style data dictionary and normalizes values to 1 when they're scaled by a consistent factor (i.e. converting a 0-255 tiff image into a 0-1 array)
        key : the data that needs to be normalized
        """
        self.k = key

    def __call__(self, data:dict) -> dict:
        d = dict(data)
        _max = np.max(d[self.k])# if isinstance(d[self.k], np.array) else torch.max(d[self.k])
        d[self.k] = d[self.k] / _max
        return d

class CropStructure:
    def __init__(self, struct_id:int, pad:int, keys:list, source_key:str) -> None:
        """
        Takes a monai style data dictionary and crops all data down to a single structure with a variable padding size on either side
        struct_id : the numerical value that represents the structure
        pad : the amount to pad around the substructure by
        keys : the keys that represent the data in the dictionary
        source_key : the key the cropping is based off of
        """
        self.id = struct_id
        self.k = source_key
        self.cropper = CropForegroundd(keys=keys, source_key=source_key, k_divisible=pad)

    def __call__(self, data:dict) -> dict:
        d = dict(data)
        d[self.k] = np.where(d[self.k] == self.id, 1, 0)# if isinstance(d[self.k], np.array) else torch.where(d[self.k] == self.id, 1, 0)
        d = self.cropper(d)
        return d