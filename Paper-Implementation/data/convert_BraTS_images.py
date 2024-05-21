# pyright: reportPrivateImportUsage=false
## Converted MSD-BraTS Images and Labels to view on 3D Slicer

import monai

import numpy as np
import time

import os
import shutil
import tempfile
import glob

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    CenterSpatialCropd,
    LoadImage,
    LoadImaged,
    SaveImage,
    SaveImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

import torch
import torch.nn.functional as F
from typing import Tuple

## Define Data Paths

## For MSD-BraTS Dataset
data_dir = "/home/neil/Lab_work/Medical_Image_Segmentation/Data_MSD_BraTS"  # for MSD_BraTS
    
## For MSD-BraTS Dataset
train_imgs = sorted(glob.glob(os.path.join(data_dir, "Images.Training", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels.Training", "*.nii.gz")))

data_dicts = [
    {"image": img_name, "label": label_name}
    for img_name, label_name in zip(train_imgs, train_labels)
]

# Assign Testing Set First (Last 24 patients)
test_data_dicts = data_dicts[-24:] # Last 24 patients for testing
# print(len(test_data_dicts))

## Load Train Images, Ground Truth (GT) Labels, convert GT Labels to Distance Maps and save the distance maps

# The following is for Train Labels ONLY (includes Validation also) 
loader = LoadImaged(keys=("image", "label"))
orient = Orientationd(keys=["image", "label"], axcodes="RAS")

distmap_transforms = Compose([loader, orient])

start_time = time.time()

for i in range(len(test_data_dicts)):
    
    data_dict = distmap_transforms(test_data_dicts[i])  
                                
    ## Save Distance Maps from GT Labels

    # np.ndarray: (1,240,240,155) 
    label = data_dict["label"] # type:ignore

    # np.ndarray: (4,240,240,155) 
    img = data_dict["image"] # type:ignore

    # print("Original GT Labels shape: ", np.shape(label))   
    # print("Original Image shape: ", np.shape(img))
    # print("Unique Labels: ", np.unique(label))
    # print("Max value of Label: ", np.max(label))
    
    # convert (1,240,240,155) -> (240,240,155) 
    label = np.squeeze(label,axis=0) 
    # print("Converted GT Labels shape: ", np.shape(label))
    
    # convert (4,240,240,155) -> (240,240,155)
    img = img[0,:,:,:] # select the first modality (FLAIR)
    # img = img[2,:,:,:] # select the third modality (T1gd)
    
    # img = np.squeeze(img,axis=0)
    # print("Converted Image shape: ", np.shape(img))

    ##############################################################################################################

    ## Save Converted Ground-Truth Labels - "channel_dim=None" needed ONLY for viewing on 3D Slicer
    '''out_dir = os.path.join(data_dir, "converted_labels") 
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    saver = SaveImage(output_dir=out_dir, output_postfix='saved', output_ext='.nii.gz', 
                      output_dtype=np.float32, resample=False, scale=None, dtype=None, squeeze_end_dims=True, 
                      data_root_dir='', separate_folder=False, print_log=False, channel_dim=None
                     )
        
    saver(label, meta_data={'filename_or_obj':'MSD_conv_label_patient_'+str(i+461)})'''

    ##############################################################################################################

    ## Save Converted Ground-Truth Images - "channel_dim=None" needed ONLY for viewing on 3D Slicer
    out_dir = os.path.join(data_dir, "converted_images_FLAIR") 
    # out_dir = os.path.join(data_dir, "converted_images_T1gd") 
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    saver = SaveImage(output_dir=out_dir, output_postfix='saved', output_ext='.nii.gz', 
                      output_dtype=np.float32, resample=False, scale=None, dtype=None, squeeze_end_dims=True, 
                      data_root_dir='', separate_folder=False, print_log=False, channel_dim=None
                     )
        
    saver(img, meta_data={'filename_or_obj':'MSD_conv_img_patient_'+str(i+461)})


end_time = time.time()
print("Time taken: ", (end_time-start_time))



