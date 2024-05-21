"""python train_SelfDistil_ViewRay_3.py -b 2 --device cuda:1 -e 1200 --img_size 96 96 96 --show_verbose -sd experiments.interpolated_volumes -exp nnUNETwSD_onSortedInterpolatedData_Jul13"""

# Python inherent
import os, glob, datetime
from typing import Union

# Torch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True

# Monai
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference

# Robert
from torchmanager_core.view import logger
from torchmanager import callbacks, losses
from torchmanager_core.protocols import MonitorType

# Our Code
import stuff.data as data
from configs import TrainingConfig
from torchmanager_monai import Manager, metrics
from networks import SelfDistillUNETRWithDictOutput as SelfDistilUNETR
from networks import SelfDistillnnUNetWithDictOutput as SelfDistilnnUNet
from loss_functions import Self_Distillation_Loss_Dice, PixelWiseKLDiv, Self_Distillation_Loss_KL, Self_Distillation_Loss_L2

import inspect
from copy import deepcopy

# For consistency - Makes it really slow
# from torchmanager_core import random
# # random.freeze_seed(100) # setting the seed for reproducability
# # cudnn.benchmark = False
# # cudnn.deterministic = True


### Current best model: "nnUNETwSD_HigherLR_wLRS6_DecSD/best_dice.model"Testing [best] model at epoch: 196
#{'DiceCE_loss': 0.5039976239204407, 'KL_loss': 0.0, 'dice': 0.6005240082740784, 'loss': 0.5039976239204407, 'dice_0': 0.7893756031990051, 'dice_1': 0.7974714636802673, 'dice_2': 0.7971574068069458, 'dice_3': 0.8866153359413147, 'dice_4': 0.7793675661087036, 'dice_5': 0.698963463306427, 'dice_6': 0.5217999815940857, 'dice_7': 0.7498135566711426, 'dice_8': 0.5207235813140869, 'dice_9': 0.2478770613670349, 'dice_10': 0.2950035631656647, 'dice_11': 0.12211958318948746}

save_ckpt = None #'experiments.interpolated_volumes/cardiac_SegResNetVAE_base.exp/last.model'

notes = f"""
Optimizer: AdamW 1e-3, 3e-5, amsgrad=True
Learning Rate Scheduler: Poly w/ gamma = 0.9
img_size = (96 96 96)
crops = 2

nnUNETwSD - 6 layers, 5 distill
filters = [32, 64, 128, 256, 512, 512]

Weights -> The same:
DiceCE weights = [1, 0.4, 0.4, 0.6, 0.8, 1.0]
KL_div weights = [0.4, 0.6, 0.8, 1.0]

Current time: {datetime.datetime.now()}
"""

################################################################
#--------------------------------------------------------------#
# Set up
#--------------------------------------------------------------#
################################################################

# Getting the input arguments
config = TrainingConfig.from_arguments()

# config.device = [torch.device('cuda:0'), torch.device('cuda:1')]
# config.use_multi_gpus = True
# config.device = torch.device('cuda:2')
if config.show_verbose: config.show_settings()

# initialize checkpoint and data dirs
data_dir = os.path.join(config.experiment_dir, "data")
best_dice_ckpt_dir = os.path.join(config.experiment_dir, "best_dice.model")
best_hd_ckpt_dir = os.path.join(config.experiment_dir, "best_hd.model")
last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

################################################################
# Setting up the data
################################################################
# src = '/mnt/data/Summerfield/Data/ViewRay.data/ViewRay.Resampled_1mm'
# data_json = '/mnt/data/Summerfield/Data/ViewRay.data/ViewRay.Resampled_1mm/cohort.json'
# roi_size = (200, 200, 200)

src = '/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data'
data_json="/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data/cohort.json"
roi_size = (128, 128, 128)

#------------------------------#
# Getting the data from files
#------------------------------#

num_classes = 13 # 12 labels + BKG 
in_channels = 1 # Only one modality input

num_workers = 1
dataset_configuration = {
    'src': src,
    'data_json': data_json,
    'fold': 1,
    'roi_size': roi_size,
    'img_size': config.img_size,
    'cached': True,
    'cache_num': (25, 25),
    'num_samples': 2,
    'num_workers': num_workers,
    }
logger.info(f'\nDataset settings:')
[logger.info(f'\t{key}: {item}') for key, item in dataset_configuration.items()]

train_ds, val_ds = data.load_ViewRay(
        testing_or_training='training',
        sim_only=False,  
        logger=logger,
        **dataset_configuration
)

training_dataset = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True)
validation_dataset = DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True)

#------------------------------#
# Quality Assurance
#------------------------------#

# Sanity Check:
logger.info(f'\nData QA:')
datasets = [training_dataset, validation_dataset] 
dataset_type = ['Training', 'Validation']
for ds, ds_type in zip(datasets, dataset_type):
    logger.info(f'  For {ds_type} data:')
    for _set in ds:
        img, lab = _set['image'], _set['label']
        logger.info(f'\tImage - shape: {img.shape}, type: {type(img)}, dtype: {img.dtype}')
        logger.info(f'\tLabel  shape: {lab.shape}, type: {type(lab)}, dtype: {lab.dtype}')
        logger.info(f'\tUnique Labels: {torch.unique(lab)}')
        break

################################################################
# Settup up a model
################################################################

#------------------------------#
# nnUNET model
#------------------------------#

strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] # input + 4 Enc-Dec Layers + Bottleneck 
kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]] # input + 4 Enc-Dec Layers + Bottleneck 
#                                 -> bottle neck
filters = [32, 64, 128, 256, 512, 512]

# filters = [64, 128, 256, 512, 1024]
model = SelfDistilnnUNet(
    spatial_dims = 3,
    in_channels = in_channels,
    out_channels = num_classes,

    kernel_size = kernel_size,
    strides = strides,
    upsample_kernel_size = strides[1:],
    filters=filters,
    norm_name="instance",

    deep_supervision=False,
    deep_supr_num=3,
    self_distillation=True, # set to false to avoid SD
    self_distillation_num=5, # Change back to 4 if everything breaks
    res_block=True,
    )
logger.info('*'*64)
logger.info('Running nnUNET')
logger.info('*'*64)

################################################################
# Configuring training settings
################################################################

#------------------------------#
# Optimizer
#------------------------------#

# optimizer = torch.optim.SGD(
#     model.parameters(), 
#     lr=1e-2, 
#     weight_decay=3e-5,
#     momentum=0.99,
#     nesterov=True,
#     dampening=0
#     )

# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=3e-5)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-5, amsgrad=True) # What I know works best

logger.info(f'\nUsing optimizer:')
logger.info(optimizer)

# Weight decay
lr_note = "Poly LR, gamma 0.9"
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: ( 1 - epoch / config.epochs) ** 0.9)

if not lr_scheduler: logger.info(f'No Learning Rate Scheduler')
else: 
    logger.info(f'Learning rate scheduler: {lr_scheduler}, {lr_note}')

#------------------------------#
# Loss functions - Self Distillation
#------------------------------#

loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

# For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main) [input]
# Dice Loss Only between GT Labels(target="out") [target] and softmax(out_dec1,out_dec2,out_dec3,out_dec4) [input]
# NOTE: weights (going down the list) are [1, 0.4, 0.4, 0.6, 0.8]
# ce_w = [1.0, 0.4, 0.4, 0.6, 0.8]
sd_w = [1.0, 0.4, 0.4, 0.6, 0.8, 1]
# sd_w = [1.0, 0.1, 0.25, 0.5, 0.75]
# sd_w = [1.0 for _ in range(5)]

loss_dice = Self_Distillation_Loss_Dice([
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[0]), #out_main and GT labels  # Allways 1
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[1]), #out_dec4 and GT labels  # Deepest
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[2]), #out_dec3 and GT labels  
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[3]), #out_dec2 and GT labels
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[4]), #out_dec1 and GT labels 
    losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0), weight=sd_w[5]), #out_dec1 and GT labels # Shallower
    ], target="out")

# Self Distillation from deepest encoder/decoder (out_enc4/out_dec1): Teacher (T), to shallower encoders/decoders (out_enc2/out_dec2,out_enc3/out_dec3,out_dec4/out_enc1): Students (S)  
# For KL Div between softmax(out_dec1/out_enc4) [target] and log_softmax((out_dec2/out_enc3,out_dec3/out_enc2,out_dec4/out_enc1)) [input]
# lambda_feat: float = 0.0001  # weight of L2 Loss Term between feature maps
temperature: int = 3 # divided by temperature (T) to smooth logits before softmax (required for KL Div)
# 3 worked the best but try 4 / 5
# NOTE: weights (going down the list) are [0.6, 0.8, 1]
# KL_w = [0.6, 0.8, 1.0]
KL_w = [0.4, 0.6, 0.8, 1]
# KL_w = [0.25, 0.5, 0.75, 1]
# KL_w = [1.0 for _ in range(5)]
# Between deepest and shallowest -> low
# KL_w = [1, 1, 1]

# 4 losses because there are 4 layers inbetween the input & bottleneck
loss_KL = Self_Distillation_Loss_KL([ 
    losses.Loss(PixelWiseKLDiv(log_target=False), weight=KL_w[0]), #out_dec4/out_enc0 (S) & out_dec1/out_enc4 (T) # Deepest / Shallowest [S] & Shallowest / Deepest [T]
    losses.Loss(PixelWiseKLDiv(log_target=False), weight=KL_w[1]), #out_dec3/out_enc1 (S) & out_dec1/out_enc4 (T)
    losses.Loss(PixelWiseKLDiv(log_target=False), weight=KL_w[2]), #out_dec2/out_enc2 (S) & out_dec1/out_enc4 (T)
    losses.Loss(PixelWiseKLDiv(log_target=False), weight=KL_w[3]), #out_dec1/out_enc3 (S) & out_dec1/out_enc4 (T) # always 1
], include_background=True, T=temperature) # pass the entire dict NOT just "out"

loss_fn = {
    "DiceCE": loss_dice,
    "KL": loss_KL
}
logger.info(f'\nRunning WITH self distillation')
logger.info(f'Dice + CE weights Weights: {sd_w}')
logger.info(f'KL divergence Weights: {KL_w}')


#------------------------------#
# Loss functions - No Distillation
#------------------------------#

# loss_dice = losses.Loss(DiceCELoss(include_background=True, to_onehot_y=True, softmax=True), target="out") # out_main and GT labels
# loss_fn = {
#     "dice": loss_dice
#     }
# logger.info(f'\nRunning WITHOUT self distillation\n')

#------------------------------#
# Metric Functions
#------------------------------#

dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")
hd_fn = metrics.CumulativeIterationMetric(HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="none", get_not_nans=False), target="out")
metric_fns: dict[str, metrics.Metric] = {
    "val_dice": dice_fn,
    # "val_hd": hd_fn
    } 

#------------------------------#
# Post processing for validation / testing
#------------------------------#

post_labels = data.transforms.AsDiscrete(to_onehot=num_classes)
post_predicts = data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes)

################################################################
# Compiling the manager & putting things together
################################################################

# compile manager
if not save_ckpt:
    manager = Manager(model, post_labels=post_labels, post_predicts=post_predicts, optimizer=optimizer, loss_fn=loss_fn, metrics=metric_fns, roi_size=config.img_size) # type: ignore
    manager.notes = notes
    
else:
    manager = Manager.from_checkpoint(save_ckpt)
    print('*'*64)
    print(f'Loading from {save_ckpt}')
    print(f'Epoch: {manager.current_epoch}')
    print(f'Notes: {manager.notes}')
    print('*'*64)
#------------------------------#
# QA
#------------------------------#
logger.info(f'\nFinal QA:')
logger.info(f'\tRunning Model: {type(manager.model)}')
logger.info(f'\tExperiment Notes: {manager.notes}')

#------------------------------#
# Putting together the checkpoints
#------------------------------#

# initialize callbacks
tensorboard_callback = callbacks.TensorBoard(data_dir)
# Constructing the list of them
callbacks_list: list[callbacks.Callback] = [
    tensorboard_callback,
    callbacks.LastCheckpoint(manager, last_ckpt_dir), # Tracking the last epoch
    callbacks.BestCheckpoint("dice", manager, best_dice_ckpt_dir), # Tracking the best epoch according to the DSC
    # callbacks.BestCheckpoint("hd", manager, best_hd_ckpt_dir, monitor_type=MonitorType.MIN), # Tracking the best epoch according to the HD
    callbacks.LrSchedueler(lr_scheduler, tf_board_writer=tensorboard_callback.writer), # Tracking the learning rate scheduler
]

logger.info(f'Using callbacks_list: {callbacks_list}')

################################################################
# train the model
################################################################

start_time = datetime.datetime.now()
logger.info(f'\nTraining Start time: {start_time}')

manager.fit(training_dataset, config.epochs, val_dataset=validation_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, show_verbose=config.show_verbose)

end_time = datetime.datetime.now()
logger.info(f'\nEnd time: {end_time}')
total_time = end_time - start_time
logger.info(f'Total time: {total_time}')

################################################################
# Load the test cases and validate
################################################################

#------------------------------#
# Getting the data sources
#------------------------------#

_, test_ds = data.load_ViewRay(
        testing_or_training='testing',  
        logger=logger,
        sim_only=True,
        **dataset_configuration
)
testing_dataset = DataLoader(test_ds, batch_size=1, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True)

#------------------------------#
# QA
#------------------------------#

logger.info(f'\nFor Testing data:')
for _set in testing_dataset:
    img, lab = _set['image'], _set['label']
    logger.info(f'\tImage - shape: {img.shape}, type: {type(img)}, dtype: {img.dtype}')
    logger.info(f'\tLabel - shape: {lab.shape}, type: {type(lab)}, dtype: {lab.dtype}')
    logger.info(f'\tUnique Labels: {torch.unique(lab)}')
    break

#------------------------------#
# Testing the LAST saved epoch model
#------------------------------#

summary = manager.test(testing_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
logger.info(f'\nTesting model at epoch: {manager.current_epoch}')
logger.info(summary)

#------------------------------#
# Testing the BEST saved epoch model
#------------------------------#

manager = Manager.from_checkpoint(best_dice_ckpt_dir)
summary = manager.test(testing_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
logger.info(f'\nTesting [best] model at epoch: {manager.current_epoch}')
logger.info(summary)

# logger.info(manager.notes)

#------------------------------#
# Testing the BEST HD save epoch model
#------------------------------#
try:
    manager = Manager.from_checkpoint(best_hd_ckpt_dir)
    summary = manager.test(testing_dataset, device=config.device, use_multi_gpus=config.use_multi_gpus, show_verbose=config.show_verbose)
    logger.info(f'\nTesting [best] model at epoch: {manager.current_epoch}')
    logger.info(summary)
except:
    ...

logger.info(manager.notes)