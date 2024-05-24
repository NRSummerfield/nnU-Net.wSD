"""
Example command:
python train_nnUNetwSD.py -b 2 --device cuda:1 -e 1200 --img_size 96 96 96 --show_verbose -exp nnU-Net.wSD_run1
"""

# Python inherent
import os, datetime

# PyTorch
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cudnn
cudnn.benchmark = True

# Monai
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data.utils import pad_list_data_collate
from monai.losses.dice import DiceCELoss
from monai.data.dataloader import DataLoader
from monai.inferers.utils import sliding_window_inference # change default value for argument "overlap" to "0.5" from "0.25"

# Torchmanager
from torchmanager_core.view import logger
from torchmanager import callbacks

# Local Imports
from Model.Networks import nnunet_wsd
from Model.Losses import CompositeLoss, PixelWiseKLDiv

import DataLoader as data
from configs import TrainingConfig
from torchmanager_monai import Manager, metrics

# optional path to the model to continue training from
save_ckpt = None

# Adding notes to keep track of experiments
notes = f"""
Optimizer: AdamW 1e-3, 3e-5, amsgrad=True
Learning Rate Scheduler: Poly w/ gamma = 0.9
img_size = (96 96 96)
crops = 2

Current time: {datetime.datetime.now()}
"""

################################################################
#--------------------------------------------------------------#
# Set up
#--------------------------------------------------------------#
################################################################

# Getting the input arguments
config = TrainingConfig.from_arguments()

# --- Optional hard code to set the device(s) directly ---
# config.device = [torch.device('cuda:0'), torch.device('cuda:1')]
# config.use_multi_gpus = True
# config.device = torch.device('cuda:2')

# how input settings
if config.show_verbose: config.show_settings()

# initialize checkpoint and data dirs
data_dir = os.path.join(config.experiment_dir, "data")
best_dice_ckpt_dir = os.path.join(config.experiment_dir, "best_dice.model")
best_hd_ckpt_dir = os.path.join(config.experiment_dir, "best_hd.model")
last_ckpt_dir = os.path.join(config.experiment_dir, "last.model")

################################################################
# Setting up the data loader
################################################################

# --- Define a dataloader for both the training and validation datasets ---

# Example:
src = '/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data'
data_json="/mnt/data/Summerfield/Data/ViewRay.data/1.5mm_volumes.data/cohort.json"
roi_size = (128, 128, 128)

#------------------------------#
# Getting the data from files
#------------------------------#

num_classes = 13 # 12 substructures + BKG 
in_channels = 1 # Only one modality input
num_workers = 1 # number of threads for dataloader

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

# pulling data from a json file and defining the needed pre-processing / augmentation steps
train_ds, val_ds = data.load_ViewRay(
        testing_or_training='training',
        sim_only=False,  
        logger=logger,
        **dataset_configuration
)

# Iterable type data loaders that gets input and reference volumes for model training / validation
training_dataset = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True)
validation_dataset = DataLoader(val_ds, batch_size=1, collate_fn=pad_list_data_collate, num_workers=num_workers, pin_memory=True)

#------------------------------#
# Quality Assurance of Data
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

# === NOTE: ============================================================================================
# strides / kernel_size / filters are all generated from the nnU-Net's pipeline and then configered here 
# ======================================================================================================
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] # input + 4 Enc-Dec Layers + Bottleneck 
kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]] # input + 4 Enc-Dec Layers + Bottleneck
filters = [32, 64, 128, 256, 512, 512]

model = nnunet_wsd(
    spatial_dims = 3,
    in_channels = in_channels,
    out_channels = num_classes,

    kernel_size = kernel_size,
    strides = strides,
    upsample_kernel_size = strides[1:],
    filters=filters,
    norm_name="instance",
    res_block=True,
    )

logger.info('*'*64)
logger.info('Running nnU-Net.wSD')
logger.info('*'*64)

# The module "torchmanager" expects the model outputs in a dictionary. Wrapping the model to return a dictionary on output.
class model_holder_for_torchmanager(torch.nn.Module):
    """
    Basic holder for a torch.nn.Module model to make it work with TorchManager
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> dict:
        out = self.model.forward(x)
        return {'out': out}       

################################################################
# Configuring training settings
################################################################

#------------------------------#
# Optimizer
#------------------------------#

# --- Define the optimizer and learning rate scheduler [optional] ---

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-5, amsgrad=True)

logger.info(f'\nUsing optimizer:')
logger.info(optimizer)

# Weight decay
lr_decay_end = config.epochs
lr_note = f"Poly LR, gamma 0.9, decays to zero @ epoch {lr_decay_end}"
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: ( 1 - epoch / lr_decay_end) ** 0.9)

if not lr_scheduler: logger.info(f'No Learning Rate Scheduler')
else: 
    logger.info(f'Learning rate scheduler: {lr_scheduler}, {lr_note}')

#------------------------------#
# Loss functions - Self Distillation
#------------------------------#

# --- Define a the loss function(s) used during training ---

# weights_segmentation_loss=[1.0, 0.4, 0.4, 0.6, 0.8, 1]
# weights_distillation_loss=[0.4, 0.6, 0.8, 1]
weights_segmentation_loss=[1.0, 1.0, 0.8, 0.6, 0.4, 0.4]
weights_distillation_loss=[1.0, 0.8, 0.6, 0.4]
loss_fn = CompositeLoss(
    n_layers=6,
    segmentation_loss=(DiceCELoss, dict(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0)),
    distillation_loss=(PixelWiseKLDiv, dict(log_target=False)),
    temperature=3,
    weights_segmentation_loss=weights_segmentation_loss,
    weights_distillation_loss=weights_distillation_loss,
    return_just_loss=True,
    target='out'
)

loss_fn = {'SelfDistill_loss':loss_fn}
logger.info(f'\nRunning WITH self distillation')
logger.info(f'Segmentation (main + supervision) Weights: {weights_segmentation_loss}')
logger.info(f'Self Distillation Weights: {weights_distillation_loss}')

#------------------------------#
# Metric Functions
#------------------------------#

# --- Define metrics used to quantify the hold-out validation dataset ---

dice_fn = metrics.CumulativeIterationMetric(DiceMetric(include_background=False, reduction="none", get_not_nans=False), target="out")
metric_fns: dict[str, metrics.Metric] = {"val_dice": dice_fn} 

#------------------------------#
# Post processing for validation / testing
#------------------------------#

# --- Post-processing to take the prediction output / reference contour, making it channeled for use in above metrics ---

post_labels = data.transforms.AsDiscrete(to_onehot=num_classes)
post_predicts = data.transforms.AsDiscrete(argmax=True, to_onehot=num_classes)

################################################################
# Compiling the manager & putting things together
################################################################

# --- Defining the torchmanager Manager to handle the training / validation looping ---

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

# --- Defining the tracking tools for model history ---

# initialize callbacks
tensorboard_callback = callbacks.TensorBoard(data_dir)
# Constructing the list of them
callbacks_list: list[callbacks.Callback] = [
    tensorboard_callback,
    callbacks.LastCheckpoint(manager, last_ckpt_dir), # Tracking the last epoch
    callbacks.BestCheckpoint("dice", manager, best_dice_ckpt_dir), # Tracking the best epoch according to the DSC
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
# Getting the data sources for test
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

# finish by repeating the notes of the experiment for easy access
logger.info(manager.notes)