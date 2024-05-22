from monai.networks import * # type: ignore
from .unetr import UNETR, UNETRWithDictOutput, SelfDistillUNETRWithDictOutput
from .nnunet import SelfDistillnnUNetWithDictOutput
from .swinunetr import SelfDistillSwinUNETRWithDictOutput