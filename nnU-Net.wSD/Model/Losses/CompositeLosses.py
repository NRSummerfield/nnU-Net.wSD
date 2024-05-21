from torchmanager import losses
from monai.losses.dice import DiceCELoss
from torch.nn.modules.loss import _Loss
import torch

from typing import Union, Optional, Callable, Any

from .loss_functions import Segmentation_Loss, PixelWiseKLDiv, Self_Distillation_Loss


known_segmentation_losses = dict(
    DiceCELoss=DiceCELoss
)

known_distillation_losses = dict(
    PixelWiseKLDiv=PixelWiseKLDiv
)

def unpack_argument(arg: Union[list, tuple], default_out: Optional[Any] = None):
    if isinstance(arg, (list, tuple)):
        assert len(arg) == 2, ValueError(f'Input argument must have only 2 elements (i.e. [item, config]) but got {len(arg)}')
        return arg
    else: return arg, default_out

class CompositeLoss(_Loss, losses.Loss):
    loss_fn: Union[losses.Loss, dict[str, losses.Loss]]

    def __init__(self, 
                 n_layers: int, 
                 segmentation_loss: Union[Callable[[torch.Tensor], torch.Tensor], str, tuple[Callable, dict], tuple[str, dict]] = DiceCELoss,
                 distillation_loss: Union[Callable[[torch.Tensor], torch.Tensor], str, tuple[Callable, dict], tuple[str, dict]] = PixelWiseKLDiv,
                 temperature:int=3, 
                 weights_segmentation_loss: Optional[list[float]] = None, 
                 weights_distillation_loss: Optional[list[float]] = None, 
                 return_just_loss: bool = True,
                 target: Optional[str] = None,
                 ):
        if weights_segmentation_loss is not None: assert len(weights_segmentation_loss) == n_layers, ValueError(f'len(weights_segmentation_loss) must match n_layers: got {len(weights_segmentation_loss)} and {n_layers} respectively')
        if weights_distillation_loss is not None: assert len(weights_distillation_loss) == n_layers - 2, ValueError(f'len(weights_distillation_loss) must match n_layers - 2: got {len(weights_distillation_loss)} and {n_layers} respectively')
        super().__init__()
        self.target = target
        self.return_just_loss = return_just_loss

        # getting the loss classes & any specified input arguments
        segmentation_loss, segmentation_loss_args = unpack_argument(segmentation_loss, default_out={})
        distillation_loss, distillation_loss_args = unpack_argument(distillation_loss, default_out={})

        # getting the weights for the loss terms
        wSegL = weights_segmentation_loss if weights_segmentation_loss is not None else [1 for _ in range(n_layers)]
        wLosL = weights_distillation_loss if weights_distillation_loss is not None else [1 for _ in range(n_layers-2)] 

        # Defining the loss functions
        loss_segmentation = Segmentation_Loss([
            losses.Loss(segmentation_loss(**segmentation_loss_args), weight=w) for w in wSegL
        ])
        loss_selfdistillation = Self_Distillation_Loss([ 
            losses.Loss(distillation_loss(**distillation_loss_args), weight=w) for w in wLosL
        ], include_background=True, T=temperature)
        self.loss_fn = {
            "distillation": loss_selfdistillation,
            "segmentation": loss_segmentation,
        }

    def __call__(self, input: torch.Tensor, target:torch.Tensor):
        if isinstance(target, dict): target = target[self.target]
        out = {}
        total = torch.tensor(0)
        for k in self.loss_fn.keys():
            k_loss = self.loss_fn[k](input=input, target=target)
            total = total +k_loss
            out[k] = k_loss.item()
        out['total_loss'] = total.item()
        if self.return_just_loss:
            return total
        else:
            return total, out
