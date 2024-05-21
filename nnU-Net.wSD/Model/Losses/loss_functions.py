# pyright: reportPrivateImportUsage=false
from turtle import forward
from typing import Optional, Iterable, Callable, List, Sequence, cast, Set, Any, Union, Dict
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.losses import TverskyLoss
from monai.utils import LossReduction

from torchmanager import losses
from torchmanager_core import _raise


# For DiceCE Loss between GT Labels(target="out") [target] and softmax(out_main,out_dec2,out_dec3,out_dec4) [input]
class Segmentation_Loss(losses.MultiLosses):
    """
    Forward pass: Order of Comparison with the output: main, deepest, ..., shallowest
    """

    def __init__(self, *args, learn_weights: bool = False, weight: float = 1.0, **kwargs) -> None:

        super().__init__(*args, **kwargs)  

        self.learn_weights = learn_weights
        if learn_weights:
            self.params_dec = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses)-1)])

    def forward(self, input: Union[dict, torch.Tensor], target: Any) -> torch.Tensor:
        if 'decoder' not in list(input.keys()): return torch.tensor(0)
        # initilaize
        loss = 0
        l = 0
        
        # Validation Mode
        if isinstance(input, torch.Tensor): 
            return self.losses[0](input, target)
        
        # Training Mode
        outs = [input['out'], input['decoder']['teacher']] + input['decoder']['students']
        if len(outs) != len(self.losses):
            raise ValueError(f'Outputs do not match loss functions: {len(outs)} vs {len(self.losses)} respectively')

        # main output, deepest decoder, second deepest, ...., shallowest decoder
        # get all losses
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            if self.learn_weights and i > 0:
                l = fn(outs[i], target) * self.params_dec[i-1]
            else:
                l = fn(outs[i], target)
            loss += l

        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


class PixelWiseKLDiv(losses.KLDiv):
    """The pixel wise KL-Divergence loss for semantic segmentation"""
    def __init__(self, *args: Any, target: Optional[str] = None, weight: float = 1, **kwargs: Any) -> None:
        super().__init__(*args, target=target, weight=weight, reduction="none", **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(input, target)
        return loss.sum(dim=1).mean()


# For KL Div Loss
class Self_Distillation_Loss(losses.MultiLosses):

    def __init__(self, *args, include_background: bool = True, T: int = 1, learn_weights: bool = False, weight: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs) 
        """
        Forward: Order of comparison with teacher: 2nd deepest, ..., shallowest
        """

        self.include_background = include_background # whether to include bkg class or not
        self.T = T  # divided by temperature (T) to smooth logits before softmax
        self.learn_weights = learn_weights
        if learn_weights:
            self.params_dec = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses))])
            self.params_enc = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses))])

    def forward(self, input: Dict[str, Union[torch.Tensor, Dict]], target: Any) -> torch.Tensor: # type:ignore
        # If in validation mode, return 0
        if (self.training == False) or ('decoder' not in list(input.keys())):
            loss = torch.tensor(0, dtype=input['out'].dtype, device=input['out'].device)
            return loss

        loss = 0 # initializing the loss value

        ############################################################################################################
        # Decoder teacher / student
        # For KL Div between softmax(out_dec1) [target/teacher] and log_softmax((out_dec2,out_dec3)) [input/students]
        ############################################################################################################
        
        num_inputs = len(input['decoder']['students'])
        if num_inputs != len(self.losses):
            raise ValueError(f'Inputs do not match loss functions: {num_inputs} vs {len(self.losses)} respectively')

        # d_heads[0]: Teacher model output (deepest decoder)
        teacher_logits: torch.Tensor = input['decoder']['teacher']
        # divide by the temperature (T) to smooth logits before softmax
        teacher_logits = teacher_logits / self.T 
        # optional ignore background
        if not self.include_background:
            teacher_logits = teacher_logits[:, 1:]
        # teacher soft max
        teacher: torch.Tensor = F.softmax(teacher_logits, dim=1)
        
        l = 0 # initialization
        for i, fn in enumerate(self.losses):
            # Pulling the student logits
            student_logits = input['decoder']['students'][i]
            # divide by the temperature (T) to smooth logits before softmax
            student_logits = student_logits / self.T
            # optional ignore background
            if not self.include_background:
                student_logits = student_logits[:, 1:]
            # log soft max of student
            student = F.log_softmax(student_logits, dim=1)

            # TODO: consider the self.learn_weights?
            # calculate the loss
            l = fn(student, teacher)

            # add to overall loss and modulate by temperature
            loss += l * (self.T ** 2)


        ############################################################################################################
        # Encoder teacher / student
        # For KL Div between softmax(out_enc4) [target/teacher] and log_softmax((out_enc2,out_enc3)) [input/students]
        ############################################################################################################
        num_inputs = len(input['encoder']['students']) 
        if num_inputs != len(self.losses):
            raise ValueError(f'Inputs do not match loss functions: {num_inputs} vs {len(self.losses)} respectively')

        # e_heads[-1]: Teacher model output (deepest encoder)
        teacher: torch.Tensor = input['encoder']['teacher']
        # divide by the temperature (T) to smooth logits before softmax
        teacher_logits = teacher_logits / self.T 
        # optional ignore background
        if not self.include_background:
            teacher_logits = teacher_logits[:, 1:]
        # teacher soft max
        teacher: torch.Tensor = F.softmax(teacher_logits, dim=1)
        
        l = 0 # initialization
        for i, fn in enumerate(self.losses):
            # Pulling the student logits
            student_logits = input['encoder']['students'][i]
            # divide by the temperature (T) to smooth logits before softmax
            student_logits = student_logits / self.T
            # optional ignore background
            if not self.include_background:
                student_logits = student_logits[:, 1:]
            # log soft max of student
            student = F.log_softmax(student_logits, dim=1)

            # TODO: consider the self.learn_weights?
            # calculate the loss
            l = fn(student, teacher)

            # add to overall loss and modulate by temperature
            loss += l * (self.T ** 2)
        
        return loss