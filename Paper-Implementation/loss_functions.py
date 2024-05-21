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
class Self_Distillation_Loss_Dice(losses.MultiLosses):

    def __init__(self, *args, learn_weights: bool = False, weight: float = 1.0, **kwargs) -> None:

        super().__init__(*args, **kwargs)  

        self.learn_weights = learn_weights
        if learn_weights:
            self.params_dec = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses)-1)])

    def forward(self, input: Union[Sequence[torch.Tensor], torch.Tensor], target: Any) -> torch.Tensor:
        # initilaize
        loss = 0
        l = 0
        
        # Validation Mode
        if isinstance(input, torch.Tensor): return self.losses[0](input, target)
        
        # Training Mode
        # get all losses
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            if self.learn_weights and i > 0:
                l = fn(input[i], target) * self.params_dec[i-1]
            else:
                l = fn(input[i], target)
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
class Self_Distillation_Loss_KL(losses.MultiLosses):

    def __init__(self, *args, include_background: bool = True, T: int = 1, learn_weights: bool = False, weight: float = 1.0, **kwargs) -> None:

        super().__init__(*args, **kwargs) 

        self.include_background = include_background # whether to include bkg class or not
        self.T = T  # divided by temperature (T) to smooth logits before softmax
        self.learn_weights = learn_weights
        if learn_weights:
            self.params_dec = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses))])
            self.params_enc = nn.ParameterList([nn.Parameter(torch.tensor(weight, dtype=torch.float), requires_grad=True) for _ in range(len(self.losses))])

    def forward(self, input: Dict[str, Union[Sequence[torch.Tensor], torch.Tensor]], target: Any) -> torch.Tensor: # type:ignore
        
        # Validation Mode - Just use the main output (out_main)
        if isinstance(input["out"], torch.Tensor): 
            assert (self.training == False), _raise(TypeError("Should be in validation mode.")) 
            loss = torch.tensor(0, dtype=input["out"].dtype, device=input["out"].device) 
            return loss

        ############################################################################################################
        loss = 0
        ## Decoder Teacher-Student
        ## For KL Div between softmax(out_dec1) [target/teacher] and log_softmax((out_dec2,out_dec3)) [input/students]
        
        # Training Mode
        # out_dec0: Teacher model output (deepest decoder)
        target_logits: torch.Tensor = input["out"][5] # for UNETR/nnUnet
                        
        target_logits = target_logits/self.T  # divided by temperature (T) to smooth logits before softmax

        if not self.include_background:
            target_logits = target_logits[:, 1:, ...] # considering only foreground classes
        
        # target softmax probs
        target: torch.Tensor = F.softmax(target_logits, dim=1)  # (B, num_cls, x,y,z): num_cls=7 
                
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize
        l = 0
        
        # get all KL Div losses between softmax(out_dec1)[target] and log_softmax((out_dec2,out_dec3,out_dec4))[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            
            # (B, num_cls, x,y,z): for Decoders - out_dec0 as Teacher and out_dec1, out_dec2, out_dec3, out_dec4 as students
            input_logits_before_log = (input["out"][i+1])/self.T  

            if not self.include_background:
                input_logits_before_log = input_logits_before_log[:, 1:, ...]  # considering only foreground classes

            log_input = F.log_softmax(input_logits_before_log, dim=1)

            if self.learn_weights:
                l = fn(log_input, target) * self.params_dec[i]
            else:
                l = fn(log_input, target)

            loss += l * (self.T ** 2)

        ############################################################################################################
        ## Encoder Teacher-Student [NOT required for special case of dual self-distillation, i.e. deep supervision]
        ## For KL Div between softmax(out_enc4) [target/teacher] and log_softmax((out_enc2,out_enc3)) [input/students]
        
        # Training Mode
        # out_enc5: Teacher model output (deepest encoder)
        target_logits: torch.Tensor = input["out"][10] # for UNETR/nnUnet
                        
        target_logits = target_logits/self.T  # divided by temperature (T) to smooth logits before softmax

        if not self.include_background:
            target_logits = target_logits[:, 1:, ...] # considering only foreground classes
        
        # target softmax probs
        target: torch.Tensor = F.softmax(target_logits, dim=1)  # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class)
                
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize        
        l = 0
        
        # get all KL Div losses between softmax(out_enc4)[target] and log_softmax((out_enc1,out_enc2,out_enc3))[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            
            # (B, num_cls, x,y,z): num_cls=7 (since excluding bkg class): for Encoders - out_enc5 as Teacher and out_enc1, out_enc2, out_enc3, out_enc4 as students
            input_logits_before_log = (input["out"][i+6])/self.T  # for UNETR/nnUnet

            if not self.include_background:
                input_logits_before_log = input_logits_before_log[:, 1:, ...]  # considering only foreground classes

            log_input = F.log_softmax(input_logits_before_log, dim=1)

            if self.learn_weights:
                l = fn(log_input, target) * self.params_enc[i]
            else:
                l = fn(log_input, target)

            loss += l * (self.T ** 2)

        ############################################################################################################
        
        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss


# For L2 loss between feature maps/hints dec1_f/enc4_f [target: Teacher (T)] and feature maps (dec3_f/enc3_f,dec2_f/enc2_f) [input: Students (S)]
class Self_Distillation_Loss_L2(losses.MultiLosses):
    
    def forward(self, input: Dict[str, Union[Sequence[torch.Tensor], torch.Tensor]], target: Any) -> torch.Tensor: # type:ignore

        if isinstance(input["out"], torch.Tensor): 
            assert (self.training == False), _raise(TypeError("Should be in validation mode.")) 
            loss = torch.tensor(0, dtype=input["out"].dtype, device=input["out"].device) 
            return loss

        ############################################################################################################
        ## Decoder Teacher-Student
        # For Decoders
        # dec1_f: Teacher model output (deepest decoder)
        target: torch.Tensor = input["out"][14] 
        
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize
        loss = 0
        l = 0

        # get all L2 losses between feature maps/hints dec1_f[target] and feature maps (dec3_f,dec2_f)[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input["out"][i+12], target) 
            loss += l
        
        ############################################################################################################
        ## Encoder Teacher-Student
        # For Encoders
        # enc4_f: Teacher model output (deepest encoder)
        target: torch.Tensor = input["out"][11] 
        
        assert isinstance(target, torch.Tensor), _raise(TypeError("Target should be a tensor.")) 

        # initilaize
        l = 0

        # get all L2 losses between feature maps/hints enc4_f[target] and feature maps (enc2_f,enc3_f)[input]
        for i, fn in enumerate(self.losses):
            assert isinstance(fn, losses.Loss), _raise(TypeError(f"Function {fn} is not a Loss object."))
            l = fn(input["out"][i+9], target) 
            loss += l
        
        ############################################################################################################

        # return loss
        assert isinstance(loss, torch.Tensor), _raise(TypeError("The total loss is not a valid `torch.Tensor`."))
        return loss

        