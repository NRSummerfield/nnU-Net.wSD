from __future__ import annotations
from typing import Any, Optional, Union, Sequence

import torch

from monai.networks.nets.unetr import UNETR
from .self_distill_unetr import SelfDistilUNETR


class UNETRWithDictOutput(UNETR):
    """
    A UNETR that returns a dictionary during training
    
    * extends: `UNETR`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        #print("Input shape:", x_in)
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)


class SelfDistillUNETRWithDictOutput(SelfDistilUNETR):
    """
    A UNETR that returns a dictionary during training
    
    * extends: `SelfDistilUNETR`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # return {"out": super().forward(x_in)} if self.training else super().forward(x_in)        
        y = super().forward(x_in)
        return {"out": y, "dist_map": y} if self.training else y # type:ignore

