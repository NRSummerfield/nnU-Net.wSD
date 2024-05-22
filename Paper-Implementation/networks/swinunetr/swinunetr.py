from __future__ import annotations
from typing import Any, Optional, Union, Sequence

import torch

from monai.networks.nets.swin_unetr import SwinUNETR
from .self_distill_swinunetr import SelfDistilSwinUNETR


class SwinUNETRWithDictOutput(SwinUNETR):
    """
    A SwinUNETR that returns a dictionary during training
    
    * extends: `SwinUNETR`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        #print("Input shape:", x_in)
        # repeat modality for input tensor
        return {"out": super().forward(x_in)} if self.training else super().forward(x_in)


class SelfDistillSwinUNETRWithDictOutput(SelfDistilSwinUNETR):
    """
    A SwinUNETR that returns a dictionary during training
    
    * extends: `SelfDistilSwinUNETR`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # return {"out": super().forward(x_in)} if self.training else super().forward(x_in)        
        y = super().forward(x_in)
        return {"out": y, "dist_map": y} if self.training else y # type:ignore

