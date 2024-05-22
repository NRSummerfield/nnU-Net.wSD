from __future__ import annotations
from typing import Any, Optional, Union, Sequence

import torch

from .self_distill_nnunet import SelfDistilDynUNet


class SelfDistillnnUNetWithDictOutput(SelfDistilDynUNet):
    """
    A nnUNet that returns a dictionary during training
    
    * extends: `SelfDistilDynUNet`
    """

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        # return {"out": super().forward(x_in)} if self.training else super().forward(x_in)        
        y = super().forward(x_in)
        return {"out": y, "dist_map": y} if self.training else y # type:ignore

