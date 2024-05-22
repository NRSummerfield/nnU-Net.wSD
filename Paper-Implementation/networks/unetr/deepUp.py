# pyright: reportPrivateImportUsage=false
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode, UpsampleMode


class DeepUp(nn.Module):
    """
    An upsampling module that can be used for UNETR/nnUNet/SwinUNETR"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        scale_factor: int, 
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            scale_factor: scale by which upsampling is needed
            mode: whether to use trainable DECONV or NONTRAINABLE mode for UpsampleMode,
            interp_mode: The type of interpolation (Linear/Bilinear/TriLinear) for NONTRAINABLE UpsampleMode
        """

        super().__init__()

        self.multiple_upsample = multiple_upsample
        self.scale_factor = scale_factor

        self.conv_block = UnetOutBlock(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels
        )

        if self.multiple_upsample and scale_factor > 2:    
            if scale_factor == 4:    
                self.transp_conv1 = UpSample(
                    spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor//2,
                    mode=mode,
                    interp_mode=interp_mode,
                    bias=True,
                    apply_pad_pool=True,
                )
            elif scale_factor == 8:    
                self.transp_conv1 = UpSample(
                    spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor//4,
                    mode=mode,
                    interp_mode=interp_mode,
                    bias=True,
                    apply_pad_pool=True,
                )
            elif scale_factor == 16:    
                self.transp_conv1 = UpSample(
                    spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor//8,
                    mode=mode,
                    interp_mode=interp_mode,
                    bias=True,
                    apply_pad_pool=True,
                )
            elif scale_factor == 32:    
                self.transp_conv1 = UpSample(
                    spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor//16,
                    mode=mode,
                    interp_mode=interp_mode,
                    bias=True,
                    apply_pad_pool=True,
                )
            elif scale_factor == 64:    
                self.transp_conv1 = UpSample(
                    spatial_dims,
                    in_channels=out_channels,
                    out_channels=out_channels,
                    scale_factor=scale_factor//32,
                    mode=mode,
                    interp_mode=interp_mode,
                    bias=True,
                    apply_pad_pool=True,
                )
            else:
                raise NotImplementedError("Too high scale factor for upsampling.")

        else:
            self.transp_conv1 = UpSample(
                spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                scale_factor=scale_factor,
                mode=mode,
                interp_mode=interp_mode,
                bias=True,
                apply_pad_pool=True,
            )

    def forward(self, inp):        
        out = self.conv_block(inp)
        if self.multiple_upsample and self.scale_factor > 2:
            if self.scale_factor == 4:
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
            elif self.scale_factor == 8:
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
            elif self.scale_factor == 16:
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
            elif self.scale_factor == 32:
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
            elif self.scale_factor == 64:
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
                out = self.transp_conv1(out)
            else:
                raise NotImplementedError("Too high scale factor for upsampling.")
        else:
            out = self.transp_conv1(out)

        return out


if __name__ == '__main__':
    block_connector = DeepUp(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 8, 
        scale_factor=16, 
        mode=UpsampleMode.DECONV, 
        interp_mode=InterpolateMode.BILINEAR
        )  

    x1 = torch.rand((1, 1, 4, 4, 4)) # (B,in_ch,x,y,z)
    print("Deep Upsampling block input shape: ", x1.shape)

    x3 = block_connector(x1)
    print("Deep Upsampling block output shape: ", x3.shape)
