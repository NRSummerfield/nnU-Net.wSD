# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat
from monai.utils import ensure_tuple_rep

if __name__ == '__main__':
    from deepUp import DeepUp
else: 
    # module to upsample the output of a convolutional layer
    from .deepUp import DeepUp

__all__ = ["BasicUnetPlusPlus", "BasicunetPlusPlus", "basicunetplusplus", "BasicUNetPlusPlus"]


class ModuleWrapper(nn.Module):
    def __init__(self, block: nn.Module, head: list[torch.Tensor], index:int = 0):
        super().__init__()
        self.block = block
        self.head = head
        self.i = index

    def forward(self, *args, **kwargs) -> torch.Tensor:
        out = self.block(*args, **kwargs)
        self.head[self.i] = out
        return out

class BasicUNetPlusPlus_withSD(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        deep_supervision: bool = False,
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet++ implementation with 1D/2D/3D supports.

        Based on:

            Zhou et al. "UNet++: A Nested U-Net Architecture for Medical Image
            Segmentation". 4th Deep Learning in Medical Image Analysis (DLMIA)
            Workshop, DOI: https://doi.org/10.48550/arXiv.1807.10165


        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            deep_supervision: whether to prune the network at inference time. Defaults to False. If true, returns a list,
                whose elements correspond to outputs at different nodes.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with deep supervision enabled
            >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), deep_supervision=True)

            # for spatial 2D, with group norm
            >>> net = BasicUNetPlusPlus(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))

        See Also
            - :py:class:`monai.networks.nets.BasicUNet`
            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        self.deep_supervision = deep_supervision

        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNetPlusPlus features: {fea}.")

        self.n_layers = 4
        self.down_heads: list[torch.Tensor] = [torch.rand(1)] * self.n_layers # 5 is the number of layers
        self.up_heads: list[torch.Tensor] = [torch.rand(1)] * self.n_layers 
        
        # Creating a "deep_up" module for each level in the model based off of feature size
        scaling_factors = [1, 2, 4, 8, 16, 1]
        # Manually defined the scaling factors to fit with the way this model's architecture is set up (?)
        for i in range(len(fea)):
            self.__setattr__(
                name = f'deep_{i}',
                value = DeepUp(
                    spatial_dims=spatial_dims,
                    in_channels=fea[i],
                    out_channels=out_channels,
                    scale_factor=scaling_factors[i],
                    # mode=mode,
                    # interp_mode=interp_mode,
                    # multiple_upsample=multiple_upsample
                    ) 
                ) 
            
        # Manually defining which "deep_up" block to use for each of the heads in encoding / decoding branches
        self.encoding_fea_list = [1, 2, 3, 4]
        self.decoding_fea_list = [0, 0, 0, 5]

        self.conv_0_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)

        # Assuming these are the MAIN encoding branch 
        self.conv_1_0 = ModuleWrapper(
            block=Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout),
            head=self.down_heads,
            index=0
        )
        self.conv_2_0 = ModuleWrapper(
            block=Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout),
            head=self.down_heads,
            index=1
        )
        self.conv_3_0 = ModuleWrapper(
            block=Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout),
            head=self.down_heads,
            index=2
        )
        # This is the DEEPEST encoder (index 3) and is the teacher for this model
        self.conv_4_0 = ModuleWrapper(
            block=Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout),
            head=self.down_heads,
            index=3
        )

        # Assuming the 0_X series are the MAIN decoding branch
        self.upcat_0_1 = ModuleWrapper(
            block=UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False),
            head=self.up_heads,
            index=0
        )
        # self.upcat_0_1 = UpCat(spatial_dims, fea[1], fea[0], fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_1 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_2_1 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_3_1 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)


        self.upcat_0_2 =  ModuleWrapper(
            block=UpCat(spatial_dims, fea[1], fea[0] * 2, fea[0], act, norm, bias, dropout, upsample, halves=False),
            head=self.up_heads,
            index=1
        )
        # self.upcat_0_2 = UpCat(spatial_dims, fea[1], fea[0] * 2, fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_2 = UpCat(spatial_dims, fea[2], fea[1] * 2, fea[1], act, norm, bias, dropout, upsample)
        self.upcat_2_2 = UpCat(spatial_dims, fea[3], fea[2] * 2, fea[2], act, norm, bias, dropout, upsample)

        self.upcat_0_3 = ModuleWrapper(
            block=UpCat(spatial_dims, fea[1], fea[0] * 3, fea[0], act, norm, bias, dropout, upsample, halves=False),
            head=self.up_heads,
            index=2
        )
        # self.upcat_0_3 = UpCat(spatial_dims, fea[1], fea[0] * 3, fea[0], act, norm, bias, dropout, upsample, halves=False)
        self.upcat_1_3 = UpCat(spatial_dims, fea[2], fea[1] * 3, fea[1], act, norm, bias, dropout, upsample)

        # This is the last ("Deepest") decoder and the teacher for this model
        self.upcat_0_4 = ModuleWrapper(
            block=UpCat(spatial_dims, fea[1], fea[0] * 4, fea[5], act, norm, bias, dropout, upsample, halves=False),
            head=self.up_heads,
            index=3
        )
        # self.upcat_0_4 = UpCat(spatial_dims, fea[1], fea[0] * 4, fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv_0_1 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_2 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_3 = Conv["conv", spatial_dims](fea[0], out_channels, kernel_size=1)
        self.final_conv_0_4 = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x_0_0 = self.conv_0_0(x)
        x_1_0 = self.conv_1_0(x_0_0)
        x_0_1 = self.upcat_0_1(x_1_0, x_0_0)

        x_2_0 = self.conv_2_0(x_1_0)
        x_1_1 = self.upcat_1_1(x_2_0, x_1_0)
        x_0_2 = self.upcat_0_2(x_1_1, torch.cat([x_0_0, x_0_1], dim=1))

        x_3_0 = self.conv_3_0(x_2_0)
        x_2_1 = self.upcat_2_1(x_3_0, x_2_0)
        x_1_2 = self.upcat_1_2(x_2_1, torch.cat([x_1_0, x_1_1], dim=1))
        x_0_3 = self.upcat_0_3(x_1_2, torch.cat([x_0_0, x_0_1, x_0_2], dim=1))

        x_4_0 = self.conv_4_0(x_3_0)
        x_3_1 = self.upcat_3_1(x_4_0, x_3_0)
        x_2_2 = self.upcat_2_2(x_3_1, torch.cat([x_2_0, x_2_1], dim=1))
        x_1_3 = self.upcat_1_3(x_2_2, torch.cat([x_1_0, x_1_1, x_1_2], dim=1))
        x_0_4 = self.upcat_0_4(x_1_3, torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=1))

        output_0_1 = self.final_conv_0_1(x_0_1)
        output_0_2 = self.final_conv_0_2(x_0_2)
        output_0_3 = self.final_conv_0_3(x_0_3)
        output_0_4 = self.final_conv_0_4(x_0_4)

        # Going through the heads and upsampling to match output
        out_down_heads = []
        out_up_heads = []
        for i in range(self.n_layers):
            # Pull the ith head and pass through the ith deep_up module
            encoding_fea_id = self.encoding_fea_list[i]
            out_down_head = self.__getattr__(f'deep_{encoding_fea_id}')(self.down_heads[i])
            print(out_down_head.shape)
            out_down_heads.append(out_down_head)

            decoding_fea_id = self.decoding_fea_list[i]
            out_up_head = self.__getattr__(f'deep_{decoding_fea_id}')(self.up_heads[i])
            print(out_up_head.shape)
            out_up_heads.append(out_up_head)
        
        # organzing the outputs in a dictionary
        self_distillation_out = {
            'encoder': {'teacher': out_down_heads[-1], 'students': out_down_heads[:-1]}, 
            'decoder': {'teacher': out_up_heads[-1], 'students': out_up_heads[:-1]}}


        if self.deep_supervision:
            output = [output_0_1, output_0_2, output_0_3, output_0_4]
        else:
            output = [output_0_4]

        return output, self_distillation_out


# BasicUnetPlusPlus = BasicunetPlusPlus = basicunetplusplus = BasicUNetPlusPlus


if __name__ == '__main__':

    model = BasicUNetPlusPlus_withSD()

    in_volume = torch.rand((1, 1, 96, 96, 96))

    out_volume, selfdistill_out = model(in_volume)

    print(f'Self Distillation Output:')

    print(f'  Encoder:')
    print(f'    Teacher:', selfdistill_out['encoder']['teacher'].shape)
    print(f'    Students:', [vol.shape for vol in selfdistill_out['encoder']['students']])

    print(f'  Decoder:')
    print(f'    Teacher:', selfdistill_out['decoder']['teacher'].shape)
    print(f'    Students:', [vol.shape for vol in selfdistill_out['decoder']['students']])
    

