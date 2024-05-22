# pyright: reportPrivateImportUsage=false
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.utils import UpsampleMode, InterpolateMode

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock


class DynUNetSkipLayer(nn.Module):
    """
    Defines a layer in the UNet topology which combines the downsample and upsample pathways with the skip connection.
    The member `next_layer` may refer to instances of this class or the final bottleneck layer at the bottom the UNet
    structure. The purpose of using a recursive class like this is to get around the Torchscript restrictions on
    looping over lists of layers and accumulating lists of output tensors which must be indexed. The `heads` list is
    shared amongst all the instances of this class and is used to store the output from the supervision heads during
    forward passes of the network.
    """

    heads: Optional[List[torch.Tensor]]

    def __init__(self, index, downsample, upsample, next_layer, heads=None, super_head=None, e_heads= None, d_heads=None, enc_head=None, dec_head=None):
        super().__init__()
        self.downsample = downsample
        self.next_layer = next_layer
        self.upsample = upsample
        self.super_head = super_head
        self.heads = heads
        self.index = index  

        self.enc_head = enc_head
        self.dec_head = dec_head
        self.e_heads = e_heads
        self.d_heads = d_heads   

    def forward(self, x):        
        downout = self.downsample(x)

        # for self-distillation
        if self.super_head is None and self.enc_head is not None and self.e_heads is not None:
            self.e_heads[self.index] = self.enc_head(downout)  

        nextout = self.next_layer(downout)

        # for self-distillation
        if self.super_head is None and self.dec_head is not None and self.d_heads is not None:
            self.d_heads[self.index] = self.dec_head(nextout)        

        upout = self.upsample(nextout, downout)

        # for deep supervision
        if self.super_head is not None and self.enc_head is None and self.dec_head is None and self.heads is not None and self.index > 0:
            self.heads[self.index - 1] = self.super_head(upout)

        return upout


class SelfDistilDynUNet(nn.Module):
    """
    This reimplementation of a dynamic UNet (DynUNet) is based on:
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
            equal to strides[1:].
        filters: number of output channels for each blocks. Different from nnU-Net, in this implementation we add
            this argument to make the network more flexible. 
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
        act_name: activation layer type and arguments. Defaults to ``leakyrelu``.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
        deep_supr_num: number of feature maps that will output during deep supervision head. Defaults to 1. 
        res_block: whether to use residual connection based convolution blocks during the network. Defaults to ``False``.
        trans_bias: whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        self_distillation: bool = False,
        self_distillation_num: int = 4,
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
        dataset: str = "MMWHS",
        res_block: bool = False,
        trans_bias: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)

        # The following is required for built-in deep supervision
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.heads: List[torch.Tensor] = [torch.rand(1)] * self.deep_supr_num
        if self.deep_supervision:
            self.deep_supervision_heads = self.get_deep_supervision_heads()
            self.check_deep_supr_num()

        #########################################
        # Upsample blocks for Self Distillation #
        #########################################

        if self_distillation:
            if dataset == "MMWHS":

                self.deep_0 = DeepUp(
                spatial_dims = 3,
                # in_channels = 320,
                in_channels = filters[-1],
                out_channels = self.out_channels,
                scale_factor = 32,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                ) 

                self.deep_1 = DeepUp(
                spatial_dims = 3,
                # in_channels = 320,
                in_channels = filters[-2],
                out_channels = self.out_channels,
                scale_factor = 16,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                ) 
                
                self.deep_2 = DeepUp(
                spatial_dims = 3,
                # in_channels = 256,
                in_channels = filters[-3],
                out_channels = self.out_channels,
                scale_factor = 8,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                ) 

                self.deep_3 = DeepUp(
                spatial_dims = 3,
                # in_channels = 128,
                in_channels = filters[-4],
                out_channels = self.out_channels,
                scale_factor = 4,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                ) 

                self.deep_4 = DeepUp(
                spatial_dims = 3,
                # in_channels = 64,
                in_channels = filters[-5],
                out_channels = self.out_channels,
                scale_factor = 2,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                )

                self.deep_5 = DeepUp(
                spatial_dims = 3,
                # in_channels = 32,
                in_channels = filters[-6],
                out_channels = self.out_channels,
                scale_factor = 1,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
                ) 

            else:
                raise NotImplementedError(f"Upsample modules not implemented for {dataset} dataset.")


        # The following is required for proposed self-distillation along with deep supervision        
        self.self_distillation = self_distillation
        self.self_distillation_num = self_distillation_num
        # initialize the typed list of supervision head outputs so that Torchscript can recognize what's going on
        self.e_heads: List[torch.Tensor] = [torch.rand(1)] * self.self_distillation_num  # Encoder Heads
        self.d_heads: List[torch.Tensor] = [torch.rand(1)] * self.self_distillation_num  # Decoder Heads
        
        if self.self_distillation: 
            if dataset == "MMWHS":
                # for MMWHS                                     SF:     1           2          4      8           16
                self.self_distillation_enc_heads = nn.ModuleList([self.deep_5,self.deep_4,self.deep_3,self.deep_2,self.deep_1])
                #                                               SF:     2           4          8      16          32
                self.self_distillation_dec_heads = nn.ModuleList([self.deep_4,self.deep_3,self.deep_2,self.deep_1,self.deep_0])

            else:
                raise NotImplementedError(f"Upsample modules not implemented for {dataset} dataset.")

        self.apply(self.initialize_weights)
        self.check_kernel_stride()

        def create_skips(index, downsamples, upsamples, bottleneck, superheads=None, distil_enc_heads=None, distil_dec_heads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """

            if len(downsamples) != len(upsamples):
                raise ValueError(f"{len(downsamples)} != {len(upsamples)}")

            if len(downsamples) == 0:  # bottom of the network, pass the bottleneck block
                return bottleneck

            if superheads is None and distil_enc_heads is None and distil_dec_heads is None:
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck)                
                return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

            if superheads is not None and distil_enc_heads is None and distil_dec_heads is None:

                super_head_flag = False
                if index == 0:  # don't associate a supervision head with self.input_block
                    rest_heads = superheads
                    
                else:
                    if len(superheads) > 0:
                        super_head_flag = True
                        rest_heads = superheads[1:]
                    else:                        
                        rest_heads = nn.ModuleList()

                # create the next layer down, this will stop at the bottleneck layer
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=rest_heads)
                
                if super_head_flag:
                    return DynUNetSkipLayer(
                        index,
                        downsample=downsamples[0],
                        upsample=upsamples[0],
                        next_layer=next_layer,
                        heads=self.heads,
                        super_head=superheads[0],
                    )

                return DynUNetSkipLayer(index, downsample=downsamples[0], upsample=upsamples[0], next_layer=next_layer)

            # HERE -> for self distillation
            if superheads is None and distil_enc_heads is not None and distil_dec_heads is not None:                  
                if len(distil_enc_heads) > 0 or len(distil_dec_heads) > 0:                  
                    rest_enc_heads = distil_enc_heads
                    rest_dec_heads = distil_dec_heads
                
                else:
                    rest_enc_heads = nn.ModuleList()
                    rest_dec_heads = nn.ModuleList()                    

                # create the next layer down, this will stop at the bottleneck layer
                # print(f'Recursive {index}')
                next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, superheads=None, distil_enc_heads=rest_enc_heads, distil_dec_heads=rest_dec_heads)

                # print(f'Returning Skip for index: {index}')
                return DynUNetSkipLayer(
                    index,
                    downsample=downsamples[0],
                    upsample=upsamples[0],
                    next_layer=next_layer,
                    heads=None,
                    super_head=None,
                    e_heads=self.e_heads,
                    d_heads=self.d_heads,
                    enc_head=distil_enc_heads[index],
                    dec_head=distil_dec_heads[index],
                )
        # ----------------------------------------------------------------------------------------------------------------------------------#

        if not self.deep_supervision and not self.self_distillation:          
            self.skip_layers = create_skips(
                0, [self.input_block] + list(self.downsamples), self.upsamples[::-1], self.bottleneck # type:ignore
            )
   
        elif self.deep_supervision and not self.self_distillation:    
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1], # type:ignore
                self.bottleneck,
                superheads=self.deep_supervision_heads,
            )

        # HERE -> self distillation: create the skips
        elif self.self_distillation and not self.deep_supervision:  
            self.skip_layers = create_skips(
                0,
                [self.input_block] + list(self.downsamples),
                self.upsamples[::-1], # type:ignore
                self.bottleneck,
                superheads=None,
                distil_enc_heads=self.self_distillation_enc_heads,
                distil_dec_heads=self.self_distillation_dec_heads,
            )
            # print(type(self.skip_layers), self.skip_layers)

        else:
            raise NotImplementedError("Deep Supervision and Self-Distillation cannot be the True at the same time. Please select one as True and the other as False.")

    def check_kernel_stride(self):
        kernels, strides = self.kernel_size, self.strides
        error_msg = "length of kernel_size and strides should be the same, and no less than 3."
        if len(kernels) != len(strides) or len(kernels) < 3:
            raise ValueError(error_msg)

        for idx, k_i in enumerate(kernels):
            kernel, stride = k_i, strides[idx]
            if not isinstance(kernel, int):
                error_msg = f"length of kernel_size in block {idx} should be the same as spatial_dims."
                if len(kernel) != self.spatial_dims:
                    raise ValueError(error_msg)
            if not isinstance(stride, int):
                error_msg = f"length of stride in block {idx} should be the same as spatial_dims."
                if len(stride) != self.spatial_dims:
                    raise ValueError(error_msg)

    def check_deep_supr_num(self):
        deep_supr_num, strides = self.deep_supr_num, self.strides
        num_up_layers = len(strides) - 1
        if deep_supr_num >= num_up_layers:
            raise ValueError("deep_supr_num should be less than the number of up sample layers.")
        if deep_supr_num < 1:
            raise ValueError("deep_supr_num should be larger than 0.")

    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]

    def forward(self, x):        
        out_layers = self.skip_layers(x)  # type:ignore

        out_main = self.output_block(out_layers)

        # The if section is not used for 3D DSD approach
        if self.training and self.deep_supervision:
            out_all = [out_layers]
            for feature_map in self.heads:
                out_all.append(interpolate(feature_map, out_layers.shape[2:]))
            return torch.stack(out_all, dim=1)

        elif self.training and self.self_distillation:
            
            # nnUNet Encoder Bottleneck Outputs
            out_enc5 = self.e_heads[4]  # Deepest Encoder -> SF 16
            out_enc4 = self.e_heads[3]  # Next Deepest Encoder -> SF 8
            out_enc3 = self.e_heads[2]  # Shallow Encoder -> SF 4
            out_enc2 = self.e_heads[1]  # Shallow Encoder -> SF 2
            out_enc1 = self.e_heads[0]  # Shallow Encoder -> SF 1

            # nnUNet Decoder Bottleneck Outputs
            out_dec0 = self.d_heads[0]  # Deepest Decoder -> SF 2 
            out_dec1 = self.d_heads[1]  # Next Deepest Decoder -> SF 4
            out_dec2 = self.d_heads[2]  # Shallow Decoder -> SF 8
            out_dec3 = self.d_heads[3]  # Shallow Decoder -> SF 16
            out_dec4 = self.d_heads[4]  # Shallow Decoder -> SF 32

            # For Deep Supervision ONLY - currently done for MMWHS CT with nnUnet
            # out = (out_main, out_dec3, out_dec2, out_dec1, out_dec0) 
            
            # For nnUNet with Self-Distillation
            out = (out_main, out_dec4, out_dec3, out_dec2, out_dec1, out_dec0, out_enc1, out_enc2, out_enc3, out_enc4, out_enc5) # Full KL Div WITHOUT feature maps - includes both encoders and decoders (Also works with encoders only - just do not add the decoder KL divergence loss for encoders only)

        else: # For Basic nnUNet ONLY and validation mode
            out = out_main
        
        return out


    def get_input_block(self):
        return self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_bottleneck(self):
        return self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            self.norm_name,
            self.act_name,
            dropout=self.dropout,
        )

    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size, trans_bias=self.trans_bias
        )

    def get_module_list(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: Type[nn.Module],
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        trans_bias: bool = False,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(i + 1) for i in range(self.deep_supr_num)])

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


if __name__ == '__main__':

    # originally used for 3D DSD experiments
    # kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    # strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    # filters = [32,64,128,256,320]  # originally used for MMWHS
    # # filters = [16,32,64,128,256] # try this

    # try the following to add one more layer to nnUnet (implemention of DynUnet from MONAI)
    kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    filters = [32,64,128,256,512,1024] 
    

    nnunet_with_self_distil = SelfDistilDynUNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 13,
        kernel_size = kernel_size,
        strides = strides,
        upsample_kernel_size = strides[1:],
        filters=filters,
        norm_name="instance",
        deep_supervision=False,
        deep_supr_num=3,
        self_distillation=True, # for self-distillation and deep supervision
        self_distillation_num=5, # the number of layers for self-distillation and deep supervision
        dataset="MMWHS",
        res_block=True,
        )

    ## Count model parameters
    total_params = sum(p.numel() for p in nnunet_with_self_distil.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')
    
    x1 = torch.rand((1, 1, 96, 96, 96)) # (B,num_ch,x,y,z)
    print("Self Distil nnUNet input shape: ", x1.shape)

    x3 = nnunet_with_self_distil(x1)
    print("Self Distil nnUNet output shape: ", x3[10].shape)

    # x3 = nnunet_with_self_distil(x1)
    # print("Self Distil nnUNet output shape: ", x3.shape)
