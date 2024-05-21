# pyright: reportPrivateImportUsage=false
from typing import List, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.utils import UpsampleMode, InterpolateMode

# Relative import for final training model
# from .deepUp import DeepUp

# Absolute import for testing this script
from .deepUp import DeepUp

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock


class SkipLayer(nn.Module):
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
    

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

class DynUNet_withDualSelfDistillation(nn.Module):
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
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
        res_block: bool = False,
        trans_bias: bool = False,
    ):
        super().__init__()
        # commiting input args to self
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
        self.filters = filters
        
        # setup functions
        self.input_block = self.get_input_block() 
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)

        # flexible creation of the DeepUp layers for self-distillation
        self.n_layers = len(filters)
        for i in range(self.n_layers):
            self.__setattr__(
                name = f'deep_{i}',
                value = DeepUp(
                    spatial_dims=spatial_dims,
                    in_channels=filters[~i],
                    out_channels=out_channels,
                    scale_factor=2 ** (self.n_layers - 1 - i),
                    mode=mode,
                    interp_mode=interp_mode,
                    multiple_upsample=multiple_upsample
                    ) 
                )

        # Lists recording the OUTPUT of the layers for self distllation
        self.e_heads: List[torch.Tensor] = [torch.rand(1)] * (self.n_layers - 1)
        self.d_heads: List[torch.Tensor] = [torch.rand(1)] * (self.n_layers - 1)

        # Lists of Modules used to output each layer in the final shape
        self.self_distillation_enc_heads = nn.ModuleList([self.__getattr__(f'deep_{i}') for i in range(self.n_layers-1, 0, -1)])
        self.self_distillation_dec_heads = nn.ModuleList([self.__getattr__(f'deep_{i}') for i in range(self.n_layers-2, -1, -1)])

        # initialize weights
        self.apply(self.initialize_weights)

        # recursive function to create the skip-layers of the model
        def create_skips(index, downsamples, upsamples, bottleneck, distil_enc_heads=None, distil_dec_heads=None):
            """
            Construct the UNet topology as a sequence of skip layers terminating with the bottleneck layer. This is
            done recursively from the top down since a recursive nn.Module subclass is being used to be compatible
            with Torchscript. Initially the length of `downsamples` will be one more than that of `superheads`
            since the `input_block` is passed to this function as the first item in `downsamples`, however this
            shouldn't be associated with a supervision head.
            """
        
            # condtion to return the bottle neck 
            if len(downsamples) == 0: 
                return bottleneck
                          
            # check to see if there are still heads left in the list or not
            if len(distil_enc_heads) > 0 or len(distil_dec_heads) > 0:                  
                rest_enc_heads = distil_enc_heads
                rest_dec_heads = distil_dec_heads
            
            else:
                rest_enc_heads = nn.ModuleList()
                rest_dec_heads = nn.ModuleList()                    

            # create the next layer down, this will stop at the bottleneck layer
            # EVERTHING below this. Picture a triangle.
            next_layer = create_skips(1 + index, downsamples[1:], upsamples[1:], bottleneck, distil_enc_heads=rest_enc_heads, distil_dec_heads=rest_dec_heads)

            # return the next level up
            return SkipLayer(
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
        # END OF RECURSIVE FUNCTION -----------------------------------------------------------------------------
        
        # trigger the recursive function
        self.skip_layers = create_skips(
            0,
            [self.input_block] + list(self.downsamples),
            self.upsamples[::-1], # type:ignore
            self.bottleneck,
            distil_enc_heads=self.self_distillation_enc_heads,
            distil_dec_heads=self.self_distillation_dec_heads,
        )

    # END OF __init__ -----------------------------------------------------------------------------
    # Functions used to help create the model
        
    # Bottle neck class -> needs to record the output of bottleneck
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

    # pulls the singular block that takes the input volumes
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

    # pulls a singular block that outputs the output volume
    def get_output_block(self, idx: int):
        return UnetOutBlock(self.spatial_dims, self.filters[idx], self.out_channels, dropout=self.dropout)

    # a list of downsamples needed for for the encoder branch after the input block
    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    # a list of upsamples needed for the decoder branch after the output block
    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size, trans_bias=self.trans_bias
        )

    # Function that returns a list of convolutions based off of kernels and strides
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

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    # forward function when called
    def forward(self, x: torch.Tensor):
        """
        If self.training == True:
            out = dict
                # Structure:
                #   out = {
                #       "out": out_main,
                #       "decoder": {
                #           "teacher": deepest,
                #           "students": [second deepest, ..., shallowest]
                #           },
                #       "encoder" : {
                #           "teacher": deepest,
                #           "students": [second deepest, ..., shallowest]
                #           }
                #       }
        else:
            out = prediction
        """
        out_layers = self.skip_layers(x)

        out_main = self.output_block(out_layers)
        
        if self.training:

            # encoder heads are REVERSED
            #   Deepest Encoder = self.e_heads[-1]
            out_enc_heads = [self.e_heads[i] for i in range(self.n_layers-2, -1, -1)]
            # deepest, next level, ..., shallowest level

            # decoder heads are IN ORDER
            #   Deepest Decoder = self.d_heads[0]
            out_dec_heads = [self.d_heads[i] for i in range(0, self.n_layers-1, 1)]
            # deepest, next level, ..., shallowest level

            out_enc = {'teacher': out_enc_heads[0], 'students': out_enc_heads[1:]}
            out_dec = {'teacher': out_dec_heads[0], 'students': out_dec_heads[1:]}

            out = {"out": out_main, 'decoder': out_dec, 'encoder': out_enc}

            return out
        else: return out_main
            


if __name__ == "__main__":
    # try the following to add one more layer to nnUnet (implemention of DynUnet from MONAI)
    kernel_size = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    filters = [32,64,128,256,512,1024] 


    nnunet_with_self_distil = DynUNet_withDualSelfDistillation(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 13,
        kernel_size = kernel_size,
        strides = strides,
        upsample_kernel_size = strides[1:],
        filters=filters,
        norm_name="instance",
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
