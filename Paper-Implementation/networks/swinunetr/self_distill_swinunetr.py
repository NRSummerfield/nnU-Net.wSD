# pyright: reportPrivateImportUsage=false
from typing import Sequence, Tuple, Union, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock

from monai.utils import ensure_tuple_rep
from monai.utils import UpsampleMode, InterpolateMode

# Relative import for final training model
from .swin_unetr_blocks import SwinTransformer

# Absolute import for testing this script
# from swin_unetr_blocks import SwinTransformer

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp

class SelfDistilSwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 36,
        self_distillation: bool = False,
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)

        self.self_distillation = self_distillation
        
        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        ############
        # Encoders #
        ############
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        ############
        # Decoders #
        ############
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        #########################################
        # Upsample blocks for Self Distillation #
        #########################################

        if self_distillation:            
            self.deep_enc0 = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size,
            out_channels = out_channels,
            scale_factor = 1,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_enc1 = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size,
            out_channels = out_channels,
            scale_factor = 2,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_enc2 = DeepUp(
            spatial_dims = 3,
            in_channels = 2 * feature_size,
            out_channels = out_channels,
            scale_factor = 4,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_enc3 = DeepUp(
            spatial_dims = 3,
            in_channels = 4 * feature_size,
            out_channels = out_channels,
            scale_factor = 8,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_enc4 = DeepUp(
            spatial_dims = 3,
            in_channels = 8 * feature_size,
            out_channels = out_channels,
            scale_factor = 16,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_dec4 = self.deep_enc4

            self.deep_dec3 = self.deep_enc3

            self.deep_dec2 = self.deep_enc2

            self.deep_dec1 = self.deep_enc1

        #############
        # Out block #
        #############
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels
        )  # type: ignore

    def forward(self, x_in):
        # Main SwinViT Transformer
        hidden_states_out = self.swinViT(x_in, self.normalize)

        #################################################
        # Encoders and Upsamplers for Self Distillation #
        #################################################

        # Encoder 0 (Shallow Encoder)
        enc0 = self.encoder1(x_in)

        if self.self_distillation:
            out_enc0 = self.deep_enc0(enc0) 
        else:
            out_enc0 = None     

        enc1 = self.encoder2(hidden_states_out[0])

        # Encoder 1 (Shallow Encoder)
        if self.self_distillation:
            out_enc1 = self.deep_enc1(enc1) 
        else:
            out_enc1 = None 


        enc2 = self.encoder3(hidden_states_out[1])

        # Encoder 2 (Shallow Encoder)
        if self.self_distillation:
            out_enc2 = self.deep_enc2(enc2) 
        else:
            out_enc2 = None 


        enc3 = self.encoder4(hidden_states_out[2])

        # Encoder 3 (Deepest Encoder)
        if self.self_distillation:
            out_enc3 = self.deep_enc3(enc3) 
        else:
            out_enc3 = None 

        # Encoder 4 (Deepest Encoder) - currently NOT used
        if self.self_distillation:
            out_enc4 = self.deep_enc4(hidden_states_out[3]) 
        else:
            out_enc4 = None 

        # Bottom-most layer between encoder and decoder
        # No DeepUp layers for this one
        dec4 = self.encoder10(hidden_states_out[4])

        #################################################
        # Decoders and Upsamplers for Self Distillation #
        #################################################

        dec3 = self.decoder5(dec4, hidden_states_out[3])

        # Decoder 4 (Shallow Decoder)
        if self.self_distillation:
            out_dec4 = self.deep_dec4(dec3) 
        else:
            out_dec4 = None 

        dec2 = self.decoder4(dec3, enc3)

        # Decoder 3 (Shallow Decoder)
        if self.self_distillation:
            out_dec3 = self.deep_dec3(dec2) 
        else:
            out_dec3 = None 

        dec1 = self.decoder3(dec2, enc2)

        # Decoder 2 (Deep Decoder)
        if self.self_distillation:
            out_dec2 = self.deep_dec2(dec1) 
        else:
            out_dec2 = None 

        dec0 = self.decoder2(dec1, enc1)

        # Decoder 1 (Deepest Decoder)
        if self.self_distillation:
            out_dec1 = self.deep_dec1(dec0) 
        else:
            out_dec1 = None 

        out = self.decoder1(dec0, enc0)

        # Prepare output layers
        out_main = self.out(out)

        # For Self Distillation (ONLY during training)
        if self.training and self.self_distillation:
            ## For Self Distillation
            # Encoders:out_enc4: deepest encoder and out_enc3, out_enc2, out_enc1: shallow encoders
            # Decoders:out_dec1: deepest decoder and out_dec2, out_dec3, out_dec4: shallow decoders                        
            
            # Full KL Div WITHOUT feature maps - includes both encoders and decoders
            out = (out_main, out_dec4, out_dec3, out_dec2, out_dec1, out_enc0, out_enc1, out_enc2, out_enc3)
            
        elif self.training and not self.self_distillation:
            # For Basic SwinUNETR ONLY (NO Self-Distillation)
            out = out_main  
        else:
            # For validation/testing (NO Self-Distillation)
            out = out_main

        return out


if __name__ == '__main__':
    self_distil_swinunetr = SelfDistilSwinUNETR(
        img_size = (96,96,96),
        in_channels = 1,
        out_channels = 8,
        feature_size=36,    
        self_distillation=True,
        mode=UpsampleMode.DECONV, 
        interp_mode=InterpolateMode.BILINEAR,     
        )

    ## Count model parameters
    total_params = sum(p.numel() for p in self_distil_swinunetr.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')
    

    x1 = torch.rand((1, 1, 96, 96, 96)) # (B,num_ch,x,y,z)
    print("SwinUNetr input shape: ", x1.shape)

    
    # x3 = self_distil_swinunetr(x1)
    # print("Basic SwinUNetr output shape: ", x3.shape)

    x4 = self_distil_swinunetr(x1)
    print("Self Distil SwinUNetr output shape: ", x4[1].shape)

