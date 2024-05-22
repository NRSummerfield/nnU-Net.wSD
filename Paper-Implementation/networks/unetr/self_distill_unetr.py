# pyright: reportPrivateImportUsage=false
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep

from monai.utils import UpsampleMode, InterpolateMode

# Relative import for final training model
from .deepUp import DeepUp

# Absolute import for testing this script
# from deepUp import DeepUp

class SelfDistilUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        self_distillation: bool = False,
        mode: Union[UpsampleMode, str] = UpsampleMode.DECONV,
        interp_mode: Union[InterpolateMode, str] = InterpolateMode.LINEAR,
        multiple_upsample: bool = True,
        use_feature_maps: bool = False,
        feature_size: int = 16, # 16 used for MSD-BraTS and MMWHS(previous)
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            self_distillation: whether to use use self-distillation or not in UNETR architecture,
            mode: whether to use trainable DECONV or NONTRAINABLE mode for UpsampleMode,
            interp_mode: The type of interpolation (Linear/Bilinear/TriLinear) for NONTRAINABLE UpsampleMode,
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.
        Examples::
            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')
             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)
            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        
        self.self_distillation = self_distillation
        self.use_feature_maps = use_feature_maps
        if self.use_feature_maps:
            assert self.self_distillation == True, f"The self-distillation has to be `True` when using feature-map distillation, got self_distillation as {self.self_distillation}."

        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
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
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        ############
        # Decoders #
        ############

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        #########################################
        # Upsample blocks for Self Distillation #
        #########################################

        if self_distillation:            
            self.deep_2_enc = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 8,
            out_channels = out_channels,
            scale_factor = 8,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_2_dec = DeepUp(
            spatial_dims = 3,
            in_channels = hidden_size,
            out_channels = out_channels,
            scale_factor = 16,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )       

            self.deep_3_enc = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 4,
            out_channels = out_channels,
            scale_factor = 4,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_3_dec = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 8,
            out_channels = out_channels,
            scale_factor = 8,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_4_enc = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 2,
            out_channels = out_channels,
            scale_factor = 2,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_4_dec = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 4,
            out_channels = out_channels,
            scale_factor = 4,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_5_enc = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size,
            out_channels = out_channels,
            scale_factor = 1,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

            self.deep_5_dec = DeepUp(
            spatial_dims = 3,
            in_channels = feature_size * 2,
            out_channels = out_channels,
            scale_factor = 2,
            mode=mode, 
            interp_mode=interp_mode,
            multiple_upsample=multiple_upsample
            )

        ###############################################
        # Upsample blocks (Required for Feature Maps) #
        ###############################################
        if self_distillation and use_feature_maps:
            self.deep_5_f = DeepUp(
                spatial_dims=3,
                in_channels=feature_size*2,
                out_channels=128,
                scale_factor=2,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
            )

            self.deep_4_f = DeepUp(
                spatial_dims=3,
                in_channels=feature_size*4,
                out_channels=128,
                scale_factor=4,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
            )

            self.deep_3_f = DeepUp(
                spatial_dims,
                in_channels=feature_size*8,
                out_channels=128,
                scale_factor=8,
                mode=mode, 
                interp_mode=interp_mode,
                multiple_upsample=multiple_upsample
            )

            ###############################################
            # Average blocks (Required for Feature Maps) #
            ###############################################
            self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))

        #############
        # Out block #
        #############

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))  # type:ignore
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x    

    def forward(self, x_in):
        # Main ViT Transformer
        x, hidden_states_out = self.vit(x_in)

        #################################################
        # Encoders and Upsamplers for Self Distillation #
        #################################################
        
        # Encoder 1 (Shallow Encoder)
        enc1 = self.encoder1(x_in) 

        if self.self_distillation:
            out_enc1 = self.deep_5_enc(enc1) 
        else:
            out_enc1 = None      
        
        x2 = hidden_states_out[3]
        # Encoder 2 (Shallow Encoder)
        enc2 = self.encoder2(self.proj_feat(x2))
        
        if self.self_distillation and self.use_feature_maps: 
            enc2_f = self.deep_5_f(enc2)   
            enc2_f = self.avgpool(enc2_f)
        else:
            enc2_f = None
        
        if self.self_distillation:
            # Upsample Encoder 2 (Shallow Encoder)
            out_enc2 = self.deep_4_enc(enc2) 
        else:
            out_enc2 = None

        x3 = hidden_states_out[6]
        # Encoder 3 (Shallow Encoder)
        enc3 = self.encoder3(self.proj_feat(x3))  

        if self.self_distillation and self.use_feature_maps:
            enc3_f = self.deep_4_f(enc3)   
            enc3_f = self.avgpool(enc3_f)   
        else:
            enc3_f = None
        
        if self.self_distillation:
            # Upsample Encoder 3 (Shallow Encoder)
            out_enc3 = self.deep_3_enc(enc3) 
        else:
            out_enc3 = None

        x4 = hidden_states_out[9]
        # Encoder 4 (Deepest Encoder)
        enc4 = self.encoder4(self.proj_feat(x4))

        if self.self_distillation and self.use_feature_maps:
            enc4_f = self.deep_3_f(enc4)   
            enc4_f = self.avgpool(enc4_f)  
        else:
            enc4_f = None     
        
        if self.self_distillation:
            # Upsample Encoder 4 (Deepest Encoder)
            out_enc4 = self.deep_2_enc(enc4) 
        else:
            out_enc4 = None

        #######################################################################################################
        #######################################################################################################
        
        #################################################
        # Decoders and Upsamplers for Self Distillation #
        #################################################

        # Decoder 3 (Shallow Decoder)
        dec4 = self.proj_feat(x)

        if self.self_distillation:
            # Upsample decoder 4
            out_dec4 = self.deep_2_dec(dec4)
        else:
            out_dec4 = None
    
        dec3 = self.decoder5(dec4, enc4) # enc 4 is the skip connection for concatenation

        if self.self_distillation and self.use_feature_maps:
            dec3_f = self.deep_3_f(dec3) 
            dec3_f = self.avgpool(dec3_f)     
        else:
            dec3_f = None  

        if self.self_distillation:
            # Upsample decoder 3 (Shallow Decoder)
            out_dec3 = self.deep_3_dec(dec3)
        else:
            out_dec3 = None

        # Decoder 2 (Shallow Decoder)
        dec2 = self.decoder4(dec3, enc3) # enc 3 is the skip connection for concatenation

        if self.self_distillation and self.use_feature_maps: 
            dec2_f = self.deep_4_f(dec2) 
            dec2_f = self.avgpool(dec2_f)
        else:
            dec2_f = None

        if self.self_distillation:
            # Upsample decoder 2 (Shallow Decoder)
            out_dec2 = self.deep_4_dec(dec2)
        else:
            out_dec2 = None     

        # Decoder 1 (Deepest Decoder)
        dec1 = self.decoder3(dec2, enc2) # enc 2 is the skip connection for concatenation

        if self.self_distillation and self.use_feature_maps:
            dec1_f = self.deep_5_f(dec1)   
            dec1_f = self.avgpool(dec1_f)
        else:
            dec1_f = None      

        if self.self_distillation:
            # Upsample decoder 1 (Deepest Decoder)
            out_dec1 = self.deep_5_dec(dec1)
        else:
            out_dec1 = None
                              
        # Prepare output layers        
        out = self.decoder2(dec1, enc1) # enc 1 is the skip connection for concatenation

        out_main = self.out(out) # Upper classifier
                      
        # For Self Distillation (ONLY during training)
        if self.training and self.self_distillation:
            ## For Self Distillation
            # Encoders:out_enc4: deepest encoder and out_enc3, out_enc2, out_enc1: shallow encoders
            # Decoders:out_dec1: deepest decoder and out_dec2, out_dec3, out_dec4: shallow decoders                        
            
            if self.use_feature_maps:
                ## For Self Distillation with Feature Maps
                # Encoders: enc4_f: deepest encoder and enc3_f, enc2_f: shallow encoders
                # Decoders: dec1_f: deepest decoder and dec3_f, dec2_f: shallow decoders 

                # Full KL Div WITH feature maps
                out = (out_main, out_dec4, out_dec3, out_dec2, out_dec1, out_enc1, out_enc2, out_enc3, out_enc4, enc2_f, enc3_f, enc4_f, dec3_f, dec2_f, dec1_f) 

            else:
                # Full KL Div WITHOUT feature maps - includes both encoders and decoders
                out = (out_main, out_dec4, out_dec3, out_dec2, out_dec1, out_enc1, out_enc2, out_enc3, out_enc4)

                # For Deep Supervision ONLY - currently done for MMWHS CT with UNETR
                # out = (out_main, out_dec3, out_dec2, out_dec1)
            
        elif self.training and not self.self_distillation and not self.use_feature_maps:
            # For Basic UNETR ONLY (NO Self-Distillation)
            out = out_main  
        else:
            # For validation/testing (NO Self-Distillation)
            out = out_main

        return out


if __name__ == '__main__':
    unetr_with_self_distil = SelfDistilUNETR(
        in_channels = 1,
        out_channels = 8,
        feature_size=32,
        self_distillation=True,
        use_feature_maps=False,
        mode=UpsampleMode.DECONV, 
        interp_mode=InterpolateMode.BILINEAR,
        img_size = (96,96,96), 
        )

    ## Count model parameters
    total_params = sum(p.numel() for p in unetr_with_self_distil.parameters() if p.requires_grad)
    print(f'The total number of model parameter is: {total_params}')
    

    x1 = torch.rand((1, 1, 96, 96, 96)) # (B,num_ch,x,y,z)
    print("UNetr input shape: ", x1.shape)

    
    # x3 = unetr_with_self_distil(x1)
    # print("Basic UNetr output shape: ", x3.shape)

    x4 = unetr_with_self_distil(x1)
    print("Self Distil UNetr output shape: ", x4[1].shape)

    # x5 = unetr_with_self_distil(x1)
    # print("Self Distil UNetr output shape: ", x5[14].shape)

