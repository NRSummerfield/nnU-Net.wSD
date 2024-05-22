from monai.networks.nets.unetr import UNETR
from torch import nn
import torch

class DoubleNET(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            img_size=(96, 96, 64),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12):
        super().__init__()
        
        self.UNETR_1 = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads
        )

        self.UNETR_2 = UNETR(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads
        )


    def forward(self, x_in):
        pred = self.UNETR_1.forward(x_in)
        # print(torch.cat((x_in, pred), dim=1).shape)
        out = self.UNETR_2.forward(torch.cat((x_in, pred), dim=1))
        return out
