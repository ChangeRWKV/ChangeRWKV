import torch
import torch.nn as nn
from model.VRWKV import VRWKV_encoder, ConvNormAct
from model.STFM import STFM

class ChangeRWKV(nn.Module):
    def __init__(self, img_size=256, num_classes=1, size="base"):
        super(ChangeRWKV, self).__init__()
        if size == "base":
            self.embed_dims = [48, 72, 144, 240]
            self.encoder = VRWKV_encoder(img_size=img_size, depths=[3, 3, 6, 3], embed_dims=self.embed_dims,
                                         exp_ratios=[2., 2.5, 4.0, 4.0], drop_path=0.05)
        elif size == "small":
            self.embed_dims = [32, 64, 128, 192]
            self.encoder = VRWKV_encoder(img_size=img_size, depths=[3, 3, 6, 3], embed_dims=self.embed_dims,
                                         exp_ratios=[2., 2.5, 3.0, 4.0], drop_path=0.05)
        elif size == "tiny":
            self.embed_dims = [32, 48, 96, 160]
            self.encoder = VRWKV_encoder(img_size=img_size, depths=[2, 2, 4, 2], embed_dims=self.embed_dims,
                                         exp_ratios=[2., 2.5, 3.0, 3.5], drop_path=0.05)
        else:
            raise NotImplementedError(f"Unimplemented model size: {size}")
        
        self.fusion_module = STFM(img_size, self.embed_dims)

        self.decoder1 = UpBlock(self.embed_dims[3], self.embed_dims[2])
        self.decoder2 =  UpBlock(self.embed_dims[2]*2, self.embed_dims[1])
        self.decoder3 =  UpBlock(self.embed_dims[1]*2, self.embed_dims[0])
        self.decoder4 =  UpBlock(self.embed_dims[0]*2, 24)
        self.final_conv = nn.Conv2d(24, num_classes, kernel_size=1)

    def forward(self, x1, x2):
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        # Encoder
        x, enc3, enc2, enc1 = self.encoder.forward_features(torch.cat([x1, x2], dim=0))
        # Fusion
        x, enc3, enc2, enc1 = self.fusion_module([x, enc3, enc2, enc1])
        # Decoder
        dec3 = self.decoder1(x)
        dec2 = self.decoder2(torch.cat([dec3, enc3], dim=1))
        dec1 = self.decoder3(torch.cat([dec2, enc2], dim=1))
        dec0 = self.decoder4(torch.cat([dec1, enc1], dim=1))
        # Final output
        out = self.final_conv(dec0)
        return torch.sigmoid(out)


class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, drop=0.):
        super().__init__()
        self.conv = ConvNormAct(dim_in, dim_in, kernel_size=1)
        self.conv_local = ConvNormAct(dim_in, dim_in, kernel_size=9, groups=dim_in, act_layer='silu')

        self.proj = ConvNormAct(dim_in, dim_out, kernel_size=1)
        self.proj_drop = nn.Dropout(drop)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_local(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x
