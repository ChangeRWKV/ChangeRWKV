import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from model.VRWKV import VRWKV_ChannelMix


class STFM(nn.Module):
    def __init__(self, img_size, embed_dims):
        super(STFM, self).__init__()
        self.SFM = SFM([embed_dims[2], embed_dims[1], embed_dims[0]], embed_dims[0], img_size // 2)
        self.TFM = nn.ModuleList([
            TFM(embed_dims[0], depth=3),
            TFM(embed_dims[1], depth=3),
            TFM(embed_dims[2], depth=3),
            TFM(embed_dims[3], depth=3),
        ])

    def forward(self, feature_list):
        x, enc3, enc2, enc1 = feature_list
        b = x.shape[0] // 2
        enc3, enc2, enc1 = self.SFM([enc3, enc2, enc1])
        enc1 = self.TFM[0](enc1[:b, :, :, :], enc1[b:, :, :, :])
        enc2 = self.TFM[1](enc2[:b, :, :, :], enc2[b:, :, :, :])
        enc3 = self.TFM[2](enc3[:b, :, :, :], enc3[b:, :, :, :])
        x = self.TFM[3](x[:b, :, :, :], x[b:, :, :, :])
        return x, enc3, enc2, enc1


class SFM(nn.Module):
    def __init__(self, in_dims, target_dim, target_size):
        super(SFM, self).__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = nn.LayerNorm(target_dim * 3)
        self.drop_path = DropPath(0.05)
        self.channel = VRWKV_ChannelMix(n_embd=target_dim * 3, channel_gamma=1 / 4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes = [target_size // 4, target_size // 2, target_size]

    def forward(self, features):
        # Projection
        upsampled_features = []
        output_features = []
        for i, feature in enumerate(features):
            feature = F.interpolate(feature, size=self.target_size, mode='bilinear', align_corners=False)
            feature = self.projections[i](feature)
            upsampled_features.append(feature)

        # Fusion
        concatenated = torch.cat(upsampled_features, dim=1)
        B, C, H, W = concatenated.shape
        concatenated = concatenated.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        attn_output = concatenated + self.drop_path(
            self.ln1(self.channel(concatenated, (self.target_size, self.target_size))))

        B, n_patch, hidden = attn_output.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.contiguous().view(B, hidden, h, w)

        # Restore to hierarchical feature map
        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            split_feature = self.final_projections[i](split_feature)
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i], mode='bilinear',
                                          align_corners=False)
            output_features.append(split_feature)
        return output_features


class TFM(nn.Module):
    def __init__(self, dim, depth=1):
        super().__init__()
        self.layers = nn.Sequential(*[TFMBlock(dim) for _ in range(depth)])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return x1 + x2


class TFMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1 = Conv(dim, dim, 3, 1, 1)
        self.conv21 = nn.Sequential(ConvRelu(dim, dim, 1, 1, 0), Conv(dim, dim, 1, 1, 0))
        self.conv22 = nn.Sequential(ConvRelu(dim, dim, 1, 1, 0), Conv(dim, dim, 1, 1, 0))
        self.conv31 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))
        self.conv32 = nn.Sequential(ConvRelu(2, 16, 3, 1, 1), Conv(16, 1, 3, 1, 1))
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)
        c1 = torch.sigmoid(self.conv21(F.adaptive_avg_pool2d(x1, output_size=(1, 1))) + self.conv21(
            F.adaptive_max_pool2d(x1, output_size=(1, 1))))
        c2 = torch.sigmoid(self.conv22(F.adaptive_avg_pool2d(x2, output_size=(1, 1))) + self.conv22(
            F.adaptive_max_pool2d(x2, output_size=(1, 1))))
        x1 = x1 * c2
        x2 = x2 * c1
        s1 = torch.sigmoid(
            self.conv31(torch.cat([torch.mean(x1, dim=1, keepdim=True), torch.max(x1, dim=1, keepdim=True)[0]], dim=1)))
        s2 = torch.sigmoid(
            self.conv32(torch.cat([torch.mean(x2, dim=1, keepdim=True), torch.max(x2, dim=1, keepdim=True)[0]], dim=1)))
        x1 = self.norm1(x1 * s2)
        x2 = self.norm2(x2 * s1)
        return x1, x2


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
