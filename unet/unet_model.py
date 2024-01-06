import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn

from .unet_parts import *

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]

class UNet(nn.Module):
    def __init__(self, marginal_prob_std, n_channels, n_classes, embed_dim = 256):
        super(UNet, self).__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.dense_in = Dense(embed_dim, 64)
        
        self.down1 = Down(64, 128)
        self.dense_d1 = Dense(embed_dim, 128)
        
        self.down2 = Down(128, 256)
        self.dense_d2 = Dense(embed_dim, 256)
        
        self.down3 = Down(256, 512)
        self.dense_d3 = Dense(embed_dim, 512)
        
        self.down4 = Down(512, 512)
        self.dense_d4 = Dense(embed_dim, 512)
        
        self.up1 = Up(1024, 256)
        self.dense_u1 = Dense(embed_dim, 256)
        
        self.up2 = Up(512, 128)
        self.dense_u2 = Dense(embed_dim, 128)
        
        self.up3 = Up(256, 64)
        self.dense_u3 = Dense(embed_dim, 64)
        
        self.up4 = Up(128, 64)
        self.dense_u4 = Dense(embed_dim, 64)
        
        self.outc = OutConv(64, n_classes)
        
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        
    def forward(self, x, t):
        embed = self.act(self.embed(t))
        
        x1 = self.inc(x)
        x1 += self.dense_in(embed)
        
        x2 = self.down1(x1)
        x2 += self.dense_d1(embed)
        
        x3 = self.down2(x2)
        x3 += self.dense_d2(embed)
        
        x4 = self.down3(x3)
        x4 += self.dense_d3(embed)
        
        x5 = self.down4(x4)
        x5 += self.dense_d4(embed)
        
        x = self.up1(x5, x4)
        x += self.dense_u1(embed)
        
        x = self.up2(x, x3)
        x += self.dense_u2(embed)
        
        x = self.up3(x, x2)
        x += self.dense_u3(embed)
        
        x = self.up4(x, x1)
        x += self.dense_u4(embed)
        
        x = self.outc(x)
        
        x = x / self.marginal_prob_std(t)[:, None, None, None]
        return x