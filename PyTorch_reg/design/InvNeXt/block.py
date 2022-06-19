import torch
import torch.nn as nn
import torch.nn.functional as F
from design.InvNeXt.involution import involution
from design.InvNeXt.layer_norm import LayerNorm

class Block(nn.Module):
    def __init__(self, dim, kernel_size, stride, dropout_rate):
        super(Block, self).__init__()

        self.inv = involution(dim, kernel_size, stride)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # self.skip_connect = nn.Conv2d(dim, dim, 1, stride=2)

    def forward(self, x):
        identity = x
        x = self.inv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)

        # First GELU-linear layer
        x = self.dropout(self.act(self.pwconv1(x)))

        # Second linear layer
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x += identity # Wrong identity
        return x