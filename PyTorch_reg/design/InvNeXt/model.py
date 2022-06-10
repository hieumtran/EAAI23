import torch
import torch.nn as nn
import torch.nn.functional as F
from design.InvNeXt.involution import involution
from design.InvNeXt.block import Block
from design.InvNeXt.layer_norm import LayerNorm

class InvNet(nn.Module):
    def __init__(self, in_channel, dims, num_per_layers, drp_rate):
        super(InvNet, self).__init__()

        self.invnet = []
        self.dims = dims

        # Stem layer
        self.invnet.append(nn.Conv2d(in_channel, out_channels=dims[0], kernel_size=7, stride=2))
        self.invnet.append(nn.MaxPool2d(2, 2))
        self.invnet.append(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))

        # Model architecture
        for i in range(len(dims)):
            for j in range(num_per_layers[i]):
                self.invnet.append(Block(dims[i], 7, 1, drp_rate))
                self.invnet.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))
            if i == len(dims)-1:
                break
            self.invnet.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2))
            self.invnet.append(LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"))
        
        # self.invnet.append(nn.Conv2d(dims[-2], dims[-1], kernel_size=2, stride=2))
        # self.invnet.append(LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"))
        self.invnet.append(nn.AvgPool2d(4, 4))
        self.invnet = nn.Sequential(*self.invnet)

        self.linear = nn.Linear(dims[-1], 2)
        # self.linear = nn.Linear(dims[-1], 8)
        

    def forward(self, x):
        x = self.invnet(x)
        x = x.view(-1, self.dims[-1])
        return self.linear(x)


