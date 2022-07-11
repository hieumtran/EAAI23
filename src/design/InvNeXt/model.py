import torch.nn as nn
import torch.nn.functional as F
from design.InvNeXt.block import Block
from design.InvNeXt.layer_norm import LayerNorm

class InvNet(nn.Module):
    def __init__(self, in_channel, dims, num_per_layers, dropout_rate, inv_kernel):
        super(InvNet, self).__init__()

        self.invnet = []
        self.dims = dims

        # Stem layer
        self.invnet.append(nn.Conv2d(in_channel, dims[0], kernel_size=7, stride=2))
        self.invnet.append(nn.MaxPool2d(2, 2))
        self.invnet.append(LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))

        # Model architecture
        for i in range(len(dims)):
            for _ in range(num_per_layers[i]):
                self.invnet.append(Block(dims[i], inv_kernel, 1, dropout_rate))
                self.invnet.append(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"))
            if i == len(dims)-1:
                break
            self.invnet.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, bias=False))
            self.invnet.append(LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"))
        
        self.invnet.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.invnet = nn.Sequential(*self.invnet)

        self.linear_reg = nn.Linear(dims[-1], 2)
        self.linear_class = nn.Linear(dims[-1], 8)
        self.log_softmax = nn.LogSoftmax(dim=1)
 
    def forward(self, x, mode):
        x = self.invnet(x)
        x = x.view(-1, self.dims[-1])
        if mode == 'reg': return self.linear_reg(x)
        elif mode == 'class': return self.log_softmax(self.linear_class(x))
        elif mode == 'class_reg': return self.linear_class(x), self.linear_reg(x)
        else: raise NotImplementedError


