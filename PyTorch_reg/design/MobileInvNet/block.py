import torch.nn as nn
from design.MobileInvNet.involution import involution

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, inv_kernel_size, inv_stride):
        super(Block, self).__init__()

        # Convolution layer
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1), # Conv1
            nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.05, affine=True),
            nn.Tanh(),

            involution(output_dim, inv_kernel_size, inv_stride), # Involution layer
            nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.05, affine=True),
            nn.Tanh(),

            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=1), #C Conv2
            nn.BatchNorm2d(output_dim, eps=1e-05, momentum=0.05, affine=True)
        )

        # Identity layer
        self.identity = nn.Identity()
        self.use_skip = inv_stride == 1 and input_dim == output_dim
        

    def forward(self, x):
        identity = self.identity(x)
        x = self.block(x)
        if self.use_skip: x += identity # Skip connection
        return x