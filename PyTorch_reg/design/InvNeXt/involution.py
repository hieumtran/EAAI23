import torch.nn as nn

class involution(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels

        # Convolution No.1
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True)
        self.act_func1 = nn.ReLU(inplace=True)

        # Convolution No.2
        self.conv2 = nn.Conv2d(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size**2 * self.groups,
            kernel_size=1,
            stride=1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, 1)

    def forward(self, x):
        if self.stride != 1: x = self.avgpool(x) 
        weight = self.conv2(self.conv1(x))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out