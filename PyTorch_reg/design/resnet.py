import torch
import torch.nn as nn


# The code is gathered from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# Based on the paper https://arxiv.org/pdf/1512.03385.pdf


# Bottle Neck implementation
class BottleNeck(nn.Module):
    def __init__(self, input_dim, output_dim, upsample, stride, padding):
        super(BottleNeck, self).__init__()

        # Convolution layer init
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3,
                               stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim*4, kernel_size=1)

        # Regularization: Batch Normalization
        self.norm_layer1 = nn.BatchNorm2d(output_dim)

        # Up sampling
        self.upsample = upsample
        self.upsampling = nn.Sequential(
            nn.Conv2d(input_dim, output_dim * 4, 1, stride),
            nn.BatchNorm2d(output_dim * 4)
        )

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Layer 1
        output = self.conv1(x)
        output = self.norm_layer1(output)
        output = self.relu(output)

        # Layer 2
        output = self.conv2(output)
        output = self.norm_layer1(output)
        output = self.relu(output)

        # Layer 3
        output = self.conv3(output)

        # Up sampling
        if self.upsample:
            x = self.upsampling(x)

        # Skip Connection
        output += x
        output = self.relu(output)

        return output


class ResNet(nn.Module):
    def __init__(self, res_learning):
        super(ResNet, self).__init__()

        # Initial variables
        self.expansion = 4
        self.res_learning = res_learning

        # Input Convolution layer
        self.input_conv = nn.Conv2d(in_channels=3, out_channels=64,
                                    kernel_size=(7, 7), stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), padding=1, stride=2)

        # Construct ResNet block: BasicBlock or BottleNeck
        # Layer 1
        self.conv1_x = BottleNeck(input_dim=64, output_dim=64, upsample=True, stride=1, padding=1)
        self.conv1_x2 = BottleNeck(input_dim=64 * self.expansion, output_dim=64, upsample=False, stride=1, padding=1)

        # Layer 2
        self.conv2_x = BottleNeck(input_dim=64 * self.expansion, output_dim=128, upsample=True, stride=2, padding=1)
        self.conv2_x2 = BottleNeck(input_dim=128 * self.expansion, output_dim=128, upsample=False, stride=1, padding=1)

        # Layer 3
        self.conv3_x = BottleNeck(input_dim=128 * self.expansion, output_dim=256, upsample=True, stride=2, padding=1)
        self.conv3_x2 = BottleNeck(input_dim=256 * self.expansion, output_dim=256, upsample=False, stride=1, padding=1)

        # Layer 4
        self.conv4_x = BottleNeck(input_dim=256 * self.expansion, output_dim=512, upsample=True, stride=2, padding=1)
        self.conv4_x2 = BottleNeck(input_dim=512 * self.expansion, output_dim=512, upsample=False, stride=1, padding=1)

        # Linear Layer
        self.linear1 = nn.Linear(2048 * 7 * 7, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 2)

        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.input_conv(x)
        x = self.relu(x)
        x = self.bn(x)

        # Conv1_x
        x = self.max_pool(x)
        x = self.conv1_x(x)
        for _ in range(self.res_learning[0] - 1):
            x = self.conv1_x2(x)

        # Conv2_x
        x = self.conv2_x(x)
        for _ in range(self.res_learning[1] - 1):
            x = self.conv2_x2(x)

        # Conv3_x
        x = self.conv3_x(x)
        for _ in range(self.res_learning[2] - 1):
            x = self.conv3_x2(x)

        # Conv4_x
        x = self.conv4_x(x)
        for _ in range(self.res_learning[3] - 1):
            x = self.conv4_x2(x)

        x = self.avgpool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear Layer
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)
