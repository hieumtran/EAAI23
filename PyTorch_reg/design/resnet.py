import torch
import torch.nn as nn

# The code is gathered from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# Based on the paper https://arxiv.org/pdf/1512.03385.pdf

# Convolution 1x1
def conv1x1(in_channels, out_channels, kernel_size=(1,1), stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# Convolution 3x3
def conv3x3(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

# Bottle Neck implementation
class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleNeck, self).__init__()
        
        # Convolution layer init
        self.conv1 = conv1x1(in_channels, out_channels[0], stride)
        self.conv2 = conv3x3(out_channels[0], out_channels[0], stride)
        self.conv3 = conv1x1(out_channels[0], out_channels[1], stride)

        # Regularization: Batch Normalization
        self.norm_layer1 = nn.BatchNorm2d(out_channels[0])
        self.norm_layer2 = nn.BatchNorm2d(out_channels[1]) 

        # Activation function
        self.relu = nn.ReLU(inplace=True) 
            
    def forward(self, x):
        # Layer 1
        output = self.conv1(x)
        output = self.relu(output)
        output = self.norm_layer1(output)

        # Layer 2
        output = self.conv2(x)
        output = self.relu(output)
        output = self.norm_layer1(output)

        # Layer 3
        output = self.conv3(x)
        output = self.relu(output)
        output = self.norm_layer2(output)

        # Skip Connection
        assert output.shape == x.shape
        output += x

        return output

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNet, self).__init__()

        # Input Convolution layer
        self.input_conv = nn.Conv2d(3, in_channels, (7,7), 2)
        self.relu = nn.ReLU(inplace=True) 
        self.avg_pool = nn.AvgPool2d(2, 2)

        # Bottle Neck usage 
        self.local_bn = BottleNeck(in_channels, out_channels, stride)

        # Linear Layer
        self.linear1 = nn.Linear()
    
    def forward(self, x):
        x = self.input_conv(x)
        
        breakpoint()

        return x
