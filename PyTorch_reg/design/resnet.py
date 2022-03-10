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

        # Activation function
        self.relu = nn.ReLU(inplace=True) 
            
    def forward(self, x):
        # Layer 1
        output = self.conv1(x)
        output = self.relu(output)
        output = self.norm_layer1(output)

        # Layer 2
        output = self.conv2(output)
        output = self.relu(output)
        output = self.norm_layer1(output)

        # Layer 3
        output = self.conv3(output)
        

        # Skip Connection
        while(output.shape != x.shape):
            zero_pad = torch.zeros_like(x)
            x = torch.concat([x, zero_pad], axis=1)
        output += x
        output = self.relu(output)

        return output

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNet, self).__init__()

        # Input Convolution layer
        self.input_conv = nn.Conv2d(3, in_channels, (7,7), 2)
        self.relu = nn.ReLU(inplace=True) 
        self.avg_pool = nn.AvgPool2d(2, 2)

        # Bottle Neck usage 
        self.local_bn1 = BottleNeck(in_channels, out_channels, stride)
        # self.local_bn2 = BottleNeck(in_channels[1], out_channels[1], stride)
        # self.local_bn3 = BottleNeck(in_channels[2], out_channels[2], stride)

        # Linear Layer
        self.linear1 = nn.Linear(256 * 54 * 54, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = self.relu(self.input_conv(x))
        x = self.avg_pool(x)

        # BottleNeck
        x = self.local_bn1(x)
        # x = self.local_bn2(x)
        # x = self.local_bn3(x)
        
        # Flatten
        x = x.view(x.size(0),-1)
        
        # Linear Layer
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)
