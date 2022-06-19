import torch
import torch.nn as nn
import torch.nn.functional as F
from design.MobileInvNet.block import Block

class MobileInvNet(nn.Module):
    def __init__(self, input_channel, final_channel, block_setting, inv_kernel_size):
        super(MobileInvNet, self).__init__()

        self.mobile_involution = []
        self.mobile_involution.append(nn.Conv2d(3, input_channel, kernel_size=1, stride=2))
        for c, n, s in block_setting:
            output_channel = c
            for i in range(n):
                if i == 0: stride = s
                else: stride = 1
                self.mobile_involution.append(Block(input_channel, output_channel, inv_kernel_size, stride))
                input_channel = output_channel
        self.mobile_involution.append(nn.Conv2d(input_channel, final_channel, kernel_size=1, stride=1))
        self.mobile_involution.append(nn.AvgPool2d(7, 1))
        
        self.mobile_involution = nn.Sequential(*self.mobile_involution)
        self.linear = nn.Linear(final_channel, 2)

        self.final_channel = final_channel

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.mobile_involution(x)
        x = torch.reshape(x, (-1, self.final_channel))
        return self.linear(x)
    

