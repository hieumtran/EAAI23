import torch
import torch.nn as nn
import torch.nn.functional as F

class simpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(simpleNet, self).__init__()

        self.basic_simple_net(input_dim)
        self.linear = nn.Linear(256, output_dim)
        self.dprt = nn.Dropout(0.1)

    def forward(self, x):
        out = self.basic_block(x)
        # out = F.max_pool2d(out, kernel_size=out.size()[2:]) 
        out = self.dprt(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def basic_simple_net(self, input_dim):
        self.basic_block = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Dropout(0.1),

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Dropout(0.1),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Dropout(0.1),

            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Dropout(0.1),

            nn.Conv2d(512, 2048, kernel_size=(1,1), stride=(1,1), padding=(0,0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), dilation=(1,1), ceil_mode=False),
            nn.Dropout(0.1),

            nn.Conv2d(2048, 256, kernel_size=(7,7), stride=(1,1), padding=(1,1)),            
        )




