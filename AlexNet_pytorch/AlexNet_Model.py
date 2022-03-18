from torch import nn
import torch


class AlexNet_Reg(nn.Module):

    def __init__(self):

        super(AlexNet_Reg, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),

            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),

            nn.Linear(4096, 2)
        )

    def forward(self, input):
        conv = self.net(input)
        conv = conv.view(conv.shape[0], -1)
        output = self.linear(conv)
        return output

    # can I separate the shared_output into 2 samples -> feed 1 through regression
    #   and feed the other one through classification.


class AlexNet_Class(nn.Module):

    def __init__(self):

        super(AlexNet_Class, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), 

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=256*7*7, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),


            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),

            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        conv = self.net(input)
        conv = conv.view(conv.shape[0], -1)
        output = self.linear(conv)
        return output

    # can I separate the shared_output into 2 samples -> feed 1 through regression
    #   and feed the other one through classification.
