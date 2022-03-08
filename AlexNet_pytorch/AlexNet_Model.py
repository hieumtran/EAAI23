from torch import nn
import torch
from Separation import Regression, Classification


class AlexNet_Reg(nn.Module):

    def __init__(self):
        """
        Args:
            regression (boolean): When True the model returns valence/arousal \
                regression result. When False returns classification result
        """

        super(AlexNet_Reg, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 9, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(3, 32, 7, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(3, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(3, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(3, 128, 3, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=True),


            nn.Linear(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=True),

            nn.Linear(1024, 2)
        )

    def forward(self, input):
        shared_output = self.net(input)
        return shared_output

    # can I separate the shared_output into 2 samples -> feed 1 through regression
    #   and feed the other one through classification.


class AlexNet_Class(nn.Module):

    def __init__(self):
        """
        Args:
            regression (boolean): When True the model returns valence/arousal \
                regression result. When False returns classification result
        """

        super(AlexNet_Class, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 9),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(16, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(32, 64, 5),  # , 5, 1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(64, 128, 3),  # , 3, 1
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Conv2d(128, 128, 3),  # , 3, 1
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(128, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=True),


            nn.Linear(1024, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=True),

            nn.Linear(1024, 10)
        )

    def forward(self, input):
        shared_output = self.net(input)
        return shared_output

    # can I separate the shared_output into 2 samples -> feed 1 through regression
    #   and feed the other one through classification.
