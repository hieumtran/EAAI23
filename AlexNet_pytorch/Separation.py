from torch import nn


class Regression(nn.Module):

    def __init__(self):

        super(Regression, self).__init__()
        self.regression = nn.Linear(1024, 2)

    def forward(self, input):
        return self.regression(input)


class Classification(nn.Module):

    def __init__(self):
        super(Classification, self).__init__()
        self.classification = nn.Linear(1024, 10)

    def forward(self, input):
        return self.classification(input)
