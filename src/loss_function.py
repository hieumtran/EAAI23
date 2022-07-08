import torch
import torch.nn as nn

def entropy_loss(pred, truth):
    criterion = nn.CrossEntropyLoss()
    return criterion(pred, truth)

def L2_dist(pred, truth):
    return torch.sum((pred - truth) ** 2) / (2 * truth.size(0))

    





