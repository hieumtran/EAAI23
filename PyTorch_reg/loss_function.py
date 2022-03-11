import torch


def L2_dist(pred, truth, y_size):
    return torch.sum((pred - truth) ** 2) / (2 * y_size)
