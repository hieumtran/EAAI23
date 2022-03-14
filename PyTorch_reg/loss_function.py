import torch


def L2_dist(pred, truth, y_size):
    return torch.sum((pred - truth) ** 2) / (2 * y_size)

def RMSE(pred, truth):
    val = torch.sum((pred[:, 0].float() - truth[:, 0].float())**2)
    ars = torch.sum((pred[:, 1].float() - truth[:, 1].float()) ** 2)
    return val, ars
