<<<<<<< HEAD
import torch


def L2_dist(pred, truth, y_size):
    return torch.sum((pred - truth) ** 2) / (2 * y_size)
=======
import torch


def L2_dist(pred, truth, y_size):
    return torch.sum((pred - truth) ** 2) / (2 * y_size)
>>>>>>> eece5f7b54bfc10d6fbfa305baff98afcf5a9e43
