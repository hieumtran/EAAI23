import torch
import scipy.stats as scst
import numpy as np

def L2_dist(pred, truth, y_size):
    print(truth.shape)
    return torch.sum((pred - truth) ** 2) / (2 * y_size)


def rmse(pred, truth):
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(pred[:, 0], truth[:, 0])), torch.sqrt(mse(pred[:, 1], truth[:, 1]))


def pear(pred, truth):
    return scst.pearsonr(pred[:, 0], truth[:, 0])[1,0], scst.pearsonr(pred[:, 1], truth[:, 1])[1,0]


def ccc(pred, truth):
    # mean
    (val_mean_pred, val_mean_truth) = (torch.mean(pred[:, 0]), torch.mean(truth[:, 0]))
    (ars_mean_pred, ars_mean_truth) = (torch.mean(pred[:, 1]), torch.mean(truth[:, 1]))
    # std
    (val_std_pred, val_std_truth) = (torch.var(pred[:, 0]), torch.var(truth[:, 0]))
    (ars_std_pred, ars_std_truth) = (torch.var(pred[:, 1]), torch.var(truth[:, 1]))
    # pearson correlation
    val_pear, ars_pear = pear(pred, truth)
    # computation
    val_ccc = (2 * val_pear * val_std_pred * val_std_truth) / \
              (val_std_pred**2 + val_std_truth**2 + (val_mean_pred - val_mean_truth)**2)
    ars_ccc = (2 * ars_pear * ars_std_pred * ars_std_truth) / \
              (ars_std_pred**2 + ars_std_truth**2 + (ars_mean_pred - ars_mean_truth)**2)
    return val_ccc, ars_ccc

def sagr(pred, truth):
    





