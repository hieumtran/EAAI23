import torch
import scipy.stats as scst
import numpy as np

def L2_dist(pred, truth):
    return torch.sum((pred - truth) ** 2) / (2 * truth.size(0))


def rmse(pred, truth):
    mse = torch.nn.MSELoss()
    return torch.sqrt(mse(pred[:, 0], truth[:, 0])), torch.sqrt(mse(pred[:, 1], truth[:, 1]))


def pear(pred, truth):
    return scst.pearsonr(pred[:, 0], truth[:, 0])[0], scst.pearsonr(pred[:, 1], truth[:, 1])[0]


def ccc(pred, truth):
    # mean
    (val_mean_pred, val_mean_truth) = (torch.mean(pred[:, 0]), torch.mean(truth[:, 0]))
    (ars_mean_pred, ars_mean_truth) = (torch.mean(pred[:, 1]), torch.mean(truth[:, 1]))
    # std
    (val_std_pred, val_std_truth) = (torch.sqrt(torch.var(pred[:, 0])), torch.sqrt(torch.var(truth[:, 0])))
    (ars_std_pred, ars_std_truth) = (torch.sqrt(torch.var(pred[:, 1])), torch.sqrt(torch.var(truth[:, 1])))
    # pearson correlation
    val_pear, ars_pear = pear(pred, truth)
    # computation
    val_ccc = (2 * val_pear * val_std_pred * val_std_truth) / \
              (val_std_pred**2 + val_std_truth**2 + (val_mean_pred - val_mean_truth)**2)
    ars_ccc = (2 * ars_pear * ars_std_pred * ars_std_truth) / (ars_std_pred**2 + ars_std_truth**2 + (ars_mean_pred - ars_mean_truth)**2)
    return val_ccc, ars_ccc

def sagr(pred, truth):
    (val_pred, ars_pred) = (np.sign(pred[:, 0]), np.sign(pred[:, 1])) 
    (val_truth, ars_truth) = (np.sign(truth[:, 0]), np.sign(truth[:, 1])) 
    
    return torch.sum(val_pred == val_truth) / val_truth.size(0), torch.sum(ars_pred == ars_truth) / ars_truth.size(0) 

    





