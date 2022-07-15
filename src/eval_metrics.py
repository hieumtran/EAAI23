import scipy.stats as scst
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score

# Regression 
def std_compute(arr):
    return np.sqrt(np.var(arr))

def rmse(pred, truth):
    return np.sqrt(((pred[:, 0] - truth[:, 0]) **2).mean()), np.sqrt(((pred[:, 1] - truth[:, 1]) **2).mean())

def pear(pred, truth):
    # Sample CC
    # return scst.pearsonr(pred[:, 0], truth[:, 0])[0], scst.pearsonr(pred[:, 1], truth[:, 1])[0]
    # Population CC 
    val_cov = np.cov(pred[:, 0], truth[:, 0])[0,1]
    pop_val_rho = val_cov / (std_compute(pred[:, 0])*std_compute(truth[:, 0]))
    ars_cov = np.cov(pred[:, 1], truth[:, 1])[0,1]
    pop_ars_rho = ars_cov / (std_compute(pred[:, 1])*std_compute(truth[:, 1]))
    return pop_val_rho, pop_ars_rho
    
def ccc(pred, truth):
    # mean
    (val_mean_pred, val_mean_truth) = (np.mean(pred[:, 0]), np.mean(truth[:, 0]))
    (ars_mean_pred, ars_mean_truth) = (np.mean(pred[:, 1]), np.mean(truth[:, 1]))
    # std
    (val_std_pred, val_std_truth) = (std_compute(pred[:, 0]), std_compute(truth[:, 0]))
    (ars_std_pred, ars_std_truth) = (std_compute(pred[:, 1]), std_compute(truth[:, 1]))
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
    
    return np.sum(val_pred == val_truth) / val_truth.shape[0], np.sum(ars_pred == ars_truth) / ars_truth.shape[0] 


# Classification
def accuracy(pred, truth):
    return np.count_nonzero(pred == truth) / truth.shape[0]

def f1_score_func(pred, truth):
    return f1_score(truth, pred, average='weighted')

def cks(pred, truth):
    return cohen_kappa_score(truth, pred)

# TODO: Krippendorf Alpha, ICC, area under ROC curve (AUC), and area under Precision-Call (AUC-PR)

def rmse_1D(pred, truth):
    return np.sqrt(((pred[:, 0] - truth[:, 0]) **2).mean())

def pear_1D(pred, truth):
    # Sample CC
    # return scst.pearsonr(pred[:, 0], truth[:, 0])[0], scst.pearsonr(pred[:, 1], truth[:, 1])[0]
    # Population CC 
    cov = np.cov(pred[:, 0], truth[:, 0])[0,1]
    pop_rho = cov / (std_compute(pred[:, 0])*std_compute(truth[:, 0]))
    return pop_rho
    
def ccc_1D(pred, truth):
    # mean
    (mean_pred, mean_truth) = (np.mean(pred[:, 0]), np.mean(truth[:, 0]))
    # std
    (std_pred, std_truth) = (std_compute(pred[:, 0]), std_compute(truth[:, 0]))
    # pearson correlation
    pear_value = pear_1D(pred, truth)
    # computation
    ccc = (2 * pear_value * std_pred * std_truth) / \
              (std_pred**2 + std_truth**2 + (mean_pred - mean_truth)**2)
    return ccc

def sagr_1D(pred, truth):
    (pred, truth) = (np.sign(pred[:, 0]), np.sign(truth[:, 0])) 
    
    return np.sum(pred == truth) / truth.shape[0]
