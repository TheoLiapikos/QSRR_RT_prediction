#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import statsmodels.api as sm

import matplotlib.pyplot as plt

# Calc R2 (R2)
# Equivalent to sklearn.metrics.r2_score()
def calc_r2(y, y_pred):
    r2 = 1 - (np.sum([(yi-y_pi)**2 for yi,y_pi in zip(y,y_pred)]) / np.sum([(yi-np.mean(y))**2 for yi,y_pi in zip(y,y_pred)]))
    return(r2)

def calc_adj_r2(y, y_pred, k):
    r2 = calc_r2(y, y_pred)
    n = len(y)
    adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
    return(adj_r2)

# Explained variance (corresponds to Q2)
# Equivalent to sklearn.metrics.explained_variance_score
def calc_expl_var(y, y_pred):
    ev = 1 - (np.var([(yi-y_pi) for yi,y_pi in zip(y,y_pred)]) / np.var(y))
    return(ev)

# Calc Mean Absolute Error of Prediction (MAE)
# Equivalent to sklearn.metrics.mean_absolute_error
def calc_mae(y, y_pred):
    mae = np.sum([np.abs(yi-y_pi) for yi,y_pi in zip(y,y_pred)])/len(y)
    return(mae)

# Calc Median Absolute Error of Prediction (MedAE)
# Equivalent to sklearn.metrics.median_absolute_error
def calc_medae(y, y_pred):
    medae = np.median([abs(i-j) for i,j in zip(y,y_pred)])
    return(medae)

# Calc % Mean Absolute Error of Prediction (%MAE)
def calc_percent_mae(y, y_pred):
    pmae = np.sum([np.abs(yi-y_pi)/yi for yi,y_pi in zip(y,y_pred)])/len(y)
    return(pmae)

# Calc Mean Squared Error of Prediction (MSE)
# Equivalent to sklearn.metrics.mean_squared_error()
def calc_mse(y, y_pred):
    mse = np.sum([(yi-y_pi)**2 for yi,y_pi in zip(y,y_pred)])/len(y)
    return(mse)

# Calc Root Mean Squared Error of Prediction (RMSE)
def calc_rmse(y, y_pred):
    rmse = np.sqrt(np.sum([(yi-y_pi)**2 for yi,y_pi in zip(y,y_pred)])/len(y))
    return(rmse)

# Calc % Root Mean Squared Error of Prediction (%RMSE)
def calc_percent_rmse(y, y_pred):
    prmse = np.sqrt(np.sum([((yi-y_pi)/yi)**2 for yi,y_pi in zip(y,y_pred)])/len(y)) * 100
    return(prmse)

def calc_metrics(y, y_pred, feats_n):
    avail_metrics = {
        'R2': calc_r2,
        'Adj_R2': calc_adj_r2,
        'Q2': calc_expl_var,
        'MAE': calc_mae,
        'MedAE': calc_medae,
        'PMAE': calc_percent_mae,
        'MSE': calc_mse,
        'RMSE': calc_rmse,
        'PRMSE': calc_percent_rmse,
    }
    y = y.squeeze()
    y_pred = y_pred.squeeze()
    metr_ls = list()
    metr_val_ls = list()
    for metric in avail_metrics.keys():
        try:
            if(metric=='Adj_R2'):
                metr_v = avail_metrics[metric](y, y_pred, feats_n)
            else:
                metr_v = avail_metrics[metric](y, y_pred)
        except:
            metr_v = 'Error'
        metr_ls.append(metric)
        metr_val_ls.append(metr_v)
    metr_series = pd.Series(metr_val_ls, index=metr_ls)
    return(metr_series)



