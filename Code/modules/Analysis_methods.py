#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
from sys import stdout
import itertools

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from random import choice
from scipy.stats import randint
from scipy.stats import uniform
from sklearn.preprocessing import maxabs_scale, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

from Evaluation_metrics import calc_metrics



data_sets = [
    'IH_BA_C18',
    'IH_BA_C8',
    'IH_FA_C18',
    'IH_FA_C8',
    'IH_FC_C8',
    'IH_LLC_C18',
    'IH_LLC_C8',
    'SMRT_75',
    'SMRT_100',
    'SMRT_200',
    'SMRT_275',
    'SMRT_350',
]


# Data to random select compounds for external datasets
comps_rs = {
    'SMRT_75': [75, 21],
    'SMRT_100': [100, 7],
    'SMRT_200': [200, 14],
    'SMRT_275': [275, 52],
    'SMRT_350': [350, 0],
}


# List of Correlation Coefficients to examine
ccs = ['CC1', 'CC96', 'CC90', 'CC80']


cor_thresh = {
    'CC1': 1,
    'CC96': 0.96,
    'CC90': 0.90,
    'CC80': 0.80,
}


# Available metrics
metrics = [
    'MAE',
    'MedAE',
    '%MedAE'
]


metrics_dic = {
    'MAE': 'neg_mean_absolute_error',
    'MedAE': 'neg_median_absolute_error',
    'MSE': 'neg_mean_squared_error',
    'RMSE': 'neg_root_mean_squared_error',
    'R2': 'r2',
}


# Maximizing metrics
max_metr = ['R2', 'Q2', 'Adj_R2']

models_dic = {
    'BayesianRidge': BayesianRidge,
    'SVR_lin': SVR,
    'SVR_nlin': SVR,
    'XGBRegressor': XGBRegressor,
}


models_shortnames_dic = {
    'BayesianRidge': 'BRidgeR',
    'SVR_lin': 'SVRl',
    'SVR_nlin': 'SVRnl',
    'XGBRegressor': 'XGBR',
}


eval_folders = [
    'Dataset Analysis',
]


eval_folders_shortnames_dic = {
    'Dataset Analysis': '',
}



models_params_dic = {
    'LinearRegression': {'normalize': [True,False], 'fit_intercept': [True],},
    'BayesianRidge': {'n_iter': randint(200,800), 'alpha_1': uniform(1e-7,1e-3), 'lambda_1': uniform(1e-7,1e-3),},
    'XGBRegressor': {'n_estimators': randint(10,200), 'max_depth': randint(1,12), 'learning_rate': uniform(0.01,0.25),
                    'gamma': uniform(0.0,10.0), 'reg_alpha': uniform(0.0,10.0), 'reg_lambda': uniform(0.0,10.0), 'objective': ['reg:squarederror'],},
    'SVR_lin': {'C': uniform(0.01,200.0), 'epsilon': uniform(0.01,300.0), 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0],
            'kernel': ["linear"], 'max_iter': [100000000], 'tol': [1e-8],},
    'SVR_nlin': {'C': uniform(0.01,200.0), 'epsilon': uniform(0.01,300.0), 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0],
            'kernel': ["rbf"], 'max_iter': [100000000], 'tol': [1e-8],},
}


def opt_model_params(X, y, model, params, cv=10, scoring='neg_mean_absolute_error', n_iters=100, rs=42, opt_mod='Best'):
    grid = RandomizedSearchCV(model, params, cv=cv, scoring=scoring, verbose=0, n_jobs=10, n_iter=n_iters, refit=False, random_state=rs)
    grid.fit(X, y)
    if(opt_mod=='Best'):
        return(grid.best_params_)
    if(opt_mod=='Random'):
        return(np.random.choice(grid.cv_results_['params']))

        
def analyse_dataset(X, y, model, params, metric='neg_mean_absolute_error', pred_splits=10, opt_cv=10, opt_iters=200, rs=42, print_iters=True, opt_mod='Best', return_modl=False):
    y_preds = list()
    pred_cv = KFold(n_splits=pred_splits, shuffle=True, random_state=rs)
    iter_n = 0
    for train_index, test_index in pred_cv.split(X):
        if(print_iters):
            iter_n += 1
            stdout.write('\rIteration %d of %d' %(iter_n, pred_splits))
            stdout.flush()
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        # Scale data
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        X_tr = pd.DataFrame(X_tr, index=X_train.index)
        X_te = pd.DataFrame(X_te, index=X_test.index)
        opt_params = opt_model_params(X_tr, y_train, model, params, cv=opt_cv, scoring=metric, n_iters=opt_iters, rs=rs, opt_mod='Best')
        cv_mod = model.set_params(**opt_params)
        cv_mod_tr = cv_mod.fit(X_tr, y_train)
        preds = cv_mod_tr.predict(X_te)
        preds = pd.Series(preds.flatten(), index=y_test.index)
        y_preds.append(preds)
    y_preds = pd.concat(y_preds).loc[y.index]
    if(return_modl):
        return(y_preds, cv_mod_tr)
    else:
        return(y_preds)


def save_res_excel(full_path, preds_df, metr_df):
    if(os.path.isfile(full_path)):
        base_name = ('.').join(full_path.split('.')[:-1])
        extens = full_path.split('.')[-1]
        i = 1
        while(os.path.isfile(full_path)):
            full_path = '%s_%d.%s' %(base_name, i, extens)
            i +=  1
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')   
    workbook=writer.book
    sht1 = 'Predictions'
    worksheet=workbook.add_worksheet(sht1)
    writer.sheets[sht1] = worksheet
    preds_df.to_excel(writer, sheet_name=sht1, startrow=0 , startcol=0, merge_cells=True)
    sht2 = 'Metrics'
    worksheet=workbook.add_worksheet(sht2)
    writer.sheets[sht2] = worksheet
    metr_df.to_excel(writer, sheet_name=sht2, startrow=0 , startcol=0, merge_cells=True)
    writer.save()
    

def save_sum_res_excel(full_path, res_dfs, mean_df, std_df, mins):
    writer = pd.ExcelWriter(full_path, engine='xlsxwriter')   
    workbook=writer.book
    sht1 = 'Total_Results'
    worksheet=workbook.add_worksheet(sht1)
    writer.sheets[sht1] = worksheet
    row = 0
    for res_df in res_dfs:
        res_df.to_excel(writer, sheet_name=sht1, startrow=row , startcol=0, merge_cells=True)
        row+=12
    sht2 = 'Mean_metrics_values_per_method'
    worksheet=workbook.add_worksheet(sht2)
    writer.sheets[sht2] = worksheet
    mean_df.to_excel(writer, sheet_name=sht2, startrow=0 , startcol=0, merge_cells=True)
    sht3 = 'St_Devs'
    worksheet=workbook.add_worksheet(sht3)
    writer.sheets[sht3] = worksheet
    std_df.to_excel(writer, sheet_name=sht3, startrow=0 , startcol=0, merge_cells=True)
    sht4 = 'Min_metric_value_per_method'
    worksheet=workbook.add_worksheet(sht4)
    writer.sheets[sht4] = worksheet
    mins.to_excel(writer, sheet_name=sht4, startrow=0 , startcol=0, merge_cells=True)
    writer.save()
    
    
