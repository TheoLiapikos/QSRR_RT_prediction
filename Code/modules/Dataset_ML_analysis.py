#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class Dataset_ML_analysis  ####
# Collects all functions for dataset analysis by selected ML methods

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime


from Analysis_methods import models_dic, models_params_dic, metrics_dic, analyse_dataset,\
                            save_res_excel, ccs
from Evaluation_metrics import calc_metrics


class Dataset_ML_analysis():
    
    dset_n = None
    dset_dic = dict()
    
    def __init__(self, dset_lnk):
        self.dset_n = dset_lnk.split('/')[-1]
        for cc in ccs:
            if (cc == 'CC1'):
                subdir_n = self.dset_n
            else:
                subdir_n = '%s_%s' %(self.dset_n, cc)
            try:
                with open(os.path.join(dset_lnk, subdir_n, 'filt_pproc_datasets.pkl'), 'rb') as handle:
                    X, y = pickle.load(handle)
                    self.dset_dic[subdir_n] = (X,y)
            except:
                print('Subdirectory %s doesn\'t exists' %subdir_n)
                continue
    
    
    def analyse_datasets(self, dset_lnk, dset_store_lnk, cc, ex_mods=list(), metric_n='MAE',
                         pred_splits=10, opt_cv=10, opt_iters=200, n=1):
        subdir_n = self.dset_n if cc == 'CC1' else '%s_%s' %(self.dset_n, cc)
        X,y = self.dset_dic[subdir_n]
        models = [i for i in models_dic.keys() if i not in ex_mods]
        metric = metrics_dic[metric_n]
        for i in range(1,n+1):
            print('\nRound %d of %d' %(i, n))
            met_ls = list()
            pred_ls = list()
            for model_n in models:
                print('\nStarting analysis with method: %s (%s)' %(model_n, datetime.now().strftime("%H:%M")))
                model = models_dic[model_n]()
                params = models_params_dic[model_n]
                ypr = analyse_dataset(X, y, model, params, metric=metric, pred_splits=pred_splits, opt_cv=opt_cv,
                                      opt_iters=opt_iters, rs=None)
                ypr.name = model_n
                pred_ls.append(ypr)
                metr = calc_metrics(y, ypr, X.shape[1])
                metr.name = model_n
                met_ls.append(metr)
            print('\n%s' %datetime.now().strftime("%H:%M"))
            y.name='RT'
            preds_df = pd.concat([y, np.round(pd.concat(pred_ls, axis=1),1)], axis=1)
            metr_df = pd.concat(met_ls, axis=1)
            print(metr_df)
            exl_n = '%s_(%s_%d_%d)_preds_metrics.xlsx' %(subdir_n, metric_n, pred_splits, opt_cv)
            full_path = os.path.join(dset_store_lnk, exl_n)
            save_res_excel(full_path, preds_df, metr_df)

