#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import random
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold


from Analysis_methods import data_sets, ccs, cor_thresh


#########################################
#####  FILTERING PROCEDURE
#########################################


### Univariate Filtering Methods

#### 0. Remove samples having depended variable == 0
def remove_y0_samples(X_t, y_t):
    zero_idxs = y_t[y_t==0].index
    X_t.drop(zero_idxs, inplace=True)
    y_t.drop(zero_idxs, inplace=True)
    return(X_t, y_t)


#### 1. Constant Features Removal
def remove_const_feats(X_t):
    const_filter = VarianceThreshold(threshold=0)
    const_filter.fit(X_t)
    const_filter.get_support()
    X_t = X_t.loc[:, const_filter.get_support()]
    return(X_t)


#### 2. Quasi (nearly constant-low variance) Features Removal
def remove_quasi_feats(X_t, quasi_thrs=0.01):
    const_filter2 = VarianceThreshold(threshold=quasi_thrs)
    const_filter2.fit(X_t)
    const_filter2.get_support()
    X_t = X_t.loc[:, const_filter2.get_support()]
    return(X_t)


#### 3. Duplicate Features Removal
def remove_dupl_feats(X_t):
    X_t_T = X_t.T
    dubl_feats = X_t_T.duplicated(keep='first')
    X_t_T = X_t_T.loc[~dubl_feats,:]
    X_t = X_t_T.T
    return(X_t)


#### 4. Duplicate Samples Removal
def remove_dupl_samples(X_t, y_t):
    dubl_samples = X_t.duplicated(keep='first')
    X_t = X_t.loc[~dubl_samples,:]
    y_t = y_t.loc[~dubl_samples]
    return(X_t, y_t)


#### 5. Remove features with a specific proportion of missing values
def remove_miss_data_feats(X_t, md_thrsh = 0.1):
    pr_thrs = md_thrsh
    mv_n = int(X_t.shape[0]*md_thrsh)
    m_cols = X_t.columns[(X_t.isnull().sum()>mv_n).values]
    X_t.drop(m_cols, axis=1, inplace=True)
    return (X_t)


#### 6. Uncorrelated to the Dependent variable Features Removal
def remove_y_uncor_feats(X_t, y_t, un_cor_thrsh=0.1):
    cor_coefs = []
    for feature in X_t.columns:
        feat1 = X_t.loc[:,feature]
        cor_coefs.append(np.corrcoef(feat1, y_t)[0][1])
    X_t = X_t.loc[:, np.array(np.abs(cor_coefs))>un_cor_thrsh]
    return(X_t)


### Multivariate Filtering Methods
#### 1. Correlated Features Removal
def LR_feat_importance(X_t, y_t):
    X_t.fillna(0, inplace=True)
    lreg = LinearRegression()
    lreg.fit(X_t, y_t)
    features = X_t.columns
    coefs = pd.Series(np.abs(lreg.coef_), index=features).sort_values(ascending=False)
    return(coefs)


# Function to calculate correlation of a list of independent variables (dataframe)
# and the corresponding dependent variable
def feat_corr_to_dep(X_t, y_t):
    cor_lst = list()
    features = X_t.columns
    for feat in features:
        cor_lst.append(X_t.loc[:,feat].corr(y_t))
    corrs = pd.Series(cor_lst, index=features).sort_values(ascending=False)
    return(corrs)


# Pairwise approach
# Check each possible pair of features, and calculate their importance
# Set the least important feature to remove
def remove_cor_feats(X_t, y_t, mul_cor_thrsh=0.85, method='y_cor'):
    feats_ign = set()
    feat_list = X_t.columns
    for i in range(len(feat_list)):
        feat1 = X_t.iloc[:,i]
        for j in range(i):
            feat2 = X_t.iloc[:,j]
            corr_coef = feat1.corr(feat2)
            if(abs(corr_coef)>=mul_cor_thrsh):
                cor_df = X_t.loc[:,[feat_list[i], feat_list[j]]]
                if(method=='y_cor'):
                    feats_imp = feat_corr_to_dep(cor_df, y_t)
                if(method=='lr_coef'):
                    feats_imp = LR_feat_importance(cor_df, y_t)
                feats_ign.add(feats_imp.keys()[1])
    X_t = X_t.drop(feats_ign, axis=1)
    return(X_t)


# Function to check duplicate samples names with different features (no duplicates)
def check_dupl_samples_names(Xs, ys):
    z = list(Xs.index)
    if(len(set(z)) < len(z)):
        for i in list(set(z)):
            z.remove(i)
        print('*** CAUTION!!! THERE ARE THE FOLLOWING DUPLICATES IN SAMPLES NAMING... ***')
        print(z)
        Xs = Xs.reset_index().drop_duplicates(subset='ID', keep='first').set_index('ID')
        ys = ys.reset_index().drop_duplicates(subset='ID', keep='first').set_index('ID')
        print('***REMOVED SAMPLES WITH DUPLICATE NAMES....***')
    return(Xs, ys.squeeze())
        

####  Main function to call all individual functions
def filter_dataset(X_in, y_in, q_thrs=0.01, md_thrs=0.1, un_cor_thrs=0.1, mul_cor_thrs=0.85, method='y_cor', print_msg=False):
    Xs = X_in.copy()
    Xs = pd.DataFrame(Xs)
    ys = y_in.copy()
    if(print_msg):
        print('Running dependent variable == 0 Samples Removal...')
    Xs, ys = remove_y0_samples(Xs, ys)
    if(print_msg):
        print('Running Constant Features Removal...')
    Xs = remove_const_feats(Xs)
    if(print_msg):
        print('Running Quasi Features Removal...')
    Xs = remove_quasi_feats(Xs, quasi_thrs=q_thrs)
    if(print_msg):
        print('Running Duplicate Features Removal...')
    Xs = remove_dupl_feats(Xs)
    if(print_msg):
        print('Running Duplicate Samples Removal...')
    Xs, ys = remove_dupl_samples(Xs, ys)
    if(print_msg):
        print('Running Features with many missing values Removal...')
    Xs = remove_miss_data_feats(Xs, md_thrsh=md_thrs)
    if(print_msg):
        print('Running Uncorrelated to dependent variable Feature Removal...')
    Xs = remove_y_uncor_feats(Xs, ys, un_cor_thrsh=un_cor_thrs)
    if(mul_cor_thrs<1):
        if(print_msg):
            print('Running Correlated Features Removal...')
        Xs = remove_cor_feats(Xs, ys, mul_cor_thrsh=mul_cor_thrs, method=method)
    if(print_msg):
        print('Checking for duplicates in samples naming...')
    Xs, ys = check_dupl_samples_names(Xs, ys)
    return(Xs, ys)


###  Apply filters an all datasets and create the various configurations
def create_dsets_configs(dsets_dir_lnk, method='y_cor'):
    dset_dirs = [d for d in os.listdir(dsets_dir_lnk) if os.path.isdir(os.path.join(dsets_dir_lnk, d))]
    act_dset_dirs = [d for d in dset_dirs if d in data_sets]
    for dset in act_dset_dirs:
        print("\nStarting preprocess procedure for dataset '%s'" %dset)
        dset_lnk = os.path.join(dsets_dir_lnk, dset)
        Xs = pd.read_csv(os.path.join(dset_lnk, 'X_input_data.csv'), index_col=0)
        ys = pd.read_csv(os.path.join(dset_lnk, 'y_input_data.csv'), index_col=0).squeeze()
        for cc in ccs:
            cc_thresh = cor_thresh[cc]
            print("\tStarting filtration procedure with Cor Coef threshold '%s'" %cc_thresh)
            X_filt, y_filt = filter_dataset(Xs, ys, mul_cor_thrs=cc_thresh, method=method)
            dset_conf_dir_n = dset if cc=='CC1' else '%s_%s' %(dset, cc)
            dset_conf_dir_lnk = os.path.join(dset_lnk, dset_conf_dir_n)
            if not os.path.exists(dset_conf_dir_lnk):
                os.makedirs(dset_conf_dir_lnk)
            X_filt.to_csv(os.path.join(dset_conf_dir_lnk, 'X_input_data.csv'))
            y_filt.to_csv(os.path.join(dset_conf_dir_lnk, 'y_input_data.csv'))
            with open(os.path.join(dset_conf_dir_lnk, 'filt_pproc_datasets.pkl'), 'wb') as handle:
                pickle.dump([X_filt, y_filt], handle)


