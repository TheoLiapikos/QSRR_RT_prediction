#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Basic functions for datasets RAW data collection  ####

import numpy as np
import pandas as pd
import os


from Analysis_methods import data_sets, ccs, metrics, max_metr, models_shortnames_dic, eval_folders, eval_folders_shortnames_dic

        
    
####  MAIN PROCEDURE TO COLLECT RAW DATA
def combine_raw_data(raw_data_lnk):
    tot_dsts_dfs = dict()
    print('RAW data will be collected for the following %d datasets' %len(data_sets))
    print(data_sets)
    for dset_n in data_sets:
        print('\tStarting dataset %s' %dset_n)
        tot_ccs_dfs = dict()
        for cc_n in ccs:
            inter_metrs = list()
            inter_preds = list()
            raw_d_fn = os.path.join(raw_data_lnk, dset_n, cc_n)
            for main_meth_n in eval_folders:
                main_meth_shrt_n = eval_folders_shortnames_dic[main_meth_n]
                tot_pred_dfs = list()
                tot_metr_dfs = list()
                tot_fns = os.listdir(os.path.join(raw_d_fn, main_meth_n))
                tot_pred_dfs = [pd.read_excel(os.path.join(raw_d_fn, main_meth_n, fn), sheet_name='Predictions', index_col=0) for fn in tot_fns if fn.endswith('.xlsx')]
                tot_metr_dfs = [pd.read_excel(os.path.join(raw_d_fn, main_meth_n, fn), sheet_name='Metrics', index_col=0) for fn in tot_fns if fn.endswith('.xlsx')]
                for i, pred_df in enumerate(tot_pred_dfs):
                    metr_df = tot_metr_dfs[i]
                    for method in metr_df.columns:
                        if(method not in models_shortnames_dic):
                            continue
                        metrs = metr_df.loc[:,method]
                        preds = pred_df.loc[:,method]
                        main_sub = '%s%s' %(main_meth_shrt_n, models_shortnames_dic[method])
                        main_sub_ser = pd.Series([main_sub, i+1], index=['Method', 'Iter'])
                        metrs_f = pd.concat([main_sub_ser, metrs])
                        preds_f = pd.concat([main_sub_ser, preds])
                        inter_metrs.append(metrs_f)
                        inter_preds.append(preds_f)

            tot_ccs_dfs[cc_n] = dict()
            metrs_df_f = pd.concat(inter_metrs, axis=1).T
            metrs_df_f = metrs_df_f.set_index(['Method', 'Iter'])
            metrs_df_f.sort_values(['Method', 'Iter'], inplace=True)
            tot_ccs_dfs[cc_n]['Metrics'] = metrs_df_f
            preds_df_f = pd.concat(inter_preds, axis=1).T
            preds_df_f = preds_df_f.set_index(['Method', 'Iter'])
            preds_df_f.sort_values(['Method', 'Iter'], inplace=True)
            exp_rt_head = pd.Series(['Exp_RT', 0], index=['Method', 'Iter'])
            exp_rt_df = pd.DataFrame(pd.concat([exp_rt_head, tot_pred_dfs[0]['RT']])).T.set_index(['Method', 'Iter'])
            tot_ccs_dfs[cc_n]['Predictions'] = pd.concat([exp_rt_df, preds_df_f])
        tot_dsts_dfs[dset_n] = tot_ccs_dfs
    tot_dsts_dfs = comp_normMedAE(tot_dsts_dfs)
    return(tot_dsts_dfs)
    
    
# Compute %MedAE for all predictions
def comp_normMedAE(tot_dsts_dfs):
    for dset in data_sets:
        for cc in ccs:
            pred_df = tot_dsts_dfs[dset][cc]['Predictions']
            metr_df = tot_dsts_dfs[dset][cc]['Metrics']
            exp_rt = pred_df.loc[('Exp_RT',0),:]
            methods = set([i for i in metr_df.index.levels[0]])
            meth_res = list()
            for method in methods:
                metd_pred_idx = [i for i in pred_df.index if method in i]
                meth_preds = pred_df.loc[metd_pred_idx,:]
                pc_normAE = 100*abs(meth_preds - exp_rt)/exp_rt
                pc_normAE_med = pc_normAE.median(axis=1)
                pc_normAE_med.name  = '%MedAE'
                meth_res.append(pc_normAE_med)
            meths_metric_vals = pd.concat(meth_res, axis=0)
            tot_dsts_dfs[dset][cc]['Metrics'] = pd.concat([metr_df, meths_metric_vals], axis=1)
        tot_dsts_dfs[dset] = best_perf_per_metric(tot_dsts_dfs[dset])
    return(tot_dsts_dfs)
    
    
## Procedure to find Top N Best Performers (BPs) for each different Metric used
def best_perf_per_metric(raw_data_dic, bf_n=4):
    for cc_n in ccs:
        cc_dic = raw_data_dic[cc_n]
        cc_metrs = cc_dic['Metrics']
        tot_metr_res = list()
        for metric_n in metrics:
            tps_lst = list()
            tps_idx = list()
            metr_vals = cc_metrs.loc[:, metric_n]
            asc = False if metric_n in max_metr else True
            metr_vals_srt = metr_vals.sort_values(ascending=asc)
            bp_idxs = metr_vals_srt.index
            i = 0
            while(len(tps_idx) < bf_n):
                bp_idx = bp_idxs[i]
                if(bp_idx[0] in tps_lst):
                    i += 1
                    continue
                else:
                    tps_idx.append(bp_idx)
                    tps_lst.append(bp_idx[0])
                    i += 1
            bps_data = list()
            for i in np.arange(bf_n):
                cur_bp_data = list()
                cur_bp_idx = tps_idx[i]
                cur_bp_n = cur_bp_idx[0]
                cur_bp_data.append([i+1, 'Method', cur_bp_n])
                cur_bp_iter = cur_bp_idx[1]
                cur_bp_data.append([i+1, 'Iter', cur_bp_iter])
                cur_bp_val = cc_metrs.loc[cur_bp_idx,metric_n]
                cur_bp_data.append([i+1, 'Value', cur_bp_val])

                cur_bp_df = pd.DataFrame(cur_bp_data, columns=['Rank', 'Infos', metric_n])
                bps_data.append(cur_bp_df)
            cur_met_df = pd.concat(bps_data)
            cur_met_df.set_index(['Rank', 'Infos'], inplace=True)
            tot_metr_res.append(cur_met_df)
        cc_conc_df = pd.concat(tot_metr_res, axis=1)
        raw_data_dic[cc_n]['Best_Performers_per_Metric'] = cc_conc_df
    return(raw_data_dic)
        
    
