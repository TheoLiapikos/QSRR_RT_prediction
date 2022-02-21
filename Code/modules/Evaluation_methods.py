#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Basic functions to compute analytical rankings for all methods and for all datasets  ####

import numpy as np
import pandas as pd
import os
import pickle
from scipy import stats
from scipy.stats import kruskal, ttest_ind, f_oneway
from scikit_posthocs import posthoc_dunn
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm


from Analysis_methods import data_sets, ccs, metrics, max_metr, models_dic, models_shortnames_dic

basic_meths_order = ['BRidgeR', 'SVRl', 'SVRnl', 'XGBR']


datasets_colors_dic = {
    'IH_BA_C18': '#808000',#'#9acd32',
    'IH_BA_C8': '#8b008b',
    'IH_FA_C18': '#ff4500',
    'IH_FA_C8': '#1e90ff',#'#00ced1',
    'IH_FC_C8': '#ffd700',
    'IH_LLC_C18': '#dc143c',#'#00ff00',
    'IH_LLC_C8': '#f4a460',#'#00fa9a',
    'SMRT_75': '#00fa9a',
    'SMRT_100': '#ee82ee',
    'SMRT_200': '#00ff00',
    'SMRT_275': '#00ced1',
    'SMRT_350': '#9acd32',
}


# Method to turn values of a Dataframe into rankings. Set which columns to examine
# Rankings are added to the dataframe as new columns
def df_cols_to_ranks(data_df, cols_to_ranks=list(), ascending=True):
    in_df = data_df.copy()
    if(len(cols_to_ranks)==0):
        print('No columns selected to turn into rankings...')
    else:
        for col in cols_to_ranks:
            in_df.sort_values(by=col, inplace=True, ascending=ascending)
            in_df['%s_rankings' %col] = np.arange(1,in_df.shape[0]+1)
    return(in_df)


# Function to compute analytical rankings of all methods on all datasets
def compute_analytical_rankings(raw_data_dic):
    dsets_dic = dict()
    for dset_n in data_sets:
        dset_rd_dic = raw_data_dic[dset_n]
        dsets_dic[dset_n] = dict()
        for cc_n in ccs:
            dsets_dic[dset_n][cc_n] = dict()
            cc_data_dic = dset_rd_dic[cc_n]
            cc_metrics_df = cc_data_dic['Metrics']
            anal_methods = cc_metrics_df.index.levels[0]
            cc_metrics_df = cc_metrics_df.astype(float)
            cc_metrics_df_mean = cc_metrics_df.groupby(level=[0]).mean().loc[anal_methods,:]
            cc_metrics_df_std = cc_metrics_df.groupby(level=[0]).std().loc[anal_methods,:]
            cc_metrics_df_median = cc_metrics_df.groupby(level=[0]).median().loc[anal_methods,:]
            cc_metrics_df_min = cc_metrics_df.groupby(level=[0]).min().loc[anal_methods,:]
            cc_metrics_df_max = cc_metrics_df.groupby(level=[0]).max().loc[anal_methods,:]
            tot_metr_lst = list()
            for metric_n in metrics:
                metr_lst = list()
                mn = cc_metrics_df_mean.loc[:,metric_n]
                mn.name = 'Mean'
                metr_lst.append(mn)
                std = cc_metrics_df_std.loc[:,metric_n]
                std.name = 'StD'
                metr_lst.append(std)
                mdn = cc_metrics_df_median.loc[:,metric_n]
                mdn.name = 'Median'
                metr_lst.append(mdn)
                opt = cc_metrics_df_max.loc[:,metric_n] if metric_n in max_metr else cc_metrics_df_min.loc[:,metric_n]
                opt.name = 'Max' if metric_n in max_metr else 'Min'
                metr_lst.append(opt)
                conc_metr_df = pd.concat(metr_lst, axis=1)
                asc = False if metric_n in max_metr else True
                cols = ['Mean', 'Median', 'Max'] if metric_n in max_metr else ['Mean', 'Median', 'Min'] 
                conc_metr_df_rnk = df_cols_to_ranks(conc_metr_df, cols_to_ranks=cols, ascending=asc)
                conc_metr_df_rnk.columns=pd.MultiIndex.from_arrays([[metric_n]*conc_metr_df_rnk.columns.shape[0], conc_metr_df_rnk.columns])
                tot_metr_lst.append(conc_metr_df_rnk)
            cc_conc_df = pd.concat(tot_metr_lst, axis=1)
            dsets_dic[dset_n][cc_n]['Analytical_Rankings'] = cc_conc_df
    return(dsets_dic)


#---------   Functions about getting the Best Performing Predictive Models for datasets  ---------#

# Get best models predictions for all datasets for all conditions (metric, CC, ranking type)
def get_total_best_models_and_predictions(raw_data_dic, anal_ranks_dic, rank_type='Median'):
    best_models_dic = dict()
    for dset in data_sets:
        best_models_dic[dset] = dict()
        for cc in ccs:
            best_models_dic[dset][cc] = dict()
            cc_res = list()
            for metric in metrics:
                opt_type = 'max' if metric in max_metr else 'min'
                cc_metr_preds = get_dset_best_model_and_predictions(dset, raw_data_dic, anal_ranks_dic, metric, cc, rank_type, opt_type)
                conds = pd.Series([metric, cc_metr_preds.name[0], cc_metr_preds.name[1]], index=['Metric', 'Best_Model', 'Iter'])
                cc_res.append(pd.concat([conds, cc_metr_preds]))
            cc_df = pd.concat(cc_res, axis=1).T
            cc_df.set_index(['Metric', 'Best_Model', 'Iter'], inplace=True)
            best_models_dic[dset][cc] = np.round(cc_df.astype(float), 1)
    return(best_models_dic)


# Get predictions for best prediction model for specific dataset, metric, CC and ranking type
def get_dset_best_model_and_predictions(dset, raw_data_dic, anal_ranks_dic, metric, cc, rank_type, opt_type):
    best_model_id = get_best_model_for_dataset(dset, anal_ranks_dic, metric, cc, rank_type)
    best_model_iter = get_model_best_iter(dset, best_model_id, raw_data_dic, metric, cc, opt_type)
    best_model_preds = get_model_predictions(dset, best_model_id, raw_data_dic, cc, best_model_iter)
    return(best_model_preds)


# Best performing prediction model for specific dataset, metric, CC and ranking type
def get_best_model_for_dataset(dset, anal_ranks_dic, metric, cc, rank_type):
    ds_cc_rnks_df = anal_ranks_dic[dset][cc]['Analytical_Rankings']
    ranks = ds_cc_rnks_df.loc[:, (metric, '%s_rankings'%rank_type)]
    best_model_id = ranks[ranks==1].index[0]
    if('MLR' in best_model_id):
        best_model_id = ranks[ranks==2].index[0]
    return(best_model_id)


# Best iteration of a prediction model for specific dataset, metric, CC
def get_model_best_iter(dset, model_id, raw_data_dic, metric, cc, opt_type):
    ds_cc_metrs_df = raw_data_dic[dset][cc]['Metrics']
    metr_vals = ds_cc_metrs_df.loc[(model_id,), metric]
    opt_val = metr_vals.max() if opt_type=='max' else metr_vals.min()
    best_model_iter = metr_vals[metr_vals==opt_val].index[0]
    return(best_model_iter)


# Predictions of a prediction model for specific dataset, CC, iteration
def get_model_predictions(dset, model_id, raw_data_dic, cc, iter_n):
    ds_cc_preds_df = raw_data_dic[dset][cc]['Predictions']
    metr_vals = ds_cc_preds_df.loc[(model_id,iter_n),:]
    return(metr_vals)



#------  Functions for Regression and Residuals plots  ------#

# Function to collect all data needed for Regression and Residuals plots
def compute_regr_resid_plots_data(datasets_dic, best_models_and_predictions_dic):
    plots_data_dic = {i:{j:{k:dict() for k in data_sets} for j in metrics} for i in ccs}
    for cc in ccs:
        for metric in metrics:
            for dset in data_sets:
                exp_rt = datasets_dic[dset]['Exp_RT']
                df = best_models_and_predictions_dic[dset][cc]
                idx = [i for i in df.index if metric in i][0]
                _, best_model, _ = idx
                pred_rt = df.loc[idx,:]
                # Residuals  *** ALWAYS Predicted-Experimental ***
                residuals = pred_rt - exp_rt
                tot_data = pd.Series([best_model, pred_rt, exp_rt, residuals],
                                     index=['Best_model', 'Predicted_RT', 'Experimental_RT', 'Residuals'],
                                     name=dset)
                plots_data_dic[cc][metric][dset] = tot_data
    return(plots_data_dic)


# Function that creates various types of Regression and/or Residuals plots for a set of
# datasets for various CC values and metrics
def regr_resid_plots(data_to_plot, cols_n=4, col_width=6, rr_plot_type=1, show_plot=False):
    plots_dic = {i:{j:None for j in metrics} for i in ccs}
    for cc in ccs:
        for metric in metrics:
            dsets_dic = data_to_plot[cc][metric]
            if(rr_plot_type==1):
                dsets_fig = regression_plots(data_dic=dsets_dic, cols_n=cols_n, col_width=col_width, show_plot=show_plot)
            if(rr_plot_type==2):
                dsets_fig = residuals_plots(data_dic=dsets_dic, cols_n=cols_n, col_width=col_width, show_plot=show_plot)
            if(rr_plot_type==3):
                dsets_fig = complex_residuals_plots(data_dic=dsets_dic, cols_n=cols_n, col_width=col_width, show_plot=show_plot)
            plots_dic[cc][metric] = dsets_fig
    return(plots_dic)


# Returns a single Regression plot for each one of a list of datasets, stored in a dictionary
def regression_plots(data_dic, cols_n=4, col_width=6, show_plot=False):
    row_height = col_width
    rows_n = int(np.ceil(len(data_dic)/cols_n))
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(cols_n*col_width, rows_n*row_height))

    for i, dset in enumerate(data_dic.keys()):
        color = datasets_colors_dic[dset]
        ds_dic = data_dic[dset]
        exp_rt = ds_dic['Experimental_RT']
        pred_rt = ds_dic['Predicted_RT']
        bm_n = ds_dic['Best_model']

        ax = axs.flatten()[i]
        ax.scatter(pred_rt, exp_rt, s=6*col_width, color=color)
        max_scale = np.ceil(np.max([ax.get_xlim()[1], ax.get_ylim()[1]]))
        ax.set_ylim(0, max_scale)
        ax.set_xlim(0, max_scale)
        ax.plot([0,max_scale],[0,max_scale], linewidth=2)
        ax.set_xlabel('Predicted $\it{t}_{R}$ (s)', fontsize=4*col_width-4)
        ax.set_ylabel('Experimental $\it{t}_{R}$ (s)', fontsize=4*col_width-5)
        ax.tick_params(axis='both', labelsize=3*col_width-4)
        ax.set_title(dset, fontsize=4*col_width-1, fontweight='bold')
        ax.text(0.025*max_scale, 0.925*max_scale, 'Best model: %s' %bm_n,
                color='blue', fontsize=3*col_width-1)
    for j in np.arange(i+1, rows_n*cols_n):
        ax = axs.flatten()[j]
        ax.set_axis_off()
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


# Returns a single Residuals plot for each one of a list of datasets, stored in a dictionary
def residuals_plots(data_dic, cols_n=4, col_width=6, show_plot=False):
    row_height = col_width
    rows_n = int(np.ceil(len(data_dic)/cols_n))
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(cols_n*col_width, rows_n*row_height))

    for i, dset in enumerate(data_dic.keys()):
        color = datasets_colors_dic[dset]
        ds_dic = data_dic[dset]
        pred_rt = ds_dic['Predicted_RT']
        res_rt = ds_dic['Residuals']
        bm_n = ds_dic['Best_model']

        ax = axs.flatten()[i]
        ax.scatter(pred_rt, res_rt, s=6*col_width, color=color)
        max_yscale = np.ceil(np.max([ax.get_ylim()[0], ax.get_ylim()[1]]))
        ax.set_ylim(-1.5*max_yscale, 1.5*max_yscale)
        if(pred_rt.min()<0):
            ax.set_xlim(0,)
        ax.axhline(linewidth=2)
        ax.set_ylabel('Residuals (s)', fontsize=4*col_width-4)
        ax.set_xlabel('Predicted $\it{t}_{R}$ (s)', fontsize=4*col_width-4)
        ax.tick_params(axis='both', labelsize=3*col_width-4)
        # Plot's title
        ax.set_title(dset, fontsize=4*col_width-1, fontweight='bold')
        ax.text(0.05, 0.925, 'Best model: %s' %bm_n, transform=ax.transAxes,
                color='blue', fontsize=3*col_width-1)
    for j in np.arange(i+1, rows_n*cols_n):
        ax = axs.flatten()[j]
        ax.set_axis_off()
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


# Creates custom QQ plot using scipy methods to calculate necessary elements
def qq_plot(in_data, line=True, dist='norm', color='r', axis=None):
    if(not isinstance(in_data, list)):
        in_data = list(in_data)
    fig_data = stats.probplot(in_data, dist=dist)
    x, y = fig_data[0]
    slope, inter, r = fig_data[1]
    axis.scatter(x, y, color=color)
    if(line):
        x1 = x[0]
        y1 = slope*x1+inter
        x2 = x[-1]
        y2 = slope*x2+inter
        axis.plot([x1,x2], [y1,y2], linewidth=3)
    return(axis)


# Creates a custom triple Residuals-Histogram-QQ plot for a dataset
def residuals_hist_qq_plot(dset, res_rt, pred_rt, best_meth, width, height, sampl_col='r'):
    fig, axs = plt.subplots(1, 3, figsize=(width,height), gridspec_kw={'width_ratios': [6, 1.5, 2.5]}, sharey=True)
    ax1 = axs[0]
    ax1.scatter(pred_rt, res_rt, s=3.6*width, color=sampl_col)
    max_yscale = np.ceil(np.max([ax1.get_ylim()[0], ax1.get_ylim()[1]]))
    ax1.set_ylim(ax1.get_ylim()[0], 1.2*ax1.get_ylim()[1])
    if(pred_rt.min()<0):
        ax1.set_xlim(0,)
    ax1.axhline(linewidth=2)
    ax1.set_ylabel('Residuals (s)', fontsize=2.5*width)
    ax1.set_xlabel('Predicted $\it{t}_{R}$ (s)', fontsize=2.5*width)
    ax1.tick_params(axis='both', labelsize=2.5*width-4)
    ax1.text(0.05, 0.925, 'Best model: %s' %best_meth, transform=ax1.transAxes,
             color='blue', fontsize=1.8*width)
    ax2 = axs[1]
    ax2.hist(res_rt, bins='auto', orientation='horizontal', color=sampl_col)
    ax2.axhline(linewidth=2)
    ax2.set_xlabel('Distribution', fontsize=1.8*width)
    ax3 = axs[2]
    qq_plot(res_rt, color=sampl_col, axis=ax3)
    ax3.set_xlabel('Theor. Quantiles', fontsize=1.8*width)
    plt.suptitle(dset, fontsize=3*width, fontweight='bold') #15
    plt.tight_layout()
    plt.close()
    return(fig)


# Returns a custom triple Residuals-Histogram-QQ plot for each one of a list of datasets,
# stored in a dictionary
def complex_residuals_plots(data_dic, cols_n=3, col_width=8, show_plot=False):
    row_height = 0.6*col_width
    rows_n = int(np.ceil(len(data_dic)/cols_n))
    fig, axs = plt.subplots(rows_n, cols_n, figsize=(cols_n*col_width, rows_n*row_height))
    for i, dset in enumerate(data_dic.keys()):
        color = datasets_colors_dic[dset]
        ds_dic = data_dic[dset]
        pred_rt = ds_dic['Predicted_RT']
        res_rt = ds_dic['Residuals']
        best_meth = ds_dic['Best_model']
        triple_fig = residuals_hist_qq_plot(dset, res_rt, pred_rt, best_meth, width=col_width, height=row_height, sampl_col=color)
        triple_fig.savefig('temp_plot.tiff', dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
        image = plt.imread('temp_plot.tiff')
        axs.flatten()[i].imshow(image)
        axs.flatten()[i].axis('off')
        plt.tight_layout()
    for j in np.arange(i+1, rows_n*cols_n):
        ax = axs.flatten()[j]
        ax.set_axis_off()
    try:
        os.remove('temp_plot.tiff')
    except:
        pass
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


#------  Functions for computation of performance matrices and corresponding plots  ------#

def comp_models_perform_matrices(raw_dic):
    methods = [models_shortnames_dic[i] for i in models_dic.keys()]
    tot_res_dic = dict()
    for metric in metrics:
        for cc in ccs:
            mc_inter_res = list()
            for dset in data_sets:
                dset_res = list()
                metr_vals = raw_dic[dset][cc]['Metrics']
                for method in methods:
                    dset_res.append(pd.Series(metr_vals.loc[(method,),metric].median(), name=method, index=[dset]))
                mc_inter_res.append(pd.concat(dset_res, axis=1))
            tot_res_dic['%s_%s' %(metric,cc)] = pd.concat(mc_inter_res, axis=0)
    return(tot_res_dic)


# Method to plot multi-group bar plots
def multi_group_bar_plot(data_dic, width, height, show_plot=False):
    plots_dic = dict()
    for key in data_dic.keys():
        data_df = data_dic[key]
        ngroups = data_df.shape[1]
        xpoints = data_df.shape[0]
        metric = key.split('_')[0]
        fig, ax = plt.subplots(figsize=(width, height))
        index = np.arange(xpoints)
        bar_width = 0.01+1/(ngroups+1)
        opacity = 0.8
        colors = cm.tab10(np.linspace(0, 1, 8), alpha=1)
        for i, col in enumerate(data_df.iloc[:,:ngroups].columns):
            xs = index + i*bar_width
            ys = data_df.loc[:,col]
            ax.bar(xs, ys, width=bar_width, color=colors[i], edgecolor='black', alpha=opacity, label=col)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlim(-0.25, xpoints-0.15)
        y_label = '%s (s)' %metric if metric in ['MAE', 'MedAE'] else '%s' %metric
        ax.set_ylabel(y_label, size=17, fontweight='bold')
        tick_labels = data_df.index.tolist()
        plt.xticks(index+(ngroups-1)*bar_width/2, tick_labels, size=13.5, fontweight='bold')#, rotation=90)
        plt.legend(fontsize=13)
        plt.tight_layout()
        if(not show_plot):
            plt.close()
        plots_dic[key] = fig
    return(plots_dic)


#------  Functions for Statistical Analysis and plots  ------#

def compute_statistical_data(rawdata_dic):
    st_data_dic = {i:dict() for i in data_sets}
    models = list(models_shortnames_dic.values())
    for dset in data_sets:
        for method in models:
            for metric in metrics:
                metr_meth_list = list()
                for cc in ccs:
                    metr_meth_list.append(pd.Series(rawdata_dic[dset][cc]['Metrics'].loc[(method,), metric], name=cc).sort_values()[::-1].reset_index(drop=True) )
                st_data_dic[dset]['%s_%s'%(method, metric)] = pd.concat(metr_meth_list, axis=1)
    return(st_data_dic)


### Perform Kruskal-Wallis test
# Returns only the computed p-values
def kw_test(metr_val_dic, dropna=False, fillna='median'):
    kw_res_dic = {i:dict() for i in data_sets}
    models = list(models_shortnames_dic.values())
    for dset in data_sets:
        for method in models:
            for metric in metrics:
                data_df = metr_val_dic[dset]['%s_%s'%(method, metric)]
                if(dropna):
                    data_df = data_df.dropna().reset_index(drop=True)
                else:
                    data_df = data_df.fillna(data_df.median()) if(fillna=='median') else data_df.fillna(data_df.mean())
                data_list = [data_df[i] for i in data_df.columns]
                stat, p = kruskal(*data_list)
                kw_res_dic[dset]['%s_%s'%(method, metric)] = p
    return(kw_res_dic)


# Plot Kruskal_Wallis test results (p-values) as a heatmap plot and store it on disk
def kw_heatmap_plot(kw_pvals_dic, a=0.05, show_plot=False):
    fig, ax = plt.subplots(figsize=(12,8))
    kw_df = pd.DataFrame.from_dict(kw_pvals_dic).T
    sns.heatmap(kw_df, vmin=0, vmax=a, cmap='coolwarm_r', ax=ax,
                linewidths=1.5, linecolor='white',
                annot=True, annot_kws={"size": 10},
                cbar_kws={'label': 'p-values'})
    ax.figure.axes[-1].yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticklabels([('\n').join(i.split('_')) for i in kw_df.columns])
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


def bonf_post_hoc_test(k_w_dic, metr_val_dic, a=0.01, p_adjust='bonferroni'):
    tot_res_lst = list()
    keys1 = k_w_dic.keys()
    for key1 in keys1:
        keys2 = k_w_dic[key1].keys()
        for key2 in keys2:
            if(k_w_dic[key1][key2]<a):
                data = [metr_val_dic[key1][key2][cc] for cc in ccs]
                pvals = posthoc_dunn(data, p_adjust=p_adjust)
                pvals=pvals.iloc[0,1:]
                pvals.index=ccs[1:]
                pvals.name='%s_%s' %(key1,key2)
                if(any([i<a for i in pvals])):
                    tot_res_lst.append(pvals)
    return(pd.concat(tot_res_lst, axis=1).T)


def bonf_test_heatmap_plot(bof_pvals_df, a=0.01, show_plot=False):
    fig, ax = plt.subplots(figsize=(9,12))
    sns.heatmap(bof_pvals_df, vmin=0, vmax=a, cmap='coolwarm_r', ax=ax,
                linewidths=1.5, linecolor='white',
                annot=True, annot_kws={"size": 10},
                cbar_kws={'label': 'p-values, Bonferroni adjusted', 'aspect':40})
    ax.figure.axes[-1].yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xticklabels(['Removal of HCFs\n(CC=0.%s)' %i.replace('CC','') for i in bof_pvals_df.columns])
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


def highlight_kw_hm(kw_pvals_dic, bon_df, a=0.01, show_plot=False):
    kw_df = pd.DataFrame.from_dict(kw_pvals_dic).T
    fig, ax = plt.subplots(figsize=(12,8))
    g = sns.heatmap(kw_df, vmin=0, vmax=a, cmap='coolwarm_r', ax=ax,
                linewidths=1.5, linecolor='white',
                annot=True, annot_kws={"size": 10},
                cbar_kws={'label': 'p-values'})
    from matplotlib.patches import Rectangle
    for i,col in enumerate(kw_df.columns):
        for j,row in enumerate(kw_df.index):
            if('%s_%s' %(row, col) in bon_df.index):
                ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='#9acd32', lw=4)) # #00ff00, #9acd32
    ax.figure.axes[-1].yaxis.label.set_size(12)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticklabels([('\n').join(i.split('_')) for i in kw_df.columns])
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)






### Perform independent t-test
# Returns only the computed p-values
def ind_ttest(metr_val_dic, dropna=False, fillna='median'):
    ttind_res_dic = {i:dict() for i in data_sets}
    other_ccs = ['CC96', 'CC90', 'CC80']
    models = list(models_shortnames_dic.values())
    for dset in data_sets:
        for metric in metrics:
            metr_lst = list()
            for method in models:
                method_lst = list()
                data_df = metr_val_dic[dset]['%s_%s'%(method, metric)]
                if(dropna):
                    data_df = data_df.dropna().reset_index(drop=True)
                else:
                    data_df = data_df.fillna(data_df.median()) if(fillna=='median') else data_df.fillna(data_df.mean())
                for cc in other_ccs:
                    ccx_lst = data_df[cc]
                    stv, pv = ttest_ind(data_df['CC1'], data_df[cc])
                    method_lst.append(pv)
                metr_lst.append(pd.Series(method_lst, index=other_ccs, name=method))
            ttind_res_dic[dset][metric]=pd.concat(metr_lst, axis=1).T
    return(ttind_res_dic)



def ttest_heatmap_plot(tt_pvals_dic, cols_n=3, col_width=8, a=0.05, show_plot=False):
    row_height = col_width/3
    rows_n = len(tt_pvals_dic)
    fig, axs = plt.subplots(rows_n, cols_n, sharey=True, figsize=(cols_n*col_width, rows_n*row_height))

    for i, dset in enumerate(tt_pvals_dic.keys()):
        for j, metric in enumerate(metrics):
            ax = axs[i,j]
            df = tt_pvals_dic[dset][metric]
            cbar = True if j==2 else False
            sns.heatmap(df, vmin=0, vmax=a, cmap='coolwarm_r', ax=ax,
                        linewidths=1.5, linecolor='white',
                        annot=True, annot_kws={"size": 8+col_width/2},
                        cbar=cbar, cbar_kws={'label': 'p-value', 'aspect':7.5})
            ax.figure.axes[-1].yaxis.label.set_size(8+col_width/2)
            ax.tick_params(axis='both', labelsize=8+col_width/2)
            if(j==0):
                ax.set_ylabel(dset, fontsize=10+col_width/2, weight='bold')
            ax.set_title(metric, fontsize=10+col_width/2, weight='bold')
    plt.tight_layout()
    if(not show_plot):
        plt.close()
    return(fig)


def compar_boxplots(st_data_dic, dropna=False, fillna='median', show_plot=False):
    bxp_res_dic = {i:dict() for i in data_sets}
    for ds in st_data_dic.keys():
        fig, axs = plt.subplots(4, 3, figsize=(3*8, 4*5))
        for i, mm in enumerate(st_data_dic[ds].keys()):
            method, metric = mm.split('_')
            df = st_data_dic[ds][mm]
            if(dropna):
                df = df.dropna().reset_index(drop=True)
            else:
                df = df.fillna(df.median()) if(fillna=='median') else df.fillna(df.mean())
            ax = axs.flatten()[i]
            df_lst = [df[i] for i in df.columns]
            ax.boxplot(df_lst, showmeans=True)
            ax.set_ylabel(metric, fontsize=15, weight='bold')
            ax.set_xticklabels(['Basic filtration']+['Removal of HCFs\n(CC=0.%s)' %i.replace('CC','') for i in df.columns[1:]])
            ax.set_title(method, fontsize=15, weight='bold')
            ax.tick_params(axis='x', labelsize=13)
        plt.tight_layout()
        if(not show_plot):
            plt.close()
        bxp_res_dic[ds] = fig
    return(bxp_res_dic)


