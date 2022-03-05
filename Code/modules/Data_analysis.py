#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class Data_analysis  ####
# Collects all functions for data analysis


import numpy as np
import pandas as pd
import pickle
import io


from Evaluation_methods import compute_analytical_rankings,\
                                get_total_best_models_and_predictions,\
                                compute_regr_resid_plots_data,\
                                regr_resid_plots, comp_models_perform_matrices, multi_group_bar_plot,\
                                compute_statistical_data, kw_test, kw_heatmap_plot,\
                                bonf_post_hoc_test, bonf_test_heatmap_plot,\
                                highlight_kw_hm, compar_boxplots


class Data_analysis():
    
    raw_data_dic = dict()
    datasets_dic = dict()
    anal_ranks_dic = dict()
    best_models_and_predictions_dic = dict()
    regr_resid_plots_data_dic = dict()
    regr_resid_plots_dic = dict()
    ffs_plots_dic = dict()
    models_perf_matrices_dic = dict()
    models_perf_plots_dic = dict()
    statistical_data_dic = dict()
    kruskal_wallis_dic = dict()
    bonferroni_post_hoc_df = dict()
    comparative_boxplots_dic = dict()
    
    
    
    def __init__(self, raw_data_obj_lnk):
        if(raw_data_obj_lnk.endswith('.pickle')):
            with open(raw_data_obj_lnk, 'rb') as handle:
                rd_obj = pickle.load(handle)
                self.raw_data_dic = rd_obj.raw_data_dic
                self.datasets_dic = rd_obj.datasets_dic
    
    
    def comp_anal_ranks(self):
        self.anal_ranks_dic = compute_analytical_rankings(self.raw_data_dic)
    
    
    # Get best models predictions for all datasets for all conditions (metric, CC, ranking type)
    def comp_best_models_and_predictions(self, rank_type='Median'):
        if(len(self.anal_ranks_dic)==0):
            self.comp_anal_ranks()
        self.best_models_and_predictions_dic = get_total_best_models_and_predictions(self.raw_data_dic, self.anal_ranks_dic, rank_type=rank_type)

    
    # Collect all data needed for Regression and Residuals plots
    def comp_regr_resid_plots_data(self, rank_type='Median'):
        if(len(self.best_models_and_predictions_dic)==0):
            self.comp_best_models_and_predictions(rank_type=rank_type)
        self.regr_resid_plots_data_dic = compute_regr_resid_plots_data(self.datasets_dic, self.best_models_and_predictions_dic)
    
    
    # Create various types of Regression and/or Residuals plots
    def create_regr_resid_plots(self, cols_n=4, col_width=6, rr_plot_type=1, show_plot=False):
        if(len(self.regr_resid_plots_data_dic)==0):
            self.comp_regr_resid_plots_data(rank_type='Median')
        self.regr_resid_plots_dic = regr_resid_plots(data_to_plot=self.regr_resid_plots_data_dic, cols_n=cols_n, col_width=col_width, rr_plot_type=rr_plot_type, show_plot=show_plot)
    
    
    # Creates plots for FFS analysis for all available datasets and all
    # available regression methods
    def create_ffs_plots(self, res_ffs_dir, metric='MAE', show_plot=False):
        self.ffs_plots_dic = ffs_plots(res_ffs_dir, metric=metric, show_plot=show_plot)
    
    
    def comp_models_perf_matrices_plots(self, width=16, height=7, show_plot=False):
        self.models_perf_matrices_dic = comp_models_perform_matrices(self.raw_data_dic)
        self.models_perf_plots_dic = multi_group_bar_plot(self.models_perf_matrices_dic, width=width,
                                                          height=height, show_plot=show_plot)
    
    
    def comp_statistical_data(self):
        self.statistical_data_dic = compute_statistical_data(self.raw_data_dic)
    
    
    def kruskal_wallis_test(self, dropna=False, fillna='median'):
        if(len(self.statistical_data_dic)==0):
            self.comp_statistical_data()
        self.kruskal_wallis_dic = kw_test(self.statistical_data_dic, dropna=dropna, fillna=fillna)
    
    
    def kruskal_wallis_heatmap_plot(self, store_link=None, a=0.01, show_plot=False):
        kw_fig = kw_heatmap_plot(self.kruskal_wallis_dic, a=a, show_plot=show_plot)
        if(store_link is not None):
            kw_fig.savefig(store_link, dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    
    
    def bonferroni_post_hoc_test(self, a=0.01, p_adjust='bonferroni'):
        self.bonferroni_post_hoc_df = bonf_post_hoc_test(self.kruskal_wallis_dic, self.statistical_data_dic, a=a, p_adjust=p_adjust)
        
        
    def bonferroni_test_heatmap_plot(self, store_link=None, a=0.01, show_plot=False):
        bonf_test_fig = bonf_test_heatmap_plot(self.bonferroni_post_hoc_df, a=a, show_plot=show_plot)
        if(store_link is not None):
            bonf_test_fig.savefig(store_link, dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    
    
    
    def highlight_kruskal_wallis_heatmap(self, store_link=None, a=0.01, show_plot=False):
        hl_kw_fig = highlight_kw_hm(self.kruskal_wallis_dic, self.bonferroni_post_hoc_df, a=a, show_plot=show_plot)
        if(store_link is not None):
            hl_kw_fig.savefig(store_link, dpi=300, format='tiff', pil_kwargs={"compression": "tiff_lzw"})
    
    
    def comparative_metric_boxplots(self, dropna=False, fillna='median', show_plot=False):
        if(len(self.statistical_data_dic)==0):
            self.comp_statistical_data()
        self.comparative_boxplots_dic = compar_boxplots(self.statistical_data_dic, dropna=dropna, fillna=fillna, show_plot=show_plot)
    
    
    ### Exports a data dictionary to a multisheet excel file
    # Use various dictionary key levels for sheet naming:
    # kl=1: Dictionary has 1 key level
    # kl=2: Dictionary has 2 key levels, etc
    def export_dict_to_file(self, data_dic, store_link, keep_index=True, kl=2):
        if(store_link.endswith('.pickle')):
            with open(store_link, 'wb') as handle:
                pickle.dump(data_dic, handle)
        elif(store_link.endswith('.xlsx')):
            writer = pd.ExcelWriter(store_link, engine='xlsxwriter')
            workbook=writer.book
            if(kl==1):
                keys1 = data_dic.keys()
                for key1 in keys1:
                    df = data_dic[key1]
                    df = np.round(df.astype(float),1)
                    sht = '%s' %(key1)
                    df.to_excel(writer, sheet_name=sht, index=keep_index, startrow=0 , startcol=0, merge_cells=True)
                    worksheet=workbook.get_worksheet_by_name(sht)
                    worksheet.set_column(0, 1, 12)
            if(kl==2):
                keys1 = data_dic.keys()
                for key1 in keys1:
                    keys2 = data_dic[key1].keys()
                    for key2 in keys2:
                        df = data_dic[key1][key2]
                        df = np.round(df.astype(float),1)
                        sht = '%s_%s' %(key1, key2)
                        df.to_excel(writer, sheet_name=sht, index=keep_index, startrow=0 , startcol=0, merge_cells=True)
                        worksheet=workbook.get_worksheet_by_name(sht)
                        worksheet.set_column(0, 1, 12)
            writer.save()
    
    
    ### Exports a plots dictionary to a pickle file or 
    ### a multisheet excel file
    # Use various dictionary key levels for sheet naming:
    # kl=1: Dictionary has 1 key level
    # kl=2: Dictionary has 2 key levels, etc
    def export_plots_dict_to_file(self, plots_dic, store_link, kl=1):
        if(store_link.endswith('.pickle')):
            with open(store_link, 'wb') as handle:
                pickle.dump(plots_dic, handle)
        elif(store_link.endswith('.xlsx')):
            writer = pd.ExcelWriter(store_link, engine='xlsxwriter')
            workbook=writer.book
            if(kl==1):
                keys1 = plots_dic.keys()
                for key1 in keys1:
                    fig = plots_dic[key1]
                    sht_n = '%s' %(key1)
                    sht = workbook.add_worksheet(sht_n)
                    imgdata=io.BytesIO()
                    fig.savefig(imgdata, format='png')
                    sht.insert_image(1, 1, '', {'image_data': imgdata})
                    sht.hide_gridlines(2)
            if(kl==2):
                keys1 = list(plots_dic.keys())
                keys2 = list(plots_dic[keys1[0]].keys())
                for key1 in keys1:
                    for key2 in keys2:
                        fig = plots_dic[key1][key2]
                        sht_n = '%s_%s' %(key1, key2)
                        sht = workbook.add_worksheet(sht_n)
                        imgdata=io.BytesIO()
                        fig.savefig(imgdata, format='png')
                        sht.insert_image(1, 1, '', {'image_data': imgdata})
                        sht.hide_gridlines(2)
            writer.save()
    
    
    ### Used in 2 cases needed special handling
    def export_dictionary_to_file(self, data_dic, store_link):
        if(store_link.endswith('_best_models_predictions.xlsx')):
            dsts = list(data_dic.keys())
            for dst in dsts:
                writer = pd.ExcelWriter(store_link.replace('X_', '%s_'%dst), engine='xlsxwriter')
                workbook=writer.book
                ccs = data_dic[dst].keys()
                for cc in ccs:
                    df = data_dic[dst][cc]
                    sht = '%s' %cc
                    df.to_excel(writer, sheet_name=sht, startrow=0 , startcol=0, merge_cells=True)
                    worksheet=workbook.get_worksheet_by_name(sht)
                    worksheet.set_column(0, 1, 12)
                writer.save()
        else:
            dsts = list(data_dic.keys())
            for dst in dsts:
                writer = pd.ExcelWriter(store_link.replace('X_', '%s_'%dst), engine='xlsxwriter')
                workbook=writer.book
                ccs = data_dic[dst].keys()
                for cc in ccs:
                    dfs_n = data_dic[dst][cc].keys()
                    for df_n in dfs_n:
                        df = data_dic[dst][cc][df_n]
                        sht = '%s_%s' %(cc, df_n)
                        df = np.round(df.astype(float),3)
                        df.to_excel(writer, sheet_name=sht, startrow=0 , startcol=0, merge_cells=True)
                        worksheet=workbook.get_worksheet_by_name(sht)
                        worksheet.set_column(0, 0, 15)
                        writer.sheets[sht].set_row(2, None, None, {'hidden': True})
                writer.save()


