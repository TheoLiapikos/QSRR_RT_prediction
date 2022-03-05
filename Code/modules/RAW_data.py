#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class RAW_data  ####
# Collects and combines all raw data for each dataset into a unique dictionary
# Info about dataset, CorCoefs, metrics, etc, used are stored at module 'Analysis_methods.py'
# Basic functions are stored in module 'Collect_RAW_data.py'

import numpy as np
import pandas as pd
import pickle


from Collect_RAW_data import combine_raw_data


class RAW_data():
    
    raw_data_dic = dict()
    datasets_dic = dict()    
    
    def __init__(self, raw_data_lnk, res_raw_data_lnk):
        self.raw_data_dic = combine_raw_data(raw_data_lnk)
        self.datasets_dic = self.set_datasets_dic()

    
    def set_datasets_dic(self):
        dsets = self.raw_data_dic.keys()
        dsets_dic = {i:dict() for i in dsets}
        for dset in dsets:
            exp_rt = self.raw_data_dic[dset]['CC1']['Predictions'].loc[('Exp_RT',0),:]
            exp_rt.name=dset
            dsets_dic[dset]['Exp_RT'] = exp_rt
            dsets_dic[dset]['Compounds'] = exp_rt.index.values
        return(dsets_dic)

        
    def export_to_file(self, store_link):
        if(store_link.endswith('.pickle')):
            with open(store_link, 'wb') as handle:
                pickle.dump(self, handle)
        elif(store_link.endswith('.xlsx')):
            dsts = list(self.raw_data_dic.keys())
            for dst in dsts:
                writer = pd.ExcelWriter(store_link.replace('X_', '%s_'%dst), engine='xlsxwriter')
                workbook=writer.book
                ccs = self.raw_data_dic[dst].keys()
                for cc in ccs:
                    dfs_n = self.raw_data_dic[dst][cc].keys()
                    for df_n in dfs_n:
                        df = self.raw_data_dic[dst][cc][df_n]
                        sht = '%s_%s' %(cc, df_n)
                        if(df_n=='Predictions'):
                            df = np.round(df.astype(float),1)
                            df.to_excel(writer, sheet_name=sht, startrow=0 , startcol=0, merge_cells=True)
                            worksheet=workbook.get_worksheet_by_name(sht)
                            worksheet.set_column(0, 0, 12)
                        elif(df_n=='Metrics'):
                            df = np.round(df.astype(float),5)
                            df.to_excel(writer, sheet_name=sht, startrow=0 , startcol=0, merge_cells=True)
                            worksheet=workbook.get_worksheet_by_name(sht)
                            worksheet.set_column(0, 0, 12)
                        elif(df_n=='Best_Performers_per_Metric'):
                            df.to_excel(writer, sheet_name=sht, startrow=0 , startcol=0, merge_cells=True)
                            worksheet=workbook.get_worksheet_by_name(sht)
                            worksheet.set_column(0, 0, 7)
                            worksheet.set_column(1, 10, 15)
                writer.save()

   