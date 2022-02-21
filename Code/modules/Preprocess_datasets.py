#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####  Class Preprocess_datasets  ####
# Collects all functions for dataset initial datasets' preprocessing

import os
import pandas as pd
import pickle
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


from Data_filtering_methods import create_dsets_configs
from Analysis_methods import data_sets, comps_rs


class Preprocess_datasets():
    
    rt_thrsh = 300
    smrt_n = 'SMRT_dataset.sdf'
    smrt_info_n = 'SMRT_RT_InChI_RT.csv'
    method = 'y_cor'
    
    def __init__(self, dsets_dir_lnk):
        print('Start calculating MDs for in-house datasets...')
        self.calc_inhouse_MDs(dsets_dir_lnk=dsets_dir_lnk)
        print('\t*** DONE ***')
        print('Start creatind external-datasets and calculating MDs...')
        self.create_ext_dsets_calc_MDs(dsets_dir_lnk=dsets_dir_lnk, smrt_n=self.smrt_n, smrt_info_n=self.smrt_info_n, rt_thrsh=self.rt_thrsh)
        print('\t*** DONE ***')
        create_dsets_configs(dsets_dir_lnk=dsets_dir_lnk, method=self.method)

        
    ### Compute Molecular Descriptors for in-house datasets
    def calc_inhouse_MDs(self, dsets_dir_lnk):
        in_excels = [f for f in os.listdir(dsets_dir_lnk) if f.endswith('.xlsx') and f.startswith('IH_')]
        for dset_n in data_sets:
            if (not dset_n.startswith('IH_')):
                continue
            in_excel = [f for f in in_excels if dset_n in f][0]
            sm_df = pd.read_excel(os.path.join(dsets_dir_lnk, in_excel), index_col=0)
            PandasTools.AddMoleculeColumnToFrame(sm_df,'SMILES','Molecule')
            PandasTools.WriteSDF(sm_df, 'SDF_input.sdf', molColName='Molecule', idName='RowID')
            mols_df = PandasTools.LoadSDF('SDF_input.sdf', molColName='Molecule')
            mols = pd.Series(mols_df['Molecule'])
            mols.index=mols_df['ID']
            des_list = [x[0] for x in Descriptors._descList]
            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
            desctrs = [calculator.CalcDescriptors(mol) for mol in mols]
            descrs_df = pd.DataFrame(desctrs, index=mols.index, columns=des_list)
            dset_subdir = os.path.join(dsets_dir_lnk, dset_n)
            if not os.path.exists(dset_subdir):
                os.makedirs(dset_subdir)
            descrs_df.to_csv(os.path.join(dset_subdir, 'X_input_data.csv'))
            sm_df['RT'].to_csv(os.path.join(dset_subdir, 'y_input_data.csv'))
        os.remove('SDF_input.sdf')
        

    ### Create external datasets and compute Molecular Descriptors
    def create_ext_dsets_calc_MDs(self, dsets_dir_lnk, smrt_n, smrt_info_n, rt_thrsh=300):
        smrt_info_df = pd.read_csv(os.path.join(dsets_dir_lnk, smrt_info_n), index_col=0)
        try:
            pickl_fn = os.path.join(dsets_dir_lnk, 'SMRT_dataset.sdf.pickle')
            with open(pickl_fn, 'rb') as handle:
                mols_df = pickle.load(handle)
        except:
            mols_df = PandasTools.LoadSDF(os.path.join(dsets_dir_lnk,smrt_n), molColName='Molecule')
            mols_df = mols_df.iloc[:,1:]
            mols_df['RETENTION_TIME'] = mols_df['RETENTION_TIME'].astype(float)
            mols_df['ID'] = mols_df['ID'].astype(int)
            mols_df = mols_df[mols_df['RETENTION_TIME'].astype(float)>rt_thrsh]
            mols_df.set_index('ID', inplace=True)
            mols_df.sort_index(inplace=True)
        for dset_n in data_sets:
            if (not dset_n.startswith('SMRT_')):
                continue
            comps_n, rs = comps_rs[dset_n]
            rand_df = mols_df.sample(n=comps_n, random_state=rs)
            rand_df.sort_index(inplace=True)
            mols = pd.Series(rand_df['Molecule'])
            des_list = [x[0] for x in Descriptors._descList]
            calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
            desctrs = [calculator.CalcDescriptors(mol) for mol in mols]
            descrs_df = pd.DataFrame(desctrs, index=mols.index, columns=des_list)
            dset_subdir = os.path.join(dsets_dir_lnk, dset_n)
            if not os.path.exists(dset_subdir):
                os.makedirs(dset_subdir)
            descrs_df.to_csv(os.path.join(dset_subdir, 'X_input_data.csv'))
            rand_df.rename({'RETENTION_TIME': 'RT'}, inplace=True, axis=1)
            rand_df['RT'].to_csv(os.path.join(dset_subdir, 'y_input_data.csv'))
            smrt_info_df.loc[rand_df.index].to_csv(os.path.join(dsets_dir_lnk, '%s_InChI_RT.csv' %dset_n))
    
    
