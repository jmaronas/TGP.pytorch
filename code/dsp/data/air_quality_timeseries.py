#-*- coding: utf-8 -*-
# regression_datasets.py : This file holds all the regression datasets
# Author: Juan Maroñas and  Ollie Hamelijnck

## python
import os
import sys
sys.path.extend(['../'])

## Standard
import pickle
import numpy as np
import pandas as pd

# Torch
import torch

## custom
from .. import config as cg
from .data import general_dataset_class

import statsmodels.api as sm
import numpy as np

class Air_Quality_Timeseries(general_dataset_class):
    def __init__(self,partition,use_validation = None, options: dict=None) -> None:

        self.shuffle=options['shuffle']

        self.seed = 0#seed used for shuffle
        if 'seed' in options.keys():
            self.seed = options['seed']

        self.sep=','
        self.header=0

        self.site = 'HP5'
        self.species = 'pm25'
        self.start_date = '03/15/2019'
        self.end_date = '04/15/2019'

        self.directory = os.path.join(cg.root_directory,'datasets','air_pollution', 'tools', 'downloaded_data')
        self.name = 'aq_data.csv'
        self.root = os.path.join(self.directory,self.name)

        self.partition =  partition
        self.options = options

        self.split_type = options['split_type']

        X_tr,Y_tr, X_te,Y_te, X_all, Y_all = self.__load_data__()
        X_va, Y_va = None, None
        
        Y_std = 1.0
        X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all, Y_all = self.standard_normalization(X_tr,Y_tr,X_va,Y_va,X_te,Y_te, X_all, Y_all, normalize_y=False)

        super().__init__(X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all, Y_all)

    def __load_data__(self):
        # Load data
        data = self.load_pandas_csv(self.root,self.shuffle,self.seed,self.sep,self.header)

        #clean data
        data['date'] = pd.to_datetime(data['date'])
        data['epoch'] = self.datetime_to_epoch(data['date'])
        data = data[data['site']==self.site]
        data = data[(data['date'] >= self.start_date) & (data['date'] < self.end_date) ]

        self.data = data

        #print(data['date'])
        #exit()

        x = np.array(data['epoch'])[:, None]
        y = np.array(data[self.species])[:, None]



        #remove nans from the dataset before running validation splits so that all splits have the same amount of observations
        #we keep x, y so that we can predict on the whole timeseries for visualisation purposes
        aq_df_no_nans = data[data[self.species].notnull()]


        x_no_nans = np.array(aq_df_no_nans['epoch'])[:, None]
        y_no_nans = np.array(aq_df_no_nans[self.species])[:, None]

        if self.split_type == 'k_fold':
            num_folds = self.options['num_folds']

            X_tr, Y_tr, X_te, Y_te = self.k_fold(x_no_nans, y_no_nans, self.partition, num_folds)
        elif self.split_type == 'random_split':
            validation_size = self.options['validation_size']
            seed = self.partition

            X_tr, Y_tr, X_te, Y_te = self.random_split_data(x_no_nans, y_no_nans, validation_size)
        else:
            raise RuntimeError('Split type {s} not supported'.format(s=self.split_type))

        X_all, Y_all = x, y

        return  X_tr, Y_tr, X_te, Y_te, X_all, Y_all

    def check_integrity(self,partition):
        pass
