#-*- coding: utf-8 -*-
# uci_dataloaders.py : This file holds UCI data loaders
# Author: Juan Maroñas and  Ollie Hamelijnck

## python
import os
import sys
sys.path.extend(['../'])

## torch
import torch

## standard
import numpy
import pandas as pd
import pickle

# custom
from .. import config as cg
from .data import general_dataset_class
from .utils_data import check_integrity, download_and_extract_archive # torchvision.datasts.utils *


############################
####### UCI DATASETS #######
############################

## The UCI datasets are loaded in the same way as the IPVI and DSVI works
## We randomize and take 90% for training except for YearMSD
## Perform 0 mean 1 std preprocessing
## The last column (-1) is used as the regressed class for all the datasets except
## Energy, that uses the -2.

## UPDATE: An updated version of this code will create fixed partitions for UCI whose indexes are saved as pickle objects
## The intention is to be able to reuse this experiments in the future, newer numpy versions will not generate different
## partitions.


class UCI_data(general_dataset_class):
    '''
    -> This general dataset class is used for UCI.
    '''
    def __init__(self,task_type = 'regression', normalize_y = True, download = False, remove_finished = True, split_from_disk = True) -> None:
        assert task_type in ['classification','regression']
        self.directory       = os.path.join(cg.root_directory,'datasets',task_type,'uci/')
        self.root            = os.path.join(cg.root_directory,'datasets',task_type,'uci/' + self.name)
        self.prop            = 0.9
        self.download        = download
        self.remove_finished = remove_finished

        X_tr,Y_tr,X_te,Y_te = self.__load__(split_from_disk)

        X_va,Y_va = None, None 
        if self.use_validation is not None:
            val_seed,val_N = self.use_validation[0],self.use_validation[1]
            X_tr,Y_tr,X_va,Y_va = self.random_split_validation(X_tr,Y_tr,val_seed,val_N)

        X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std = self.standard_normalization(X_tr,Y_tr,X_va,Y_va,X_te,Y_te, normalize_y = normalize_y)

        super().__init__(X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std)

    def __load__(self,split_from_disk) -> list:

        ''' This function returns X_tr,X_te,Y_tr,Y_te'''

        if self.download:
            self._download(self.url,self.directory,self.name,self.md5sum, self.remove_finished)

        if not self._check_integrity(self.directory,self.name,self.md5sum):
            raise ValueError("Dataset {} is corrupted".format(self.name))

        if split_from_disk:
            # If true then reads the split from the disk. Avoids changing partition when updating numpy
            data = self.load_csv_data(self.root,False,None,self.sep,self.header)

            X,Y = data[ : , :self.index ],data[ : , self.index ] 
            Y = numpy.reshape(Y,(Y.shape[0],1))

            abs_path   = self.directory
            abs_name   = os.path.join(abs_path, 'splits_idx_'+self.name.split('.')[0]+'.pkl')
            f          = open(abs_name,"rb")
            split_dict = pickle.load(f)

            tr_idx = split_dict['seed_'+str(self.seed)]['train']
            te_idx = split_dict['seed_'+str(self.seed)]['test']

            data_tr = data[tr_idx]
            data_te = data[te_idx]

            X_tr    = data_tr[: , :self.index]
            X_te    = data_te[: , :self.index]

            Y_tr    = data_tr[: , self.index]
            Y_te    = data_te[: , self.index]

            Y_tr = numpy.reshape(Y_tr,(Y_tr.shape[0],1))
            Y_te = numpy.reshape(Y_te,(Y_te.shape[0],1))

        else:
            data = self.load_csv_data(self.root,self.shuffle,self.seed,self.sep,self.header)

            X,Y = data[ : , :self.index ],data[ : , self.index ]
            Y = numpy.reshape(Y,(Y.shape[0],1))

            X_tr,Y_tr,X_te,Y_te = self.random_split_data(X,Y,self.prop)

        return X_tr,Y_tr,X_te,Y_te 

    def __generate_splits__(self,split: list):
        '''Generate a pickle object saving the split index for each seed in split list'''
        splits_idx = {}
        X    = pd.pandas.read_csv(self.root, sep = self.sep, header = self.header)
        X    = X.to_numpy()
        rows = X.shape[0]

        for seed in split:
            numpy.random.seed(seed)
            perm = numpy.random.permutation(rows)
            N_tr = int(rows*self.prop)
            train_idx, test_idx = perm[0:N_tr],perm[N_tr:]

            aux_dic = {'train': train_idx, 'test': test_idx}
            splits_idx.update(
                              {
                                  'seed_'+str(seed) : aux_dic
                              }
                             )

        # save partitions to pickle objects
        abs_path = self.directory
        abs_name = os.path.join(abs_path, 'splits_idx_'+self.name.split('.')[0]+'.pkl')
        f = open(abs_name,"wb")
        pickle.dump(splits_idx,f)
        f.close()


class YearMsd(UCI_data):
    # This UCI dataset has specified train and test partitions
    def __init__(self, use_validation = None, split_from_disk = False) -> None :
        self.url   = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip' # this dataset is downloaded  
        self.md5sum = '9b07d8011e0d8add1150dee90a38c548'

        self.sep = ','
        self.header = None
        self.name = 'YearPredictionMSD.txt'
        self.train_ind = 463715 
        self.test_ind  = 51630

        self.use_validation = use_validation

        super().__init__(download = True)

    def __load__(self, split_from_disk):
        # Download dataset if it hasnt been downloaded 
        if self.download:
            self._download(self.url,self.directory,self.name,self.md5sum)

        #check everything is correctly downloaded
        if not self._check_integrity(self.directory,self.name,self.md5sum):
            raise Exception("Files corrupted. Set download=True") 

        # load data 
        data = self.load_csv_data(self.root, shuffle=False, seed = None, sep = self.sep, header = self.header) # train and test is already given for this dataset

        X,Y = data[ : , 1: ],data[ : , 0 ]
        Y = numpy.reshape(Y,(Y.shape[0],1))

        X_tr,Y_tr = X[0:self.train_ind,:], Y[0:self.train_ind,:]
        X_te,Y_te = X[self.train_ind:,:],  Y[self.train_ind:,:] 

        return X_tr, Y_tr, X_te, Y_te

class Boston(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'boston.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '686696c995bd450cdd718dad546014e0'
        super().__init__(split_from_disk = split_from_disk)


class Concrete(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'concrete.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '5b5b69bd932abfcab8062214cb48d5aa'
        super().__init__(split_from_disk = split_from_disk)

    
class Kin8nm(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'kin8nm.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '5d5e02dacbad6451fc6310bc2163cd7a'
        super().__init__(split_from_disk = split_from_disk)

class Protein(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'protein.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '2617524f6f3393d482b12a17329556eb'
        super().__init__(split_from_disk = split_from_disk)

class Energy(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'energy.csv'
        self.index = -2
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = 'fdc160476bde85d01ab75b1e2b1c63c8'
        super().__init__(split_from_disk = split_from_disk)

class Power(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'power.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '4c0814dc6aec16aad6500f37243f16a0'
        super().__init__(split_from_disk = split_from_disk)

class Wine_Red(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'wine-red.csv'
        self.index = -1
        self.sep = ','
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '3a55dc882b07a5c342137152622d07fa'
        super().__init__(split_from_disk = split_from_disk)

class Wine_White(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'wine-white.csv'
        self.index = -1
        self.sep = ';'
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '024acd1a22808344471a56f318a49016'
        super().__init__(split_from_disk = split_from_disk)

class Naval(UCI_data):
    def __init__(self,seed, use_validation = None, split_from_disk = True) -> None :
        self.seed = seed # to generate training test partition
        self.name = 'naval.tsv'
        self.index = -1
        self.sep = '   '
        self.header = None
        self.shuffle = True
        self.use_validation = use_validation
        self.md5sum = '4a95e860425c9cf765b56f0134ad99cf'
        super().__init__(split_from_disk = split_from_disk)
