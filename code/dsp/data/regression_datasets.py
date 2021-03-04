#-*- coding: utf-8 -*-
# regression_datasets.py : This file holds all the regression datasets
# Author: Juan Maroñas and  Ollie Hamelijnck

## python
import os
import sys
sys.path.extend(['../'])

## Standard
import pickle
import numpy
import pandas as pd

# Torch
import torch

## custom
from .. import config as cg
from .data import general_dataset_class

class RainFall(general_dataset_class):
    def __init__(self,partition,use_validation = None) -> None:
        self.train = 'data_train_'+str(partition)+'.pickle'
        self.test  = 'data_test_'+str(partition)+'.pickle'
        self.raw   = 'data_raw_'+str(partition)+'.pickle'
        self.directory = os.path.join(cg.root_directory,'datasets','regression','rainfall')

        self.check_integrity(partition) 

        X_tr,Y_tr, X_te,Y_te = self.__load_data__()

        X_va,Y_va = None, None 
        if use_validation is not None:
            val_seed,val_N = use_validation[0],use_validation[1]
            X_tr,Y_tr,X_va,Y_va = self.random_split_validation(X_tr,Y_tr,val_seed,val_N)

        Y_std = 1

        super().__init__(X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std)

    def check_integrity(self,partition):
        md5_raw, md5_train, md5_test = self.md5_sum_list(partition)

        #check everything is correctly downloaded and preprocessed
        if not self._check_integrity(self.directory,self.raw,md5_raw):
            raise Exception("Files corrupted. Check preprocessing step") 

        if not self._check_integrity(self.directory,self.train,md5_train):
            raise Exception("Files corrupted. Check preprocessing step") 

        if not self._check_integrity(self.directory,self.test,md5_test):
            raise Exception("Files corrupted. Check preprocessing step") 

    def md5_sum_list(self, partition):
        if partition == 0:
            md5_raw   = 'f087f0dfd56b8357bb6a02176d1aba88'  
            md5_train = '04a2bcd3f4309d9d16cd04d3c9687530' 
            md5_test  = '05ee908f5e9f69ba1c741432fb3c7039' 
        elif partition == 1:
            md5_raw   = '2b9400903279ee04232bf032f248c978' 
            md5_train = '1446d57a5b1061735b58827fc82b05bc' 
            md5_test  = 'fe41f0e2fea5a6724550ed6a6e18ce40' 
        elif partition == 2:
            md5_raw   = '4e288f2f16a0336e746f5f7bf153ddf3' 
            md5_train = '55e27def7e64c9d7258149f3c25a66b2' 
            md5_test  = 'e853b6e7eed6c7cd152e03cb1b2eac74' 
        elif partition == 3:
            md5_raw   = '975ff8ec4cfc47a73c9778cccc00f25c' 
            md5_train = '497f3caec66e984e6fd4514273e999b9' 
            md5_test  = '4313a66ecdd48334602c20e71c1e8f7d' 
        elif partition == 4:
            md5_raw   = '223d477d970dc42b0587ebf09c15c1cb' 
            md5_train = '64c10d2af3f12ca5bbf20b3e4a004c40' 
            md5_test  = '139b4042c5de8063151f95a9c8ecbf35' 

        return md5_raw, md5_train, md5_test

         
    def __load_data__(self):

        train = os.path.join(self.directory,self.train)
        test = os.path.join(self.directory,self.test)

        train = pickle.load(open(train,'rb'))
        test = pickle.load(open(test,'rb'))['test']


        X_tr,Y_tr = torch.tensor(train['X'], dtype = cg.dtype),torch.tensor(train['Y'],dtype = cg.dtype)
        X_te,Y_te = torch.tensor(test['X'], dtype = cg.dtype),torch.tensor(test['Y'], dtype = cg.dtype)

        return X_tr,Y_tr, X_te,Y_te


class Airline(general_dataset_class):
    def __init__(self, seed, use_validation = None, split_from_disk = True) -> None :

        self.directory = os.path.join(cg.root_directory,'datasets','regression','airline')
        self.name = 'airline.csv'
        self.root = os.path.join(self.directory,self.name)
        self.md5sum = '14be75ad425a4c4f47fe75d60829a955'
        self.shuffle = True
        self.header = None

        self.seed = seed # to generate training test partition
        self.index = -1 # regressed label is here 
        self.sep = ','

        self.use_validation = use_validation

        self.N_tr = 2058097 - 100000

        X_tr,Y_tr,X_te,Y_te = self.__load__(split_from_disk)

        X_va,Y_va = None, None 
        if self.use_validation is not None:
            val_seed,val_N = self.use_validation[0],self.use_validation[1]
            X_tr,Y_tr,X_va,Y_va = self.random_split_validation(X_tr,Y_tr,val_seed,val_N)
        
        X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std = self.standard_normalization(X_tr,Y_tr,X_va,Y_va,X_te,Y_te)

        super().__init__(X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std)


    def __load__(self, split_from_disk) -> list:
        ''' This function returns X_tr,X_te,Y_tr,Y_te'''
        if not self._check_integrity(self.directory,self.name,self.md5sum):
            raise ValueError("Dataset {} is corrupted or hasnt been download. Check README file from main repository".format(self.name))

        if split_from_disk:
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

            X_tr,Y_tr = X[0:self.N_tr,:], Y[0:self.N_tr,:]
            X_te,Y_te = X[self.N_tr:,:],  Y[self.N_tr:,:] 

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
            N_tr = self.N_tr
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


