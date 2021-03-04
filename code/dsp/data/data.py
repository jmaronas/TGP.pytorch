#-*- coding: utf-8 -*-
# data.py : This file holds function and utilities for loading data into dataloaders and so on
# Author: Juan Maroñas and  Ollie Hamelijnck

## python
import os
import sys
sys.path.extend(['../'])

## standard
import pandas as pd
import numpy
from sklearn.model_selection import KFold

## torch
import torch

## custom
from .. import  config as cg
from .utils_data import check_integrity, download_and_extract_archive # torchvision.datasts.utils *


###############################
## General Data Loader
###############################

def return_data_loader(dataset: torch.utils.data.Dataset, batch_size: int, workers: int = 0 , shuffle: bool = True , sampler_policy = None, use_generator: bool = False) -> torch.utils.data.DataLoader :
    # workers are the number of threads used to load the data. If -1 then use system nproc.
    persistent_worker = False
    if workers < 0 and cg.is_linux : # for mac compatibility
        workers = (int)(os.popen('nproc').read())
    if workers < 0: #use main thread if -1 and not linux system
        workers = 0

    if workers > 0:
        persistent_worker = True

    ## Juan. Message for all: I usually use multithreading to load datasets, but for this particular project it can incur an overhead of 1 order of magnitude. My deep learning background where I dealt with images and costly data augmentation pipelines made me use multithreading to reduce the overhead. But since UCI doesnt need data preprocessing it seems one thread works fine. Set workers to -1 vs workers to 0 and launch a SVGP. You will see!

    # Generator dont work in pytorch 1.5.0
    gen = None
    if use_generator and torch.__version__ == '1.7.0':
        gen = torch.Generator()
        gen.manual_seed(cg.config_seed)

    if isinstance(sampler_policy,torch.utils.data.SubsetRandomSampler):
        # subsetrandomsampler should be instanced with a generator 
        if torch.__version__ == '1.7.0':
            data_loader =  torch.utils.data.DataLoader( dataset, batch_size = batch_size, num_workers=workers, drop_last = False, shuffle = False, sampler = sampler_policy, persistent_worker = persistent_worker, generator = gen) 
        else:
            data_loader =  torch.utils.data.DataLoader( dataset, batch_size = batch_size, num_workers=workers, drop_last = False, shuffle = False, sampler = sampler_policy)

    elif sampler_policy is not None:
        raise ValueError("Only torch.utils.data.SubsetRandomSampler are allowed, yours is {}".format(type(sampler_policy)))

    else:
        if torch.__version__ == '1.7.0':
            data_loader =  torch.utils.data.DataLoader( dataset, batch_size = batch_size, num_workers = workers, drop_last = False, shuffle = shuffle, persistent_workers = persistent_worker, generator = gen)
        else:
            data_loader =  torch.utils.data.DataLoader( dataset, batch_size = batch_size, num_workers = workers, drop_last = False, shuffle = shuffle)
    return data_loader

###############################
## General vector dataset class
###############################
class dataset_class(torch.utils.data.Dataset):
    ''' General dataset class

    Description: This is the general class for a dataset where inputs X and targets Y are vectors in standard Machine Learning form. This dataset is usefull for data that fitsin RAM+SWAP memory.

        Args:
                X: input -> torch.tensor
                Y: targets -> torch.tensor
    '''

    def __init__(self,X: torch.tensor, Y: torch.tensor):
        super(dataset_class, self).__init__()
        assert X is not None , "Invalid None type"
        assert Y is not None, "Invalid None type"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self,idx):
        # Not remember very well, but this class can in principle return numpy, torch.tensor, even objects.
        return self.X[idx],self.Y[idx]

############################################################
## regression dataset class with training test and valid partitions
############################################################
class general_dataset_class(object):
    def __init__(self,X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all=None, Y_all=None) -> None:
        ''' General  dataset class. It supports useful methods and features for the datasets consider in this project
                 
        -> Description: This class keep training validation and test set in memory and 
                        also holds a torch tensor data loader. It also holds the output std (Y_std)
        -> For large datasets given in a single csv file we will have to think
                        how to code it. Perhaps using chunks a csv. If images 
                        were the input then just place them in a folder and
                        keep a list in memory with image identifiers
                        so when calling __getitem__ method each image can be 
                        loaded. For images we can also use the folder dataset


        #TODO: Deal with Large datasets that does not fit in RAM memory

        '''
        X_tr = torch.tensor(X_tr,dtype = cg.dtype)
        Y_tr = torch.tensor(Y_tr,dtype = cg.dtype)

        # torch datasets
        self.train_dataset = dataset_class(X_tr, Y_tr)
        N_tr = X_tr.size(0)

        N_te = 0
        if Y_te is not None:
            X_te = torch.tensor(X_te,dtype = cg.dtype)
            Y_te = torch.tensor(Y_te,dtype = cg.dtype)
            self.test_dataset  = dataset_class(X_te, Y_te)
            N_te  = X_te.size(0)
 
        N_va = 0
        if Y_va is not None:
            X_va = torch.tensor(X_va,dtype = cg.dtype)
            Y_va = torch.tensor(Y_va,dtype = cg.dtype)
            self.valid_dataset = dataset_class(X_va, Y_va)
            N_va  = X_va.size(0)

        N_all = 0
        if Y_all is not None:
            X_all = torch.tensor(X_all,dtype = cg.dtype)
            Y_all = torch.tensor(Y_all,dtype = cg.dtype)
            self.all_dataset = dataset_class(X_all, Y_all)
            N_all  = X_all.size(0)


        self.N = N_tr + N_te + N_va
        self.N_tr = N_tr
        self.N_te = N_te
        self.N_va = N_va
        self.N_all = N_all

        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_te = X_te
        self.Y_te = Y_te 
        self.X_va = X_va
        self.Y_va = Y_va
        self.Y_std = Y_std

        self.X_all = X_all
        self.Y_all = Y_all

    def load_pandas_csv(self,root,shuffle,seed,sep,header):
        """
            Similar to load_csv_data except this returns a pandas dataframe instead of a numpy method
        """
        X_pd = pd.pandas.read_csv(root, sep=sep, header = header)

        if shuffle:
            X_pd = X_pd.sample(frac=1, random_state=seed) 

        return X_pd

    def cast_outputs(self):
        if self.Y_tr is not None:
            self.Y_tr  = self.Y_tr.long()
            self.train_dataset = dataset_class(self.X_tr,self.Y_tr)

        if self.Y_te is not None:
            self.Y_te  = self.Y_te.long()
            self.test_dataset = dataset_class(self.X_te,self.Y_te)

        if self.Y_va is not None:
            self.Y_va  = self.Y_va.long()
            self.valid_dataset = dataset_class(self.X_va,self.Y_va)

        if self.Y_all is not None:
            self.Y_all = self.Y_all.long()
            self.all_dataset = dataset_class(self.X_all,self.Y_all)

    def load_csv_data(self,root,shuffle,seed,sep,header, return_pandas = False):

        X = pd.pandas.read_csv(root, sep=sep, header = header)
        if return_pandas: # this will use the method incorporated by Ollie once we merge
            return X 

        X = X.to_numpy()
        if shuffle:
            numpy.random.seed(seed)
            numpy.random.shuffle(X)
        return X

    def random_split_data(self,data_X,data_Y,prop):
        """
            args:
                prop: percentage of how many testing points
        """
        assert data_X.shape[0] == data_Y.shape[0], "Different shapes for data_X and data_y"
        N = data_X.shape[0]

        # Not used anymore
        # numpy.random.seed(0) # always zero so that the split is always the same. The randomness is introduced in the self.load_csv_data method
        # rand_index = numpy.random.permutation(N)

        # number of training points
        N_tr =  int(N * prop) 

        X_tr,Y_tr = data_X[0:N_tr,:],data_Y[0:N_tr,:]
        X_te,Y_te = data_X[N_tr:,:], data_Y[N_tr:,:]

        return X_tr,Y_tr,X_te,Y_te

    def random_split_validation(self,data_X,data_Y,seed,N_val):
        """
            Create a random split with shuffling
        """

        assert data_X.shape[0] == data_Y.shape[0], "Different shapes for data_X and data_y"

        N_tr = data_X.shape[0]
        assert N_val <= N_tr, "Got more validation points {} than total training size {}".format(N_val,N_tr)
        numpy.random.seed(seed)
        aux = numpy.random.permutation(N_tr)

        train_idx = aux[0:N_tr-N_val]
        valid_idx = aux[N_tr-N_val:]

        X_tr,  Y_tr = data_X[train_idx,:], data_Y[train_idx]
        X_va,  Y_va = data_X[valid_idx,:], data_Y[valid_idx]

        return X_tr,Y_tr,X_va,Y_va

    def k_fold(self, data_X, data_Y, seed, N_val):
        """
            - seed refers to the k'th k-fold partition 
            - N_val refers to the number of folds
        """

        fold_id = seed
        num_folds = N_val

        assert data_X.shape[0] == data_Y.shape[0], "Different shapes for data_X and data_y"
        assert fold_id < num_folds, "fold id >= num_folds"

        #always use random state as shuffling happens when the data is loaded
        kf = KFold(n_splits=N_val, shuffle=False)

        #get all folds and select current one
        folds = [[train_idx, test_idx] for train_idx, test_idx in kf.split(data_X)]
        train_idx, test_idx = folds[fold_id]
        
        X_tr,  Y_tr = data_X[train_idx,:], data_Y[train_idx]
        X_te,  Y_te = data_X[test_idx,:], data_Y[test_idx]

        return X_tr, Y_tr, X_te, Y_te

    def standard_normalization(self,X_tr,Y_tr,X_va,Y_va,X_te,Y_te, X_all=None, Y_all=None, normalize_y=True):

        self.epsilon = 1e-15
        X_mean = numpy.mean(X_tr, 0)
        X_std = numpy.std(X_tr, 0) + self.epsilon

        if normalize_y:
            Y_mean = numpy.mean(Y_tr, 0)
            Y_std = numpy.std(Y_tr, 0) + self.epsilon
        else:
            Y_std = 1.0
            Y_mean = 0.0
    
        if hasattr(self,'are_categorical'):
            cat_idx = numpy.ones((X_mean.shape[0]))
            cat_idx[self.are_categorical] = 0

        else:
            cat_idx = numpy.ones((X_mean.shape[0]))
        cat_idx = cat_idx.astype('bool')

        X_tr[:,cat_idx] = (X_tr[:,cat_idx] - X_mean[cat_idx]) / X_std[cat_idx]
        X_te[:,cat_idx] = (X_te[:,cat_idx] - X_mean[cat_idx]) / X_std[cat_idx]

        if normalize_y:
            Y_tr = (Y_tr - Y_mean) / Y_std
            Y_te = (Y_te - Y_mean) / Y_std

        if X_va is not None:
            X_va[:,cat_idx] = (X_va[:,cat_idx] - X_mean[cat_idx]) / X_std[cat_idx]
            if normalize_y:
                Y_va        = (Y_va            - Y_mean)          / Y_std

        if X_all is not None:
            X_all[:,cat_idx] = (X_all[:,cat_idx] - X_mean[cat_idx]) / X_std[cat_idx]
            Y_all            = (Y_all            - Y_mean)          / Y_std

        if X_all is not None:
            return X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std, X_all, Y_all
        return X_tr,Y_tr,X_va,Y_va,X_te,Y_te,Y_std

    def datetime_to_epoch(self, datetime):
        """
            Converts a datetime to a number
            args:
                datatime: is a pandas column

        """
        return datetime.astype('int64')//1e9


    def _check_integrity(self,directory,filename,md5sum):
        fpath=os.path.join(directory,filename)
        if not check_integrity(fpath, md5sum):
            return False
        return True


    def _download(self,url,directory,filename,md5sum,remove_finished = True):
        if self._check_integrity(directory,filename,md5sum):
            print('Files already downloaded and verified')
            return

        #fpath=os.path.join(directory,filename)
        download_and_extract_archive(url, directory, remove_finished = remove_finished)


 
