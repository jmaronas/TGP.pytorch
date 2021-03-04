#-*- coding: utf-8 -*-
# datasets.py : This file holds an interface with all the datasets that return the dataloader and properties from the data
# Author: Juan Maroñas and  Ollie Hamelijnck

import numpy

from .data import return_data_loader
from .uci_datasets import Boston, Concrete, Kin8nm, Protein, Energy, Power, Wine_Red, Wine_White, Naval, YearMsd
from .uci_datasets_classification import Avila, Banknote, Movement, Activity, Heart
from .regression_datasets import RainFall, Airline
from .air_quality_timeseries import Air_Quality_Timeseries

from .rainfall_spatial import Rainfall_Spatial

## ================ ##
## Interface to UCI ##

uci_list = ['boston', 'concrete', 'yearmsd', 'kin8nm', 'protein', 'energy', 'power', 'wine_red', 'wine_white', 'naval','year', 'avila', 'banknote','movement','activity', 'heart']

classification_datasets = ['avila', 'banknote','movement','activity', 'heart'] 

def return_UCI_dataset(name, seed, use_validation, split_from_disk):

    if name != 'year' and name != 'avila':
        assert seed is not None, "Specify a seed for the partition in UCI"

    if name == 'boston':
        return Boston(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'concrete':
        return Concrete(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'kin8nm':
        return Kin8nm(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'protein':
        return Protein(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'energy':
        return Energy(seed, use_validation = use_validation, split_from_disk = split_from_disk)
            
    elif name == 'power':
        return Power(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'wine_red':
        return Wine_Red(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'wine_white':
        return Wine_White(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'naval':
        return Naval(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'year':
        return YearMsd(use_validation = use_validation)

    elif name == 'avila':
        return Avila(use_validation = use_validation)

    elif name == 'banknote':
        return Banknote(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'movement':
        return Movement(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'activity':
        return Activity(seed, use_validation = use_validation, split_from_disk = split_from_disk)

    elif name == 'heart':
        return Heart(seed,use_validation = use_validation, split_from_disk = split_from_disk)
    else:
        raise ValueError("Got UCI dataset {} which is not available".format(name))


## ========================= ##
##  Return Dataset Function  ##
## ========================= ##
# This function is used as an interfaze to return
# any dataset

def return_dataset(dataset_name, batch_size, use_validation, seed = None, options:dict = None):

    ''' return dataset

    Description: This is the interface to any of our datasets

        Args:
                dataset_name   (str) :->: name of our dataset
                batch_size     (int) :->: batch size
                use_validation (*)   :->: The format depend on each specific dataset
                seed           (int) :->: Random seed use to generate train test split on the datasets that perform this partition on the run. Also specifies the partition in those datasets in which the train-test partition is done at preprocessing time
                options        (dict) :->: optional dictionary to pass options to the dataset

     Note: 

         -> This next code is useful when the validation
            dataset does not require statistics from the
            training dataset for preprocessing steps such
            as normalization. This is the case of calssification
            of images when one tipically divide by 255. 
            This code can be used anyway. The only problem
            is that the validation is normalized with
            statistics from the validation and that does 
            not reflect the reality.

            tot_tr = len(E.train_dataset)
            N_valid = 100
            N_tr    = tot_tr - N_valid

            numpy.random.seed(0)
            aux = numpy.random.permutation(tot_tr)
            train_idx = aux[0:N_tr]
            valid_idx  = aux[N_tr:]

            train_idx = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_idx = torch.utils.data.SubsetRandomSampler(valid_idx)

            train_loader = return_data_loader(E.train_dataset, batch_size,sampler_policy = train_idx)
            valid_loader = return_data_loader(E.train_dataset, batch_size,sampler_policy = valid_idx)
            test_loader = return_data_loader(E.test_dataset, batch_size, shuffle = False)

    '''
    if options == None:
        shuffle_train   = True
        use_generator   = True
        split_from_disk = True
        n_workers       = 0 
    else:
        shuffle_train   = options['shuffle_train']   if 'shuffle_train'   in options.keys() else True
        use_generator   = options['use_generator']   if 'use_generator'   in options.keys() else True
        split_from_disk = options['split_from_disk'] if 'split_from_disk' in options.keys() else True 
        n_workers       = options['n_workers']       if 'n_workers'       in options.keys() else 0

    if dataset_name == 'rainfall':
        Dataset = RainFall(partition = seed, use_validation = use_validation)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train, use_generator = use_generator)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False, use_generator = use_generator)
        data_loaders = [train_loader,test_loader]

    elif dataset_name in uci_list:
        Dataset = return_UCI_dataset(dataset_name, seed, use_validation, split_from_disk)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train, use_generator = use_generator, workers = n_workers)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False, use_generator = use_generator, workers = n_workers)
        data_loaders = [train_loader,test_loader]

    elif dataset_name == 'airline':
        Dataset = Airline(seed,use_validation, split_from_disk)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train, use_generator = use_generator, workers = n_workers)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False, use_generator = use_generator, workers = n_workers)
        data_loaders = [train_loader,test_loader]

    elif dataset_name == 'air_quality_timeseries':
        partition = seed
        Dataset = Air_Quality_Timeseries(partition, use_validation = None, options = options)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False)
        all_loader  = return_data_loader(Dataset.all_dataset, batch_size, shuffle = False)
        data_loaders = [train_loader,test_loader, all_loader]

    elif dataset_name == 'audio':
        partition = seed
        Dataset = Audio(partition, use_validation = None, options = options)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False)
        all_loader  = return_data_loader(Dataset.all_dataset, batch_size, shuffle = False)
        data_loaders = [train_loader,test_loader, all_loader]

    elif dataset_name == 'rainfall_spatial':
        partition = seed
        Dataset = Rainfall_Spatial(partition, use_validation = None, options = options)
        train_loader = return_data_loader(Dataset.train_dataset, batch_size,shuffle = shuffle_train)
        test_loader  = return_data_loader(Dataset.test_dataset, batch_size, shuffle = False)
        all_loader  = return_data_loader(Dataset.all_dataset, batch_size, shuffle = False)
        data_loaders = [train_loader,test_loader, all_loader]

    else:
        raise ValueError("Unkown dataset provided {}".format(dataset_name))

    if use_validation is not None and dataset_name not in toy_list:
        valid_loader = return_data_loader(Dataset.valid_dataset, batch_size, shuffle = shuffle_train, use_generator = use_generator)
        data_loaders = [train_loader, valid_loader, test_loader]

    N_tr = Dataset.N_tr
    N_te = Dataset.N_te
    N_va = Dataset.N_va
    X_tr = Dataset.X_tr
    Y_tr = Dataset.Y_tr
    X_va = Dataset.X_va
    Y_va = Dataset.Y_va
    X_te = Dataset.X_te
    Y_te = Dataset.Y_te
    Y_std = Dataset.Y_std

    X_all = Dataset.X_all
    Y_all = Dataset.Y_all

    D = X_tr.size(1) # input dimensionality
    Do = Dataset.Y_tr.size(1) # output dimensionality

    if dataset_name in classification_datasets:
        Do = len(numpy.unique(Dataset.Y_tr))

    # Data config: used to instance the model or compute performance metrics
    data_config = { 
                    'X_tr' : X_tr,
                    'Y_tr' : Y_tr,
                    'X_va' : X_va,
                    'Y_va' : Y_va,
                    'X_te' : X_te,
                    'Y_te' : Y_te,
                    'N_tr' : N_tr,
                    'N_va' : N_va, 
                    'N_te' : N_te, 
                    'Dx'   : D,
                    'Dy'   : Do , 
                    'Y_std': Y_std,
                    'X_all': X_all,
                    'Y_all': Y_all,
                  }

    return data_loaders, data_config



