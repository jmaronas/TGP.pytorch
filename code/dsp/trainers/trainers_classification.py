#-*- coding: utf-8 -*-
# trainers_classification.py : Trainers methods for classification
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import time

# Standard
import numpy

# Torch
import torch
import torch.distributions as td

# Custom
from .trainers_regression import Trainer_GP_regression
from .. import config as cg


### Trainer method for the sparse GP. ###
class Trainer_GP_classification(Trainer_GP_regression):
    def __init__(self,model:torch.nn.Module, data_loaders: list, validate_each: int, plot: bool, track: bool, Y_std: float, plot_each: int = 1, S_test : int = None, inference_in_cpu = False) -> None:
        ''' 
                Args:
                        `model`           (torch.nn.Module)  :->: the model to be optimized
                        `data_loaders`    (list)             :->: dataloaders [train,valid,test]. Possibilities: [train], [train,test] and [train,valid,test]
                        `validate_each`   (int)              :->: evaluate validation and test each this number epochs
                        `plot`            (bool)             :->: whether to plot or not. If true, then call self.plot() 
                        `track`           (bool)             :->: whether to track or not. If true, then call self.track()  and self.track_global()
                        `Y_std`           (torch.tensor)     :->: Standard desviation of your regressed variable. Used to compute performance metrics. Should have shape (Dy,)
                        `plot_each`       (int)              :->: each plot_each epochs function plot is call. Default 1
                        `S_test`          (int)              :->: Used to compute the predictive distribution of the Copula process when input dependent flows use dropout
                        `inference_in_cpu (bool)             :->: If true, then inference for computing metrics is done in CPU. This is specially usefull when S_test >>> S_train
        '''

        Trainer_GP_regression.__init__(self,model,data_loaders,validate_each,plot, track, Y_std, plot_each, S_test, inference_in_cpu) 	


    def initialize_to_last_results(self):
        self.to_last_results = [self.ELogL_train, self.acc_train, self.ELogL_valid, self.acc_valid, self.ELogL_test, self.acc_test] # save initial values for metrics.

    def reset_before_epoch(self):
        self.tot_batch, self.tot_train, self.tot_test, self.tot_valid = [0.0]*4 # summarized statistics of the datasets
        self.acc_loss, self.acc_ell, self.acc_kld = [0.0]*3 # to monitor the elbo

        ##  == Performance measures == ##

        # Expected predictive log likelihood
        self.ELogL_train = numpy.zeros(1,) 
        self.ELogL_valid = numpy.zeros((1,)) 
        self.ELogL_test  = numpy.zeros((1,))

        # Accuracy
        self.acc_train   = numpy.zeros((1,))
        self.acc_valid   = numpy.zeros((1,)) 
        self.acc_test    = numpy.zeros((1,)) 

        # Monitor Time per epoch
        self.current_t = time.time()

    def compute_metrics(self):

        '''
        Computes all the metrics for all the dataset loaders
        '''
        self.reset_before_epoch()

        tot_train, tot_valid, tot_test = [0.0]*3
        for x,y in self.train_loader:
            x,y = x.to(cg.device), y.to(cg.device)
            self.performance_metrics(x,y,dataset='train')
            tot_train += x.size(0)

        if self.is_valid:
            for x,y in self.valid_loader:
                x,y = x.to(cg.device), y.to(cg.device)
                self.performance_metrics(x,y,dataset='valid')
                tot_valid += x.size(0)

        if self.is_test:
            for x,y in self.test_loader:
                x,y = x.to(cg.device), y.to(cg.device)
                self.performance_metrics(x,y,dataset='test')
                tot_test += x.size(0)

        # Performance Metrics normalized
        logL_train = self.ELogL_train / float(tot_train)
        acc_train  = self.acc_train   / float(tot_train)

        logL_valid = self.ELogL_valid
        acc_valid  = self.acc_valid
        if self.is_valid:
            logL_valid /= float(tot_valid)
            acc_valid  /= float(tot_valid)

        logL_test = self.ELogL_test
        acc_test  = self.acc_test
        if self.is_test:
            logL_test /= float(tot_test)
            acc_test  /= float(tot_test)

        return logL_train.item(), acc_train.item(), logL_valid.item(), acc_valid.item(), logL_test.item(), acc_test.item()

    def performance_metrics(self,X,Y,dataset):

        Y_std = self.Y_std

        if self.inference_in_cpu:
            current_device = cg.device
            cg.device      = 'cpu'
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
            Y_std = Y_std.to(cg.device)
            X = X.to(cg.device)
            Y = Y.to(cg.device)

        lpy, predictive_params = self.model.test_log_likelihood(X, Y, return_moments = True, Y_std = Y_std, S_MC_NNet = self.S_test)

        if dataset == 'train':
            ElogL , acc = self.ELogL_train, self.acc_train

        elif dataset == 'valid':
            ElogL , acc = self.ELogL_valid, self.acc_valid
            self.tot_valid += X.size(0)

        elif dataset == 'test':
            ElogL , acc = self.ELogL_test, self.acc_test
            self.tot_test += X.size(0)

        # Compute performance measures and accumulate them
        my = predictive_params[0]
        num_success = (my.argmax(dim=1)==Y.squeeze()).sum()
       
        # Save the errors
        ElogL += lpy.to('cpu').numpy()
        acc   += num_success.to('cpu').numpy()

        # Accumulate in its accumulators
        if dataset == 'train':
            self.ELogL_train = ElogL 
            self.acc_train   = acc
        
        elif dataset == 'valid':
            self.ELogL_valid = ElogL 
            self.acc_valid   = acc

        elif dataset == 'test':
            self.ELogL_test = ElogL 
            self.acc_test   = acc

        if self.inference_in_cpu:
            cg.device      = current_device
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
        
    def stats_summary(self,lr,epochs,ep):

        if ((ep+1)%self.validate_each) == 0: 
            Y_std = self.Y_std
            # Performance Metrics
            logL_train  = self.ELogL_train / float(self.tot_train)
            acc_train   = self.acc_train   / float(self.tot_train)

            logL_valid = self.ELogL_valid
            acc_valid  = self.acc_valid
            if self.is_valid:
                logL_valid /= float(self.tot_valid)
                acc_valid   = acc_valid / float(self.tot_valid)

            logL_test = self.ELogL_test
            acc_test  = self.acc_test
            if self.is_test:
                logL_test /= float(self.tot_test)
                acc_test   = acc_test / float(self.tot_test)

            #logL_train,acc_train = float(logL_train),float(acc_train)
            #logL_valid,acc_valid = float(logL_valid),float(acc_valid)
            #logL_test,acc_test   = float(logL_test), float(acc_test)

            # Save the last computed results.
            self.to_last_results = [logL_train, acc_train, logL_valid, acc_valid, logL_test, acc_test]


        # Training Metrics
        acc_loss = float(self.acc_loss)/float(self.tot_train)
        acc_ell  = float(self.acc_ell)/float(self.tot_train) 
        acc_kld  = float(self.acc_kld)/float(self.tot_train)

        string_print_common =  "\n|| Epoch [{}/{}] took {:.5f} minutes LR: {:.4f} \t Loss ==>> ELBO: {:.5f} ELL: {:.5f} KLD: {:.5f} \n \t == STATS == \n".format(
                                        ep+1,epochs,(time.time()-self.current_t)/60., lr, acc_loss, acc_ell, acc_kld)
        string_print_common += "\n"

        string_train, string_valid, string_test = [""]*3

        if ((ep+1)%self.validate_each) == 0: 
            string_train  =  "\t => Train: total samples {} \t".format(self.tot_train) 
            string_train += "\t \t  LOGL {:.5f}".format(logL_train.item()) + "  ACC [%] {:.5f}".format(acc_train.item()*100)
            string_train += "\n"
                    
            string_valid  =  "\t => Valid: total samples {} \t".format(self.tot_valid) 
            string_valid += "\t \t  LOGL {:.5f}".format(logL_valid.item()) + "  ACC [%] {:.5f}".format(acc_valid.item()*100)
            string_valid += "\n"

            string_test  =  "\t => Test: total_samples {}  \t".format(self.tot_test) 
            string_test +=  "\t \t LOGL {:.5f}".format(logL_test.item()) + "  ACC [%] {:.5f}".format(acc_test.item()*100)
                    
        string_test += "\n"

        print(string_print_common + string_train + string_valid + string_test)


    def get_last_results(self) -> list:
        logL_train, acc_train, logL_valid, acc_valid, logL_test, acc_test = self.to_last_results
        return logL_train.item(), acc_train.item(), logL_valid.item(), acc_valid.item(), logL_test.item(), acc_test.item()		


### === Trainer for sparse SP classification === ###
class Trainer_SP_classification(Trainer_GP_classification):
    def __init__(self,model:torch.nn.Module, data_loaders: list, validate_each: int, plot: bool,track: bool, Y_std:torch.tensor, plot_each: int = 1, S_test : int = None, inference_in_cpu = False ) -> None:
        ''' Main trainer class. All trainer classes should inherit from this one
                Args:
                        `model`        (torch.nn.Module)  :->: the model to be optimized
                        `data_loaders` (list)             :->: dataloaders [train,valid,test]. Possibilities: [train], [train,test] and [train,valid,test]
                        `validate_each`(int)              :->: evaluate validation and test each this value epochs
                        `plot`         (bool)             :->: wether to plot or not. This calls the method plot that must be overwritten
                        `track`        (bool)             :->: whether to track or not. This calls the method track that must be overwritten
                        `Y_std`        (torch.tensor)     :->: Standard desviation of your regressed variable. Used to compute performance metrics. Should have shape (Dy,)
                        `plot_each`    (int)              :->: each plot_each epochs function plot is call. Default 1
                        `S_test`       (int)              :->: Used to compute the predictive distribution of the Copula process when input dependent flows use dropout
                        `inference_in_cpu (bool)          :->: If true, then inference for computing metrics is done in CPU. This is specially usefull when S_test >>> S_train
        '''
        ## Difference previous to refactorization. Monte Carlo samples are now handled inside likelihood instance
        Trainer_GP_classification.__init__(self,model, data_loaders, validate_each, plot,track, Y_std, plot_each, S_test, inference_in_cpu = inference_in_cpu) 	

