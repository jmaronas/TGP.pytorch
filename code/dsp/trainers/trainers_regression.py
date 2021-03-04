#-*- coding: utf-8 -*-
# trainers_regression.py : Trainers methods for standard regression
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import time

# Standard
import numpy

# Torch
import torch
import torch.distributions as td

# Custom
from .trainer_base import Trainer_base
from ..utils import positive_transform, inverse_positive_transform 
from .. import config as cg


### Trainer method for the sparse GP. ###
class Trainer_GP_regression(Trainer_base):
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
                        `S_test`          (int)              :->: Used to sample from predictive distribution. Although moments can be comptued in closed form in this case. This function computes them
                                                                  with sampling so that the estimator is the same as for the TGP and DGP and reuse same code (bug safer)
                        `inference_in_cpu (bool)             :->: If true, then inference for computing metrics is done in CPU. This is specially usefull when S_test >>> S_train
        '''
        Trainer_base.__init__(self,model,data_loaders,plot, track, plot_each) 	
        self.model = model

        self.num_outputs      = self.model.out_dim
        self.is_plotting      = plot
        self.is_tracking      = track
        self.inference_in_cpu = inference_in_cpu
        self.validate_each    = validate_each

        self.ELL_arr  = []                # Keeps ELL of all the epochs
        self.KLD_arr  = []                # Keeps KLD of all the epochs
        self.S_test   = S_test

        self.reset_before_epoch()             # initialize self.ElogL_train etc 
        self.initialize_to_last_results()
 
        assert len(Y_std.shape) == 1 and Y_std.shape[0] == self.num_outputs, "Y_std isnot torch tensor of size torch.Size([{}])".format(self.num_outputs)
        self.Y_std = Y_std

    def initialize_to_last_results(self):
        self.to_last_results = [self.ELogL_train, self.rmse_train, self.coverage_train, self.ELogL_valid, self.rmse_valid, self.coverage_valid, self.ELogL_test, self.rmse_test, self.coverage_test] # save initial values for metrics.

    def reset_before_epoch(self):
        self.tot_batch, self.tot_train, self.tot_test, self.tot_valid = [0.0]*4 # summarized statistics of the datasets
        self.acc_loss, self.acc_ell, self.acc_kld = [0.0]*3 # to monitor the elbo

        ##  == Performance measures == ##

        # Expected predictive log likelihood
        self.ELogL_train = numpy.zeros((self.num_outputs)) 
        self.ELogL_valid = numpy.zeros((self.num_outputs)) 
        self.ELogL_test  = numpy.zeros((self.num_outputs))

        # Root mean squared error
        self.rmse_train   = numpy.zeros((self.num_outputs))
        self.rmse_valid   = numpy.zeros((self.num_outputs)) 
        self.rmse_test    = numpy.zeros((self.num_outputs)) 

        # Coverage error
        self.coverage_train = numpy.zeros((self.num_outputs))
        self.coverage_valid = numpy.zeros((self.num_outputs)) 
        self.coverage_test  = numpy.zeros((self.num_outputs)) 

        # Monitor Time per epoch
        self.current_t = time.time()

    def ELBO_call(self,x,y):

        loss,elogl,kld = self.model.ELBO(x,y)
        loss *= -1 # negative because we minimize

        self.ELL_arr.append(elogl.item())
        self.KLD_arr.append(kld.item())
        
        self.param_elbo =  [loss,elogl,kld] # save this to be used by self.print_train_summary
        return loss

    def print_train_summary(self,epochs,ep):
        loss,_,_ = self.param_elbo
        print('| Epoch [{}/{}] Iter[{:.0f}/{:.0f}]\t\t Loss: {:.3f}'.format(ep+1,epochs,self.tot_batch,len(self.train_loader),loss.item()),end="\r")

    def train_metrics(self,X):
        self.tot_batch += 1
        self.tot_train += X.size(0)
        
        # elbo stats
        loss,elogl,kld = self.param_elbo
        self.acc_loss += loss.item()*-1 # negate ELBO to display it increasing 
        self.acc_ell += elogl.item()
        self.acc_kld += kld.item()

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
        Y_std = self.Y_std

        logL_train     = self.ELogL_train / float(tot_train)
        rmse_train     = Y_std.to('cpu') * numpy.sqrt(self.rmse_train / float(tot_train))
        coverage_train = self.coverage_train / float(tot_train)

        logL_valid     = self.ELogL_valid
        rmse_valid     = self.rmse_valid
        coverage_valid = self.coverage_valid
        if self.is_valid:
            logL_valid     /= float(tot_valid)
            rmse_valid     = Y_std.to('cpu') * numpy.sqrt(rmse_valid / float(tot_valid))
            coverage_valid /= float(tot_valid)

        logL_test     = self.ELogL_test
        rmse_test     = self.rmse_test
        coverage_test = self.coverage_test
        if self.is_test:
            logL_test /= float(tot_test)
            rmse_test  =  Y_std.to('cpu') * numpy.sqrt(rmse_test / float(tot_test))
            coverage_test /= float(tot_test)

        return logL_train.item(), rmse_train.item(), coverage_train.item(), logL_valid.item(), rmse_valid.item(), coverage_valid.item(), logL_test.item(), rmse_test.item(), coverage_test.item()

    def performance_metrics(self,X,Y, dataset):

        Y_std = self.Y_std

        if self.inference_in_cpu:
            current_device = cg.device
            cg.device      = 'cpu'
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
            Y_std = Y_std.to(cg.device)
            X = X.to(cg.device)
            Y = Y.to(cg.device)

        log_p_y_f, predictive_params = self.model.test_log_likelihood(X, Y, return_moments = True, Y_std = Y_std, S_MC_NNet = None)
        pos_pred_samples,_,_         = self.model.sample_from_predictive_distribution(X, S = self.S_test) # this could be replaced by taking the quantiles directly. However in this case all the 
                                                                                                          # models in this work share _performance_metric function 

        self._performance_metrics(X,Y,dataset, log_p_y_f, predictive_params, pos_pred_samples)

        if self.inference_in_cpu:
            cg.device      = current_device
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
 
    def _performance_metrics(self,X,Y,dataset, log_p_y_f, predictive_params, pos_pred_samples):

        quantiles95 = numpy.quantile(pos_pred_samples.to('cpu').numpy(),[0.025,0.975], axis = 1) # move to torch.quantile when updating to pytorch 1.7

        if dataset == 'train':
            ElogL , rmse, coverage = self.ELogL_train, self.rmse_train, self.coverage_train

        elif dataset == 'valid':
            ElogL , rmse, coverage = self.ELogL_valid, self.rmse_valid, self.coverage_valid
            self.tot_valid += X.size(0)

        elif dataset == 'test':
            ElogL , rmse, coverage = self.ELogL_test, self.rmse_test, self.coverage_test
            self.tot_test += X.size(0)

        # Compute performance measures and accumulate them
        my,Ky = predictive_params
        for idx, (lpy, m) in enumerate(zip(log_p_y_f,my)):
            ElogL[idx] += lpy.to('cpu').numpy()
            m = torch.squeeze(m)
            rmse[idx]   += ((m-Y[:,idx])**2).sum()

            q = quantiles95[:,idx,:].squeeze(-1)
            y = Y[:,idx]
            low = y >= torch.tensor(q[0,:])
            up  = y <= torch.tensor(q[1,:])
            vec = torch.logical_and(low,up).float()
            coverage[idx] += vec.sum()

        # Accumulate in its accumulators
        if dataset == 'train':
            self.ELogL_train    = ElogL 
            self.rmse_train     = rmse
            self.coverage_train = coverage
        
        elif dataset == 'valid':
            self.ELogL_valid    = ElogL 
            self.rmse_valid     = rmse
            self.coverage_valid = coverage

        elif dataset == 'test':
            self.ELogL_test    = ElogL 
            self.rmse_test     = rmse
            self.coverage_test = coverage

       
    def stats_summary(self,lr,epochs,ep):

        if ((ep+1)%self.validate_each) == 0: 
            Y_std = self.Y_std
            # Performance Metrics
            logL_train     = self.ELogL_train / float(self.tot_train)
            rmse_train     = Y_std.to('cpu') * numpy.sqrt(self.rmse_train / float(self.tot_train))
            coverage_train = self.coverage_train / float(self.tot_train)

            logL_valid     = self.ELogL_valid
            rmse_valid     = self.rmse_valid
            coverage_valid = self.coverage_valid 
            if self.is_valid:
                logL_valid     /= float(self.tot_valid)
                rmse_valid     = Y_std.to('cpu') * numpy.sqrt(rmse_valid / float(self.tot_valid))
                coverage_valid /= float(self.tot_valid)

            logL_test     = self.ELogL_test
            rmse_test     = self.rmse_test
            coverage_test = self.coverage_test
            if self.is_test:
                logL_test     /= float(self.tot_test)
                rmse_test     =  Y_std.to('cpu') * numpy.sqrt(rmse_test / float(self.tot_test))
                coverage_test /= float(self.tot_test)

            # Save the last computed results.
            self.to_last_results = [logL_train, rmse_train, coverage_train, logL_valid, rmse_valid, coverage_valid, logL_test, rmse_test, coverage_test]


        # Training Metrics
        acc_loss = float(self.acc_loss)/float(self.tot_train)
        acc_ell  = float(self.acc_ell)/float(self.tot_train) 
        acc_kld  = float(self.acc_kld)/float(self.tot_train)

        string_print_common =  "\n|| Epoch [{}/{}] took {:.5f} minutes LR: {:.4f} \t Loss ==>> ELBO: {:.5f} ELL: {:.5f} KLD: {:.5f} \n \t == STATS == \n".format(
                                        ep+1,epochs,(time.time()-self.current_t)/60., lr, acc_loss, acc_ell, acc_kld)

        string_print_common +=  "\t\t Monitor Model Parameters: Likelihood noise std  "
        for idx,out_id in enumerate(self.model.likelihood.log_var_noise):
            noise = positive_transform(out_id.detach()).sqrt().item()
            string_print_common +=  "Output " + str(idx)+": {:.5f}   ".format(noise)	
            if self.model.likelihood.noise_is_shared:
                break

        string_print_common += "\n"

        string_train, string_valid, string_test = [""]*3

        if ((ep+1)%self.validate_each) == 0: 
            string_train        =  "\t => Train: total samples {} \t".format(self.tot_train) 
            for out_id, (ell,rmse,coverage) in enumerate(zip(logL_train,rmse_train,coverage_train)):
                string_train += "Output " + str(out_id) + ":  LOGL {:.5f}".format(ell) + "  RMSE {:.5f}".format(rmse) + "  COVERAGE {:.5f}".format(coverage) + "\t"
            string_train += "\n"
                    
            string_valid        =  "\t => Valid: total samples {} \t".format(self.tot_valid) 
            for out_id, (ell,rmse,coverage) in enumerate(zip(logL_valid,rmse_valid,coverage_valid)):
                string_valid += "Output " + str(out_id) + ":  LOGL {:.5f}".format(ell) + "  RMSE {:.5f}".format(rmse) + "  COVERAGE {:.5f}".format(coverage) + "\t"
            string_valid += "\n"

            string_test         =  "\t => Test: total_samples {}  \t".format(self.tot_test) 
            for out_id, (ell,rmse,coverage) in enumerate(zip(logL_test,rmse_test,coverage_test)):
                string_test += "Output " + str(out_id) + ":  LOGL {:.5f}".format(ell) + "  RMSE {:.5f}".format(rmse) + "  COVERAGE {:.5f}".format(coverage) + "\t"
                    
        string_test += "\n"

        print(string_print_common + string_train + string_valid + string_test)

    def get_last_results(self) -> list:
        logL_train, rmse_train, coverage_train, logL_valid, rmse_valid, coverage_valid,  logL_test, rmse_test, coverage_test = self.to_last_results
        return logL_train.item(), rmse_train.item(), coverage_train.item(), logL_valid.item(), rmse_valid.item(), coverage_valid.item(), logL_test.item(), rmse_test.item(), coverage_test.item()		


### === Trainer for sparse SP regression === ###
class Trainer_SP_regression(Trainer_GP_regression):
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
                        `inference_in_cpu (bool)             :->: If true, then inference for computing metrics is done in CPU. This is specially usefull when S_test >>> S_train
        '''
        ## Difference previous to refactorization. Monte Carlo samples are now handled inside likelihood instance
        Trainer_GP_regression.__init__(self,model, data_loaders, validate_each, plot,track, Y_std, plot_each, S_test, inference_in_cpu = inference_in_cpu) 	

    def performance_metrics(self,X,Y, dataset):

        Y_std = self.Y_std

        if self.inference_in_cpu:
            current_device = cg.device
            cg.device      = 'cpu'
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
            Y_std = Y_std.to(cg.device)
            X = X.to(cg.device)
            Y = Y.to(cg.device)

        log_p_y_f, predictive_params = self.model.test_log_likelihood(X, Y, return_moments = True, Y_std = Y_std, S_MC_NNet = self.S_test)
        pos_pred_samples,_,_         = self.model.sample_from_predictive_distribution(X, S = self.S_test) 

        self._performance_metrics(X,Y,dataset, log_p_y_f, predictive_params, pos_pred_samples)

        if self.inference_in_cpu:
            cg.device      = current_device
            self.model.to(cg.device)
            self.model.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))
 


