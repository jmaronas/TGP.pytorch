#-*- coding: utf-8 -*-
# trainers_base.py : Base class for all the trainers
# Author: Juan Maroñas and  Ollie Hamelijnck

## python 
from collections import OrderedDict
import sys
sys.path.extend(['../'])

## Torch
import torch

## custom
from .. import config as cg
from .optimizers import return_optimizer

import warnings


### Based class to instance a trainer method ###
class Trainer_base(object):
    def __init__(self,model:torch.nn.Module, data_loaders: list, plot: bool,track: bool, plot_each: int = 1 ) -> None:
        ''' Main trainer class. All trainer classes should inherit from this one
                Args:
                        `model`        (torch.nn.Module)  :->: the model to be optimized
                        `data_loaders` (list)             :->: dataloaders [train,valid,test]. Possibilities: [train], [train,test] and [train,valid,test]
                        `plot`         (bool)             :->: whether to plot or not. This calls the method plot that must be overwritten
                        `track`        (bool)             :->: whether to call the track method plot or not. This calls the method track that must be overwritten
                        `plot_each`    (int)              :->: each plot_each epochs function plot is call. Default 1. Similar to the validate_each argument from the
                                                               trainer method.
        '''
        self.model = model                # the model to optimize
 
        self.is_train = False             # has train set
        self.is_test = False              # has test set
        self.is_valid = False             # has validation set
        self.optimizer_is_freezed = False # whether the optimizer is reset or not
        self.optimizer = None             # optimizer
        self.is_plotting = plot           # if true, then call the plot function after each gradient update
        self.is_tracking = track
        self.plot_each   = plot_each      # each how many epochs we call the plot function
        self.to_last_results = 0.0        # initialize the list where the results of the evaluation are stored


        self.total_trainer_epochs = 0     # Keeps all the epochs run with an instance of each trainer, i.e it accumulates epoch within subsequent calls to self.train()
        self.local_trainer_epochs = -1    # Keeps all the epochs of a single call to self.train() 
        self.loss_arr = []                # Keeps the ELBO of all the epochs in total_trainer_epochs

        ## data loaders -> this might be changed if we only want to do testing. For the moment is okei
        if len(data_loaders) == 3:
                self.train_loader,self.valid_loader,self.test_loader = data_loaders
                self.is_train, self.is_valid, self.is_test = [True]*3

        elif len(data_loaders) == 2:
                self.train_loader,self.test_loader = data_loaders
                self.is_train, self.is_test = [True]*2
        else:
                self.train_loader = data_loaders[0]
                self.only_train = True

    def reset_before_epoch(self):
        ''' Reset all the variables that must be reset before each epoch. So place any variable you use to compute any metric, monitor time etc
            Set these variables as attributes of the model that will be use in subsequent methods like print_train_summary or stats_summary
        '''
        raise NotImplementedError("reset_before_epoch must be overwritten")

    def plot(self):
        ''' This function must be overwritten if you wish to plot anything. For example for toy problems. It is call if self.is_plotting = True'''
        raise NotImplementedError("plot must be overwritten")

    def track(self):
        ''' This function must be overwritten. This can be used to track anything after each gradient update e.g parameter values, norms of the gradients.'''
        raise NotImplementedError("track must be overwritten")

    def track_global(self):
        ''' This function must be overwritten. This can be used to track anything after each epoch (i.e after seeing all the dataset). '''
        raise NotImplementedError("track must be overwritten")

    def ELBO_call(self,x,y):
        ''' This function computes the ELBO
                -> Inputs must be the training input X and the targets Y
                -> Returns the loss to be optimized. Must register anything from the training algorithm you want to monitor/print to stdout. This will be then used by other methods as print_train_summary
        '''
        raise NotImplementedError("ELBO_call must be overwritten")

    def performance_metrics(self,X,Y,dataset):
        ''' Computes performance metrics, e.g RMSE for regression or accuracy for classification. It can keep the computed metrics in self.to_last_results so that the last computation of performance metrics is available through the method self.get_last_results.
                X -> Inputs
                Y -> Outputs
                dataset -> which dataset is being evaluated: train, test, valid
        '''
        raise NotImplementedError("performance_metrics must be overwritten")

    def print_train_summary(self,epochs,ep):
        ''' Used to print after each batch is processed. Might used whatever is kept in the ELBO_call method. It is call after each gradient update '''
        raise NotImplementedError("print_train_summary must be overwritten")	
                
    def stats_summary(self,lr,epochs,ep):
        ''' Used to plot any summary after each epoch (loss, training/validatoin/test metrics etc)'''
        raise NotImplementedError("stats_summary must be overwritten")

    def get_last_results(self):
        ''' If called, then it returns the last results kept in self.to_last_results '''
        raise NotImplementedError("get_last_results must be overwritten")

    def set_optimized_params(self,optimisation_step,epochs,global_lr):
        ''' Used to set which parameters are optimized '''
        model = self.model
        step_epochs = int(epochs*optimisation_step[0])
        parameters_to_process = optimisation_step[1]

        total_parameters = []
        if parameters_to_process != None:
            # Create a list containing all the parameters with the learning rates that are going to use
            for k,v in parameters_to_process.items():
                if len(total_parameters) == 0:
                    total_parameters.append({'params':[],'lr': v['lr'], 'weight_decay':v['weight_decay'] })
                    continue

                is_added = 0
                for d in total_parameters:
                    if v['lr'] == d['lr'] and v['weight_decay'] == d['weight_decay']:
                        is_added = 1
                if not is_added:
                    total_parameters.append({'params': [], 'lr': v['lr'], 'weight_decay': v['weight_decay']})

        # Add the parameters to the corresponding lists
        rest_of_parameters = [] # this keeps the parameters not specified

        for n,p in model.named_parameters():

            if self.optimizer_is_freezed:
                if n in self.names_parameters_added:
                    continue
                    # We cannot raise and error because we will always loop over all named_parameters()
                    # This means that we silently ignore if a parameter is specified twice in the specs.
                    # raise ValueError("The parameter {} has already been added to the optimizer".format(n))

            if parameters_to_process is None:
                print('Learning Rate Global \t Parameter {} '.format(n))
                rest_of_parameters.append(p)
                if self.optimizer_is_freezed: # This has to be done hear to avoid adding to this list the lr = 0.0 parameters
                    self.names_parameters_added.append(n) # add the parameter to be tracked later
                continue
        
            is_added = 0
            for k,v in parameters_to_process.items():

                p_list = v['params']
                lr_a   = v['lr']
                wd_a   = v['weight_decay']

                if n in p_list:
                    for idx,dicts in enumerate(total_parameters):
                        if lr_a == 0.0: # when the learning rate is 0.0 the parameter is not added to the rest_of_parameters list
                            is_added = 1
                            break # once the parameter is added to its corresponding learning rate there is no need to loop over the entire total_parameter list as each position
                                  # corresponds to a dict containing the parameters for a learning rate.
                        if dicts['lr'] == lr_a and dicts['weight_decay'] == wd_a:
                            print('Learning Rate {} weight decay {} \t Parameter {} '.format(dicts['lr'],dicts['weight_decay'],n))
                            total_parameters[idx]['params'].append(p) 
                            is_added = 1
                            if self.optimizer_is_freezed: # This has to be done hear to avoid adding to this list the lr = 0.0 parameters
                                self.names_parameters_added.append(n) # add the parameter to be tracked later
                            break

            if not is_added:
                print('Learning Rate Global \t Parameter {} '.format(n))
                if self.optimizer_is_freezed: # This has to be done hear to avoid adding to this list the lr = 0.0 parameters
                    self.names_parameters_added.append(n) # add the parameter to be tracked later
                rest_of_parameters.append(p)

        if len(rest_of_parameters) != 0.0:
            total_parameters.append({'params' : rest_of_parameters, 'lr' : global_lr, 'weight_decay' : 0.0})

        ## remove the parameter with the list with rate = 0.0 
        Tot_Param = []
        for el in total_parameters:
            if el['lr'] != 0.0:
                Tot_Param.append(el)

        if self.optimizer_is_freezed:
            # Add the new parameter groups to the parameter_groups list
            self.parameter_groups_added.extend(Tot_Param)

        return Tot_Param,step_epochs

    def parse_optimizer_specs(self,percentages,specifications):
        ''' Parse the arguments to the train method specifying optimization schedule '''

        model = self.model
        optimisation_schedule = []

        if sum(percentages) != 1.0:
            if cg.strict_flag:
                raise ValueError("percentages must sum 1, got {}".format(sum(percentages)))
            else:
                warnings.warn("percentages must sum 1, got {}".format(sum(percentages)))

        if len(percentages) != len(specifications):
            raise ValueError("Percentages and specifications must have same length")

        # this is only performed when the optimizer is not reset and 
        # acts as a sanity check for the user so that the we do not silently ignore
        # when two set of parameters are specified.
        # only if self.optimizer_is_freezed  = True
        is_repeated_parameter = []

        for per,specs in zip(percentages,specifications):
            if specs == None:
                optimisation_schedule.append([ per, None ])
                continue

            dict_params = OrderedDict() # so the order is preserved. Starting from python3.7 {} and OrderedDict() preserves order

            for _specs in specs:
           
                if len(_specs) == 3: # proceed in this way for backward compatibility
                    lr, wd, param = _specs
                elif len(_specs) == 2:
                    lr, param = _specs
                    wd        = 0.0
                else:
                    raise ValueError("Unvalid argument optimisation_schedule. Parameters should be specified as lr, param or lr, weight_decay, param")


                if param in dict_params.keys():
                    raise ValueError("Parameter {} already added to the parameter list".format(param))

                for n,_ in model.named_parameters():

                    if param in n:

                        if self.optimizer_is_freezed and n in is_repeated_parameter:
                            raise ValueError("Got repeated parameter {}. Either set keep_parameter_groups to false or set just one specs for param {}".format(n,param))
                        if self.optimizer_is_freezed and n in self.names_parameters_added:
                            raise ValueError("Got repeated parameter {} with root {}. This parameter already belongs to a parameter group".format(n,param))

                        if lr != 0.0:
                            is_repeated_parameter.append(n)

                        if param not in dict_params.keys():
                            dict_params.update( {param: {'lr': lr, 'weight_decay': wd, 'params': []}  }  )
                        dict_params[param]['params'].append(n)

            optimisation_schedule.append([ per, dict_params ])

        return optimisation_schedule

    def train(self, epochs: int , lr_ALL: float, opt: str, keep_parameter_groups: bool, lr_groups: list = None, optimisation_schedule: list = None) -> None:
        ''' Train method
                Args:
                        `epochs`                 (int)       :->: total epochs
                        `lr_ALL`                 (float)     :->: initial learning rate and common to all the parameters if not specified through the optimization_schedule nor lr_groups
                        `opt`                    (str)       :->: optimizer.
                        `keep_parameter_groups`  (bool)      :->: If true, the optimizer is not reinstated. This means that the parameter groups that are created are kept after
                                                                  the call to self.train() has finished. This is usefull when one wants to sequentially add and finnetune 
                                                                  parameters. It is important to keep in mind that if a set of parameters are asigned to a parameter group
                                                                  there is no chance to reagrup them at the moment.
                        `lr_groups`              (list)      :->: Specifies the learning rate used for the already added groups. It must have len equal to the number of parameter
                                                                  groups already kept in the optimizer or None. If None is provided then the learning rate of all the groups is
                                                                  set to `lr_ALL`
                        `optimisation_schedule`  (list)      :->: list detailing which params to train for how many percentage of epochs of total epochs
        '''

        self.loss_arr_shot = [] # this holds the ELBO for the current call to self.train() while self.total_loss_arr holds the ELBO for all the calls to self.train()

        if optimisation_schedule is None:
            optimisation_schedule = [[1.0],[None]]

        ## Check if optimizer should be freezed:
        if keep_parameter_groups:
            self.optimizer_is_freezed = True

            if self.optimizer is None:
                self.names_parameters_added = [] # this list holds the parameters already added to the optimizer
                self.parameter_groups_added = [] # this list holds the parameter groups so that we can change learning rate 

            else:
                if lr_groups is not None:
                    if len(lr_groups) != len(self.parameter_groups_added):
                        raise ValueError("The provided `lr_groups` {} does not match the number of parameter groups in the optimizer {}".format(lr_groups,len(self.parameter_groups_added)))
                if lr_groups is None:
                    lr_groups = [lr_ALL]*len(self.parameter_groups_added)

                ## Set learning rates to the previous parameter_groups:
                for idx,lr_i in enumerate(lr_groups):
                    self.optimizer.param_groups[idx]['lr'] = lr_i

        else:
            self.optimizer_is_freezed = False
            self.optimizer = None

        ## model
        model = self.model

        ## optimizer
        per,specs = optimisation_schedule
        optimisation_schedule = self.parse_optimizer_specs(per,specs)

        ## loop
        self.local_trainer_epochs = -1
        ep = -1

        for optimisation_step in optimisation_schedule:

            if not self.optimizer_is_freezed or not isinstance(self.optimizer,torch.optim.Optimizer):
                # create the optimizer
                parameters,step_epochs = self.set_optimized_params(optimisation_step,epochs,lr_ALL)
                self.optimizer = return_optimizer( opt, parameters, lr_ALL)

            else:

                parameters,step_epochs = self.set_optimized_params(optimisation_step,epochs,lr_ALL)

                ## Add new parameter groups.
                for group in parameters:
                    self.optimizer.add_param_group(group)

            #for param_group in self.optimizer.param_groups:
            #    print(param_group)
            for epoch in range(step_epochs):
                #update total number of epochs
                ep += 1
                self.local_trainer_epochs = ep

                self.reset_before_epoch()

                for b,(x,y) in enumerate(self.train_loader): 
                    x,y = x.to(cg.device),y.to(cg.device)
                    # For a batch of size MB x has shape (MB,Dx) and y has shape (MB,Dy). Even if Dx or Dy has dimension one the array has to be a 2 dimensional array. 

                    assert len(x.shape) == 2 and len(y.shape)==2, "x.shape and y.shape must be (MB,D) and (MB,D') with D and D' being input and output dimensions"
                    self.model.set_is_training(mode = True) # set the whole model to train mode

                    ## loss
                    loss = self.ELBO_call(x,y)

                    ## stochastic gradient update
                    self.optimizer.zero_grad() # just in case
                    loss.backward()
                    self.optimizer.step()

                    ## after optimization set back to test mode
                    self.model.set_is_training(mode = False) # set the whole model to train mode

                    self.loss_arr.append(loss.item())
                    self.loss_arr_shot.append(loss.item())
                    self.total_trainer_epochs += 1
        
                    # Get some performance metrics			
                    self.train_metrics(x)
                    if ((ep+1)%self.validate_each) == 0: 
                        self.performance_metrics(x,y,dataset='train')	

                    # Print batch summary
                    self.print_train_summary(epochs,ep)

                    # Call the track function to track things after each gradient step
                    if self.is_tracking:
                        self.track()
    
                if self.is_plotting:
                    if ((ep+1)%self.plot_each) == 0:
                        self.plot()
        
                with torch.no_grad():	

                    if ((ep+1)%self.validate_each) == 0: 
                        if self.is_valid:
                            for b,(x,y) in enumerate(self.valid_loader): 
                                x,y = x.to(cg.device),y.to(cg.device)
                                self.performance_metrics(x,y,dataset='valid')

                    if ((ep+1)%self.validate_each) == 0: 
                        if self.is_test:
                            for b,(x,y) in enumerate(self.test_loader): 
                                x,y = x.to(cg.device),y.to(cg.device)
                                self.performance_metrics(x,y,dataset='test')


                self.stats_summary(lr_ALL,epochs,ep)

                # Call the track global function, which is call after each epoch.
                if self.is_tracking:
                    self.track_global()


        # After the whole training is finished set self.optimizer to None
        if not self.optimizer_is_freezed:
            self.optimizer = None
            


