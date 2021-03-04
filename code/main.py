#-*- coding: utf-8 -*-
# Author: Juan Maroñass

# Python
import os
import sys
sys.path.extend(['../../../'])

# Libraries
import numpy
import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=40)
plt.rc('ytick',labelsize=40)
import hashlib
import datetime
import argparse

# Torch
import torch

# Gpytorch
import gpytorch as gpy

# Custom
import dsp
import dsp.config as cg
from dsp.models import instance_kernel, sparse_MF_SP, sparse_MF_GP
from dsp.models.flow import instance_flow
from dsp.likelihoods import GaussianNonLinearMean, GaussianLinearMean
from dsp.data import return_dataset
from dsp.utils import KMEANS
from dsp.initializers import find_forward_params_input_dependent_flow, find_forward_params
from dsp.trainers import Trainer_SP_regression

from exp_config import return_hyperparams
from exp_utils import return_flow_architecture


## ============================= ##
## ===== PARSER DEFINITION ===== ##
def parse_args():

    parser = argparse.ArgumentParser( description = 'TGP in PyTorch' )

    ## == Model == ##
    parser.add_argument('--model', required = True, help = 'Which model to run: ID_TGP, TGP or SVGP')

    # == Dataset Arguments == #
    parser.add_argument('--dataset', required = True, help='Dataset to use', choices = ['boston', 'power'],type = str)
    parser.add_argument('--train_test_seed_split', required = True, help='Train test random partition for UCI',type = int)

    ## == Common to all models == #
    parser.add_argument('--num_inducing'     , required = True, type = int,                   help='Number of inducing points to run' )

    # == Config Parsing == #
    args = parser.parse_args()
    return args

args = parse_args()


## =================================== ##
## =================================== ##
## ====== EXPERIMENT DEFINITION ====== ## 
## =================================== ##
## =================================== ##

num_valid_points = {'boston': 100, 'energy': 150, 'concrete': 200, 'kin8nm': 1000,  'naval': 1500, 'power': 1000, 'protein': 5000,   'wine_white': 600 ,'wine_red': 300, 'airline': 500000, 'year': 100000, 'avila': 2000, 'heart' : 50, 'movement' : 1000 , 'activity' : 5000, 'banknote' : 200 }


## ========================
## == DATASET DEFINITION ==
batch_size        = 10000                                   # batch size
dataset_name      = args.dataset                            # dataset 

train_test_seed    = args.train_test_seed_split # train test split

## ======================
## ======================
## == MODEL DEFINITION ==
## ======================
## ======================

## == Flow definition ==
flow_arch, num_blocks, num_steps, flow_hidden_act, flow_num_hidden_layers, flow_DR, flow_BN, flow_hidden_dim, flow_inference = return_hyperparams(args.model,args.dataset)
flow_input_dependent = True if args.model == 'ID_TGP' else False
be_fully_bayesian    = False 

## == GP Part ==
num_Z        = args.num_inducing  # number of inducing points
n_kmeans_run = 10                 # number of times kmeans is run. The best one is used for initialization of inducing points.
obs_noise        = 0.05           # observation noise from p(y|f). Not optimized.

init_params_kernel = { # set initial values for the kernel parameters based on salimbeni's code
                      'length_scale'   : 2.0,
                      'kernel_scale'   : 2.0,
                      'noisy_variance' : 1e-6,

                     }


## == Variational posterior definition ==
whiten            = True  # use whitenned representation   
init_params_model =  {    # initial parameters for the variational posterior
                        'variational_distribution' : {
                                                      'variance_scale' : 1e-5,
                                                      'mean_scale'     : 0.0
                                                     }
                     }

## ======================================== ##
## ======================================== ##
## ========== SET EXPERIMENT  ============= ##
## ======================================== ##
## ======================================== ##

##ctual Some config switch
if torch.cuda.is_available():
    cg.device = 'cuda:0'
else:
    cg.device = 'cpu'

cg.set_maximum_precission()

S_test  = 100

## Load dataset
options = { 
           'shuffle_train'   : True, 
           'split_from_disk' : True,
           'use_generator'   : True,
           'n_workers'       : 0, 
        }
data_loaders, data_config = return_dataset(dataset_name, batch_size, use_validation = None, seed = train_test_seed, options = options)

## ================== ##
## ================== ##
## Instance of the model
## ================== ##
## ================== ##

Dy = data_config['Dy']
Dx = data_config['Dx']
init_Z = KMEANS(data_config['X_tr'], num_Z, n_init = n_kmeans_run, seed = cg.config_seed) 

## =================================================== ##
## Output flow instance: common to single and multi-layer

flow_arch, num_blocks, num_steps, flow_hidden_act, flow_num_hidden_layers, flow_DR, flow_BN, flow_hidden_dim, flow_inference 

flow_rest_of_parameters = {
                           'input_dependent'  : flow_input_dependent,
                           'input_dim'        : Dx,
                           'num_hidden_layers': flow_num_hidden_layers,
                           'batch_norm'       : flow_BN,
                           'dropout'          : flow_DR,
                           'hidden_dim'       : flow_hidden_dim,
                           'hidden_activation': flow_hidden_act,
                           'inference'        : flow_inference
                          }
## = Flow instance
run_initializer = False
if args.model != 'SVGP':
    flow_specs, random_flow_fn, run_initializer = return_flow_architecture(flow_arch, num_blocks, num_steps, flow_rest_of_parameters)

## = Initializer run if flow need it
if run_initializer:
    print("=====================================================")
    print("=====================================================")
    print("==== RUNNING INITIALIZER TO MAKE THE FLOW LINEAR ====")
    print("=====================================================")
    print("=====================================================")

    tr_ds = data_loaders[0].dataset
    optimizer_fn = None 
    min_x = tr_ds.Y.min()-1
    max_x = tr_ds.Y.max()+1
    x_input = numpy.linspace(min_x,max_x, 5000 )
    y_output = x_input.copy()

    eps_init = 2000

    ## Make the flow the identity mapping over the output training range given by [ Y.min() , Y_max() ]
    T_flow,MSE_loss = find_forward_params(x_input, y_output, random_flow_fn, num_restarts = 1, num_epochs = eps_init, optimizer_fn = optimizer_fn, verbose = True, verbose_level = 1)
  
    if numpy.any(numpy.isnan( numpy.array(MSE_loss) )) :
        raise RuntimeError("Got MSE loss to Nan on the flow initializer.")

    flow_specs = T_flow

flow_init_eps = None
if flow_input_dependent:
    
    flow_init_eps = 2000

    if not run_initializer:
        T_flow = instance_flow(flow_specs) # we instance the flow
    else:
        T_flow.input_dependent = True # random_flow_fn set it to False to run the initializer


    train_loader = data_loaders[0]
    ## This function makes the NN parameters match those that make the flow be the identity mapping. It is done by minimizing sum of squared errores between NNet(X) and parameter, where X is a minibatch
    ## of samples from the dataloader
    T_flow,_ = find_forward_params_input_dependent_flow(train_loader, FLOW = T_flow, num_epochs = flow_init_eps, noise_var = 0.0)

    flow_specs = T_flow

## ===================
## Instance likelihood
## Regression
Obs_noise_shared = False
if args.model == 'SVGP':
    likelihood = GaussianLinearMean(out_dim = Dy, noise_init = obs_noise, noise_is_shared = Obs_noise_shared)
else:
    likelihood = GaussianNonLinearMean(out_dim = Dy, noise_init = obs_noise, noise_is_shared = Obs_noise_shared, quadrature_points = cg.quad_points )


K_is_shared         = False
mean_is_shared      = False
K_is_shared         = False
Z_is_shared         = False
Q_is_shared         = False
noise_power         = 0.0
GP_per_hidden_layer = None

# instance kernel object
K = instance_kernel('scale_rbf',ard_num_dim = Dx, num_multioutput = Dy, kernel_is_shared = K_is_shared, init_params = init_params_kernel )

if args.model == 'SVGP': 
    model = sparse_MF_GP(  model_specs        = ['zero',K], 
                           X                  = data_config['X_tr'],
                           init_Z             = init_Z,
                           N                  = data_config['N_tr'],
                           likelihood         = likelihood,
                           num_outputs        = Dy, 
                           is_whiten          = whiten,
                           K_is_shared        = K_is_shared, 
                           mean_is_shared     = mean_is_shared, 	
                           Z_is_shared        = Z_is_shared, 
                           q_U_is_shared      = Q_is_shared , 
                           add_noise_inducing = noise_power,
                           init_params        = init_params_model
                        )
else:
    model = sparse_MF_SP(  model_specs        = ['zero',K], 
                           X                  = data_config['X_tr'],
                           init_Z             = init_Z,
                           N                  = data_config['N_tr'],
                           likelihood         = likelihood,
                           num_outputs        = Dy, 
                           is_whiten          = whiten,
                           K_is_shared        = K_is_shared, 
                           mean_is_shared     = mean_is_shared, 	
                           Z_is_shared        = Z_is_shared, 
                           q_U_is_shared      = Q_is_shared , 
                           flow_specs         = [flow_specs],
                           flow_connection    = 'single',
                           add_noise_inducing = noise_power,
                           init_params        = init_params_model,
                           be_fully_bayesian  = be_fully_bayesian
                        )
model.to(cg.device)

## ==============================
## == OPTIMIZER SPECIFICATIONS ==
global_eps    = 15000
validate_each = 1e20    # how many epochs do we wait to compute validation metrics. Set to -1 so it is not done

lr  = 0.01

## set weight decay for input dep flows
specifications_1 = [[]]
percentages_1    = [1.0]
if flow_input_dependent:
    wd_nnets = 1e-5

    # Get the instanced parameters that are not NNets and set them the same learning rate. The optimizar will group them
    flow_scheduler = []
    for n,p in model.named_parameters():
        if 'G_matrix' in n:
            if 'NNets' not in n:
                flow_scheduler.append([lr,n])

    flow_scheduler.append([lr,wd_nnets,'NNets'])

    specifications_1[0].extend(flow_scheduler)

## ============================================ ##
## ======   Train/Evaluate the model     ====== ##
Y_std   = (torch.ones((Dy,))*data_config['Y_std']).to(cg.device) # Shape (Dy,)

trainer = Trainer_SP_regression(model = model, data_loaders = data_loaders, validate_each = validate_each, plot = False, track = False, Y_std = Y_std, plot_each = -1, S_test = S_test, inference_in_cpu = True)

## the parameters not specified through optimization_schedule get the lr_ALL 
optimisation_schedule = (percentages_1, specifications_1)
trainer.train(epochs = global_eps, lr_ALL = lr, opt = 'adam', keep_parameter_groups = True, optimisation_schedule = optimisation_schedule, lr_groups = None)

## Compute LogL rmse and overage
logL_train, rmse_train, coverage_train, logL_valid, rmse_valid, coverage_valid, logL_test, rmse_test, coverage_test = trainer.compute_metrics()

print("\n"*2)
print("=====================================")
print("== Results obtained after training ==")
print("=====================================")

if args.model == 'ID_TGP':

    print("Dataset {}, num inducing points {}, POINT ESTIMATE FLOW , Test Negative LOGL {:.3f}, Test RMSE {:.3f}".format(args.dataset,args.num_inducing,-1*logL_test,rmse_test))

    ## ======================================================
    ## compute the predictions integrating out Bayesian flows.
    ## Note this just requires using the Monte carlo version of Dropout, hence we use the same model already trained

    ## Set bayesian flag up
    model.be_fully_bayesian(True)

    ## compute predictions again
    logL_train, rmse_train, coverage_train, logL_valid, rmse_valid, coverage_valid, logL_test, rmse_test, coverage_test = trainer.compute_metrics()

    print("Dataset {}, num inducing points {}, BAYESIAN FLOW , Test Negative LOGL {:.3f}, Test RMSE {:.3f}".format(args.dataset,args.num_inducing,-1*logL_test,rmse_test))

else:
    print("Dataset {}, num inducing points {}, model {}, Test Negative LOGL {:.3f}, Test RMSE {:.3f}".format(args.dataset,args.num_inducing,args.model,-1*logL_test,rmse_test))

exit()


