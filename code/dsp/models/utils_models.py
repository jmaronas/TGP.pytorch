#-*- coding: utf-8 -*-
# utils_models.py : this file contain some utilities to be shared by the models module
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import sys
sys.path.extend(['../'])

import typing
from typing import List
# Standard
import numpy 
import numpy as np

# Torch
import torch
import torch.nn as nn

# Gpytorch
import gpytorch as gpy
from gpytorch.means import MultitaskMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, AdditiveKernel, ProductKernel, MaternKernel

# Custom
from .. import config as cg
from .kernels import WhiteNoiseKernel
from .means import Linear, Identity



## ================================== ##
## ====== Confidence Intervals ====== ##
def confidence_intervals(model, X:torch.tensor, intervals:list, S:int, distribution: str, is_deep:bool) -> List[List[np.ndarray]]:
    """
        Confidence intervals are computed point wise and just calculate the quantiles at each point
        Returns:
            A list over each output. Each element is itself a list ove each of the intervals.
            
    """

    #imports are here to fix circular imports
    from ..models import sparse_MF_GP
    from ..likelihoods import WarpedGaussianLinearMean

    found_intervals = []
    with torch.no_grad():

        if (type(model) is sparse_MF_GP) and (type(model.likelihood) is WarpedGaussianLinearMean):
            print('special case')
            if distribution == 'posterior':
                raise NotImplementedError()
            elif distribution =='predictive':

                if model.out_dim > 1:
                    raise NotImplementedError()

                N = X.shape[0]

                if intervals != [0.025, 0.5, 0.975]:
                    raise NotImplementedError()

                #manually get moments for q(f) to avoid sampling in model.predictive_distribution
                X = X.repeat(model.out_dim,1,1)
                mean_q_f, cov_q_f = model.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = False, init_Z = None)

                #returns moments of p(Y_0, | f)
                m_Ys,K_Ys  = model.likelihood.unwarped_marginal_moments(mean_q_f.squeeze(dim = 2), cov_q_f.squeeze(dim = 2), diagonal = True)

                def warp(x):
                    return model.likelihood.flow[0].inverse(x)

                median = warp(m_Ys)
                lower_band = warp(m_Ys-1.96*torch.sqrt(K_Ys))
                upper_band = warp(m_Ys+1.96*torch.sqrt(K_Ys))

                median = median.detach().to('cpu').numpy()
                lower_band = lower_band.detach().to('cpu').numpy()
                upper_band = upper_band.detach().to('cpu').numpy()

                median = np.reshape(median, [N, 1])
                lower_band = np.reshape(lower_band, [N, 1])
                upper_band = np.reshape(upper_band, [N, 1])

                found_intervals = [lower_band, median, upper_band] 
                #assuming one task
                found_intervals = [found_intervals]

            else:
                raise NotImplementedError()

        else:
            #Dy x S x N x 1
            if distribution == 'predictive':
                samples = model.sample_from_predictive_distribution(X, S)

            elif distribution == 'posterior':

                if is_deep:
                    Fk_list, _, _, _ = model.sample_from_variational_posterior(X,S = S)
                    samples = Fk_list[-1]

                else:
                    MB, Dx = X.shape
                    Dy     = model.out_dim
                    samples,_,_,_ = model.sample_from_variational_marginal(X, S, diagonal=True, is_duvenaud=False, init_Z=None)
                    samples = samples.view(Dy,S,MB,1)
            else:
                raise ValueError("Invalid arguments")

            found_intervals = []
            for l_idx in range(model.out_dim):
                sample_idx = samples[l_idx].detach().to('cpu').numpy()

                arr = []
                for interval in intervals:
                    ci_i = np.quantile(sample_idx, interval, axis=0)
                    arr.append(ci_i)

                found_intervals.append(arr)

    return found_intervals

def compute_95_and_median_confidence_intervals(model, X:torch.tensor, S:int, distribution: str, is_deep: bool) -> torch.tensor:
    """
        Args:
            model        : is an object of one of the classes defined in dsp/models.
            X            : input to compute the confidence intervals at : NxDx
            S            : number of samples to compute the confidence at
            distribution : choices (posterior or predictive). Computes confidence interval for q(f_k) or q(y*|x*)
            is_deep      : whether model is a deep model (DGP) or a shallow model

        Returns the [0.025, 0.5, and 0.975] confidence intervals  at location X using S samples
            0.5 refers to the median
            0.025, 0.975 are the \pm 95% confidence intervals

        Note: is_deep should be removed in a future. The point is that sampling from the posterior in a shallow model and deep model is done with different methods. We should
        provide a common interface for sampling for both shallow and deep model so that we can sample from the posterior calling the same method
    """
    assert distribution in ['posterior','predictive'], "Invalid arguments for distribution. Got {}, expected one of [posterior, predictive]".format(distribution)
    return confidence_intervals(model, X, [0.025, 0.5, 0.975], S, distribution, is_deep)

## ============================= ##
## ====== Instance Kernel ====== ##

def instance_kernel(name: str, ard_num_dim : int, num_multioutput:int, kernel_is_shared:bool, init_params : dict = {}, kernels : list = None) -> gpy.kernels :
    '''
    This function acts as a wrapper of gpytorch kernels. It return instances of kernels to pass to the classes.

        Args:

            name             (str)  : string identifying the kernel
            ard_num_dim      (int)  : to perform ARD kernels. None if not required.
            num_multioutput  (int)  : number of kernels created. This is useful to perform the computation of several independent GPs in parallel
            kernel_is_shared (bool) : if true, then the kernel is shared by all the GPs. This means that num_multioutput is set to one
            init_params      (dict) : hold specific values to initialize each of the kernels
            kernels          (list) : A list of kernels used to make a composition of kernels specified through argument name (e.g name = additive)

    '''

    if ard_num_dim is not None and not isinstance(ard_num_dim,int):
        raise ValueError("ard_num_dim must be None or int, got {}".format(type(ard_num_dim)))

    ## Initialization parameters for kernels
    if 'length_scale' in init_params.keys():
        ls = init_params['length_scale']
    else:
        ls = 1.0

    if 'kernel_scale' in init_params.keys():
        ks = init_params['kernel_scale']
    else:
        ks = 1.0

    if 'noisy_variance' in init_params.keys():
        variance = init_params['noisy_variance']
    else:
        variance = 1e-9

    if kernel_is_shared:
        num_multioutput = 1

    ## Instance kernels
    if name == 'rbf':
        K = RBFKernel(ard_num_dims = ard_num_dim ,  batch_shape=torch.Size([num_multioutput]))
        K.raw_lengthscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,1,K.raw_lengthscale.size(1),dtype = cg.dtype)*ls)

            
    elif name == 'scale_rbf':
        RBF = RBFKernel(ard_num_dims = ard_num_dim , batch_shape = torch.Size([num_multioutput]))
        RBF.raw_lengthscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,1,RBF.raw_lengthscale.size(-1),dtype = cg.dtype)*ls)

        K =  ScaleKernel(RBF, batch_shape = torch.Size([num_multioutput]))
        K.raw_outputscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,dtype = cg.dtype)*ks)

    elif name == 'matern32':
        K = MaternKernel(ard_num_dims = ard_num_dim , batch_shape = torch.Size([num_multioutput]), nu=1.5)
        K.raw_lengthscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,1,K.raw_lengthscale.size(1),dtype = cg.dtype)*ls)

    elif name == 'scale_matern32':
        MATERN = MaternKernel(ard_num_dims = ard_num_dim , batch_shape = torch.Size([num_multioutput]), nu=1.5)
        MATERN.raw_lengthscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,1,MATERN.raw_lengthscale.size(1),dtype = cg.dtype)*ls)

        K =  ScaleKernel(MATERN, batch_shape = torch.Size([num_multioutput]))
        K.raw_outputscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,dtype = cg.dtype)*ks)

    elif name == 'whitenoise':
        K = WhiteNoiseKernel(variance,num_multioutput)

    elif name == 'constant_whitenoise':
        #init data such that the transformed variance is equal to variance
        K = ConstantWhiteNoiseKernel(variance,num_multioutput)

    elif name == 'additive':
        return AdditiveKernel(*kernels)

    elif name == 'spectral_mixture':
        num_components = init_params['K']
        periods = init_params['periods']
        lengthscales = init_params['length_scales']
        magnitudes = init_params['magnitudes']


        kernels = []

        for k in range(num_components):

            period = torch.tensor(periods[k])
            lengthscale = torch.tensor(lengthscales[k])
            magnitude = torch.tensor(magnitudes[k])

            kernel_per = gpy.kernels.PeriodicKernel(ard_num_dims = ard_num_dim ,  batch_shape=torch.Size([num_multioutput]))
            kernel_per.raw_period_length=nn.Parameter(gpy.utils.transforms.inv_softplus(period))

            kernel_rbf = gpy.kernels.RBFKernel(ard_num_dims = ard_num_dim ,  batch_shape=torch.Size([num_multioutput]))
            kernel_rbf.raw_lengthscale=nn.Parameter(gpy.utils.transforms.inv_softplus(lengthscale))
            kernel_rbf = gpy.kernels.ScaleKernel(kernel_rbf, batch_shape = torch.Size([num_multioutput]))
            kernel_rbf.raw_outputscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,dtype = cg.dtype)*magnitude)

            kernel_k = ProductKernel(kernel_per, kernel_rbf)

            kernels.append(kernel_k)


        return AdditiveKernel(*kernels)

    elif name == 'sm':
        num_components = init_params['K']
        periods = init_params['periods']
        lengthscales = init_params['length_scales']
        magnitudes = init_params['magnitudes']


        kernels = []

        for k in range(num_components):

            period = torch.tensor(periods[k])
            lengthscale = torch.tensor(lengthscales[k])
            magnitude = torch.tensor(magnitudes[k])

            kernel_per = gpy.kernels.CosineKernel(ard_num_dims = ard_num_dim ,  batch_shape=torch.Size([num_multioutput]))
            kernel_per.raw_period_length=nn.Parameter(gpy.utils.transforms.inv_softplus(period))

            kernel_rbf = gpy.kernels.RBFKernel(ard_num_dims = ard_num_dim ,  batch_shape=torch.Size([num_multioutput]))
            kernel_rbf.raw_lengthscale=nn.Parameter(gpy.utils.transforms.inv_softplus(lengthscale))
            kernel_rbf = gpy.kernels.ScaleKernel(kernel_rbf, batch_shape = torch.Size([num_multioutput]))
            kernel_rbf.raw_outputscale.data = gpy.utils.transforms.inv_softplus(torch.ones(num_multioutput,dtype = cg.dtype)*magnitude)

            kernel_k = ProductKernel(kernel_per, kernel_rbf)

            kernels.append(kernel_k)


        return AdditiveKernel(*kernels)


    else:
        raise NotImplemented("Not Implemented kernel. Add it to the list.")

    return K

## ============================= ##
## ======= Instance Mean ======= ##

def return_mean(name: str, input_dim : int, output_dim : int, W: torch.tensor) -> gpy.means :
    if name == 'zero' :
        # use multitask wrapper to directly handle batch_shape and keep shapes as in other mean functions
        return ZeroMean()
    elif name == 'identity':
        return Identity(W,input_dim,output_dim)
    elif name == 'linear':
        return Linear(input_dim,output_dim)
    else:
        raise NotImplemented("Not implemented, add it to the list")

## ============================= ##
## ========= PCA Matrix ======== ##

def return_projection_matrix(input_dim: int, output_dim: int, X: torch.tensor) -> torch.tensor : 

    X = X.detach().to('cpu').numpy()
    if output_dim == input_dim:	
        # Identity Mapping
        W = numpy.eye(input_dim)
    elif output_dim > input_dim:
        # Identity Mapping + Zero PAD
        W = numpy.concatenate([numpy.eye(input_dim), numpy.zeros((input_dim, output_dim - input_dim ))], 1) 
    elif output_dim < input_dim:
        # PCA
        _, _, V = numpy.linalg.svd(X, full_matrices=False)
        W = V[:output_dim, :].T
    else:
        raise NotImplemented("Check what happened")

    return torch.tensor(W,dtype=cg.dtype)


## ================================= ##
## ======== Propagate X ============ ##
## Propagate locations X through hierarchy
## mainly used for inducing points initialization

def propagate_inducing( model: nn.Module, propagate_Z_as: str, Z: torch.tensor, out_dim: int ) -> torch.tensor:

    in_dim = Z.size(-1)
    if propagate_Z_as == 'salimbeni':
        W = return_projection_matrix( in_dim, out_dim, Z ) 
        # 1. If in_dim == out_dim Z_L = Z_[L-1]
        # 2. If in_dim >  out_dim Z_L = PCA( Z_[L-1] )
        # 3. If in_dim <  out_dim  
        Z_out = torch.mm(Z, W)

    elif propagate_Z_as == 'sampling':
        Z_out = model.sample_prior(Z,N = 1)[0] # sample from the previous GP at Z locations and initialize using function values at those locations
        Z_out = Z_out.squeeze(-1).t()

    elif propagate_Z_as == 'salimbeni+sampling':
        # the problem of salimbeni propagation when out_dim > in_dim is that some inducing points are initalized to zero. 
        # What we do is change that zero for a function sample

        W = return_projection_matrix( in_dim, out_dim, Z )
        # 1. If in_dim == out_dim Z_L = Z_[L-1]
        # 2. If in_dim >  out_dim Z_L = PCA( Z_[L-1] )
        # 3. If in_dim <  out_dim  
        Z_out = torch.mm(Z, W)

        if out_dim > Z.size(1):
            Z_aux = model.sample_prior(Z, N = 1)[0] # sample from the previous GP at Z locations and initialize using function values at those locations
            Z_aux = Z_aux.squeeze(-1).t()
            Z_out[:,in_dim:out_dim] = Z_aux[:, in_dim:out_dim]
    
    return Z_out


## ======================================== ##
## ==== Enables Dropout for MC dropout ==== ##
## credit for pytorch forum https://discuss.pytorch.org/t/using-nn-dropout2d-at-eval-time-for-modelling-uncertainty/45274
def enable_eval_dropout(modules):
    is_dropout = False
    for module in modules:
        if 'Dropout' in type(module).__name__:
            module.train()
            is_dropout = True
    return is_dropout



