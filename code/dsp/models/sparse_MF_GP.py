#-*- coding: utf-8 -*-
# sparse_MF_GP.py : file containing base class for the sparse GP model as in Hensman et al.
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import sys
sys.path.extend(['../'])

# Standard
import numpy
import matplotlib.pyplot as plt

#Typing
import typing
from typing import List, Optional

# Torch
import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence

# Gpytorch
import gpytorch as gpy
from gpytorch.variational import CholeskyVariationalDistribution

# Custom
from .. import config as cg
from ..utils import batched_log_Gaussian
from ..likelihoods import GaussianLinearMean, MulticlassCategorical, Bernoulli

# Module specific
from .config_models import get_init_params
from .utils_models import return_mean, return_projection_matrix
from .sparse_MF_SP import sparse_MF_SP


## Sparse GP Hensman et al
class sparse_MF_GP(sparse_MF_SP):
    def __init__(self,model_specs: list, X: torch.tensor, init_Z: torch.tensor, N: float, likelihood : nn.Module, num_outputs: int, is_whiten: bool, K_is_shared: bool, mean_is_shared: bool, Z_is_shared: bool, q_U_is_shared: bool , add_noise_inducing: float, init_params: dict={}) -> None:
        """
        Stochastic Variational Sparse GP Hensman et al
                Args: 
                        :attr:  `model_specs`         (list)         :->: tuple (A,B) where A and B are string representing the desired mean and covariance functions.
                                                                          For the moment all the GPs at a layer shared the functional form of these
                                `X`                   (torch.tensor) :->: Full training set (or subset) of samples used for the SVD 
                                `init_Z`              (torch.tensor) :->: initial inducing point locations
                                `N`                   (float)        :->: total training size	
                                `likelihood`          (nn.Module)    :->: Likelihood instance that will depend on the task to carry out
                                `num_outputs`         (int)          :->: number of output GP. The number of inputs is taken from the dimensionality of init_Z
                                `is_whiten`           (bool)         :->: use whitened representation of inducing points.
                                `K_is_shared`         (bool)         :->: True if the covariance of the output GPs are shared
                                `mean_is_shared`      (bool)         :->: True if the mean of the output GPs are shared
                                `Z_is_shared`         (bool)         :->: True if the inducing point locations are shared
                                `q_U_is_shared`       (bool)         :->: True if the variational distribution is shared
                                `add_noise_inducing`  (float)        :->: Standard desviation of Gaussian noise added to init_Z
                                `init_params`         (dict)         :->: Initial parameters of model. If not use the defaults in config_models.py


          # -> Some notes for deciding if whitenning or not: https://gpytorch.readthedocs.io/en/latest/variational.html#gpytorch.variational.UnwhitenedVariationalStrategy	
        """
        flow_specs      =[ [('identity',[])]  for i in range(num_outputs) ]
        flow_connection = 'single'
        super(sparse_MF_GP, self).__init__(model_specs, X, init_Z, N, likelihood, num_outputs, is_whiten, K_is_shared, mean_is_shared, Z_is_shared, q_U_is_shared, flow_specs, flow_connection, add_noise_inducing, be_fully_bayesian = False, init_params = init_params) 

    ## ====================== ##
    ## == SAMPLING METHODS == ## 
    ## ====================== ##

    def sample_from_variational_marginal(self, X: torch.tensor, S : int, diagonal: bool, is_duvenaud: bool, init_Z: torch.tensor = None) -> list:
        """ Sample from the marginal variational q(f)
        
              q(f) = \int q(f|u) q(u) du
                
                Args: 
                        `X`           (torch.tensor)  :->: Locations where the sample is drawn from. It shoud be (Dy,SMB,Dx) or (SMB,Dx). Note that for DGP MB = S*MB given S 
                                                           Monte Carlo samples
                        `S`           (int)           :->: Number of samples to draw. Specify S = 1 if your input is already S*MB
                        `diagonal`    (bool)          :->: Ff true, samples are drawn independently. For the moment is always true.
                        `is_duvenaud` (bool)          :->: Indicate if we are using duvenaud mean function. Only useful in DGPs
                        `init_Z`      (torch.tensor)  :->: Only used if is_duvenad = True

                Returns:
                        `f`           (torch.tensor)  :->:  A sample from the posterior distribution with shape (Dy,SMB)
                        `mu_q_f`      (torch.tensor)  :->:  Mean from marginal q(f) with shape Dy,MB,1
                        `cov_q_f`     (torch.tensor)  :->:  Covariance from marginal q(f) with shape (Dy,MB,1) if diagonal else (Dy,MB,MB)
                        `f`           (torch.tensor)  :->:  Same as aboved, just provided for compatibility with TGP.
        
        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to work batched and multioutput respectively
        assert len(X.shape) == 3, 'Invalid input X.shape' 
        X = X.repeat(1,S,1)

        f, mean_q_f, cov_q_f = self.sample_from_variational_marginal_base(X = X, diagonal = diagonal, is_duvenaud = is_duvenaud, init_Z = init_Z)
        return f, mean_q_f, cov_q_f, f


