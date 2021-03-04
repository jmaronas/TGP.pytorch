#-*- coding: utf-8 -*-
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import sys
sys.path.extend(['../'])

# Torch
import torch
import torch.nn as nn
import torch.distributions as td

# Gpytorch 
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

# custom
from .. import config as cg
from ..utils import batched_log_Gaussian, positive_transform,inverse_positive_transform 

from . import GaussianLinearMean
from ..models.flow import Flow


class WarpedGaussianLinearMean(GaussianLinearMean):
    '''
        Warps output Y by a transform Y:
            T(Y) = f + eps
            Y = T^{-1}(f + eps)

        The resulting likelihood is
            N(T(Y) | f, Sigma) |dT/dY|

        Assumes single output
    '''
    def __init__(self,out_dim : int, noise_init: float, noise_is_shared : bool, flow:Flow, quad_points:int):
        super(WarpedGaussianLinearMean,self).__init__(out_dim, noise_init, noise_is_shared)

        self.flow = nn.ModuleList([flow])
        self.quad_points = quad_points

        self.quad = GaussHermiteQuadrature1D(self.quad_points) # quadrature integrator.

    def sample_from_output(self, f: torch.tensor, i:int) -> td.Distribution:
        """
            Args:
                f: likelihood input - shape Nx1
                i: returns a sample from p(T(y_i) | f) |dT/dY|

            Returns sample by:
                1) Sample f^(s) from N(f, \sigma^2)
                2) Y = T^-1(f^(s))
        """
        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        #select the noise term for output i
        var = positive_transform(log_var_noise[i])

        dist = td.Normal(f, torch.ones_like(f)*torch.sqrt(var))
        return self.flow[0].inverse(dist.rsample())

    def expected_log_prob(self, Y, gauss_mean, gauss_cov):
        """ Expected Log Likelihood

            Computes E_q(f) [\log p(y|f)] = E_q(f) [\log p(T^{-1}(y)|f)] + \log |dT^{-1} / dY|

            Args:

                `Y`          (torch.tensor)  :->:  Labels representing the mean. Shape (Dy,MB)
                `gauss_mean` (torch.tensor)  :->:  mean from q(f). Shape (Dy,MB)
                `gauss_cov`  (torch.tensor)  :->:  diagonal covariance from q(f). Shape (Dy,MB)

        """
        #call closed form ELL from standard Gaussian likelihood
        base_ell = super(WarpedGaussianLinearMean,self).expected_log_prob(self.flow[0].forward(Y), gauss_mean, gauss_cov)

        #compute \log |dT^{-1} / dY|
        log_jacobian_l_idx = torch.sum(torch.log(self.flow[0].forward_grad(Y)))
        
        ell = base_ell + log_jacobian_l_idx

        return ell

    def unwarped_marginal_moments(self,gauss_mean,gauss_cov, diagonal):
        """
            Returns moments from p(Y_0, | f)
        """
        return super(WarpedGaussianLinearMean,self).marginal_moments(gauss_mean,gauss_cov, diagonal)

    def marginal_moments(self,gauss_mean,gauss_cov, diagonal):
        """ Computes the moments of order 1 and non centered 2 of the observation model integrated out w.r.t a Gaussian with means and covariances. 

            p(y|x) = \int p(y|f) p(f) df

            - Note that p(f) can be any distribution, hence this is useful also for the marginal likelihood

            Args:
                `gauss_mean` (torch.tensor)  :->: mean from q(f). Shape (Dy,MB)
                `gauss_cov`  (torch.tensor)  :->: covariance from q(f). Shape (Dy,MB) if diagonal is True, else (Dy,MB,MB)
                `diagonal`   (bool)          :->: wether the covariance from the distribution is diagonal or not

        """


        N = gauss_mean.size(1)

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        C_Y = positive_transform(log_var_noise).expand(-1,N) # shape (Dy,MB)

        if not diagonal:
            C_Y = torch.diag_embed(C_Y) # shape (Dy,MB,MB)

        C_Y_0  = C_Y + gauss_cov
        mu_Y_0 = gauss_mean.clone() 

        ## Base distribution q(f_0)
        q_y0 = td.Normal(mu_Y_0, C_Y_0.sqrt())

        def warp(f):
            return self.flow[0].inverse(f)

        def aux_moment1(f):
            return warp(f)
            
        def aux_moment2(f):
            f = warp(f)
            return f*f

        ## Moment of order 1 (mean) ##
        mean_lambda = lambda samples: aux_moment1(samples)
        moment1     = self.quad.forward(mean_lambda, q_y0)

        ## Centered 2 moment (variance) ##
        # -> VAR = E[fk^2] - moment1**2

        # compute second moment E[fk^2] with quadrature
        quad_lambda = lambda samples: aux_moment2(samples)
        moment2     = self.quad.forward(quad_lambda,q_y0)
        moment2 = moment2 - moment1**2

        return moment1, moment2


    def log_marginal(self, Y, gauss_mean, gauss_cov):
        """ Computes the log marginal likelihood 
        
            log p(Y|x) =   log p(T^{-1}(Y)|x) + + log |dT^{-1} / dY| 

            Args:
                `Y` (torch.tensor)  :->: Observations Y with shape (Dy,MB)
                `gauss_mean` (torch.tensor)  :->:  mean from p(f). Shape (Dy,MB)
                `gauss_cov`  (torch.tensor)  :->:  full covariance from p(f). Shape (Dy,MB,MB)
        """


        base_log_ml = super(WarpedGaussianLinearMean,self).log_marginal(self.flow[0].forward(Y), gauss_mean, gauss_cov)

        #compute \log |dT^{-1} / dY|
        log_jacobian_l_idx = torch.sum(torch.log(sel.flow[0].forward_grad(Y)))

        return base_log_ml + log_jacobian_l_idx





