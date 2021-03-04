#-*- coding: utf-8 -*-
# Bernoulli.py : Parent class for Bernoulli Likelihood for binary class problems
# Author: Juan Maroñas and  Ollie Hamelijnc

# Python
import sys
sys.path.extend(['../'])

# Torch
import torch
import torch.nn as nn
import torch.distributions as td

# Gpytorch
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

# custom
from .. import config as cg
from ..models.flow import IdentityFlow, CompositeFlow

class Bernoulli(nn.Module):
    ''' Implements Bernouilli likelihood using Gaussian CDF link functions

        All integrations required are done using quadrature as integration is done w.r.t
        univariate Gaussians

        Predictive distribution is done followin equation 3.77 in the GP book Rasmussen and Williams et al
    '''

    def __init__(self):
        super(Bernoulli,self).__init__()

        self.C                       = 2
        self.quad_points             = cg.quad_points                  
        self.quadrature_distribution = GaussHermiteQuadrature1D(self.quad_points)
        self.loss                    = nn.BCELoss(reduction = 'none')
        self.link_function           = td.normal.Normal(0,1).cdf

    def sample_from_output(self, f: torch.tensor, i:int, **kwargs) -> td.Distribution:
        """
            Args:
                f: likelihood input - shape Nx1
                i: returns a sample from p(y_i | f)
        """
        probs = self.link_function(f)
        dist  = td.Bernoulli(probs = probs)
        return dist.sample().to(cg.dtype)


    def expected_log_prob(self,Y:torch.tensor, gauss_mean: torch.tensor, gauss_cov: torch.tensor, flow: list, X: torch.tensor, **kwargs):

        ''' Computes

            \int q(f_0) \log p(y|G(f_0)) d_f_0

            Each of the samples f_0 are squeezed through the flow first. The rest of the integration is done with the quadrature

        
            Args:
                `Y`             (torch.tensor)  :->:  Categorical Labels in standard format (i.e NO one-hot). Shape (1,MB)
                `gauss_mean`    (torch.tensor)  :->:  mean from q(f). Shape (n_classes,MB)
                `gauss_cov`     (torch.tensor)  :->:  diagonal covariance from q(f). Shape (n_classes,MB)
                `non_linearity` (list)          :->:  List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor)  :->:  Input with shape (n_classes,MB,Dx). Needed for input dependent flows
                
        '''
        assert len(flow) == 1, "Flow list must be size 1 for Bernoulli likelihood"
        assert gauss_mean.size(0) == 1, "Binary classification just require one GP for both classes"
        assert len(X.shape) == 3, 'Bad input X, expected (n_class,MB*S,Dx)'

        MB     = gauss_mean.size(1)
        S_quad = self.quad_points
        Y      =  Y.type(cg.dtype)

        Y          = Y.t().view(1,MB).repeat((S_quad,1)) # S,Dy,MB
        gauss_mean = gauss_mean #.t()
        gauss_cov[gauss_cov < 0] = 0 ## add this as sometimes there is a very small negative value (PW model mainly)
        gauss_std  = gauss_cov.sqrt() #.t()

        distr = td.Normal(gauss_mean,gauss_std)

        def _compute_BCE(f0,_Y,_flow,_X):
            fk = flow[0](f0[:,0,:],_X[0]) # warp the sample previous to apply 
                                          # the link function
                                          # Note we follow warping standard 
                                          # by GaussianNonLinearMean Likelihood
            yk = self.link_function(fk)   # map to probability
            log_LLH = self.loss(yk,_Y)
            return log_LLH

        compute_binary_crossentropy = lambda f_samples: _compute_BCE(f_samples,Y,flow,X)
        log_p_y = self.quadrature_distribution(compute_binary_crossentropy, distr)

        log_p_y = log_p_y.sum() # reduce minibatch dimension
        return log_p_y*-1 # we return the negation because the cross entropy is the negative log likelihood, while our code assumes we maximize the ELBO (although optimization is done by finally negating and minimizing)


    def marginal_moments(self, gauss_mean, gauss_cov, flow, X, **kwargs):
        """ Computes marginal moments w.r.t single Gaussian:

            p(ys|xs) = int p(y*|f*) q(f*)df* = int gauss_CDF(f*) q(f*)df*
        
            Note: We use exact integration by following equation 3.77 in the GP book
                  The value of m is 0 and of s is 1. Hence the solution to the integral is
                  equation 3.80

            Args:
                `gauss_mean`       (torch.tensor)  :->: mean from q(f). Shape (n_classes,MB)
                `gauss_cov`        (torch.tensor)  :->: covariance from q(f). Shape (n_classes,MB)
                `non_linearity`    (list)          :->: flow list
                `X`                (torch.tensor)  :->: Input locations used by input dependent flows

        """
        assert len(flow) == 1, "Flow list must be size 1 for Bernoulli likelihood"
        assert gauss_mean.size(0) == 1, "Binary classification just require one GP for both classes"
        assert len(X.shape) == 3, 'Bad input X, expected (n_class,MB*S,Dx)'

        gauss_mean = gauss_mean # .t()
        gauss_cov  = gauss_cov  # .t()
        gauss_std  = gauss_cov.std()

        # This operation could be done by creating a specific class for the Bernoulli 
        # likelihood that differs when the mapping applied over F0 is the Identity or not
        # However this is not as the GaussianLikelihood, where computations clearly
        # differ in the number of code lines to use. Thus we follow what GPflow does 
        # which internally check the link function and swithches between quadrature
        # and eq 3.77.
        if isinstance(flow[0],CompositeFlow):
            for f in flow[0].flow_arr:
                is_identity_flow = isinstance(f,IdentityFlow)
                if not is_identity_flow:
                    break
        else:
            is_identity_flow = isinstance(flow[0],IdentityFlow)

        if is_identity_flow: # use equation 3.77

            P = self.link_function( gauss_mean.t() / torch.sqrt(1 + gauss_cov.t() )) # use transpose so that output is (MB,classes)

        else: # use quadrature
            distr = td.Normal(gauss_mean,gauss_std)

            def _warp_and_squeeze(f0,_flow,_X):

                fk = _flow[0](f0[:,0,:],_X[0]) # warp the sample previous to apply 
                                              # the link function
                yk = self.link_function(fk)   # map to probability
                return yk

            compute_moments = lambda f_samples: _warp_and_squeeze(f_samples,flow,X)
            P = self.quadrature_distribution(compute_moments, distr)
            P = P.unsqueeze(dim = 1) # to be compatible when output (MB,classes)

            P[P < 0.0] = 0.0 # quadrature can incorporate small numericall error that gives probability around negative 0, i.e 0 - 1e-20
            P[P > 1.0] = 1.0 # quadrature can incorporate small numericall error that gives probability around negative 0, i.e 1 + 1e-20

        return P
