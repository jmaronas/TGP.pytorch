#-*- coding: utf-8 -*-
# GaussianNonLinearMean.py : Keeps a Non Linear Gaussian Mean p(y|T(f)) likelihood with homoceodastic noise and non linearity given by T(f).
# Author: Juan Maroñas and  Ollie Hamelijnck

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
from ..utils import batched_log_Gaussian, positive_transform,inverse_positive_transform 



class GaussianNonLinearMean(nn.Module):
    """Place a GP over the mean of a Gaussian likelihood $p(y|G(f))$ 
    with noise variance $\sigma^2$ and with a NON linear transformation $G$ over $f$.
    It supports multi-output (independent) GPs and the possibility of sharing 
    the noise between the different outputs. In this case integrations wrt to a 
    Gaussian distribution can only be done with quadrature."""

    def __init__(self,out_dim : int, noise_init: float, noise_is_shared : bool, quadrature_points: int):
        super(GaussianNonLinearMean,self).__init__()

        self.out_dim = out_dim
        self.noise_is_shared = noise_is_shared

        if noise_is_shared: # if noise is shared create one parameter and expand to out_dim shape
            log_var_noise = nn.Parameter(torch.ones(1,1,dtype = cg.dtype)*inverse_positive_transform(torch.tensor(noise_init,dtype = cg.dtype)))

        else: # creates a vector of noise variance parameters.
            log_var_noise = nn.Parameter(torch.ones(out_dim,1,dtype = cg.dtype)*inverse_positive_transform(torch.tensor(noise_init,dtype = cg.dtype)))

        self.log_var_noise = log_var_noise

        self.quad_points = quadrature_points
        self.quadrature_distribution = GaussHermiteQuadrature1D(quadrature_points)

    def sample_from_output(self, f: torch.tensor, i:int, **kwargs) -> td.Distribution:
        """
            Args:
                f: likelihood input - shape Nx1
                i: returns a sample from p(y_i | f)
        """
        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        #select the noise term for output i
        var = positive_transform(log_var_noise[i])

        dist = td.Normal(f, torch.ones_like(f)*torch.sqrt(var))
        return dist.sample()

    def log_non_linear(self, f: torch.tensor, Y: torch.tensor, noise_var: torch.tensor, flow: list, X: torch.tensor, **kwargs):
        """ Return the log likelihood of S Gaussian distributions, each of this S correspond to a quadrature point.
            The only samples f have to be warped with the composite flow G().
            -> f is assumed to be stacked samples of the same dimension of Y. Here we compute (apply lotus rule):

          \int \log p(y|fK) q(fK) dfK = \int \log p(y|fk) q(f0) df0 \approx 1/sqrt(pi) sum_i w_i { \log[ p( y | G( sqrt(2)\sigma f_i + mu), sigma^2 ) ] };

          where q(f0) is the initial distribution. We just face the problem of computing the expectation under a log Gaussian of a 
          non-linear transformation of the mean, given by the flow.
      
                Args:
                        `f`         (torch.tensor)  :->:  Minibatched - latent function samples in (S,Dy,MB), being S the number of quadrature points and MB the minibatch.
                                                          This is directly given by the gpytorch.GaussHermiteQuadrature1D method in this format and corresponds to 
                                                          \sqrt(2)\sigma f_i + mu see https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
                        `Y`         (torch.tensor)  :->:  Minibatched Observations in Dy x MB.
                        `noise_var` (torch.tensor)  :->:  Observation noise
                        'flow'      (CompositeFlow) :->:  Sequence of flows to be applied to each of the outputs
                        'X'         (torch.tensor)  :->:  Input locations used for input dependent flows. Has shape [Dy,S*MB,Dx] or shape [S*MB,Dx]. N
        """
        assert len(flow) == self.out_dim, "This likelihood only supports a flow per output_dim. Got {} for Dy {}".format(self.out_dim,len(flow))
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        S  = self.quad_points
        MB = Y.size(1)
        Dy = self.out_dim

        Y = Y.view(Dy,MB,1).repeat((S,1,1,1))                 # S,Dy,MB,1
        noise_var = noise_var.view(Dy,MB,1).repeat((S,1,1,1)) # S,Dy,MB,1

        fK = f.clone()

        # Be aware that as we expand X we will be performing self.quad_points forwards through the NNets for each output. This might be inneficient unless pytorch only performs the operation once and returned the expanded dimension 

        # expanded_size = [self.quad_points] + [-1]*(len(X.size()))
        # X = X.expand(expanded_size) # no need for this broadcasting as pytorch will broadcast automatically

        for idx,fl in enumerate(flow):
            # warp the samples 
            fK[:,idx,:] = fl(f[:,idx,:],X[idx]) 

        fK = fK.view(S,Dy,MB,1) # we add extra dimension so that batched_log_gaussian does not reduce minibatch dimension. This will be reduced at the end as the
                              # GaussHermiteQudrature from gpytorch reduces S by default. Although sum is associative we prefer to separate for clarity.

        log_p_y = batched_log_Gaussian( obs = Y, mean = fK, cov = noise_var, diagonal = True, cov_is_inverse = False) # (S,Dy,MB)

        return log_p_y # return (S,Dy,MB) so that reduction is done for S.


    def expected_log_prob(self, Y, gauss_mean, gauss_cov, flow, X, **kwargs):
        """ Expected Log Likelihood

            Computes E_q(f) [\log p(y|G(f))] = \int q(f) \log p(y|G(f)) df \approx with quadrature

                - Acts on batched form. Hence returns vector of shape (Dy,)

            Args:

                `Y`             (torch.tensor)  :->:  Labels representing the mean. Shape (Dy,MB)
                `gauss_mean`    (torch.tensor)  :->:  mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor)  :->:  diagonal covariance from q(f). Shape (Dy,MB)
                `non_linearity` (list)          :->:  List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor)  :->:  Input with shape (Dy,MB,Dx) or shape (MB,Dx). Needed for input dependent flows

        """

        assert len(flow) == self.out_dim, "The number of callables representing non linearities is different from out_dim"
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        N = Y.size(1)
        C_y = positive_transform(log_var_noise).expand(-1,N)

        distr = td.Normal(gauss_mean,gauss_cov.sqrt()) # Distribution of shape (Dy,MB). Gpytorch samples from it

        log_likelihood_lambda = lambda f_samples: self.log_non_linear(f_samples, Y, C_y, flow, X)
        ELL = self.quadrature_distribution(log_likelihood_lambda,distr)
        #ELL shape is Dy x N 

        ELL = ELL.sum(1) # reduce minibatch dimension. Shape (Dy,)

        return ELL

    def marginal_moments(self,gauss_mean,gauss_cov, flow, X, **kwargs):
        """ Computes the moments of order 1 and non centered 2 of the observation model integrated out w.r.t a Gaussian with means and covariances. 
            There is a non linear relation between the mean and integrated variable

            p(y|x) = \int p(y|G(f)) p(f) df

            - Note that p(f) can only be diagonal as this function only supports quadrature integration.
            - Moment1: \widehat{\mu_y} = \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]
            - Moment2: \sigma^2_o + \frac{1}{\sqrt\pi} \sum^S_{s=1} w_s \mathtt{G}[\sqrt2 \sigma f_s + \mu]^2 - \widehat{\mu_y}^2

            Args:
                `gauss_mean`    (torch.tensor) :->: mean from q(f). Shape (Dy,MB)
                `gauss_cov`     (torch.tensor) :->: covariance from q(f). Shape (Dy,MB) if diagonal is True, else (Dy,MB,MB). For the moment only supports diagonal true
                `non_linearity` (list)         :->: List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor) :->: Input locations used by input dependent flows
        """
        assert len(X.shape) == 3, 'Bad input X, expected (out_dim,MB*S,Dx)'
        assert X.size(0) == self.out_dim, 'Wrong first dimension in X, expected out_dim'

        if self.noise_is_shared:
            log_var_noise = self.log_var_noise.expand(self.out_dim,1)
        else:
            log_var_noise = self.log_var_noise

        MB  = gauss_mean.size(1)
        C_Y = positive_transform(self.log_var_noise).expand(-1,MB) # shape (Dy,MB)
        # expanded_size = [self.quad_points] + [-1]*(len(X.size())-1)
        # X = X.expand(expanded_size) # no need for this expansion as pytorch will broadcast automatically
        def aux_moment1(f,_flow):
            # f.shape (S,Dy,MB)
            for idx,fl in enumerate(_flow):
                # warp the samples 
                f[:,idx,:] = fl(f[:,idx,:],X[idx])
            return f

        def aux_moment2(f,_flow):
            # f.shape (S,Dy,MB)
            # x.shape (Dy,MB) # pytorch automatically broadcast to sum over S inside the flow fl
            for idx,fl in enumerate(_flow):
                # warp the samples 
                f[:,idx,:] = fl(f[:,idx,:],X[idx])
            return f**2

        aux_moment1_lambda = lambda f_samples: aux_moment1(f_samples,flow)
        aux_moment2_lambda = lambda f_samples: aux_moment2(f_samples,flow)
        distr = td.Normal(gauss_mean,gauss_cov.sqrt()) # Distribution of shape (Dy,MB). Gpytorch samples from it

        m1         =  self.quadrature_distribution(aux_moment1_lambda,distr)
        E_square_y =  self.quadrature_distribution(aux_moment2_lambda,distr)
        m2         =  C_Y + E_square_y - m1**2 
        
        return m1, m2




