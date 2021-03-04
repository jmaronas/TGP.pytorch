#-*- coding: utf-8 -*-
# utils.py : For the moment this is a utils file where we place everything that might be useful and common to different things
# Author: Juan Maroñas and  Ollie Hamelijnck

## Python
import warnings
from gpytorch.utils.errors import NanError
from gpytorch.utils.warnings import NumericalWarning


## Standard
import numpy
import numpy as np
from sklearn.cluster import KMeans

# Torch
import torch
import torch.distributions as td

## Custom
from . import config as cg

#used to estimate spectral density
import scipy
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import cumtrapz
import math
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


import contextlib

#used to disable with statements
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

def positive_transform(x: torch.tensor) -> torch.tensor:
    if cg.positive_transform is 'exp':
        return torch.exp(x)
    elif cg.positive_transform is 'softplus':
        return torch.log(torch.exp(x)+1)
    
    raise NotImplementedError('positive_transform function ', cg.positive_transform, ' is not implemented.' )


def inverse_positive_transform(x: torch.tensor) -> torch.tensor:
    if cg.positive_transform is 'exp':
        return torch.log(x)
    elif cg.positive_transform is 'softplus':
        #beta is assumed to be 1
        return torch.log(torch.exp(x) - 1.)
    
    raise NotImplementedError('inverse_positive_transform function for ', cg.positive_transform, ' is not implemented.' )


def estimate_spectral_density(x, y, prominence=1000, plot=True):
    """
        Computes the fourier transform y and computed the emperical spectral density.
        The peaks of the density are found (with tuning parameter) prominence and the frequence and periods 
            relating to these peaks are returned.

        Assumes that x is evenely spaced

        Args: 
            x : N x *
            y : N x *
        Returns:
            frequences: List[np.ndarray], periods: List[np.ndarray]
    """
    x = np.squeeze(x)
    y = np.squeeze(y)

    N = x.shape[0]

    emp_spect = np.abs(np.fft.rfft(y) ** 2) / N
    freq = np.fft.rfftfreq(N, d=np.abs(x[1]-x[0]))

    if plot:
        plt.plot(freq, emp_spect)

    peaks, _ = find_peaks(emp_spect,  prominence=prominence)

    for p in peaks:
        if plot:
            plt.axvline(freq[p])

    if plot:
        plt.show()

    num_freq = len(peaks)

    freqs = [freq[i] for i in peaks]
    periods = [1/f for f in freqs]

    return freqs, periods

def get_approximate_min_distance(x: np.ndarray, axis=0):
    """
        Returns the absolute distance between the first two points on the axis specificed.
        Assumes that x is ordered, 

        TODO: add a flag to sort first

        Args:
            X: N x D
        return:
            approx_min_dis : np.ndarray
    """
    approx_min_dist = np.abs(x_train[0, axis] - x_train[1, axis])
    return approx_min_dist

## ======================== ##
## === Is Identity Flow === ##
def is_identity_flow(flow_specs,connection):

    if connection == 'shared':
        ch1  = len(flow_specs[0])  == 1
        ch2  = flow_specs[0][0][0] == 'identity'
        flag = ch1 and ch2
    elif connection == 'single': # loop over each output
        flag = True
        for f in flow_specs:
            if type(f) != list:
                # Assume that any instance flow is not an identity flow.
                flag = False
                continue
            ch1 = len(f)  == 1
            ch2 = f[0][0] == 'identity'
            if not(ch1 and ch2):
                flag = False
                break

    else:
        raise NotImplementedError("Not implemented")
   
    return flag


## ======================== ##
## === KMEANS algorithm === ##
def KMEANS(X: torch.tensor,num_Z: int, n_init:int = 1, seed = None) -> torch.tensor:
    """
    Initialize inducing point locations using KMeans
            Args:
                    X : observed data -> column vector
                    num_Z : number of inducing points
                    n_init: number of Kmeans iterations
    """
    # study this option:https://pypi.org/project/kmeans-pytorch/
    # for the moment we use sklearn. 
    if seed is None:
        seed = cg.config_seed

    kmeans = KMeans(n_clusters=num_Z,init='k-means++',n_init = n_init, random_state = seed).fit(X.to('cpu').numpy())
    Z_init = torch.tensor(kmeans.cluster_centers_,dtype=cg.dtype).to(cg.device)

    return Z_init


## ================================================= ##
##  Log Batched Multivariate Gaussian: log N(x|mu,C) ##
def batched_log_Gaussian(obs: torch.tensor, mean: torch.tensor, cov: torch.tensor, diagonal:bool, cov_is_inverse: bool)-> torch.tensor:
    """
    Computes a batched of * log p(obs|mean,cov) where p(y|f) is a  Gaussian distribution, with dimensionality N. 
    Returns a vector of shape *.
    -0.5*N log 2pi -0.5*\log|Cov| -0.5[ obs^T Cov^{-1} obs -2 obs^TCov^{-1} mean + mean^TCov^{-1}mean]
            Args: 
                    obs            :->: random variable with shape (*,N)
                    mean           :->: mean -> matrix of shape (*,N)
                    cov            :->: covariance -> Matrix of shape (*,N) if diagonal=True else batch of matrix (*,N,N)
                    diagonal       :->: if covariance is diagonal or not 
                    cov_is_inverse :->: if the covariance provided is already the inverse
    
    #TODO: Check argument shapes
    """

    N = mean.size(-1)
    cte =  N*torch.log(2*cg.pi.to(cg.device).type(cg.dtype))
    
    if diagonal:

        log_det_C = torch.sum(torch.log(cov),-1)
        inv_C = cov
        if not cov_is_inverse:
            inv_C = 1./cov # Inversion given a diagonal matrix. Use torch.cholesky_solve for full matrix.
        else:
            log_det_C *= -1 # switch sign

        exp_arg = (obs*inv_C*obs).sum(-1) -2 * (obs*inv_C*mean).sum(-1) + (mean*inv_C*mean).sum(-1)

    else:
        raise NotImplemented("log_Gaussian for full covariance matrix is not implemented yet.")
    return -0.5*( cte + log_det_C + exp_arg )


## =============================================
## Add Jitter for safe cholesky decomposition ##
def add_jitter_MultivariateNormal(mu_q_f,K):
    # -> Copied from gpytorch: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
    #    We have to add it sometimes because td.MultivariateNormal does not add jitter when necessary

    # TODO: give support for diagonal instance of Multivariate Normals. 

    jitter = 1e-6 if K.dtype == torch.float32 else 1e-8
    Kprime = K.clone()
    jitter_prev = 0
    for i in range(5):
        jitter_new = jitter * (10 ** i)
        Kprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
        jitter_prev = jitter_new
        try:
            q_f = td.multivariate_normal.MultivariateNormal(mu_q_f, Kprime)
            return q_f
        except RuntimeError: 
            continue
    raise RuntimeError("Cannot compute a stable td.MultivariateNormal instance. Got singular covariance matrix")

## ========================================================================= ##
## Safe cholesky from gpytorch but returning the covariance with jitter also ##
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        if cg.constant_jitter is not None:
            A.diagonal(dim1=-2, dim2=-1).add_(cg.constant_jitter)

        L = torch.cholesky(A, upper=upper, out=out)

        ## For a weird reason sometimes even A has nan torch.cholesky doesnt fail and goes into except. We check it manually and raise it ourselves
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        return L, A
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise NanError(
                f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
            )

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(3):
            jitter_new = jitter * (10 ** i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(f"A not p.d., added jitter of {jitter_new} to the diagonal", NumericalWarning)
                return L, Aprime
            except RuntimeError:
                continue
        raise e
