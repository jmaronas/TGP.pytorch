#-*- coding: utf-8 -*-
# kernels.py : kernels not provided by gpytorch
# Author: Juan Maroñas and  Ollie Hamelijnck

import sys
sys.path.extend(['../'])

import torch
import torch.nn as nn

from gpytorch.kernels import Kernel
from gpytorch.lazy import DiagLazyTensor, ZeroLazyTensor, NonLazyTensor

# custom
from .. import config as cg


class WhiteNoiseKernel(Kernel):
    """
      White noise kernel that adds a variance to the diagonal elements.
      This Kernels is only preprared to work with batch_shaped kernels.

      Attr:
        variance       (float) : initial value of the noise variance
        batch_shape    (int)   : number of GPs run in parallel. Similar to gpy.kernel batch_shape argument
        constant_noise (bool)  : if false only adds noise for K(X,X) evaluations
    """


    def __init__(self, variance, batch_shape = int, constant_noise = False):
        super(WhiteNoiseKernel, self).__init__()
        
        assert batch_shape >= 1, "Batch shape should be at least 1"

        variances           = torch.ones((batch_shape,1),dtype = cg.dtype)*variance
        init_noise          = torch.log(variances)
        self.noise_log_var  = nn.Parameter(init_noise)
        self.batch_shape    = torch.Size([batch_shape]) 
        self.constant_noise = constant_noise

    
    def forward(self, x1, x2, diag = False, are_equal = True, **params):
        batch_shape = self.batch_shape
        leading_dim = x1.size()[:-2]
        
        if self.constant_noise:
            _are_equal = (x1.shape == x2.shape)
        else:
            _are_equal = torch.equal(x1,x2) and are_equal
            
        if  _are_equal:
            noise_var = torch.exp(self.noise_log_var).expand(-1,x1.size(-2))
            K = DiagLazyTensor(noise_var)
        else:
            K = ZeroLazyTensor(*leading_dim, x1.size(-2), x2.size(-2), dtype = x1.dtype, device = x1.device)

        if diag:
            K = K.diag()
            if not leading_dim:
                K = K.unsqueeze(0)
            return K # return torch.tensor rather than lazy. Consistent with other's kernels behavior

        return K
