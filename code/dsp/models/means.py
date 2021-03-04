#-*- coding: utf-8 -*-
# means.py : this file contains means functions not provided in gpytorch
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import sys
sys.path.extend(['../'])

# Torch 
import torch
import torch.nn as nn

# Standard
import numpy

# Gpytorch
import gpytorch as gpy

# custom
from .. import config as cg

## ============================ ##
## === Identity Mean Function === ##
## Salimbeni et al
class Identity(gpy.means.Mean):
    '''
       Identity Mean Function:

          m(x) = W*x 

          Note: W is a projection matrix that maps from input_dimension to output_dimension. It could be created inside this class but for the moment we keep it outside.

          IMPORTANT: This mean function works differently from the others (for convenience). This mean function is R^D -> R^D instead of R^D -> R (as it should be). 
                     This means that this function is usefull for multioutput GP. However, to be consistent with how the kernels work, we need to reshape W to perform a
                     batch of dot products. The input X will have shape (Dy,MB,Dx), and this matrix, by design, have shape (Dx,Dy). This means we have to perform a 
                     batch of dot products with the columns of W.

    '''
    def __init__(self, W: torch.tensor, num_inputs : int, num_outputs) -> None :
        super(Identity,self).__init__()
        W      = W.t()                            # shape (Dy,Dx). Each row represents the vector that projects X to each output Dy
        W      = W.view(num_outputs,num_inputs,1) # reshape and add 1 dim to perform batched product of dot products 
        W      = W.to(cg.device)
        if torch.__version__ == '1.5.0':
            self.register_buffer('W', W)
        else:
            self.register_buffer('W', W, False)

    def __call__(self,X) -> torch.tensor :
        # X has shape (Dy,MB,Dx)
        # W has shape (Dy,Dx,1)
        # Batched product X@W gives a tensor of shape (Dy,MB,1).
        return torch.bmm(X,self.W)

## ============================ ##
## === Linear Mean Function === ##
class Linear(gpy.means.Mean):
    '''
       Linear Mean Function:

          m(x) = a x + b 

            Args:
                    `num_dim`  (int)  :->: number of input dimensions
                    `out_dim`  (int)  :->: number of output dimensions.
    '''
    def __init__(self, input_dim, output_dim) -> None :
        super(Linear,self).__init__()
        numpy.random.seed(cg.seed) # ensure same initialization always
        self.a = nn.Parameter( torch.tensor(numpy.random.randn(output_dim,input_dim,1),dtype=cg.dtype) )
        self.b = nn.Parameter(torch.zeros(output_dim,1,1,dtype=cg.dtype))

    def __call__(self,X) -> torch.tensor :
        return torch.bmm(X,self.a) + self.b


