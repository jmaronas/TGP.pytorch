#-*- coding: utf-8 -*-
# config.py : In this file we place config things of the project
# Author: Juan Maroñas and  Ollie Hamelijnck
import torch
import gpytorch
import numpy
import math
import platform
import os
import warnings

def check_device():
    device = 'cpu'
    if torch.cuda.is_available():
       device = 'cuda'
    return device

def check_versions(torch_version,gpytorch_version):
    if torch.__version__ not in torch_version:
        raise ImportError('Torch does not match correct version {}'.format(torch_version))
        #warnings.warn('Torch does not match correct version {}'.format(torch_version))

    if gpytorch.__version__ not in gpytorch_version:
        raise ImportError('Gpy Torch does not match correct version {}'.format(gpytorch_version))
        #warnings.warn('Gpy Torch does not match correct version {}'.format(gpytorch_version))

def get_root_dir():
    from . import data
    #import data
    abs_dir = os.path.abspath(data.__file__)
    return os.path.split(os.path.split(os.path.split(abs_dir)[0])[0])[0]

def set_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def set_maximum_precission():
    global dtype
    global maximum_precision
    global quad_points
    global device
    maximum_precision = True
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    quad_points = 100
    #device = 'cpu'

## Config Variables
torch_version    = ['1.7.0','1.5.0']
gpytorch_version = ['1.3.0', '1.1.1']

config_seed  = 0
dtype        = torch.float32
torch.set_default_dtype(dtype)

maximum_precision  = False # This basically adds jitter to all the covariance matrices whose cholesky wont be stable
is_linux           = 'linux' in platform.platform().lower()
quad_points        = 50  # number of quadrature_points
S_train            = 1   # monte carlo samples for training deep instances
S_test             = 100 # monte carlo samples for testing deep instances
positive_transform = 'exp' #function to force positive. Options : 'exp', 'softplus'.
strict_flag        = True
constant_jitter    = None # if different from None always used this, else use gpytorch like
global_jitter      = None # if different from None it uses this jitter value as a drop in replacement of jitter in psd_safe_cholesky

set_seed(config_seed)
device=check_device()
root_directory = get_root_dir()

## Constant definitions
pi = torch.tensor(math.pi)

## Callers
check_versions(torch_version,gpytorch_version)

## Plot palette
palette = [ 'C'+str(j) for j in range(1000) ]

## Color palette from Jeremias
color_palette_J = {
                    'black': '#000000',
                    'orange': '#E69F00',
                    'blue': '#56B4E9',
                    'green': '#009E73',
                    'orange': '#F0E442',
                    'dark_blue': '#0072B2',
                    'dark_orange': '#D55E00',
                    'pink': '#CC79A7',
                    'white': '#111111',
                    'grey': 'grey'
                  }
