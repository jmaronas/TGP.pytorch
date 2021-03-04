#-*- coding: utf-8 -*-
# initializers.py : Holds different flow initializers
# Author: Juan Maroñas and  Ollie Hamelijnck

# Torch
import torch 
import torch.distributions as td
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import softplus

# Gpytorch
import gpytorch

# Standard
import numpy
import numpy as np
import pandas as pd
import scipy
import typing
import warnings
import inspect

# Custom
from ..models.flow import *
from .. import config as cg


def find_forward_params(x_input: torch.tensor, y_ouput: torch.tensor, random_flow_fn: typing.Callable=None, num_restarts: int=1, optimizer_fn=None, num_epochs=None, seed=0, verbose = 0, verbose_level = 0 ) -> Flow:
    """
        Attempts to find a flow that maps from x_input to y_output by minimising the MSE between T(x_input) and y_output
        Args:
            random_flow_fn (callable): defines the flow to optimize, everytime this is called it should return a a randomly initalised version
            num_restarts (int): number of random restarts to try
    """

    if random_flow_fn is None:
        raise RuntimeError('random_flow_fn must be specified')

    if optimizer_fn is None:
        warnings.warn("Using default optimizer (optim.Adam(trainable_params, lr=0.01))", Warning)
        optimizer_fn = lambda trainable_params:  optim.Adam(trainable_params, lr=0.01)

    if num_epochs is None:
        warnings.warn("Using default number of epochs (100)", Warning)
        num_epochs = 100

    np.random.seed(seed)
    
    found_flows = []
    found_min_losses = []
    found_losses = []
    
    for r in range(num_restarts):

        _flow = random_flow_fn()
        
        #Create MSE loss
        def mse_loss(x_input_t, y_ouput_t):
            return torch.mean((_flow.forward(x_input_t)-y_ouput_t)**2)

        #Get parameters of the flow
        trainable_params = []
        for n,p in _flow.named_parameters():
            trainable_params.append(p)


        optimizer = optimizer_fn(trainable_params)

        #Minimise distance
        if verbose:
                print('Restart {r}'.format(r=r), end = '\r')

        loss_arr = []
        for e in range(num_epochs):
            ## stochastic gradient update
            
            optimizer.zero_grad() # just in case

            loss = mse_loss(torch.tensor(x_input), torch.tensor(y_ouput))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                _end = '\r'
                if e >= num_epochs-1:
                     _end = '\n'

                if verbose_level != 1:
                    _end =  '\n'

                print('Restart {} Step {} - {}'.format(r,e,loss.detach().item()), end=_end)

                                        
            loss_arr.append(loss.detach().to('cpu').numpy())
        
        found_flows.append(_flow)
        found_min_losses.append(loss_arr[-1])
        found_losses.append(loss_arr)

    # In case it saturets we must take out the nan from the initializer list otherwise it will always be returned
    idx = numpy.logical_not(numpy.isnan(found_min_losses))
    found_min_losses = numpy.array(found_min_losses)[idx]
    found_losses     = numpy.array(found_losses)[idx].tolist()
    found_flows      = numpy.asarray(found_flows, dtype=object)[idx].tolist()

    return found_flows[np.argmin(found_min_losses)], found_losses[np.argmin(found_min_losses)]

def find_forward_params_input_dependent_flow(x_loader: torch.utils.data.dataloader, FLOW : Flow, optimizer_fn=None, num_epochs=None, seed=0, verbose = 0, verbose_level = 0, noise_var = 0.0 ) -> Flow:
    """
        Attempts to find a flow that maps from x_input to y_output by minimising the MSE between T(x_input) and y_output
        Works with flows that are input dependent. As a consequence one of the arguments is a dataloader so that we can work in the range of the input that is going to be used.
        The other argument is a init_flow that holds the value of the parameters towards the input dependent flow is initialized. Thus init_flow will in general be the output of another initializer
    """

    if optimizer_fn is None:
        warnings.warn("Using default optimizer (optim.Adam(trainable_params, lr=0.01))", Warning)
        optimizer_fn = lambda trainable_params:  optim.Adam(trainable_params, lr=0.01)

    if num_epochs is None:
        warnings.warn("Using default number of epochs (100)", Warning)
        num_epochs = 100

    np.random.seed(seed)

    found_flows = []
    found_min_losses = []
    found_losses = []
    

    #Get parameters of the flow
    trainable_params = []
    for n,p in FLOW.named_parameters():
        trainable_params.append(p)

    optimizer = optimizer_fn(trainable_params)

    state = FLOW.training
    FLOW.train() # switch layers to training mode (dropout, batchnorm etc)
    FLOW.to(cg.device)
    loss_acc = 0.0
    for e in range(num_epochs):
        ## stochastic gradient update
        loss_acc = 0.0
        for x,y in x_loader:
            x,y = x.to(cg.device),y.to(cg.device)

            if type(noise_var) is float:
                if noise_var > 0.0:
                    x = x + torch.zeros_like(x).normal_()*numpy.sqrt(noise_var)
            elif type(noise_var) is list:
                idx = numpy.random.randint(len(noise_var))
                x = x + torch.zeros_like(x).normal_()*numpy.sqrt(noise_var[idx])
            else:
                raise NotImplementedError()

            optimizer.zero_grad() # just in case
            loss = FLOW.forward_initializer(x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # TODO: for doing this we need to compute the minimum dataset_norm per dataset. We keep it for future once we know how this works
            # The same norm could be added when training the model
            # loss = FLOW.forward_initializer(x + torch.normal(0,1,x.shape)*dataset_norm ) # also regularize a bit in the surroundings
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            print('Epoch {} Loss {}'.format(e,loss.detach().item()), end='\r')

            loss_acc += loss.item()
    
    print("\n")
    # Return it in its operating mode 
    FLOW.training = state
    FLOW.turn_off_initializer_parameters()
    FLOW.to('cpu') # map back to cpu

    return FLOW, loss_acc



def get_empty_flow(flow, params=None):
    def get_flow(flow_l):
        sig = inspect.getfullargspec(flow_l.__class__.__init__)
        num_params = sum([s.startswith('init_') for s in sig.args])

        add_init_f0_flag = False

        add_constraint_flag = False

        kwargs = { }

        if 'add_init_f0' in sig.args:
            kwargs['add_init_f0'] = flow_l.add_init_f0


        if 'constraint' in sig.args:
            kwargs['constraint'] = flow_l.constraint

        if 'set_restrictions' in sig.args:
            kwargs['set_restrictions'] = flow_l.set_restrictions

        if 'approx_sign' in sig.args:
            kwargs['approx_sign'] = flow_l.approx_sign

        if 'jitter' in sig.args:
            kwargs['jitter'] = flow_l.jitter

        if params is None:
            random_flows_params = np.random.randn(num_params)
        else:
            random_flows_params = params

        if len(kwargs.keys()) == 0:
            empty_flow = flow_l.__class__(
                *random_flows_params
            )
        else:
            empty_flow = flow_l.__class__(
                *random_flows_params, 
                **kwargs
            )
        
        return empty_flow

    if flow.__class__ is CompositeFlow:
        empty_arr = []
        
        for flow_l in flow.flow_arr:
            empty_flow = get_flow(flow_l)
            empty_arr.append(empty_flow)

        empty_flow = CompositeFlow(empty_arr)

    elif flow.__class__ is StepFlow:
        empty_arr = []
        
        for flow_l in flow.flow_arr:
            empty_flow = get_flow(flow_l)
            empty_arr.append(empty_flow)

        empty_flow = StepFlow(empty_arr, add_init_f0=flow.add_init_f0)
    else:
        empty_flow = get_flow(flow)

    return empty_flow

def newton_flow_inverse(t, flow):
    """
        Not all flows have the inverse implemented so approximate it with netwton raphson optimisation
    """
    if type(t) is torch.Tensor:
        t = t.detach().to('cpu').numpy()
    inv =  scipy.optimize.newton(lambda x: flow.forward(torch.tensor(x)).detach().numpy()-t, np.zeros_like(t))
    return inv

def find_flow_params_that_makes_marginals_standard_normal(y_output: torch.tensor, flow: Flow=None, step_1_optimizer_fn: typing.Callable=None, step_2_optimizer_fn: typing.Callable=None, step_1_num_epochs: int = None, step_2_num_epochs: int = None, step_2_num_restarts=1, seed=0, verbose=False, inv_flow: Flow=None, use_inverse_in_step_1: bool = False):
    """
        This fucntion optimizes the flow parameters that transforms the histogram of data into an (approx) standard normal.
        This does so by minimizing the KL between the true data generating process p(x) and a transformated standard normal:

        Let R = T^-1 (reverse/inverses flow), and N() be the standard norm then:
            
            KL[ p(x) || N(R(x)|dR/dx| ] = - E_p [ log N(R(x)|dR/dx| ] + E_p [ p ]

        The second term is constant w.r.t the flow parameters and so we only need to minimize the first term, the cross entropy term.

        NOTE: we first optimize the forward flow because optimizing the reverse flow is difficult. Then we fit a forward flow to the inverse of the original forward flow.
            This is convoluted but necessary until directly optimizing the reverse flow is possible.

    """

    #=============================SETUP ARGUMENTS=============================
    if flow is None:
        raise RuntimeError('flow must be specified')

    if step_1_optimizer_fn is None:
        warnings.warn("Using default optimizer for optim.SGD(trainable_params, lr=0.001, momentum=0.9)", Warning)
        step_1_optimizer_fn = lambda trainable_params:  optim.SGD(trainable_params, lr=0.001, momentum=0.9)
    
    if step_2_optimizer_fn is None:
        warnings.warn("Using default optimizer for step_2_optimizer_fn (optim.Adam(trainable_params, lr=0.01))", Warning)
        step_2_optimizer_fn = lambda trainable_params:  optim.Adam(trainable_params, lr=0.01)

    if step_1_num_epochs is None:
        warnings.warn("Using default number of step_1_num_epochs (100)", Warning)
        step_1_num_epochs = 100    

    if step_2_num_epochs is None:
        warnings.warn("Using default number of step_2_num_epochs (100)", Warning)
        step_2_num_epochs = 100    

    if inv_flow is None:
        inv_flow = flow

    #=============================DEFINE HELPER FUNCTIONS=============================
    def flow_jacobian(f, flow):

        if use_inverse_in_step_1:
            try:
                f_inv = flow.inverse(f)
            except NotImplementedError: 
                #flow inverse is not implemented
                f_inv = newton_flow_inverse(f, flow)
                f_inv = torch.tensor(f_inv)

            #x = 1/flow.forward_grad(f_inv) 
            x = flow.inverse_grad(f) 
        else:
            x = flow.forward_grad(f) 

        return x

    def cross_entropy(x, flow):
        """
            Implements  
                - E_p [ log N(R(x)|dR/dx| ] \approx (1/S) ∑ log N(R(s)|dR/ds|  for s ~ p(x)
            where s is just the input observations
        """
        N = Y=x.shape[0]
        if use_inverse_in_step_1:
            try:
                x_0 = flow.inverse(x)
            except NotImplementedError: 
                #flow inverse is not implemented
                x_0 = newton_flow_inverse(x, flow)
                x_0 = torch.tensor(x_0)
        else:
            x_0 = flow.forward(x)

        log_det = torch.mean(torch.log(torch.abs(flow_jacobian(x, flow)) + 1e-6))
        log_gauss = torch.mean(td.Normal(torch.zeros(x_0.shape), torch.ones(x_0.shape)).log_prob(x_0))

        ce = -(log_gauss+log_det)

        return ce

    #=============================GAUSSIANISATION=============================
    Y = torch.tensor(y_output)
    #Minmise the cross entrop
    trainable_params = []
    for n,p in inv_flow.named_parameters():
        trainable_params.append(p)
        
    optimizer = step_1_optimizer_fn(trainable_params)

    loss_arr = []

    if verbose:
        print('Starting step 1: optimising flow for gaussianiatity')

    if type(optimizer) is optim.LBFGS:
        for e in range(step_1_num_epochs):        
            def closure():
                optimizer.zero_grad()

                loss = cross_entropy(Y, inv_flow)

                loss.backward()

                return loss
            
            optimizer.step(closure)
            loss = cross_entropy(Y, inv_flow)

            if verbose:
                #Make it so we plot the last epoch
                _end = '\r'
                if e >= step_1_num_epochs-1:
                    _end = '\n'

                print('Step {e} - {loss}'.format(e=e, loss=loss.detach().numpy()), end=_end)
                
            loss_arr.append(loss.detach().numpy())
    else:
        for e in range(step_1_num_epochs):        
            loss = cross_entropy(Y, inv_flow)


            ## stochastic gradient update
            optimizer.zero_grad() # just in case
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                #Make it so we plot the last epoch
                _end = '\r'
                if e >= step_1_num_epochs-1:
                    _end = '\n'

                print('Step {e} - {loss}'.format(e=e, loss=loss.detach().numpy()), end=_end)
                
            
            loss_arr.append(loss.detach().numpy())

    if use_inverse_in_step_1:
        return None, inv_flow

    standard_normal_samples = np.random.randn(10000)

    if verbose:
        print('Starting step 2: finding forward flow that matches inverse flow')

    y_output = newton_flow_inverse(standard_normal_samples, flow)

    _flow, _ = find_forward_params(
        x_input = standard_normal_samples, 
        y_ouput = newton_flow_inverse(standard_normal_samples, inv_flow), 
        random_flow_fn = lambda: get_empty_flow(inv_flow), 
        num_restarts = step_2_num_restarts, 
        optimizer_fn = step_2_optimizer_fn,
        num_epochs = step_2_num_epochs,
        seed = seed,
        verbose=verbose
    )    

    return inv_flow, _flow


def initalize_step_flow_as_ladder(K=1, output_range=[],  smoothness_scale=0.1, remove_tails=False):
    """
        This function finds an approximately identiy step flow in the range given by output_range
            but with K steps. Such that the function resemebles a ladder.
    """
    softminus = lambda x: np.log(np.exp(x)-1)
    softplus = lambda x: np.log(np.exp(x)+1)
    
    output_diff = np.abs(output_range[1]-output_range[0])

    #find k points in range. ignore start and end of range.
    #these will be the points where the steps happen.

    if remove_tails:
        step_points = np.linspace(output_range[0], output_range[1], K)
    else:
        step_points = np.linspace(output_range[0], output_range[1], K+2)[1:-1]
    
    flow_arr = []
    for k in range(K):
        #if K>1 then shift tanh between (0, 2)
        shift=0
        if k == 0:
            #if K>1 then start tanh at output_range[0]
            shift = output_range[0]
            
        #each signmoid has the same height
        a = (output_diff/2)/K
        b = softminus(a)

        c = step_points[k]*smoothness_scale
        
        #smooth steps
        d = softminus(smoothness_scale)

        a = a+shift
        b = softplus(b)
        c = c/softplus(d)
        d = 1/softplus(d)

        _flow = TanhFlow(
            init_a=a,
            init_b=softminus(b),
            init_c=c,
            init_d=softminus(d),
            set_restrictions=True,
            add_init_f0=False
        )
        flow_arr.append(_flow)

    return StepFlow(flow_arr,add_init_f0=False)
