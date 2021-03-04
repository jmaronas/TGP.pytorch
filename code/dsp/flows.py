#-*- coding: utf-8 -*-
# flow.py -> interface to return flows
# Author: Juan Maroñas and  Ollie Hamelijnck

import torch
from gpytorch.utils.transforms import inv_softplus
import numpy
import numpy as np
from .models.flow import *

def common_config(options):
    if 'set_res' in options.keys():
        set_res = options['set_res']
    else:
        set_res = False

    if 'add_f0' in options.keys():
        addf0 = options['add_f0']
    else:
        addf0 = False

    if 'init_random' in options.keys():
        init_random = options['init_random']
    else:
        init_random = False

    if 'constraint' in options.keys():
        constraint = options['constraint']
    else:
        constraint = None

    return set_res, addf0, init_random, constraint

def set_input_dependent_config(options):
    input_dependent_config = {}
    # BN,DR,H,act,num_H,inference = 0,0.0,input_dim,'relu',1,MC_dropout # these are the default input dependent parameters of the neural network if we return an empty input_dependent_config dict

    input_dependent = False
    if 'input_dependent' in options.keys():
        input_dependent = bool(options['input_dependent'])

    if input_dependent:
        assert 'input_dim' in options.keys(), "You set to use input_dependent flows but the input dimension is not provided."

    if 'input_dim' in options.keys():
        input_dim = options['input_dim']
    else:
        input_dim = -1

    if 'batch_norm' in options.keys():
        input_dependent_config['batch_norm'] = options['batch_norm']

    if 'dropout'    in options.keys():
        input_dependent_config['dropout']    = options['dropout']

    if 'hidden_dim' in options.keys():
        input_dependent_config['hidden_dim'] = options['hidden_dim']

    if 'hidden_activation' in options.keys():
        input_dependent_config['hidden_activation'] = options['hidden_activation']

    if 'num_hidden_layers' in options.keys():
        input_dependent_config['num_hidden_layers'] = options['num_hidden_layers']

    if 'inference' in options.keys():
        input_dependent_config['inference'] = options['inference']


    return input_dependent, input_dim, input_dependent_config

def build_chain(flow_combination, num_blocks, **kwargs):
    # @TODO: create a way to learn chains of combined flows. Might worth do it for a large scale evaluation/combination. For the moment 
    # We use it just as a shortcut to return some combinations

    if flow_combination   == 'SAL_BCL':
        block_array = []
        constraint = kwargs['constraint']
        for nb in range(num_blocks):
            block_array.extend(SAL(1))
            block_array.extend(BoxCoxL(1,constraint = constraint))

    elif flow_combination   == 'SAL_InvBCL':
        block_array = []
        constraint = kwargs['constraint']
        for nb in range(num_blocks):
            block_array.extend(SAL(1))
            block_array.extend(InverseBoxCoxL(1, constraint = constraint))

    elif flow_combination == 'SAL_AL':
        block_array = []
        for nb in range(num_blocks):
            block_array.extend(SAL(1))
            block_array.extend(ArcSL(1))

    elif flow_combination == 'BCL_AL':
        constraint = kwargs['constraint']
        block_array = []
        for nb in range(num_blocks):
            block_array.extend(BoxCoxL(1,constraint = constraint))
            block_array.extend(ArcSL(1))

    elif flow_combination   == 'InvBCL_AL':
        constraint = kwargs['constraint']
        block_array = []
        for nb in range(num_blocks):
            block_array.extend(InverseBoxCoxL(1, constraint = constraint))
            block_array.extend(ArcSL(1))

    return block_array

# ====== SAL FLOW Generator ====== #
## SA + L ( sinh_arcsinh + affine )

# This initialization makes the flow a linear function
def SAL(num_blocks, **kwargs):

    set_res, addf0, init_random, constraint            = common_config(kwargs)
    input_dependent, input_dim, input_dependent_config = set_input_dependent_config(kwargs)
    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
            a_sal,b_sal = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0
            a_sal,b_sal = 0.0,1.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': set_res}
        init_sinh_arcsinh = {
                              'init_a': a_sal, 'init_b': b_sal,   'add_init_f0': addf0, 'set_restrictions': set_res, 
                              'input_dependent': input_dependent, 'input_dim': input_dim, 'input_dependent_config' : input_dependent_config
                            }
        block = [ ('sinh_arcsinh',init_sinh_arcsinh), ('affine',init_affine)  ] 
        block_array.extend(block) 

    return block_array

# ====== BoxCox FLow Generator ====== #
## BoxCox + affine
def BoxCoxL(num_blocks,**kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
            init_lam    = numpy.random.randn(1)+1. # center it around 1.0 which is the one that makes the flow the identity 
            constraint  = None 
        else:
            a_aff,b_aff = 1.0,0.0
            init_lam    = 5.0 # if applied the constraint below makes the flow the identity
            #def constraint(lam): 
            #    lam = 2*torch.sigmoid(0.3*lam-1.5)+0.05
            #    if lam == 0:
            #        lam = lam + 1e-11
            #        return lam

        init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': set_res}

        block = [ ('boxcox',init_bc), ('affine',init_affine)  ] 
        block_array.extend(block) # note that we append instead of extend because we will add each block per add_fow step

    return block_array

# ====== Inverse BoxCox Flow Generator ====== #
## InverseBoxCox + affine
def InverseBoxCoxL(num_blocks,**kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
            init_lam    = numpy.random.randn(1)+1. # center it around 1.0 which is the one that makes the flow the identity 
        else:
            a_aff,b_aff = 1.0,0.0
            init_lam    = 5.0 # if applied the constraint below makes the flow the identity
            # def constraint(lam): 
            #    lam = 2*torch.sigmoid(0.3*lam-1.5)+0.05
            #    if lam == 0:
            #        lam = lam + 1e-11
            #        return lam

        init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': set_res}

        block = [ ('inverseboxcox',init_bc), ('affine',init_affine)  ] 
        block_array.extend(block) # note that we append instead of extend because we will add each block per add_fow step

    return block_array


# ===== ArcSinh Flow Generator ===== #
## Arcsinh + affine
def ArcSL(num_blocks,**kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)

    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a_aff,b_aff             = numpy.random.randn(2)
            a_arc,b_arc,c_arc,d_arc = numpy.random.randn(4)
        else:
            a_aff,b_aff = 1.0,0.0
            a_arc,b_arc,c_arc,d_arc = numpy.random.randn(4)
            b_arc += 1
            d_arc += 1

        init_arcsinh = {'init_a': a_arc, 'init_b': b_arc, 'init_c': c_arc, 'init_d': d_arc, 'add_init_f0': addf0, 'set_restrictions': set_res}
        init_affine  = {'init_a': a_aff, 'init_b': b_aff, 'set_restrictions': set_res}

        block = [ ('arcsinh',init_arcsinh), ('affine',init_affine)  ] 
        block_array.extend(block) # note that we append instead of extend because we will add each block per add_fow step

    return block_array

# ====== AFFINE FLOW Generator ====== #
## L ( affine )

# This initialization makes the flow a linear function
def Affine(num_blocks, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)

    block_array = []
    for nb in range(num_blocks):
        if init_random:
            a,b = numpy.random.randn(2)
        else:
            a,b = 1.0,0.0
        init_affine = {'init_a':a, 'init_b': b, 'set_restrictions': set_res}
        block = [ ('affine',init_affine)  ] 
        block_array.extend(block) # note that we append instead of extend because we will add each block per add_fow step

    return block_array


# ====== StepTANH Flow Generator ====== #
## StepTanh + L (Tanh flow + affine flow)
## sum_i [ a_i + b_i*(tanh((f-c_i)/d_i)) ]*a + b
def StepTanhL(num_blocks, num_steps, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step tanh flow set_res has to be True for num_steps > 1'
    set_res = True 

    input_dependent, input_dim, input_dependent_config = set_input_dependent_config(kwargs)

    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]
        for st in range(num_steps): # each step in the linear combination should be initialized to different values otherwise gradients 
                                    # will always point in the same direction and the parameters will be equal after each gradient update.
            # w = numpy.log(numpy.exp(1./float(num_steps))-1) # softmin
            e1,e2,e3,e4 = numpy.multiply(numpy.random.randn(4,),numpy.array([1.0,1.0,1.0,1.0]))
            if not init_random:
                e2 = inv_softplus(torch.abs(torch.tensor( (e2+1.0) / float(num_steps)))).item()
                e4 = inv_softplus(torch.abs(torch.tensor( (e4+1.0) / float(num_steps)))).item()

            init_tanh = { 
                          'init_a':e1, 'init_b':e2, 'init_c':e3, 'init_d':e4, 'add_init_f0': False, 'set_restrictions' : set_res,
                          'input_dependent': input_dependent, 'input_dim': input_dim, 'input_dependent_config' : input_dependent_config
                        }

            step_flow_arr.append(('tanh',init_tanh))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step_tanh = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step_tanh), ('affine',init_affine)] 
        block_array.extend(block)

    return block_array



# ===== Step SA Flow generator ===== 
## Step SA + L -> linear combination of SAL flows plus affine flow

def StepSAL(num_blocks,num_steps, **kwargs):

    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step sa flow set_res has to be True for num_steps > 1'
    set_res = True 

    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]
        for st in range(num_steps): # each step in the linear combination should be initialized to different values otherwise gradients 
            a_sal,b_sal = numpy.random.randn(2)
            if not init_random:
                b_sal += 1.0
                b_sal = inv_softplus(torch.abs(torch.tensor(b_sal))).item()

            init_sinh_arcsinh = {'init_a': a_sal, 'init_b': b_sal, 'add_init_f0': False, 'set_restrictions': set_res}
            step_flow_arr.append(('sinh_arcsinh',init_sinh_arcsinh))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step_sa = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step_sa), ('affine',init_affine)] 
        block_array.extend(block)

    return block_array



# ====== StepArcsinh Flow Generator ====== #
## StepArcsinh + L (Arcsinh flow + affine flow)
## sum_i [ a_i + b_i*(arcsinh((f-c_i)/d_i)) ]*a + b
def StepArcSL(num_blocks, num_steps, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step tanh flow set_res has to be True for num_steps > 1'
    set_res = True 

    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]
        for st in range(num_steps): # each step in the linear combination should be initialized to different values otherwise gradients 
                                    # will always point in the same direction and the parameters will be equal after each gradient update.
            # w = numpy.log(numpy.exp(1./float(num_steps))-1) # softmin
            e1,e2,e3,e4 = numpy.multiply(numpy.random.randn(4,),numpy.array([1.0,1.0,1.0,1.0]))
            if not init_random:
                e2 = inv_softplus(torch.abs(torch.tensor((e2+1.0) / float(num_steps)))).item()
                e4 = inv_softplus(torch.abs(torch.tensor((e4+1.0) / float(num_steps)))).item()
            # apply inv_softplus to parameter b and d. As set_res = True, the flow applies the softplus to the parameters, hence for initialization we should use the inverse_softplus    
            init_arcsinh = {'init_a':e1, 'init_b': e2, 'init_c':e3, 'init_d': e4, 'add_init_f0': False, 'set_restrictions' : set_res}
            step_flow_arr.append(('arcsinh',init_arcsinh))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step_arcsinh = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step_arcsinh), ('affine',init_affine)] 
        block_array.extend(block)

    return block_array


# ====== StepBoxCox + L Flow Generator ====== #
def StepBoxCoxL(num_blocks,num_steps, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step tanh flow set_res has to be True for num_steps > 1'
    set_res = True 

    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]
        for st in range(num_steps): # each step in the linear combination should be initialized to different values otherwise gradients 

            init_lam = numpy.random.randn(1,)
            if not init_random:
                init_lam += 5.0

            init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
            step_flow_arr.append(('boxcox',init_bc))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0


        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step_box_cox = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step_box_cox), ('affine',init_affine)] 
        block_array.extend(block)

    return block_array

def StepInverseBoxCoxL(num_blocks,num_steps, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step tanh flow set_res has to be True for num_steps > 1'
    set_res = True 

    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]
        for st in range(num_steps): # each step in the linear combination should be initialized to different values otherwise gradients 

            init_lam = numpy.random.randn(1,)
            if not init_random:
                init_lam += 5.0

            init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
            step_flow_arr.append(('inverseboxcox',init_bc))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0


        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step_box_cox = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step_box_cox), ('affine',init_affine)] 
        block_array.extend(block)

    return block_array


def StepAllL(num_blocks, **kwargs):
    set_res, addf0, init_random, constraint = common_config(kwargs)
    if 'set_res' in kwargs.keys():
        assert kwargs['set_res']  == True, 'In the step tanh flow set_res has to be True for num_steps > 1'
    set_res = True 

    num_steps = 5
    block_array = []
    for nb in range(num_blocks):

        step_flow_arr=[]

        ## Inverse Box Cox
        init_lam = numpy.random.randn(1,)
        if not init_random:
            init_lam += 5.0

        init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
        step_flow_arr.append(('inverseboxcox',init_bc))

        ## Box Cox Flow
        init_lam = numpy.random.randn(1,)
        if not init_random:
            init_lam += 5.0

        init_bc = {'init_lam': init_lam, 'add_init_f0': addf0 , 'constraint': constraint}
        step_flow_arr.append(('boxcox',init_bc))


        ## Arcsinh Flow
        e1,e2,e3,e4 = numpy.multiply(numpy.random.randn(4,),numpy.array([1.0,1.0,1.0,1.0]))
        if not init_random:
            e2 = inv_softplus(torch.abs(torch.tensor((e2+1.0) / float(num_steps)))).item()
            e4 = inv_softplus(torch.abs(torch.tensor((e4+1.0) / float(num_steps)))).item()
        
        init_arcsinh = {'init_a':e1, 'init_b': e2, 'init_c':e3, 'init_d': e4, 'add_init_f0': False, 'set_restrictions' : set_res}
        step_flow_arr.append(('arcsinh',init_arcsinh))

        ## SAL Flow
        a_sal,b_sal = numpy.random.randn(2)
        if not init_random:
            b_sal += 1.0
            b_sal = inv_softplus(torch.abs(torch.tensor(b_sal))).item()

        init_sinh_arcsinh = {'init_a': a_sal, 'init_b': b_sal, 'add_init_f0': False, 'set_restrictions': set_res}
        step_flow_arr.append(('sinh_arcsinh',init_sinh_arcsinh))

        ## Tanh Flow
        e1,e2,e3,e4 = numpy.multiply(numpy.random.randn(4,),numpy.array([1.0,1.0,1.0,1.0]))
        if not init_random:
            e2 = inv_softplus(torch.abs(torch.tensor( (e2+1.0) / float(num_steps)))).item()
            e4 = inv_softplus(torch.abs(torch.tensor( (e4+1.0) / float(num_steps)))).item()

        init_tanh = {'init_a':e1, 'init_b':e2, 'init_c':e3, 'init_d':e4, 'add_init_f0': False, 'set_restrictions' : set_res}
        step_flow_arr.append(('tanh',init_tanh))

        if init_random:
            a_aff,b_aff = numpy.random.randn(2)
        else:
            a_aff,b_aff = 1.0,0.0

        init_affine = {'init_a':a_aff, 'init_b': b_aff, 'set_restrictions': False}
        init_step   = {'flow_arr' : step_flow_arr, 'add_init_f0': addf0}

        block = [('step_flow',init_step), ('affine',init_affine)] 
        block_array.extend(block)

        return block_array

def get_flow_combinations_randomly_initalised(flow_names):
    if type(flow_names) is list:
        flow_arr = []
        easy_inv_flow_arr = []
        for flow in flow_names:
            flow_random, easy_inv_flow = get_flow_combinations_randomly_initalised(flow)
            flow_arr.append(flow_random)
            easy_inv_flow_arr.append(easy_inv_flow)
            
        flow_random = CompositeFlow(flow_arr)
        easy_inv_flow = CompositeFlow(easy_inv_flow_arr)
        return flow_random, easy_inv_flow
    
    #flow_names is a 'leaf' string now
    flow_name = flow_names
    
    if flow_name == 'affine':
        _a, _b = np.random.randn(2)
        flow_random =  AffineFlow(_a, _b, set_restrictions=True)
        easy_inv_flow = flow_random
    elif flow_name == 'arcsinh':
        _a, _b, _c, _d = np.random.randn(4)
        flow_random =  ArcsinhFlow(_a, _b, _c, _d, add_init_f0=False, set_restrictions=True)
        easy_inv_flow = flow_random
    elif flow_name == 'inverse_arcsinh':
        _a, _b, _c, _d = np.random.randn(4)
        flow_random = InverseArchsinhFlow(_a, _b, _c, _d, add_init_f0=False, set_restrictions=True)
        easy_inv_flow = flow_random
    elif flow_name == 'sinh_arcsinhflow':
        _a, _b = np.random.randn(2)
        flow_random = Sinh_ArcsinhFlow(_a, _b, add_init_f0=False, set_restrictions=True)
        easy_inv_flow = flow_random
    elif flow_name == 'inverse_sinh_arcsinhflow':
        _a, _b = np.random.randn(2)
        flow_random = Inverse_Sinh_ArcsinhFlow(_a, _b, add_init_f0=False, set_restrictions=True)
        easy_inv_flow = flow_random

    elif flow_name == 'exp':
        flow_random = ExpFlow()
        easy_inv_flow = flow_random
    elif flow_name == 'softplus':
        flow_random = SoftplusFlow()
        easy_inv_flow = flow_random

    elif flow_name == 'inverse_boxcox':
        n = 2.0
        #restrict lam to be between 0.01 and 2
        n = 2.0
        boxcox_constraint = lambda lam:  n*torch.sigmoid(lam)+0.01
        identity_init = 1.0

        flow_random = CompositeFlow([
            TranslationFlow(0),
            InverseBoxCoxFlow(0.01, add_init_f0=0.0, constraint =boxcox_constraint)
        ])
        easy_inv_flow = flow_random
    elif flow_name == 'step_flow':
        min_y = np.min(y_train)
        max_y = np.max(y_train)
        step_flow = initalize_step_flow_as_ladder(
            K=5,
            output_range=[min_y, max_y],
            smoothness_scale=0.01,
            remove_tails=False
        )
        step_flow = [instance_flow([f], is_composite=False)[0] for f in step_flow]
        step_flow = StepFlow(step_flow, add_init_f0=False)
        
        flow_random = step_flow
        easy_inv_flow = step_flow

    elif flow_name == 'tukey_right':
        _g, _h = np.random.randn(2)
        flow_random = TukeyRightFlow(init_g=_g, init_h = _h, add_init_f0=False)
        easy_inv_flow = flow_random

    else:
        print('Flow ', flow_name, ' not found')
        
    return flow_random, easy_inv_flow
