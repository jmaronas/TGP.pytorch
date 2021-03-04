import sys
import copy
sys.path.extend(['../'])

import torch

from dsp.models.flow import instance_flow
from dsp.flows import SAL, StepTanhL

## ================================ ##
## ================================ ##
## ================================ ##
##  Acts as a wrapper for returning
##  different flow combinations
##  with specified blocks and steps.

def return_flow_architecture(flow_arch, num_blocks, num_steps, kwargs):
    # Return different things. If the flow can be initialized to the identity it returns the format in string mode. Otherwise it returns a ranfom_flow_fn function to be used with the initializer. 

    run_initializer = False
    random_flow_fn  = None
    # SAL flow 
    if flow_arch == 'SAL':
        assert num_steps is None, "Num_steps has to be None for {} flow".format(flow_arch) 
        flow_specs = SAL(num_blocks, **kwargs)

    # Step Tanh 
    elif flow_arch == 'StepTanhL':
        run_initializer = True
        def random_flow_fn():    
            flow_list = StepTanhL(num_blocks,num_steps, add_f0 = True, **kwargs)
            instance                 = instance_flow(flow_list)
            instance.input_dependent = False
            return instance

    else:
        raise NotImplementedError("Unrecognized flow argument {}".format(flow_arch))

    if run_initializer:
        flow_specs = None

    return flow_specs, random_flow_fn, run_initializer



