## The flows selected using a validation search

## Input dependent flows 
ID_TGP_boston = {
                    'flow_arch'               : 'SAL' ,      ## flow functional
                    'num_blocks'              : 1     ,      ## number of composite flows
                    'num_steps'               : None  ,      ## number of steps from the linear combination of the flow (e.g flow from Snelson et al 2003)
                    'flow_hidden_act'         : 'tanh',      ## activation of the Neural Network in input dependent flows
                    'flow_num_hidden_layers'  : 1     ,      ## number of hidden layers
                    'flow_DR'                 : 0.5   ,      ## dropout probability
                    'flow_BN'                 : 0     ,      ## batch normalization
                    'flow_hidden_dim'         : 25    ,      ## number of neurons per layer
                    'flow_inference'          : 'MC_dropout' ## Inference
                }


ID_TGP_power  = {
                    'flow_arch'               : 'SAL' ,      
                    'num_blocks'              : 3     ,      
                    'num_steps'               : None  ,      
                    'flow_hidden_act'         : 'relu',      
                    'flow_num_hidden_layers'  : 2     ,      
                    'flow_DR'                 : 0.25  ,      
                    'flow_BN'                 : 0     ,      
                    'flow_hidden_dim'         : 50    ,      
                    'flow_inference'          : 'MC_dropout' 
                }


## Non Input dependent flows
TGP_boston  = {
                'flow_arch'               : 'StepTanhL' ,      
                'num_blocks'              : 10          ,      
                'num_steps'               : 2           ,      
                'flow_hidden_act'         : None        ,      
                'flow_num_hidden_layers'  : None        ,     
                'flow_DR'                 : None        ,     
                'flow_BN'                 : None        ,
                'flow_hidden_dim'         : None        ,
                'flow_inference'          : None        ,
              }



TGP_power  = {
                'flow_arch'               : 'SAL'       ,      
                'num_blocks'              : 2           ,      
                'num_steps'               : None        ,      
                'flow_hidden_act'         : None        ,      
                'flow_num_hidden_layers'  : None        ,     
                'flow_DR'                 : None        ,     
                'flow_BN'                 : None        ,
                'flow_hidden_dim'         : None        ,
                'flow_inference'          : None        ,
             }

SVGP        = {
                'flow_arch'               : None ,      
                'num_blocks'              : None ,      
                'num_steps'               : None ,      
                'flow_hidden_act'         : None ,      
                'flow_num_hidden_layers'  : None ,     
                'flow_DR'                 : None ,     
                'flow_BN'                 : None ,
                'flow_hidden_dim'         : None ,
                'flow_inference'          : None ,
              }



def return_hyperparams(model,dataset):

    if model == 'ID_TGP' and dataset == 'boston':
        return ID_TGP_boston.values()

    elif model == 'TGP' and dataset == 'boston':
        return TGP_boston.values()

    elif model == 'ID_TGP' and dataset == 'power':
        return ID_TGP_power.values()

    elif model == 'TGP' and dataset == 'power':
        return TGP_power.values()

    elif model == 'SVGP':
        return SVGP.values()


