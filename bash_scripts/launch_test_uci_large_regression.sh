NUM_Z=( 100 )                 # number of inducing points for ablation
NUM_H_LAYERS=( 1 2 3 )        # number of hidden layers for DGP 
device=None                   # place here the id of your GPU if you want to use GPU
MEDIUM_REGRESSION_DATASETS=( 'year' 'airline' )

## Common hyperparameters to ALL experiments
lr=0.01

## =============================== ##
## =============================== ##
## SVTGP Juan and Ollie et al 2020 ##
## =============================== ##
## =============================== ##

## ========================= ##
## NON INPUT DEPENDENT FLOWS ##
## ========================= ##

if [[ "$dataset" = 'year' ]]
then

    # YEAR SP_SAL_5_blocks_None_steps
    flow_arch='SAL'
    num_blocks=5
    num_steps="None"

elif [[ "$dataset" = 'airline' ]]
then

    # AIRLINE SP_StepTanhL_5_blocks_6_steps
    flow_arch='StepTanhL'
    num_blocks=5
    num_steps=6
         
else
    echo "This dataset does not exist"
    exit
fi


## ========================= ##
##   INPUT DEPENDENT FLOWS   ##
## ========================= ##

if [[ "$dataset" = 'year' ]]
then

    # YEAR SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #      hidden_act_tanh_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=2
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'airline' ]]
then

    # AIRLINE SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #      hidden_act_tanh_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=2
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

else
    echo "This dataset does not exist"
    exit
fi



