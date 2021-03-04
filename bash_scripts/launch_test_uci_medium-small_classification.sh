SEED=( 1 2 3 4 5 6 7 8 9 10 ) # seed for random train-test split
NUM_Z=( 100 )                 # number of inducing points for ablation
device=None                   # place here the id of your GPU if you want to use GPU
MEDIUM_CLASSIFICATION_DATASETS_with_random_test_split=( 'movement' 'heart' 'banknote' 'activity' )
MEDIUM_CLASSIFICATION_DATASETS_with_fixed_test_split=( 'avila' )

## Common hyperparameters to ALL experiments
lr=0.01
hold_K_params=2000  # hold kernel parameters during $hkp epochs following Hensman 2015b. This parameter has been previously searched on a validation split.

## =============================== ##
## =============================== ##
## SVTGP Juan and Ollie et al 2020 ##
## =============================== ##
## =============================== ##

## ========================= ##
## NON INPUT DEPENDENT FLOWS ##
## ========================= ##

if [[ "$dataset" = 'movement' ]]
then

    # MOVEMENT SP_ArcSL_2_blocks_None_steps
    flow_arch='ArcSL'
    num_blocks=2
    num_steps="None"

elif [[ "$dataset" = 'heart' ]]
then

    # HEART SP_SAL_InvBCL_1_block_None_steps
    flow_arch='SAL_InvBCL'
    num_blocks=1
    num_steps="None"

elif [[ "$dataset" = 'banknote' ]]
then

    # BANKNOTE SP_BCL_AL_5_blocks_None_steps
    flow_arch='BCL_AL'
    num_blocks=5
    num_steps="None"

elif [[ "$dataset" = 'avila' ]]
then

    # AVILA SP_SAL_AL_1_blocks_None_steps
    flow_arch='SAL_AL'
    num_blocks=1
    num_steps="None"     

elif [[ "$dataset" = 'activity' ]]
then

    # ACTIVITY SP_BCL_AL_1_blocks_None_steps
    flow_arch='BCL_AL'
    num_blocks=1
    num_steps="None"    

else
    echo "This dataset does not exist"
    exit
fi

 
## ========================= ##
##   INPUT DEPENDENT FLOWS   ##
## ========================= ##



if [[ "$dataset" = 'movement' ]]
then

    # MOVEMENT SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #          hidden_act_relu_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='relu'
    num_H=2
    DR=0.25
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'heart' ]]
then

    # HEART SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #       hidden_act_tanh_num_hidden_layers_1_batch_norm_False_dropout_0.5_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=1
    DR=0.5
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'banknote' ]]
then

    # BANKNOTE SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #          hidden_act_tanh_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=2
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'avila' ]]
then

    # AVILA SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #       hidden_act_tanh_num_hidden_layers_1_batch_norm_False_dropout_0.75_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=1
    DR=0.75
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'activity' ]]
then

    # ACTIVITY SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #          hidden_act_tanh_num_hidden_layers_1_batch_norm_False_dropout_0.75_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=1
    DR=0.75
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

else
    echo "This dataset does not exist"
    exit
fi


