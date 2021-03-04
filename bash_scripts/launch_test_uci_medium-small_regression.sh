SEED=( 1 2 3 4 5 6 7 8 9 10 ) # seed for random train-test split
NUM_Z=( 100 50 20 10 5 )      # number of inducing points for ablation
NUM_H_LAYERS=( 1 2 3 )        # number of hidden layers for DGP 
device=None                   # place here the id of your GPU if you want to use GPU
MEDIUM_REGRESSION_DATASETS=( 'boston' 'concrete' 'energy' 'kin8nm' 'naval' 'power' 'wine_red' 'wine_white' 'protein' )

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

if [[ "$dataset" = 'boston' ]]
then

    # BOSTON SP_StepTanhL_10_blocks_2_steps
    flow_arch='StepTanhL'
    num_blocks=10
    num_steps=2

elif [[ "$dataset" = 'concrete' ]]
then

    # CONCRETE SP_StepInvBoxCoxL_5_blocks_2_steps
    flow_arch='StepInvBoxCoxL'
    num_blocks=5
    num_steps=2

elif [[ "$dataset" = 'energy' ]]
then

    # ENERGY SP_StepTanhL_15_blocks_4_steps
    flow_arch='StepTanhL'
    num_blocks=15
    num_steps=4     

elif [[ "$dataset" = 'kin8nm' ]]
then

    # KIN8NM SP_InvBCL_1_blocks_None_steps
    flow_arch='InvBCL'
    num_blocks=1
    num_steps="None"     

elif [[ "$dataset" = 'naval' ]]
then

    # NAVAL SP_SAL_InvBCL_1_blocks_None_steps
    flow_arch='SAL_InvBCL'
    num_blocks=1
    num_steps="None"    

elif [[ "$dataset" = 'power' ]]
then

    # POWER SP_SAL_2_blocks_None_steps
    flow_arch='SAL'
    num_blocks=2
    num_steps="None"   

elif [[ "$dataset" = 'protein' ]]
then

    # PROTEIN SP_StepTanhL_10_blocks_2_steps
    flow_arch='StepTanhL'
    num_blocks=10
    num_steps=2

elif [[ "$dataset" = 'wine_red' ]]
then

    # WINE RED SP_SAL_3_blocks_None_steps
    flow_arch='SAL'
    num_blocks=3
    num_steps="None"
       
elif [[ "$dataset" = 'wine_white' ]]
then

    # WINE WHITE SP_SAL_BCL_10_blocks_None_steps
    flow_arch='SAL_BCL'
    num_blocks=10
    num_steps="None"
         
else
    echo "This dataset does not exist"
    exit
fi


## ========================= ##
##   INPUT DEPENDENT FLOWS   ##
## ========================= ##


if [[ "$dataset" = 'boston' ]]
then

    # BOSTON SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #        hidden_act_tanh_num_hidden_layers_1_batch_norm_False_dropout_0.5_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='tanh'
    num_H=1
    DR=0.5
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'concrete' ]]
then

    # CONCRETE SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #          hidden_act_relu_num_hidden_layers_1_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=1
    num_steps="None"
    act_H='relu'
    num_H=1
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'energy' ]]
then

    # ENERGY SP_SAL_3_blocks_None_steps_input_dependent_True                      
    #        hidden_act_relu_num_hidden_layers_2_batch_norm_False_dropout_0.5_hidden_dim_50
    num_blocks=3
    num_steps="None"
    act_H='relu'
    num_H=2
    DR=0.5
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'kin8nm' ]]
then

    # KIN8NM SP_SAL_3_blocks_None_steps_input_dependent_True                      
    #        hidden_act_relu_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=3
    num_steps="None"
    act_H='relu'
    num_H=2
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'naval' ]]
then

    # NAVAL SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #       hidden_act_relu_num_hidden_layers_1_batch_norm_False_dropout_0.5_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='relu'
    num_H=1
    DR=0.5
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'power' ]]
then

    # POWER SP_SAL_3_blocks_None_steps_input_dependent_True                      
    #       hidden_act_relu_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=3
    num_steps="None"
    act_H='relu'
    num_H=2
    DR=0.25
    BN=0
    dim_H=50
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'protein' ]]
then

    # PROTEIN SP_SAL_1_blocks_None_steps_input_dependent_True                      
    #         hidden_act_relu_num_hidden_layers_1_batch_norm_False_dropout_0.25_hidden_dim_25
    num_blocks=1
    num_steps="None"
    act_H='relu'
    num_H=1
    DR=0.25
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'wine_red' ]]
then

    # WINE-RED SP_SAL_3_blocks_None_steps_input_dependent_True                      
    #          hidden_act_tanh_num_hidden_layers_1_batch_norm_False_dropout_0.25_hidden_dim_25
    num_blocks=3
    num_steps="None"
    act_H='tanh'
    num_H=1
    DR=0.25
    BN=0
    dim_H=25
    NNet_inference='MC_dropout'
    input_dependent=$act_H" "$num_H" "$DR" "$BN" "$dim_H" "$NNet_inference

elif [[ "$dataset" = 'wine_white' ]]
then

    # WINE-WHITE SP_SAL_3_blocks_None_steps_input_dependent_True                      
    #            hidden_act_tanh_num_hidden_layers_2_batch_norm_False_dropout_0.25_hidden_dim_50
    num_blocks=3
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

            

