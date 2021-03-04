#-*- coding: utf-8 -*-
# sparse_MF_SP.py : file containing base class for the sparse SP model.
# Author: Juan Maroñas and  Ollie Hamelijnck

# Python
import sys
sys.path.extend(['../'])

# Standard
import numpy
import numpy as np
import matplotlib.pyplot as plt

#Typing
import typing
from typing import List, Optional

# Torch
import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence

# Gpytorch
import gpytorch  as gpy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.kernels import AdditiveKernel
from gpytorch.utils.broadcasting import _pad_with_singletons
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D

# Pytorch Library
from pytorchlib import compute_calibration_measures

# Custom
from .. import config as cg
from ..utils import batched_log_Gaussian, add_jitter_MultivariateNormal, psd_safe_cholesky
from ..likelihoods import GaussianLinearMean, GaussianNonLinearMean, MulticlassCategorical, Bernoulli

# Module specific
from .config_models import get_init_params
from .utils_models import return_mean, return_projection_matrix, enable_eval_dropout
from .flow import instance_flow


## Sparse TGP. Warping prior p(f) using normalizing flows
class sparse_MF_SP(nn.Module):
    def __init__(self,model_specs: list, X: torch.tensor, init_Z: torch.tensor, N: float, likelihood : nn.Module, num_outputs: int, is_whiten: bool, K_is_shared: bool, mean_is_shared: bool, Z_is_shared: bool, q_U_is_shared: bool , flow_specs: list, flow_connection: str, add_noise_inducing: float, be_fully_bayesian : bool = False, init_params: dict={}) -> None:
        """
                Args: 
                        :attr:  `model_specs`         (list)         :->: tuple (A,B) where A is a string representing the mean used and B is a kernel instance.
                                                                          This kernel instance should have batch_shape = number of outputs if K_is_shared = False,
                                                                          and batch_shape = 1 else. For the moment all the GPs at a layer shared the functional form of these.
                                `X`                   (torch.tensor) :->: Full training set (or subset) of samples used for the SVD 
                                `init_Z`              (torch.tensor) :->: initial inducing point locations
                                `N`                   (float)        :->: total training size	
                                `likelihood`          (nn.Module)    :->: Likelihood instance that will depend on the task to carry out
                                `num_outputs`         (int)          :->: number of output GP. The number of inputs is taken from the dimensionality of init_Z
                                `is_whiten`           (bool)         :->: use whitened representation of inducing points.
                                `K_is_shared`         (bool)         :->: True if the covariance of the output GPs are shared
                                `mean_is_shared`      (bool)         :->: True if the mean of the output GPs are shared
                                `Z_is_shared`         (bool)         :->: True if the inducing point locations are shared
                                `q_U_is_shared`       (bool)         :->: True if the variational distribution is shared
                                `add_noise_inducing`  (float)        :->: Standard desviation of Gaussian noise added to init_Z
                                `flow_specs`          (list)         :->: A list of list containing lists of strings (or flows instances) specifying the composition 
                                                                          and interconnection of flows per output dimension. If flow connection is shared then pass a 
                                                                          list containing a single list specifying the flow to use. If flow connection is single then
                                                                          the list contains num_output lists, specifying the flows per output GP.
                                `flow_connection`     (str)          :->: Specifies how flows are connected. Possibilities are: shared, single
                                                                             shared : all GPs share the same flow
                                                                             single : each GP has a unique flow
                                `be_fully_bayesian`   (bool)         :->: If true, then input dependent flows are integrated with monte carlo dropout when possible
                                `init_params`         (dict)         :->: Initial parameters of model. If not use the defaults in config_models.py


          # -> Some notes for deciding if whitenning or not: https://gpytorch.readthedocs.io/en/latest/variational.html#gpytorch.variational.UnwhitenedVariationalStrategy	
        """
        super(sparse_MF_SP, self).__init__()
        ## ==== Check assertions ==== ##
        assert len(model_specs) == 2, 'Parameter model_specs should be len 2. First position string with the mean and second position string with the kernels'

        ## ==== Config Variables ==== ##
        self.out_dim          = int(num_outputs)    # output dimension
        self.inp_dim          = int(init_Z.size(1)) # input dimension
        self.kernel_is_shared = K_is_shared         # if the kernel is shared between GPs (done in DSVI code)
        self.mean_is_shared   = mean_is_shared      # if the mean is shared. 
        self.Z_is_shared      = Z_is_shared         # if the inducing points are shared 
        self.q_U_is_shared    = q_U_is_shared       # if the variational distribution is shared.
        self.N                = float(N)            # training size
        self.M                = init_Z.size(0)      # number of inducing points
        self.likelihood       = likelihood
        
        self.fully_bayesian   = be_fully_bayesian
        self.init_params = get_init_params(init_params) # get init params of model

        ## ==== Tools ==== ##
        self.standard_sampler = td.MultivariateNormal(torch.zeros(1,).to(cg.device),torch.eye(1).to(cg.device))  # used in the reparameterization trick.
        self.is_training      = True # This flag controls where the model is in train or test mode. We cant use the self.eval() self.train() provided by PyTorch for our model as we need to activate
                                     # dropout during inference for the Bayesian flows

        if isinstance(self.likelihood, GaussianNonLinearMean):
            self.quad_points = self.likelihood.quad_points 
            self.quad        = GaussHermiteQuadrature1D(self.quad_points) # quadrature integrator. 
        else:
            self.quad_points = cg.quad_points
            self.quad        = GaussHermiteQuadrature1D(self.quad_points)
            

        ## ==== Set the Model ==== ##
        # Variational distribution
        self.initialize_inducing(init_Z, add_noise_inducing)	
        self.initialize_variational_distribution(is_whiten)

        # Model distribution
        self.initialize_mean_function(X, model_specs)
        self.initialize_covariance_function(model_specs)
        G_matrix,G_flow_connection =  self.initialize_flows(flow_connection,flow_specs)
        self.G_matrix          = G_matrix
        self.G_flow_connection = G_flow_connection

        self.l2_regularize = False

    ## ================================ ##
    ## == Some configuration Methods == ## 
    ## ================================ ##

    ## ========================================== ##
    ##  Set model to work in fully bayesian mode
    def be_fully_bayesian(self, mode):
        self.fully_bayesian = mode

    ## ========================================== ##
    ##  Set flag is training
    def set_is_training(self, mode):
        self.is_training = mode


    ## ================================================ ##
    ##   Variational distribution q(f,u) = p(f|u)q(u)   ##

    ### Inducing point locations Z ###
    def initialize_inducing(self,init_Z: torch.tensor, add_noise_inducing: float) -> None:
        """Initializes inducing points using the argument init_Z. It prepares this inducing points
           to be or not to be shared by all kernels.
        """
        if self.Z_is_shared:
            self.Z = nn.Parameter(init_Z.unsqueeze(dim = 0)) # this parameter is repeated when needed
        else:
            Z = torch.zeros(self.out_dim,self.M,self.inp_dim)
            for l in range(self.out_dim):
                aux_init = init_Z.clone()
                if add_noise_inducing > 0.0: ## add some noise to inducing points
                    power = numpy.random.randn( self.M, self.inp_dim )
                    aux_init = init_Z * torch.tensor(add_noise_inducing*power,dtype=cg.dtype).detach() # we detach just in case.
                Z[l,:] = aux_init
            self.Z = nn.Parameter(Z)

    ## Variational Distribution q(U)
    def initialize_variational_distribution(self,is_whiten:bool) -> None:
        """Initializes the GP variational distribution for work with batched inputs and not be or not to be shared.
           It initializes its parameters using the init_dict passed as argument or the default values at config.models
        """

        if self.q_U_is_shared:
            q_U =  CholeskyVariationalDistribution(self.M)
            o_d = 1
        else:
            q_U =  CholeskyVariationalDistribution(self.M,batch_shape = torch.Size([self.out_dim]))
            o_d = self.out_dim

        init_S_U = torch.eye(self.M, self.M).view(1,self.M,self.M).repeat(o_d,1,1)*numpy.sqrt(self.init_params['variational_distribution']['variance_scale']) # take sqrt as we parameterize the lower triangular
        init_m_U = torch.ones(o_d,self.M)*self.init_params['variational_distribution']['mean_scale']

        q_U.chol_variational_covar.data = init_S_U
        q_U.variational_mean.data = init_m_U

        self.is_whiten = is_whiten
        self.q_U = q_U


    ## ========================== ##
    ## === Model Distribution === ##

    ## == Mean function == ##
    def initialize_mean_function(self, X:torch.tensor, model_specs:list) -> None:
        """Initializes the mean function.

        """
        ### Mean Function ###
        ## Projection matrix for identity gpy.identity mean function
        name = model_specs[0]
        W = None
        if name == 'identity':
            # Get projection matrix. Cases of study:
              # 1. If num_inputs == num_outputs, the mean for each output GP is given by x_i i=1 : num_inputs
              # 2. If num_inputs > num_outputs, the mean for each output GP is given by x_i i=1 : num_outputs < num_inputs
              #  -> Salimbeni here performed SVD decomposition
              # 3. If num_inputs < num_outputs -> Some kind of padding must be performed.
            W = return_projection_matrix(self.inp_dim,  self.out_dim, X)

        mean_function = return_mean(name, self.inp_dim, self.out_dim, W) 

        if self.mean_is_shared:
            assert name == 'linear', "mean_is_shared = True only with Linear mean function, got {}".format(name)
            mean_function = return_mean(name,self.inp_dim,1,W)

        self.mean_function = mean_function


    ## == Covariance function == ##
    def initialize_covariance_function(self,model_specs) -> None:
        """Initializes the kernel using an already instanced one as given by the instance_kernel function in utils_models.py. This function double checks that
           the kernels has been properly initialized depending on wether the kernel is shared by all the GPs or not, i.e it checks its batch_shape argument.

        """
        if cg.strict_flag:
            if isinstance(model_specs[1],AdditiveKernel):
                for kernel in model_specs[1].kernels:
                    if self.kernel_is_shared:
                        assert kernel.batch_shape == torch.Size([1]), "Got a kernel with batch_shape {} but self.kernel_is_shared is True. Expected batch_shape 1 got {}".format(kernel.batch_shape)
                    else:
                        assert kernel.batch_shape == torch.Size([self.out_dim]), "Got a kernel with batch_shape {} but self.kernel_is_shared is False. Expected batch_shape {} got {}.".format(self.out_dim,kernel.batch_shape)

            else:
                kernel = model_specs[1]
                if self.kernel_is_shared:
                    assert model_specs[1].batch_shape == torch.Size([1]), "Got a kernel with batch_shape {} but self.kernel_is_shared is True. Expected batch_shape 1".format(kernel.batch_shape)
                else:
                    assert model_specs[1].batch_shape == torch.Size([self.out_dim]), "Got a kernel with batch_shape {} but self.kernel_is_shared is False. Expected batch_shape {}".format(kernel.batch_shape, self.out_dim)

        # model_specs[1] keeps the kernel object with the batch_shape set.
        self.covariance_function = model_specs[1] 

    ## == Flows == ##
    ## Initialize Flow. G (prior) in the paper ##
    def initialize_flows(self,flow_connection, flow_specs) -> None:
        """ Initializes the flows applied on the prior depending if G is shared or single. It works with flows passed as instances or flows passed through strings. 
            Flows should be passed as lists where the list contains a single list for the shared and num_output lists for single.

        """
        if   flow_connection == 'shared':
            assert len(flow_specs) == 1, "The number of elements provided in {} is different from 1 for option shared".format(len(flow_specs))

        elif flow_connection == 'single':
            assert len(flow_specs) == self.out_dim, "The number of elements provided in {} is different from the number of outputs {}".format(len(flow_specs),self.out_dim)

        else:
            raise ValueError("Invalid value {} for argument flow_connection".format(flow_connection))

        G_matrix = []
    
        if flow_connection == 'single':
            for idx,fl in enumerate(flow_specs):
                if type(fl) == list:
                    #flow has been defined by a list with flow name and parameters
                    G_matrix.append(instance_flow(fl))
                else:
                    G_matrix.append(fl)

        flow_connection = flow_connection
        G_matrix        = nn.ModuleList(G_matrix)

        # This is now returned so that this funciton can be used for prior and posterior
        # flows
        return G_matrix, flow_connection

    ## ============================ ##
    ## ==  FINISH CONFIG METHODS == ##
    ## ============================ ##

    ## =============================== ##
    ## == MODEL COMPUTATION METHODS == ## 
    ## =============================== ##

    def marginal_variational_qf_parameters(self, X : torch.tensor, diagonal : bool, is_duvenaud: bool, init_Z : torch.tensor = None) -> torch.tensor:
        """ Marginal Variational posterior q(f) = \int p(f|u) q(u) d_u
            q(f) = int p(f|u) q(u) d_u = N(f|K_xz K_zz_inv m + m_x -K_xz K_zz_inv \mu_z, 
                                                 K_xx -K_xz K_zz_inv K_zx + [K_xz K_zz_inv] S [K_xz K_zz_inv]^T)
                Args:
                        `X`           (torch.tensor)  :->:  input locations where the marginal distribution q(f) is computed. Can hace shape (S*MB,Dx) or (Dy,S*MB,Dx)
                        `diagonal`    (bool)          :->:  If true, return only the diagonal covariance
                        `is_duvenaud` (bool)          :->:  Indicate if we are using duvenaud mean function. Only useful in DGPs
                        `init_Z`      (torch.tensor)  :->:  Only used if is_duvenaud = True. It is used to concatenate the input inducing points to 
                                                            the inducing points at each layer

                Returns:
                        `mu_q_f`      (torch.tensor)  :->:  shape Dy,MB,1
                        `cov_q_f`     (torch.tensor)  :->:  shape (Dy,MB,1) if diagonal else (Dy,MB,MB)
        """
        ## ================================= ##
        ## ===== Pre-Compute Variables ===== ##
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to work batched and multioutput respectively
        assert len(X.shape) == 3, 'Invalid input X.shape' 

        Dy,MB,M  = self.out_dim,X.size(1),self.M
        Z        = self.Z

        kernel   = self.covariance_function
        mean     = self.mean_function

        if self.Z_is_shared:
            # In this case this repeat is not particulary needed because the kernel will repeat Z
            # when doing forward both if batch_shape is out_dim or is 1 (self.kernel_is_shared True)
            # Keep it explicitely for better understanding of the code.
            Z = Z.repeat(self.out_dim,1,1) 

        # Concatenate inducing points if is duvenaud
        if is_duvenaud:
            #z_concat = X[0,0:self.M,-1].view(self.M,1)
            init_Z    = init_Z.view(1,self.M,-1).repeat(self.out_dim,1,1)
            Z = torch.cat((Z,init_Z),2)

        K_xx = kernel(X,are_equal = True, diag = diagonal)
        mu_x = gpy.lazy.delazify( mean(X) ).view(Dy, MB, 1)

        K_zz = kernel(Z,are_equal = False).evaluate()
        mu_z = gpy.lazy.delazify( mean(Z) ).view(Dy, M , 1)

        K_xz = kernel(X,Z,are_equal = False).evaluate()

        # stabilize K_xz. In case Z = X we should add jitter if psd_safe_cholesky adds jitter to K_zz
        # jitter can only be added to square matrices
        if cg.maximum_precision and self.M == MB: 
            pass
            #_, K_xz  = psd_safe_cholesky(K_xz, upper = False) # The K_xz returned is that with noise

        K_zx = torch.transpose(K_xz,1,2) # pre-compute the transpose as it is required several times

        # cholesky from K_zz
        L_zz, K_zz  = psd_safe_cholesky(K_zz, upper = False, jitter = cg.global_jitter) # The K_zz returned is that with noise

        if self.is_whiten:
            L_zz_t = L_zz.transpose(1,2) 

        # variational distribution
        q_U    = self.q_U
        m_q_U  = q_U.variational_mean
        K_q_U  = q_U.chol_variational_covar
        
        if self.q_U_is_shared:
            m_q_U = m_q_U.repeat(Dy,1)
            K_q_U = K_q_U.repeat(Dy,1,1)

        lower_mask = torch.ones(K_q_U.shape[-2:], dtype=cg.dtype, device=cg.device).tril(0)
        L_q_U = K_q_U.mul(lower_mask)
        K_q_U = torch.matmul( L_q_U,L_q_U.transpose(1,2) )
        m_q_U  = m_q_U.view(Dy,M,-1)

        ## =================== ##
        ## ==== mean q(f) ==== ##

        if self.is_whiten:
            # mu_qf = K_{xz}[L_{zz}^T]^{-1}m_0+\mu_x
            sol,_ = torch.triangular_solve(m_q_U, L_zz_t, upper = True)
            mu_q_f = torch.matmul(K_xz,sol) + mu_x

        else:
            # mu_qf = K_xz K_zz_inv( m - mu_Z) + m_x
            lhs = torch.cholesky_solve(m_q_U-mu_z, L_zz, upper = False)
            mu_q_f = torch.matmul(K_xz,lhs) + mu_x

        
        ## ========================= ##
        ## ==== covariance q(f) ==== ##
        ## Note:
            # To compute the diagonal q(f) we perform the following identity. Here @ indicates matrix product and .* element-wise product
            # For K_xz @ K_zz_inv @ K_zx the diagonal is:
            #   sum(K_zx .* [K_zz_inv @ K_zx],0)
            # This means that the identity can be written down as:
            #  A @ B @ A^T = A^T .* [ B @ A^T ]							
            # For the covariance note that: [K_xz K_zz_inv] S [K_xz K_zz_inv]^T = [K_zz_inv K_zx]^T S [K_zz_inv K_zx] =
            # where the output of the linear solver is sol = [K_zz_inv K_zx]. So we have: sol^T S sol. Thus we have: sum(sol.*[S @ sol],0) to compute the diagonal
            # note that as the operations are batched we have to reduce dimension 1 instead of dimension 0. Also use matmul to perform the batched operation.

        # sol = K_zz^{-1}@K_zx
        sol = torch.cholesky_solve(K_zx, L_zz, upper = False)

        if self.is_whiten:
            # cov_qf = K_{xx} -K_{xz} K_{zz}^{-1} K_{zx} + K_{xz} {L_{zz}^T}^{-1} S L_{zz}^{-1}K_{zx} 
            rhs,_ = torch.triangular_solve(K_zx, L_zz, upper = False)
            if diagonal:
                cov_q_f = K_xx - torch.sum(torch.mul(K_zx,sol),1) + torch.sum(torch.mul(rhs,torch.matmul(K_q_U,rhs)),1)
            else:
                cov_q_f = K_xx - torch.matmul(K_xz,sol) + torch.matmul(torch.matmul(torch.transpose(rhs,1,2),K_q_U),rhs)

        else:
            # cov_qf = K_{xx} -K_{xz} K_{zz}^{-1} K_{zx} + [K_{xz} K_{zz}^{-1}] S [K_{xz} K_{zz}^{-1}]^T 
            if diagonal:
                cov_q_f = K_xx - torch.sum(torch.mul(K_zx,sol),1) + torch.sum(torch.mul(sol,torch.matmul(K_q_U,sol)),1)
            else:
                cov_q_f = K_xx - torch.matmul(K_xz,sol) + torch.matmul(torch.matmul(torch.transpose(sol,1,2),K_q_U),sol)

        if diagonal:
            cov_q_f = torch.unsqueeze(cov_q_f,2)

        return mu_q_f, cov_q_f

    def KLD(self) -> torch.tensor :
            """ Kullback Lieber Divergence between q(U) and p(U) 
                Computes KLD of all the GPs at a layer.
                Returns shape (Dy,) with Dy number of outputs GP
            """
            ## whitened representation of inducing points.
            # -> computations got from https://arxiv.org/pdf/2003.01115.pdf

            if self.is_whiten:

                q_U    = self.q_U
                m_q_V  = q_U.variational_mean
                K_q_V  = q_U.chol_variational_covar

                if self.q_U_is_shared:
                    m_q_V  = m_q_V.repeat(self.out_dim,1)
                    K_q_V = K_q_V.repeat(self.out_dim,1,1)

                # Variational mean
                m_q_V = m_q_V.view(self.out_dim,self.M,1)

                # Cholesky decomposition of K_vv
                lower_mask = torch.ones(K_q_V.shape[-2:], dtype=cg.dtype, device=cg.device).tril(0)
                L_q_V = K_q_V.mul(lower_mask)
                K_q_V = torch.matmul( L_q_V,L_q_V.transpose(1,2) )

                # KLD
                dot_mean = torch.matmul(m_q_V.transpose(1,2),m_q_V).squeeze()
                log_det_K_q_v = torch.log(torch.diagonal(L_q_V, dim1 = 1, dim2 = 2)**2).sum(1)

                #edit over comment: only true for diagonal matrix
                trace = torch.diagonal(K_q_V,dim1=-2,dim2=-1).sum(-1)

                KLD = 0.5*(-log_det_K_q_v + dot_mean + trace - float(self.M))

            else:
                Z = self.Z
                if self.Z_is_shared:
                    Z = Z.repeat(self.out_dim,1,1)

                ## Posterior
                q_U = self.q_U() # This call generates a torch.distribution.MultivaraiteNormal distribution with the parameters 
                                 # of the variational distribution given by:
                                 # q_mean_U = self.q_U.variational_mean
                                 # q_K_U    = self.q_U.chol_variational_covar


                ## Prior p(U)
                p_mean_U = gpy.lazy.delazify(self.mean_function(Z)).squeeze(-1)
                p_K_U    = self.covariance_function(self.Z, are_equal = False).evaluate() # are_equal = False. We dont add noise to the inducing points, only samples X 
                # shapes (Dy,M,1) and (Dy,M,M)
                #p_U = td.multivariate_normal.MultivariateNormal(p_mean_U,p_K_U)
                p_U = add_jitter_MultivariateNormal(p_mean_U, p_K_U)

                ## KLD -> use built in td.distributions
                KLD = kl_divergence(q_U,p_U)

            return KLD

    def predictive_distribution(self,X: torch.tensor, diagonal: bool=True, S_MC_NNet: int = None)-> list:
        """ This function computes the moments 1 and 2 from the predictive distribution. 
            It also returns the posterior mean and covariance over latent functions.

            p(Y*|X*) = \int p(y*|G(f*)) q(f*,f|u) q(u) df*,df,du
   
                # Homoceodastic Gaussian observation model p(y|f)
                # GP variational distribution q(f)
                # G() represents a non-linear transformation

                Args:
                        `X`                (torch.tensor)  :->: input locations where the predictive is computed. Can have shape (MB,Dx) or (Dy,MB,Dx)
                        `diagonal`         (bool)          :->: if true, samples are drawn independently. For the moment is always true.
                        `S_MC_NNet`        (int)           :->: Number of samples from the dropout distribution is fully_bayesian is true

                Returns:
                        `m1`       (torch.tensor)  :->:  Predictive mean with shape (Dy,MB)
                        `m2`       (torch.tensor)  :->:  Predictive variance with shape (Dy,MB). Takes None for classification likelihoods
                        `mean_q_f` (torch.tensor)  :->:  Posterior mean of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]
                        `cov_q_f`  (torch.tensor)  :->:  Posterior covariance of q(f) with shape (Dy,MB,1) [same shape as returned by marginal_variational_qf]

        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1)
        assert len(X.shape) == 3, "Bad input specificaton"

        assert not self.is_training, "This method only works in eval mode"

        self.eval() # set parameters for eval mode. Batch normalization, dropout etc
        if self.fully_bayesian:
            # activate dropout if required
            is_dropout = enable_eval_dropout(self.modules())
            assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

            assert S_MC_NNet is not None, "The default parameter S_MC_NNet is not provided and set to default None, which is invalid for self.be_bayesian" 

        with torch.no_grad():
            if not diagonal:
                raise NotImplemented("This function does not support returning the predictive distribution with correlations")

            mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = False, init_Z = None)

            if self.fully_bayesian: # @NOTE: this has not been refactored as with the rest of the code. But note that we could do both point estimate and bayesian by setting S_MC_NNet = 1 for the non
                                    #  bayesian case.
                # If it is fully Bayesian then do it as in the DGP with flows in the output layer
                Dy,MB,_ = mean_q_f.shape

                # 1. Reshape mean_q_f and cov_q_f to shape (Dy,S_MC_NNet*MB)
                mean_q_f_run = mean_q_f.view(Dy,MB).repeat(1,S_MC_NNet)
                cov_q_f_run  = cov_q_f.view(Dy,MB).repeat(1,S_MC_NNet)

                # 2. Compute moments of each of the montecarlos. Just need to provide X extended to S_MC so that each forward computes a monte carlo
                X = X.repeat(1,S_MC_NNet,1) # expand to shape (Dy,S*MB,Dx). 
                MOMENTS = self.likelihood.marginal_moments(mean_q_f_run, cov_q_f_run, self.G_matrix, X) # get the moments of each S*MB samples

                # 3. Compute the moments from the full predictive distribution, e.g the mixture of Gaussians for Gaussian Likelihood
                if isinstance(self.likelihood,GaussianNonLinearMean):
                    m_Y,C_Y = MOMENTS
                    m_Y = m_Y.view(Dy,S_MC_NNet,MB)
                    C_Y = C_Y.view(Dy,S_MC_NNet,MB)

                    m1 = m_Y.mean(1)
                    m2 = ( C_Y + m_Y**2 ).mean(1) - m1**2 # var = 1/S * sum[K_Y + mu_y^2 ] -[1/S sum m1]^2

                elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli):
                    m1,m2 = MOMENTS,None
                        
                    m1 = m1.view(S_MC_NNet,MB,Dy)
                    m1 = m1.mean(0) # reduce the monte carlo dimension

                else:
                    raise ValueError("Unsupported likelihood [{}] for class [{}]".format(type(self.likelihood),type(self)))

            else:

                MOMENTS = self.likelihood.marginal_moments(mean_q_f.squeeze(dim = 2), cov_q_f.squeeze(dim = 2), diagonal = True, flow = self.G_matrix, X = X) # diagonal True always. Is an element only used by the sparse_MF_GP with SVI. Diag = False is used by standard GP's marginal likelihood

                if isinstance(self.likelihood,GaussianLinearMean) or isinstance(self.likelihood,GaussianNonLinearMean):
                    m1,m2 = MOMENTS 
                elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood, Bernoulli):
                    m1,m2 = MOMENTS,None

        self.train() # switch back to train mode. 
        return m1,m2, mean_q_f, cov_q_f


    ## ====================================== ##
    ## == FINISH MODEL COMPUTATION METHODS == ## 
    ## ====================================== ##


    ## ====================== ##
    ## == TRAINING METHODS == ## 
    ## ====================== ##

    def ELBO(self,X: torch.tensor, Y: torch.tensor) -> torch.tensor:
        """ Evidence Lower Bound

            ELBO = \int log p(y|f) q(f|u) q(u) df,du -KLD[q||p] 

                Args:
                        `X` (torch.tensor)  :->:  Inputs
                        `Y` (torch.tensor)  :->:  Targets

            Returns possitive loss, i.e: ELBO = LLH - KLD; ELL and KLD

        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) # repeat here as if not this operation will be done twice by the marginal_qf_parameter and likelihood to work batched and multioutput respectively
        assert len(X.shape) == 3, 'Invalid input X.shape' 

        ## ============================= ##
        ## === Compute KL divergence === ##

        KLD = self.KLD()

        ## Get flow KLD for variational Bayes flow
        KLD_flow = 0.0
        for flow in self.G_matrix:
            KLD_flow += flow.KLD()

        ## ================================================= ##
        ## === Computes Variational posterior parameters === ##
        mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = False, init_Z = None)
            # mean_q_f shape (Dy,MB,1)
            # cov_q_f  shape (Dy,MB,1)

        ## =============================== ##
        ## Compute Expected Log Likelihood ##
        ELL = self.ELL(X = X, Y = Y, mean = mean_q_f, cov = cov_q_f)

        ## =============================== ##
        ## ======== Computes ELBO ======== ##
        ELL = ELL.sum()
        KLD = KLD.sum()

        ELBO = ELL - KLD - KLD_flow

        ## Accumulate Loss
        KLD  = KLD + KLD_flow # Return Possitive KLD

        return ELBO, ELL, KLD # returns positive ELBO. This will be change to negative to optimize


    def ELL(self, X: torch.tensor, Y: torch.tensor, mean: torch.tensor, cov: torch.tensor):
        """ Expected Log Likelihood w.r.t to Gaussian distribution given a non linear mean p(y|G(f))q(f)
            
            ELL = \int log p(y|G(f)) q(f|u) q(u) df,du

                Args:
                        `X`    (torch.tensor)  :->: Inputs, shape  (MB,Dx) or (Dy,MB,Dx)
                        `Y`    (torch.tensor)  :->: Targets, shape (MB,Dy)
                        `mean` (torch.tensor)  :->: Mean from q(f). Shape (Dy,MB,1)
                        `cov`  (torch.tensor)  :->: diagonal covariance from q(f). Shape (Dy,MB,1)

            Computes the stochastic estimator of the ELBO properly re-normalized by N/MB with N number of training points and MB minibatch.

        """
        ## ================ ##
        ## == Assertions == ##
        # Most of the methods of this class can be re-used by any flow base GP. However, the way in which the ELL handles the transformed GP requires specific task likelihoods.
        # In this case, each of the GP is assumed to be the mean of an independent Dy dimensional output Y. The user can simply overwrite the method ELL and the assertions
        # at this point to handle other tasks.
        assert self.G_flow_connection == 'single', "The current sparse_MF_SP only supports G_matrix being single, got {}. This class works by placing independent flows in each of the outputs. Overwrite  self.ELL method for performing other tasks.".format(self.G_flow_connection)
        assert isinstance(self.likelihood,GaussianNonLinearMean) or isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli) or isinstance(self.likelihood,GaussianLinearMean), "The current sparse_MF_SP only supports GaussianNonLinearMean likelihood and classification likelihoods. Overwrite this method to perform other tasks and pass the adequate likelihood."

        N,MB = self.N, Y.size(0)
        ELL = self.likelihood.expected_log_prob(Y.t(), mean.squeeze(dim = 2), cov.squeeze(dim = 2), flow = self.G_matrix, X = X)

        return self.N/MB * ELL

    ## ============================= ##
    ## == FINISH TRAINING METHODS == ## 
    ## ============================= ##


    ## ========================= ##
    ## == PERFORMANCE METHODS == ## 
    ## ========================= ##

    def test_log_likelihood(self, X: torch.tensor, Y:torch.tensor, return_moments:bool ,Y_std: float, S_MC_NNet: int = None) -> torch.tensor:
        """ Computes Predictive Log Likelihood 
                \log p(Y*|X*) = \log \int p(y*|G(f*),C_y) q(f*,f|u) q(u) df*,df,du 
                   -> We take diagonal of C_Y as samples are assumed to be i.i.d
                   -> Integration can be approximated either with Monte Carlo or with quadrature. This function uses quadrature.
                
                Args:
                        `X`                 (torch.tensor) :->: Input locations. Shape (MB,Dx) or shape (Dy,MB,Dx)
                        `Y`                 (torch.tensor) :->: Ground truth labels. Shape (MB,Dy)
                        `return_moments`    (bool)         :->: If true, then return the moments 1 and 2 from the predictive distribution.
                        `Y_std`             (float)        :->: Standard desviation of your regressed variable. Used to re-scale output.
                        `S_MC_NNet`         (int)          :->: Number of samples from the dropout distribution is fully_bayesian is true

                Returns:
                        `log_p_y`           (torch.tensor) :->: Log probability of each of the outpus with a tensor of shape (Dy,)
                        `predictive_params` (list)         :->: if return_moments True then returns a list with mean and variance from the predictive distribution. This is done in this funciton
                                                                because for some test log likelihood we need to compute the predictive. Hence support is given for any likelihood. Moments have shape
                                                                (Dy,MB,1)
        """
        MB = X.size(0)
        Dx = X.size(1)
        Dy = self.out_dim
        
        X_run  = X  # didnt realized the rest of function used X_run, so it is easier to do it here.
        if len(X_run.shape) == 2:
            X_run = X_run.repeat(self.out_dim,1,1) 
        assert len(X_run.shape) == 3, 'Invalid input X.shape'

        assert not self.is_training, "This method only works in eval mode"

        self.eval() # set parameters for eval mode. Batch normalization, dropout etc
        if self.fully_bayesian:
            # activate dropout if required
            is_dropout = enable_eval_dropout(self.modules())
            assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"
            assert S_MC_NNet is not None, "The default parameter S_MC_NNet is not provided and set to default None, which is invalid for self.be_bayesian" 

        with torch.no_grad():

            ## ================================================ ##
            ## =========== GAUSSIAN LIKELIHOOOD =============== ##
            ## == with non linear mean
            if isinstance(self.likelihood,GaussianNonLinearMean):
                # retrieve the noise and expand
                log_var_noise = self.likelihood.log_var_noise
                if self.likelihood.noise_is_shared:
                    log_var_noise = self.likelihood.log_var_noise.expand(Dy,1)

                ## ================================================== ##
                ## === Compute moments of predictive distribution === ##
                #  In this model this is not necessary to compute log likelihood.
                #  However, we give the option of returning this parameters to be consistent
                #  with the standard GP.
                predictive_params = None
                if return_moments:
                    m1,m2, mean_q_f, cov_q_f = self.predictive_distribution(X_run, diagonal = True, S_MC_NNet = S_MC_NNet)
                    predictive_params = [m1,m2]
                else:
                    mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X_run, diagonal = True, is_duvenaud = False, init_Z = None)
                mean_q_f, cov_q_f = mean_q_f.squeeze(dim = -1),cov_q_f.squeeze(dim = -1)

                self.eval()
                if self.fully_bayesian:
                    ## Call again self.eval() as self.predictive_distribution call self.train() before return
                    is_dropout = enable_eval_dropout(self.modules())
                    assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

                ## Common functions used by bayesian and non bayesian flows
                def get_quad_weights_shifted_locations(mean_q_f,cov_q_f):
                    ## Get the quadrature points and the weights
                    locations = self.likelihood.quadrature_distribution.locations
                    locations = _pad_with_singletons(locations, num_singletons_before=0, num_singletons_after = mean_q_f.dim())
                    shifted_locs = torch.sqrt(2.0 * cov_q_f) * locations + mean_q_f # Shape (S_quad,Dy,S,MB)

                    weights = self.likelihood.quadrature_distribution.weights
                    weights = _pad_with_singletons(weights, num_singletons_before=0, num_singletons_after = shifted_locs.dim() - 1) # Shape (S_quad,1,1,1)

                    return shifted_locs, weights

                def compute_log_lik(Y,Y_std,shifted_locs,C_Y):
                    ## Re-scale by Y_std same as what other people does to compare in UCI
                    Y   = Y_std*Y
                    m_Y = Y_std*shifted_locs
                    C_Y = (Y_std*torch.sqrt(C_Y))**2

                    log_p_y = batched_log_Gaussian( Y, m_Y, C_Y, diagonal = True, cov_is_inverse = False) # (S_quad,Dy,S_MC,MB)
                    
                    return log_p_y



                S_MC_NNet = 1 if not self.fully_bayesian else S_MC_NNet # Note that the estimator is the same for input dependent and Bayesian. Just need to expand or not this dimension
                                                                        
                S_quad = self.quad_points 
                G_mat  = self.G_matrix

                # noise retrieve and reshape
                C_Y = torch.exp(log_var_noise).expand(-1,MB).view(Dy,1,MB,1).repeat((S_quad,1,S_MC_NNet,1,1)) # (Squad,Dy,S_MC_NNet,MB,1). Add extra dimension 1 so that we can compute 
                                                                                                                  #                           likelihood using batched_log_gaussian function    
                # observation reshape
                Y = Y.t().view(1,Dy,1,MB,1).repeat((S_quad,1,S_MC_NNet,1,1))   # S,Dy,S_MC_NNet,MB,1

                # Y_std reshape
                Y_std = Y_std.view(1,Dy,1,1,1).repeat(S_quad,1,S_MC_NNet,MB,1) # S,Dy,S_MC_NNet,MB,1

                # this operation could be done by repeating X and computing mean_q_f as in DGP but is not necessary to do extra computation here as X is constant: just repeat. 
                mean_q_f, cov_q_f = mean_q_f.unsqueeze(dim = 1),cov_q_f.unsqueeze(dim = 1) # Remove last dimension, so that we can warp. We add it later for batched_log_lik
                mean_q_f = mean_q_f.repeat(1,S_MC_NNet,1) # (Dy,S_MC_NNet,MB)
                cov_q_f  = cov_q_f.repeat(1,S_MC_NNet,1)

                ## =================================== ##
                ## === Compute test log likelihood === ##
                shifted_locs, weights =  get_quad_weights_shifted_locations(mean_q_f,cov_q_f)

                ## Warp quadrature points
                # expand X to perform MC dropout over NNets parameters
                X_run = X_run.unsqueeze(dim = 1).repeat(1,S_MC_NNet,1,1) # Just add one extra dimension. No need for repeat for S_quad as pytorch automatically broadcasts. 
                                                                         # It is important to repeat over S_MC_NNet. In this way each forward thorugh X computes a different 
                                                                         # MC for the flow parameters. Otherwise pytorch would broadcast S_MC_NNet as well hence we would only 
                                                                         # be using one sample from the posterior over W.
                for idx,fl in enumerate(G_mat):
                     shifted_locs[:,idx,:,:] = fl(shifted_locs[:,idx,:,:],X_run[idx]) # (S_quad,Dy,S_MC_NNet,MB)

                shifted_locs = shifted_locs.view(S_quad,Dy,S_MC_NNet,MB,1) # shape (S_quad,Dy,S,MB,1)

                log_p_y = compute_log_lik(Y,Y_std,shifted_locs,C_Y)

                if self.fully_bayesian: # the only difference between bayesian and the rest is here, where we perform a double integration for this case

                    # Reduce with double logsumexp operation. Check estimator here: @TODO: add link once we releasea github
                    reduce_lse = torch.log(weights)  + log_p_y
                    log_p_y = torch.logsumexp( torch.logsumexp(reduce_lse, dim = 0) -0.5*torch.log(cg.pi) ,dim = 1).sum(1) - MB*numpy.log(S_MC_NNet)
                else:
                    # Note that we just need to remove the extra dimension we added for using the same code
                    log_p_y = log_p_y.squeeze(dim = 2)
                    weights = weights.squeeze(dim = 2)
        
                    ## Reduce log ws + log_p_y_s using logsumexp trick. Also reduce MB and add the constant
                    reduce_lse = torch.log(weights) + log_p_y
                    log_p_y = (torch.logsumexp(reduce_lse, dim = 0)).sum(-1) - 0.5*MB*torch.log(cg.pi)

            ## ===================
            ## == with linear mean
            elif isinstance(self.likelihood,GaussianLinearMean):
                ## ================================================== ##
                ## === Compute moments of predictive distribution === ##
                m_Y,K_Y, mean_q_f, cov_q_f = self.predictive_distribution(X_run, diagonal = True)

                ## =================================== ##
                ## === Compute test log likelihood === ##
                # Re-scale Y_std
                Y = Y.t() # (Dy,MB)
                Y_std = Y_std.view(self.out_dim,1) # (Dy,1)

                log_p_y = batched_log_Gaussian( obs = Y_std*Y, mean = Y_std*m_Y, cov = (Y_std*torch.sqrt(K_Y))**2, diagonal = True, cov_is_inverse = False)

                predictive_params = None
                if return_moments:
                    predictive_params = [m_Y,K_Y]

            ## =============================================================== ##
            ## ============ BERNOULLI/CATEGORICAL LIKELIHOOOD ================ ##
            elif isinstance(self.likelihood,MulticlassCategorical) or isinstance(self.likelihood,Bernoulli):

                # as we cant do exact integration here either we warp or we dont the proceedure is very similar to GP classification. The only difference is of
                # binary classification with Gauss CDF link function
                m_Y, _, mean_q_f, cov_q_f = self.predictive_distribution(X_run,diagonal = True, S_MC_NNet = S_MC_NNet)

                check = torch.logical_not(torch.isfinite(m_Y)).float()
                assert check.sum() == 0.0, "Got saturated probabilities"

                if isinstance(self.likelihood,Bernoulli): # turn the vector as if it became from the MulticlassCategorical so that this is transparent to the trainer
                    m_Y     = m_Y.squeeze() 
                    neg_m_Y = 1.0-m_Y # compute the probability of class 0
                    m_Y     = torch.stack((neg_m_Y,m_Y),dim = 1) 

                _, _ , _ , log_p_y = compute_calibration_measures(m_Y.float() ,Y ,apply_softmax = False ,bins = 15)  

                log_p_y = -1*((log_p_y*MB).sum()) # the compute_calibration_measures returns log_p_y.mean(), hence we remove that by multiplying by MB and then summing up

                predictive_params = None
                if return_moments:
                    predictive_params = [m_Y]

            else:
                raise ValueError("Unsupported likelihood [{}] for class [{}]".format(type(self.likelihood),type(self)))

        self.train() # set parameters for train mode. Batch normalization, dropout etc
        return log_p_y, predictive_params

    ## Predictive distribution method is used to compute root mean squared error.

    ## ================================ ##
    ## == FINISH PERFORMANCE METHODS == ## 
    ## ================================ ##

    ## ====================== ##
    ## == SAMPLING METHODS == ## 
    ## ====================== ##

    def sample_from_variational_marginal_base(self, X: torch.tensor, diagonal: bool, is_duvenaud: bool, init_Z: torch.tensor = None) -> list:
        """ Sample from the marginal variational q(f0), i.e without warping. 

            Note: this method should not be called. Instead call self.sample_from_variational_marginal
        
              q(f) = \int q(f|u) q(u) du
                
                Args: 
                        `X`           (torch.tensor)  :->: Locations where the sample is drawn from. It shoud be (Dy,SMB,Dx). Note that for DGP MB = S*MB given S 
                                                           Monte Carlo samples
                        `diagonal`    (bool)          :->: Ff true, samples are drawn independently. For the moment is always true.
                        `is_duvenaud` (bool)          :->: Indicate if we are using duvenaud mean function. Only useful in DGPs
                        `init_Z`      (torch.tensor)  :->: Only used if is_duvenad = True

                Returns:
                        `f`           (torch.tensor) :->: Returns a sample from the posterior with shape (Dy,S*MB) 
                        `mean_q_f`    (torch.tensor) :->: Returns mean from q(f) with shape (Dy,S*MB,1)
                        `cov_q_f`     (torch.tensor) :->: Returns cov  from q(f) with shape (Dy,S*MB,1)

        """
        ## We sample from the Dy's GPs at a layer at the same time ##
        if not diagonal:
            raise NotImplementedError("This function only works with diagonal=True")

        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) 
        assert len(X.shape) == 3, 'Invalid input shape for X'

        Dy,SMB,Dx = X.shape

        mean_q_f, cov_q_f = self.marginal_variational_qf_parameters(X, diagonal = True, is_duvenaud = is_duvenaud, init_Z = init_Z)
            # mean_q_f shape (Dy,MB,1)
            # cov_q_f  shape (Dy,MB,1)
        std_q_f = cov_q_f.sqrt()

        ## ============================================================================== ##
        ##  1. == Get i.i.d sample for each sample in the minibatch and each dimension == ##
        e = self.standard_sampler.sample(sample_shape = torch.Size([Dy,SMB,]))

        ## ================================= ##
        ##  2. ==  Reparameterized Sample == ##
        f = e*std_q_f + mean_q_f

        ## ================================================== ##
        ##  3. ==  Reshape and permute to match dimensions == ##
        f = f.squeeze(dim = 2) # shape (Dy,SMB)

        return f, mean_q_f, cov_q_f

    def sample_from_variational_marginal(self, X: torch.tensor, S : int, diagonal: bool, is_duvenaud: bool, init_Z: torch.tensor = None) -> list:
        """ Sample from the marginal variational q(fK), i.e after warping 
        
              q(f) = \int q(f|u) q(u) du
                
                Args: 
                        `X`           (torch.tensor)  :->: Locations where the sample is drawn from. It shoud be (Dy,SMB,Dx). Note that for DGP MB = S*MB given S 
                                                           Monte Carlo samples. This S correspond also to the Monte Carlo dropout marginalization.
                        `S`           (int)           :->: Number of samples to draw. Specify S = 1 if your input is already S*MB,Dx (DGP). 
                        `diagonal`    (bool)          :->: Ff true, samples are drawn independently. For the moment is always true.
                        `is_duvenaud` (bool)          :->: Indicate if we are using duvenaud mean function. Only useful in DGPs
                        `init_Z`      (torch.tensor)  :->: Only used if is_duvenad = True

            Note: Returns a sample from the posterior with shape (Dy,S*MB,Dy) 
            
            Returns:

                f           (torch.tensor)  :->: Warped samples. Can have different shape depend on connection, check note above. In general is (Dy,S*MB)
                mean_q_f0   (torch.tensor)  :->: Mean from variational marginal q(f0) with shape (Dy,SMB,1)
                cov_q_f0    (torch.tensor)  :->: Diagonal covariance from variational marginal q(f0) with shape (Dy,SMB,1)
                f0          (torch.tensor)  :->: Samples from base GP with shape (Dy,SMB)
        """
        if len(X.shape) == 2:
            X = X.repeat(self.out_dim,1,1) 
        assert len(X.shape) == 3, 'Invalid input X.shape'
        X = X.repeat(1,S,1)

        if self.is_training: 
            self.train() 
        else: # During evaluation we switch everything to eval mode but the MC dropout in case fully bayesian is true
            self.eval() # set parameters for eval mode. Batch normalization, dropout etc
            if self.fully_bayesian:
                # activate dropout if required
                is_dropout = enable_eval_dropout(self.modules())
                assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

        f0, mean_q_f0, cov_q_f0 = self.sample_from_variational_marginal_base(X = X, diagonal = diagonal, is_duvenaud = is_duvenaud, init_Z = init_Z)

        Dy  = self.out_dim
        SMB = f0.size(0)
        G   = self.G_matrix

        ## == Warp the samples == ##
        f = f0.clone() 

        f_aux = torch.zeros(f.shape)
        for idx,g in enumerate(G):
            f_aux[idx,:] = g(f[idx,:],X[idx])
        f = f_aux

        self.train()
        return f, mean_q_f0, cov_q_f0, f0 

    def sample_from_predictive_distribution(self, X: torch.tensor, S:int) -> List[torch.tensor]:
        """
            This functions computes from q(y)
            Steps:
                1) sample from q(f)
                2) for each sample draw a sample from the predictive likelihood 

            Args:
                X: input to predict at. X = N x D.
                S: number of samples

            Returns:
                sample_arr (torch.tensor) - returns a tensor of samples for each output dimension shape is Dy x S x N x 1
                f_k_samples  (torch.tensor)  :->: warped locations where sample from predictive is drawn p(y|fk)
                f_0_samples  (torch.tensor)  :->: unwarped location where predictive sample is drawn f0 = G^{-1}(fk)
        """
        #@TODO: Refactorize this function to work batched. Just requires likelihood.sample_from_output to be implemented as batched because sampling from the posterior already does it
        #@TODO: IMPORTANT: Sample from predictive with classification likelihoods is wrong as the likelihood is 1D but self.out_dim is equal to the number of GPs. I have just given support to sample_from_output in the classification likelihoods for compatibility with my experiments (I dont really use this feature) but we should clearly refactorize this to work batched. If it works batched then there is no problem here.

        assert not self.is_training, "This method only works in eval mode"
        self.eval() # set parameters for eval mode. Batch normalization, dropout etc
        if self.fully_bayesian:
            # activate dropout if required
            is_dropout = enable_eval_dropout(self.modules())
            assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

        assert len(X.shape) == 2, 'Invalid input X.shape' # when refactorized to work batched it should allow 3D input as in the rest of the code.

        N, D = X.shape
        # X = X.repeat([1, S, 1]) # this is done now inside sample_from_variaitonal_marginal

        #array of samples for each output dimension
        samples_arr = []
        with torch.no_grad():

            #sample_from_variational_marginal:
            #   input: X: [1, NxS, D]
            #   output: f_k_samples : [NxS, D_out]
            f_k_samples, _, _, f_0_samples = self.sample_from_variational_marginal(X, S, diagonal=True, is_duvenaud=False, init_Z=None)

            self.eval() # set again as above method calls train before return
            if self.fully_bayesian:
                # activate dropout if required
                is_dropout = enable_eval_dropout(self.modules())
                assert is_dropout, "You set the model to work on fully bayesian but there are no dropout layers in your model. I assert this error as otherwise the the code will work in non_bayesian operating mode"

            #go through every output and get the respective likelihood for it
            for output_idx in range(self.out_dim):
                samples = self.likelihood.sample_from_output(f_k_samples, output_idx)
                samples = samples.view(S, N, 1)
                samples_arr.append(samples)

            self.train() # set parameters for train mode. Batch normalization, dropout etc
            return torch.stack(samples_arr, dim=0), f_k_samples, f_0_samples


