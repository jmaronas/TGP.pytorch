#-*- coding: utf-8 -*-
# MultiClassCategorical.py : Keeps a Categorical Likelihood p(y|f) for multiclass problems
# Author: Juan Maroñas and  Ollie Hamelijnc

# Python
import sys
sys.path.extend(['../'])

# Torch
import torch
import torch.nn as nn
import torch.distributions as td
from  torch.nn.functional import softmax

# custom
from .. import config as cg


class MulticlassCategorical(nn.Module):
    ''' Implements multiclass categorical likelihood using a softmax link function
        All integrations required are done using Monte Carlo using S_quad parameters

        Although Softmax  = Heavised + noise, we wont keep noise in the likelihood. The user can specify noise by using a white noise kernel on the Base model

        Note: For those familiar in coding up Bayesian Neural Networks this likelihood is basically what you do with BNNs, just changing samples from the parameters by samples from the GP.
    '''

    def __init__(self,num_classes : int):
        super(MulticlassCategorical,self).__init__()

        self.C             = num_classes
        self.SMC           = cg.quad_points                      # number of monte carlo used to compute the integrals
        self.loss          = nn.CrossEntropyLoss(reduction = 'none')
        self.link_function = softmax

        assert num_classes > 2, "If you have a binary classification problem use the Bernouilli"

    def sample_from_output(self, f: torch.tensor, i:int, **kwargs) -> td.Distribution:
        """
            Args:
                f: likelihood input - shape Nx1
                i: returns a sample from p(y_i | f)
        """
        assert f.size(0) == self.C, "Bad specified input"

        probs = self.link_function(f.t(), dim = 1)
        dist  = td.Categorical(probs = probs)
        return dist.sample().to(cg.dtype)


    def expected_log_prob(self,Y:torch.tensor, gauss_mean: torch.tensor, gauss_cov: torch.tensor, flow: list, X: torch.tensor, **kwargs):

        ''' Computes

            \int q(f_0) \log p(y|G(f_0)) d_f_0

            Each of the samples f_0 are squeezed through the flow and through the softmax. In practice, pytorch do not require you to apply the softmax as the categorical cross entropy function handles everything for numerical stability

        
            Args:
                `Y`             (torch.tensor)  :->:  Categorical Labels in standard format (i.e NO one-hot). Shape (1,MB)
                `gauss_mean`    (torch.tensor)  :->:  mean from q(f). Shape (n_classes,MB)
                `gauss_cov`     (torch.tensor)  :->:  diagonal covariance from q(f). Shape (n_classes,MB)
                `non_linearity` (list)          :->:  List of callable representing the non linearity applied to the mean.
                `X`             (torch.tensor)  :->:  Input with shape (n_classes,MB,Dx) or (MB,Dx). Needed for input dependent flows
                
        '''
        SMC   = self.SMC
        num_C = self.C
        MB    = gauss_mean.size(1)

        assert len(flow) == num_C, "Flow list must be size {} for MultiClass likelihood".format(num_C)
        assert gauss_mean.size(0) == num_C, "Multiclass classification requires {} GPs, got {}".format(num_C,gauss_mean.size(0))
        assert len(X.shape) == 3, 'Bad input X, expected (n_class,MB*S,Dx)'
        assert X.size(0) == num_C, 'Wrong first dimension in X, expected n_classes'

        Y          = Y.t().squeeze(dim = 1)
        # gauss_mean = gauss_mean
        gauss_std  = gauss_cov.sqrt()

        distr = td.Normal(gauss_mean,gauss_std)

        F0    = distr.rsample(torch.Size([SMC])) # use rsample so that we can backpropagate
                                                 # shape (S, num_C, MB)   
        # warp the samples using the flow
        FK = F0.clone()
        for c in range(num_C):
            FK[:,c,:] = flow[c](F0[:,c,:],X[c]) # X is the same for the single layer 
                                                # but not for the deep one, hence
                                                # the indexing X[c]

        # FK has shape (S,Dy,MB) so transpose to (S,MB,Dy) and view to (S*MB,Dy) for compatibility with torch
        FK = FK.transpose(2,1)
        FK = FK.reshape(SMC*MB,num_C)

        # reshape to work with pytorch built in
        Y  = Y.repeat(SMC)

        ## use built-in torch categorical likelihood
        log_p_y = self.loss(FK,Y).view(SMC,MB)
      
        # reduce mean monte carlo dimension and sum minibatch dimension
        log_p_y = log_p_y.mean(0).sum()

        return log_p_y*-1 # we return the negation because the cross entropy is the negative log likelihood, while our code assumes we maximize the ELBO (although optimization is done by finally negating and minimizing)



    def marginal_moments(self, gauss_mean, gauss_cov, flow, X, **kwargs):
        """ Computes marginal moments w.r.t single Gaussian using
            monte carlo and sofmax

            Args:
                `gauss_mean`    (torch.tensor)  :->: mean from q(f). Shape (n_classes,MB)
                `gauss_cov`     (torch.tensor)  :->: covariance from q(f). Shape (n_classes,MB)
                `non_linearity` (list)          :->: flow list
                `X`             (torch.tensor)  :->: Input locations used by input dependent flows

        """
        SMC   = self.SMC
        num_C = self.C
        MB    = gauss_mean.size(1)

        assert len(flow) == num_C, "Flow list must be size {} for MultiClass likelihood".format(num_C)
        assert gauss_mean.size(0) == num_C, "Multiclass classification requires {} GPs, got {}".format(num_C,gauss_mean.size(0))
        assert len(X.shape) == 3, 'Bad input X, expected (n_class,MB*S,Dx)'
        assert X.size(0) == num_C, 'Wrong first dimension in X, expected n_classes'

        #gauss_mean = gauss_mean
        gauss_std  = gauss_cov.sqrt()

        distr = td.Normal(gauss_mean,gauss_std)

        F0    = distr.sample(torch.Size([SMC])) # use sample so that we CAN'T backpropagate
                                                # shape (S, num_C, MB)   
        # warp the samples using the flow
        FK = F0.clone()
        for c in range(num_C):
            FK[:,c,:] = flow[c](F0[:,c,:],X[c]) # X is the same for the single layer 
                                                # but not for the deep one, hence
                                                # the indexing X[c]

        ## use built-in torch softmax function. I think passing dim = 1 should work fine but I feel safer if I transpose first. Also more clear for someone reading the code
        FK = FK.transpose(2,1)
        P  = self.link_function(FK, dim = 2).mean(0) # softmax plus reduce montecarlo dimension 

        # print(P.sum(1)) # This has to give a vector of 1s
        # print(P.sum(1).sum()) # This has to give a value of MB
        # print(MB)

        return P 
