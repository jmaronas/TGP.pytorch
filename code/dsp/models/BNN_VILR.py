# torch
import torch
import torch.nn as nn

from .. import config as cg

from pytorchlib import return_activation


class FC_VI_LR(nn.Module):
    def __init__(self,indim,outdim,activation,prior_mean,prior_logvar):
        super(FC_VI_LR, self).__init__()
        ## Model Definitinion 
        # -> Prior: p(w,b)=Normal(0,1)
        
        self.activation= return_activation(activation)
        self.outdim=outdim

        ## Variational Distribution
        self.w_mean=nn.Parameter(torch.randn((indim,outdim)))
        self.w_logvar=nn.Parameter(torch.randn((indim,outdim))*-1e-20)
        self.b_mean=nn.Parameter(torch.randn((outdim)))
        self.b_logvar=nn.Parameter(torch.randn((outdim))*-1e-20)

    def sample(self,ind_mu,ind_var):
        # Reparameterization trick ( locally )
        u = torch.normal(torch.tensor(0),torch.tensor(1),ind_var.shape).to(cg.device)
        s=(u*ind_var.sqrt() + ind_mu)
        return s

    def forward(self,x, is_initializing):
        # Induce distribution over activations
        ind_mu = torch.matmul(x,self.w_mean) + self.b_mean	
        ind_var = torch.matmul(x**2,torch.exp(self.w_logvar)) + torch.exp(self.b_logvar)

        # Sample the the batch of activations
        if is_initializing:
            s = ind_mu
        else:
            s = self.sample(ind_mu,ind_var)

        return self.activation(s)
                    

## Mean-field Gaussian VI over a BNN with Local Reparameterization

class BNN_VILR(nn.Module):
    def __init__(self,dim_layer,number_layers,in_dim,out_dim,prior_mean,prior_var, hidden_act):
        super(BNN_VILR, self).__init__()

        # prior N(w,b| )
        self.prior_mean=torch.tensor(prior_mean).to(cg.device)
        self.prior_logvar=torch.log(torch.tensor(prior_var).to(cg.device))

        self.invalid = nn.Dropout(0.0) # add this thing so that the code doesnt assert if there is no dropout layers

        # Instance model Likelihood 
        self.Layers = nn.ModuleList()
        if number_layers == 0:
            Lin=FC_VI_LR(in_dim,out_dim,'linear',self.prior_mean,self.prior_logvar)
            self.Layers.append(Lin)
        else:
            Lin=FC_VI_LR(in_dim,dim_layer,hidden_act,self.prior_mean,self.prior_logvar)
            self.Layers.append(Lin)
            for l in range(number_layers-1):
                self.Layers.append(FC_VI_LR(dim_layer,dim_layer,hidden_act,self.prior_mean,self.prior_logvar))
            self.Layers.append(FC_VI_LR(dim_layer,out_dim,'linear',self.prior_mean,self.prior_logvar))


    # Compute the likelihood p(t|x)
    def forward(self,x, is_initializing = False):
        for l in self.Layers:
            x=l(x, is_initializing)
        return x

    # Gaussian KLD
    def GAUSS_KLD(self,qmean,qlogvar,pmean,plogvar):
        # Computes the DKL(q(x)//p(x)) between the variational and the prior 
        # distribution assuming Gaussians distribution with arbitrary prior
        qvar,pvar = torch.exp(qlogvar),torch.exp(plogvar)
        DKL=(0.5 * (-1 + plogvar - qlogvar + (qvar/pvar) + torch.pow(pmean-qmean,2)/pvar)).sum()

        return DKL

    # Kullback Liber Divergence of the Full Model
    def KLD(self):
        DKL=0.0
        for l in self.Layers:
            w_mean,w_logvar,b_mean,b_logvar=l.w_mean,l.w_logvar,l.b_mean,l.b_logvar
            DKL+=(self.GAUSS_KLD(w_mean,w_logvar,self.prior_mean,self.prior_logvar)+self.GAUSS_KLD(b_mean,b_logvar,self.prior_mean,self.prior_logvar))

        return DKL



