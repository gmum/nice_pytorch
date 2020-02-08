"""
Implementation of NICE log-likelihood loss.
"""
import torch
import torch.nn as nn
import numpy as np
import scipy.special

# ===== ===== Loss Function Implementations ===== =====
"""
We assume that we final output of the network are components of a multivariate distribution that
factorizes, i.e. the output is (y1,y2,...,yK) ~ p(Y) s.t. p(Y) = p_1(Y1) * p_2(Y2) * ... * p_K(YK),
with each individual component's prior distribution coming from a standardized family of
distributions, i.e. p_i == Gaussian(mu,sigma) for all i in 1..K, or p_i == Logistic(mu,scale).
"""


def gaussian_nice_loglkhd(h, diag):
    """
    Definition of log-likelihood function with a Gaussian prior, as in the paper.
    
    Args:
    * h: float tensor of shape (N,D). First dimension is batch dim, second dim consists of components
      of a factorized probability distribution.
    * diag: scaling diagonal of shape (D,).

    Returns:
    * loss: torch float tensor of shape (N,).
    """
    # \sum^D_i s_{ii} - { (1/2) * \sum^D_i  h_i**2) + (D/2) * log(2\pi) }
    return torch.sum(diag) - (0.5*torch.sum(torch.pow(h,2),dim=1) + h.size(1)*0.5*torch.log(torch.tensor(2*np.pi)))


def logistic_nice_loglkhd(h, diag):
    """
    Definition of log-likelihood function with a Logistic prior.
    
    Same arguments/returns as gaussian_nice_loglkhd.
    """
    # \sum^D_i s_{ii} - { \sum^D_i log(exp(h)+1) + torch.log(exp(-h)+1) }
    return (torch.sum(diag) - (torch.sum(torch.log1p(torch.exp(h)) + torch.log1p(torch.exp(-h)), dim=1)))


def binomial_coeff(n, alpha):
    factorials = torch.from_numpy(np.array([scipy.special.binom(n, k) for k in range(n)])).type(torch.FloatTensor)
    coeffs = torch.from_numpy(np.array([(alpha ** k) * ((1 - alpha) ** (n - k)) for k in range(n)]))
    return torch.mul(factorials, coeffs).type(torch.FloatTensor)


def binomial_nice(h, diag, DEVICE, alpha):
    radius = torch.sort(torch.stack([torch.sqrt(torch.sum(i)) for i in torch.pow(h, 2)]), descending=True)
    #radius = torch.sort(torch.stack([torch.pow(torch.sqrt(torch.sum(i)), h.shape[1]) for i in torch.pow(h, 2)]), descending=True)
    loss = torch.sum(diag) - torch.mul(torch.tensor(2.0), torch.log(torch.dot(radius[0], binomial_coeff(radius[0].shape[0], alpha).to(DEVICE))))
    #loss = torch.div(torch.dot(radius[0], binomial_coeff(radius[0].shape[0], alpha).to(DEVICE)), torch.exp(diag[0]*diag[1]))
    return loss


# wrap above loss functions in Modules:
class GaussianPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(GaussianPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag, DEVICE, alpha):
        if self.size_average:
            return torch.mean(-gaussian_nice_loglkhd(fx, diag))
        else:
            return torch.sum(-gaussian_nice_loglkhd(fx, diag))


class LogisticPriorNICELoss(nn.Module):
    def __init__(self, size_average=True):
        super(LogisticPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag, DEVICE, alpha):
        if self.size_average:
            return torch.mean(-logistic_nice_loglkhd(fx, diag))
        else:
            return torch.sum(-logistic_nice_loglkhd(fx, diag))


class BinomialPriorNICELoss(nn.Module):
    def __init__(self, size_average=True, alpha=0.05):
        super(BinomialPriorNICELoss, self).__init__()
        self.size_average = size_average

    def forward(self, fx, diag, DEVICE, alpha):
        if self.size_average:
            return torch.mean(-binomial_nice(fx, diag, DEVICE, alpha))
        else:
            return torch.sum(-binomial_nice(fx, diag, DEVICE, alpha))
