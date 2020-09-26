"""

"""

import numpy as np
import torch
from torch.autograd import Function
from .. import emd,gromov_wasserstein


# Author: Remi Flamary <remi.flamary@unice.fr>
#
# License: MIT License


# Inherit from Function
class OptimalTransportLossFunction(Function):
    """Return OT Loss for input (a,b,M) """

    @staticmethod
    # bias is an optional argument
    def forward(ctx, a, b, M, num_iter_max=100000):

        # convert to numpy
        a2 = a.detach().cpu().numpy().astype(np.float64)
        b2 = b.detach().cpu().numpy().astype(np.float64)
        M2 = M.detach().cpu().numpy().astype(np.float64)

        # project on simplex for float64 or else numerical errors
        a2 /= a2.sum()
        b2 /= b2.sum()

        G, log = emd(a2, b2, M2, log=True, numItermax=num_iter_max)

        G = torch.from_numpy(G).type_as(M)
        grad_a = torch.from_numpy(log['u']).type_as(a)
        grad_b = torch.from_numpy(log['v']).type_as(b)
        grad_M = G

        ctx.save_for_backward(grad_a, grad_b, grad_M)
        return torch.sum(G * M)

    @staticmethod
    def backward(ctx, grad_output):

        grad_a0, grad_b0, grad_M0 = ctx.saved_tensors
        grad_a = grad_b = grad_M = None

        if ctx.needs_input_grad[0]:
            grad_a = grad_a0
        if ctx.needs_input_grad[1]:
            grad_b = grad_b0
        if ctx.needs_input_grad[2]:
            grad_M = grad_M0

        return grad_a, grad_b, grad_M, None  # last one is param

class GromovWassersteinLossFunction(Function):
    """Return GW Loss for input (C1,C2,p,q) """

    @staticmethod
    # bias is an optional argument
    def forward(ctx, C1,C2,p,q,func='square_loss'):

        # convert to numpy
        C1 = C1.detach().cpu().numpy().astype(np.float64)
        C2 = C2.detach().cpu().numpy().astype(np.float64)
        p = p.detach().cpu().numpy().astype(np.float64)
        q = q.detach().cpu().numpy().astype(np.float64)

        # project on simplex for float64 or else numerical errors
        p /= p.sum()
        q /= q.sum()

        T= gromov_wasserstein(C1,C2,p,q, log=False,loss_fun=func)

        T = torch.from_numpy(T)
        grad_T = T

        ctx.save_for_backward(grad_T)
        return torch.sum(T) 

    @staticmethod
    def backward(ctx, grad_output):

        grad_T0 = ctx.saved_tensors
        grad_T = None

        if ctx.needs_input_grad[0]:
            grad_T = grad_T0

        return grad_T


def ot_loss(a, b, M, num_iter_max=100000):
    """loss=emd2(a,b,M)"""
    return OptimalTransportLossFunction.apply(a, b, M, num_iter_max)


def otgw_loss(C1,C2,p,q):
    """loss=gromov_wasserstein(C1,C2,p,q)"""
    return GromovWassersteinLossFunction.apply(C1,C2,p,q)


def ot_solve(a, b, M, num_iter_max=100000, log=False):

    a2 = a.detach().cpu().numpy().astype(np.float64)
    b2 = b.detach().cpu().numpy().astype(np.float64)
    M2 = M.detach().cpu().numpy().astype(np.float64)

    # project on simplex for float64 or else numerical errors
    a2 /= a2.sum()
    b2 /= b2.sum()

    if log:

        G, log = emd(a2, b2, M2, log=False, numItermax=num_iter_max)

        log['u'] = torch.from_numpy(log['u']).type_as(a)
        log['v'] = torch.from_numpy(log['v']).type_as(a)

        return torch.from_numpy(G).type_as(M), log

    else:

        G = emd(a2, b2, M2, log=False, numItermax=num_iter_max)

        return torch.from_numpy(G).type_as(M)
    
def otgw_solve(C1,C2,p,q,func='kl_loss'):

    C1 = C1.detach().cpu().numpy().astype(np.float64)
    C2 = C2.detach().cpu().numpy().astype(np.float64)
    p = p.detach().cpu().numpy().astype(np.float64)
    q = q.detach().cpu().numpy().astype(np.float64)
    # project on simplex for float64 or else numerical errors
    p /= p.sum()
    q /= q.sum()


    T = gromov_wasserstein(C1,C2,p,q, log=False,loss_fun=func)

    return torch.from_numpy(T)
