## Implement the specific derivatives
## O = \int_0^T (y[1](t) - v(t)) ^ 2 dt 
# To compute dO / d \beta, one needs to implement six functions
# In all tensors, the last dimension is time
## Import standard librarys
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import scipy.optimize as opt
import numpy as np

from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# \partial o(y, yDot, t; \beta) / \partial y
def DoDy(y, v, t, MFParams):
    DoDy = torch.zeros(y.shape)
    DoDy[1, :] = 2. * (y[1, :] - v)
    return DoDy

# \partial o / \partial yDot
def DoDyDot(y, v, t, MFParams):
    return torch.zeros(y.shape)

# d/dt (\partial o / \partial yDot)
def DDoDyDotDt(y, v, t, MFParams):
    return torch.zeros(y.shape)

# \partial o / \partial \beta
def DoDBeta(y, v, t, MFParams):
    return torch.zeros([MFParams.RSParams.shape[0], y.shape[1]])

# \partial C / \partial y, unregularized
def DCDy(y, v, t, MFParams):
    DCDy = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    DCDy[0, 1, :] = -1.
    DCDy[1, 0, :] = MFParams.k / MFParams.m
    DCDy[1, 1, :] = MFParams.g * MFParams.RSParams[0] / y[1, :]
    DCDy[1, 2, :] = MFParams.g * MFParams.RSParams[1] / y[2, :]
    # DCDy[2, 1, :] = y[2, :] / MFParams.RSParams[2]
    # DCDy[2, 2, :] = y[1, :] / MFParams.RSParams[2]
    DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2]
    DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2]
    return DCDy

# \partial C / \partial y, regularized
def DCDy_regularized(y, v, t, MFParams):
    # Compute auxiliary Q1 and !2
    # Q1 = MFParams.RSParams[3] + MFParams.RSParams[1] * torch.log(1.e-6 * y[2, :] / MFParams.RSParams[2])
    Q1 = MFParams.RSParams[3] + MFParams.RSParams[1] * torch.log(1.e-6 * y[2, :] * MFParams.RSParams[2])
    Q1 = Q1 / MFParams.RSParams[0]
    Q2 = y[1, :] / 2 / 1.e-6 * torch.exp(Q1)
    Q2_cliped = torch.clamp(Q2, min = -1.e10, max = 1.e10)
    Q2Term = Q2_cliped / torch.sqrt(Q2_cliped**2 + 1.)
    
    # Calculate partial derivatives
    pfpy1 = MFParams.RSParams[0] * Q2Term / y[1, :]
    pfpy2 = MFParams.RSParams[1] * Q2Term / y[2, :]
    
    DCDy = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    DCDy[0, 1, :] = -1.
    DCDy[1, 0, :] = MFParams.k / MFParams.m
    DCDy[1, 1, :] = MFParams.g * pfpy1
    DCDy[1, 2, :] = MFParams.g * pfpy2
    # DCDy[2, 1, :] = y[2, :] / MFParams.RSParams[2]
    # DCDy[2, 2, :] = y[1, :] / MFParams.RSParams[2]
    DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2]
    DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2]
    
#     # DEBUG LINES
#     print("DCDy: ", DCDy)
    
    return DCDy

# \partial C / \partial yDot, unregularized
def DCDyDot(y, v, t, MFParams):
    DCDyDot = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    DCDyDot[0, 0, :] = 1.
    DCDyDot[1, 1, :] = 1.
    DCDyDot[2, 2, :] = 1.
    return DCDyDot

# d/dt (\partial C / \partial yDot)
def DDCDyDotDt(y, v, t, MFParams):
    DDCDyDotDt = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    return DDCDyDotDt

# \partial C / \partial \beta, unregularized
def DCDBeta(y, v, t, MFParams):
    DCDBeta = torch.zeros([y.shape[0], MFParams.RSParams.shape[0], y.shape[1]])
    DCDBeta[1, 0, :] = MFParams.g * torch.log(y[1, :] / 1.e-6)
    # DCDBeta[1, 1, :] = MFParams.g * torch.log(1.e-6 * y[2, :] / MFParams.RSParams[2])
    # DCDBeta[1, 2, :] = -MFParams.g * MFParams.RSParams[1] / MFParams.RSParams[2]
    
    DCDBeta[1, 1, :] = MFParams.g * torch.log(1.e-6 * y[2, :] * MFParams.RSParams[2])
    DCDBeta[1, 2, :] = MFParams.g * MFParams.RSParams[1] / MFParams.RSParams[2]

    DCDBeta[1, 3, :] = MFParams.g
    # DCDBeta[2, 2, :] = -y[1, :] * y[2, :] / MFParams.RSParams[2] / MFParams.RSParams[2]
    DCDBeta[2, 2, :] = y[1, :] * y[2, :]
    return DCDBeta

# \partial C / \partial \beta, regularized
def DCDBeta_regularized(y, v, t, MFParams):
    # Compute auxiliary Q1 and !2
    # Q1 = MFParams.RSParams[3] + MFParams.RSParams[1] * torch.log(1.e-6 * y[2, :] / MFParams.RSParams[2])
    Q1 = MFParams.RSParams[3] + MFParams.RSParams[1] * torch.log(1.e-6 * y[2, :] * MFParams.RSParams[2]) 
    Q1 = Q1 / MFParams.RSParams[0]
    Q2 = y[1, :] / 2 / 1.e-6 * torch.exp(Q1)
    Q2_cliped = torch.clamp(Q2, min = -1.e10, max = 1.e10)
    Q2Term = Q2_cliped / torch.sqrt(Q2_cliped**2 + 1.)
#     # DEBUG LINES
#     print('Q1: ', Q1)
#     print('Q2: ', Q2)
    
    # Partial derivatives
    pfpbeta0 = torch.asinh(Q2) - Q1 * Q2Term
    # pfpbeta1 = Q2Term * torch.log(1.e-6 * y[2, :] / MFParams.RSParams[2])
    pfpbeta1 = Q2Term * torch.log(1.e-6 * y[2, :] * MFParams.RSParams[2])
    # pfpbeta2 = -Q2Term * MFParams.RSParams[1] / MFParams.RSParams[2]
    pfpbeta2 = Q2Term * MFParams.RSParams[1] / MFParams.RSParams[2]
    pfpbeta3 = Q2Term
    
    DCDBeta = torch.zeros([y.shape[0], MFParams.RSParams.shape[0], y.shape[1]])
    DCDBeta[1, 0, :] = MFParams.g * pfpbeta0
    DCDBeta[1, 1, :] = MFParams.g * pfpbeta1
    DCDBeta[1, 2, :] = MFParams.g * pfpbeta2
    DCDBeta[1, 3, :] = MFParams.g * pfpbeta3
    # DCDBeta[2, 2, :] = -y[1, :] * y[2, :] / MFParams.RSParams[2] / MFParams.RSParams[2]
    DCDBeta[2, 2, :] = y[1, :] * y[2, :]
#     # DEBUG LINES
#     print("DCDBeta: ", DCDBeta)
    
    return DCDBeta

# ------------------------ Calculate the derivative: Do / D\beta -------------------------
# Observation
def O(y, v, t, MFParams):
    # Least square error
    O = torch.trapezoid(
        torch.square(y[1, :] - v), 
        t
    )
    
    # print("Relative L2 error: ", torch.sqrt(O) / torch.linalg.norm(v))
    return O

    