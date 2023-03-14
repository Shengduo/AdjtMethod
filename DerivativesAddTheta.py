## Implement the specific derivatives
## O = \int_0^T (y[1](t) - v(t)) ^ 2 + (log(y[1](t)) - log(v(t))) ^ 2
#               (y[2](t) - theta(t)) ^ 2 + (log(y[2](t)) - log(theta(t))) ^ 2 dt 
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

# ------------------------ Calculate the derivative: Do / D\beta -------------------------
# Observation
def O(y, y_targ, t, MFParams, MFParams_targ, normalization=True):
    ff_targ = computeF(y_targ, MFParams_targ)
    ff = computeF(y, MFParams)
    
    # DEBUG LINES
    print("~"*80)
    print("torch.mean(y_targ[1, :]): ", torch.mean(y_targ[1, :]))
    print("torch.mean(y_targ[2, :]): ", torch.mean(y_targ[2, :]))
    print("torch.mean(torch.log(y_targ[1, :])): ", torch.mean(torch.log(y_targ[1, :])))
    print("torch.mean(torch.log(y_targ[2, :])): ", torch.mean(torch.log(y_targ[2, :])))
    print("torch.mean(ff_targ): ", torch.mean(ff_targ))
    print("~"*80, "\n")

    # Least square error
    if normalization == True:
        O = torch.trapezoid(
            torch.square((y[1, :] - y_targ[1, :]) / torch.mean(y_targ[1, :])) 
            + torch.square((y[2, :] - y_targ[2, :]) / torch.mean(y_targ[2, :]))  
            + torch.square((torch.log(y[1, :]) - torch.log(y_targ[1, :])) / torch.mean(torch.log(y_targ[1, :])))
            + torch.square((torch.log(y[2, :]) - torch.log(y_targ[2, :])) / torch.mean(torch.log(y_targ[2, :])))
            + torch.square((ff - ff_targ) / torch.mean(ff_targ)), 
            t
        )
    else:
        O = torch.trapezoid(
            torch.square(y[1, :] - y_targ[1, :]) + torch.square(y[2, :] - y_targ[2, :])  
            + torch.square(torch.log(y[1, :]) - torch.log(y_targ[1, :]))
            + torch.square(torch.log(y[2, :]) - torch.log(y_targ[2, :])) 
            + torch.square(ff - ff_targ), 
            t
        )
    
    # print("Relative L2 error: ", torch.sqrt(O) / torch.linalg.norm(v))
    return O

# \partial o(y, yDot, t; \beta) / \partial y
def DoDy(y, y_targ, t, MFParams, MFParams_targ, normalization=True):
    DoDy = torch.zeros(y.shape)

    if normalization == True:
        DoDy[1, :] = 2. * (y[1, :] - y_targ[1, :]) / torch.mean(y_targ[1, :])**2 + 2. * (torch.log(y[1, :]) - torch.log(y_targ[1, :])) / y[1, :] / torch.mean(torch.log(y_targ[1, :]))**2
        DoDy[2, :] = 2. * (y[2, :] - y_targ[2, :]) / torch.mean(y_targ[2, :])**2 + 2. * (torch.log(y[2, :]) - torch.log(y_targ[2, :])) / y[2, :] / torch.mean(torch.log(y_targ[2, :]))**2
        # DoDy[1, :] = 2. * (y[1, :] - y_targ[1, :]) + 2. * (torch.log(y[1, :]) - torch.log(y_targ[1, :])) / y[1, :]
        # DoDy[2, :] = 2. * (y[2, :] - y_targ[2, :]) + 2. * (torch.log(y[2, :]) - torch.log(y_targ[2, :])) / y[2, :]
    
    else:
        DoDy[1, :] = 2. * (y[1, :] - y_targ[1, :]) + 2. * (torch.log(y[1, :]) - torch.log(y_targ[1, :])) / y[1, :]
        DoDy[2, :] = 2. * (y[2, :] - y_targ[2, :]) + 2. * (torch.log(y[2, :]) - torch.log(y_targ[2, :])) / y[2, :]
    
    # # Add the f terms
    ff_targ = computeF(y_targ, MFParams_targ)
    ff = computeF(y, MFParams)
    dfdy = computeDFdy(y, MFParams)
    
    if normalization == True: 
        DoDy += 2. * (ff - ff_targ) / (torch.mean(ff_targ)**2) * dfdy 
    else:
        DoDy += 2. * (ff - ff_targ) * dfdy 
    return DoDy

# \partial o / \partial yDot
def DoDyDot(y, y_targ, t, MFParams):
    return torch.zeros(y.shape)

# d/dt (\partial o / \partial yDot)
def DDoDyDotDt(y, y_targ, t, MFParams):
    return torch.zeros(y.shape)

# \partial o / \partial \beta
def DoDBeta(y, y_targ, t, MFParams, MFParams_targ, normalization=True):
    DoDbeta = torch.zeros([MFParams.RSParams.shape[0], y.shape[1]])

    # # Add the f terms
    ff_targ = computeF(y_targ, MFParams_targ)
    ff = computeF(y, MFParams)
    dfdbeta = computeDFDBeta(y, MFParams)
    if normalization == True:
        DoDbeta += 2. * (ff - ff_targ) / (torch.mean(ff_targ)**2) * dfdbeta 
    else:
        DoDbeta += 2. * (ff - ff_targ) * dfdbeta 
    return DoDbeta

# ------------------------ Calculate the derivative: DC / D\beta -------------------------
# \partial C / \partial y, unregularized
def DCDy(y, y_targ, t, MFParams):
    DCDy = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    DCDy[0, 1, :] = -1.
    DCDy[1, 0, :] = MFParams.k / MFParams.m
    DCDy[1, 1, :] = MFParams.g * MFParams.RSParams[0] / y[1, :]
    DCDy[1, 2, :] = MFParams.g * MFParams.RSParams[1] / y[2, :]
    # DCDy[2, 1, :] = y[2, :] / MFParams.RSParams[2]
    # DCDy[2, 2, :] = y[1, :] / MFParams.RSParams[2]
    
    # Switch between different laws
    if MFParams.lawFlag == "slip":
        DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2] * (1. + torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]))
        DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2] * (1. + torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]))
    else: # aging law
        DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2]
        DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2]
    
    return DCDy

# \partial C / \partial y, regularized
def DCDy_regularized(y, y_targ, t, MFParams):
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
    
    # Switch between different laws
    if MFParams.lawFlag == "slip":
        DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2] * (1. + torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]))
        DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2] * (1. + torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]))
    else: # aging law
        DCDy[2, 1, :] = y[2, :] * MFParams.RSParams[2]
        DCDy[2, 2, :] = y[1, :] * MFParams.RSParams[2]
#     # DEBUG LINES
#     print("DCDy: ", DCDy)
    
    return DCDy

# \partial C / \partial yDot, unregularized
def DCDyDot(y, y_targ, t, MFParams):
    DCDyDot = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    DCDyDot[0, 0, :] = 1.
    DCDyDot[1, 1, :] = 1.
    DCDyDot[2, 2, :] = 1.
    return DCDyDot

# d/dt (\partial C / \partial yDot)
def DDCDyDotDt(y, y_targ, t, MFParams):
    DDCDyDotDt = torch.zeros([y.shape[0], y.shape[0], y.shape[1]])
    return DDCDyDotDt

# \partial C / \partial \beta, unregularized
def DCDBeta(y, y_targ, t, MFParams):
    DCDBeta = torch.zeros([y.shape[0], MFParams.RSParams.shape[0], y.shape[1]])
    DCDBeta[1, 0, :] = MFParams.g * torch.log(y[1, :] / 1.e-6)
    # DCDBeta[1, 1, :] = MFParams.g * torch.log(1.e-6 * y[2, :] / MFParams.RSParams[2])
    # DCDBeta[1, 2, :] = -MFParams.g * MFParams.RSParams[1] / MFParams.RSParams[2]
    
    DCDBeta[1, 1, :] = MFParams.g * torch.log(1.e-6 * y[2, :] * MFParams.RSParams[2])
    DCDBeta[1, 2, :] = MFParams.g * MFParams.RSParams[1] / MFParams.RSParams[2]

    DCDBeta[1, 3, :] = MFParams.g
    # DCDBeta[2, 2, :] = -y[1, :] * y[2, :] / MFParams.RSParams[2] / MFParams.RSParams[2]
    
    # Switch between slip and aging law
    if MFParams.lawFlag == "slip":
        DCDBeta[2, 2, :] = y[1, :] * y[2, :] * (torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]) + 1.)
    else:
        DCDBeta[2, 2, :] = y[1, :] * y[2, :]
    return DCDBeta

# \partial C / \partial \beta, regularized
def DCDBeta_regularized(y, y_targ, t, MFParams):
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
    
     # Switch between slip and aging law
    if MFParams.lawFlag == "slip":
        DCDBeta[2, 2, :] = y[1, :] * y[2, :] * (torch.log(y[1, :] * y[2, :] * MFParams.RSParams[2]) + 1.)
    else:
        DCDBeta[2, 2, :] = y[1, :] * y[2, :]
    return DCDBeta
#     # DEBUG LINES
#     print("DCDBeta: ", DCDBeta)
    
    return DCDBeta

def computeF(y, MFParams):
    a = MFParams.RSParams[0]
    b = MFParams.RSParams[1]
    DRSInv = MFParams.RSParams[2]
    fStar = MFParams.RSParams[3]

    if MFParams.regularizedFlag == True:
        res = a * torch.asinh(y[1, :] / 2.e-6 * torch.exp((fStar + b * torch.log(1.e-6 * y[2, :] * DRSInv)) / a))
    else:
        res = fStar + a * torch.log(y[1, :] / 1.e-6) + b * torch.log(1.e-6 * y[2, :] * DRSInv)
    
    return res

def computeDFdy(y, MFParams):
    a = MFParams.RSParams[0]
    b = MFParams.RSParams[1]
    DRSInv = MFParams.RSParams[2]
    fStar = MFParams.RSParams[3]
    dfdy = torch.zeros(y.shape)

    if MFParams.regularizedFlag == True:
        Q1 = fStar + b * torch.log(1.e-6 * y[2, :] * DRSInv) 
        Q1 = Q1 / a
        Q2 = y[1, :] / 2 / 1.e-6 * torch.exp(Q1)
        Q2_cliped = torch.clamp(Q2, min = -1.e10, max = 1.e10)
        Q2Term = Q2_cliped / torch.sqrt(Q2_cliped**2 + 1.)

        dfdy[1, :] = a * Q2Term / y[1, :]
        dfdy[2, :] = b * Q2Term / y[2, :] 
    
    else:
        dfdy[1, :] = a / y[1, :]
        dfdy[2, :] = b / y[2, :]
    
    return dfdy

def computeDFDBeta(y, MFParams):
    a = MFParams.RSParams[0]
    b = MFParams.RSParams[1]
    DRSInv = MFParams.RSParams[2]
    fStar = MFParams.RSParams[3]
    dfdBeta = torch.zeros([len(MFParams.RSParams), y.shape[1]])

    if MFParams.regularizedFlag == True:
        Q1 = fStar + b * torch.log(1.e-6 * y[2, :] * DRSInv) 
        Q1 = Q1 / a
        Q2 = y[1, :] / 2 / 1.e-6 * torch.exp(Q1)
        Q2_cliped = torch.clamp(Q2, min = -1.e10, max = 1.e10)
        Q2Term = Q2_cliped / torch.sqrt(Q2_cliped**2 + 1.)

        dfdBeta[0, :] = torch.asinh(Q2Term) - Q1 * Q2Term
        dfdBeta[1, :] = Q2Term * torch.log(1.e-6 * y[2, :] * DRSInv)
        dfdBeta[2, :] = Q2Term * b / DRSInv
        dfdBeta[3, :] = Q2Term
    
    else:
        dfdBeta[0, :] = torch.log(y[1, :] / 1.e-6)
        dfdBeta[1, :] = torch.log(1.e-6 * y[2, :] * DRSInv)
        dfdBeta[2, :] = b / DRSInv
        dfdBeta[3, :] = 1.

    return dfdBeta

    