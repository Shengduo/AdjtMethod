## Import standard librarys
import grp
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import scipy.optimize as opt
import numpy as np

from pathlib import Path
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from random import shuffle

# Target Rate and state properties
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

## For April 3, show the un-convexness of the last two dimensions
# Different start beta, closer to target
DRS_low = -3.
DRS_high = 1.
# beta_low = torch.tensor([0.011, 0.016, pow(10., DRS_low), 0.2])
# beta_high = torch.tensor([0.011, 0.016, pow(10., DRS_high), 0.8])

beta_low = torch.tensor([0.001, 0.001, 1.e-1, 0.58])
beta_high = torch.tensor([0.021, 0.021, 1.e-1, 0.58])

beta_fixed = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
beta0 = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# Number of grid points per dimension in the parameter space
NofGridPts = 50

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)

# Multi data2
ones = 10 * [1.]
tens = 10 * [10.]
VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
                    ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
                    ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
                    tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

# Multi data3

# # Multi data1
# VVs = torch.tensor([[1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                     [1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.], 
#                     [1., 1., 1., 1., 1., 1. ,1., 1., 1., 1., 1., 1., 1., 1., 1.], 
#                     [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]])

tts = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1])])
VtFuncs = []

# Functions
for VV, tt in zip(VVs, tts):
    VtFuncs.append(interp1d(tt, VV))

# Solver to be used
solver = 'dopri5'

# Store all keyword arguments
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VtFuncs' : VtFuncs, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
    'beta_fixed' : beta_fixed,
    'beta_low' : beta_low, 
    'beta_high' : beta_high, 
    'beta_targ' : beta_targ, 
    'NofGridPts' : NofGridPts, 
    'solver' : solver, 
}


# Compute f history based on VtFunc and beta
def cal_f(beta, kwgs):
    tts = kwgs['tts']
    NofTpts = kwgs['NofTpts']
    theta0 = kwgs['theta0']
    VtFuncs = kwgs['VtFuncs']

    # Get all sequences
    Vs = []
    thetas = []
    fs = []
    for VtFunc in VtFuncs:
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        V = torch.tensor(VtFunc(t), dtype=torch.float)

        theta = torch.zeros(t.shape)

        a = beta[0]
        b = beta[1]
        DRSInv = beta[2]
        fStar = beta[3]
        thetaFunc = lambda t, theta: 1. - torch.tensor(VtFunc(torch.clip(t, tt[0], tt[-1])), dtype=torch.float) * theta * DRSInv
        theta = odeint(thetaFunc, theta0, t, atol = 1.e-10, rtol = 1.e-8, method = kwgs['solver'])
        
        f = fStar + a * torch.log(V / 1.e-6) + b * torch.log(1.e-6 * theta * DRSInv)
        Vs.append(V)
        thetas.append(theta)
        fs.append(f)
    
    Vs = torch.stack(Vs)
    thetas = torch.stack(thetas)
    fs = torch.stack(fs)

    return Vs, thetas, fs

def O(f, f_targ, t):
    return torch.trapezoid(
        torch.square(f - f_targ), t
    )


# Plot sequences of friction coefficient
def plotSequences(param1s, param2s, Os, pwd):
    # Plot the data
    plt.figure(figsize = (7, 6), dpi = 300)
    cp = plt.contourf(param1s, param2s, Os)
    
    # Give the color bar
    cbar = plt.colorbar(cp)
    # plt.clim([-5., 0.])
    cbar.set_label('Objective value', fontsize = 20)
    
    plt.xlabel("DRSInv [1/m]", fontsize=20)
    plt.xscale('log')
    plt.ylabel("fStar ", fontsize=20)
    plt.savefig(pwd + "Surf.png", dpi = 300.)
    plt.close()


## Investigate the 2 parameters
t = torch.linspace(tt[0], tt[-1], NofTpts)
AllOs = torch.zeros([kwgs['NofGridPts'], kwgs['NofGridPts']])

# DRS grid
param1s = torch.ones([kwgs['NofGridPts'], kwgs['NofGridPts']]) \
          * (torch.linspace(kwgs['beta_low'][0], kwgs['beta_high'][0], kwgs['NofGridPts']).reshape([-1, 1]))

# fStar grid
param2s = torch.ones([kwgs['NofGridPts'], kwgs['NofGridPts']]) \
          * (torch.linspace(kwgs['beta_low'][1], kwgs['beta_high'][1], kwgs['NofGridPts']))

# Get target fs
V_targs, theta_targs, f_targs = cal_f(beta_targ, kwgs)

# Get all values for the grid
for i in range(kwgs['NofGridPts']):
    for j in range(kwgs['NofGridPts']):
        beta_this = torch.clone(beta0)
        beta_this[0] = param1s[i, j]
        beta_this[1] = param2s[i, j]
        V_thiss, theta_thiss, f_thiss = cal_f(beta_this, kwgs)
        print("Solved for beta_this = ", beta_this, flush=True)
        for f_this, f_targ in zip(f_thiss, f_targs):
            AllOs[i, j] += O(f_this, f_targ, t)

kwgs['AllOs'] = AllOs
kwgs['param1s'] = param1s
kwgs['param2s'] = param2s

# Save the results
pwd ="./plots/FricSeqGen_Plot2Params_ab/"
Path(pwd).mkdir(parents=True, exist_ok=True)
plotSequences(param1s, param2s, AllOs, pwd)

# Append to the keywords arguments
torch.save(kwgs, pwd + "kwgs.pt")
print("AllOs: ", AllOs)
