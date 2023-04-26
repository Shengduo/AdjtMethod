## Import standard librarys
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import torch.optim as optim
import scipy.optimize as opt
import numpy as np

from pathlib import Path
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# ----------------------------------------- README ----------------------------------------------
# Here we try to approximate the rate-and-state friction model using the formulation below:
# f(V, \xi) &= f_0 + NN1(V, \xi, log(V), log(\xi))
# d\xi / dt &= NN2(V, \xi, log(V), log(\xi))
# -----------------------------------------------------------------------------------------------
# Class that computes f-sequence given V-sequence
class NN_computeF(nn.Module):
    # Constructor
    def __init__(self, kwgs):
        super().__init__()

        # Store the kwgs
        self.kwgs = kwgs

        # For f = NN1(V, log(V), Xi, log(Xi))
        NN1s = kwgs['NN1s']
        NN1_input_dim = kwgs['NN1_input_dim']
        NN1_output_dim = kwgs['NN1_output_dim']

        # \dot{\xi} = NN2(V, log(V), \xi, log(\xi))
        NN2s = kwgs['NN2s']
        NN2_input_dim = kwgs['NN2_input_dim']
        NN2_output_dim = kwgs['NN2_output_dim']

        # Define function NN1
        self.NN1 = nn.Sequential(
            nn.Linear(NN1_input_dim, NN1s[0]), 
            nn.ReLU(),
        )
        
        for i in range(len(NN1s) - 1):
            self.NN1.append(nn.Linear(NN1s[i], NN1s[i + 1]))
            self.NN1.append(nn.ReLU())
        
        self.NN1.append(nn.Linear(NN1s[-1], NN1_output_dim))

        # Define function NN2
        self.NN2 = nn.Sequential(
            nn.Linear(NN2_input_dim, NN2s[0]), 
            nn.ReLU(),
        )
        
        for i in range(len(NN2s) - 1):
            self.NN2.append(nn.Linear(NN2s[i], NN2s[i + 1]))
            self.NN2.append(nn.ReLU())
        
        self.NN2.append(nn.Linear(NN2s[-1], NN2_output_dim))

    def forward(self, VtFunc):
        NofTpts = kwgs['NofTpts']

        # Get all sequences
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        V = torch.tensor(VtFunc(t), dtype=torch.float)

        XiFunc = lambda t, Xi: self.NN2(torch.concat([VtFunc(t), torch.log(VtFunc(t)), Xi(t), torch.log(Xi(t))]))
        Xi = odeint(XiFunc, Xi0, t, atol = 1.e-10, rtol = 1.e-8)
        
        f = f0 + self.NN1(torch.transpose(torch.stack([V, torch.log(V), Xi, torch.log(Xi)])))

        return f




# Define the class multi-layer perceptron for function NN1 and function NN2
class PP(nn.Module):
    # Constructor
    def __init__(self, NNs, input_dim = 1, output_dim = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, NNs[0]), 
            nn.ReLU(),
        )
        
        for i in range(len(NNs) - 1):
            self.fc.append(nn.Linear(NNs[i], NNs[i + 1]))
            self.fc.append(nn.ReLU())
        
        self.fc.append(nn.Linear(NNs[-1], output_dim))
    
    # Forward function
    def forward(self, x):
        return self.fc(x)

# Target Rate and state properties for the target data generation
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)

# Multi data2
ones = 10 * [1.]
tens = 10 * [10.]

# Prescribed velocities
VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
                    ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
                    ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
                    tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])


# Times at which the velocities are prescribed
tts = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1])])
VtFuncs = []


# Functions for V-t interpolation
for VV, tt in zip(VVs, tts):
    VtFuncs.append(interp1d(tt, VV))

# Number of hidden variables, i.e. the dimension of xi
DimXi = 1
Xi0 = torch.ones(DimXi)

# NN parameters for f = fStar + NN1(V, log(V), \xi, log(\xi))
f0 = 0.58
NN1_input_dim = 2 + 2 * DimXi
NN1s = [16, 64, 64, 16]
NN1_output_dim = 1

# NN parameters for \dot{\xi} = NN2(V, log(V), \xi, log(\xi))
NN2_input_dim = 2 + 2 * DimXi
NN2s = [16, 64, 64, 16]
NN2_output_dim = DimXi

# Store all the input parameters as a keyword dictionary
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VtFuncs' : VtFuncs, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
    'Xi0' : Xi0, 
    'NN1_input_dim' : NN1_input_dim, 
    'NN1s' : NN1s, 
    'NN1_output_dim' : NN1_output_dim, 
    'NN2_input_dim' : NN2_input_dim, 
    'NN2s' : NN2s, 
    'NN2_output_dim' : NN2_output_dim, 
    'f0' : f0, 
}


# Compute f history based on VtFunc and beta
def cal_f_beta(beta, kwgs):
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
        theta = odeint(thetaFunc, theta0, t, atol = 1.e-10, rtol = 1.e-8)
        
        f = fStar + a * torch.log(V / 1.e-6) + b * torch.log(1.e-6 * theta * DRSInv)
        Vs.append(V)
        thetas.append(theta)
        fs.append(f)
    
    Vs = torch.stack(Vs)
    thetas = torch.stack(thetas)
    fs = torch.stack(fs)

    return Vs, thetas, fs

# Compute f history based on VtFunc and the two NNs
def cal_f_NNs(NN1, NN2, kwgs):
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

        XiFunc = lambda t, Xi: NN2(torch.concat([VtFunc(t), torch.log(VtFunc(t)), Xi(t), torch.log(Xi(t))]))
        Xi = odeint(XiFunc, Xi0, t, atol = 1.e-10, rtol = 1.e-8)
        
        f = f0 + NN1(torch.transpose(torch.stack([V, torch.log(V), Xi, torch.log(Xi)])))
        Vs.append(V)
        fs.append(f)
    
    Vs = torch.stack(Vs)
    fs = torch.stack(fs)

    return Vs, fs

def O(f, f_targ, t):
    return torch.trapezoid(
        torch.square(f - f_targ), t
    )


# Plot sequences of friction coefficient
def plotSequences(beta_targ, NN1, NN2, kwgs, pwd):
    Vs, fs = cal_f_NNs(NN1, NN2, kwgs)
    V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs)
    lws = torch.linspace(3.0, 1.0, len(Vs))

    for idx, (f_targ, f) in enumerate(zip(f_targs, fs)):
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f, linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        plt.title("Sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "Seq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, V in enumerate(Vs):
        plt.plot(t, V, linewidth=lws[idx])
        lgd.append("seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "GenSeqs.png", dpi = 300.)
    plt.close()


## Invert on an problem
t = torch.linspace(tt[0], tt[-1], NofTpts)

V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs)
# print('V_targ: ', V_targ)
# print('theta_targ: ', theta_targ)
# print('f_targ: ', f_targ)

## Gradient descent:
max_iters = 200

# Gradient descent
for epoch in range(max_iters):
    
    print("=" * 40, " Iteration ", str(i + 1), " ", "=" * 40)
    print("Optimized beta: ", beta_this)
    print("O: ", O_this)
    print("Gradient: ", grad_this, flush=True)

# Save a figure of the result
pwd ="./plots/FricSeqGen0404_DRSfStar_2/"
Path(pwd).mkdir(parents=True, exist_ok=True)
plotSequences(beta_this, beta_targ, kwgs, pwd)


# ## Check numerical derivatives
# beta0=torch.tensor([0.0109, 0.0161, 0.2000, 0.5800])

# # Gradient descent
# beta_this = beta0
# V_thiss, theta_thiss, f_thiss = cal_f(beta_this, kwgs)

# O_this = 0.
# grad_this = torch.zeros(4)
# for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, f_targs):
#     O_this += O(f_this, f_targ, t)
#     grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)

# print("Grad by Adjoint: ", grad_this)

# # Numerical gradients
# inc = 0.01
# numerical_grad0 = torch.zeros(beta0.shape)
# for i in range(len(beta0)):
#     beta_plus = torch.clone(beta0)
#     beta_plus[i] *= (1 + inc)
#     print("beta_plus: ", beta_plus)
#     Vps, thetasp, fps = cal_f(beta_plus, kwgs)

#     Op = 0.
#     for f_targ, fp in zip(f_targs, fps):
#         Op += O(fp, f_targ, t)

#     beta_minus = torch.clone(beta0)
#     beta_minus[i] *= (1 - inc)
#     print("beta_minus: ", beta_minus)
#     Vms, thetams, fms = cal_f(beta_minus, kwgs)
    
#     Om = 0.
#     for f_targ, fm in zip(f_targs, fms):
#         Om += O(fm, f_targ, t)

#     numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

# print("Grad by finite difference: ", numerical_grad0)