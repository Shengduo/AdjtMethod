## Import standard librarys
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


# Target Rate and state properties
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# # Start beta
# beta0 = torch.tensor([0.009, 0.012, 1. / 1.e2, 0.3])

# Different start beta, closer to target
beta0 = torch.tensor([0.010, 0.017, 2. / 1.e1, 0.6])

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

# Store all keyword arguments
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VtFuncs' : VtFuncs, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
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
        theta = odeint(thetaFunc, theta0, t, atol = 1.e-10, rtol = 1.e-8)
        
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

def grad(beta, t, V, theta, f, f_targ, kwgs):
    integrand = torch.zeros([len(beta), len(t)])
    integrand[0, :] = torch.log(V / 1.e-6)
    integrand[1, :] = torch.log(1.e-6 * theta * beta[2])
    integrand[2, :] = beta[1] / beta[2]
    integrand[3, :] = 1.
    integrand = 2 * (f - f_targ) * integrand
    dodTheta = interp1d(t, 2. * (f - f_targ) * beta[1] / theta)
    dCdTheta = interp1d(t, V * beta[2])

    # Adjoint
    # print(torch.flip(t[-1] - t, [0]))
    laFunc = lambda tau, la: -torch.tensor(dodTheta(torch.clip(t[-1]-tau, t[0], t[-1])), dtype=torch.float) - la * torch.tensor(dCdTheta(torch.clip(t[-1]-tau, t[0], t[-1])), dtype=torch.float)
    la = odeint(laFunc, torch.tensor(0.), torch.flip(t[-1] - t, [0]), atol = 1.e-10, rtol = 1.e-8)
    la = torch.flip(la, [0])
    # print("la: ", la)
    integrand[2, :] += la * V * theta
    res = torch.trapezoid(
        integrand, t
    )

    
    return res

# Plot sequences of friction coefficient
def plotSequences(beta, beta_targ, kwgs, pwd):
    Vs, thetas, fs = cal_f(beta, kwgs)
    V_targs, theta_targs, f_targs = cal_f(beta_targ, kwgs)
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

V_targs, theta_targs, f_targs = cal_f(beta_targ, kwgs)
# print('V_targ: ', V_targ)
# print('theta_targ: ', theta_targ)
# print('f_targ: ', f_targ)

## Gradient descent:
max_iters = 6
# max_step = torch.tensor([0.005, 0.005, 0.01, 0.1])
max_step = torch.tensor([1., 1., 1., 1.])

# Gradient descent
beta_this = beta0
V_thiss, theta_thiss, f_thiss = cal_f(beta_this, kwgs)

O_this = 0.
grad_this = torch.zeros(4)
for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, f_targs):
    O_this += O(f_this, f_targ, t)
    grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)

print("=" * 40, " Iteration ", str(0), " ", "=" * 40)
print("Initial beta: ", beta_this)
print("O: ", O_this)
print("Gradient: ", grad_this, flush=True)

for i in range(max_iters):
    max_eta = torch.min(torch.abs(max_step / grad_this))
    # Line search
    iter = 0
    O_trial = O_this
    while (iter <= 20 and O_trial >= O_this):
        beta_trial = beta_this - grad_this * max_eta * pow(2, -iter)
        V_trials, theta_trials, f_trials = cal_f(beta_trial, kwgs)
        O_trial = 0.
        
        for V_trial, theta_trial, f_trial, f_targ in zip(V_trials, theta_trials, f_trials, f_targs):
            O_trial += O(f_trial, f_targ, t)
        
        iter += 1

    beta_this = beta_trial
    V_thiss = V_trials
    theta_thiss = theta_trials
    f_thiss = f_trials
    O_this = O_trial

    # Get new grad
    grad_this = torch.zeros(4)
    for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, f_targs):
        grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)
    
    print("=" * 40, " Iteration ", str(i + 1), " ", "=" * 40)
    print("Optimized beta: ", beta_this)
    print("O: ", O_this)
    print("Gradient: ", grad_this, flush=True)

# Save a figure of the result
pwd ="./plots/FricSeqGen0322_multi2_closerBeta/"
Path(pwd).mkdir(parents=True, exist_ok=True)
plotSequences(beta_this, beta_targ, kwgs, pwd)


## Check numerical derivatives
beta0=torch.tensor([0.0103, 0.0168, 0.2000, 0.6000])
# Gradient descent
beta_this = beta0
V_thiss, theta_thiss, f_thiss = cal_f(beta_this, kwgs)

O_this = 0.
grad_this = torch.zeros(4)
for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, f_targs):
    O_this += O(f_this, f_targ, t)
    grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)

print("Grad by Adjoint: ", grad_this)

# Numerical gradients
inc = 0.01
numerical_grad0 = torch.zeros(beta0.shape)
for i in range(len(beta0)):
    beta_plus = torch.clone(beta0)
    beta_plus[i] *= (1 + inc)
    print("beta_plus: ", beta_plus)
    Vps, thetasp, fps = cal_f(beta_plus, kwgs)

    Op = 0.
    for f_targ, fp in zip(f_targs, fps):
        Op += O(fp, f_targ, t)

    beta_minus = torch.clone(beta0)
    beta_minus[i] *= (1 - inc)
    print("beta_minus: ", beta_minus)
    Vms, thetams, fms = cal_f(beta_minus, kwgs)
    
    Om = 0.
    for f_targ, fm in zip(f_targs, fms):
        Om += O(fm, f_targ, t)

    numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

print("Grad by finite difference: ", numerical_grad0)