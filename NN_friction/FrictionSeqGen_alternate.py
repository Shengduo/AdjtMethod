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
from random import shuffle

# Target Rate and state properties
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# # Start beta
# beta0 = torch.tensor([0.009, 0.012, 1. / 1.e2, 0.3])

# Different start beta, closer to target
beta_low = torch.tensor([0.001, 0.001, 0.001, 0.1])
beta_high = torch.tensor([1., 1., 1.e6, 0.9])

# # For 0323 alternating drs fstar
# beta0 = torch.tensor([0.011, 0.016, 1. / 2.e1, 0.7])
# beta_fixed = torch.tensor([1, 1, 0, 0], dtype=torch.bool)

# For 0323 alternating a b drs fstar
beta0 = torch.tensor([0.009, 0.012, 1. / 2.e1, 0.7])
beta_fixed = torch.tensor([0, 0, 0, 0], dtype=torch.bool)

# Document the unfixed groups
# beta_unfixed_groups = [[0], [1], [2], [3]]
# beta_unfixed_NofIters = torch.tensor([3, 3, 3, 3])
beta_unfixed_groups = [[0, 1, 2, 3]]
beta_unfixed_NofIters = torch.tensor([3])

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)

# Multi data2
ones = 10 * [1.-2]
tens = 10 * [10.]
VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
                    ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
                    ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
                    tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

# Prescribed velocities - testing
VV_tests = torch.tensor([ones + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + tens + tens + ones + ones, \
                         ones + ones + tens + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + ones + ones])

tts = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1])])

# Times at which the velocities are prescribed - testing
tt_tests = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                        torch.linspace(0., 30., VVs.shape[1])])

VtFuncs = []
VtFunc_tests = []

# Functions
for VV, tt in zip(VVs, tts):
    VtFuncs.append(interp1d(tt, VV))

# Functions for V-t interpolation
for VV, tt in zip(VV_tests, tt_tests):
    VtFunc_tests.append(interp1d(tt, VV))

# Store all keyword arguments
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VtFuncs' : VtFuncs, 
    'VV_tests' : VV_tests,
    'tt_tests' : tt_tests, 
    'VtFunc_tests' : VtFunc_tests, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
    'beta_fixed' : beta_fixed,
    'beta_unfixed_groups' : beta_unfixed_groups, 
    'beta_unfixed_NofIters' : beta_unfixed_NofIters, 
    'beta0' : beta0, 
    'beta_low' : beta_low, 
    'beta_high' : beta_high, 
}

# Compute f history based on VtFunc and beta
def cal_f_beta(beta, kwgs, tts, VtFuncs, std_noise = 0.001):
    NofTpts = kwgs['NofTpts']
    theta0 = kwgs['theta0']

    # Get all sequences
    Vs = []
    thetas = []
    fs = []
    for tt, VtFunc in zip(tts, VtFuncs):
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
        mean_f = torch.mean(f);
        f = f + std_noise * mean_f * torch.randn(f.shape)
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
def plotSequences(beta, kwgs, pwd):
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['tts'], kwgs['VtFuncs'], 0.)
    f_targs = kwgs['f_targs']
    lws = torch.linspace(3.0, 1.0, len(Vs))
    NofTpts = kwgs['NofTpts']
    for idx, (tt, f_targ, f) in enumerate(zip(kwgs['tts'], f_targs, fs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f, linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        plt.title("Train sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "TrainSeq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, (tt, V) in enumerate(zip(kwgs['tts'], Vs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.plot(t, V, linewidth=lws[idx])
        lgd.append("Train Seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TrainSeqs.png", dpi = 300.)
    plt.close()

    # -------------------- For test data --------------------------
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['tt_tests'], kwgs['VtFunc_tests'], 0.)
    f_targs = kwgs['f_targ_tests']
    lws = torch.linspace(3.0, 1.0, len(Vs))

    for idx, (tt, f_targ, f) in enumerate(zip(kwgs['tt_tests'], f_targs, fs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f, linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        plt.title("Test sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "TestSeq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, (tt, V) in enumerate(zip(kwgs['tt_tests'], Vs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.plot(t, V, linewidth=lws[idx])
        lgd.append("Test Seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TestSeqs.png", dpi = 300.)
    plt.close()

# Load data
shit =  torch.load('./data/RandnData1_std_1e-3_0504.pt')
kwgs['f_targs'] = shit['f_targs']
kwgs['f_targ_tests'] = shit['f_targ_tests']

## Invert on an problem
t = torch.linspace(tt[0], tt[-1], NofTpts)

## ------------------------------------ Gradient descent ------------------------------------ 
# Maximum alternative iterations
max_iters = 50

# Store all betas and all Os
All_betas = []
All_Os = []
All_grads = []

# Start the outer loop of iterations
beta_this = beta0
for alt_iter in range(max_iters):
    # Print out section
    print("*" * 100)
    print("*", " "*40, "Outer Iteration ", str(alt_iter), " " * 40, "*")
    print("*" * 100)

    ## Inner iteration, get fixed randomly
    inner_groups = []
    for idx, NofIters in enumerate(kwgs['beta_unfixed_NofIters']):
        inner_groups = inner_groups + [kwgs['beta_unfixed_groups'][idx] for i in range(NofIters)]
    
    # Permute it 
    shuffle(inner_groups)
    print("Inner groups: ", inner_groups)
    
    for (grp_idx, release_idx) in enumerate(inner_groups):
    # for grp_idx, release_idx in enumerate(kwgs['beta_unfixed_groups']):
        beta_fixed_this = torch.ones(len(kwgs['beta0']), dtype=torch.bool)
        beta_fixed_this[release_idx] = False
        
        # Print out which values are fixed
        print("~" * 20, "beta_fixed: ", beta_fixed_this, "~"*20)


        # max_step = torch.tensor([0.005, 0.005, 0.01, 0.1])
        max_step = torch.tensor([1., 1., 1., 1.])
        max_step[beta_fixed_this] = 1.e30
        beta0_this = beta_this
        V_thiss, theta_thiss, f_thiss = cal_f_beta(beta_this, kwgs, kwgs['tts'], kwgs['VtFuncs'], 0.)

        O_this = 0.
        grad_this = torch.zeros(4)
        for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, kwgs['f_targs']):
            O_this += O(f_this, f_targ, t)
            grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)

        # print("=" * 40, "Inner Iteration ", str(0), " ", "=" * 40)
        # print("Initial beta: ", beta_this)
        # print("O: ", O_this)
        # print("Gradient: ", grad_this, flush=True)
        
        # Store the first O and grad if both the outside and inside iteration number is 0
        if alt_iter == 0 and grp_idx == 0:
            print("=" * 40, "Inner Iteration ", str(0), " ", "=" * 40)
            print("Initial beta: ", beta_this)
            print("O: ", O_this)
            print("Gradient: ", grad_this, flush=True)
            All_betas.append(beta_this)
            All_Os.append(O_this)
            All_grads.append(grad_this)
        
        for i in range(1):
        # for i in range(kwgs['beta_unfixed_NofIters'][grp_idx]):
            max_eta = torch.min(torch.abs(max_step / grad_this))
            # Line search
            iter = 0
            O_trial = O_this
            while (iter <= 12 and O_trial >= O_this):
                beta_trial = beta_this - grad_this * max_eta * pow(2, -iter)
                beta_trial[beta_fixed_this] = beta0_this[beta_fixed_this]
                beta_trial = torch.clip(beta_trial, kwgs['beta_low'], kwgs['beta_high'])
                V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['tts'], kwgs['VtFuncs'], 0.)
                O_trial = 0.
                
                for V_trial, theta_trial, f_trial, f_targ in zip(V_trials, theta_trials, f_trials, kwgs['f_targs']):
                    O_trial += O(f_trial, f_targ, t)
                # print("beta, O" + str(iter) + ": ", beta_trial, O_trial)
                iter += 1

            beta_this = beta_trial
            V_thiss = V_trials
            theta_thiss = theta_trials
            f_thiss = f_trials
            O_this = O_trial

            # Get new grad
            grad_this = torch.zeros(4)
            for V_this, theta_this, f_this, f_targ in zip(V_thiss, theta_thiss, f_thiss, kwgs['f_targs']):
                grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)
            
            print("=" * 40, " Inner Iteration ", str(i + 1), " ", "=" * 40)
            print("Optimized beta: ", beta_this)
            print("Training data O: ", O_this)
            print("Gradient: ", grad_this, flush=True)
            All_betas.append(beta_this)
            All_Os.append(O_this)
            All_grads.append(grad_this)
    if alt_iter % 10 == 0:
        O_test = 0.
        V_tests, theta_tests, f_tests = cal_f_beta(beta_this, kwgs, kwgs['tt_tests'], kwgs['VtFunc_tests'], 0.)
        for V_test, theta_test, f_test, f_targ in zip(V_tests, theta_tests, f_tests, kwgs['f_targs']):
            O_test += O(f_trial, f_targ, t)
        print("-!" * 40)
        print("Testing O: ", O_test), 
        print("-!" * 40)


# Save a figure of the result
# pwd ="./plots/FricSeqGen0323_alternating_DrsFStar/"
pwd ="./plots/Test0504_std_1e-3_AdjMtd/"
Path(pwd).mkdir(parents=True, exist_ok=True)

# Append to the keywords arguments
kwgs['All_betas'] = All_betas
kwgs['All_Os'] = All_Os
kwgs['All_grads'] = All_grads
torch.save(kwgs, pwd + "kwgs.pt")

# Print out best performance and plot sequences
best_idx = All_Os.index(min(All_Os))
best_grad = All_grads[best_idx]
best_O = All_Os[best_idx]
best_beta = All_betas[best_idx]

# Print results
print("~" * 40, " Final Optimization Answer ", "~" * 40)
print("Optimized beta: ", best_beta)
print("O: ", best_O)
print("Gradient: ", best_grad, flush=True)

plotSequences(best_beta, kwgs, pwd)


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