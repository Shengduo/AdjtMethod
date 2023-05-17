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
beta_unfixed_groups = [[0], [1], [2], [3]]
beta_unfixed_NofIters = torch.tensor([3, 3, 3, 3])
# beta_unfixed_groups = [[0, 1, 2, 3]]
# beta_unfixed_NofIters = torch.tensor([1])

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)

# Multi data2
ones = 10 * [1.e-8]
tens = 10 * [10.]

    
# VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
#                     ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
#                     ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
#                     tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

# VVseeds = torch.randint(-10, 3, [4, 15])
# print("VVseeds: ", VVseeds)

VVseeds = torch.tensor([[-10,  -9,  -2,  -1,  -4,   0,  -6,  -3,  -6,  -6, -10,   0,  -9,  -5,  -9],
                        [ -7,   2,  -3,  -3,  -6,   1,  -5,  -9,  -8,  -4, -10,  -9,   0,  -3,  -7],
                        [ -5,  -8,  -7,  -3,  -4,  -3,   0,  -9,  -1,   0,  -6,  -8,  -6,  -2,  -6],
                        [ -5,  -6,   1,   1,  -2,   1,   2,  -8,  -6,   1,  -2,   2, -10,  -6,   1], 
                        [-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10], 
                        [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1], 
                        [-10, -10, -10, -10, -10, -10, -10, -10,   1,   1,   1,   1,   1,   1,   1], 
                        [  1,   1,   1,   1,   1,   1,   1, -10, -10, -10, -10, -10, -10, -10, -10]])

VV_seeds_len = [10, 10, 10, 10, 100, 100, 100, 100]

# Selected
selected = [0, 2, 6, 7]
VVseeds = VVseeds[selected, :]
VV_seeds_len = [VV_seeds_len[i] for i in selected]

VVs = []
tts = []

# Generate VVs and tts
for idx, VVseed in enumerate(VVseeds):
    VV = torch.zeros(len(VVseed) * VV_seeds_len[idx])
    for j in range(len(VVseed)):
        VV[VV_seeds_len[idx] * j : VV_seeds_len[idx] * (j + 1)] = torch.pow(10., VVseed[j])
    VVs.append(VV)
    tt = torch.linspace(0., 0.2 * len(VV), len(VV))
    tts.append(tt)

# Figure out the jump points of V-sequences
JumpIdxs = []
for VV in VVs:
    JumpIdx = [0]
    for i in range(1, len(VV)):
        if VV[i] != VV[i - 1]:
            JumpIdx.append(i)
    JumpIdx.append(len(VV) - 1)
    JumpIdxs.append(JumpIdx)

# Prescribed velocities - testing
VV_tests = torch.tensor([ones + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + tens + tens + ones + ones, \
                         ones + ones + tens + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + ones + ones])

# Figure out the jump points of V-test-sequences
JumpIdx_tests = []
for VV in VV_tests:
    JumpIdx = [0]
    for i in range(1, len(VV)):
        if VV[i] != VV[i - 1]:
            JumpIdx.append(i)
    JumpIdx.append(len(VV) - 1)
    JumpIdx_tests.append(JumpIdx)




# Times at which the velocities are prescribed - testing
tt_tests = torch.stack([torch.linspace(0., 30., VV_tests.shape[1]),
                        torch.linspace(0., 30., VV_tests.shape[1])])

VtFuncs = []
VtFunc_tests = []
ts = []
t_tests = []
t_JumpIdxs = []
t_JumpIdx_tests = []

# Functions, ts and t_JumpIdxs
t_tt_times = [10, 10, 10, 10, 3, 3, 3, 3]
for JumpIdx, VV, tt, t_tt_time in zip(JumpIdxs, VVs, tts, t_tt_times):
    VtFunc = []
    t = torch.linspace(tt[0], tt[-1], t_tt_time * len(tt))
    t_JumpIdx = [0]

    for i in range(len(JumpIdx) - 1):
        this_tt = tt[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
        this_VV = VV[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
        this_VV[-1] = this_VV[-2]
        VtFunc.append(interp1d(this_tt, this_VV))

        isIdx =  (t <= this_tt[-1])
        if isIdx[-1] == True:
            t_JumpIdx.append(len(isIdx))
        else:
            for j in range(len(isIdx)):
                if isIdx[j] == False:
                    t_JumpIdx.append(j)
                    break
    
    t_JumpIdxs.append(t_JumpIdx)
    ts.append(t)
    VtFuncs.append(VtFunc)

# Functions, ts and t_JumpIdxs
for JumpIdx, VV, tt in zip(JumpIdx_tests, VV_tests, tt_tests):
    VtFunc = []
    t = torch.linspace(tt[0], tt[-1], 10 * len(tt))
    t_JumpIdx = [0]

    for i in range(len(JumpIdx) - 1):
        this_tt = tt[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
        this_VV = VV[JumpIdx[i] : JumpIdx[i + 1] + 1].clone()
        this_VV[-1] = this_VV[-2]
        VtFunc.append(interp1d(this_tt, this_VV))

        isIdx =  (t <= this_tt[-1])
        if isIdx[-1] == True:
            t_JumpIdx.append(len(isIdx))
        else:
            for j in range(len(isIdx)):
                if isIdx[j] == False:
                    t_JumpIdx.append(j)
                    break
    
    t_JumpIdx_tests.append(t_JumpIdx)
    t_tests.append(t)
    VtFunc_tests.append(VtFunc)

# Store all keyword arguments
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VtFuncs' : VtFuncs, 
    'JumpIdxs' : JumpIdxs, 
    'VV_tests' : VV_tests, 
    'tt_tests' : tt_tests, 
    'VtFunc_tests' : VtFunc_tests, 
    'JumpIdx_tests' : JumpIdx_tests, 
    'NofTpts' : NofTpts, 
    'ts' : ts, 
    't_tests' : t_tests, 
    't_JumpIdxs' : t_JumpIdxs, 
    't_JumpIdx_tests' : t_JumpIdx_tests, 
    'theta0' : theta0, 
    'beta_fixed' : beta_fixed, 
    'beta_unfixed_groups' : beta_unfixed_groups, 
    'beta_unfixed_NofIters' : beta_unfixed_NofIters, 
    'beta0' : beta0, 
    'beta_low' : beta_low, 
    'beta_high' : beta_high, 
}

# Compute f history based on VtFunc and beta
def cal_f_beta(beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001):
    theta0 = kwgs['theta0']

    # Get all sequences
    Vs = []
    thetas = []
    fs = []

    for t_this, t_JumpIdx, VtFunc, tt, JumpIdx in zip(ts, t_JumpIdxs, VtFuncs, tts, JumpIdxs):
        # V = torch.tensor(VtFunc(t), dtype=torch.float)
        V = torch.zeros(t_this.shape)
        theta = torch.zeros(t_this.shape)

        a = beta[0]
        b = beta[1]
        DRSInv = beta[2]
        fStar = beta[3]

        # Loop thru all sections of VtFunc
        theta0_this = theta0
        for index, vtfunc in enumerate(VtFunc):
            t_this_interval = t_this[t_JumpIdx[index] : t_JumpIdx[index + 1]] 

            # Append with the first and last tt
            t_this_interval = torch.cat([torch.tensor([tt[JumpIdx[index]]]), t_this_interval, torch.tensor([tt[JumpIdx[index + 1]]])])
            
            # Update V
            V[t_JumpIdx[index] : t_JumpIdx[index + 1]] = torch.tensor(vtfunc(t_this[t_JumpIdx[index] : t_JumpIdx[index + 1]]), dtype=torch.float)

            # Compute theta
            thetaFunc = lambda t, theta: 1. - torch.tensor(vtfunc(torch.clip(t, tt[JumpIdx[index]], tt[JumpIdx[index + 1]])), dtype=torch.float) * theta * DRSInv
            i = 0
            j = len(t_this_interval)
            if (t_this_interval[0] == t_this_interval[1]):
                i = i + 1
            if (t_this_interval[-1] == t_this_interval[-2]):
                j = -1
            theta_this = odeint(thetaFunc, theta0_this, t_this_interval[i : j], atol = 1.e-10, rtol = 1.e-8)

            # Update theta
            if i == 1:
                i = 0
            else:
                i = 1
            if j == -1:
                j = len(theta_this)
            else:
                j = -1

            theta[t_JumpIdx[index] : t_JumpIdx[index + 1]] = theta_this[i : j]
            theta0_this = theta_this[-1]

        
        f = fStar + a * torch.log(V / 1.e-6) + b * torch.log(1.e-6 * theta * DRSInv)
        mean_f = torch.mean(f);
        f = f + std_noise * mean_f * torch.randn(f.shape)
        Vs.append(V)
        thetas.append(theta)
        fs.append(f)
    
    # Vs = torch.stack(Vs)
    # thetas = torch.stack(thetas)
    # fs = torch.stack(fs)

    return Vs, thetas, fs

def O(f, f_targ, t):
    return torch.trapezoid(
        torch.square(f - f_targ), t
    ) / (t[-1] - t[0])

def grad(beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs):
    integrand = torch.zeros([len(beta), len(t)])
    integrand[0, :] = torch.log(V / 1.e-6)
    integrand[1, :] = torch.log(1.e-6 * theta * beta[2])
    integrand[2, :] = beta[1] / beta[2]
    integrand[3, :] = 1.
    integrand = 2 * (f - f_targ) * integrand
    # dodTheta = interp1d(t, 2. * (f - f_targ) * beta[1] / theta)
    # dCdTheta = interp1d(t, V * beta[2])

    # Adjoint
    # print(torch.flip(t[-1] - t, [0]))
    la = torch.zeros(t.shape)

    la_this0 = torch.tensor(0.)
    for index in range(len(JumpIdx) - 1):
        idx_this = len(JumpIdx) - 1 - index
        t_this_interval = torch.cat(
            [torch.tensor([tt[JumpIdx[idx_this - 1]]]), 
            t[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]], 
            torch.tensor([tt[JumpIdx[idx_this]]])]
            )
        
        # Remove the duplicates
        original_t_shape = t_this_interval.shape
        i = 0
        j = len(t_this_interval)
        if t_this_interval[i] == t_this_interval[i + 1]:
            i = i + 1
        if t_this_interval[-1] == t_this_interval[-2]:
            j = - 1
        t_this_interval = t_this_interval[i : j]

        f_this_interval = torch.zeros(original_t_shape)
        f_this_interval[1: -1] = f[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]]
        f_this_interval[0] = f_this_interval[1]
        f_this_interval[-1] = f_this_interval[-2]
        f_this_interval = f_this_interval[i : j]
        
        f_targ_this_interval = torch.zeros(original_t_shape)
        f_targ_this_interval[1: -1] = f_targ[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]]
        f_targ_this_interval[0] = f_targ_this_interval[1]
        f_targ_this_interval[-1] = f_targ_this_interval[-2]
        f_targ_this_interval = f_targ_this_interval[i : j]

        theta_this_interval = torch.zeros(original_t_shape)
        theta_this_interval[1: -1] = theta[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]]
        theta_this_interval[0] = theta_this_interval[1]
        theta_this_interval[-1] = theta_this_interval[-2]
        theta_this_interval = theta_this_interval[i : j]

        V_this_interval = torch.zeros(original_t_shape)
        V_this_interval[1: -1] = V[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]]
        V_this_interval[0] = V_this_interval[1]
        V_this_interval[-1] = V_this_interval[-2]
        V_this_interval = V_this_interval[i : j]

        dodTheta = interp1d(t_this_interval, 2. * (f_this_interval - f_targ_this_interval) * beta[1] / theta_this_interval)
        dCdTheta = interp1d(t_this_interval, V_this_interval * beta[2])

        laFunc = lambda tau, la: -torch.tensor(dodTheta(torch.clip(t[-1]-tau, t_this_interval[0], t_this_interval[-1])), dtype=torch.float) \
                                - la * torch.tensor(dCdTheta(torch.clip(t[-1]-tau, t_this_interval[0], t_this_interval[-1])), dtype=torch.float)
        la_flipped = odeint(laFunc, la_this0, torch.flip(t[-1] - t_this_interval, [0]), atol = 1.e-10, rtol = 1.e-8)
        la_this_interval = torch.flip(la_flipped, [0])
        if i == 1:
            i = 0
        else:
            i = 1
        if j == -1:
            j = len(la_this_interval)
        else:
            j = -1

        la[t_JumpIdx[idx_this - 1] : t_JumpIdx[idx_this]] = la_this_interval[i : j]

        # Refresh la_this0
        la_this0 = la_this_interval[0]

    # print("la: ", la)
    integrand[2, :] += la * V * theta
    res = torch.trapezoid(
        integrand, t
    ) / (t[-1] - t[0])

    
    return res

# Plot sequences of friction coefficient
def plotSequences(beta, kwgs, pwd):
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
    f_targs = kwgs['f_targs']
    lws = torch.linspace(3.0, 1.0, len(Vs))
    for idx, (tt, t, f_targ, f) in enumerate(zip(kwgs['tts'], kwgs['ts'], f_targs, fs)):
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

    for idx, (tt, t, V) in enumerate(zip(kwgs['tts'], kwgs['ts'], Vs)):
        plt.semilogy(t, V, linewidth=lws[idx])
        lgd.append("Train Seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TrainSeqs.png", dpi = 300.)
    plt.close()

    # -------------------- For test data --------------------------
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0.)
    f_targs = kwgs['f_targ_tests']
    lws = torch.linspace(3.0, 1.0, len(Vs))

    for idx, (tt, t, f_targ, f) in enumerate(zip(kwgs['tt_tests'], kwgs['t_tests'], f_targs, fs)):
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

    for idx, (tt, t, V) in enumerate(zip(kwgs['tt_tests'], kwgs['t_tests'], Vs)):
        plt.semilogy(t, V, linewidth=lws[idx])
        lgd.append("Test Seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TestSeqs.png", dpi = 300.)
    plt.close()

# Let's not load the data and calculate f_targs for now
# # Load data
# shit =  torch.load('./data/RandnData1_std_1e-3_0504.pt')

V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                           kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
V_targ_tests, theta_targ_tests, f_targ_tests = cal_f_beta(beta_targ, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                                          kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0.)
kwgs['f_targs'] = f_targs
kwgs['f_targ_tests'] = f_targ_tests

## ------------------------------------ Gradient descent ------------------------------------ 
# Maximum alternative iterations
max_iters = 30

# Store all betas and all Os
All_betas = []
All_Os = []
All_grads = []

# Early stop criteria
early_stop_rounds = 20
best_O = 1.e4
notImprovingRounds = 0

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
        # beta_targ, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0.
        V_thiss, theta_thiss, f_thiss = cal_f_beta(beta_this, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                   kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)

        O_this = 0.
        grad_this = torch.zeros(4)
        for V_this, theta_this, f_this, t_this, f_targ, t_JumpIdx, tt, VV, JumpIdx in zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
            O_this += O(f_this, f_targ, t_this)
            # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
            grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs)

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

            best_O = O_this
            notImprovingRounds = 0
        
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
                V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                              kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
                O_trial = 0.
                
                for V_trial, theta_trial, f_trial, f_targ, t_this in zip(V_trials, theta_trials, f_trials, kwgs['f_targs'], kwgs['ts']):
                    O_trial += O(f_trial, f_targ, t_this)
                # print("beta, O" + str(iter) + ": ", beta_trial, O_trial)
                iter += 1

            beta_this = beta_trial
            V_thiss = V_trials
            theta_thiss = theta_trials
            f_thiss = f_trials
            O_this = O_trial

            # Get new grad
            grad_this = torch.zeros(4)
            for V_this, theta_this, f_this, t_this, f_targ, t_JumpIdx, tt, VV, JumpIdx in zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
                O_this += O(f_this, f_targ, t_this)
                # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
                grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs)
            
            print("=" * 40, " Inner Iteration ", str(i + 1), " ", "=" * 40)
            print("Optimized beta: ", beta_this)
            print("Training data O: ", O_this)
            print("Gradient: ", grad_this, flush=True)
            All_betas.append(beta_this)
            All_Os.append(O_this)
            All_grads.append(grad_this)

            # Set up early stop
            if O_this < best_O:
                best_O = O_this
                notImprovingRounds = 0
            else:
                notImprovingRounds += 1
            if notImprovingRounds > early_stop_rounds:
                break

    if alt_iter % 10 == 0:
        O_test = 0.
        # V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
        #                                                       kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
        V_tests, theta_tests, f_tests = cal_f_beta(beta_this, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                                   kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0.)
        for V_test, theta_test, f_test, f_targ, t_this in zip(V_tests, theta_tests, f_tests, kwgs['f_targ_tests'], kwgs['t_tests']):
            O_test += O(f_test, f_targ, t_this)
        print("-!" * 40)
        print("Testing O: ", O_test), 
        print("-!" * 40)

    # Set up early stop
    if notImprovingRounds > early_stop_rounds:
        print("~" * 40, " Early stop criteria has been met ", "~" * 40)
        break


# Save a figure of the result
# pwd ="./plots/FricSeqGen0323_alternating_DrsFStar/"
pwd = "./plots/Test0516_std_0_AdjMtd_selected_intervals/"
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


## Check numerical derivatives
# beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001
# beta0=torch.tensor([0.0109, 0.0161, 0.2000, 0.5800])

# # Gradient descent
# beta_this = beta0
# V_thiss, theta_thiss, f_thiss = cal_f_beta(beta_this, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0.)

# O_this = 0.
# grad_this = torch.zeros(4)
# for V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx in zip(V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs):
#     O_this += O(f_this, f_targ, t)
#     # beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
#     grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs)

# print("Grad by Adjoint: ", grad_this)

# # Numerical gradients
# inc = 0.001
# numerical_grad0 = torch.zeros(beta0.shape)
# for i in range(len(beta0)):
#     beta_plus = torch.clone(beta0)
#     beta_plus[i] *= (1 + inc)
#     print("beta_plus: ", beta_plus)
#     Vps, thetasp, fps = cal_f_beta(beta_plus, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0.)

#     Op = 0.
#     for f_targ, fp in zip(f_targs, fps):
#         Op += O(fp, f_targ, t)

#     beta_minus = torch.clone(beta0)
#     beta_minus[i] *= (1 - inc)
#     print("beta_minus: ", beta_minus)
#     Vms, thetams, fms = cal_f_beta(beta_minus, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0.)
    
#     Om = 0.
#     for f_targ, fm in zip(f_targs, fms):
#         Om += O(fm, f_targ, t)

#     numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

# print("Grad by finite difference: ", numerical_grad0)