## Import standard librarys
from tkinter import NW
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
from joblib import Parallel, delayed, effective_n_jobs

# Output number of available workers
print("Number of workers available: ", effective_n_jobs(-1))

# Initialize the parallel pool
nWorkers = 16
parallel_pool = Parallel(n_jobs=nWorkers, backend='threading')

# Directly compute the sequences
DirectComputeFlag = True

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
beta0 = torch.tensor([0.005, 0.02, 1. / 5.e1, 0.4])
beta_fixed = torch.tensor([0, 0, 0, 0], dtype=torch.bool)

# Set p values, p works fine between 2 and 12, even
p = 6
# Use same standard for test data
p_test = 2

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

# Generate or load data
generate_data = False
dataFilename = "./data/VVTTs0517.pt"
totalNofSeqs = 4
VVseeds = []
VVseeds_len = []
for i in range(totalNofSeqs):
    NofSds = torch.randint(5, 11, [1])
    VVseed = torch.randint(-10, 3, [NofSds])
    VVseed_len = 10 * torch.randint(1, 11, [NofSds])
    VVseeds.append(VVseed)
    VVseeds_len.append(VVseed_len)

# VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
#                     ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
#                     ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
#                     tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

# VVseeds = torch.randint(-10, 3, [4, 15])
# print("VVseeds: ", VVseeds)
if generate_data == True:
    VVs = []
    tts = []

    # Generate VVs and tts
    for idx, (VVseed, VVseed_len) in enumerate(zip(VVseeds, VVseeds_len)):
        VV = torch.zeros(torch.sum(VVseed_len))
        st = 0
        for j in range(len(VVseed_len)):
            VV[st : st + VVseed_len[j]] = torch.pow(10., VVseed[j])
            st += VVseed_len[j]
        VVs.append(VV)
        tt = torch.linspace(0., 0.2 * len(VV), len(VV))
        tts.append(tt)
    
    data = {
        "VVs" : VVs, 
        "tts" : tts
    }
    torch.save(data, dataFilename)
else:
    shit = torch.load(dataFilename)
    VVs = shit['VVs']
    tts = shit['tts']

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
t_tt_times = [10 for i in range(totalNofSeqs)]
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

# Divide the sequences and distribute to different workers
def cal_f_beta_parallel(beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, 
                        n_workers = nWorkers, pool = parallel_pool):
    # For now, partition the tts such that each chunk only has one sequence
    t_splits = [[t] for t in ts]
    t_JumpIdx_splits = [[t_JumpIdx] for t_JumpIdx in t_JumpIdxs]
    tt_splits = [[tt] for tt in tts]
    JumpIdx_splits = [[JumpIdx] for JumpIdx in JumpIdxs]
    VtFunc_splits = [[VtFunc] for VtFunc in VtFuncs]

    # Get all the sequences
    res = pool(delayed(cal_f_beta)(
                beta, kwgs, t_split, t_JumpIdx_split, tt_split, JumpIdx_split, VtFunc_split, std_noise, directCompute
            ) for t_split, t_JumpIdx_split, tt_split, JumpIdx_split, VtFunc_split in zip(t_splits, t_JumpIdx_splits, tt_splits, JumpIdx_splits, VtFunc_splits)
        )

    # Join the list
    Vs = [res[i][0] for i in range(len(res))]
    thetas = [res[i][1] for i in range(len(res))]
    fs = [res[i][2] for i in range(len(res))]
    
    Vs = [piece for this in Vs for piece in this] 
    thetas = [piece for this in thetas for piece in this] 
    fs = [piece for this in fs for piece in this] 

    # ## Debug lines
    # print("len(Vs): ", len(Vs))
    # print("len(thetas): ", len(thetas))
    # print("len(fs): ", len(fs))

    # print("Vs: ", Vs)
    # print("thetas: ", thetas)
    # print("fs: ", fs)
    # Partition the sets
    return Vs, thetas, fs

# Compute f history based on VtFunc and beta
def cal_f_beta(beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True):
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
            i = 0
            j = len(t_this_interval)
            if (t_this_interval[0] == t_this_interval[1]):
                i = i + 1
            if (t_this_interval[-1] == t_this_interval[-2]):
                j = -1

            if directCompute == True:
                V_thisStep = V[t_JumpIdx[index]]
                if V_thisStep * DRSInv < 1.e-4:
                    alp = V_thisStep * DRSInv
                    deltaT = t_this_interval[i : j] - t_this_interval[i]
                    theta_this = theta0_this + deltaT - alp * (deltaT * theta0_this + torch.square(deltaT) / 2.) \
                                 + alp * alp * theta0_this * torch.square(deltaT) / 2.
                else: 
                    InsideExp = -DRSInv * V_thisStep * (t_this_interval[i : j] - t_this_interval[i])
                    ExpTerm = torch.exp(InsideExp)
                    theta_this = (1 - (1 - DRSInv * V_thisStep * theta0_this) * ExpTerm) / (DRSInv * V_thisStep)
                # # DEBUG LINES
                # print("!"*100)
                # print("theta0_this: ", theta0_this)
                # print("theta_this[0]: ", theta_this[0])
                # print("DRSInv * V_thisStep: ", DRSInv * V_thisStep)
                # print("!"*100)

            else:
                thetaFunc = lambda t, theta: 1. - torch.tensor(vtfunc(torch.clip(t, tt[JumpIdx[index]], tt[JumpIdx[index + 1]])), dtype=torch.float) * theta * DRSInv
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

        # # Inside the cal_f_beta
        # print("="*10, " inside cal_f_beta ", "="*10)
        # print("V: ", V)
        # print("theta: ", theta)
        # print("="*10, " after cal_f_beta ", "="*10)
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

# Objective function, parallel version
def O_parallel(fs, f_targs, ts, p = 2, pool = parallel_pool):
    Os = pool(
        delayed(O)(f, f_targ, t, p) for f, f_targ, t in zip(fs, f_targs, ts)  
    )
    
    # To the power 1/p
    res = torch.sum(torch.stack(Os))
    # # To the power 1
    # res = torch.pow(torch.sum(torch.stack(Os)), 1.) 
    return res

# Add l_p for O
def O(f, f_targ, t, p):
    res = torch.trapezoid(
        torch.pow(f - f_targ, p), t
    ) / (t[-1] - t[0])

    # To the power of one over p
    res = torch.pow(res, 1. / p)
    return res


# Gradient of objective function, parallel
def grad_parallel(beta, ts, Vs, thetas, fs, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs, kwgs, p = 2, pool = parallel_pool):
    grads = pool(
        delayed(grad)(beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) \
            for t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx in zip(ts, Vs, thetas, fs, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs)
    )
    grads = torch.stack(grads)
    # print("grads shape: ", grads.shape)
    # print("grads: ", grads)

    res = torch.sum(grads, dim = 0)
    return res


def grad(beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p):
    integrand = torch.zeros([len(beta), len(t)])
    integrand[0, :] = torch.log(V / 1.e-6)
    integrand[1, :] = torch.log(1.e-6 * theta * beta[2])
    integrand[2, :] = beta[1] / beta[2]
    integrand[3, :] = 1.
    integrand = p * torch.pow(f - f_targ, p - 1) * integrand
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

        dodTheta = interp1d(t_this_interval, p * torch.pow(f_this_interval - f_targ_this_interval, p - 1) * beta[1] / theta_this_interval)
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
                                kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)
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

# Let's not load the data and calculate f_targs this time
V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                           kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., DirectComputeFlag)
V_targ_tests, theta_targ_tests, f_targ_tests = cal_f_beta(beta_targ, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                                          kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)

# Compute p_norm of target fs for trainig and testing
f_targ_pnorms = []
f_targ_testPnorms = []

for t, f_targ in zip(kwgs['ts'], f_targs):
    res = torch.pow(torch.trapz(torch.pow(f_targ, p), t), 1./p)
    f_targ_pnorms.append(res)

for t, f_targ in zip(kwgs['t_tests'], f_targ_tests):
    res = torch.pow(torch.trapz(torch.pow(f_targ, p_test), t), 1./p_test)
    f_targ_testPnorms.append(res)

kwgs['f_targs'] = f_targs
kwgs['f_targ_tests'] = f_targ_tests
kwgs['f_targ_pnorms'] = f_targ_pnorms
kwgs['f_targ_testPnorms'] = f_targ_testPnorms

## DEBUG LINE
print('Train P norms: ', kwgs['f_targ_pnorms'])
print('Test P norms: ', kwgs['f_targ_testPnorms'])
print('Test ts: ', kwgs['t_tests'])
print('f_targ_tests: ', f_targ_tests)
# Save data 
torch.save(kwgs, './data/VVTTs_0601_std0_kwgs.pt')

# # Load data
# kwgs = torch.load('./data/VVTTs_0517_std1e-3_kwgs.pt')
## ------------------------------------ Gradient descent ------------------------------------ 
# Maximum alternative iterations
max_iters = 50

# Store all betas and all Os
All_betas = []
All_Os = []
All_grads = []

# Early stop criteria
early_stop_rounds = 60
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
                                                   kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., DirectComputeFlag)

        O_this = 0.
        grad_this = torch.zeros(4)
        for V_this, theta_this, f_this, t_this, f_targ, f_targ_pnorm, t_JumpIdx, tt, VV, JumpIdx \
            in zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
            O_this += O(f_this, f_targ, t_this, p) / f_targ_pnorm
            # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
            grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) / torch.pow(f_targ_pnorm, p)

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
                                                              kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., DirectComputeFlag)
                O_trial = 0.
                
                for V_trial, theta_trial, f_trial, f_targ, f_targ_pnorm, t_this in \
                    zip(V_trials, theta_trials, f_trials, kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['ts']):
                    O_trial += O(f_trial, f_targ, t_this, p) / f_targ_pnorm
                # print("beta, O" + str(iter) + ": ", beta_trial, O_trial)
                iter += 1

            beta_this = beta_trial
            V_thiss = V_trials
            theta_thiss = theta_trials
            f_thiss = f_trials
            O_this = O_trial

            # Get new grad
            grad_this = torch.zeros(4)
            for V_this, theta_this, f_this, t_this, f_targ, f_targ_pnorm, t_JumpIdx, tt, VV, JumpIdx in \
                zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
                O_this += O(f_this, f_targ, t_this, p) / f_targ_pnorm
                # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
                grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) / torch.pow(f_targ_pnorm, p)
            
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
                                                   kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)
        for V_test, theta_test, f_test, f_targ, f_targ_pnorm, t_this in \
            zip(V_tests, theta_tests, f_tests, kwgs['f_targ_tests'], kwgs['f_targ_testPnorms'], kwgs['t_tests']):
            O_test += O(f_test, f_targ, t_this, p_test) / f_targ_pnorm
        print("-!" * 40)
        print("Testing O (p = 2): ", O_test), 
        print("-!" * 40)

    # Set up early stop
    if notImprovingRounds > early_stop_rounds:
        print("~" * 40, " Early stop criteria has been met ", "~" * 40)
        break


# Save a figure of the result
pwd = "./plots/Test0609_std_0_AdjMtd_generated_intervals_p6/"
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

# Calculate best test data
O_test = 0.
# V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
#                                                       kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
V_tests, theta_tests, f_tests = cal_f_beta(best_beta, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                            kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)
for V_test, theta_test, f_test, f_targ, f_targ_pnorm, t_this in \
    zip(V_tests, theta_tests, f_tests, kwgs['f_targ_tests'], kwgs['f_targ_testPnorms'], kwgs['t_tests']):
    O_test += O(f_test, f_targ, t_this, p_test) / f_targ_pnorm

# Print results
print("~" * 40, " Final Optimization Answer ", "~" * 40)
print("Optimized beta: ", best_beta)
print("Training O under optimized neta: ", best_O)
print("Testing O under optimized beta (p = 2): ", O_test), 
print("Gradient: ", best_grad, flush=True)
print("VVs: ", VVs)
print("tts: ", tts)
plotSequences(best_beta, kwgs, pwd)


# # Check numerical derivatives
# # beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001
# DirectComputeFlag = True

# # Set p values, p works fine between 2 and 12, even
# p = 12
# beta0=torch.tensor([0.009, 0.012, 0.2000, 0.3800])

# # Time start
# st = time.time()

# # Gradient descent
# beta_this = beta0
# # beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, n_workers = nWorkers, pool = parallel_pool
# V_thiss, theta_thiss, f_thiss = cal_f_beta_parallel(beta_this, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0., DirectComputeFlag, nWorkers, parallel_pool)
# V_targs, theta_targs, f_targs = cal_f_beta_parallel(beta_targ, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0., DirectComputeFlag, nWorkers, parallel_pool)

# O_this = 0.
# grad_this = torch.zeros(4)
# O_this = O_parallel(f_thiss, f_targs, ts, p, parallel_pool)
# grad_this = grad_parallel(beta_this, ts, V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs, kwgs, p, parallel_pool)
# # for V_this, theta_this, f_this, f_targ, t_JumpIdx, t, tt, VV, JumpIdx in zip(V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, ts, tts, VVs, JumpIdxs):
# #     # # Debug lines
# #     # print("f_this shape: ", f_this.shape)
# #     # print("f_targ shape: ", f_targ.shape)
# #     # print("t shape: ", t.shape)
# #     # print("f_this: ", f_this)
# #     # print("theta_this: ", theta_this)
# #     # print("V_this: ", V_this)

# #     O_this += O(f_this, f_targ, t)
# #     # beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
# #     grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs)

# print("Grad by Adjoint: ", grad_this)

# # Numerical gradients
# inc = 0.001
# numerical_grad0 = torch.zeros(beta0.shape)
# for i in range(len(beta0)):
#     beta_plus = torch.clone(beta0)
#     beta_plus[i] *= (1 + inc)
#     print("beta_plus: ", beta_plus)
#     Vps, thetasp, fps = cal_f_beta_parallel(beta_plus, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0., DirectComputeFlag, nWorkers, parallel_pool)

#     Op = O_parallel(fps, f_targs, ts, p, parallel_pool)
#     # Op = 0.
#     # for f_targ, fp, t in zip(f_targs, fps, ts):
#     #     Op += O(fp, f_targ, t)

#     beta_minus = torch.clone(beta0)
#     beta_minus[i] *= (1 - inc)
#     print("beta_minus: ", beta_minus)
#     Vms, thetams, fms = cal_f_beta_parallel(beta_minus, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0., DirectComputeFlag, nWorkers, parallel_pool)
    
#     Om = O_parallel(fms, f_targs, ts, p, parallel_pool)
#     # Om = 0.
#     # for f_targ, fm , t in zip(f_targs, fms, ts):
#     #     Om += O(fm, f_targ, t)

#     numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

# print("Grad by finite difference: ", numerical_grad0)

# # grad_par_this = grad_parallel(beta_this, ts, V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs, kwgs, pool = parallel_pool)
# # print("Gradient by parallel function: ", grad_par_this)

# # End the timer
# timeCost = time.time() - st
# print("Time costed: ", timeCost)