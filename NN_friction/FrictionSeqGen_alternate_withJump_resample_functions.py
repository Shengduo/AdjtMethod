# This script re-sample training sequences after every training epoch, and use the worst training samples for next epoch
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
from joblib import Parallel, delayed, effective_n_jobs

# Function that generates VVs and tts
def genVVtt(totalNofSeqs, NofIntervalsRange, VVRange, VVLenRange):
    VVseeds = []
    VVseeds_len = []

    # Generate the seeds of VVs and tts
    for i in range(totalNofSeqs):
        NofSds = torch.randint(NofIntervalsRange[0], NofIntervalsRange[1], [1])
        VVseed = torch.randint(VVRange[0], VVRange[1], [NofSds])
        VVseed_len = 10 * torch.randint(VVLenRange[0], VVLenRange[1], [NofSds])
        VVseeds.append(VVseed)
        VVseeds_len.append(VVseed_len)


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
    
    # data = {
    #     "VVs" : VVs, 
    #     "tts" : tts
    # }
    # torch.save(data, dataFilename)
    
    return VVs, tts

# Function to get ts, JumpIdxs, t_JumpIdxs, VtFuncs
def calVtFuncs(VVs, tts):
    # get JumpIdxs
    JumpIdxs = []
    for VV in VVs:
        JumpIdx = [0]
        for i in range(1, len(VV)):
            if VV[i] != VV[i - 1]:
                JumpIdx.append(i)
        JumpIdx.append(len(VV) - 1)
        JumpIdxs.append(JumpIdx)

    # Get VtFuncs, ts, t_JumpIdxs
    VtFuncs = []
    ts = []
    t_JumpIdxs = []

    # Functions, ts and t_JumpIdxs
    t_tt_times = [10 for i in range(len(VVs))]
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
    return ts, JumpIdxs, t_JumpIdxs, VtFuncs

# Divide the sequences and distribute to different workers
def cal_f_beta_parallel(beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, 
                        n_workers = 16, pool = Parallel(n_jobs=16, backend='threading')):
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
def O_parallel(fs, f_targs, ts, p = 2, pool = Parallel(n_jobs=16, backend='threading')):
    Os = pool(
        delayed(O)(f, f_targ, t, p) for f, f_targ, t in zip(fs, f_targs, ts)  
    )
    
    # To the power 1/p
    res = torch.sum(torch.stack(Os))

    return Os, res

# Add l_p for O
def O(f, f_targ, t, p):
    res = torch.trapezoid(
        torch.pow(f - f_targ, p), t
    ) / (t[-1] - t[0])

    # To the power of one over p
    res = torch.pow(res, 1. / p)
    return res

# Generate and then find 8 training sequences
def findVtFuncs(beta_this, beta_targ, kwgs, VVs_prev = [], tts_prev = [], n_Workers=16, parallel_pool=Parallel(n_jobs=16, backend='threading')):
    # Generate N of VVs and tts
    VVs, tts = genVVtt(kwgs['totalNofSeqs'], kwgs['NofIntervalsRange'], kwgs['VVRange'], kwgs['VVLenRange'])

    # Concat the previous VVs
    VVs = VVs + VVs_prev
    tts = tts + tts_prev

    # Calculate VtFuncs based on VVs and tts
    ts, JumpIdxs, t_JumpIdxs, VtFuncs = calVtFuncs(VVs, tts)

    # Grab the Top n seqs
    resIdx, resOs = findTopNSeqs(beta_this, beta_targ, kwgs['selectedNofSeqs'], kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, n_Workers, parallel_pool)

    VVs_res = [VVs[i] for i in resIdx]
    tts_res = [tts[i] for i in resIdx]
    ts_res = [ts[i] for i in resIdx]
    JumpIdxs_res = [JumpIdxs[i] for i in resIdx]
    t_JumpIdxs_res = [t_JumpIdxs[i] for i in resIdx]
    VtFuncs_res = [VtFuncs[i] for i in resIdx]

    # Return all resulting res
    return VVs_res, tts_res, ts_res, JumpIdxs_res, t_JumpIdxs_res, VtFuncs_res

# Find n sequences with the highest O values
def findTopNSeqs(beta_this, beta_targ, n, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, n_Workers, parallel_pool):
    # beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, n_workers = nWorkers, pool = parallel_pool
    V_thiss, theta_thiss, f_thiss = cal_f_beta_parallel(beta_this, kwgs, ts, t_JumpIdxs, 
                                                        tts, JumpIdxs, VtFuncs, 
                                                        0., True, n_Workers, parallel_pool)
    
    V_targs, theta_targs, f_targs = cal_f_beta_parallel(beta_targ, kwgs, ts, t_JumpIdxs, 
                                                        tts, JumpIdxs, VtFuncs, 
                                                        0., True, n_Workers, parallel_pool)
    
    Os, O = O_parallel(f_thiss, f_targs, ts, kwgs['p'], parallel_pool)
    Os = torch.tensor(Os)
    
    # Sort
    sorted, indices = torch.sort(Os, descending=True)
    resIdx = indices[0:n]
    resOs = sorted[0:n]

    # # DEBUG LINE
    # print('Os: ', Os)
    # print('First n indices: ', resIdx)
    # print('First n values: ', resOs)
    return resIdx, resOs



# Gradient of objective function, for all sequences
def grad_parallel(beta, ts, Vs, thetas, fs, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs, kwgs, p = 2, pool = Parallel(n_jobs=16, backend='threading')):
    grads = pool(
        delayed(grad)(beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) \
            for t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx in zip(ts, Vs, thetas, fs, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs)
    )
    grads = torch.stack(grads)
    # print("grads shape: ", grads.shape)
    # print("grads: ", grads)

    res = torch.sum(grads, dim = 0)
    return res

# Gradient of objective function, for one sequence
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
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['t_origs'], kwgs['t_JumpIdx_origs'], 
                                kwgs['tt_origs'], kwgs['JumpIdx_origs'], kwgs['VtFunc_origs'], 0.)
    f_targs = kwgs['f_targ_origs']
    lws = torch.linspace(3.0, 1.0, len(Vs))
    for idx, (tt, t, f_targ, f) in enumerate(zip(kwgs['tt_origs'], kwgs['t_origs'], f_targs, fs)):
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f, linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        plt.title("Orignal train sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "OrigTrainSeq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, (tt, t, V) in enumerate(zip(kwgs['tt_origs'], kwgs['t_origs'], Vs)):
        plt.semilogy(t, V, linewidth=lws[idx])
        lgd.append("Original train Seq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "OrigTrainSeqs.png", dpi = 300.)
    plt.close()

    # -------------------- For test data --------------------------
    Vs, thetas, fs = cal_f_beta(beta, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., kwgs['DirectComputeFlag'])
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

# Check numerical derivatives by finite difference method
def checkNumericalDerivatives(beta0, beta_targ, p, kwgs, nWorkers, parallel_pool, outputFile):
    # Check numerical derivatives
    # beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001

    # Time start
    st = time.time()

    # Gradient descent
    beta_this = beta0
    # beta, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, std_noise = 0.001, directCompute = True, n_workers = nWorkers, pool = parallel_pool
    V_thiss, theta_thiss, f_thiss = cal_f_beta_parallel(beta_this, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                        kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 
                                                        0., kwgs['DirectComputeFlag'], nWorkers, parallel_pool)
    
    V_targs, theta_targs, f_targs = cal_f_beta_parallel(beta_targ, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                        kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 
                                                        0., kwgs['DirectComputeFlag'], nWorkers, parallel_pool)
    O_this = 0.
    grad_this = torch.zeros(4)
    O_thiss, O_this = O_parallel(f_thiss, f_targs, kwgs['ts'], p, parallel_pool)
    grad_this = grad_parallel(beta_this, kwgs['ts'], V_thiss, theta_thiss, f_thiss, f_targs, 
                              kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs'], 
                              kwgs, p, parallel_pool)
    # for V_this, theta_this, f_this, f_targ, t_JumpIdx, t, tt, VV, JumpIdx in zip(V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, ts, tts, VVs, JumpIdxs):
    #     # # Debug lines
    #     # print("f_this shape: ", f_this.shape)
    #     # print("f_targ shape: ", f_targ.shape)
    #     # print("t shape: ", t.shape)
    #     # print("f_this: ", f_this)
    #     # print("theta_this: ", theta_this)
    #     # print("V_this: ", V_this)

    #     O_this += O(f_this, f_targ, t)
    #     # beta, t, V, theta, f, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
    #     grad_this += grad(beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs)
    with open(outputFile, 'a') as myFile:
        print("Grad by Adjoint: ", grad_this, file=myFile)

        # Numerical gradients
        inc = 0.001
        numerical_grad0 = torch.zeros(beta0.shape)
        for i in range(len(beta0)):
            beta_plus = torch.clone(beta0)
            beta_plus[i] *= (1 + inc)
            # print("beta_plus: ", beta_plus, file=myFile)
            Vps, thetasp, fps = cal_f_beta_parallel(beta_plus, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                    kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 
                                                    0., kwgs['DirectComputeFlag'], nWorkers, parallel_pool)

            Ops, Op = O_parallel(fps, f_targs, kwgs['ts'], p, parallel_pool)
            # Op = 0.
            # for f_targ, fp, t in zip(f_targs, fps, ts):
            #     Op += O(fp, f_targ, t)

            beta_minus = torch.clone(beta0)
            beta_minus[i] *= (1 - inc)
            # print("beta_minus: ", beta_minus, file=myFile)
            Vms, thetams, fms = cal_f_beta_parallel(beta_minus, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                                    kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 
                                                    0., kwgs['DirectComputeFlag'], nWorkers, parallel_pool)
            
            Oms, Om = O_parallel(fms, f_targs, kwgs['ts'], p, parallel_pool)
            # Om = 0.
            # for f_targ, fm , t in zip(f_targs, fms, ts):
            #     Om += O(fm, f_targ, t)

            numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

        print("Grad by finite difference: ", numerical_grad0, file=myFile)

        # grad_par_this = grad_parallel(beta_this, ts, V_thiss, theta_thiss, f_thiss, f_targs, t_JumpIdxs, tts, VVs, JumpIdxs, kwgs, pool = parallel_pool)
        # print("Gradient by parallel function: ", grad_par_this)

        # End the timer
        timeCost = time.time() - st
        print("Time costed: ", timeCost, file=myFile)