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
from FrictionSeqGen_alternate_withJump_resample_functions import genVVtt, calVtFuncs, \
    cal_f_beta_parallel, cal_f_beta, O_parallel, O, grad_parallel, grad, plotSequences, \
    checkNumericalDerivatives

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
generate_VVtts = False
loadDataFilename = "./data/VVTTs0517.pt"
saveDataFilename = "./data/VVTTs0614.pt"
totalNofSeqs = 4
NofIntervalsRange = [5, 11]
VVRange = [-10, 3]
VVLenRange = [1, 11]

# Determine method of getting data
if generate_VVtts == True:
    VVs, tts = genVVtt(totalNofSeqs, NofIntervalsRange, VVRange, VVLenRange)
    data = {
        "VVs": VVs, 
        "tts": tts, 
    }
    torch.save(data, saveDataFilename)
else:
    shit = torch.load(loadDataFilename)
    VVs = shit['VVs']
    tts = shit['tts']

# Prescribed velocities - testing
VV_tests = torch.tensor([ones + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + tens + tens + ones + ones, \
                         ones + ones + tens + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + ones + ones])
# Times at which the velocities are prescribed - testing
tt_tests = torch.stack([torch.linspace(0., 30., VV_tests.shape[1]),
                        torch.linspace(0., 30., VV_tests.shape[1])])

# Get ts, JumpIdxs, t_JumpIdxs, VtFuncs for train and test data set
ts, JumpIdxs, t_JumpIdxs, VtFuncs = calVtFuncs(VVs, tts)
t_tests, JumpIdx_tests, t_JumpIdx_tests, VtFunc_tests = calVtFuncs(VV_tests, tt_tests)

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
    'DirectComputeFlag' : DirectComputeFlag, 
}

# Let's not load the data and calculate f_targs this time
V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
                                           kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., kwgs['DirectComputeFlag'])
V_targ_tests, theta_targ_tests, f_targ_tests = cal_f_beta(beta_targ, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
                                                          kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., kwgs['DirectComputeFlag'])

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

# ## DEBUG LINE
# print('Train P norms: ', kwgs['f_targ_pnorms'])
# print('Test P norms: ', kwgs['f_targ_testPnorms'])
# print('Test ts: ', kwgs['t_tests'])
# print('f_targ_tests: ', f_targ_tests)

# Save data 
torch.save(kwgs, './data/VVTTs_0615_std0_kwgs.pt')

# Test numerical derivatives
outputFile = "log/testDerivatives0615"
checkNumericalDerivatives(beta0, beta_targ, p, kwgs, nWorkers, parallel_pool, outputFile)

# # Load data
# kwgs = torch.load('./data/VVTTs_0517_std1e-3_kwgs.pt')
# ## ------------------------------------ Gradient descent ------------------------------------ 
# # Maximum alternative iterations
# max_iters = 50

# # Store all betas and all Os
# All_betas = []
# All_Os = []
# All_grads = []

# # Early stop criteria
# early_stop_rounds = 60
# best_O = 1.e4
# notImprovingRounds = 0

# # Start the outer loop of iterations
# beta_this = beta0
# for alt_iter in range(max_iters):
#     # Print out section
#     print("*" * 100)
#     print("*", " "*40, "Outer Iteration ", str(alt_iter), " " * 40, "*")
#     print("*" * 100)

#     ## Inner iteration, get fixed randomly
#     inner_groups = []
#     for idx, NofIters in enumerate(kwgs['beta_unfixed_NofIters']):
#         inner_groups = inner_groups + [kwgs['beta_unfixed_groups'][idx] for i in range(NofIters)]
    
#     # Permute it 
#     shuffle(inner_groups)
#     print("Inner groups: ", inner_groups)
    
#     for (grp_idx, release_idx) in enumerate(inner_groups):
#     # for grp_idx, release_idx in enumerate(kwgs['beta_unfixed_groups']):
#         beta_fixed_this = torch.ones(len(kwgs['beta0']), dtype=torch.bool)
#         beta_fixed_this[release_idx] = False
        
#         # Print out which values are fixed
#         print("~" * 20, "beta_fixed: ", beta_fixed_this, "~"*20)


#         # max_step = torch.tensor([0.005, 0.005, 0.01, 0.1])
#         max_step = torch.tensor([1., 1., 1., 1.])
#         max_step[beta_fixed_this] = 1.e30
#         beta0_this = beta_this
#         # beta_targ, kwgs, ts, t_JumpIdxs, tts, JumpIdxs, VtFuncs, 0.
#         V_thiss, theta_thiss, f_thiss = cal_f_beta(beta_this, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
#                                                    kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., DirectComputeFlag)

#         O_this = 0.
#         grad_this = torch.zeros(4)
#         for V_this, theta_this, f_this, t_this, f_targ, f_targ_pnorm, t_JumpIdx, tt, VV, JumpIdx \
#             in zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
#             O_this += O(f_this, f_targ, t_this, p) / f_targ_pnorm
#             # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
#             grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) / torch.pow(f_targ_pnorm, p)

#         # print("=" * 40, "Inner Iteration ", str(0), " ", "=" * 40)
#         # print("Initial beta: ", beta_this)
#         # print("O: ", O_this)
#         # print("Gradient: ", grad_this, flush=True)
        
#         # Store the first O and grad if both the outside and inside iteration number is 0
#         if alt_iter == 0 and grp_idx == 0:
#             print("=" * 40, "Inner Iteration ", str(0), " ", "=" * 40)
#             print("Initial beta: ", beta_this)
#             print("O: ", O_this)
#             print("Gradient: ", grad_this, flush=True)
#             All_betas.append(beta_this)
#             All_Os.append(O_this)
#             All_grads.append(grad_this)

#             best_O = O_this
#             notImprovingRounds = 0
        
#         for i in range(1):
#         # for i in range(kwgs['beta_unfixed_NofIters'][grp_idx]):
#             max_eta = torch.min(torch.abs(max_step / grad_this))
#             # Line search
#             iter = 0
#             O_trial = O_this
#             while (iter <= 12 and O_trial >= O_this):
#                 beta_trial = beta_this - grad_this * max_eta * pow(2, -iter)
#                 beta_trial[beta_fixed_this] = beta0_this[beta_fixed_this]
#                 beta_trial = torch.clip(beta_trial, kwgs['beta_low'], kwgs['beta_high'])
#                 V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
#                                                               kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0., DirectComputeFlag)
#                 O_trial = 0.
                
#                 for V_trial, theta_trial, f_trial, f_targ, f_targ_pnorm, t_this in \
#                     zip(V_trials, theta_trials, f_trials, kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['ts']):
#                     O_trial += O(f_trial, f_targ, t_this, p) / f_targ_pnorm
#                 # print("beta, O" + str(iter) + ": ", beta_trial, O_trial)
#                 iter += 1

#             beta_this = beta_trial
#             V_thiss = V_trials
#             theta_thiss = theta_trials
#             f_thiss = f_trials
#             O_this = O_trial

#             # Get new grad
#             grad_this = torch.zeros(4)
#             for V_this, theta_this, f_this, t_this, f_targ, f_targ_pnorm, t_JumpIdx, tt, VV, JumpIdx in \
#                 zip(V_thiss, theta_thiss, f_thiss, kwgs['ts'], kwgs['f_targs'], kwgs['f_targ_pnorms'], kwgs['t_JumpIdxs'], kwgs['tts'], kwgs['VVs'], kwgs['JumpIdxs']):
#                 O_this += O(f_this, f_targ, t_this, p) / f_targ_pnorm
#                 # beta_this, t, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs
#                 grad_this += grad(beta_this, t_this, V_this, theta_this, f_this, f_targ, t_JumpIdx, tt, VV, JumpIdx, kwgs, p) / torch.pow(f_targ_pnorm, p)
            
#             print("=" * 40, " Inner Iteration ", str(i + 1), " ", "=" * 40)
#             print("Optimized beta: ", beta_this)
#             print("Training data O: ", O_this)
#             print("Gradient: ", grad_this, flush=True)
#             All_betas.append(beta_this)
#             All_Os.append(O_this)
#             All_grads.append(grad_this)

#             # Set up early stop
#             if O_this < best_O:
#                 best_O = O_this
#                 notImprovingRounds = 0
#             else:
#                 notImprovingRounds += 1
#             if notImprovingRounds > early_stop_rounds:
#                 break

#     if alt_iter % 10 == 0:
#         O_test = 0.
#         # V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
#         #                                                       kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
#         V_tests, theta_tests, f_tests = cal_f_beta(beta_this, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
#                                                    kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)
#         for V_test, theta_test, f_test, f_targ, f_targ_pnorm, t_this in \
#             zip(V_tests, theta_tests, f_tests, kwgs['f_targ_tests'], kwgs['f_targ_testPnorms'], kwgs['t_tests']):
#             O_test += O(f_test, f_targ, t_this, p_test) / f_targ_pnorm
#         print("-!" * 40)
#         print("Testing O (p = 2): ", O_test), 
#         print("-!" * 40)

#     # Set up early stop
#     if notImprovingRounds > early_stop_rounds:
#         print("~" * 40, " Early stop criteria has been met ", "~" * 40)
#         break


# # Save a figure of the result
# pwd = "./plots/Test0609_std_0_AdjMtd_generated_intervals_p6/"
# Path(pwd).mkdir(parents=True, exist_ok=True)

# # Append to the keywords arguments
# kwgs['All_betas'] = All_betas
# kwgs['All_Os'] = All_Os
# kwgs['All_grads'] = All_grads
# torch.save(kwgs, pwd + "kwgs.pt")

# # Print out best performance and plot sequences
# best_idx = All_Os.index(min(All_Os))
# best_grad = All_grads[best_idx]
# best_O = All_Os[best_idx]
# best_beta = All_betas[best_idx]

# # Calculate best test data
# O_test = 0.
# # V_trials, theta_trials, f_trials = cal_f_beta(beta_trial, kwgs, kwgs['ts'], kwgs['t_JumpIdxs'], 
# #                                                       kwgs['tts'], kwgs['JumpIdxs'], kwgs['VtFuncs'], 0.)
# V_tests, theta_tests, f_tests = cal_f_beta(best_beta, kwgs, kwgs['t_tests'], kwgs['t_JumpIdx_tests'], 
#                                             kwgs['tt_tests'], kwgs['JumpIdx_tests'], kwgs['VtFunc_tests'], 0., DirectComputeFlag)
# for V_test, theta_test, f_test, f_targ, f_targ_pnorm, t_this in \
#     zip(V_tests, theta_tests, f_tests, kwgs['f_targ_tests'], kwgs['f_targ_testPnorms'], kwgs['t_tests']):
#     O_test += O(f_test, f_targ, t_this, p_test) / f_targ_pnorm

# # Print results
# print("~" * 40, " Final Optimization Answer ", "~" * 40)
# print("Optimized beta: ", best_beta)
# print("Training O under optimized neta: ", best_O)
# print("Testing O under optimized beta (p = 2): ", O_test), 
# print("Gradient: ", best_grad, flush=True)
# print("VVs: ", VVs)
# print("tts: ", tts)
# plotSequences(best_beta, kwgs, pwd)
