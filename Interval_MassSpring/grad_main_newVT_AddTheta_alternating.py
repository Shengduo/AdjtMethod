"""
Use class GenerateVT to generate imposed sliprate-time history
"""
## Import standard libraries
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

## Import local classes and functions
from MassFricParams import MassFricParams
from TimeSequenceGen import TimeSequenceGen
from AdjointMethod import AdjDerivs
from GradientDescent import GradDescent, objGradFunc, empiricalGrad, objFunc_parallel, \
                            get_yts_parallel, get_yt, genVVtt, SampleAndFindTopNSeqs
from plotSequences import plot_differences
from joblib import Parallel, delayed, effective_n_jobs
# from GenerateVT import GenerateVT

torch.set_default_dtype(torch.float)

# Gradient descent on fixed $\alpha = [k, m, g]$ and $V$ 
# Set up the parameters
plotsName = "GenSeqs"
res_path = "./plots/0713ADRSfStar_f1_aging_AddFricVTs_Normed_data2_unAlternating/"
Path(res_path).mkdir(parents=True, exist_ok=True)
gen_plt_save_path = res_path + plotsName + ".png"

# Multi data2
alpha = torch.tensor([500., 5., 9.8])

# # Multi data2
ones = 10 * [1.e-2]
tens = 10 * [100.]

# Generate or load data
generate_VVtts = False
loadDataFilename = "./data/VVTTs0713.pt"
saveDataFilename = "./data/VVTTs0713.pt"
totalNofSeqs = 256
selectedNofSeqs = 4
NofIntervalsRange = [5, 11]
VVRange = [-2, 2]
VVLenRange = [1, 11]

# Determine method of getting data
if generate_VVtts == True:
    VVs, tts = genVVtt(4, NofIntervalsRange, VVRange, VVLenRange)
    data = {
        "VVs": VVs, 
        "tts": tts, 
    }
    torch.save(data, saveDataFilename)
else:
    shit = torch.load(loadDataFilename)
    VVs = shit['VVs']
    tts = shit['tts']

# # For prescribed VT
NofTPts = 10

# Initial condition
y0 = torch.tensor([0., 1.0, 1.0])

# Start beta
beta0 = torch.tensor([0.008, 0.012, 1. / 2.e1, 0.3])

# # Different start beta, closer to target
# beta0 = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# Target beta
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])
# beta_targ = torch.tensor([0.008, 0.012, 1. / 2.e1, 0.3])

# Beta ranges
beta_low = torch.tensor([0.001, 0.001, 0.001, 0.1])
beta_high = torch.tensor([1., 1., 1.e2, 0.9])

beta_fixed = torch.tensor([0, 0, 0, 0], dtype=torch.bool)

# Document the unfixed groups
beta_unfixed_groups = [[0], [1], [2], [3]]
beta_unfixed_NofIters = torch.tensor([1, 1, 1, 1])
# beta_unfixed_groups = [[0, 1, 2, 3]]
# beta_unfixed_NofIters = torch.tensor([1])

scaling = torch.tensor([1., 1., 1., 1.])

# Other arguments for optAlpha function
max_iters = 100
maxFuncCalls = 200
regularizedFlag = True
noLocalSearch = True
stepping = 'lsrh'
# stepping = 'BB'
lsrh_steps = 12

# Sequence specific parameters
# T = VT_Trange[1]

# Tolerance parameters
this_rtol = 1.e-6
this_atol = 1.e-8

# Solver
solver = 'rk4'
# solver = 'dopri5'

# LawFlag
# lawFlag = "slip"
lawFlag = "aging"

# Alternating gradient descent, default False
alter_grad_flag = True

# Lp norm for the error
p = 6

# Store the keywords for optAlpha
kwgs = {
    'totalNofSeqs' : totalNofSeqs, 
    'selectedNofSeqs' : selectedNofSeqs, 
    'NofIntervalsRange' : NofIntervalsRange, 
    'VVRange' : VVRange, 
    'VVLenRange' : VVLenRange, 
    'y0' : y0, 
    'alpha' : alpha, 
    "VV_origs" : VVs, 
    "tt_origs" : tts, 
    'NofTPts' : NofTPts, 
    'max_iters' : max_iters, 
    'beta_this' : beta0, 
    'beta_targ' : beta_targ, 
    'beta_low' : beta_low, 
    'beta_high' : beta_high, 
    'scaling' : scaling, 
    'regularizedFlag' : regularizedFlag, 
    'maxFuncCalls' : maxFuncCalls, 
    'noLocalSearch' : noLocalSearch, 
    'stepping' : stepping, 
    'lsrh_steps' : lsrh_steps, 
    'NofTPts' : NofTPts, 
    'this_rtol': this_rtol, 
    'this_atol' : this_atol, 
    'solver' : solver, 
    'lawFlag' : lawFlag, 
    'p' : p, 
    'alter_grad_flag' : alter_grad_flag, 
    'beta_fixed' : beta_fixed,
    'beta_unfixed_groups' : beta_unfixed_groups, 
    'beta_unfixed_NofIters' : beta_unfixed_NofIters, 
}

# Initialize the parallel pool
n_Workers = 1
parallel_pool = Parallel(n_jobs=n_Workers, backend='threading')

# Test out adjoint and empirical gradients
# Generate target v
y_orig_targs, t_orig_targs, MFParams_orig_targs = get_yts_parallel(kwgs, 
                                                                   kwgs['alpha'], 
                                                                   kwgs['VV_origs'], 
                                                                   kwgs['tt_origs'], 
                                                                   kwgs['beta_targ'], 
                                                                   kwgs['y0'], 
                                                                   n_Workers, 
                                                                   parallel_pool)
# Store target MFParams
kwgs['MFParams_targ'] = MFParams_orig_targs[0]

obj = objGradFunc(kwgs, 
                  kwgs['alpha'], 
                  kwgs['VV_origs'], 
                  kwgs['tt_origs'], 
                  beta0, 
                  kwgs['y0'], 
                  y_orig_targs, 
                  MFParams_orig_targs, 
                  objOnly = True)

# empirical_grad = empiricalGrad(kwgs, 
#                                kwgs['alpha'], 
#                                kwgs['VV_origs'], 
#                                kwgs['tt_origs'], 
#                                beta0, 
#                                kwgs['y0'], 
#                                y_orig_targs, 
#                                MFParams_orig_targs, 
#                                proportion = 0.01)

# print("-$" * 20, " Gradient test ", "-$" * 20)
# print("Adjoint gradient: ", grad)
# print("Finite difference gradient: ", empirical_grad)
# print("-$" * 20, "               ", "-$" * 20)

## Number of total alpha-beta iterations
N_AllIters = 1
# this_alpha = alpha
this_beta = beta0.clone()
VVs_prev = VVs
tts_prev = tts

## Run alpha-beta iterations
obj_origs = [obj]
optimal_betas = [this_beta]
MFParams_targs = [kwgs['MFParams_targ'] for i in range(kwgs['selectedNofSeqs'])]

for i in range(N_AllIters):
    # Print out info
    print("#" * 40, " Total Iteration {0} ".format(i) + "#" * 40)
    this_VVs, this_tts, this_targ_ys = SampleAndFindTopNSeqs(kwgs, 
                                                             kwgs['totalNofSeqs'], 
                                                             kwgs['selectedNofSeqs'], 
                                                             VVs_prev, 
                                                             tts_prev, 
                                                             this_beta, 
                                                             kwgs['y0'], 
                                                             kwgs['MFParams_targ'], 
                                                             n_Workers, 
                                                             parallel_pool)
    ## First optimize alpha
    # kwgs['alpha'] = this_alpha
    kwgs['beta_this'] = this_beta
    kwgs['VVs'] = this_VVs
    kwgs['tts'] = this_tts

    # # Run grad descent on beta
    # Generate target v
    # ts, ys, MFParams_targs = generate_target_v(kwgs, this_VVs, this_tts, beta0)
                #  kwgs, 
                #  alpha,
                #  VVs, tts, 
                #  beta0, beta_low, beta_high, 
                #  y0, targ_ys, MFParams_targs, 
                #  objGrad_func, max_steps, scaling = torch.tensor([1., 1., 1., 1.]), 
                #  stepping = 'BB', obs_rtol = 1e-5, grad_atol = 1.e-10, lsrh_steps = 10, 
                #  regularizedFlag = False, 
                #  NofTPts = 1000, this_rtol = 1.e-6, this_atol = 1.e-8, 
                #  solver = 'dopri5', lawFlag = "aging", alter_grad_flag = False
    # Run gradient descent
    myGradBB = GradDescent(kwgs, kwgs['alpha'], this_VVs, this_tts,
                           this_beta, kwgs['beta_low'], kwgs['beta_high'], 
                           kwgs['y0'], this_targ_ys, MFParams_targs, 
                           objGrad_func = objGradFunc, scaling = kwgs['scaling'], 
                           max_steps = kwgs['max_iters'], stepping = kwgs['stepping'], obs_rtol = 1e-5, lsrh_steps = kwgs['lsrh_steps'], 
                           regularizedFlag = kwgs['regularizedFlag'], 
                           NofTPts = kwgs['NofTPts'], this_rtol = kwgs['this_rtol'], this_atol = kwgs['this_atol'], 
                           solver = kwgs['solver'], lawFlag = kwgs['lawFlag'], alter_grad_flag = kwgs['alter_grad_flag'])
    
    myGradBB.run()
    
    # Update parameters
    this_beta = myGradBB.beta_optimal
    optimal_betas.append(this_beta)
    print("Optimal beta: ", this_beta)

    VVs_prev = this_VVs
    tts_prev = this_tts
    this_obj_orig = objGradFunc(kwgs, 
                                kwgs['alpha'], 
                                kwgs['VV_origs'], 
                                kwgs['tt_origs'], 
                                this_beta, 
                                kwgs['y0'], 
                                y_orig_targs, 
                                MFParams_orig_targs, 
                                objOnly = True)
    obj_origs.append(this_obj_orig)
    optimal_betas.append(this_beta)

# Find optimal obj
sorted, indices = torch.sort(obj_origs)
final_optimal_beta = optimal_betas[indices[0]]

# Plot sequences
print("[k, m, g]: ", alpha)
print("beta_targ: ", beta_targ)
print("beta0: ", beta0)
print("final_optimal_beta: ", final_optimal_beta)
print("stepping: ", stepping)
print("solver: ", solver)
print("lawFlag: ", lawFlag)
print("alter_grad_flag", alter_grad_flag)
betas = [beta_targ, beta0, final_optimal_beta]
betas_legend = ["True", "Init", "Finl"]
# betas = [beta0, beta_targ]
# betas_legend = ["Init", "True"]
plot_differences(kwgs, betas, betas_legend, res_path)