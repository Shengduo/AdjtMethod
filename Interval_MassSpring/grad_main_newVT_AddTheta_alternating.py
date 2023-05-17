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
from GradientDescent import GradDescent, objGradFunc
from plotSequences import plot_differences
from GenerateVT import GenerateVT

torch.set_default_dtype(torch.float)

# Gradient descent on fixed $\alpha = [k, m, g]$ and $V$ 
# Set up the parameters
plotsName = "LinearGen"

# Generate VT series
VT_Vrange = torch.tensor([5., 15.])
# VT_NofTpts = 2000
# VT_flag = "simple"
# VT_flag = "prescribed_simple"
VT_flag = "prescribed_linear"
VT_nOfTerms = 5
VT_nOfFourierTerms = 100
res_path = "./plots/0420ADRSfStar_f1_aging_AddFricVTs_Normed_data2_unAlternating/"
Path(res_path).mkdir(parents=True, exist_ok=True)
gen_plt_save_path = res_path + plotsName + ".png"

# # For prescribed VT
VT_NofTpts = 1500
# VT_VVs = torch.tensor([[1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                        [1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.]])

# VT_VVs = torch.tensor([[1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                        [1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1, 1.e1]])

# # Data 3
# alphas = torch.tensor([[100., 5., 9.8], 
#                        [100., 5., 9.8], 
#                        [100., 5., 9.8], 
#                        [100., 5., 9.8]])

# VT_VVs = torch.tensor([[1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                        [1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.], 
#                        [1., 1., 1., 1., 1., 1. ,1., 1., 1., 1., 1., 1., 1., 1., 1.], 
#                        [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.]])

# Data 4
# alphas = torch.tensor([[100., 5., 9.8], 
#                        [100, 5., 9.8], 
#                        [1.e5, 5., 9.8], 
#                        [1.e5, 5., 9.8]])

# Multi data2
alphas = torch.tensor([[100., 5., 9.8], 
                       [100., 5., 9.8], 
                       [100., 5., 9.8], 
                       [100., 5., 9.8]])

# VT_VVs = torch.tensor([[1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                        [1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.], 
#                        [1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.], 
#                        [1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.]])

# VT_tts = torch.stack([torch.linspace(0., 30., 15), 
#                       torch.linspace(0., 30., 15), 
#                       torch.linspace(0., 30., 15), 
#                       torch.linspace(0., 30., 15)])

# # Multi data2
ones = 10 * [1.]
tens = 10 * [10.]
VT_VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
                       ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
                       ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
                       tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

VT_tts = torch.stack([torch.linspace(0., 30., VT_VVs.shape[1]),
                      torch.linspace(0., 30., VT_VVs.shape[1]),
                      torch.linspace(0., 30., VT_VVs.shape[1]),
                      torch.linspace(0., 30., VT_VVs.shape[1])])

VT_Trange = torch.tensor([0., 30.])
VT_Trange = torch.tensor([0., 30.])

# VT_VVs = torch.tensor([[1., 1., 1., 1., 1., 1. ,1., 10., 10., 10., 10., 10., 10., 10., 10.]])
# VT_tts = torch.linspace(0., 30., 15).reshape([1, -1])

# Initialize VT_kwgs
VT_kwgs = {
    "nOfTerms" : VT_nOfTerms, 
    "nOfFourierTerms" : VT_nOfFourierTerms,
    "Trange" : VT_Trange, 
    "Vrange" : VT_Vrange, 
    "flag" : VT_flag, 
    "NofTpts" : VT_NofTpts, 
    "VVs" : VT_VVs, 
    "tts" : VT_tts, 
    "plt_save_path" : gen_plt_save_path, 
}

# Get the series
VT_instance = GenerateVT(VT_kwgs)
print("Shit!")

VTs = VT_instance.VT

# Plot VT (optional)
VT_instance.plotVT()

# Alpha range
alp_low = torch.tensor([50., 0.5, 1., 9.])
alp_hi = torch.tensor([100., 2., 10., 10.])
y0 = torch.tensor([0., 1.0, 1.0])

# Start beta
beta0 = torch.tensor([0.008, 0.016, 1. / 2.e1, 0.3])

# # Different start beta, closer to target
# beta0 = torch.tensor([0.010, 0.017, 2. / 1.e1, 0.6])

# Target beta
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# Beta ranges
# beta_low = torch.tensor([0.001, 0.006, 1. / 5., 0.3])
# beta_low = torch.tensor([-1., -1., 1. / 1.e3, 0.3])

# beta_high = torch.tensor([1., 1., 1. / 1.e-1, 0.8])

# Different start beta, closer to target
beta_low = torch.tensor([0.001, 0.001, 0.001, 0.1])
beta_high = torch.tensor([1., 1., 1.e6, 0.9])

beta_fixed = torch.tensor([0, 0, 0, 0], dtype=torch.bool)

# Document the unfixed groups
# beta_unfixed_groups = [[0], [1], [2], [3]]
# beta_unfixed_NofIters = torch.tensor([1, 1, 1, 1])
beta_unfixed_groups = [[0, 1, 2, 3]]
beta_unfixed_NofIters = torch.tensor([1])

scaling = torch.tensor([1., 1., 1., 1.])

# Other arguments for optAlpha function
max_iters = 100
maxFuncCalls = 200
regularizedFlag = True
noLocalSearch = True
stepping = 'lsrh'
# stepping = 'BB'
lsrh_steps = 20

# Sequence specific parameters
# T = VT_Trange[1]
NofTPts = VT_NofTpts

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

# Store the keywords for optAlpha
kwgs = {
    'y0' : y0, 
    'alphas' : alphas, 
    'VTs' : VTs,
    'alp_low' : alp_low, 
    'alp_high' : alp_hi, 
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
    'alter_grad_flag' : alter_grad_flag, 
    'beta_fixed' : beta_fixed,
    'beta_unfixed_groups' : beta_unfixed_groups, 
    'beta_unfixed_NofIters' : beta_unfixed_NofIters, 
}


# Function to get target v
def generate_target_v(alphas, VTs, beta, y0, this_rtol, this_atol, regularizedFlag, solver, lawFlag):
    ts = []
    ys = []
    MFParams_targs = []
    for idx, (alpha, VT) in enumerate(zip(alphas, VTs)):
        targ_SpringSlider = MassFricParams(alpha, VT, beta, y0, lawFlag, regularizedFlag)
        # targ_SpringSlider.print_info()
        targ_seq = TimeSequenceGen(VT[1, -1], NofTPts, targ_SpringSlider, 
                                   rtol=this_rtol, atol=this_atol, regularizedFlag=regularizedFlag, solver=solver)
        
        ts.append(targ_seq.t)
        ys.append(targ_seq.default_y)
        MFParams_targs.append(targ_SpringSlider)

    # v = targ_seq.default_y[1, :]
    # t = targ_seq.t
    return torch.stack(ts), torch.stack(ys), MFParams_targs


## Number of total alpha-beta iterations
N_AllIters = 1
this_alphas = alphas
this_beta = beta0

## Run alpha-beta iterations
for i in range(N_AllIters):
    # Print out info
    print("#" * 40, " Total Iteration {0} ".format(i) + "#" * 40)
    
    ## First optimize alpha
    kwgs['alphas'] = this_alphas
    kwgs['beta_this'] = this_beta
    
    # Timing alpha
    # Update this Alpha
    # this_alpha = optAlpha(O_GAN, kwgs)
    
    
    ## Run grad descent on beta
    # Generate target v
    ts, vs, MFParams_targs = generate_target_v(this_alphas, kwgs['VTs'], kwgs['beta_targ'], kwgs['y0'], 
                                               kwgs['this_rtol'], kwgs['this_atol'], kwgs['regularizedFlag'], 
                                               kwgs['solver'], kwgs['lawFlag'])

    # Run gradient descent
    myGradBB = GradDescent(kwgs, this_alphas, kwgs['alp_low'], kwgs['alp_high'], kwgs['VTs'], 
                           this_beta, kwgs['beta_low'], kwgs['beta_high'], 
                           kwgs['y0'], vs, ts, MFParams_targs, 
                           objGrad_func = objGradFunc, scaling = kwgs['scaling'], 
                           max_steps = kwgs['max_iters'], stepping = kwgs['stepping'], obs_rtol = 1e-5, lsrh_steps = kwgs['lsrh_steps'], 
                           regularizedFlag = kwgs['regularizedFlag'], 
                           NofTPts = kwgs['NofTPts'], this_rtol = kwgs['this_rtol'], this_atol = kwgs['this_atol'], 
                           solver = kwgs['solver'], lawFlag = kwgs['lawFlag'], alter_grad_flag = kwgs['alter_grad_flag'])
    
    myGradBB.run()
    
    # Update parameters
    this_beta = myGradBB.beta_optimal
    print("Optimal beta: ", this_beta)

# Plot sequences
print("[k, m, g]: ", alphas)
print("VV: ", VT_VVs)
print("tt: ", VT_tts)
print("beta_targ: ", beta_targ)
print("beta0: ", beta0)
print("this_beta: ", this_beta)
print("stepping: ", stepping)
print("solver: ", solver)
print("lawFlag: ", lawFlag)
print("alter_grad_flag", alter_grad_flag)
betas = [beta_targ, beta0, this_beta]
betas_legend = ["True", "Init", "Finl"]
plot_differences(kwgs, betas, betas_legend, res_path)