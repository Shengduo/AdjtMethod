## Import standard librarys
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import scipy.optimize as opt
import numpy as np

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

torch.set_default_dtype(torch.float)

# Gradient descent on fixed $\alpha = [k, m, g]$ and $V$ 
# Set up the parameters
alpha0 = torch.tensor([50., 1., 9.8])
VT = torch.tensor([[1., 1.], [0., 5.]])

# Alpha range
alp_low = torch.tensor([50., 0.5, 1., 9.])
alp_hi = torch.tensor([100., 2., 10., 10.])
y0 = torch.tensor([0., 1.0, 1.0])

# Start beta
beta0 = torch.tensor([0.008, 0.012, 1. / 2.e0, 0.5])

# Target beta
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e0, 0.58])

# Beta ranges
beta_low = torch.tensor([0.001, 0.006, 1. / 5., 0.3])
beta_high = torch.tensor([0.021, 0.026, 1. / 0.5e-3, 0.8])
scaling = torch.tensor([1., 1., 1., 1.])

# Other arguments for optAlpha function
max_iters = 100
maxFuncCalls = 200
regularizedFlag = False
noLocalSearch = True
stepping = 'lsrh'
lsrh_steps = 10

# Sequence specific parameters
T = 5.
NofTPts = 1000

# Tolerance parameters
this_rtol = 1.e-6
this_atol = 1.e-8

# Store the keywords for optAlpha
kwgs = {
    'y0' : y0, 
    'alpha0' : alpha0, 
    'VT' : VT,
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
    'T' : T, 
    'NofTPts' : NofTPts, 
    'this_rtol': this_rtol, 
    'this_atol' : this_atol
}

# Function to get target v
def generate_target_v(alpha, VT, beta, y0, this_rtol, this_atol, regularizedFlag):
    # y0[1] = alpha[2]
    targ_SpringSlider = MassFricParams(alpha, VT, beta, y0)
    # targ_SpringSlider.print_info()
    targ_seq = TimeSequenceGen(T, NofTPts, targ_SpringSlider, 
                               rtol=this_rtol, atol=this_atol, regularizedFlag=regularizedFlag)
    v = targ_seq.default_y[1, :], 
    t = targ_seq.t
    return v[0], t


## Number of total alpha-beta iterations
N_AllIters = 1
this_alpha = alpha0
this_beta = beta0

## Run alpha-beta iterations
for i in range(N_AllIters):
    # Print out info
    print("#" * 40, " Total Iteration {0} ".format(i) + "#" * 40)
    
    ## First optimize alpha
    kwgs['alpha0'] = this_alpha
    kwgs['beta_this'] = this_beta
    
    # Timing alpha
    # Update this Alpha
    # this_alpha = optAlpha(O_GAN, kwgs)
    
    
    ## Run grad descent on beta
    # Generate target v
    v, t = generate_target_v(this_alpha, kwgs['VT'], kwgs['beta_targ'], kwgs['y0'], kwgs['this_rtol'], kwgs['this_atol'], kwgs['regularizedFlag'])
    
    # Run gradient descent
    myGradBB = GradDescent(this_alpha, kwgs['alp_low'], kwgs['alp_high'], kwgs['VT'], 
                           this_beta, kwgs['beta_low'], kwgs['beta_high'], 
                           kwgs['y0'], v, t, 
                           objGrad_func = objGradFunc, scaling = kwgs['scaling'], 
                           max_steps = kwgs['max_iters'], stepping = kwgs['stepping'], obs_rtol = 1e-5, lsrh_steps = kwgs['lsrh_steps'], 
                           regularizedFlag = kwgs['regularizedFlag'], 
                           T = kwgs['T'], NofTPts = kwgs['NofTPts'], this_rtol = kwgs['this_rtol'], this_atol = kwgs['this_atol'])
    
    myGradBB.run()
    
    # Update parameters
    this_beta = myGradBB.beta_optimal
    print("Optimal beta: ", this_beta)

