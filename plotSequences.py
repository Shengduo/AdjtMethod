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
beta0 = torch.tensor([0.008, 0.012, 2.e0, 0.5])

# Target beta
beta_targ = torch.tensor([0.011, 0.016, 1.e0, 0.58])

# Beta ranges
beta_low = torch.tensor([0.001, 0.006, 0.5e-3, 0.3])
beta_high = torch.tensor([0.021, 0.026, 5, 0.8])
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

## Function to plot the diagrams of a given set of sequences
def plot_differences(kwgs, betas, betas_legend, savePath = './plots/shit.png'):
    # Store all sequeces as a list
    ys = []
    ts = []
    lwidths = np.linspace(2., 1., len(betas))

    # Generate the sequences
    for beta in betas:
        targ_SpringSlider = MassFricParams(kwgs['alpha0'], kwgs['VT'], beta, kwgs['y0'])
        # targ_SpringSlider.print_info()
        targ_seq = TimeSequenceGen(kwgs['T'], kwgs['NofTPts'], targ_SpringSlider, 
                                   rtol=kwgs['this_rtol'], atol=kwgs['this_atol'], regularizedFlag=kwgs['regularizedFlag'])
        
        # Append the sequences into the list
        ys.append(targ_seq.default_y)
        ts.append(targ_seq.t)

    # Plot all (2) sequences
    # Plot Sequence of V(t) and theta(t) given sample-index
    f, axs = plt.subplots(2, 2, figsize = (15, 15))

    # Plot x_1(t)
    for (t, y, lw) in zip(ts, ys, lwidths):
        axs[0][0].plot(t, y[0, :], linewidth=lw)
    # axs[0][0].legend(betas_legend, loc='best', fontsize=20)
    axs[0][0].set_xlabel('Time [s]', fontsize=20)
    axs[0][0].set_ylabel('Slip $x_1(t)\  \mathrm{[m]}$', fontsize=20)
    # axs[0][0].set_ylim([1e-15, 1e2])
    axs[0][0].grid()

    # Plot v_1(t)
    for (t, y, lw) in zip(ts, ys, lwidths):
        axs[0][1].semilogy(t, y[1, :], linewidth=lw)
    # axs[0][1].legend(betas_legend, loc='best', fontsize=20)
    axs[0][1].set_xlabel('Time [s]', fontsize=20)
    axs[0][1].set_ylabel('Slip rate $v_1(t)\ \mathrm{[m/s]}$', fontsize=20)
    # axs[0][1].set_ylim([0, 15])
    axs[0][1].grid()

    # Plot theta(t)
    for (t, y, lw) in zip(ts, ys, lwidths):
        axs[1][0].semilogy(t, y[2, :], linewidth=lw)
    # axs[1][0].semilogy(1e6 * t, self.MFParams.RSParams[2] / y[1, :], linewidth=2.0)
    axs[1][0].set_xlabel('Time [$\mu$ s]', fontsize=20)
    axs[1][0].set_ylabel('State Variable $\\theta(t)\ \mathrm{[s]}$', fontsize=20)
    axs[1][0].legend(betas_legend, loc='best', fontsize=20)
    axs[1][0].grid()
    f.savefig(savePath, dpi=500.)


## Main executions 
betas = torch.tensor([[0.011, 0.016, 1. / 1.e0, 0.58], 
                      [0.008, 0.012, 1. / 2.e0, 0.5], 
                      # [0.0110, 0.0100, 1. / 1.9997, 0.5050]])
                      [0.0110, 0.0099, 0.5007, 0.5032]])
betas_legend = ["True", "Init", "Finl"]
plot_differences(kwgs, betas, betas_legend, './plots/shit.png')