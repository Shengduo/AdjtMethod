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
# from Derivatives import *
from DerivativesAddTheta import *


torch.set_default_dtype(torch.float)

# ----------------------------- Calculate gradients via Adjoint Method -----------------------------
## Fixed parameters
# Parameters for the spring-slider
k = 100.
m = 1.
VT = torch.tensor([[1., 1.], [0., 5.]])
g = 9.8
y0 = torch.tensor([0., 1.0, 1.0])
kmg = torch.tensor([k, m, g])
# Sequence specific parameters
T = 5.
NofTPts = 1000

# Tolerance parameters
this_rtol = 1.e-8
this_atol = 1.e-10

# Regularized flag
regularizedFlag = True

# Generate target v
targ_RSParams = torch.tensor([0.006, 0.010, 1. / 1., 0.58])
targ_SpringSlider = MassFricParams(kmg, VT, targ_RSParams, y0)
# targ_SpringSlider.print_info()
targ_seq = TimeSequenceGen(T, NofTPts, targ_SpringSlider, rtol=this_rtol, atol=this_atol, regularizedFlag=regularizedFlag)
v = targ_seq.default_y
# targ_seq.plotY(targ_seq.t, targ_seq.default_y)


# A new set of RS params
new_RSParams = torch.tensor([0.008, 0.012, 1. / 5., 0.3])
# new_RSParams = torch.tensor([0.011, 0.016, 1.e-3, 0.58])
new_SpringSlider = MassFricParams(kmg, VT, new_RSParams, y0)
new_seq = TimeSequenceGen(T, NofTPts, new_SpringSlider, rtol=this_rtol, atol=this_atol, regularizedFlag = regularizedFlag)
# new_seq.plotY(new_seq.t, new_seq.default_y)


# Report observation:
Obs = O(new_seq.default_y, v, new_seq.t, new_SpringSlider)
print('Objective value: ', Obs)


# Calculate DoDBeta
myRegADJ = AdjDerivs(new_seq.default_y, v, new_seq.t, new_SpringSlider, regularizedFlag = regularizedFlag, 
                     rtol = 1.e-8, atol = 1.e-10)
# Calculate DoDBeta
myUnRegADJ = AdjDerivs(new_seq.default_y, v, new_seq.t, new_SpringSlider, regularizedFlag = not regularizedFlag, 
                       rtol = 1.e-8, atol = 1.e-10)

# Report gradients
print('='*20, ' Gradient via Adjoint method ', '='*20)
print('myRegADJ.dOdBeta: ', myRegADJ.dOdBeta)
print('myUnRegADJ.dOdBeta: ', myUnRegADJ.dOdBeta)
print('\n')

# ----------------------------- Calculate gradients via Adjoint Method -----------------------------
print('='*20, ' Gradient via Finite Difference ', '='*20)
perturbRatio = 0.01
numericalGrad = torch.zeros(new_RSParams.shape)
Rtol = 1.e-8
Atol = 1.e-10


# Loop through all beta's components
for i in range(len(new_RSParams)):
    RSParamsPlus = torch.clone(new_RSParams)
    RSParamsPlus[i] = RSParamsPlus[i] * (1 + perturbRatio)
    RSParamsMinus = torch.clone(new_RSParams)
    RSParamsMinus[i] = RSParamsMinus[i] * (1 - perturbRatio)
    
    print("-" * 40)
    print("RSParams: ", new_RSParams)
    print("RSParamsPlus: ", RSParamsPlus)
    print("RSParamsMinus: ", RSParamsMinus)
    
    # kmg, VT, targ_RSParams, y0
    # Calculate two observations
    SpringSliderPlus = MassFricParams(kmg, VT, RSParamsPlus, y0)
    seqPlus = TimeSequenceGen(T, NofTPts, SpringSliderPlus, Rtol, Atol, regularizedFlag=regularizedFlag)
    OPlus = O(seqPlus.default_y, v, seqPlus.t, SpringSliderPlus)
    
    SpringSliderMinus = MassFricParams(kmg, VT, RSParamsMinus, y0)
    seqMinus = TimeSequenceGen(T, NofTPts, SpringSliderMinus, Rtol, Atol, regularizedFlag=regularizedFlag)
    OMinus = O(seqMinus.default_y, v, seqMinus.t, SpringSliderMinus)
    
    numericalGrad[i] = (OPlus - OMinus) / (RSParamsPlus[i] - RSParamsMinus[i])

print('Numerical gradient via Finite Difference: ', numericalGrad)