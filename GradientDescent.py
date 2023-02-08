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

from MassFricParams import MassFricParams
from TimeSequenceGen import TimeSequenceGen
from AdjointMethod import AdjDerivs
from Derivatives import *


## Fixed parameters
# # Sequence specific parameters
# T = 5.
# NofTPts = 1000

# # Tolerance parameters
# this_rtol = 1.e-8
# this_atol = 1.e-10

# Function observation function
def objGradFunc(alpha, VT, beta, y0, targ_y, scaling, regularizedFlag = False, objOnly = False, 
                T = 5., NofTPts = 1000, this_rtol = 1.e-6, this_atol = 1.e-8):
    # Generate target v
    this_RSParams = beta * scaling
    this_SpringSlider = MassFricParams(alpha, VT, this_RSParams, y0)
    
    this_seq = TimeSequenceGen(T, NofTPts, this_SpringSlider, 
                               rtol=this_rtol, atol=this_atol, regularizedFlag = regularizedFlag)
    
    # Compute the value of objective function
    obj = O(this_seq.default_y, targ_y, this_seq.t, this_SpringSlider)
    
#     # DEBUG LINES
#     print("-"*30)
#     print("This RS params: ", this_RSParams)
#     print("Objective value: ", obj)
#     print("-"*30)
    
    # Compute dOdBeta
    if objOnly:
        grad = 0.
    else:
        myAdj = AdjDerivs(this_seq.default_y, targ_y, this_seq.t, this_SpringSlider, 
                          rtol = this_rtol, atol = this_atol, regularizedFlag = regularizedFlag)
        grad = myAdj.dOdBeta / scaling
        
    return obj, grad


# Give the initial position and gradient updating function
class GradDescent:
    # Constructor, initial value position
    def __init__(self, 
                 alpha0, alpha_low, alpha_high, 
                 VT, # Temperarily fix the VT relation now
                 beta0, beta_low, beta_high, 
                 y0, targ_y, t, 
                 objGrad_func, max_steps, scaling = torch.tensor([1., 1., 1., 1.]), 
                 stepping = 'BB', obs_rtol = 1e-5, grad_atol = 1.e-10, lsrh_steps = 10):
        # Initial parameters, and their lower and upper bound
        # Alpha contains the non-gradient-able parameters
        self.alpha0 = alpha0
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.VT = VT
        
        # Beta are the differentiable parameters
        self.beta0 = beta0 / scaling
        self.beta_low = beta_low / scaling
        self.beta_high = beta_high / scaling
        
        # y0 is the initial condition to solve the odes
        self.y0 = y0
        # Sync the spring speed and the initial mass block speed
        # self.y0[1] = self.alpha0[2]
        
        # Scale the gradients to get precisions match
        self.scaling = scaling
        
        # Target sequence
        self.targ_y = targ_y
        
        # Time at which targ_y was observed
        self.t = t
        
        # Objective and gradient function
        self.objGrad_func = objGrad_func
        
        # Maximum number of steps allowed
        self.max_steps = max_steps
        
        # Stepping scheme, either BB or lsrh
        self.stepping = stepping
        
        # Tolerance of observation relative error
        self.obs_tol = obs_rtol
        
        # Tolerance of gradient absolute error
        self.grad_atol = grad_atol
        
        # Sequence of parameters
        self.betas = [self.beta0]
        
        # Maximum line search steps
        self.lsrh_steps = lsrh_steps
        
        # Get Initial observations
        self.objs = []
    
    # First descent, cannot use Barzilai–Borwein stepping, using linesearch
    def firstDescent(self):
        # Compute obj and grad
        obj, grad = self.objGrad_func(self.alpha0, self.VT, self.betas[-1], self.y0, self.targ_y, self.scaling)
        self.objs = [obj]
        self.grads = [grad]
        
        # Norm of gradients
        self.grad_norms = torch.linalg.norm(grad).reshape([-1])
        
        # Perform linesearch
        return self.lineSearch()
    
    # Run one descent using either Barzilai–Borwein stepping or linesearch
    def oneDescent(self):
        # Make sure there are more than 1 steps beforehand
        assert(len(self.objs) >= 2)
        
        # Compute BB stepsize
        BBStepSize = abs(torch.dot(self.betas[-1] - self.betas[-2], self.grads[-1] - self.grads[-2])) / \
                       torch.sum(torch.square(self.grads[-1] - self.grads[-2]))
        
        # Calculate the step size
        if self.stepping == 'BB':
            stepSize = BBStepSize
            beta_trial = self.project(self.betas[-1] - stepSize * self.grads[-1])

            # Append the betas and objs
            obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VT, beta_trial, self.y0, self.targ_y, 
                                                      self.scaling, objOnly = False)
            self.betas.append(beta_trial)
            self.objs.append(obj_trial)
            self.grads.append(grad_trial)
            self.grad_norms = torch.concat([self.grad_norms, torch.linalg.norm(grad_trial).reshape([-1])])
            
            # Return if obj_trial is smaller than obj
            return obj_trial < self.objs[-2]
        
        # Line search mechanism
        elif self.stepping == 'lsrh':
            return self.lineSearch(BBStepSize)
        
    
    # Run gradient descent
    def run(self):
        # Run initial descent
        success = self.firstDescent()
        print("=" * 40)
        print("Initial descent succeeds: ", success)
        print("Observation: ", self.objs[-1])
        print("Gradient (scaled): ", self.grads[-1])
        print("Relative error of observation: ", torch.sqrt(self.objs[-1]) / torch.linalg.norm(self.targ_y))
        
        if torch.min(self.grad_norms) < self.grad_atol:
            print("The final predicted parameters: ", self.betas[torch.argmin(self.grad_norms)])
            return
        
        # Run max_iters number of iterations
        for i in range(self.max_steps):
            success = self.oneDescent()
            print("=" * 40)
            print("The {0}th descent succeeds: ".format(i + 1), success)
#             print("Observation: ", self.objs[-1])
            print("Gradient (scaled): ", self.grads[-1])
            print("Relative error of observation: ", torch.sqrt(self.objs[-1]) / torch.linalg.norm(self.targ_y))
            
            # Check if the gradient is small enough
            if torch.min(self.grad_norms) < self.grad_atol:
                break
        
        # Return
        print("The final predicted parameters: ", self.betas[torch.argmin(self.grad_norms)] * self.scaling)
        self.beta_optimal = self.betas[torch.argmin(self.grad_norms)] * self.scaling
        return
    
    # Line search functino
    def lineSearch(self, minStepSize = 0.):
        # Find stepsize
        maxStepSize = 0.1 * min(abs(self.betas[-1] / self.grads[-1]))
            
        # Line search
        stepSize = max(maxStepSize, minStepSize)
        
        for i in range(self.lsrh_steps):
            beta_trial = self.project(self.betas[-1] - stepSize * self.grads[-1])
            obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VT, beta_trial, self.y0, self.targ_y, 
                                                      self.scaling, objOnly = True)
            print("shit")
            
            # Break if this beta is good
            if (stepSize < minStepSize) or (obj_trial < self.objs[-1]):
                break
            
            # Half the stepSize
            stepSize = stepSize / 2.
        
        # If linesearch exits but minimum step size hasn't been reached, try with minimum step size
        if (minStepSize > 0.) and (stepSize >= minStepSize) and (obj_trial > self.objs[-1]):
            beta_trial = self.project(self.betas[-1] - minStepSize * self.grads[-1])
        
        # Append the betas and objs
        obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VT, beta_trial, self.y0, self.targ_y, 
                                                  self.scaling, objOnly = False)
        self.betas.append(beta_trial)
        self.objs.append(obj_trial)
        self.grads.append(grad_trial)
        self.grad_norms = torch.concat([self.grad_norms, torch.linalg.norm(grad_trial).reshape([-1])])
        
        # Return if obj_trial is smaller than obj
        return obj_trial < self.objs[-2]
        
    
    # Project the new point into the constraint set
    def project(self, pt):
        fun = lambda u: np.linalg.norm(u - np.array(pt))
        prjted = opt.minimize(fun, x0 = np.array((self.beta_low + self.beta_high) / 2.), 
                              bounds = [(self.beta_low[i], self.beta_high[i]) 
                                        for i in range(len(self.beta_low))]
                             ).x
        return torch.tensor(prjted, dtype=torch.float)