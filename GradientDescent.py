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
# from Derivatives import *
from DerivativesAddTheta import *

## Fixed parameters
# # Sequence specific parameters
# T = 5.
# NofTPts = 1000

# # Tolerance parameters
# this_rtol = 1.e-8
# this_atol = 1.e-10

# Function observation function
def objGradFunc(alpha, VTs, beta, y0, targ_ys, MFParams_targs, scaling, regularizedFlag, objOnly = False, 
                NofTPts = 1000, this_rtol = 1.e-6, this_atol = 1.e-8, 
                solver = 'dopri5', lawFlag = "aging"):
    # Initialize objective and gradient
    obj = 0.
    grad = torch.zeros(beta.shape)

    # Generate target v
    for (VT, targ_y, MFParams_targ) in zip(VTs, targ_ys, MFParams_targs):
        this_RSParams = beta * scaling
        this_SpringSlider = MassFricParams(alpha, VT, this_RSParams, y0, lawFlag, regularizedFlag)
        
        this_seq = TimeSequenceGen(VT[1, -1], NofTPts, this_SpringSlider, 
                                   rtol = this_rtol, atol = this_atol, regularizedFlag = regularizedFlag, 
                                   solver = solver)
        
        # Compute the value of objective function
        obj = obj + O(this_seq.default_y, targ_y, this_seq.t, this_SpringSlider, MFParams_targ)
        
    #     # DEBUG LINES
    #     print("-"*30)
    #     print("This RS params: ", this_RSParams)
    #     print("Objective value: ", obj)
    #     print("-"*30)
        
        # Compute dOdBeta
        if objOnly:
            grad = 0.
        else:
            myAdj = AdjDerivs(this_seq.default_y, targ_y, this_seq.t, this_SpringSlider, MFParams_targ, 
                              rtol = this_rtol, atol = this_atol, regularizedFlag = regularizedFlag, solver = solver)
            grad = grad + myAdj.dOdBeta / scaling
        
    return obj, grad


# Give the initial position and gradient updating function
class GradDescent:
    # Constructor, initial value position
    def __init__(self, 
                 alpha0, alpha_low, alpha_high, 
                 VTs, # Temperarily fix the VTs relation now
                 beta0, beta_low, beta_high, 
                 y0, targ_ys, ts, MFParams_targs, 
                 objGrad_func, max_steps, scaling = torch.tensor([1., 1., 1., 1.]), 
                 stepping = 'BB', obs_rtol = 1e-5, grad_atol = 1.e-10, lsrh_steps = 10, 
                 regularizedFlag = False, NofTPts = 1000, this_rtol = 1.e-6, this_atol = 1.e-8, 
                 solver = 'dopri5', lawFlag = "aging"):
        # Initial parameters, and their lower and upper bound
        # Alpha contains the non-gradient-able parameters
        self.alpha0 = alpha0
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.VTs = VTs
        
        # Beta are the differentiable parameters
        self.beta0 = beta0 / scaling
        self.beta_low = beta_low / scaling
        self.beta_high = beta_high / scaling
        self.beta_matters = (torch.abs(self.beta_high - self.beta_low) >= 1.e-8)

        # y0 is the initial condition to solve the odes
        self.y0 = y0
        # Sync the spring speed and the initial mass block speed
        # self.y0[1] = self.alpha0[2]
        
        # Scale the gradients to get precisions match
        self.scaling = scaling
        
        # Target sequence
        self.targ_ys = targ_ys

        # Time at which targ_y was observed
        self.ts = ts

        # Spring sliders for target generation
        self.MFParams_targs = MFParams_targs

        # Compute L2(t) norm of targ_ys
        self.targ_ys_norm = 0.
        for t, y in zip(self.ts, self.targ_ys):
            # V and theta
            self.targ_ys_norm += torch.trapz(t, y[1, :] * y[1, :] + y[2, :] * y[2, :])

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

        # Time sequence parameters
        self.regularizedFlag = regularizedFlag
        # self.T = T
        self.NofTPts = NofTPts
        self.this_rtol = this_rtol
        self.this_atol = this_atol
        self.solver = solver
        self.lawFlag = lawFlag

        # Get Initial observations
        self.objs = []
    
    # First descent, cannot use Barzilai–Borwein stepping, using linesearch
    def firstDescent(self):
        # Compute obj and grad
        # print("self.targ_ys.shape: ", self.targ_ys.shape )
        obj, grad = self.objGrad_func(self.alpha0, self.VTs, self.betas[-1], self.y0, self.targ_ys, self.MFParams_targs, self.scaling, 
                                      self.regularizedFlag, False, 
                                      self.NofTPts, self.this_rtol, self.this_atol, 
                                      self.solver, self.lawFlag)
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
        BBStepSize = abs(torch.dot(self.betas[-1][self.beta_matters] - self.betas[-2][self.beta_matters], 
                                   self.grads[-1][self.beta_matters] - self.grads[-2][self.beta_matters])) / \
                         torch.sum(torch.square(self.grads[-1][self.beta_matters] - self.grads[-2][self.beta_matters]))
        
        # Calculate the step size
        if self.stepping == 'BB':
            stepSize = BBStepSize
            beta_trial = self.project(self.betas[-1], stepSize * self.grads[-1])

            # Append the betas and objs
            obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VTs, beta_trial, self.y0, self.targ_ys, self.MFParams_targs, 
                                                      self.scaling, self.regularizedFlag, False, 
                                                      self.NofTPts, self.this_rtol, self.this_atol, 
                                                      self.solver, self.lawFlag)
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
        print("beta: ", self.betas[-1])
        print("Relative error of observation: ", torch.sqrt(self.objs[-1]) / self.targ_ys_norm)
        
        if torch.min(self.grad_norms) < self.grad_atol:
            print("The final predicted parameters: ", self.betas[torch.argmin(self.grad_norms)])
            return
        
        # Run max_iters number of iterations
        for i in range(self.max_steps):
            success = self.oneDescent()
            print("=" * 40)
            print("The {0}th descent succeeds: ".format(i + 1), success)
            print("Observation: ", self.objs[-1])
            print("Gradient (scaled): ", self.grads[-1])
            print("beta: ", self.betas[-1])
            print("Relative error of observation: ", torch.sqrt(self.objs[-1]) / self.targ_ys_norm)
            
            # Check if the gradient is small enough
            if torch.min(self.grad_norms) < self.grad_atol:
                break
        
        # Return
        print("The final predicted parameters: ", self.betas[torch.argmin(self.grad_norms)] * self.scaling)
        self.beta_optimal = self.betas[torch.argmin(self.grad_norms)] * self.scaling
        return
    
    # Line search function
    def lineSearch(self, minStepSize = 0.):
        # Find stepsize
        # maxStepSize = 1.0 * min(abs(self.betas[-1] / self.grads[-1]))

        # Consider a only
        maxStepSize = 0.2 * min(abs(self.betas[-1][self.beta_matters] / self.grads[-1][self.beta_matters]))
        
        # maxStepSize = 1.
            
        # Line search
        stepSize = max(maxStepSize, minStepSize)
        # print("Initial step size: ", stepSize)
        # print("Initial beta trial: ", self.project(self.betas[-1], stepSize * self.grads[-1]))

        # # DEBUG LINES
        # print("~"*40, " linesearch initiates ", "~"*40)
        # print("Last objective value: ", self.objs[-1])

        for i in range(self.lsrh_steps):
            beta_trial = self.project(self.betas[-1], stepSize * self.grads[-1])
            obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VTs, beta_trial, self.y0, self.targ_ys, self.MFParams_targs, 
                                                      self.scaling, self.regularizedFlag, True, 
                                                      self.NofTPts, self.this_rtol, self.this_atol, 
                                                      self.solver, self.lawFlag)
            # # DEBUG LINES
            # # print("shit")
            # print("The ", i, " th lsrh iteration:")
            # print("self.betas[-1]: ", self.betas[-1])
            # print("self.grads[-1]: ", self.grads[-1])
            # print("Step size: ", stepSize)
            # print("beta_trial: ", beta_trial)
            # print("self.beta_matters: ", self.beta_matters)
            # beta_trial_unprojected = self.betas[-1] - stepSize * self.grads[-1]
            # beta_trial_unprojected[~self.beta_matters] = self.beta0[~self.beta_matters]
            # print("beta_trial (unprojected): ", beta_trial_unprojected)
            # print("obj_trial: ", obj_trial)
            # print("grad_trial: ", grad_trial, "\n")

            # Break if this beta is good
            if (stepSize < minStepSize) or (obj_trial < self.objs[-1]):
                break
            
            # Half the stepSize
            stepSize = stepSize / 2.
        
        # If linesearch exits but minimum step size hasn't been reached, try with minimum step size
        if (minStepSize > 0.) and (stepSize >= minStepSize) and (obj_trial > self.objs[-1]):
            beta_trial = self.project(self.betas[-1], minStepSize * self.grads[-1])
        
        

        # Append the betas and objs
        obj_trial, grad_trial = self.objGrad_func(self.alpha0, self.VTs, beta_trial, self.y0, self.targ_ys, self.MFParams_targs, 
                                                  self.scaling, self.regularizedFlag, False, 
                                                  self.NofTPts, self.this_rtol, self.this_atol, 
                                                  self.solver, self.lawFlag)
        
        # # DEBUG LINES
        # print("Final step size: ", stepSize)
        # print("Final beta trial: ", beta_trial)
        # print("Final obj: ", obj_trial)
        # print("Final grad: ", grad_trial)
        # print("~"*40, " linesearch finalizes ", "~"*40)

        self.betas.append(beta_trial)
        self.objs.append(obj_trial)
        self.grads.append(grad_trial)
        self.grad_norms = torch.concat([self.grad_norms, torch.linalg.norm(grad_trial).reshape([-1])])
        
        # Return if obj_trial is smaller than obj
        return obj_trial < self.objs[-2]
        
    
    # Project the new point into the constraint set
    def project(self, pt, subtraction):
        # First calculate the subtraction
        prjted = torch.clone(pt)
        prjted[~self.beta_matters] = self.beta0[~self.beta_matters]
        prjted[self.beta_matters] -= subtraction[self.beta_matters]

        # Project on the dimensions that matter
        fun = lambda u: np.linalg.norm(u - np.array(prjted[self.beta_matters]))
        beta_high_matters = self.beta_high[self.beta_matters]
        beta_low_matters = self.beta_low[self.beta_matters]

        prjted_matters = opt.minimize(fun, x0 = np.array((beta_high_matters + beta_low_matters) / 2.), 
                                      bounds = [(beta_low_matters[i], beta_high_matters[i]) 
                                                for i in range(len(beta_low_matters))]
                                     ).x
        
        prjted[self.beta_matters] = torch.tensor(prjted_matters, dtype=torch.float)
        return prjted