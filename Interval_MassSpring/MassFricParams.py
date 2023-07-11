# Import packages
import torch
import torch.nn as nn
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import scipy.optimize as opt
import pickle
import numpy as np
import time


# torch.set_default_dtype(torch.float)

"""
Class MassFricParams, manages data of a mass block sliding on rate-and-state friction surface, contains 
    Data:
        k : Spring stiffness
        m : Mass of the block
        V : Leading head speed of the spring
        g : Gravity
        RSParams : rate and state parameters, torch.tensor([a, b, DRS, f*])
        y0 : torch.tensor([initial x_1, initial v_1, initial state variable])
"""
class MassFricParams: 
    # Constructor
    def __init__(self, kmg, VV, TT, RSParams, y0, lawFlag = "aging", regularizedFlag = True):
        # Define constant parameters k, m and g
        self.k = kmg[0]
        self.m = kmg[1]
        self.g = kmg[2]
        
        # Get the VT relation
        # print("VT: ", VT)
        self.VV = VV
        self.TT = TT
        
        # Get the displacement at T
        self.S = torch.zeros([len(self.VV)])
        # self.S[1:] = torch.cumulative_trapezoid(self.VV, self.TT)
        
        self.RSParams = RSParams
        self.y0 = y0
        self.y0[1] = VV[0]

        self.lawFlag = lawFlag
        self.regularizedFlag = regularizedFlag
        
        # Get the jump points
        self.JumpIdx = [0]
        self.JumpT = [self.TT[0]]
        for i in range(1, len(self.VV)):
            if self.VV[i] != self.VV[i - 1]:
                self.JumpIdx.append(i)
                self.JumpT.append(self.TT[i])
        self.JumpIdx.append(len(self.VV) - 1)
        self.JumpT.append(self.TT[-1])

        # Get the function of V, S at T
        self.VtFuncs = []
        self.stFuncs = []
        for i in range(len(self.JumpIdx) - 1):
            # this_V = torch.tensor(self.VV[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1]).clone()
            # this_T = torch.tensor(self.TT[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1]).clone()
            this_V = self.VV[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1].clone()
            this_T = self.TT[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1].clone()
            self.S[self.JumpIdx[i] + 1 : self.JumpIdx[i + 1] + 1] = torch.cumulative_trapezoid(this_V, this_T) + self.S[self.JumpIdx[i]]
            this_V[-1] = this_V[-2]
            this_vtFunc = interp1d(this_T, this_V)
            this_stFunc = interp1d(this_T, self.S[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1])
            self.VtFuncs.append(this_vtFunc)
            self.stFuncs.append(this_stFunc)

            # # DEBUG LINES
            # print("~+"*30, " In MassFricParams ", "+~"*30)
            # print("this_V: ", this_V)
            # print("this_S: ", self.S[self.JumpIdx[i] : self.JumpIdx[i + 1] + 1])
            # print("~+"*30, "                   ", "+~"*30, flush=True)

    # Define the function that gives V at t
    def VatT_interp(self, t):
        t_clipped = torch.clip(t, self.TT[0], self.TT[-1])
        for idx, jumpT in enumerate(self.JumpT):
            if jumpT > t_clipped:
                return torch.tensor(self.VtFuncs[idx - 1](t_clipped), dtype=torch.float)

        # If the last interval
        return torch.tensor(self.VtFuncs[-1](t_clipped), dtype=torch.float) 
    
    # Define the function that gives S at t
    def SatT_interp(self, t):
        t_clipped = torch.clip(t, self.TT[0], self.TT[-1])
        for idx, jumpT in enumerate(self.JumpT):
            if jumpT > t_clipped:
                return torch.tensor(self.stFuncs[idx - 1](t_clipped), dtype=torch.float)

        # If the last interval
        return torch.tensor(self.stFuncs[-1](t_clipped), dtype=torch.float)
    
    # Output the information of this class
    def print_info(self):
        print("-" * 20, " Mass and spring parameters ", "-"*20)
        print('k:        ', self.k)
        print('m:        ', self.m)
        print('g:        ', self.g)
        print('\n')
        
        print("-" * 20, " Rate-and-state parameters ", "-"*20)
        print('fr:       ', self.RSParams[3])
        print('a:        ', self.RSParams[0])
        print('b:        ', self.RSParams[1])
        # print('DRS:      ', self.RSParams[2])
        print('1 / DRS:  ', self.RSParams[2])
        print('y0:       ', self.y0)
        print('law:      ', self.lawFlag)
        # # Plot V at t
        # plt.figure()
        # plt.plot(self.TT, self.VV, linewidth = 2.0)
        # plt.show()
        