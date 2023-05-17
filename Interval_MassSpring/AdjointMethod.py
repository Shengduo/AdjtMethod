# Class Adjoint derivatives
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
# from Derivatives import *
from DerivativesAddTheta import *

class AdjDerivs:
    # Constructor
    def __init__(self, y, y_targ, t, MFParams, MFParams_targ, regularizedFlag = False, rtol = 1.e-6, atol = 1.e-8, solver = 'dopri5'):
        self.y = y
        self.y_targ = y_targ
        self.t = t
        
        # Define tau = T - t
        self.T = self.t[-1]
        self.tau = self.T - self.t
        
        # self.tau = torch.flip(self.T - self.t, [0])
        self.MFParams = MFParams
        self.MFParams_targ = MFParams_targ
        self.regularizedFlag = regularizedFlag
        self.MFParams.regularizedFlag = regularizedFlag
        self.MFParams_targ.regularizedFlag = regularizedFlag
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        
        # The jump indexes in t and tt
        self.tt = self.MFParams.T
        self.JumpIdx = self.MFParams.JumpIdx
        self.JumpTT = self.MFParams.JumpT

        # Calculate t_JumpIdx
        self.t_JumpIdx = []
        for jumpT in self.JumpTT:
            for idx in range(len(self.t)):
                if self.t[idx] >= jumpT:
                    self.t_JumpIdx.append(idx)
                    break
        
        self.t_JumpIdx[-1] = len(self.t)

        ## Calculate the partial derivatives ##
        if regularizedFlag:
            self.dCdy = DCDy_regularized(y, y_targ, t, MFParams)
            self.dCdBeta = DCDBeta_regularized(y, y_targ, t, MFParams)
        else:
            self.dCdy = DCDy(y, y_targ, t, MFParams)
            self.dCdBeta = DCDBeta(y, y_targ, t, MFParams)
        
        self.dody = DoDy(y, y_targ, t, MFParams, MFParams_targ)
        self.dCdyDot = DCDyDot(y, y_targ, t, MFParams)
        self.ddCdyDotdt = DDCDyDotDt(y, y_targ, t, MFParams)
        self.dodyDot = DoDyDot(y, y_targ, t, MFParams)
        self.ddodyDotdt = DDoDyDotDt(y, y_targ, t, MFParams)
        self.dodBeta = DoDBeta(y, y_targ, t, MFParams, MFParams_targ)
        self.dodyDot = DoDyDot(y, y_targ, t, MFParams)
        self.dCdyDot = DCDyDot(y, y_targ, t, MFParams)
        
        # # DEBUG LINES
        # print("self.dodBeta: ", self.dodBeta)
        # print("self.dody: ", self.dody)

        # Calculate A_z and u_z
        self.Az = self.A_z()
        self.uz = self.u_z()
        
        # Calculate A_l and u_l
        self.Als = self.A_l()
        self.uls = self.u_l()
        
        # Calculate dOdBeta
        st = time.time()
        self.dOdBeta = self.DODBeta()
        self.time_cost = time.time() - st
        print("Time cost in computing gradients: ", self.time_cost)
    
    ## d\lambda / dtau = f_l(l, tau) = -(l \cdot A_l(tau) + u_l(tau)) ##
    # calculate A_l at tau
    def A_l(self):
        dCdyDot = torch.transpose(self.dCdyDot, 2, 0)
        dCdy = torch.transpose(self.dCdy, 2, 0)
        ddCdyDotdt = torch.transpose(self.ddCdyDotdt, 2, 0)
        
        # Calculate A_l at tSteps
        A_l_discrete = torch.linalg.solve(dCdyDot, dCdy - ddCdyDotdt)
        A_l_discrete = torch.transpose(A_l_discrete, 0, 2)
        
        # self.A_l_discrete = A_l_discrete
        # Define the series of interpolation functions
        A_ls = []
        for idx in range(len(self.t_JumpIdx) - 1):
            # The real index
            this_idx = len(self.t_JumpIdx) - idx - 1
            this_A_l_discrete = A_l_discrete[:, :, self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx]]
            this_A_l_discrete = torch.cat([this_A_l_discrete[:, :, [0]], this_A_l_discrete, this_A_l_discrete[:, :, [-1]]], -1)
            this_t = self.t[self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx]]
            this_t = torch.cat([torch.tensor([self.JumpTT[idx - 1]]), this_t, torch.tensor([self.JumpTT[idx]])])
            this_A_l = interp1d(self.T - this_t, this_A_l_discrete)
            A_ls.append(this_A_l)
        
        return A_ls
    
    # Calculate u_l at tau
    def u_l(self):
        dCdyDot = torch.transpose(self.dCdyDot, 2, 0)
        dody = torch.transpose(self.dody, 1, 0)
        ddodyDotdt = torch.transpose(self.ddodyDotdt, 1, 0)
        
        # Calculate u_l at tSteps
        u_l_discrete = torch.linalg.solve(dCdyDot, dody - ddodyDotdt)
        u_l_discrete = torch.movedim(u_l_discrete, 0, 1)
        
        u_ls = []
        for idx in range(len(self.t_JumpIdx) - 1):
            # The real index
            this_idx = len(self.t_JumpIdx) - idx - 1
            this_u_l_discrete = u_l_discrete[:, self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx]]
            this_u_l_discrete = torch.cat([this_u_l_discrete[:, [0]], this_u_l_discrete, this_u_l_discrete[:, [-1]]], -1)
            this_t = self.t[self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx]]
            this_t = torch.cat([torch.tensor([self.JumpTT[idx - 1]]), this_t, torch.tensor([self.JumpTT[idx]])])
            this_u_l = interp1d(self.T - this_t, this_u_l_discrete)
            u_ls.append(this_u_l)

        return u_ls
    
    # Calculate f_l(self, tau, l)
    def f_l(self, tau, l):
        tau = torch.clip(tau, self.t[0], self.t[-1])
        res = -torch.matmul(l, torch.tensor(self.Al(tau), dtype=torch.float)) - \
              torch.tensor(self.ul(tau), dtype=torch.float)
        return res
    
    ## dz / dt = f_z(z, t) = A_z(t) \cdot z + u_z(t) ##
    # Calculate A_z at t
    def A_z(self):
        dCdyDot = torch.movedim(self.dCdyDot, 2, 0)
        dCdy = torch.movedim(self.dCdy, 2, 0)
        
        # Calculate A_z at tSteps
        A_z_discrete = -torch.linalg.solve(dCdyDot, dCdy)
        A_z_discrete = torch.movedim(A_z_discrete, 0, 2)
        
        # DEBUG LINES
        self.A_z_discrete = A_z_discrete
        
        # Compute the interpolation for slip rate Ds
        t_temp = torch.concat([torch.tensor([self.t[0] - 1.0]), self.t, torch.tensor([self.t[-1] + 1.0])], 0)
        A_z_discrete_temp = torch.concat([A_z_discrete[:, :, [0]], A_z_discrete, A_z_discrete[:, :, [-1]]], -1)
        
        # Return the function
        A_z = interp1d(t_temp, A_z_discrete_temp, kind="cubic")
        return A_z
    
    # Calculate u_z at t
    def u_z(self):
        dCdyDot = torch.movedim(self.dCdyDot, 2, 0)
        dCdBeta = torch.movedim(self.dCdBeta, 2, 0)
        
        # Calculate A_z at tSteps
        u_z_discrete = -torch.linalg.solve(dCdyDot, dCdBeta)
        u_z_discrete = torch.movedim(u_z_discrete, 0, 2)
        
        # DEBUG LINES
        self.u_z_discrete = u_z_discrete
    
        # Compute the interpolation for slip rate Ds
        t_temp = torch.concat([torch.tensor([self.t[0] - 1.0]), self.t, torch.tensor([self.t[-1] + 1.0])], 0)
        u_z_discrete_temp = torch.concat([u_z_discrete[:, :, [0]], u_z_discrete, u_z_discrete[:, :, [-1]]], -1)
        
        # Return the function
        u_z = interp1d(t_temp, u_z_discrete_temp, kind="cubic")
        return u_z
    
    # Calculate f_z
    def f_z(self, t, z):
        
        res = torch.matmul(torch.tensor(self.Az(t), dtype=torch.float), z) + \
              torch.tensor(self.uz(t), dtype=torch.float)
        
        
#         # DEBUG LINES
#         print("-" * 40)
#         print('t = ', t)
#         print('z = ', z)
#         print('f_z: ', res)
        
        return res
    
    # d observation / d \beta
    def DODBeta(self):
        ## First solve for lambda(t) ##
        A = self.dCdy - self.ddCdyDotdt
        B = -self.dody + self.ddodyDotdt

        # Switch dimensions by torch.transpose
        A = torch.transpose(A, 0, 2)
        B = torch.transpose(B, 0, 1)

        # Solve for lambda [Tsteps, ]
        # L = torch.linalg.solve(A, B)
        L0 = torch.zeros(self.y.shape[0])
        
        # Solve for L(t)
        L = torch.zeros([len(L0), 1, len(self.t)])
        
        # Loop thru all intervals
        this_L0 = L0
        for idx in range(len(self.t_JumpIdx) - 1):
            this_idx = len(self.t_JumpIdx) - 1 - idx
            this_t = self.t[self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx]]
            this_t = torch.cat([torch.tensor([self.JumpTT[this_idx - 1]]), this_t, torch.tensor([self.JumpTT[this_idx]])])
            
            # Deal with duplicates
            i = 0
            j = len(this_t)
            if this_t[0] == this_t[1]:
                i = 1
            if this_t[-1] == this_t[-2]:
                j = -1
            this_t = this_t[i : j]
            this_tau = self.T - this_t
            this_tau_low = self.T - self.JumpTT[this_idx]
            this_tau_high = self.T - self.JumpTT[this_idx - 1]
            f_l = lambda tau, l: -torch.matmul(l, torch.tensor(self.Als[idx](torch.clip(tau, this_tau_low, this_tau_high)), dtype=torch.float)) - \
                                  torch.tensor(self.uls[idx](torch.clip(tau, this_tau_low, this_tau_high)), dtype=torch.float)
            this_L = odeint(f_l, this_L0, torch.flip(this_tau, [0]), 
                            rtol = self.rtol, atol = self.atol, method = self.solver)
            if i == 0:
                i = 1
            else:
                i = 0
            if j == -1:
                j = len(this_tau)
            else:
                j = -1
            
            this_L = this_L.reshape([this_L.shape[0], 1, this_L.shape[1]])
            this_L = torch.flip(this_L, [0])
            this_L0 = this_L[0, :, :].reshape(-1)
            L[self.t_JumpIdx[this_idx - 1] : self.t_JumpIdx[this_idx], 0, :] = this_L[i : j, 0, :]

        # # DEBUG LINES
        # print("L: ", L)
        
        LDCDBeta = torch.movedim(self.dCdBeta, 2, 0)
        LDCDBeta = torch.matmul(L, LDCDBeta)
        LDCDBeta = LDCDBeta.reshape([LDCDBeta.shape[0], LDCDBeta.shape[2]])

#         # DEBUG LINES
#         print('L shape: ', L.shape)
#         print('LDCDBeta shape: ', LDCDBeta.shape)
#         print('L[-1, :, :]: ', L[-1, :, :])
        
        integrand = self.dodBeta + torch.transpose(LDCDBeta, 0, 1)
        DODBeta = torch.trapezoid(
            integrand, 
            self.t
        )
        
        
        
#         ## Then calculate z(T) ##
#         z0 = torch.zeros([self.y.shape[0], self.MFParams.RSParams.shape[0]])
#         zT = odeint(self.f_z, z0, self.t, 
#                     rtol = 1.e-10, atol = 1.e-12, method = 'dopri5')
#         zT_last = zT[-1, :, :]
        
#         # DEBUG LINES 
#         print("zT_last: ", zT_last)
#         print("LT: ", L[-1, :, :])
        
        # DEBUG LINES 
        # print('DODBeta: ', DODBeta)
        
#         DODBeta += ((self.dodyDot[:, -1] + L[-1, :, :] @ self.dCdyDot[:, :, -1]) @ zT_last).reshape([-1])
#         print('DODBeta after boundary term: ', DODBeta)
        
#         ## USE THE ALTERNATIVE WAY
#         DODBeta_alter = torch.trapezoid(
#             2 * (self.y[1, :] - self.v) * torch.transpose(zT[:, 1, :], 0, 1), 
#             self.t
#         )
#         print('DODBeta alternative ', DODBeta_alter)
        
        return DODBeta
        