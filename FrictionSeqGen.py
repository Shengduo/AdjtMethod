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


# Target Rate and state properties
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])
beta0 = torch.tensor([0.009, 0.012, 1. / 2.e1, 0.3]) 

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)
VV = torch.tensor([1., 1., 10., 10., 1., 1., 10., 10., 1., 1., 1., 1., 1., 1., 1.])
tt = torch.linspace(0., 30., 15)
VtFunc = interp1d(tt, VV)

kwgs = {
    'VV' : VV, 
    'tt' : tt, 
    'VtFunc' : VtFunc, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
}

# Compute f history based on VtFunc and beta
def cal_f(beta, kwgs):
    tt = kwgs['tt']
    NofTpts = kwgs['NofTpts']
    theta0 = kwgs['theta0']
    VtFunc = kwgs['VtFunc']

    t = torch.linspace(tt[0], tt[-1], NofTpts)
    V = torch.tensor(VtFunc(t), dtype=torch.float)

    theta = torch.zeros(t.shape)

    a = beta[0]
    b = beta[1]
    DRSInv = beta[2]
    fStar = beta[3]
    thetaFunc = lambda t, theta: 1. - torch.tensor(VtFunc(torch.clip(t, tt[0], tt[-1])), dtype=torch.float) * theta * DRSInv
    theta = odeint(thetaFunc, theta0, t, atol = 1.e-10, rtol = 1.e-8)
    
    f = fStar + a * torch.log(V / 1.e-6) + b * torch.log(1.e-6 * theta * DRSInv)

    return V, theta, f

def O(f, f_targ, t):
    return torch.trapezoid(
        torch.square(f - f_targ), t
    )

def grad(beta, t, V, theta, f, f_targ, kwgs):
    integrand = torch.zeros([len(beta), len(t)])
    integrand[0, :] = torch.log(V / 1.e-6)
    integrand[1, :] = torch.log(1.e-6 * theta * beta[2])
    integrand[2, :] = beta[1] / beta[2]
    integrand[3, :] = 1.
    integrand = 2 * (f - f_targ) * integrand
    dodTheta = interp1d(t, 2. * (f - f_targ) * beta[1] / theta)
    dCdTheta = interp1d(t, V * beta[2])

    # Adjoint
    # print(torch.flip(t[-1] - t, [0]))
    laFunc = lambda tau, la: -torch.tensor(dodTheta(torch.clip(t[-1]-tau, t[0], t[-1])), dtype=torch.float) - la * torch.tensor(dCdTheta(torch.clip(t[-1]-tau, t[0], t[-1])), dtype=torch.float)
    la = odeint(laFunc, torch.tensor(0.), torch.flip(t[-1] - t, [0]), atol = 1.e-10, rtol = 1.e-8)
    la = torch.flip(la, [0])
    # print("la: ", la)
    integrand[2, :] += la * V * theta
    res = torch.trapezoid(
        integrand, t
    )

    
    return res

# Plot sequences of friction coefficient
def plotSequences(beta, beta_targ, kwgs, pwd):
    V, theta, f = cal_f(beta, kwgs)
    V_targ, theta_targ, f_targ = cal_f(beta_targ, kwgs)
    plt.figure(figsize=[15, 10])
    plt.plot(t, f_targ, linewidth=2.0)
    plt.plot(t, f, linewidth=1.5)
    plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("Friction coefficient", fontsize=20)
    plt.savefig(pwd + "Fric_0322.png", dpi = 300.)
    plt.close()

    plt.figure(figsize=[15, 10])
    plt.plot(kwgs['tt'], kwgs['VV'], linewidth=2.0)
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "VProfile_0322.png", dpi = 300.)
    plt.close()


## Invert on an problem
t = torch.linspace(tt[0], tt[-1], NofTpts)

V_targ, theta_targ, f_targ = cal_f(beta_targ, kwgs)
# print('V_targ: ', V_targ)
# print('theta_targ: ', theta_targ)
# print('f_targ: ', f_targ)

V0, theta0, f0 = cal_f(beta0, kwgs)
O0 = O(f0, f_targ, t)
print("O0: ", O0)


## Gradient descent:
max_iters = 100
max_step = torch.tensor([0.005, 0.005, 0.01, 0.1])

# Gradient descent
beta_this = beta0
V_this, theta_this, f_this = cal_f(beta_this, kwgs)
O_this = O(f_this, f_targ, t)
grad_this = grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)
max_eta = torch.min(torch.abs(max_step / grad_this))
print("=" * 40, " Iteration ", str(0), " ", "=" * 40)
print("Initial beta: ", beta_this)
print("O: ", O_this)
print("Gradient: ", grad_this, flush=True)

for i in range(max_iters):
    max_eta = torch.min(torch.abs(max_step / grad_this))
    # Line search
    iter = 0
    O_trial = O_this
    while (iter <= 20 and O_trial >= O_this):
        beta_trial = beta_this - grad_this * max_eta * pow(2, -iter)
        V_trial, theta_trial, f_trial = cal_f(beta_trial, kwgs)
        O_trial = O(f_trial, f_targ, t)
        iter += 1

    beta_this = beta_trial
    V_this = V_trial
    theta_this = theta_trial
    f_this = f_trial
    O_this = O_trial
    grad_this = grad(beta_this, t, V_this, theta_this, f_this, f_targ, kwgs)

    print("=" * 40, " Iteration ", str(i + 1), " ", "=" * 40)
    print("Optimized beta: ", beta_this)
    print("O: ", O_this)
    print("Gradient: ", grad_this, flush=True)

# Save a figure of the result
pwd="./plots/FricSeqGen/"
plotSequences(beta_this, beta_targ, kwgs, pwd)
# grad0 = grad(beta0, t, V0, theta0, f0, f_targ, kwgs)
# print("grad0: ", grad0)

# # Numerical gradients
# inc = 0.01
# numerical_grad0 = torch.zeros(beta0.shape)
# for i in range(len(beta0)):
#     beta_plus = torch.clone(beta0)
#     beta_plus[i] *= (1 + inc)
#     print("beta_plus: ", beta_plus)
#     Vp, thetap, fp = cal_f(beta_plus, kwgs)
#     Op = O(fp, f_targ, t)

#     beta_minus = torch.clone(beta0)
#     beta_minus[i] *= (1 - inc)
#     print("beta_minus: ", beta_minus)
#     Vm, thetam, fm = cal_f(beta_minus, kwgs)
#     Om = O(fm, f_targ, t)
#     numerical_grad0[i] = (Op - Om) / (2 * inc * beta0[i])

# print("Numerical_grad0: ", numerical_grad0)