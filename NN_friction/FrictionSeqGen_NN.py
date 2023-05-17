## Import standard librarys
import torch
import torchdiffeq
import pickle
import time
import torch.nn as nn
import torch.optim as optim
import scipy.optimize as opt
import numpy as np

from pathlib import Path
from torchdiffeq import odeint
from xitorch.interpolate import Interp1D
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
torch.autograd.set_detect_anomaly(True)
# ----------------------------------------- README ----------------------------------------------
# Here we try to approximate the rate-and-state friction model using the formulation below:
# f(V, \xi) &= f_0 + NN1(V, \xi, log(V), log(\xi))
# d\xi / dt &= NN2(V, \xi, log(V), log(\xi))
# -----------------------------------------------------------------------------------------------
# Class that computes f-sequence given V-sequence
class NN_computeF(nn.Module):
    # Constructor
    def __init__(self, kwgs):
        super().__init__()

        # Store the kwgs
        self.kwgs = kwgs

        # For f = NN1(V, log(V), Xi, log(Xi))
        NN1s = kwgs['NN1s']
        NN1_input_dim = kwgs['NN1_input_dim']
        NN1_output_dim = kwgs['NN1_output_dim']

        # \dot{\xi} = NN2(V, log(V), \xi, log(\xi))
        NN2s = kwgs['NN2s']
        NN2_input_dim = kwgs['NN2_input_dim']
        NN2_output_dim = kwgs['NN2_output_dim']

        # Define function NN1
        self.NN1 = nn.Sequential(
            nn.Linear(NN1_input_dim, NN1s[0]), 
            nn.ReLU(),
        )
        
        for i in range(len(NN1s) - 1):
            self.NN1.append(nn.Linear(NN1s[i], NN1s[i + 1]))
            self.NN1.append(nn.ReLU())
        
        self.NN1.append(nn.Linear(NN1s[-1], NN1_output_dim))

        # Define function NN2
        self.NN2 = nn.Sequential(
            nn.Linear(NN2_input_dim, NN2s[0]), 
            nn.ReLU(),
        )
        
        for i in range(len(NN2s) - 1):
            self.NN2.append(nn.Linear(NN2s[i], NN2s[i + 1]))
            self.NN2.append(nn.ReLU())
        
        self.NN2.append(nn.Linear(NN2s[-1], NN2_output_dim))

    def forward(self, VtFunc):
        NofTpts = kwgs['NofTpts']

        # Get all sequences
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        V = torch.tensor(VtFunc(t), dtype=torch.float)
        # shit = torch.concat([torch.tensor(VtFunc(1.), dtype=torch.float).reshape(-1), 
        #                      torch.log(torch.tensor(VtFunc(1.), dtype=torch.float)).reshape(-1), 
        #                      torch.tensor([2.]), 
        #                      torch.log(torch.tensor([2.]))])
        # print("Shit: ", shit)

        # XiFunc = lambda t, Xi: self.NN2(torch.concat([torch.tensor(VtFunc(t), dtype=torch.float).reshape(-1), 
        #                                               torch.log(torch.tensor(VtFunc(t), dtype=torch.float)).reshape(-1), 
        #                                               Xi.reshape(-1), 
        #                                               torch.log(Xi).reshape(-1)])
        #                                 )
        # Xi = odeint(XiFunc, Xi0, t, atol = 1.e-10, rtol = 1.e-8)
        
        # Forward Euler
        Xi = torch.ones([self.kwgs['DimXi'], NofTpts]) * kwgs['Xi0'];
        for i in range(1, NofTpts):
            DXiDt = self.NN2(
                torch.concat(
                    [V[i].reshape(-1), 
                    torch.log(V[i]).reshape(-1),  
                    Xi[:, i - 1].reshape(-1), 
                    torch.exp(Xi[:, i - 1]).reshape(-1)]
                            )
                )
            Xi[:, i] = Xi[:, i - 1] + DXiDt * (t[i] - t[i - 1])
        
        # ass = torch.transpose(torch.concat([V.reshape([1, -1]), torch.log(V).reshape([1, -1]), Xi, torch.exp(Xi)], dim = 0), 0, 1)
        # print("Ass shape: ", ass.shape)
        f = f0 + self.NN1(torch.transpose(torch.concat([V.reshape([1, -1]), torch.log(V).reshape([1, -1]), Xi, torch.exp(Xi)], dim = 0), 0, 1))

        return f.reshape(-1)


# Target Rate and state properties for the target data generation
beta_targ = torch.tensor([0.011, 0.016, 1. / 1.e1, 0.58])

# VV_tt history
NofTpts = 1500
theta0 = torch.tensor(1.)

# Multi data2
ones = 10 * [1.]
tens = 10 * [10.]

# Prescribed velocities - training
VVs = torch.tensor([ones + ones + tens + tens + ones + ones + tens + tens + ones + ones + ones + ones + ones + ones + ones, \
                    ones + ones + ones + ones + ones + ones + ones + tens + tens + tens + tens + tens + tens + tens + tens, \
                    ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones + ones, \
                    tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens + tens])

# Prescribed velocities - testing
VV_tests = torch.tensor([ones + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + tens + tens + ones + ones, \
                         ones + ones + tens + tens + tens + ones + ones + tens + tens + tens + tens + ones + ones + ones + ones])

# Times at which the velocities are prescribed
tts = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1]),
                   torch.linspace(0., 30., VVs.shape[1])])

# Times at which the velocities are prescribed - testing
tt_tests = torch.stack([torch.linspace(0., 30., VVs.shape[1]),
                        torch.linspace(0., 30., VVs.shape[1])])

VtFuncs = []
VtFunc_tests = []

# Functions for V-t interpolation
for VV, tt in zip(VVs, tts):
    VtFuncs.append(interp1d(tt, VV))

# Functions for V-t interpolation
for VV, tt in zip(VV_tests, tt_tests):
    VtFunc_tests.append(interp1d(tt, VV))

# Number of hidden variables, i.e. the dimension of xi
DimXi = 1
Xi0 = torch.zeros(DimXi)

# NN parameters for f = fStar + NN1(V, log(V), \xi, log(\xi))
f0 = 0.58
NN1_input_dim = 2 + 2 * DimXi
NN1s = [128, 512, 512, 128]
NN1_output_dim = 1

# NN parameters for \dot{\xi} = NN2(V, log(V), \xi, log(\xi))
NN2_input_dim = 2 + 2 * DimXi
NN2s = [128, 512, 512, 128]
NN2_output_dim = DimXi

# Store all the input parameters as a keyword dictionary
kwgs = {
    'VVs' : VVs, 
    'tts' : tts, 
    'VV_tests' : VV_tests,
    'tt_tests' : tt_tests, 
    'VtFuncs' : VtFuncs, 
    'VtFunc_tests' : VtFunc_tests, 
    'NofTpts' : NofTpts,
    'theta0' : theta0, 
    'Xi0' : Xi0, 
    'NN1_input_dim' : NN1_input_dim, 
    'NN1s' : NN1s, 
    'DimXi' : DimXi, 
    'NN1_output_dim' : NN1_output_dim, 
    'NN2_input_dim' : NN2_input_dim, 
    'NN2s' : NN2s, 
    'NN2_output_dim' : NN2_output_dim, 
    'f0' : f0, 
}

# Compute f history based on VtFunc and beta
def cal_f_beta(beta, kwgs, tts, VtFuncs, std_noise = 0.001):
    NofTpts = kwgs['NofTpts']
    theta0 = kwgs['theta0']

    # Get all sequences
    Vs = []
    thetas = []
    fs = []
    for tt, VtFunc in zip(tts, VtFuncs):
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
        mean_f = torch.mean(f);
        f = f + std_noise * mean_f * torch.randn(f.shape)
        Vs.append(V)
        thetas.append(theta)
        fs.append(f)
    
    Vs = torch.stack(Vs)
    thetas = torch.stack(thetas)
    fs = torch.stack(fs)

    return Vs, thetas, fs

def O(f, f_targ, t):
    return torch.trapezoid(
        torch.square(f - f_targ), t
    )


# Plot sequences of friction coefficient
def plotSequences(NN_model, kwgs, pwd):
    # Calculate NN_model on the train dataset
    fs = []
    for VtFunc in kwgs['VtFuncs']:
        fs.append(NN_model(VtFunc))
    
    f_targs = kwgs['f_targs']
    lws = torch.linspace(3.0, 1.0, len(fs))

    tts = kwgs['tts']
    NofTpts = kwgs['NofTpts']
    for idx, (tt, f_targ, f) in enumerate(zip(tts, f_targs, fs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f.detach().numpy(), linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        # plt.ylim([0.2, 1.])
        plt.title("Train sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "TrainSeq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, (t, V) in enumerate(zip(kwgs['tts'], kwgs['VVs'])):
        plt.plot(t, V, linewidth=lws[idx])
        lgd.append("TrainSeq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TrainSeqs.png", dpi = 300.)
    plt.close()

    # Calculate NN_model on the test dataset
    fs = []
    for VtFunc in kwgs['VtFunc_tests']:
        fs.append(NN_model(VtFunc))
    
    f_targs = kwgs['f_targ_tests']
    lws = torch.linspace(3.0, 1.0, len(fs))

    tts = kwgs['tt_tests']
    NofTpts = kwgs['NofTpts']
    for idx, (tt, f_targ, f) in enumerate(zip(tts, f_targs, fs)):
        t = torch.linspace(tt[0], tt[-1], NofTpts)
        plt.figure(figsize=[15, 10])
        plt.plot(t, f_targ, linewidth=2.0)
        plt.plot(t, f.detach().numpy(), linewidth=1.5)
        plt.legend(["Target", "Optimized"], fontsize=20, loc='best')
        plt.xlabel("t [s]", fontsize=20)
        plt.ylabel("Friction coefficient", fontsize=20)
        # plt.ylim([0.2, 1.])
        plt.title("Test sequence " + str(idx), fontsize=20)
        plt.savefig(pwd + "TestSeq_" + str(idx) + ".png", dpi = 300.)
        plt.close()

    # Plot the generating sequences
    plt.figure(figsize=[15, 10])
    lgd = []

    for idx, (t, V) in enumerate(zip(kwgs['tt_tests'], kwgs['VV_tests'])):
        plt.plot(t, V, linewidth=lws[idx])
        lgd.append("TestSeq " + str(idx))
    
    plt.legend(lgd, fontsize=20, loc='best')
    plt.xlabel("t [s]", fontsize=20)
    plt.ylabel("V [m/s]", fontsize=20)
    plt.savefig(pwd + "TestSeqs.png", dpi = 300.)
    plt.close()


## Invert on an problem
t = torch.linspace(tt[0], tt[-1], NofTpts)

# # Generate f_targs
# std_noise = 0.001
# V_targs, theta_targs, f_targs = cal_f_beta(beta_targ, kwgs, kwgs['tts'], kwgs['VtFuncs'], std_noise)
# V_targ_tests, theta_targ_tests, f_targ_tests = cal_f_beta(beta_targ, kwgs, kwgs['tt_tests'], kwgs['VtFunc_tests'], 0.)
# kwgs['f_targs'] = f_targs
# kwgs['f_targ_tests'] = f_targ_tests
# torch.save(kwgs, './data/RandnData1_std_1e-3_0504.pt')

# Load data
shit =  torch.load('./data/RandnData1_std_1e-3_0504.pt')
kwgs['f_targs'] = shit['f_targs']
kwgs['f_targ_tests'] = shit['f_targ_tests']

# print('V_targ: ', V_targ)
# print('theta_targ: ', theta_targ)
# print('f_targ: ', f_targ)

# Calculate model
NNModel = NN_computeF(kwgs);

# # Calculate the NNs 
# f_NNs = []
# for VtFunc in VtFuncs:
#     f_NNs.append(NNModel(VtFunc))

# print("f_NNs: ", f_NNs)

## Gradient descent:
max_epochs = 501
optimizer = optim.SGD(NNModel.parameters(), lr=0.001, momentum=0.9)

# Train NNModel
for epoch in range(max_epochs):
    print("=" * 40, " Epoch ", str(epoch + 1), " ", "=" * 40, flush=True)
    this_epoch_loss_sum = 0.
    for i, (VtFunc, f_targ) in enumerate(zip(kwgs['VtFuncs'], kwgs['f_targs'])):
        optimizer.zero_grad()
        f_pred = NNModel(VtFunc)
        loss = O(f_pred, f_targ, t)
        this_epoch_loss_sum += loss
        loss.backward()
        optimizer.step()
        # print("f_targ.shape: ", f_targ.shape)

    # Print this epoch training loss sum
    print("Sum of training loss in this epoch: ", this_epoch_loss_sum, flush=True)
    if (epoch % 10 == 0):
        this_epoch_test_loss = 0.
        for i, (VtFunc, f_targ) in enumerate(zip(kwgs['VtFunc_tests'], kwgs['f_targ_tests'])):
            f_pred = NNModel(VtFunc)
            loss = O(f_pred, f_targ, t)
            this_epoch_test_loss += loss

        # Print this epoch loss sum
        print("Testing loss in this epoch: ", this_epoch_test_loss, flush=True)

# Save a figure of the result
pwd ="./plots/Test0517_std_1e-3_NN128_512/"
Path(pwd).mkdir(parents=True, exist_ok=True)
plotSequences(NNModel, kwgs, pwd)