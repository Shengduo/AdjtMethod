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

# Generate VT using several constraints
class GenerateVT:
    # Constructor
    def __init__(self, kwgs):
        # Flag gives whether the series is truncated Fourier or simple 
        self.Vrange = kwgs['Vrange']
        self.Trange = kwgs['Trange']
        self.NofTpts = kwgs['NofTpts']
        self.flag = kwgs['flag']
        self.kwgs = kwgs

        self.VT = torch.zeros([2, self.NofTpts])

        ## ------------- Decide which function to use based on the flag -------------- ##
        # Random-generated simple
        if self.flag == "simple":
            self.VT = self.VTSimple()
        
        # Random-generated fourier
        elif self.flag == "fourier":
            self.VT = self.VTFourier()

        # Prescribed simple function
        elif self.flag == "prescribed_simple":
            self.VT = self.VTPrescribedSimple()
        
    # Simple function series
    def VTSimple(self):
        # Get all evaluation points
        T = torch.linspace(self.Trange[0], self.Trange[1], self.NofTpts)

        # Number of terms of simple functions
        nOfTerms = self.kwgs['nOfTerms']
        tt = self.Trange[0] + (self.Trange[1] - self.Trange[0]) * torch.sort(torch.rand(nOfTerms)).values
        tt[0] = self.Trange[0] 
        tt[-1] = self.Trange[1]

        # Generate the V values
        VV = self.Vrange[0] + (self.Vrange[1] - self.Vrange[0]) * torch.rand(nOfTerms)

        # Get the V terms evalute at T
        V = VV[0] * torch.ones(T.shape)
        for i in range(1, len(tt)):
            V[T >= tt[i]] = VV[i]
        
        # Update self.tt, self.VV
        self.tt = tt
        self.VV = VV

        return torch.stack([V, T])

    # Fourier function series
    def VTFourier(self):
        # Get all evaluation points
        T = torch.linspace(self.Trange[0], self.Trange[1], self.NofTpts)

        # Number of fourier terms
        nOfFourierTerms = self.kwgs['nOfFourierTerms']

        # Base amplitude
        base_amp = 0.5 * (self.Vrange[1] - self.Vrange[0])

        # Generate random amplitudes
        amps = base_amp * (1. + torch.rand(nOfFourierTerms))

        # Generate velocity
        V = 0.5 * (self.Vrange[1] - self.Vrange[0]) + torch.ones(T.shape)
        
        for i in range(len(amps)):
            V = V + amps[i] * torch.sin((i + 1) * torch.pi * T / self.Trange[1])

        # Clip V
        V = torch.clip(V, min=self.Vrange[0], max=self.Vrange[1])

        # Update self.tt, self.VV
        self.tt = 0.
        self.VV = 0.

        return torch.stack([V, T])
    
    def plotVT(self):
        # Plot V at t
        plt.figure(figsize=[15, 10])
        plt.plot(self.VT[1, :], self.VT[0, :], linewidth = 2.0)
        plt.xlabel('Time [s]', fontsize = 20)
        plt.ylabel('V [m/s]', fontsize = 20)
        plt.savefig(self.kwgs['plt_save_path'], dpi = 300.)

    # Prescribed simple function series
    def VTPrescribedSimple(self):
        # Get all evaluation points
        T = torch.linspace(self.Trange[0], self.Trange[1], self.NofTpts)

        # Number of terms of simple functions
        nOfTerms = self.kwgs['nOfTerms']
        # tt = self.Trange[0] + (self.Trange[1] - self.Trange[0]) * torch.sort(torch.rand(nOfTerms))
        tt = self.kwgs['tt']
        tt[0] = self.Trange[0] 
        tt[-1] = self.Trange[1]

        # Generate the V values
        # VV = self.Vrange[0] + (self.Vrange[1] - self.Vrange[0]) * torch.sort(torch.rand(nOfTerms))
        VV = self.kwgs['VV']

        # Get the V terms evalute at T
        V = VV[0] * torch.ones(T.shape)
        for i in range(1, len(tt)):
            V[T >= tt[i]] = VV[i]
        
        # Update self.tt, self.VV
        self.tt = tt
        self.VV = VV

        return torch.stack([V, T])

