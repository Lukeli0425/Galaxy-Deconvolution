import os
import json
import logging
import argparse
from tkinter import YES
import numpy as np
from skimage import io
import torch
import torch.nn as nn
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from utils_poisson_deblurring.utils_torch import conv_fft_batch, psf_to_otf

class Richard_Lucy(nn.Module):
    def __init__(self, n_iters):
        super(Richard_Lucy, self).__init__()
        self.n_iters = n_iters
        
    def forward(self, y, psf):
        psf = psf/psf.sum() # normalize PSF
        ones = torch.ones_like(y)
        _, H = psf_to_otf(psf, y.size())
        Ht = torch.conj(H)
        x = torch.ones_like(y) # initial guess
        for i in range(self.n_iters):
            Hx = conv_fft_batch(H, x)
            numerator = conv_fft_batch(Ht, y/Hx)
            divisor = conv_fft_batch(H, ones)
            x = x*numerator/divisor
        return x
    
if __name__ == "__main__":
    model = Richard_Lucy()
