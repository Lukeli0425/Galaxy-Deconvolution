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

class Richard_Lucy(nn.module):
    def __init__(self, n_iters):
        self.n_iters = n_iters
        
        
    def forward(self, x, psf):
        y = torch.ones_like(x)
        for i in range(self.n_iters):
            
        return y