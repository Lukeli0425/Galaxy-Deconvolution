import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
import galsim
import fpfs

def PSNR(img1, img2, normalize=True):
    """Calculate the PSNR of two images."""
    if not img1.shape == img2.shape:
        raise('Images have inconsistent Shapes!')

    img1 = np.array(img1)
    img2 = np.array(img2)

    if normalize:
        img1 = (img1 - img1.min())/(img1.max() - img1.min())
        img2 = (img2 - img2.min())/(img2.max() - img1.min())
        PIXEL_MAX = 1.0
    else:
        PIXEL_MAX = np.max([img1.max(), img2.max()])

    MSE = ((img1 - img2)**2).mean()
    psnr = 20*np.log10(PIXEL_MAX/np.sqrt(MSE))

    return psnr


def estimate_shear(obs, psf=None, use_psf=False):
    """Estimate shear from input 2D image."""
    if not use_psf: # Use delta for PSF if not given, equivalent to no deconvolution
        psf = np.zeros(obs.shape)
        psf[np.int(obs.shape[0]/2)+1, np.int(obs.shape[1]/2)+1] = 1
    else: # Crop out PSF
        # beg = psf.shape[0]//2 - rcut
        # end = beg + 2*rcut + 1
        # psf = psf[beg:end,beg:end]
        psf_pad = np.zeros((obs.shape[0], obs.shape[1]))
        starti = (obs.shape[0] - psf.shape[0]) // 2
        endi = starti + psf.shape[0]
        startj = (obs.shape[1] // 2) - (psf.shape[1] // 2)
        endj = startj + psf.shape[1]
        psf_pad[starti:endi, startj:endj] = psf
        psf = psf_pad

    fpTask = fpfs.fpfsBase.fpfsTask(psf, beta=0.75)
    modes = fpTask.measure(obs)
    ells = fpfs.fpfsBase.fpfsM2E(modes, const=1, noirev=False)
    resp = ells['fpfs_RE'][0]
    g_1 = ells['fpfs_e1'][0] / resp
    g_2 = ells['fpfs_e2'][0] / resp

    return (g_1, g_2) 


def plot_loss(train_loss, test_loss):
    n_epochs = len(train_loss)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=4, label='Train Loss')
    plt.plot(range(1, n_epochs+1), test_loss, '-o', markersize=4, label='Test Loss')
    plt.title('Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=15)
    plt.show()

if __name__ == "__main__":
    train_loss = [0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029]
    test_loss = [0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335]
    plot_loss(train_loss, test_loss)
