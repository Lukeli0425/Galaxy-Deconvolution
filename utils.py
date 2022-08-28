import os
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import logging
from skimage import io
import torch
import galsim
import fpfs



def PSNR(img1, img2, normalize=True):
    """Calculate the PSNR of two images."""
    if not img1.shape == img2.shape:
        logging.raiseExceptions('Images have inconsistent Shapes!')

    img1 = np.array(img1)
    img2 = np.array(img2)

    if normalize:
        img1 = (img1 - img1.min())/(img1.max() - img1.min())
        img2 = (img2 - img2.min())/(img2.max() - img1.min())
        pixel_max = 1.0
    else:
        pixel_max = np.max([img1.max(), img2.max()])

    mse = ((img1 - img2)**2).mean()
    psnr = 20*np.log10(pixel_max/np.sqrt(mse))

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


def plot_loss(train_loss, val_loss, llh, PnP, n_iters, n_epochs, survey, I):
    n_epochs = len(train_loss)
    plt.figure(figsize=(12,7))
    plt.plot(range(1, n_epochs+1), train_loss, '-o', markersize=3.5, label='Train Loss')
    plt.plot(range(1, n_epochs+1), val_loss, '-o', markersize=3.5, label='Test Loss')
    plt.title('Loss Curve', fontsize=18)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    # plt.yscale("log")
    plt.legend(fontsize=15)
    file_name = f'./saved_models/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_loss_curve.jpg'
    plt.savefig(file_name)
    plt.close()

def plot_psnr(n_iters, llh, PnP, n_epochs, survey, I):
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')
    except:
        logging.raiseExceptions(f'Failed loading in {results_file}.')

    # Plot PSNR distribution
    try:
        obs_psnr, rec_psnr = np.array(results['obs_psnr']), np.array(results['rec_psnr'])
        plt.figure(figsize=(12,10))
        plt.plot([10,35],[10,35],'r') # plot y=x line
        xy = np.vstack([obs_psnr, rec_psnr])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = obs_psnr[idx], rec_psnr[idx], z[idx]
        plt.scatter(x, y, c=z, s=8, cmap='Spectral_r')
        plt.colorbar()
        plt.title('PSNR of Test Results', fontsize=18)
        plt.xlabel('PSNR of Observed Galaxies', fontsize=15)
        plt.ylabel('PSNR of Recovered Galaxies', fontsize=15)
        plt.savefig(os.path.join(result_path, 'psnr.jpg'), bbox_inches='tight')
        plt.close()
    except:
        logging.warning('No PSNR data found!')

def plot_shear_err(n_iters, llh, PnP, n_epochs, survey, I):
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')
    except:
        logging.raiseExceptions(f'Failed loading in {results_file}.')

    # Plot shear error density distribution
    gt_shear = np.array(results['gt_shear'])
    obs_shear = np.array(results['obs_shear'])
    rec_shear = np.array(results['rec_shear'])
    fpfs_shear = np.array(results['fpfs_shear'])
    plt.figure(figsize=(15,4.2))
    plt.subplot(1,3,1)
    x = (obs_shear - gt_shear)[:,0]
    y = (obs_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Observed Galaxy', fontsize=13)

    plt.subplot(1,3,2)
    x = (rec_shear - gt_shear)[:,0]
    y = (rec_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Recovered Galaxy', fontsize=13)

    plt.subplot(1,3,3)
    x = (fpfs_shear - gt_shear)[:,0]
    y = (fpfs_shear - gt_shear)[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=5, cmap='Spectral_r')
    plt.xlabel('$e_1$', fontsize=13)
    plt.ylabel('$e_2$', fontsize=13)
    plt.xlim([-0.8,0.8])
    plt.ylim([-0.8,0.8])
    plt.title('Fourier Power Spectrum Deconvolution', fontsize=13)
    plt.savefig(os.path.join(result_path, 'shear_err.jpg'), bbox_inches='tight')

if __name__ == "__main__":
    train_loss = [0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.082,0.079,0.072,0.062,0.051,0.047,0.039,0.035,0.032,0.029,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335]
    val_loss = [0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335,0.09,0.08,0.072,0.066,0.058,0.051,0.04,0.037,0.036,0.0335]
    plot_loss(train_loss, val_loss, 'Poisson', True, 11, True, 50, 23.5)
