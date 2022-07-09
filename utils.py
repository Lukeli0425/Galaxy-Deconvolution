import numpy as np
import math
import torch
import galsim

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

if __name__ == "__main__":
    a = np.array([[1,2],[3,4]])
    img1 = np.array([[0.5,1.5],[2.5,3.5]])
    img1 = (img1 - img1.min())/(img1.max() - img1.min())
    print(img1.sum())