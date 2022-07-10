
from pickle import TRUE
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def loaddata(psf_file, img_file, f, show_im=True):
    psf = Image.open(psf_file)
    psf = np.array(psf, dtype='float32')
    obs = Image.open(img_file)
    obs = np.array(obs, dtype='float32')
    
    """Resize to a more manageable size to do reconstruction on. 
    Because resizing is downsampling, it is subject to aliasing 
    (artifacts produced by the periodic nature of sampling). Demosaicing is an attempt
    to account for/reduce the aliasing caused. In this application, we do the simplest
    possible demosaicing algorithm: smoothing/blurring the image with a box filter"""
    
    def resize(img, factor):
        num = int(-np.log2(factor))
        for i in range(num):
            img = 0.25*(img[::2,::2,...]+img[1::2,::2,...]+img[::2,1::2,...]+img[1::2,1::2,...])
        return img    
    
    def pad(img, pad_shape):
        img_pad = np.zeros((pad_shape[0], pad_shape[1]))
        starti = (pad_shape[0] - img.shape[0]) // 2
        endi = starti + img.shape[0]
        startj = (pad_shape[1] // 2) - (img.shape[1] // 2)
        endj = startj + img.shape[1]
        img_pad[starti:endi, startj:endj] = img

        return img_pad
    
    # Pad the images to ensure same sizes for psf and image
    if psf.shape[0] > obs.shape[0]:
        obs = pad(obs, psf.shape)
    elif psf.shape[0] < obs.shape[0]:
        psf = pad(psf, obs.shape)

    # """ nmormalizing copy from shreyas"""
    # psf /= np.linalg.norm(psf.ravel())
    # obs /= np.linalg.norm(obs.ravel())
    
    if show_im:
        fig1 = plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.title('PSF')
        plt.show()
        fig2 = plt.figure()
        plt.imshow(obs, cmap='gray')
        plt.title('Raw data')
        plt.show()
    return psf, obs

def initMatrices(h):
    pixel_start = (np.max(h) + np.min(h))/2
    x = np.ones(h.shape)*pixel_start

    init_shape = h.shape
    padded_shape = [nextPow2(2*n - 1) for n in init_shape]
    starti = (padded_shape[0]- init_shape[0])//2
    endi = starti + init_shape[0]
    startj = (padded_shape[1]//2) - (init_shape[1]//2)
    endj = startj + init_shape[1]
    hpad = np.zeros(padded_shape)
    hpad[starti:endi, startj:endj] = h

    H = fft.fft2(hpad, norm="ortho")
    Hadj = np.conj(H)

    def crop(X):
        return X[starti:endi, startj:endj]

    def pad(v):
        vpad = np.zeros(padded_shape).astype(np.complex64)
        vpad[starti:endi, startj:endj] = v
        return vpad

    utils = [crop, pad]
    v = np.real(pad(x))
    print(H.shape)
    return H, Hadj, v, utils

def nextPow2(n):
    return int(2**np.ceil(np.log2(n)))

def grad(Hadj, H, vk, b, crop, pad):
    Av = calcA(H, vk, crop)
    diff = Av - b
    return np.real(calcAHerm(Hadj, diff, pad))

def calcA(H, vk, crop):
    Vk = fft.fft2(vk, norm="ortho")
    return crop(fft.ifftshift(fft.ifft2(H*Vk, norm="ortho")))

def calcAHerm(Hadj, diff, pad):
    xpad = pad(diff)
    X = fft.fft2(xpad, norm="ortho")
    return fft.ifftshift(fft.ifft2(Hadj*X, norm="ortho"))


def grad_descent(h, b, n_iters, disp_pic=100):
    H, Hadj, v, utils = initMatrices(h)
    crop = utils[0]
    pad = utils[1]
        
    alpha = np.real(2/(np.max(Hadj * H)))
    iterations = 0

    def non_neg(xi):
        xi = np.maximum(xi,0)
        return xi

    #proj = lambda x: x #Do no projection
    proj = non_neg #Enforce nonnegativity at every gradient step. Comment out as needed.

    parent_var = [H, Hadj, b, crop, pad, alpha, proj]
    
    vk = v

    #### uncomment for Nesterov momentum update ####   
    p = 0
    mu = 0.9
    ################################################

    #### uncomment for FISTA update ################
    tk = 1
    xk = v
    ################################################
        
    for iterations in range(n_iters):   
        # uncomment for regular GD update
        # vk = gd_update(vk, parent_var)
        
        # uncomment for Nesterov momentum update 
        # vk, p = nesterov_update(vk, p, mu, parent_var)
        
        # uncomment for FISTA update
        vk, tk, xk = fista_update(vk, tk, xk, parent_var)

        if iterations % disp_pic == 0:
            print(iterations)
            image = proj(crop(vk))
            f = plt.figure(1)
            plt.imshow(image, cmap='gray')
            plt.title('Reconstruction after iteration {}'.format(iterations))
            plt.show()
    
    return proj(crop(vk)) 
    
def gd_update(vk, parent_var):
    H, Hadj, b, crop, pad, alpha, proj = parent_var
    
    gradient = grad(Hadj, H, vk, b, crop, pad)
    vk -= alpha*gradient
    vk = proj(vk)
    
    return vk    

def nesterov_update(vk, p, mu, parent_var):
    H, Hadj, b, crop, pad, alpha, proj = parent_var
    
    p_prev = p
    gradient = grad(Hadj, H, vk, b, crop, pad)
    p = mu*p - alpha*gradient
    vk += -mu*p_prev + (1+mu)*p
    vk = proj(vk)
    
    return vk, p

def fista_update(vk, tk, xk, parent_var):
    H, Hadj, b, crop, pad, alpha, proj = parent_var
    
    x_k1 = xk
    gradient = grad(Hadj, H, vk, b, crop, pad)
    vk -= alpha*gradient
    xk = proj(vk)
    t_k1 = (1+np.sqrt(1+4*tk**2))/2
    vk = xk+(tk-1)/t_k1*(xk - x_k1)
    tk = t_k1
    
    return vk, tk, xk


if __name__ == "__main__":
    ### Reading in params from config file (don't mess with parameter names!)
    # params = yaml.load(open("gd_config.yml"))
    # for k,v in params.items():
    #     exec(k + "=v")
    n_iters = 10000

    psf, data = loaddata(psf_file='./dataset/COSMOS_23.5/psf/psf_23.5_13.tiff',
                        img_file='./dataset/COSMOS_23.5/obs/obs_23.5_13.tiff',
                        f=1)
    final_im = grad_descent(psf, data, n_iters, disp_pic=500)
    
    plt.imshow(final_im, cmap='gray')
    plt.title('Final reconstruction after {} iterations'.format(n_iters))
    plt.show()
    saveim = input('Save final image? (y/n) ')
    if saveim == 'y':
        filename = input('Name of file: ')
        plt.imshow(final_im, cmap='gray')
        plt.axis('off')
        plt.savefig(filename+'.png', bbox_inches='tight')

