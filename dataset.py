import os
import json
import logging
import argparse
import numpy as np
from skimage import io
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import galsim
import webbpsf
from utils import PSNR

# webbpsf.setup_logging('ERROR')

class Galaxy_Dataset(Dataset):
    """A galaxy image dataset generated with Galsim."""
    def __init__(self,  train=True, data_path='/mnt/WD6TB/tianaoli/dataset/', train_split = 0.7, n_total=0,
                        COSMOS_path='/mnt/WD6TB/tianaoli/', atmos=True, I=23.5, img_size=(48,48),
                        gal_max_shear=0.5, atmos_max_shear=0.2, 
                        pixel_scale=0.2, seeing=0.7):
        logging.info('Constructing dataset.')
        # Initialize parameters
        self.train= train # Using train data or test data
        self.COSMOS_dir = os.path.join(COSMOS_path, f"COSMOS_{I}_training_sample")
        self.train_split = train_split # n_train/n_total
        self.n_total = 0
        self.n_train = 0
        self.n_test = 0
        self.sequence = []
        self.info = {}

        self.atmos = atmos # ground-based or space-based
        self.I = I # I = 23.5 or 25.2 COSMOS data
        self.img_size = img_size
        self.gal_max_shear = gal_max_shear
        self.atmos_max_shear = atmos_max_shear
        self.pixel_scale = pixel_scale # arcsec
        self.seeing = seeing # arcsec (0.7 for LSST)
        
        # Create directory for the dataset
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.data_path = os.path.join(data_path, f'COSMOS_{self.I}_{"ground" if atmos else "space"}')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(os.path.join(self.data_path, 'obs')): # create directory for obs images
            os.mkdir(os.path.join(self.data_path, 'obs'))
        if not os.path.exists(os.path.join(self.data_path, 'gt')): # create directory for ground truth
            os.mkdir(os.path.join(self.data_path, 'gt'))
        if not os.path.exists(os.path.join(self.data_path, 'psf')): # create directory for PSF
            os.mkdir(os.path.join(self.data_path, 'psf'))
        if not os.path.exists(os.path.join(self.data_path, 'visualization')): # create directory for visualization
            os.mkdir(os.path.join(self.data_path, 'visualization'))

        # Read in real galaxy catalog
        try:
            self.real_galaxy_catalog = galsim.RealGalaxyCatalog(dir=self.COSMOS_dir, sample=str(self.I))
            self.n_total = self.real_galaxy_catalog.nobjects if n_total==0 else n_total
            logging.info(f'Successfully read in {self.n_total} real galaxies from {self.COSMOS_dir}.')
        except:
            logging.critical(f'Failed reading in real galaxies from {self.COSMOS_dir}.')

        # Read in information file
        self.info_file = os.path.join(self.data_path, f'info_{self.I}.json')
        try:
            logging.info(f'Successfully loaded in {self.info_file}.')
            with open(self.info_file, 'r') as f:
                self.info = json.load(f)
            self.n_total = self.info['n_total']
            self.n_train = self.info['n_train']
            self.n_test = self.info['n_test']
            self.sequence = self.info['sequence']
        except:
            self.info = {'atmos':atmos, 'I':I, 'img_size':img_size, 'gal_max_shear':gal_max_shear, 'atmos_max_shear':atmos_max_shear, 'pixel_scale':pixel_scale, 'seeing':seeing}
            # Generate random sequence for data
            logging.warning(f'Failed reading information from {self.info_file}.')
            

    def create_images(self):
        """Generate and save LSST images with Galsim."""
        logging.info('Simulating LSST images.')
        
        self.sequence = [i for i in range(self.n_total)]
        np.random.shuffle(self.sequence)
        n_train = int(self.train_split * self.n_total)
        self.info['n_total'] = self.n_total
        self.info['n_train'] = n_train
        self.info['n_test'] = self.real_galaxy_catalog.nobjects - n_train
        self.info['sequence'] = self.sequence
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)
        logging.info(f'Successfully created {self.info_file}.')
        
        random_seed = 425879
        psnr_list = []
        for k in range(self.n_total):
            idx = self.sequence[k] # index pf galaxy in the catalog
            # Galaxy parameters 
            rng = galsim.UniformDeviate(seed=random_seed+k+1) # Initialize the random number generator
            sky_level = 2.5e4                   # ADU / arcsec^2
            gal_flux = 1.e5                     # arbitrary choice, makes nice (not too) noisy images
            gal_e = rng() * self.gal_max_shear  # shear of galaxy
            gal_beta = 2. * np.pi * rng()       # radians
            gal_g1 = gal_e * np.cos(gal_beta)
            gal_g2 = gal_e * np.sin(gal_beta)
            gal_mu = 1 + rng() * 0.1            # mu = ((1-kappa)^2 - g1^2 - g2^2)^-1 (1.082)
            theta = 2. * np.pi * rng()          # radians
            # PSF parameters
            psf_flux = 1.e5
            if self.atmos:
                rng_gaussian = galsim.GaussianDeviate(seed=random_seed+k+1, mean=self.seeing, sigma=0.18)
                atmos_fwhm = 0 # arcsec (mean 0.7 for LSST)
                while atmos_fwhm < 0.35 or atmos_fwhm > 1.1: # sample fwhm
                    atmos_fwhm = rng_gaussian()
                # atmos_fwhm = self.seeing + (rng()-0.5)*0.25 # sample uniformly in [0.45, 0.95]
                atmos_e = rng() * self.atmos_max_shear # ellipticity of atmospheric PSF
                atmos_beta = 2. * np.pi * rng()     # radians
                atmos_g1 = atmos_e * np.cos(atmos_beta)
                atmos_g2 = atmos_e * np.sin(atmos_beta)
            opt_defocus = 0.3 + 0.4 * rng()     # wavelengths
            opt_a1 = 2*0.5*(rng() - 0.5)        # wavelengths (-0.29)
            opt_a2 = 2*0.5*(rng() - 0.5)        # wavelengths (0.12)
            opt_c1 = 2*1.*(rng() - 0.5)         # wavelengths (0.64)
            opt_c2 = 2*1.*(rng() - 0.5)         # wavelengths (-0.33)
            opt_obscuration = 0.165             # linear scale size of secondary mirror obscuration $(3.4/8.36)^2$
            lam = 700                           # nm    NB: don't use lambda - that's a reserved word.
            tel_diam = 8.36 if self.atmos else 6.5# telescope diameter / meters (8.36 for LSST, 6.5 for JWST)
            # psf_beta = 3
            # psf_trunc= 2 * atmos_fwhm
            # CCD parameters
            read_noise = 9. # e-
            gain = 0.34 # e-/ADU

            # Read out real galaxy from catalog
            gal_ori = galsim.RealGalaxy(self.real_galaxy_catalog, index = idx, flux = gal_flux)
            psf_ori = self.real_galaxy_catalog.getPSF(i=idx)
            gal_ori_image = self.real_galaxy_catalog.getGalImage(idx)
            psf_ori_image = self.real_galaxy_catalog.getPSFImage(idx)

            gal_ori = galsim.Convolve([psf_ori, gal_ori]) # concolve wth original PSF of HST
            gal = gal_ori.rotate(theta * galsim.radians) # Rotate by a random angle
            gal = gal.shear(e=gal_e, beta=gal_beta * galsim.radians) # Apply the desired shear
            gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.
            
            # Simulated PSF (optical + atmospheric)
            # Define the optical component of PSF
            lam_over_diam = lam * 1.e-9 / tel_diam # radians
            lam_over_diam *= 206265  # arcsec
            optics = galsim.OpticalPSF( lam_over_diam = lam_over_diam,
                                        defocus = opt_defocus,
                                        coma1 = opt_c1, coma2 = opt_c2,
                                        astig1 = opt_a1, astig2 = opt_a2,
                                        obscuration = opt_obscuration,
                                        flux=1)
            # Define the atmospheric component of PSF
            atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1) # Note: the flux here is the default flux=1.
            atmos = atmos.shear(e=atmos_e, beta=atmos_beta*galsim.radians)
            psf = galsim.Convolve([atmos, optics], real_space=True)
            
            psf_image = galsim.ImageF(self.img_size[0], self.img_size[1])
            psf.drawImage(psf_image, scale=self.pixel_scale, method='auto')
            psf_image = torch.from_numpy(psf_image.array)
            psf_image = torch.max(torch.zeros_like(psf_image), psf_image)
            # final = galsim.Convolve([psf, gal]) # Make the combined profile
            # Offset by up to 1/2 pixel in each direction
            dx = rng() - 0.5
            dy = rng() - 0.5

            # Dobs the profile
            gal_image = galsim.ImageF(self.img_size[0], self.img_size[1])
            gal.drawImage(gal_image, scale=self.pixel_scale, offset=(dx,dy), method='auto')
            gal_image += sky_level * (self.pixel_scale**2)
            gal_image = torch.from_numpy(gal_image.array)
            gal_image = torch.max(torch.zeros_like(gal_image), gal_image)
            
            
            # Concolve with PSF
            conv = ifftshift(ifft2(fft2(psf_image) * fft2(gal_image))).real
            conv = torch.max(torch.zeros_like(conv), conv) # set negative pixels to zero

            # Add CCD noise (Poisson + Gaussian)
            obs = torch.poisson(conv) + torch.normal(mean=torch.zeros_like(conv), std=torch.ones_like(conv))
            obs = torch.max(torch.zeros_like(obs), obs) # set negative pixels to zero
            
            # gal_image.addNoise(galsim.PoissonNoise(rng)) # no noise for ground truth
            # obs += sky_level * (self.pixel_scale**2) # Add a constant background level
            # obs.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise)) # add noise for observation
            # obs.addNoise(galsim.PoissonNoise(rng))
            # psf_image = psf_image*psf_flux + sky_level * (self.pixel_scale**2) # Add a constant background level
            # psf_image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise)) # add noise for observation
            # psf_image.addNoise(galsim.PoissonNoise(rng))

            # Save images
            psnr = PSNR(obs, gal_image)
            psnr_list.append(psnr)
            io.imsave(os.path.join(self.data_path, 'gt', f"gt_{self.I}_{k}.tiff"), np.array(gal_image), check_contrast=False)
            io.imsave(os.path.join(self.data_path, 'psf', f"psf_{self.I}_{k}.tiff"), np.array(psf_image), check_contrast=False)
            io.imsave(os.path.join(self.data_path, 'obs', f"obs_{self.I}_{k}.tiff"), np.array(obs), check_contrast=False)
            logging.info("Simulating Image:  [{:}/{:}]   PSNR={:.2f}".format(k+1,self.real_galaxy_catalog.nobjects, psnr))

            # Visualization
            if idx < 200:
                plt.figure(figsize=(10,10))
                plt.subplot(2,2,1)
                plt.imshow(gal_ori_image.array)
                plt.title('Original Galaxy')
                plt.subplot(2,2,2)
                plt.imshow(gal_image)
                plt.title('Simulated Galaxy\n($e_1={:.3f}$, $e_2={:.3f}$)'.format(gal_g1, gal_g2))
                plt.subplot(2,2,3)
                plt.imshow(psf_image)
                plt.title('PSF\n($e_1={:.3f}$, $e_2={:.3f}$, FWHM={:.2f})'.format(atmos_g1, atmos_g2, atmos_fwhm) if self.atmos else 'PSF')
                plt.subplot(2,2,4)
                plt.imshow(obs)
                plt.title('Observed Galaxy\n($PSNR={:.2f}$)'.format(psnr))
                plt.savefig(os.path.join(self.data_path, 'visualization', f"COSMOS_{self.I}_{k}.jpg"), bbox_inches='tight')
                plt.close()
        
        self.info['PSNR'] = psnr_list
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)

    def __len__(self):
        return self.n_train if self.train else self.n_test

    def __getitem__(self, i):
        idx = i if self.train else i + self.n_train
        
        psf_path = os.path.join(self.data_path, 'psf')
        psf = torch.from_numpy(io.imread(os.path.join(psf_path, f"psf_{self.I}_{idx}.tiff"))).unsqueeze(0)
        # psf = (psf - psf.min())/(psf.max() - psf.min())
        # psf /= psf.sum()

        obs_path = os.path.join(self.data_path, 'obs')
        obs = torch.from_numpy(io.imread(os.path.join(obs_path, f"obs_{self.I}_{idx}.tiff"))).unsqueeze(0)
        # obs = (obs - obs.min())/(obs.max() - obs.min())

        gt_path = os.path.join(self.data_path, 'gt')
        gt = torch.from_numpy(io.imread(os.path.join(gt_path, f"gt_{self.I}_{idx}.tiff"))).unsqueeze(0)
        # gt = (gt - gt.min())/(gt.max() - gt.min())

        M = obs.ravel().mean()
        M = torch.Tensor(M).view(1,1,1)

        return (obs, psf, M), gt


def get_Webb_PSF(insts=['NIRCam', 'NIRSpec','NIRISS', 'MIRI', 'FGS']):
    """Calculate all PSFs for given JWST instruments.

    Args:
        insts (list, optional): Instruments of JWST for PSF calculation. Defaults to ['NIRCam', 'NIRSpec','NIRISS', 'MIRI', 'FGS'].

    Returns:
        tuple: List of PSF names and dictionary containing PSF images.
    """
    psfs = dict() # all PSF images
    for instname in insts:
        inst = webbpsf.instrument(instname)
        filters = inst.filter_list
        for filter in filters:
            inst.filter = filter
            try:
                logging.info(f'Calculating Webb PSF: {instname} {filter}')
                psf_list = inst.calc_psf(fov_pixels=self.fov_pixels, oversample=1)
                psf = torch.from_numpy(psf_list[0].data)
                psf = torch.max(torch.zeros_like(psf), psf) # set negative pixels to zero
                psf /= psf.sum()
                psfs[instname+filter] = (psf, inst.pixelscale)
            except:
                pass
    psf_names = list(psfs.keys())

    return psf_names, psfs

def get_COSMOS_Galaxy(catalog, idx, gal_flux, sky_level, gal_e, gal_beta, theta, gal_mu, fov_pixels, pixel_scale, rng):

    # Read out real galaxy from catalog
    gal_ori = galsim.RealGalaxy(catalog, index = idx, flux = gal_flux)
    psf_ori = catalog.getPSF(i=idx)
    gal_ori_image = catalog.getGalImage(idx)
    psf_ori_image = catalog.getPSFImage(idx)

    gal_ori = galsim.Convolve([psf_ori, gal_ori]) # concolve wth original PSF of HST
    gal = gal_ori.rotate(theta * galsim.radians) # Rotate by a random angle
    gal = gal.shear(e=gal_e, beta=gal_beta * galsim.radians) # Apply the desired shear
    gal = gal.magnify(gal_mu) # Also apply a magnification mu = ( (1-kappa)^2 - |gamma|^2 )^-1, this conserves surface brightness, so it scales both the area and flux.
    
    # Offset by up to 1/2 pixel in each direction
    dx = rng() - 0.5
    dy = rng() - 0.5
    
    gal_image = galsim.ImageF(fov_pixels, fov_pixels)
    gal.drawImage(gal_image, scale=pixel_scale, offset=(dx,dy), method='auto')
    gal_image += sky_level * (pixel_scale**2)
    gal_image = torch.from_numpy(gal_image.array)
    gal_image = torch.max(torch.zeros_like(gal_image), gal_image)
    
    return gal_image, gal_ori_image.array


class JWST_Dataset(Dataset):
    """Simulated Galaxy Image Dataset inherited from torch.utils.data.Dataset."""
    def __init__(self, survey, I, fov_pixels, gal_max_shear, 
                train, train_split = 0.7, 
                data_path='/mnt/WD6TB/tianaoli/dataset/', 
                COSMOS_path='/mnt/WD6TB/tianaoli/'):
        """Construction function for the PyTorch Galaxy Dataset.

        Args:
            survey (str): The telescope to simulate, 'LSST' or 'JWST'.
            I (float): A keyword argument that can be used to specify the sample to use, "23.5" or "25.2".
            fov_pixels (int): Width of the simulated images in pixels.
            gal_max_shear (float): Maximum shear applied to galaxies.
            train (bool): Whether the dataset is generated for training or testing.
            train_split (float, optional): Proportion of data used in train dataset, the rest will be used in test dataset. Defaults to 0.7.
            data_path (str, optional): Directory to save the galaxy. Defaults to '/mnt/WD6TB/tianaoli/dataset/'.
            COSMOS_path (str, optional): Path to COSMOS data. Defaults to '/mnt/WD6TB/tianaoli/'.
        """
        
        logging.info('Constructing JWST dataset.')
        
        # Initialize parameters
        self.train= train # Using train data or test data
        self.COSMOS_dir = os.path.join(COSMOS_path, f"COSMOS_{I}_training_sample")
        self.train_split = train_split # n_train/n_total
        self.n_total = 0
        self.n_train = 0
        self.n_test = 0
        self.sequence = []
        self.info = {}

        self.survey = survey # LSST or JWST
        self.I = I # I = 23.5 or 25.2 COSMOS data
        self.fov_pixels = fov_pixels # numbers of pixels in FOV
        self.gal_max_shear = gal_max_shear
        
        # Create directory for the dataset
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.data_path = os.path.join(data_path, f'JWST_{self.I}')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if not os.path.exists(os.path.join(self.data_path, 'obs')): # create directory for obs images
            os.mkdir(os.path.join(self.data_path, 'obs'))
        if not os.path.exists(os.path.join(self.data_path, 'gt')): # create directory for ground truth
            os.mkdir(os.path.join(self.data_path, 'gt'))
        if not os.path.exists(os.path.join(self.data_path, 'psf')): # create directory for PSF
            os.mkdir(os.path.join(self.data_path, 'psf'))
        if not os.path.exists(os.path.join(self.data_path, 'visualization')): # create directory for visualization
            os.mkdir(os.path.join(self.data_path, 'visualization'))

        # Read in real galaxy catalog
        try:
            self.real_galaxy_catalog = galsim.RealGalaxyCatalog(dir=self.COSMOS_dir, sample=str(self.I))
            self.n_total = self.real_galaxy_catalog.nobjects
            logging.info(f'Successfully read in {self.n_total} real galaxies from {self.COSMOS_dir}.')
        except:
            logging.critical(f'Failed reading in real galaxies from {self.COSMOS_dir}.')

        # Read in information file
        self.info_file = os.path.join(self.data_path, f'info_{self.I}.json')
        try:
            logging.info(f'Successfully loaded in {self.info_file}.')
            with open(self.info_file, 'r') as f:
                self.info = json.load(f)
            self.n_total = self.info['n_total']
            self.n_train = self.info['n_train']
            self.n_test = self.info['n_test']
            self.sequence = self.info['sequence']
        except:
            self.info = {'fov_pixels':fov_pixels, 'gal_max_shear':gal_max_shear}
            logging.warning(f'Failed reading information from {self.info_file}.')
            

    def create_images(self):
        """Generate and save JWST images with Galsim and WebbPSF."""
        
        logging.info('Simulating JWST images.')
        
        # Generate random sequence for data
        self.sequence = [i for i in range(self.n_total)]
        np.random.shuffle(self.sequence)
        n_train = int(self.train_split * self.n_total)
        self.info['n_total'] = self.n_total
        self.info['n_train'] = n_train
        self.info['n_test'] = self.real_galaxy_catalog.nobjects - n_train
        self.info['sequence'] = self.sequence
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)
        logging.info(f'Successfully created {self.info_file}.')
        
        random_seed = 243
        psnr_list = []
        
        # Calculate all PSFs and split for train/test
        psf_names, psfs = get_Webb_PSF()
        np.random.shuffle(psf_names)
        train_psfs = psf_names[:int(len(psf_names) * self.train_split)]
        test_psfs = psf_names[int(len(psf_names) * self.train_split):]
        
        for k in range(self.n_total):
            idx = self.sequence[k] # index pf galaxy in the catalog
            
            # Choose a Webb PSF 
            psf_name = np.random.choice(train_psfs) if k < self.n_train else np.random.choice(test_psfs)
            psf_image, pixel_scale = psfs[psf_name]
            
            # Galaxy parameters 
            rng = galsim.UniformDeviate(seed=random_seed+k+1) # Initialize the random number generator
            sky_level = 2.5e4                   # ADU / arcsec^2
            gal_flux = 1.e5                     # arbitrary choice, makes nice (not too) noisy images
            gal_e = rng() * self.gal_max_shear  # shear of galaxy
            gal_beta = 2. * np.pi * rng()       # radians
            gal_g1 = gal_e * np.cos(gal_beta)
            gal_g2 = gal_e * np.sin(gal_beta)
            gal_mu = 1 + rng() * 0.1            # mu = ((1-kappa)^2 - g1^2 - g2^2)^-1 (1.082)
            theta = 2. * np.pi * rng()          # radians
            
            gal_image, gal_orig = get_COSMOS_Galaxy(catlog=self.real_galaxy_catalog, idx=idx, 
                                                    gal_flux=gal_flux, sky_level=sky_level, 
                                                    gal_e=gal_e, gal_beta=gal_beta, 
                                                    theta=theta, gal_mu=gal_mu, 
                                                    fov_pixels=self.fov_pixels, pixel_scale=pixel_scale, 
                                                    rng=rng)

            # Convolution via FFT
            conv = ifftshift(ifft2(fft2(psf_image) * fft2(gal_image))).real
            conv = torch.max(torch.zeros_like(conv), conv) # set negative pixels to zero

            # Add CCD noise (Poisson + Gaussian)
            obs = torch.poisson(conv) + torch.normal(mean=torch.zeros_like(conv), std=torch.ones_like(conv))
            obs = torch.max(torch.zeros_like(obs), obs) # set negative pixels to zero
            
            # Save images
            psnr = PSNR(obs, gal_image)
            psnr_list.append(psnr)
            torch.save(gal_image.clone(), os.path.join(self.data_path, 'gt', f"gt_{self.I}_{k}.pth"))
            torch.save(psf_image.clone(), os.path.join(self.data_path, 'psf', f"psf_{self.I}_{k}.pth"))
            torch.save(obs.clone(), os.path.join(self.data_path, 'obs', f"obs_{self.I}_{k}.pth"))
            logging.info("Simulating Image:  [{:}/{:}]   PSNR={:.2f}".format(k+1, self.n_total, psnr))

            # Visualization
            if idx < 200:
                plt.figure(figsize=(10,10))
                plt.subplot(2,2,1)
                plt.imshow(gal_orig)
                plt.title('Original Galaxy')
                plt.subplot(2,2,2)
                plt.imshow(gal_image)
                plt.title('Simulated Galaxy\n($e_1={:.3f}$, $e_2={:.3f}$)'.format(gal_g1, gal_g2))
                plt.subplot(2,2,3)
                plt.imshow(psf_image)
                plt.title(f'PSF: {psf_name}')
                plt.subplot(2,2,4)
                plt.imshow(obs)
                plt.title('Observed Galaxy\n($PSNR={:.2f}$)'.format(psnr))
                plt.savefig(os.path.join(self.data_path, 'visualization', f"COSMOS_{self.I}_{k}.jpg"), bbox_inches='tight')
                plt.close()
        
        self.info['PSNR'] = psnr_list
        with open(self.info_file, 'w') as f:
            json.dump(self.info, f)

    def __len__(self):
        return self.n_train if self.train else self.n_test

    def __getitem__(self, i):
        idx = i if self.train else i + self.n_train
        
        psf_path = os.path.join(self.data_path, 'psf')
        psf = torch.load(os.path.join(psf_path, f"psf_{self.I}_{idx}.pth")).unsqueeze(0)

        obs_path = os.path.join(self.data_path, 'obs')
        obs = torch.load(os.path.join(obs_path, f"obs_{self.I}_{idx}.pth")).unsqueeze(0)

        gt_path = os.path.join(self.data_path, 'gt')
        gt = torch.load(os.path.join(gt_path, f"gt_{self.I}_{idx}.pth")).unsqueeze(0)

        M = obs.ravel().mean().float()
        M = torch.Tensor(M).view(1,1,1)

        return (obs, psf, M), gt/M
            
            

def get_dataloader(survey='LSST', I=23.5, train_test_split=0.857, batch_size=32):
    """Generate dataloaders from Galaxy Dataset.

    Args:
        survey (str, optional): The survey of the dataset. Defaults to 'LSST'.
        I (float, optional): _description_. Defaults to 23.5.
        train_test_split (float, optional): Proportion of data used in train dataloader in train dataset, the rest will be used in valid dataloader. Defaults to 0.857.
        batch_size (int, optional): Batch size for training dataset. Defaults to 32.

    Returns:
        train_loader:  PyTorch data loader for train dataset.
        val_loader: PyTorch data loader for valid dataset.
    """
    if survey == 'LSST':
        train_dataset = Galaxy_Dataset(train=True, I=I, data_path='/mnt/WD6TB/tianaoli/dataset/')
    elif survey == 'JWST':
        train_dataset = JWST_Dataset(train=True, I=I, fov_pixels=64)
    
    train_size = int(train_test_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for dataset.')
    parser.add_argument('--survey', type=str, default='JWST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    if opt.survey == 'LSST':
        LSST_Dataset = Galaxy_Dataset(atmos=True, I=opt.I, pixel_scale=0.2)
        LSST_Dataset.create_images()
    elif opt.survey == 'JWST':
        JWST_Dataset = JWST_Dataset(I=opt.I, fov_pixels=64)
        JWST_Dataset.create_images()