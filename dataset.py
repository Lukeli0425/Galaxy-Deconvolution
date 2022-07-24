import os
import json
import logging
import argparse
import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import galsim
from utils import PSNR

class Galaxy_Dataset(Dataset):
    """A galaxy image dataset generated with Galsim."""
    def __init__(self,  train=True, data_path='./dataset/', train_split = 0.7, n_total=0,
                        COSMOS_path='./data/', I=23.5, img_size=(48,48),
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

        self.I = I # I = 23.5 or 25.2 COSMOS data
        self.img_size = img_size
        self.gal_max_shear = gal_max_shear
        self.atmos_max_shear = atmos_max_shear
        self.pixel_scale = pixel_scale # arcsec
        self.seeing = seeing # arcsec (0.7 for LSST)
        
        # Create directory for the dataset
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.data_path = os.path.join(data_path, f'COSMOS_{self.I}')
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
            self.info = {'I':I, 'img_size':img_size, 'gal_max_shear':gal_max_shear, 'atmos_max_shear':atmos_max_shear, 'pixel_scale':pixel_scale, 'seeing':seeing}
            # Generate random sequence for data
            logging.warning(f'Failed reading information from {self.info_file}.')
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
            self.create_imgaes()

    def create_imgaes(self):
        """Generate and save real galaxy images with Galsim."""
        logging.info('Simulating real galaxy images.')
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
            tel_diam = 8.36                     # telescope diameter / meters (8.36 for LSST)
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
            
            # Simulated PSF (optical + atmospgeric)
            # Define the atmospheric component of PSF
            atmos = galsim.Kolmogorov(fwhm=atmos_fwhm, flux=1) # Note: the flux here is the default flux=1.
            atmos = atmos.shear(e=atmos_e, beta=atmos_beta*galsim.radians)
            # Define the optical component of PSF
            lam_over_diam = lam * 1.e-9 / tel_diam # radians
            lam_over_diam *= 206265  # arcsec
            optics = galsim.OpticalPSF( lam_over_diam,
                                        defocus = opt_defocus,
                                        coma1 = opt_c1, coma2 = opt_c2,
                                        astig1 = opt_a1, astig2 = opt_a2,
                                        obscuration = opt_obscuration,
                                        flux=1)
            
            psf = galsim.Convolve([atmos, optics], real_space=True)
            final = galsim.Convolve([psf, gal]) # Make the combined profile
            # Offset by up to 1/2 pixel in each direction
            dx = rng() - 0.5
            dy = rng() - 0.5

            # Dobs the profile
            obs = galsim.ImageF(self.img_size[0], self.img_size[1])
            final.drawImage(obs, scale=self.pixel_scale, offset=(dx,dy), method='auto')
            gal_image = galsim.ImageF(self.img_size[0], self.img_size[1])
            gal.drawImage(gal_image, scale=self.pixel_scale, offset=(dx,dy), method='auto')
            psf_image = galsim.ImageF(self.img_size[0]-1, self.img_size[1]-1)
            psf.drawImage(psf_image, scale=self.pixel_scale, offset=(dx,dy), method='auto')
            
            gal_image += sky_level * (self.pixel_scale**2)
            # gal_image.addNoise(galsim.PoissonNoise(rng)) # no noise for ground truth
            obs += sky_level * (self.pixel_scale**2) # Add a constant background level
            obs.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise)) # add noise for observation
            # obs.addNoise(galsim.PoissonNoise(rng))
            psf_image = psf_image*psf_flux + sky_level * (self.pixel_scale**2) # Add a constant background level
            psf_image.addNoise(galsim.CCDNoise(rng, gain=gain, read_noise=read_noise)) # add noise for observation
            # psf_image.addNoise(galsim.PoissonNoise(rng))

            # Save images
            psnr = PSNR(obs.array, gal_image.array)
            psnr_list.append(psnr)
            io.imsave(os.path.join(self.data_path, 'gt', f"gt_{self.I}_{k}.tiff"), gal_image.array, check_contrast=False)
            io.imsave(os.path.join(self.data_path, 'psf', f"psf_{self.I}_{k}.tiff"), psf_image.array, check_contrast=False)
            io.imsave(os.path.join(self.data_path, 'obs', f"obs_{self.I}_{k}.tiff"), obs.array, check_contrast=False)
            logging.info("Simulating Image:  [{:}/{:}]   PSNR={:.2f}".format(k+1,self.real_galaxy_catalog.nobjects, psnr))

            # Visualization
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(gal_ori_image.array)
            plt.title('Original Galaxy')
            plt.subplot(2,2,2)
            plt.imshow(gal_image.array)
            plt.title('Sheared Galaxy\n($e_1={:.3f}$, $e_2={:.3f}$)'.format(gal_g1, gal_g2))
            plt.subplot(2,2,3)
            plt.imshow(psf_image.array)
            plt.title('PSF\n($e_1={:.3f}$, $e_2={:.3f}$, FWHM={:.2f})'.format(atmos_g1, atmos_g2, atmos_fwhm))
            plt.subplot(2,2,4)
            plt.imshow(obs.array)
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
        psf = (psf - psf.min())/(psf.max() - psf.min())
        psf /= psf.sum()

        obs_path = os.path.join(self.data_path, 'obs')
        obs = torch.from_numpy(io.imread(os.path.join(obs_path, f"obs_{self.I}_{idx}.tiff"))).unsqueeze(0)
        obs = (obs - obs.min())/(obs.max() - obs.min())

        gt_path = os.path.join(self.data_path, 'gt')
        gt = torch.from_numpy(io.imread(os.path.join(gt_path, f"gt_{self.I}_{idx}.tiff"))).unsqueeze(0)
        gt = (gt - gt.min())/(gt.max() - gt.min())

        M = obs.ravel().mean()/0.33
        M = torch.Tensor(M).view(1,1,1)

        return (obs, psf, M), gt

def get_dataloader(train_test_split=0.857, batch_size=32):
    """Create dataloaders from Galaxy Dataset."""
    train_dataset = Galaxy_Dataset(train=True)
    train_size = int(train_test_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for dataset.')
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    dataset = Galaxy_Dataset(COSMOS_path='/mnt/WD6TB/tianaoli/', I=opt.I, data_path='./dataset/')