import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.Unrolled_ADMM import Unrolled_ADMM
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR, estimate_shear
from scipy.stats import gaussian_kde

class ADMM_deconvolver:
    """Wrapper class for unrolled ADMM deconvolution."""
    def __init__(self, n_iters=8, poisson=True, PnP=False, model_file=None):
        self.model_file = model_file
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Unrolled_ADMM(n_iters=n_iters, poisson=poisson, PnP=PnP)
        self.model.to(self.device)
        # Load pretrained model
        try:
            self.model.load_state_dict(torch.load(model_file, map_location=torch.device(self.device)))
            logging.info(f'Successfully loaded in {model_file}.')
        except:
            logging.raiseExceptions(f'Failed loading {model_file}!')

    def deconvolve(self, obs, psf):
        """Deconvolve PSF with unrolled ADMM model."""
        psf = torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0) if type(psf) is np.ndarray else psf
        obs = torch.from_numpy(obs).unsqueeze(dim=0).unsqueeze(dim=0) if type(obs) is np.ndarray else obs
        alpha = obs.ravel().mean()/0.33
        alpha = torch.Tensor(alpha.float()).view(1,1,1,1)

        output = self.model(obs.to(self.device), psf.to(self.device), alpha.to(self.device))
        rec = (output.cpu() * alpha.cpu()).squeeze(dim=0).squeeze(dim=0).numpy()

        return rec

def test(n_iters, poisson, PnP, n_epochs, survey, I):
    """Test the model."""     
    logging.info(f'Start testing unrolled {"PnP-" if PnP else ""}ADMM model with {"Poisson" if poisson else "Gaussian"} likelihood on {survey} data.')
    results = {} # dictionary to record the test results
    result_path = f'./results/{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(os.path.join(result_path, 'rec')): # create directory for recovered image
        os.mkdir(os.path.join(result_path, 'rec'))
    if not os.path.exists(os.path.join(result_path, 'visualization')): # create directory for visualization
        os.mkdir(os.path.join(result_path, 'visualization'))
    
    test_dataset = Galaxy_Dataset(train=False, survey=survey, I=I)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unrolled_ADMM(n_iters=n_iters, poisson=poisson, PnP=PnP)
    model.to(device)
    
    # Load the model
    model_file = f'saved_models/{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs.pth'
    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
    except:
        logging.raiseExceptions('Failed loading pretrained model!')
    
    loss_fn = MultiScaleLoss()

    test_loss = 0.0
    obs_psnr = []
    rec_psnr = []
    model.eval()
    for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
        with torch.no_grad():
            obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
            rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
            loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
            test_loss += loss.item()
            rec *= alpha.view(1,1,1)
            
            gt = gt.squeeze(dim=0).squeeze(dim=0).cpu()
            psf = psf.squeeze(dim=0).squeeze(dim=0).cpu()
            obs = obs.squeeze(dim=0).squeeze(dim=0).cpu()
            rec = rec.squeeze(dim=0).squeeze(dim=0).cpu()
        
        # Save image
        # io.imsave(os.path.join(result_path, 'rec', f"rec_{I}_{idx}.tiff"), rec.numpy(), check_contrast=False)
        
        # Visualization
        if idx < 100:
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.imshow(gt)
            plt.title('Sheared Galaxy')
            plt.subplot(2,2,2)
            plt.imshow(psf)
            plt.title('PSF')
            plt.subplot(2,2,3)
            plt.imshow(obs)
            plt.title('Observed Galaxy\n($PSNR={:.2f}$)'.format(PSNR(gt, obs)))
            plt.subplot(2,2,4)
            plt.imshow(rec)
            plt.title('Recovered Galaxy\n($PSNR={:.2f}$)'.format(PSNR(gt, rec)))
            plt.savefig(os.path.join(result_path, 'visualization', f"vis_{I}_{idx}.jpg"), bbox_inches='tight')
            plt.close()
        
        obs_psnr.append(PSNR(gt, obs))
        rec_psnr.append(PSNR(gt, rec))
        
        logging.info("Testing Image:  [{:}/{:}]  loss={:.4f}  PSNR: {:.2f} -> {:.2f}".format(
                        idx+1, len(test_loader), 
                        loss.item(), PSNR(gt, obs), PSNR(gt, rec)))
        
    logging.info("test_loss={:.4f}  PSNR: {:.2f} -> {:.2f}".format(
                    test_loss/len(test_loader),
                    np.mean(obs_psnr), np.mean(rec_psnr)))
        
    # Save results to json file
    results['test_loss'] = test_loss/len(test_loader)
    results['obs_psnr_mean'] = np.mean(obs_psnr)
    results['rec_psnr_mean'] = np.mean(rec_psnr)
    results['obs_psnr'] = obs_psnr
    results['rec_psnr'] = rec_psnr
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Test results saved to {results_file}.")

    return results

def test_shear(n_iters, poisson, PnP, n_epochs, survey, I):
    """Estimate shear with saved model."""
    test_dataset = Galaxy_Dataset(train=False, survey=survey, I=I)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    gt_shear = []
    obs_shear = []
    rec_shear = []
    fpfs_shear = []
    
    result_path = f'./results/{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results.json')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')
    except:
        logging.warning(f'Failed loading in {results_file}.')
        results = {}

    model_file = f'saved_models/{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs.pth'
    model = ADMM_deconvolver(n_iters=n_iters, poisson=poisson, PnP=PnP, model_file=model_file)
    
    for idx, ((obs, psf, M), gt) in enumerate(test_loader):
        with torch.no_grad():
            try:
                rec = io.imread(os.path.join(result_path, 'rec/', f"rec_{I}_{idx}.tiff"))
            except:
                rec = model.deconvolve(obs, psf)
            gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()

            # Calculate shear
            try:
                fpfs_shear.append(estimate_shear(obs, psf, use_psf=True))
                if abs(fpfs_shear[-1][0]) + abs(fpfs_shear[-1][1]) > 10:
                    fpfs_shear.pop()
                else:
                    gt_shear.append(estimate_shear(gt))
                    obs_shear.append(estimate_shear(obs))
                    rec_shear.append(estimate_shear(rec))
                    logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})  fpfs:({:.3f},{:.3f})'.format(
                        idx+1, len(test_loader),
                        gt_shear[-1][0], gt_shear[-1][1],
                        obs_shear[-1][0], obs_shear[-1][1],
                        rec_shear[-1][0], rec_shear[-1][1],
                        fpfs_shear[-1][0], fpfs_shear[-1][1]))
            except:
                pass
    gt_shear = np.array(gt_shear)
    obs_shear = np.array(obs_shear)
    rec_shear = np.array(rec_shear)
    fpfs_shear = np.array(fpfs_shear)
    obs_err_mean = np.mean(abs(obs_shear - gt_shear), axis=0)
    rec_err_mean = np.mean(abs(rec_shear - gt_shear), axis=0)
    fpfs_err_mean = np.mean(abs(fpfs_shear - gt_shear), axis=0)
    obs_err_median = np.median(abs(obs_shear - gt_shear), axis=0)
    rec_err_median = np.median(abs(rec_shear - gt_shear), axis=0)
    fpfs_err_median = np.median(abs(fpfs_shear - gt_shear), axis=0)
    obs_err_rms = np.sqrt(np.mean((obs_shear - gt_shear)**2, axis=0))
    rec_err_rms = np.sqrt(np.mean((rec_shear - gt_shear)**2, axis=0))
    fpfs_err_rms = np.sqrt(np.mean((fpfs_shear - gt_shear)**2, axis=0))
    logging.info('Shear error (mean): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
                obs_err_mean[0], obs_err_mean[1],
                rec_err_mean[0], rec_err_mean[1],
                fpfs_err_mean[0], fpfs_err_mean[1]))
    logging.info('Shear error (median): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
                obs_err_median[0], obs_err_median[1],
                rec_err_median[0], rec_err_median[1],
                fpfs_err_median[0], fpfs_err_median[1]))
    logging.info('Shear error (RMS): ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
                obs_err_rms[0], obs_err_rms[1],
                rec_err_rms[0], rec_err_rms[1],
                fpfs_err_rms[0], fpfs_err_rms[1]))
    
    # Save shear estimation
    results['gt_shear'] = gt_shear.tolist()
    results['obs_shear'] = obs_shear.tolist()
    results['rec_shear'] = rec_shear.tolist()
    results['fpfs_shear'] = fpfs_shear.tolist()
    results['obs_err_mean'] = obs_err_mean.tolist()
    results['rec_err_mean'] = rec_err_mean.tolist()
    results['fpfs_err_mean'] = fpfs_err_mean.tolist()
    results['obs_err_median'] = obs_err_median.tolist()
    results['rec_err_median'] = rec_err_median.tolist()
    results['fpfs_err_median'] = fpfs_err_median.tolist()
    results['obs_err_rms'] = obs_err_rms.tolist()
    results['rec_err_rms'] = rec_err_rms.tolist()
    results['fpfs_err_rms'] = fpfs_err_rms.tolist()
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Shear estimation results saved to {results_file}.")
    
    return results

def plot_results(n_iters, poisson, PnP, n_epochs, survey, I):
    result_path = f'./results/{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
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




if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for tesing unrolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--poisson', type=bool, default=True)
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=8)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    
    # test(n_iters=opt.n_iters, poisson=opt.poisson, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
    test_shear(n_iters=opt.n_iters, poisson=opt.poisson, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
    plot_results(n_iters=opt.n_iters, poisson=opt.poisson, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
