import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

def test_psf_shear_err(n_iters, llh, PnP, n_epochs, survey, I, 
                       shear_errs=[0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.3]):
    logging.info(f'Testing unrolled {"PnP-" if PnP else ""}ADMM model with {llh} likelihood on {survey} data with PSF shear error.')
    results = {} # dictionary to record the test results
    results['shear_errs'] = shear_errs
    result_path = f'./results/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs/'
    results_file = os.path.join(result_path, 'results_psf_shear_eer.json')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(os.path.join(result_path, 'visualization')): # create directory for visualization
        os.mkdir(os.path.join(result_path, 'visualization'))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unrolled_ADMM(n_iters=n_iters, llh=llh, PnP=PnP)
    model.to(device)
    # Load the model
    model_file = f'saved_models/{llh}{"_PnP" if PnP else ""}_{n_iters}iters_{survey}{I}_{n_epochs}epochs.pth'
    try:
        model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
        logging.info(f'Successfully loaded in {model_file}.')
    except:
        logging.raiseExceptions('Failed loading pretrained model!')
    loss_fn = MultiScaleLoss()
    
    for k, shear_err in enumerate(shear_errs):
        test_dataset = Galaxy_Dataset(train=False, survey=survey, I=I, psf_folder=f'psf_shear_err{shear_err}')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
        obs_psnr, rec_psnr = [], []
        gt_shear, obs_shear, rec_shear = [], [], []
        rec_err_mean = []
        model.eval()
        for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
            with torch.no_grad():
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                loss = loss_fn(gt.squeeze(dim=1), rec.squeeze(dim=1))
                rec *= alpha.view(1,1,1)
                
                gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            
            # Calculate PSNR & shear
            if k == 0:
                obs_psnr.append(PSNR(gt, obs))
                gt_shear.append(estimate_shear(gt))
                obs_shear.append(estimate_shear(obs))
            rec_psnr.append(PSNR(gt, rec))
            rec_shear.append(estimate_shear(rec))
            
            logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
                idx+1, len(test_loader),
                gt_shear[-1][0], gt_shear[-1][1],
                obs_shear[-1][0], obs_shear[-1][1],
                rec_shear[-1][0], rec_shear[-1][1]))
        
        if k == 0:
            results['gt_shear'] = gt_shear
            results['obs_shear'] = obs_shear
            results['obs_psnr'] = obs_psnr
        results['rec_shear'][str(shear_err)] = rec_shear
        results['rec_psnr'][str(shear_err)] = rec_psnr
        gt_shear, rec_shear = np.array(gt_shear), np.array(rec_shear)
        rec_err_mean.append(np.mean(abs(rec_shear - gt_shear), axis=0))
    results['rec_err_mean'] = rec_err_mean
    
    # Save results to json file
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Test results saved to {results_file}.")
    
    return results
    
def plot_results(shear_errs=[0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.3]):
    # Line plot for systematic shear error in PSF vs shear estimation error
    result_path = f'./results/'
    with open(results_file, 'r') as f:
        results = json.load(f)
    results_file = os.path.join(result_path, 'results_psf_shear_eer.json')
    logging.info(f'Successfully loaded in {results_file}.')

    shear_errs = results['shear_errs']
    rec_err_mean = results['rec_err_mean']
    plt.figure(figsize=(10,8))
    plt.plot(shear_errs, rec_err_mean)

    plt.legend()
    plt.savefig(os.pth.join(result_path, 'psf_shear_err.jpg'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for tesing unrolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--llh', type=str, default='Poisson', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
        
    test_psf_shear_err(n_iters=opt.n_iters, llh=opt.llh, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
    plot_results()