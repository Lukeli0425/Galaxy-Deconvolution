import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.Unrolled_ADMM import Unrolled_ADMM
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR, estimate_shear
from scipy.stats import gaussian_kde

def test_psf_shear_err(shear_errs=[0.01,0.02,0.03,0.05,0.1,0.15,0.2,0.3]):
    methods = ['No_deconv', 'Fourier', 'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)', 'Unrolled_ADMM(12)']
    model_files = [None, None,
                   "saved_models/Poisson_PnP_4iters_LSST23.5_30epochs.pth",
                   "saved_models/Poisson_PnP_8iters_LSST23.5_30epochs.pth",
                   "saved_models/Poisson_PnP_12iters_LSST23.5_20epochs.pth"]
    n_iters = [None, None, 4, 8, 12]
    
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logging.info(f'Tesing PSF with shear error: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results_psf_shear_eer.json')
        
        results = {} # dictionary to record the test results
        results['shear_errs'] = shear_errs
        results['rec_shear'] = {}
        gt_shear, obs_shear = [], []
        rec_err_mean = []
        
        if not model_file == None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
            model.to(device)
            # Load the model
            model_file = f'saved_models/{method}.pth'
            try:
                model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                logging.info(f'Successfully loaded in {model_file}.')
            except:
                logging.raiseExceptions('Failed loading pretrained model!')   
            model.eval()     
        
        for k, shear_err in enumerate(shear_errs):
            test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder=f'psf_shear_err{shear_err}')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
            rec_shear = []
            for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
                with torch.no_grad():
                    if method == 'No_deconv':
                        if k>0:
                            rec_shear = obs_shear
                            break
                        gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        gt_shear.append(estimate_shear(gt))
                        obs_shear.append(estimate_shear(obs))
                        rec_shear.append(estimate_shear(obs))
                    elif method == 'Fourier':
                        psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                    else:
                        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                        rec = model(obs, psf, alpha) #* M.view(batch_size,1,1)
                        # rec *= alpha.view(1,1,1)
                        rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                    
                        # Calculate shear
                        rec_shear.append(estimate_shear(rec))
                
                logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
                    idx+1, len(test_loader),
                    gt_shear[idx][0], gt_shear[idx][1],
                    obs_shear[-1][0], obs_shear[-1][1],
                    rec_shear[-1][0], rec_shear[-1][1]))
            results['rec_shear'][str(shear_err)] = rec_shear
            gt_shear, rec_shear = np.array(gt_shear), np.array(rec_shear)
            rec_err_mean.append(np.mean(abs(rec_shear - gt_shear), axis=0))
        results['gt_shear'] = gt_shear.tolist()
        results['obs_shear'] = obs_shear
        results['rec_err_mean'] = np.array(rec_err_mean).tolist()
    
        # Save results to json file
        with open(results_file, 'w') as f:
            json.dump(results, f)
        logging.info(f"Test results saved to {results_file}.")
    
    return results
    
def plot_results(methods = ['No_deconv', 'Fourier', 'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)', 'Unrolled_ADMM(12)']):
    """Draw line plot for systematic shear error in PSF vs shear estimation error."""
    plt.figure(figsize=(10,8))
    for method in methods:
        result_path = os.path.join('results', method)
        results_file = os.path.join(result_path, 'results_psf_shear_eer.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')

        shear_errs = results['shear_errs']
        rec_err_mean = results['rec_err_mean']
        
        plt.plot(shear_errs, rec_err_mean, '-o', label=method)
    plt.xlim([0, 0.3])
    plt.ylim([0, 0.2])
    plt.legend()
    plt.savefig(os.path.join('results', 'psf_shear_err.jpg'))
    plt.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description='Arguments for tesing unrolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=4)
    parser.add_argument('--llh', type=str, default='Poisson', choices=['Poisson', 'Gaussian'])
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--survey', type=str, default='LSST', choices=['LSST', 'JWST'])
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
        
    test_psf_shear_err(n_iters=opt.n_iters, llh=opt.llh, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)
    plot_results(n_iters=opt.n_iters, llh=opt.llh, PnP=opt.PnP, n_epochs=opt.n_epochs, survey=opt.survey, I=opt.I)