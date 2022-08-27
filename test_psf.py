from cmath import inf
import os
import logging
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.Unrolled_ADMM import Unrolled_ADMM
from models.Richard_Lucy import Richard_Lucy
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR, estimate_shear

def test_psf_shear_err(methods, shear_errs, n_iters, model_files, n_gal):   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logging.info(f'Tesing PSF with shear error: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results_psf_shear_err.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
            results['shear_errs'] = shear_errs
            results['rec_shear'] = {}
        rec_err_mean = []
        
        if n_iter > 0:
            if method == 'Richard-Lucy':
                model = Richard_Lucy(n_iters=n_iter)
                model.to(device)
            else:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
                try: # Load the model
                    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                    logging.info(f'Successfully loaded in {model_file}.')
                except:
                    logging.raiseExceptions(f'Failed loading in {model_file} model!')   
            model.eval()     
        
        for k, shear_err in enumerate(shear_errs):
            test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder=f'psf_shear_err{shear_err}/' if shear_err>0 else 'psf/')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
            rec_shear = []
            for idx, ((obs, psf, alpha), gt) in enumerate(test_loader):
                with torch.no_grad():
                    if method == 'No_deconv':
                        if k > 0:
                            rec_shear = obs_shear
                            break
                        gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        gt_shear.append(estimate_shear(gt))
                        obs_shear.append(estimate_shear(obs))
                        rec_shear.append(estimate_shear(obs))
                    elif method == 'FPFS':
                        psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        try:
                            rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                        except:
                            rec_shear.append(obs_shear[idx])
                    elif method == 'Richard-Lucy':
                        obs, psf = obs.to(device), psf.to(device)
                        rec = model(obs, psf) 
                        rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        # Calculate shear
                        rec_shear.append(estimate_shear(rec))
                    else:
                        obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                        rec = model(obs, psf, alpha) #*= alpha.view(1,1,1)
                        rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        # Calculate shear
                        rec_shear.append(estimate_shear(rec))
                logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
                    idx+1, len(test_loader),
                    gt_shear[idx][0], gt_shear[idx][1],
                    obs_shear[idx][0], obs_shear[idx][1],
                    rec_shear[idx][0], rec_shear[idx][1]))
                if idx > n_gal:
                    break
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
    
def test_psf_seeing_err(methods, seeing_errs, n_iters, model_files, n_gal):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gt_shear, obs_shear = [], []
    for method, model_file, n_iter in zip(methods, model_files, n_iters):
        logging.info(f'Tesing PSF with shear error: {method}')
        result_path = os.path.join('results/', method)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        results_file = os.path.join(result_path, 'results_psf_seeing_err.json')
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            results = {} # dictionary to record the test results
            results['seeing_errs'] = seeing_errs
            results['rec_shear'] = {}
        rec_err_mean = []
        
        if n_iter > 0:
            if method == 'Richard-Lucy':
                model = Richard_Lucy(n_iters=n_iter)
                model.to(device)
            else:
                model = Unrolled_ADMM(n_iters=n_iter, llh='Poisson', PnP=True)
                try: # Load the model
                    model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
                    logging.info(f'Successfully loaded in {model_file}.')
                except:
                    logging.raiseExceptions(f'Failed loading in {model_file} model!')   
            model.eval()  
        
        for k, seeing_err in enumerate(seeing_errs):
            test_dataset = Galaxy_Dataset(train=False, survey='LSST', I=23.5, psf_folder=f'psf_seeing_err{seeing_err}/' if seeing_err>0 else 'psf/')
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
                    elif method == 'FPFS':
                        psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        try:
                            rec_shear.append(estimate_shear(obs, psf, use_psf=True))
                        except:
                            rec_shear.append(obs_shear[idx])
                    elif method == 'Richard-Lucy':
                        obs, psf = obs.to(device), psf.to(device)
                        rec = model(obs, psf) 
                        rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        # Calculate shear
                        rec_shear.append(estimate_shear(rec))
                    else:
                        obs, psf, alpha = obs.to(device), psf.to(device), alpha.to(device)
                        rec = model(obs, psf, alpha) # rec *= alpha.view(1,1,1)
                        rec = rec.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
                        # Calculate shear
                        rec_shear.append(estimate_shear(rec))
                logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})'.format(
                    idx+1, len(test_loader),
                    gt_shear[idx][0], gt_shear[idx][1],
                    obs_shear[idx][0], obs_shear[idx][1],
                    rec_shear[idx][0], rec_shear[idx][1]))
                if idx > n_gal:
                    break
            results['rec_shear'][str(seeing_err)] = rec_shear
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
    
def plot_results(methods):
    """Draw line plot for systematic shear error in PSF vs shear estimation error."""
    color_list = ['tab:red', 'tab:olive', 'tab:purple', 'tab:blue', 'tab:cyan', 'tab:green', 'tab:orange']
    # Systematic shear error in PSF vs shear estimation error
    fig = plt.figure(figsize=(12,8))
    for method, color in zip(methods, color_list):
        result_path = os.path.join('results', method)
        results_file = os.path.join(result_path, 'results_psf_shear_err.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')

        shear_errs = results['shear_errs']
        rec_err_mean = np.array(results['rec_err_mean'])
        
        plt.plot(shear_errs, rec_err_mean[:,0], '-o', label='$g_1$, '+method, color=color)
        plt.plot(shear_errs, rec_err_mean[:,1], '--v', label='$g_2$, '+method, color=color)
    
    plt.xlabel('Shear Error($\Delta_{g_1}$, $\Delta_{g_2}$) in PSF', fontsize=12)
    plt.ylabel('Average shear estimated error', fontsize=12)
    plt.xlim([-0.01, 0.41])
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join('results', 'psf_shear_err.jpg'), bbox_inches='tight')
    plt.close()
    
    # Seeing error in PSF vs shear estimation error
    fig = plt.figure(figsize=(12,8))
    for method, color in zip(methods, color_list):
        result_path = os.path.join('results', method)
        results_file = os.path.join(result_path, 'results_psf_seeing_err.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        logging.info(f'Successfully loaded in {results_file}.')

        seeing_errs = results['seeing_errs']
        rec_err_mean = np.array(results['rec_err_mean'])
        
        plt.plot(seeing_errs, rec_err_mean[:,0], '-o', label='$g_1$, '+method, color=color)
        plt.plot(seeing_errs, rec_err_mean[:,1], '--v', label='$g_2$, '+method, color=color)
    
    plt.xlabel('Seeing Error in PSF (arcsec)', fontsize=12)
    plt.ylabel('Average shear estimated error', fontsize=12)
    plt.xlim([-0.01, 0.31])
    # plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.savefig(os.path.join('results', 'psf_seeing_err.jpg'), bbox_inches='tight')
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
    parser.add_argument('--n_gal', type=int, default=inf)
    opt = parser.parse_args()
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    
    methods = ['No_deconv', 'FPFS', 'Richard-Lucy', 'Unrolled_ADMM(1)', 'Unrolled_ADMM(2)', 
               'Unrolled_ADMM(4)', 'Unrolled_ADMM(8)']
    n_iters = [0, 0, 100, 1, 2, 4, 8]
    model_files = [None, None, None,
                   "saved_models/Poisson_PnP_1iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_2iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_4iters_LSST23.5_50epochs.pth",
                   "saved_models/Poisson_PnP_8iters_LSST23.5_50epochs.pth"]
                #    "saved_models/Poisson_PnP_12iters_LSST23.5_25epochs.pth"]
    shear_errs=[0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    seeing_errs=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    test_psf_shear_err(methods=methods, shear_errs=shear_errs, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal)
    test_psf_seeing_err(methods=methods, seeing_errs=seeing_errs, n_iters=n_iters, model_files=model_files, n_gal=opt.n_gal)
    # plot_results(methods=methods)