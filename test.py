import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import torch
from torch.utils.data import DataLoader
from dataset import Galaxy_Dataset
from models.network_p4ip import P4IP_Net
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR, estimate_shear

class p4ip_deconvolver:
    """Wrapper class for P4IP deconvolution."""
    def __init__(self, model_path):
        self.model_path = model_path
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = P4IP_Net(n_iters=8)
        model.to(device)
        # Load the p4ip model
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        except:
            logging.raiseExceptions('Failed loading P4IP model!')

    def deconvolve(self, obs, psf):
        """Deconvolve PSF with P4IP model."""
        psf = torch.from_numpy(psf).unsqueeze(dim=0).unsqueeze(dim=0)
        obs = torch.from_numpy(obs).unsqueeze(dim=0).unsqueeze(dim=0)
        M = obs.ravel().mean()/0.33
        M = torch.Tensor(M).view(1,1,1,1)

        output = self.model(obs, psf, M)
        rec = output[-1].squeeze(dim=0).squeeze(dim=0).cpu().numpy()

        return rec

def test_p4ip(n_iters=8, result_path='./results/p4ip/', model_path='./saved_models/p4ip_10.pth', I=23.5):
    """Test the model."""    
    logging.info('Start testing p4ip model.')
    results = {} # dictionary to record the test results
    results_file = os.path.join(result_path, 'p4ip_results.json')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(os.path.join(result_path, 'rec')): # create directory for recovered image
        os.mkdir(os.path.join(result_path, 'rec'))
    if not os.path.exists(os.path.join(result_path, 'visualization')): # create directory for visualization
        os.mkdir(os.path.join(result_path, 'visualization'))
    
    test_dataset = Galaxy_Dataset(train=False, I=23.5)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = P4IP_Net(n_iters=n_iters)
    model.to(device)
    # Load the p4ip model
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    except:
        logging.raiseExceptions('Failed loading P4IP model!')
    
    loss_fn = MultiScaleLoss()

    test_loss = 0.0
    obs_psnr = []
    rec_psnr = []
    model.eval()
    for idx, ((obs, psf, M), gt) in enumerate(test_loader):
        with torch.no_grad():
            obs, psf, M, gt = obs.to(device), psf.to(device), M.to(device), gt.to(device)
            output = model(obs, psf, M)
            rec = output[-1]
            loss = loss_fn(gt, rec)
            test_loss += loss.item()
            rec *= M.view(1,1,1)
            
            gt = gt.squeeze(dim=0).squeeze(dim=0).cpu()
            psf = psf.squeeze(dim=0).squeeze(dim=0).cpu()
            obs = obs.squeeze(dim=0).squeeze(dim=0).cpu()
            rec = rec.squeeze(dim=0).squeeze(dim=0).cpu()
        
        # Save image
        io.imsave(os.path.join(result_path, 'rec', f"rec_{I}_{idx}.tiff"), rec.numpy(), check_contrast=False)
        
        # Visualization
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
        plt.savefig(os.path.join(result_path, 'visualization', f"p4ip_{I}_{idx}.jpg"), bbox_inches='tight')
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

    # Plot results
    with open(results_file, 'r') as f:
        results = json.load(f)
    obs_psnr = results['obs_psnr']
    rec_psnr = results['rec_psnr']
    plt.figure(figsize=(10,10))
    plt.plot([10,35],[10,35],'r') # plt y=x line
    plt.plot(obs_psnr, rec_psnr, '.')
    plt.title('PSNR of P4IP Test Results', fontsize=18)
    plt.xlabel('PSNR of Observed Galaxies', fontsize=16)
    plt.ylabel('PSNR of Recovered Galaxies', fontsize=16)
    plt.savefig(os.path.join(result_path, 'p4ip_psnr.jpg'), bbox_inches='tight')
    plt.close()

    return results

def test_shear(results_file='./results/p4ip/p4ip_results.json', I=23.5):
    """Estimate shear"""
    test_dataset = Galaxy_Dataset(train=False, I=I)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    gt_shear = []
    obs_shear = []
    rec_shear = []
    fpfs_shear = []
    with open(results_file, 'r') as f:
        results = json.load(f)

    for idx, ((obs, psf, M), gt) in enumerate(test_loader):
        with torch.no_grad():
            gt = gt.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            psf = psf.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            obs = obs.squeeze(dim=0).squeeze(dim=0).cpu().numpy()
            rec = io.imread(os.path.join('./results/p4ip/rec/', f"rec_{I}_{idx}.tiff"))

            # Calculate shear
            gt_shear.append(estimate_shear(gt))
            obs_shear.append(estimate_shear(obs))
            rec_shear.append(estimate_shear(rec))
            fpfs_shear.append(estimate_shear(obs, psf, use_psf=True))
            logging.info('Estimating shear: [{}/{}]  gt:({:.3f},{:.3f})  obs:({:.3f},{:.3f})  rec:({:.3f},{:.3f})  fpfs:({:.3f},{:.3f})'.format(
                idx, len(test_loader),
                gt_shear[-1][0], gt_shear[-1][1],
                obs_shear[-1][0], obs_shear[-1][1],
                rec_shear[-1][0], rec_shear[-1][1],
                fpfs_shear[-1][0], fpfs_shear[-1][1]
            ))
            
    gt_shear = np.array(gt_shear)
    obs_shear = np.array(obs_shear)
    rec_shear = np.array(rec_shear)
    fpfs_shear = np.array(fpfs_shear)
    obs_shear_err = np.sqrt(np.mean((obs_shear - gt_shear)**2, axis=0))
    rec_shear_err = np.sqrt(np.mean((rec_shear - gt_shear)**2, axis=0))
    fpfs_shear_err = np.sqrt(np.mean((fpfs_shear - gt_shear)**2, axis=0))
    logging.info('Shear error ({:.5f},{:.5f}) -> ({:.5f},{:.5f})   fpfs:({:.5f},{:.5f})'.format(
        obs_shear_err[0], obs_shear_err[1],
        rec_shear_err[0], rec_shear_err[1],
        fpfs_shear_err[0], fpfs_shear_err[1]
    ))

    # Save shear estimation
    results['gt_shear'] = gt_shear.tolist()
    results['obs_shear'] = obs_shear.tolist()
    results['rec_shear'] = rec_shear.tolist()
    results['fpfs_shear'] = fpfs_shear.tolist()
    results['obs_shear_err'] = obs_shear_err.tolist()
    results['rec_shear_err'] = rec_shear_err.tolist()
    results['fpfs_shear_err'] = fpfs_shear_err.tolist()
    with open(results_file, 'w') as f:
        json.dump(results, f)
    logging.info(f"Shear estimation results saved to {results_file}.")



if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)
    
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    # test_p4ip(n_iters=8, result_path='./results/p4ip/', model_path='./saved_models/p4ip_10.pth')
    test_shear()
    # a = np.array([(1,2), (3,4)])
    # b = np.array([(1,1), (1,1)])
    # print(a.tolist())
    # print(np.sqrt(np.mean(a, axis=0)))
