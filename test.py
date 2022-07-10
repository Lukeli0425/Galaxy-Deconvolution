import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import Adam, SGD
from dataset import Galaxy_Dataset
from models.network_p4ip import P4IP_Net
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR


def test_p4ip(n_iters=8, result_path='./results/', model_path='./saved models/model_20.pth'):
    """Test the model."""
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    test_dataset = Galaxy_Dataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = P4IP_Net(n_iters=n_iters)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    loss_fn = MultiScaleLoss()

    test_loss = 0.0
    obs_psnr = 0.0
    rec_psnr = 0.0
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
        plt.savefig(os.path.join(result_path, f"result_{idx}.jpg"), bbox_inches='tight')
        plt.close()
        
        obs_psnr += PSNR(gt, obs)
        rec_psnr += PSNR(gt, rec)
            
    print("PSNR: {:.2f} -> {:.2f}".format(obs_psnr/len(test_loader), rec_psnr/len(test_loader)))

    return

if __name__ =="__main__":
    test_p4ip()