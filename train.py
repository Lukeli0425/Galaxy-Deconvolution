import os
import json
import logging
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim import Adam, SGD
from dataset import Galaxy_Dataset
from models.network_p4ip import P4IP_Net
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import PSNR

def get_dataloader(train_test_split=0.7, batch_size=1):
    """Generate dataloaders."""
    full_dataset = Galaxy_Dataset(train=True)
    train_size = int(train_test_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train(n_epochs=10, n_iters=8, lr=1e-4, train_test_split=0.7, batch_size=32,
          model_save_path='./saved_models/', load_pretrain=True):
    """Train model"""
    train_loader, test_loader = get_dataloader(train_test_split=train_test_split, batch_size=batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = P4IP_Net(n_iters=n_iters)
    model.to(device)
    MODEL_FILE = './poisson-deblurring/model_zoo/p4ip_100epoch.pth'
    if load_pretrain:
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device(device)))

    optimizer = Adam(params=model.parameters(), lr = lr)
    loss_fn = MultiScaleLoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_psnr = 0.0
        for step, ((obs, psf, M), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, M, gt = obs.to(device), psf.to(device), M.to(device), gt.to(device)
            output = model(obs, psf, M)
            rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
            loss = loss_fn(gt.squeeze(dim=1), rec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # evaluate on test dataset
            if (step+1) % 10 == 0:
                test_loss = 0.0
                test_psnr = 0.0
                model.eval()
                optimizer.zero_grad()
                for _, ((obs, psf, M), gt) in enumerate(test_loader):
                    with torch.no_grad():
                        obs, psf, M, gt = obs.to(device), psf.to(device), M.to(device), gt.to(device)
                        output = model(obs, psf, M)
                        rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
                        loss = loss_fn(gt.squeeze(dim=1), rec)
                        test_loss += loss.item()

                print("[{}: {}/{}]  train_loss={:.4f}  test_loss={:.4f}".format(
                                epoch+1, step+1, len(train_loader),
                                train_loss/(step+1), 0,
                                test_loss/len(test_loader), 0))
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, f'p4ip_{n_epochs}.pth'))

    return

if __name__ =="__main__":
    train()