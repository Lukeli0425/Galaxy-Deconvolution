import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import logging
import argparse
import torch
from torch import nn
from torch.optim import Adam
from dataset import get_dataloader
from models.Unrolled_ADMM import Unrolled_ADMM
from utils_poisson_deblurring.utils_torch import MultiScaleLoss
from utils import plot_loss

def train(n_iters=8, poisson=True, PnP=True, 
            n_epochs=10, lr=1e-4, I=23.5, train_val_split=0.857, batch_size=32, 
            model_save_path='./saved_models/', load_pretrain=False,
            pretrained_file = None):
    logging.info(f'\nStart training unrolled {"PnP-" if PnP else ""}ADMM with {"Poisson" if poisson else "Gaussian"} likelihood for {n_epochs} epochs.')
    train_loader, val_loader = get_dataloader(I=I, train_test_split=train_val_split, batch_size=batch_size)
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Unrolled_ADMM(n_iters=n_iters, poisson=poisson, PnP=PnP)
    model.to(device)
    if load_pretrain:
        try:
            model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))
            logging.info(f'Successfully loaded in {pretrained_file}')
        except:
            logging.critical(f'Failed loading in {pretrained_file}')

    optimizer = Adam(params=model.parameters(), lr = lr)
    loss_fn = MultiScaleLoss()

    train_loss_list = []
    val_loss_list = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, ((obs, psf, alpha), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
            output = model(obs, psf, alpha)
            rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
            loss = loss_fn(gt.squeeze(dim=1), rec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Evaluate on valid dataset
            if (idx+1) % 25 == 0:
                val_loss = 0.0
                model.eval()
                optimizer.zero_grad()
                for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
                    with torch.no_grad():
                        obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                        output = model(obs, psf, alpha)
                        rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
                        loss = loss_fn(gt.squeeze(dim=1), rec)
                        val_loss += loss.item()

                logging.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
                                epoch+1, idx+1, len(train_loader),
                                train_loss/(idx+1),
                                val_loss/len(val_loader)))
    
        # Evaluate on train & valid dataset after every epoch
        train_loss = 0.0
        model.eval()
        optimizer.zero_grad()
        for _, ((obs, psf, alpha), gt) in enumerate(train_loader):
            with torch.no_grad():
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                output = model(obs, psf, alpha)
                rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
                loss = loss_fn(gt.squeeze(dim=1), rec)
                train_loss += loss.item()
        train_loss_list.append(train_loss)
        
        val_loss = 0.0
        model.eval()
        optimizer.zero_grad()
        for _, ((obs, psf, alpha), gt) in enumerate(val_loader):
            with torch.no_grad():
                obs, psf, alpha, gt = obs.to(device), psf.to(device), alpha.to(device), gt.to(device)
                output = model(obs, psf, alpha)
                rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
                loss = loss_fn(gt.squeeze(dim=1), rec)
                val_loss += loss.item()
        val_loss_list.append(val_loss)

        logging.info(" [{}: {}/{}]  train_loss={:.4f}  val_loss={:.4f}".format(
                        epoch+1, len(train_loader), len(train_loader),
                        train_loss/(idx+1),
                        val_loss/len(val_loader)))

        if (epoch + 1) % 10 == 0:
            model_file_name = f'{"Poisson" if poisson else "Gaussian"}{"_PnP" if PnP else ""}_{epoch+1}epochs.pth'
            torch.save(model.state_dict(), os.path.join(model_save_path, model_file_name))
            logging.info(f'P4IP model saved to {os.path.join(model_save_path, model_file_name)}')

    # Plot loss curve
    plot_loss(train_loss_list, val_loss_list, poisson, PnP, n_epochs, I)

    return


if __name__ =="__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Arguments for training urolled ADMM.')
    parser.add_argument('--n_iters', type=int, default=8)
    parser.add_argument('--poisson', type=bool, default=False)
    parser.add_argument('--PnP', action="store_true")
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--I', type=float, default=23.5, choices=[23.5, 25.2])
    parser.add_argument('--train_val_split', type=float, default=0.857)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--load_pretrain', action="store_true")
    opt = parser.parse_args()

    train(  n_iters=opt.n_iters, poisson=opt.poisson, PnP=opt.PnP,
            n_epochs=opt.n_epochs,lr=opt.lr,
            I=opt.I, train_val_split=opt.train_val_split, batch_size=opt.batch_size,
            load_pretrain=opt.load_pretrain,
            model_save_path='./saved_models/',
            pretrained_file='./saved_models/Poisson_PnP_20epochs.pth')
