import os
import logging
import torch
from torch.optim import Adam, SGD
from dataset import get_dataloader
from models.network_p4ip import P4IP_Net
from utils_poisson_deblurring.utils_torch import MultiScaleLoss

def train_p4ip( n_epochs=10, n_iters=8, lr=1e-4, train_test_split=0.7, batch_size=32,
                model_save_path='./saved_models/', load_pretrain=False,
                pretrained_file = None):
    """Train p4ip (unrolled PnP-ADMM) model."""
    logging.info('Start training pip4.')
    train_loader, test_loader = get_dataloader(train_test_split=train_test_split, batch_size=batch_size)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = P4IP_Net(n_iters=n_iters)
    model.to(device)
    if load_pretrain:
        model.load_state_dict(torch.load(pretrained_file, map_location=torch.device(device)))

    optimizer = Adam(params=model.parameters(), lr = lr)
    loss_fn = MultiScaleLoss()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for idx, ((obs, psf, M), gt) in enumerate(train_loader):
            optimizer.zero_grad()
            obs, psf, M, gt = obs.to(device), psf.to(device), M.to(device), gt.to(device)
            output = model(obs, psf, M)
            rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
            loss = loss_fn(gt.squeeze(dim=1), rec)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # evaluate on test dataset
            if (idx+1) % 20 == 0:
                test_loss = 0.0
                model.eval()
                optimizer.zero_grad()
                for _, ((obs, psf, M), gt) in enumerate(test_loader):
                    with torch.no_grad():
                        obs, psf, M, gt = obs.to(device), psf.to(device), M.to(device), gt.to(device)
                        output = model(obs, psf, M)
                        rec = output[-1].squeeze(dim=1) #* M.view(batch_size,1,1)
                        loss = loss_fn(gt.squeeze(dim=1), rec)
                        test_loss += loss.item()

                logging.info("[{}: {}/{}]  train_loss={:.4f}  test_loss={:.4f}".format(
                                epoch+1, idx+1, len(train_loader),
                                train_loss/(idx+1), 0,
                                test_loss/len(test_loader), 0))
    
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    torch.save(model.state_dict(), os.path.join(model_save_path, f'p4ip_{n_epochs}.pth'))
    logging.info(f'pip4 model saved to {os.path.join(model_save_path, f"p4ip_{n_epochs}.pth")}')

    return

if __name__ =="__main__":
    train_p4ip(load_pretrain=True, pretrained_file='./poisson-deblurring/model_zoo/p4ip_100epoch.pth')