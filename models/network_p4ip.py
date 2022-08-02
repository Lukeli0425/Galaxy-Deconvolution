import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

from models.ResUNet import ResUNet
from utils_poisson_deblurring.utils_deblur import pad
from utils_poisson_deblurring.utils_torch import conv_fft, conv_fft_batch, psf_to_otf



def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		# nn.init.uniform(m.weight.data, 1.0, 0.02)
		m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
		nn.init.constant(m.bias.data, 0.0)



class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)

class X_Update(nn.Module):
	def __init__(self):
		super(X_Update, self).__init__()

	def forward(self, x1, x2, AtA_fft, rho1, rho2):
		lhs = rho1*AtA_fft + rho2 
		rhs = tfft.fftn( rho1*x1 + rho2*x2, dim=[2,3] )
		x = tfft.ifftn(rhs/lhs, dim=[2,3])
		return x.real

class U_Update(nn.Module):
	def __init__(self):
		super(U_Update, self).__init__()

	def forward(self, v_tilde, y, rho2, M):
		t1 = rho2*v_tilde - M 
		return 0.5*(1/rho2)*(-t1 + torch.sqrt(t1**2 + 4*y*rho2))

class Z_Update(nn.Module):
	"""Updating Z with zero partial derivative."""
	def __init__(self):
		super(Z_Update, self).__init__()		

	def forward(self, z):
		z_out = self.net(z.float())
		return z_out

class Z_Update_ResUNet(nn.Module):
	"""Updating Z with ResUNet as denoiser."""
	def __init__(self):
		super(Z_Update_ResUNet, self).__init__()		
		self.net = ResUNet()

	def forward(self, z):
		z_out = self.net(z.float())
		return z_out
	
class InitNet(nn.Module):
	def __init__(self, n):
		super(InitNet,self).__init__()
		self.n = n

		self.conv_layers = nn.Sequential(
			Down(1,4),
			Down(4,8),
			Down(8,16),
			Down(16,16))
		
		self.mlp = nn.Sequential(
			nn.Linear(16*8*8+1, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 2*self.n),
			nn.Softplus())
		self.resize = nn.Upsample(size=[256,256], mode='bilinear', align_corners=True)
		
	def forward(self, kernel, M):
		N, C, H, W  = kernel.size()
		h1, h2 = int(np.floor(0.5*(128-H))), int(np.ceil(0.5*(128-H)))
		w1, w2 = int(np.floor(0.5*(128-W))), int(np.ceil(0.5*(128-W)))
		k_pad = F.pad(kernel, (w1,w2,h1,h2), "constant", 0)
		A = tfft.fftn(k_pad,dim=[2,3])
		AtA_fft = torch.abs(A)**2
		x = self.conv_layers(AtA_fft.float())
		x = torch.cat((x.view(N,1,16*8*8),  M.float().view(N,1,1)), axis=2).float()
		h = self.mlp(x)+1e-6

		rho1_iters = h[:,:,0:self.n].view(N, 1, 1, self.n)
		rho2_iters = h[:,:,self.n:2*self.n].view(N, 1, 1, self.n)
		return rho1_iters, rho2_iters
		
		
class P4IP_Net(nn.Module):
	def __init__(self, n_iters=8):
		super(P4IP_Net, self).__init__()
		self.n =  n_iters
		self.init = InitNet(self.n)
		self.X = X_Update() # FFT based quadratic solution
		self.U = U_Update()	# Poisson MLE
		self.Z = Z_Update_ResUNet() # BW Denoiser
	
	def init_l2(self, y, A, M):
		N, C, H, W = y.size()
		At, AtA_fft = torch.conj(A), torch.abs(A)**2
		rhs = tfft.fftn( conv_fft_batch(At, y/M), dim=[2,3] )
		lhs = AtA_fft + (1/M)
		x0 = torch.real(tfft.ifftn(rhs/lhs, dim=[2,3]))
		x0 = torch.clamp(x0,0,1)
		return x0

	def forward(self, y, kernel, M):
		device = torch.device("cuda:0" if y.is_cuda else "cpu")
		x_list = []

		N, C, H, W = y.size()
		# Generate auxiliary variables for convolution
		k_pad, A = psf_to_otf(kernel, y.size())
		A =  A.to(device)
		At, AtA_fft = torch.conj(A), torch.abs(A)**2
		rho1_iters, rho2_iters = self.init(kernel, M) 	# Hyperparameters
		x = self.init_l2(y, A, M)	# Initialization using Weiner Deconvolution
		x_list.append(x)
		# Other ADMM variables
		z = Variable(x.data.clone()).to(device)
		u = Variable(y.data.clone()).to(device)
		v1 = torch.zeros(y.size()).to(device)
		v2 = torch.zeros(y.size()).to(device)
			
		for n in range(self.n):
			rho1 = rho1_iters[:,:,:,n].view(N,1,1,1)
			rho2 = rho2_iters[:,:,:,n].view(N,1,1,1)
			# U, Z and X updates
			u = self.U(conv_fft_batch(A,x) + v1, y, rho1, M)
			z = self.Z(x + v2)
			x = self.X(conv_fft_batch(At,u - v1), z - v2, AtA_fft, rho1, rho2)
			# Lagrangian updates			
			v1 = v1 + conv_fft_batch(A,x) - u
			v2 = v2 + x - z
			x_list.append(x)

		return x_list

	
