import math

import torch
import torch.nn as nn
from models.modules import (gaussian_likelihood, gaussian_sample, View)

class VAE(nn.Module):
    def __init__(self, image_shape, hidden_channels):
        super().__init__()
        self.encoder = nn.Sequential(*[
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 16, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels * 2),
        ])
        self.decoder = nn.Sequential(*[
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 8 * 8 * 16),
            nn.BatchNorm1d(8 * 8 * 16),
            nn.ReLU(),
            View((-1, 16, 8, 8)),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3 * 2, kernel_size=3, stride=1, padding=1, bias=False),
        ])
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        z_mu, z_logvar = torch.chunk(self.encoder(x),2,dim=1)
        z_var = torch.exp(z_logvar)
        z = z_mu
        if self.training:
            z_std = torch.exp(0.5 * z_logvar)
            eps = torch.randn_like(z_mu)
            z += z_std * eps
        x_mu, x_logvar = torch.chunk(self.decoder(z),2,dim=1)
        x_var = torch.exp(x_logvar)
        
        loss_DKL = - 0.5 * (1 + z_logvar - z_mu**2 - z_var).sum(dim=1)
        loss_rec = 0.5 * (c * h * w) * torch.log(2 * math.pi * x_var.view(b,-1).sum(dim=1)) 
                + 0.5 * ((x - x_mu) ** 2 / x_var).view(b, -1).sum(dim=1)
        objective = -(loss_rec + loss_DKL)
        bpd = (-objective) / (math.log(2.) * c * h * w)
        
        summary_image = torch.cat([x[:16], x_mu[:16]], dim=0)
        
        return z, bpd, None, summary_image