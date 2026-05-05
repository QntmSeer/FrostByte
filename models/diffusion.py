import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def get_loss(self, x_0, mask=None):
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device).long()
        epsilon = torch.randn_like(x_0)
        
        broadcast_shape = [B] + [1] * (x_0.dim() - 1)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(*broadcast_shape)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(*broadcast_shape)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * epsilon
        
        if mask is not None:
             epsilon_pred = self.model(x_t, t, mask=mask)
        else:
             epsilon_pred = self.model(x_t, t)
        
        if isinstance(epsilon_pred, list):
            epsilon_pred = torch.cat(epsilon_pred, dim=1)

        if mask is not None:
            loss = F.mse_loss(epsilon_pred, epsilon, reduction='none')
            while mask.dim() < epsilon.dim():
                mask = mask.unsqueeze(-1)
            loss = (loss * mask).sum() / (mask.sum() * (epsilon.numel() / mask.numel()))
        else:
            loss = F.mse_loss(epsilon_pred, epsilon)
        return loss
    
    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        """Standard DDPM sampling for Stage 1 (64x64)."""
        B = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps_theta = self.model(x, t)
            if isinstance(eps_theta, list):
                eps_theta = torch.cat(eps_theta, dim=1)
            
            beta_t = self.betas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            alpha_t = 1 - beta_t
            
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x - coeff2 * eps_theta)
            
            if i > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
            x.clamp_(-6.0, 6.0) # Standard manifold range
        return x

    @torch.no_grad()
    def sample_cascaded(self, shape, low_res, device='cpu'):
        """Conditioned sampling for Stage 2 (Super-Resolution)."""
        B = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            # Upsampler model expects (x_t, t, low_res)
            eps_theta_list = self.model(x, t, low_res)
            eps_theta = torch.cat(eps_theta_list, dim=1)
            
            beta_t = self.betas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            alpha_t = 1 - beta_t
            
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x - coeff2 * eps_theta)
            
            if i > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
            x.clamp_(-10.0, 10.0) # Atomic range
        return x
