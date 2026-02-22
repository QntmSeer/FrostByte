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

class SimplePointNet(nn.Module):
    """
    A simple PointNet-like architecture for 3D point cloud diffusion.
    Modified to take time/noise level as input.
    """
    def __init__(self, in_dim=3, hidden_dim=128, time_embed_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim) # Output noise prediction (same shape as input)
        )
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, t):
        # x: (B, N, 3)
        # t: (B,)
        
        B, N, D = x.shape
        
        # Embed time
        t_emb = self.time_mlp(t) # (B, H)
        
        # Per-point embedding
        h = self.input_mlp(x) # (B, N, H)
        
        # Add time embedding to per-point features
        h = h + t_emb.unsqueeze(1)
        
        # Process (simplified PointNet checks)
        # Here we really need interactions. 
        # A true PointNet has a global feature agg. 
        # For diffusion on structure, we want to update local coordinates based on global context.
        
        # Global feature extraction
        # (B, N, H) -> (B, H, N)
        h_trans = h.transpose(1, 2)
        global_feat = F.max_pool1d(h_trans, kernel_size=N).squeeze(2) # (B, H)
        
        # Concat global to local
        feature = torch.cat([h, global_feat.unsqueeze(1).repeat(1, N, 1)], dim=2) # (B, N, 2H)
        
        # Refine (Simulating layers)
        # We need a bit more capacity here usually, but keeping it simple for "SimplePointNet"
        # Since we just output noise, let's project back.
        # But wait, dimensionality mismatch. 
        # Let's add a middle layer that mixes global info.
        
        # Re-defining structure to be more proper for dense prediction
        return self._param_free_forward(x, t_emb)

    def _param_free_forward(self, x, t_emb):
        # Simplified dense skip architecture
        h = self.input_mlp(x) + t_emb.unsqueeze(1)
        return self.output_mlp(h)

class ResBlock(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.time_proj = nn.Linear(time_dim, dim)
    
    def forward(self, x, t_emb):
        # x: (B, N, dim)
        # t_emb: (B, time_dim)
        h = self.mlp(x + self.time_proj(t_emb).unsqueeze(1))
        return x + h

class GlobalAttention(nn.Module):
    """Simplified Self-Attention to aggregate global context."""
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.scale = dim ** -0.5
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # mask: (B, N)
            m = mask.unsqueeze(1).repeat(1, N, 1) # (B, N, N)
            attn = attn.masked_fill(~m, -1e9)
            
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        return self.proj(x)

class GeometricAttention(nn.Module):
    """
    SE(3)-Equivariant Message Passing Layer.
    Ensures that if the input point cloud rotates, the output score rotates identically.
    
    Formula: m_i = sum_j f(dist_ij, h_i, h_j) * (x_i - x_j) / dist_ij
    """
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        # MLP for computing invariant messages from distances and features
        self.message_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Output a scalar weight for the vector
        )
        self.feat_update = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x, h, mask=None):
        """
        Args:
            x: coordinates (B, N, 3)
            h: invariant features (B, N, dim)
            mask: padding mask (B, N)
        Returns:
            dx: equivariant coordinate update (B, N, 3)
            h_new: updated invariant features (B, N, dim)
        """
        B, N, D = x.shape
        # 1. Compute pairwise distances (Invariant)
        # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True) # (B, N, N, 1)
        
        # Normalize diff for direction
        direction = diff / (dist + 1e-8)
        
        # 2. Compute messages
        # Concat features: (B, N, N, dim*2 + 1)
        h_i = h.unsqueeze(2).repeat(1, 1, N, 1)
        h_j = h.unsqueeze(1).repeat(1, N, 1, 1)
        msg_input = torch.cat([h_i, h_j, dist], dim=-1)
        
        weights = self.message_mlp(msg_input) # (B, N, N, 1)
        
        if mask is not None:
            m = mask.unsqueeze(1).repeat(1, N, 1).unsqueeze(-1)
            weights = weights.masked_fill(~m, 0.0)
            
        # Equivariant update: weighted sum of relative vectors
        dx = torch.sum(weights * direction, dim=2) # (B, N, 3)
        
        # Invariant feature update
        h_agg = torch.sum(weights * dist, dim=2) # (B, N, 1)
        h_new = self.feat_update(torch.cat([h, h_agg], dim=-1))
        
        return dx, h_new

class PointDiffusionTransformer(nn.Module):
    """
    SE(3)-Equivariant Point Cloud Diffusion Model.
    Uses Geometric Message Passing to ensure symmetry.
    """
    def __init__(self, in_dim=3, hidden_dim=256, num_layers=4):
        super().__init__()
        self.time_dim = 64
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initial invariant features from coordinates?
        # To be truly invariant, initial features should be from distances or atom types.
        # Since we use C-alpha only, we'll use a constant feature or local density.
        self.init_feat = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.blocks = nn.ModuleList([
            GeometricAttention(hidden_dim) for _ in range(num_layers)
        ])
        
        # Each layer produces an equivariant dx. We sum them up.
        # Actually, we can refine x or just output a final sum.
        # Typical equivariant nets output vector field.
        
    def forward(self, x, t, mask=None):
        B, N, _ = x.shape
        t_emb = self.time_mlp(t)
        
        # Initialize invariant features
        h = self.init_feat.repeat(B, N, 1)
        h = h + t_emb.unsqueeze(1)
        
        total_dx = torch.zeros_like(x)
        
        for block in self.blocks:
            dx, h = block(x, h, mask=mask)
            total_dx = total_dx + dx
            
        return total_dx

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
        """
        Calculates diffusion loss.
        x_0: (B, N, 3) or (B, C, D, H, W)
        mask: optional bool tensor
        """
        B = x_0.shape[0]
        # B, *dims = x_0.shape
        
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device).long()
        epsilon = torch.randn_like(x_0)
        
        # Reshape alpha for broadcasting
        # We need (B, 1, 1) for points or (B, 1, 1, 1, 1) for volumes
        # Universal approach: (B, *([1]*len(dims)))
        broadcast_shape = [B] + [1] * (x_0.dim() - 1)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(*broadcast_shape)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(*broadcast_shape)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * epsilon
        
        # Model forward
        # PointNet expects (x, t, mask)
        # UNet expects (x, t)
        if mask is not None:
             epsilon_pred = self.model(x_t, t, mask=mask)
        else:
             epsilon_pred = self.model(x_t, t)
        
        if mask is not None:
            loss = F.mse_loss(epsilon_pred, epsilon, reduction='none')
            # Handle mask broadcasting
            # Mask is usually (B, N) for points.
            # Epsilon is (B, N, 3).
            while mask.dim() < epsilon.dim():
                mask = mask.unsqueeze(-1)
            
            loss = (loss * mask).sum() / (mask.sum() * (epsilon.numel() / mask.numel()))
        else:
            loss = F.mse_loss(epsilon_pred, epsilon)
            
        return loss
    
    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        # Sampling (p sample loop) - DDPM Ancestral Sampling
        # shape: tuple, e.g. (B, N, 3) or (B, 1, 64, 64, 64)
        B = shape[0]
        x = torch.randn(shape, device=device)
        
        broadcast_shape = [B] + [1] * (len(shape) - 1)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            # Predict noise
            epsilon_theta = self.model(x, t)
            
            # If the model is the TriPlaneUNet, it returns a list of 3 planes [XY, XZ, YZ].
            # We must concatenate them back into the shape of x (B, 3C, H, W) to perform diffusion math.
            if isinstance(epsilon_theta, list):
                epsilon_theta = torch.cat(epsilon_theta, dim=1)
            
            # Inverse parameters
            beta_t = self.betas[i]
            alpha_t = 1 - beta_t
            alpha_bar_t = self.alphas_cumprod[i]
            
            # Mean
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            
            mean = coeff1 * (x - coeff2 * epsilon_theta)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t) # approx
                x = mean + sigma_t * noise
            else:
                x = mean
                
        return x
