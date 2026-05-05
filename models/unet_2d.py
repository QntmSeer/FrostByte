import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block2D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.spatial = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        else:
            self.spatial = nn.Conv2d(in_ch, out_ch, 3, 2, 1)
            
        self.conv1 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.transform = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        
    def forward(self, x, t):
        h = self.spatial(x)
        time_emb = self.time_mlp(t)
        time_emb = time_emb[(...,) + (None,) * 2] # (B, C, 1, 1)
        
        h = h + time_emb
        h = self.conv1(h)
        return self.transform(h)

class TriPlaneUNet(nn.Module):
    """
    A 2D U-Net designed to denoise Tri-Plane representations.
    Instead of processing a 3D volume (1 channel, 64^3), it processes
    3 concatenated 2D feature planes (3 * C channels, 64^2), slashing VRAM requirements.
    """
    def __init__(self, plane_channels=32, time_dim=64, in_channels=None, out_channels=None):
        super().__init__()
        self.plane_channels = plane_channels
        self.time_dim = time_dim
        
        # Tri-planes: XY, XZ, YZ concatenated along channel dimension
        # If in_channels is provided (e.g. for cascaded models), use it.
        self.in_ch = in_channels if in_channels is not None else plane_channels * 3
        # Output is the predicted noise in the tri-planes
        self.out_ch = out_channels if out_channels is not None else plane_channels * 3
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        self.init_conv = nn.Conv2d(self.in_ch, 64, 3, padding=1)
        
        # Down
        self.down1 = Block2D(64, 128, time_dim) # 32x32
        self.down2 = Block2D(128, 256, time_dim) # 16x16
        
        # Bottleneck (Simulated self-attention substitute for simplicity)
        self.bot1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bot2 = nn.Conv2d(256, 256, 3, padding=1)
        
        # Up
        self.up1 = Block2D(256, 128, time_dim, up=True) # 32x32
        self.up2 = Block2D(128 * 2, 64, time_dim, up=True) # 64x64
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, 64 * 2),
            nn.SiLU(),
            nn.Conv2d(64 * 2, self.out_ch, 3, padding=1)
        )

    def forward(self, planes, time):
        """
        planes: [XY, XZ, YZ] OR concatenated tensor
        time: (B,)
        Returns: [noise_XY, noise_XZ, noise_YZ] OR planes if matching out_ch
        """
        if isinstance(planes, list):
            x = torch.cat(planes, dim=1) 
        else:
            x = planes 
            
        B, C_total, H, W = x.shape
        assert C_total == self.in_ch, f"Expected {self.in_ch} channels, got {C_total}"
        
        t = self.time_mlp(time)
        
        x0 = self.init_conv(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        
        b = self.bot1(x2); b = F.silu(b)
        b = self.bot2(b); b = F.silu(b)
        
        u1 = self.up1(b, t)
        u2 = self.up2(torch.cat([u1, x1], dim=1), t)
        
        out = self.final_conv(torch.cat([u2, x0], dim=1))
        
        # Split back if possible
        if self.out_ch == self.plane_channels * 3:
            return [out[:, 0:self.plane_channels], 
                    out[:, self.plane_channels:2*self.plane_channels], 
                    out[:, 2*self.plane_channels:]]
        return out

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriPlaneUNet(plane_channels=32).to(device)
    
    # Simulate Tri-Planes
    B, C, H, W = 2, 32, 64, 64
    xy = torch.randn(B, C, H, W).to(device)
    xz = torch.randn(B, C, H, W).to(device)
    yz = torch.randn(B, C, H, W).to(device)
    
    t = torch.randint(0, 1000, (B,)).to(device)
    
    out_xy, out_xz, out_yz = model([xy, xz, yz], t)
    print(f"Input XY shape: {xy.shape}")
    print(f"Output XY noise shape: {out_xy.shape}")
    print(f"Total Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
