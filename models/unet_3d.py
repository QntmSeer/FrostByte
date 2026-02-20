
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv3d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose3d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv3d(out_ch, out_ch, 4, 2, 1)
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm3d(out_ch)
        self.bnorm2 = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.conv1(x) # (B, out_ch, D, H, W)
        h = self.bnorm1(h)
        h = self.relu(h)
        
        # Time Embedding
        time_emb = self.relu(self.time_mlp(t)) # (B, out_ch)
        time_emb = time_emb[(..., ) + (None, ) * 3] # (B, out_ch, 1, 1, 1)
        h = h + time_emb
        
        # Second Conv
        h = self.conv2(h)
        h = self.bnorm2(h)
        h = self.relu(h)
        
        # Down or Up Transform
        return self.transform(h)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU()
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.conv(x)
        time_emb = self.time_mlp(t)[(..., ) + (None, ) * 3]
        return h + time_emb

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x, t):
        x = self.conv(x, t)
        p = self.pool(x)
        return x, p

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch, time_emb_dim)

    def forward(self, x, skip, t):
        x = self.up(x)
        # diffY = skip.size()[2] - x.size()[2]
        # diffX = skip.size()[3] - x.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
        #               diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x, t)

class UNet3D(nn.Module):
    """
    Simple 3D U-Net for Volumetric Diffusion.
    Input: (B, 1, 64, 64, 64)
    Output: (B, 1, 64, 64, 64)
    """
    def __init__(self, in_ch=1, out_ch=1, time_dim=64):
        super().__init__()
        
        # Time Embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        n_channels = 32
        
        # Down
        self.down1 = Down(in_ch, n_channels, time_dim) # 64 -> 32
        self.down2 = Down(n_channels, n_channels*2, time_dim) # 32 -> 16
        self.down3 = Down(n_channels*2, n_channels*4, time_dim) # 16 -> 8
        self.down4 = Down(n_channels*4, n_channels*8, time_dim) # 8 -> 4

        # Bottleneck
        self.bot = ConvBlock(n_channels*8, n_channels*16, time_dim) # 4

        # Up
        self.up1 = Up(n_channels*16, n_channels*8, time_dim) # 4 -> 8
        self.up2 = Up(n_channels*8, n_channels*4, time_dim) # 8 -> 16
        self.up3 = Up(n_channels*4, n_channels*2, time_dim) # 16 -> 32
        self.up4 = Up(n_channels*2, n_channels, time_dim) # 32 -> 64
        
        self.out = nn.Conv3d(n_channels, out_ch, 1)

    def forward(self, x, t):
        # x: (B, C, D, H, W)
        t = self.time_mlp(t)
        
        # 64
        skip1, x = self.down1(x, t)
        # 32
        skip2, x = self.down2(x, t)
        # 16
        skip3, x = self.down3(x, t)
        # 8
        skip4, x = self.down4(x, t)
        
        # 4
        x = self.bot(x, t)
        
        # 8
        x = self.up1(x, skip4, t)
        # 16
        x = self.up2(x, skip3, t)
        # 32
        x = self.up3(x, skip2, t)
        # 64
        x = self.up4(x, skip1, t)
        
        return self.out(x)

if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D().to(device)
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    t = torch.randint(0, 1000, (2,)).to(device)
    out = model(x, t)
    print(out.shape)
