import torch
import torch.nn as nn
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_2d import Block2D, TriPlaneUNet, SinusoidalPositionEmbeddings
from models.diffusion import DiffusionModel
from models.speculative_diffusion import SpeculativeDiffusionModel

# ============================================================
# Define a Lightweight Draft U-Net
# ============================================================

class SmallTriPlaneUNet(nn.Module):
    """
    A lightweight version of TriPlaneUNet to act as a fast draft model.
    It uses significantly fewer channels (16 -> 32 -> 64) compared to the
    target model (64 -> 128 -> 256), leading to extremely fast evaluations.
    """
    def __init__(self, plane_channels=32, time_dim=64):
        super().__init__()
        self.plane_channels = plane_channels
        self.time_dim = time_dim
        self.in_ch = plane_channels * 3
        self.out_ch = plane_channels * 3
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        self.init_conv = nn.Conv2d(self.in_ch, 16, 3, padding=1)
        
        # Down
        self.down1 = Block2D(16, 32, time_dim) # 32x32
        self.down2 = Block2D(32, 64, time_dim) # 16x16
        
        # Bottleneck
        self.bot1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bot2 = nn.Conv2d(64, 64, 3, padding=1)
        
        # Up
        self.up1 = Block2D(64, 32, time_dim, up=True) # 32x32
        self.up2 = Block2D(32 * 2, 16, time_dim, up=True) # 64x64
        
        self.final_conv = nn.Sequential(
            nn.GroupNorm(4, 16 * 2),
            nn.SiLU(),
            nn.Conv2d(16 * 2, self.out_ch, 3, padding=1)
        )

    def forward(self, planes, time):
        if isinstance(planes, list):
            x = torch.cat(planes, dim=1) 
        else:
            x = planes 
            
        t = self.time_mlp(time)
        
        x0 = self.init_conv(x)
        x1 = self.down1(x0, t)
        x2 = self.down2(x1, t)
        
        b = self.bot1(x2); b = torch.relu(b)
        b = self.bot2(b); b = torch.relu(b)
        
        u1 = self.up1(b, t)
        u2 = self.up2(torch.cat([u1, x1], dim=1), t)
        
        out = self.final_conv(torch.cat([u2, x0], dim=1))
        
        if self.out_ch == self.plane_channels * 3:
            return [out[:, 0:self.plane_channels], 
                    out[:, self.plane_channels:2*self.plane_channels], 
                    out[:, 2*self.plane_channels:]]
        return out


# ============================================================
# Benchmarking script
# ============================================================

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Speculative Speculative Decoding (SSD) on device: {device}")
    
    # 1. Instantiate models
    target_net = TriPlaneUNet(plane_channels=32).to(device)
    target_diffusion = DiffusionModel(target_net, timesteps=100).to(device) 
    
    draft_net = SmallTriPlaneUNet(plane_channels=32).to(device)
    draft_diffusion = DiffusionModel(draft_net, timesteps=100).to(device)
    
    spec_diffusion = SpeculativeDiffusionModel(target_diffusion, draft_diffusion).to(device)
    
    # Shape of Tri-Planes: (B, 3 * C, H, W) -> (B, 96, 64, 64)
    shape = (1, 96, 64, 64)
    
    # Warmup
    print("Warming up models...")
    for _ in range(3):
        dummy_x = torch.randn(shape, device=device)
        dummy_t = torch.randint(0, 100, (1,), device=device)
        _ = target_net(dummy_x, dummy_t)
        _ = draft_net(dummy_x, dummy_t)
    
    # 2. Benchmark standard sequential sampling
    print("\n--- Running Standard DDPM Sampling ---")
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    _ = target_diffusion.sample(shape, device=device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    std_duration = time.time() - start_time
    print(f"Standard Sampling: {std_duration:.3f}s")
    
    for K in [3]:
        # 3. Benchmark standard speculative sampling (sequential execution)
        print(f"\n--- Running Standard Speculative Sampling (K={K}, Synchronous) ---")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        _, stats_sync = spec_diffusion.sample_speculative(shape, K=K, device=device, async_mode=False)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        sync_duration = time.time() - start_time
        sync_speedup = std_duration / sync_duration
        print(f"Synchronous Speculative: {sync_duration:.3f}s (Speedup: {sync_speedup:.2f}x)")
        print(f"  Hit Rate: {stats_sync['hit_rate']*100:.1f}% (Hits: {stats_sync['cache_hits']}, Misses: {stats_sync['cache_misses']})")
        print(f"  Total Speculated Steps: {stats_sync['total_speculations']}")
        
        # 4. Benchmark Tanishq Kumar's Speculative Speculative Decoding (concurrent streams)
        print(f"\n--- Running Tanishq Kumar's Speculative Speculative Decoding (K={K}, Asynchronous SSD) ---")
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        _, stats_async = spec_diffusion.sample_speculative(shape, K=K, device=device, async_mode=True)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        async_duration = time.time() - start_time
        async_speedup = std_duration / async_duration
        
        print(f"Asynchronous SSD: {async_duration:.3f}s (Speedup vs Standard: {async_speedup:.2f}x)")
        if sync_duration > 0:
            print(f"  SSD Speedup vs Synchronous Speculative: {sync_duration / async_duration:.2f}x")
        print(f"  Hit Rate: {stats_async['hit_rate']*100:.1f}% (Hits: {stats_async['cache_hits']}, Misses: {stats_async['cache_misses']})")
        print(f"  Total Speculated Steps: {stats_async['total_speculations']}")

if __name__ == "__main__":
    run_benchmark()
