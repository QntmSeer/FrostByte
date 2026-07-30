"""
RTX A2000 3D Generative Diffusion & Cryo-EM Workstation Stress Test
===================================================================
Domain: 3D Diffusion Models / Structural Biology & Cryo-EM

Benchmarks:
1. Sampling latency per 3D density grid (32^3, 64^3, 128^3).
2. Denoising score-matching loss curves across 1,000 timesteps.
3. Real-space cross-correlation vs. resolution (FSC 0.143 resolution threshold).
"""

import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from utils.metrics import compute_cc, compute_fsc

def benchmark_grid_sampling_latency(device):
    print("\n" + "="*70)
    print(" 1. BENCHMARK: 3D Density Grid Sampling Latency")
    print("="*70)
    
    grid_sizes = [32, 64, 128]
    timesteps_test = 50 # Sub-sampled for benchmark timing
    
    results = {}
    for L in grid_sizes:
        print(f"\n---> Testing Grid Size: {L}^3 ({L}x{L}x{L} voxels)...")
        try:
            model = UNet3D(in_ch=1, out_ch=1, time_dim=64).to(device)
            model.eval()
            diffusion = DiffusionModel(model, timesteps=1000).to(device)
            
            shape = (1, 1, L, L, L)
            
            # Warmup
            dummy_x = torch.randn(shape, device=device)
            dummy_t = torch.randint(0, 1000, (1,), device=device)
            with torch.no_grad():
                _ = model(dummy_x, dummy_t)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            with torch.no_grad():
                x_t = torch.randn(shape, device=device)
                for i in reversed(range(1000 - timesteps_test, 1000)):
                    t_step = torch.full((1,), i, device=device, dtype=torch.long)
                    eps = model(x_t, t_step)
                    x_t = x_t - 0.01 * eps
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_mem = 0.0
                
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / timesteps_test) * 1000.0
            est_1000_step_sec = (elapsed / timesteps_test) * 1000.0
            
            results[L] = {
                'ms_per_step': ms_per_step,
                'est_1000_step_sec': est_1000_step_sec,
                'peak_vram_mb': peak_mem
            }
            
            print(f"     [OK] {L}^3 Grid | Single Step Latency: {ms_per_step:.2f} ms")
            print(f"     [OK] Est. 1,000-Step Sampling Wall-Clock: {est_1000_step_sec:.2f} s")
            print(f"     [OK] Peak VRAM Allocated: {peak_mem:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"     [OOM] {L}^3 Grid exceeded GPU VRAM limit.")
                results[L] = 'OOM'
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise e
                
    return results

def benchmark_score_matching_loss_curve(device):
    print("\n" + "="*70)
    print(" 2. BENCHMARK: Denoising Score-Matching Loss Across 1,000 Timesteps")
    print("="*70)
    
    L = 64
    model = UNet3D(in_ch=1, out_ch=1, time_dim=64).to(device)
    model.eval()
    diffusion = DiffusionModel(model, timesteps=1000).to(device)
    
    # Generate continuous density sphere target
    coords = torch.linspace(-1, 1, L, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    vol_gt = (torch.exp(-(xx**2 + yy**2 + zz**2)/0.2) > 0.3).float().unsqueeze(0).unsqueeze(0)
    
    t_samples = np.linspace(0, 999, 20, dtype=int)
    losses = []
    
    print("\n   Timestep (t) | Noise MSE Loss | SNR (dB)")
    print("   ----------------------------------------")
    for t_val in t_samples:
        t_tensor = torch.full((1,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(vol_gt)
        
        sqrt_alpha = diffusion.sqrt_alphas_cumprod[t_val]
        sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t_val]
        
        x_t = sqrt_alpha * vol_gt + sqrt_one_minus_alpha * noise
        
        with torch.no_grad():
            pred_noise = model(x_t, t_tensor)
            loss = F.mse_loss(pred_noise, noise).item()
            
        snr_db = 10 * np.log10((sqrt_alpha.item()**2) / (sqrt_one_minus_alpha.item()**2 + 1e-8))
        losses.append((t_val, loss, snr_db))
        print(f"     t = {t_val:4d}   |  MSE = {loss:.5f}  | SNR = {snr_db:6.2f} dB")
        
    return losses

def benchmark_fsc_resolution(device):
    print("\n" + "="*70)
    print(" 3. BENCHMARK: Real-Space Cross-Correlation & FSC_0.143 Threshold")
    print("="*70)
    
    L = 64
    coords = torch.linspace(-1, 1, L, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    vol_gt = torch.exp(-(xx**2 + yy**2 + zz**2)/0.15)
    
    # Add noise to simulate reconstructed density at target resolution
    noise_level = 0.25
    vol_recon = vol_gt + torch.randn_like(vol_gt) * noise_level
    
    cc = compute_cc(vol_recon, vol_gt, align=False)
    freqs, fsc = compute_fsc(vol_recon, vol_gt, align=False)
    
    # Find FSC 0.143 cutoff
    idx_cutoff = np.where(fsc <= 0.143)[0]
    if len(idx_cutoff) > 0:
        freq_cutoff = freqs[idx_cutoff[0]]
        res_spatial_voxels = 1.0 / max(freq_cutoff, 1e-5)
    else:
        freq_cutoff = freqs[-1]
        res_spatial_voxels = 2.0
        
    print(f"\n   [OK] 3D Real-Space Pearson Cross-Correlation (CC): {cc:.4f}")
    print(f"   [OK] FSC_0.143 Spatial Frequency Cutoff: {freq_cutoff:.4f} (Nyquist fraction)")
    print(f"   [OK] FSC_0.143 Estimated Resolution Limit: {res_spatial_voxels:.2f} voxels")

def run_stress_test():
    print("="*70)
    print(" FrostByte: PyTorch/CUDA 3D Diffusion & Cryo-EM Stress Test Suite")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Execution Device: {device}")
    if device.type == 'cuda':
        print(f" GPU Model:        {torch.cuda.get_device_name(0)}")
        print(f" Total VRAM:       {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f" CUDA Version:     {torch.version.cuda}")
    else:
        print(" Warning: CUDA device not detected. Running CPU benchmark fallback.")
        
    benchmark_grid_sampling_latency(device)
    benchmark_score_matching_loss_curve(device)
    benchmark_fsc_resolution(device)
    
    print("\n" + "="*70)
    print(" [COMPLETE] All Workstation Stress Tests Completed Successfully.")
    print("="*70)

if __name__ == "__main__":
    run_stress_test()
