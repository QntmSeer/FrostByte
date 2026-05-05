import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_2d import TriPlaneUNet
from models.triplane import TriPlaneDecoder
from models.diffusion import DiffusionModel

def debug_planes():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the AutoEncoder
    from models.triplane_encoder import TriPlaneEncoder
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    encoder = TriPlaneEncoder(channels=32).to(device)
    encoder.load_state_dict(torch.load(os.path.join(BASE_DIR, 'experiments/checkpoints/triplane_encoder_v2.pth'), map_location=device))
    encoder.eval()
    
    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(os.path.join(BASE_DIR, 'experiments/checkpoints/triplane_decoder_v2.pth'), map_location=device))
    decoder.eval()
    
    # 2. Check Autoencoder over actual data
    from data.triplane_dataset import TriPlaneDataset
    dataset = TriPlaneDataset(os.path.join(BASE_DIR, 'data/processed/cath_subset.pt'), num_samples=1000, grid_size=64)
    vol_gt, coords_gt, density_gt = dataset[0]
    vol_gt = vol_gt.unsqueeze(0).to(device) # (1, 1, 64, 64, 64)
    coords_gt = coords_gt.unsqueeze(0).to(device) # (1, N, 3)
    
    with torch.no_grad():
        planes_gt = encoder(vol_gt)
        print("GT Planes stats:")
        print(f"  XY: mean={planes_gt[0].mean():.4f}, std={planes_gt[0].std():.4f}")
        
        # Test GT continuous evaluation on the sampled coordinates
        pred_density = decoder(planes_gt, (coords_gt / (32.0 * dataset.voxel_size))) # scale to [-1, 1]
        
        print("Density GT vs Pred on random points:")
        print(f"  GT: max={density_gt.max():.4f}, mean={density_gt.mean():.4f}, nonzeros={(density_gt > 0.01).sum()}")
        print(f"  Pred: max={pred_density.max():.4f}, mean={pred_density.mean():.4f}")
        
        # Test full grid query on the AutoEncoder
        grid_1d = torch.linspace(-1.0, 1.0, 64, device=device)
        z, y, x = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        queries_flat = torch.stack([x, y, z], dim=-1).reshape(1, -1, 3)
        density_full = decoder(planes_gt, queries_flat).reshape(64, 64, 64)
        
        print("AutoEncoder Full Grid stats:")
        print(f"  Max={density_full.max():.4f}, Mean={density_full.mean():.4f}, NonZeros={(density_full > 0.01).sum()}")
        
    print("\n-------------------------------\n")
    # 3. Load the Latent Diffusion Model (UNet 2D)
    unet = TriPlaneUNet(plane_channels=32, time_dim=64).to(device)
    unet.load_state_dict(torch.load(os.path.join(BASE_DIR, 'experiments/checkpoints/ddpm_triplane_2d_v2.pth'), map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, 1000).to(device)
    
    with torch.no_grad():
        # Sample 1 protein, 96 concatenated channels, 64x64 spatial resolution
        planes_latent = diffusion.sample((1, 96, 64, 64), device=device)
        planes_diff = [planes_latent[:, 0:32], planes_latent[:, 32:64], planes_latent[:, 64:96]]
        
        print("Diffusion Planes stats:")
        print(f"  XY: mean={planes_diff[0].mean():.4f}, std={planes_diff[0].std():.4f}")
        
        density_diff = decoder(planes_diff, queries_flat).reshape(64, 64, 64)
        print("Diffusion Full Grid stats:")
        print(f"  Max={density_diff.max():.4f}, Mean={density_diff.mean():.4f}, NonZeros={(density_diff > 0.01).sum()}")

if __name__ == "__main__":
    debug_planes()
