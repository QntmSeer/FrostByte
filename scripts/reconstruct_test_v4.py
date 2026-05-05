import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.triplane_encoder import TriPlaneEncoder
from models.triplane import TriPlaneDecoder
from data.volume_dataset import VolumeDataset

def run_sanity_check():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running v4 Atomic Sanity Check on {device}")
    
    CKPT_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cath_subset.pt")
    
    # 1. Load v4 Models
    encoder = TriPlaneEncoder(channels=32, plane_res=128, signal_scale=4.0).to(device)
    encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "triplane_encoder_v4.pth"), map_location=device))
    
    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "triplane_decoder_v4.pth"), map_location=device))
    
    encoder.eval(); decoder.eval()
    
    # 2. Get 1A3N Complex (Hemoglobin)
    data_all = torch.load(DATA_PATH, weights_only=False)
    coords = data_all['1a3n'].to(device)
    coords -= coords.mean(dim=0)
    
    L = 128
    VS = 0.6
    vol_gt = VolumeDataset.voxelize_gaussian(coords, L, VS, 0.6).to(device)
    
    # 3. Recon Path
    with torch.no_grad():
        planes = encoder(vol_gt.unsqueeze(0).unsqueeze(0))
        
        # Grid Query
        grid_1d = torch.linspace(-1.0, 1.0, L, device=device)
        zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        q_coords = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
        
        vol_rec = decoder(planes, q_coords).reshape(L, L, L)
        
    # 4. Save MIP Comparison
    gt_mip = vol_gt.max(axis=2).values.cpu().numpy()
    rec_mip = vol_rec.max(axis=2).values.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_mip, cmap='Greys_r')
    axes[0].set_title("Ground Truth (1A3N)")
    axes[0].axis('off')
    
    axes[1].imshow(rec_mip, cmap='inferno')
    axes[1].set_title("v4 AE Recon (No Diffusion)")
    axes[1].axis('off')
    
    out_path = os.path.join(BASE_DIR, "experiments", "results", "v4_sanity_check.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Sanity Check saved to {out_path}")

if __name__ == "__main__":
    run_sanity_check()
