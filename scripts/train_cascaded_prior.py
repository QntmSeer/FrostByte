import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.triplane_encoder import TriPlaneEncoder
from models.unet_upsampler import CascadedTriPlaneUNet
from models.diffusion import DiffusionModel
from data.triplane_dataset import TriPlaneDataset

CKPT_DIR = "experiments/checkpoints"

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Cascaded SR-Prior (64->128) on {device}")

    # 1. Dataset (128x128 Targets)
    dataset = TriPlaneDataset(os.path.join(BASE_DIR, 'data/processed/cath_subset.pt'), 
                              num_samples=10, grid_size=128, voxel_size=0.6, augment=True)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    # 2. Frozen Encoder (Used to generate both 64 and 128 sketches)
    encoder_base = TriPlaneEncoder(channels=32, plane_res=64, signal_scale=2.0).to(device)
    encoder_high = TriPlaneEncoder(channels=32, plane_res=128, signal_scale=4.0).to(device)
    
    enc_v4_path = os.path.join(BASE_DIR, CKPT_DIR, "triplane_encoder_v4.pth")
    encoder_high.load_state_dict(torch.load(enc_v4_path, map_location=device))
    # Base encoder can also use v4 weights roughly
    encoder_base.load_state_dict(torch.load(enc_v4_path, map_location=device), strict=False)
    
    for e in [encoder_base, encoder_high]: 
        e.eval()
        for p in e.parameters(): p.requires_grad = False

    # 3. Trainable Upsampler
    upsampler = CascadedTriPlaneUNet(plane_channels=32, time_dim=64).to(device)
    diffusion = DiffusionModel(upsampler, 1000).to(device) # We handle the SR logic manually
    
    opt = torch.optim.AdamW(upsampler.parameters(), lr=2e-4)
    epochs = 400

    for epoch in range(1, epochs + 1):
        upsampler.train()
        total_loss = 0.0

        for vol_gt, _, _ in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            vol_gt = vol_gt.to(device)
            B = vol_gt.shape[0]

            with torch.no_grad():
                # Get the "Sketch" (64) and the "Goal" (128)
                planes_lr = encoder_base(vol_gt)
                low_res = torch.cat(planes_lr, dim=1) # (B, 96, 64, 64)
                
                planes_hr = encoder_high(vol_gt)
                x_0 = torch.cat(planes_hr, dim=1)     # (B, 96, 128, 128)

            t = torch.randint(0, 1000, (B,), device=device).long()
            noise = torch.randn_like(x_0)

            sqrt_alpha = diffusion.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
            sqrt_one_minus_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
            x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

            opt.zero_grad()
            # Feed low_res conditioning to upsampler
            noise_pred_list = upsampler(x_t, t, low_res)
            noise_pred = torch.cat(noise_pred_list, dim=1)
            
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | SR-Loss: {total_loss/len(loader):.4f}")
            torch.save(upsampler.state_dict(), os.path.join(BASE_DIR, CKPT_DIR, "triplane_upsampler_v5.pth"))

    print("Cascaded Training Complete.")

if __name__ == "__main__":
    train()
