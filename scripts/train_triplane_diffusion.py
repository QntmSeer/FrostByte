import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse

# Internal project imports
from data.triplane_dataset import TriPlaneDataset
from models.triplane_encoder import TriPlaneEncoder
from models.triplane import TriPlaneDecoder
from models.unet_2d import UNet2D
from scripts.diffusion_utils import GaussianDiffusion

# Constants
BASE_DIR = os.getcwd()
CKPT_DIR = "experiments/checkpoints"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/processed/cath_s40.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--ckpt_suffix", type=str, default="_v7")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    dataset_path = os.path.join(BASE_DIR, args.dataset)
    dataset = TriPlaneDataset(dataset_path, grid_size=128, voxel_size=0.6, sigma=1.0)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # 1. Load Pre-trained AutoEncoder (Must match train_triplane.py)
    encoder = TriPlaneEncoder(channels=128, plane_res=128, signal_scale=4.0).to(device)
    ckpt_dir = os.path.join(BASE_DIR, CKPT_DIR)
    encoder_path = os.path.join(ckpt_dir, f"triplane_encoder{args.ckpt_suffix}.pth")
    
    if os.path.exists(encoder_path):
        print(f"Loading pre-trained encoder from {encoder_path}...")
        sd = torch.load(encoder_path, map_location=device)
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        encoder.load_state_dict(sd)
    encoder.eval()

    # 2. Diffusion Setup
    unet = UNet2D(in_channels=384, model_channels=128, out_channels=384).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=1000).to(device)
    
    # --- Auto-Resume Logic ---
    diffusion_path = os.path.join(ckpt_dir, f"ddpm_triplane_2d{args.ckpt_suffix}.pth")
    meta_path      = os.path.join(ckpt_dir, f"diffusion_meta{args.ckpt_suffix}.pth")
    start_epoch = 1
    
    if os.path.exists(diffusion_path):
        print(f"Resuming Diffusion from {diffusion_path}...")
        sd = torch.load(diffusion_path, map_location=device)
        if any(k.startswith('_orig_mod.') for k in sd.keys()):
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        unet.load_state_dict(sd)
        
        if os.path.exists(meta_path):
            start_epoch = torch.load(meta_path).get('epoch', 0) + 1
            print(f"Resuming from Epoch {start_epoch}")

    opt   = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    print("Pre-computing latent planes for fast training...")
    latents = []
    for vol_gt, _, _ in tqdm(loader, desc="Encoding Latents"):
        vol_gt = vol_gt.to(device)
        with torch.no_grad():
            planes_gt = encoder(vol_gt)
            x_0 = torch.cat(planes_gt, dim=1)
            latents.append(x_0.cpu())

    latent_dataset = TensorDataset(torch.cat(latents, dim=0))
    latent_loader  = DataLoader(latent_dataset, batch_size=16, shuffle=True, pin_memory=True)

    if torch.cuda.get_device_capability()[0] >= 8:
        unet = torch.compile(unet)

    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        total_loss = 0.0
        pbar = tqdm(latent_loader, desc=f"Diffusion Epoch {epoch}/{args.epochs}")

        for (x_0,) in pbar:
            x_0 = x_0.to(device)
            t   = torch.randint(0, 1000, (x_0.shape[0],), device=device).long()
            x_t, noise = diffusion.forward_diffusion(x_0, t)

            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                noise_pred_list = unet(x_t, t)
                noise_pred      = torch.cat(noise_pred_list, dim=1)
                loss            = F.mse_loss(noise_pred, noise)
            
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        sched.step()

        if epoch % 50 == 0 or epoch == 1 or epoch == start_epoch:
            avg = total_loss / len(latent_loader)
            print(f"Epoch {epoch:3d} | loss={avg:.4f} | lr={sched.get_last_lr()[0]:.2e}")
            
            import threading
            def async_save(state_dict, path):
                cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
                threading.Thread(target=lambda: torch.save(cpu_state, path), daemon=True).start()

            u_sd = getattr(unet, '_orig_mod', unet).state_dict()
            async_save(u_sd, diffusion_path)
            torch.save({'epoch': epoch}, meta_path)

    print("Diffusion Training Complete.")

if __name__ == "__main__":
    train()
