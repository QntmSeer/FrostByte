import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Internal project imports
from data.triplane_dataset import TriPlaneDataset
from models.triplane_encoder import TriPlaneEncoder
from models.triplane import TriPlaneDecoder

# Constants
BASE_DIR = os.getcwd()
CKPT_DIR = "experiments/checkpoints"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/processed/cath_s40.pt")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--ckpt_suffix", type=str, default="_v7")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True # Production optimization
    
    dataset_path = os.path.join(BASE_DIR, args.dataset)
    dataset = TriPlaneDataset(dataset_path, 
                              num_samples=16384, grid_size=128, voxel_size=0.6, sigma=1.0,
                              coordinate_scale=1.0, augment=False)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    encoder = TriPlaneEncoder(channels=128, plane_res=128, signal_scale=4.0).to(device)
    decoder = TriPlaneDecoder(channels=128).to(device)

    params  = list(encoder.parameters()) + list(decoder.parameters())
    opt     = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    # --- Auto-Resume Logic ---
    ckpt_dir = os.path.join(BASE_DIR, CKPT_DIR)
    encoder_path = os.path.join(ckpt_dir, f"triplane_encoder{args.ckpt_suffix}.pth")
    decoder_path = os.path.join(ckpt_dir, f"triplane_decoder{args.ckpt_suffix}.pth")
    meta_path    = os.path.join(ckpt_dir, f"triplane_meta{args.ckpt_suffix}.pth")
    start_epoch = 1

    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print(f"Resuming AutoEncoder from {encoder_path}...")
        
        def robust_load(model, path):
            sd = torch.load(path, map_location=device)
            if any(k.startswith('_orig_mod.') for k in sd.keys()):
                sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd)

        robust_load(encoder, encoder_path)
        robust_load(decoder, decoder_path)
        
        if os.path.exists(meta_path):
            meta = torch.load(meta_path)
            start_epoch = meta.get('epoch', 0) + 1
            print(f"Resuming from Epoch {start_epoch}")

    epochs  = args.epochs
    # last_epoch should be start_epoch - 2 because it is 0-indexed internally and incremented on step
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    alpha   = 100.0
    scaler  = torch.amp.GradScaler('cuda')

    print("Pre-computing AE samples...")
    ae_data = []
    for vol_gt, coords, density_gt in tqdm(loader, desc="Caching AE Data"):
        ae_data.append((vol_gt.cpu(), coords.cpu(), density_gt.cpu()))

    ae_dataset = torch.utils.data.TensorDataset(
        torch.cat([item[0] for item in ae_data], dim=0),
        torch.cat([item[1] for item in ae_data], dim=0),
        torch.cat([item[2] for item in ae_data], dim=0)
    )
    
    ae_batched_loader = torch.utils.data.DataLoader(
        ae_dataset, 
        batch_size=8, 
        shuffle=True, 
        drop_last=True,
        pin_memory=True
    )

    if torch.cuda.get_device_capability()[0] >= 8:
        print("Compiling models with torch.compile (Triton)...")
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)

    for epoch in range(start_epoch, epochs + 1):
        encoder.train(); decoder.train()
        total_loss = 0.0
        pbar = tqdm(ae_batched_loader, desc=f"Epoch {epoch}/{epochs}")

        for vol_gt, coords, density_gt in pbar:
            vol_gt      = vol_gt.to(device)
            coords      = coords.to(device)
            density_gt  = density_gt.to(device)

            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                planes       = encoder(vol_gt)                  
                density_pred = decoder(planes, coords)          
                weights  = 1.0 + alpha * (density_gt ** 2)
                loss     = (F.mse_loss(density_pred, density_gt, reduction='none') * weights).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        sched.step()
        avg = total_loss / len(ae_batched_loader)
        
        if epoch % 25 == 0 or epoch == 1 or epoch == start_epoch:
            print(f"Epoch {epoch:3d} | loss={avg:.4f} | lr={sched.get_last_lr()[0]:.2e}")
            
            import threading
            def async_save(state_dict, path):
                cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
                threading.Thread(target=lambda: torch.save(cpu_state, path), daemon=True).start()

            # Save uncompiled state dict
            enc_sd = getattr(encoder, '_orig_mod', encoder).state_dict()
            dec_sd = getattr(decoder, '_orig_mod', decoder).state_dict()
            async_save(enc_sd, encoder_path)
            async_save(dec_sd, decoder_path)
            torch.save({'epoch': epoch}, meta_path)

    print("Training Complete.")

if __name__ == "__main__":
    train()
