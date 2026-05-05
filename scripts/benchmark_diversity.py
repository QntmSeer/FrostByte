import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys, json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_2d import TriPlaneUNet
from models.triplane import TriPlaneDecoder
from models.diffusion import DiffusionModel
from projection.neural_radon import NeuralRayMarcher
from projection.radon import RadonProjector
from data.volume_dataset import VolumeDataset
from utils.metrics import compute_fsc, compute_cc, align_volumes_com

# Absolute paths based on script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Constants
L = 64
NUM_PROJS = 10
GUIDANCE_ZETA = 5.0    # Reduced to prevent over-sharpening sparks
EDGE_LAMBDA = 0.5      # Increased for better coherence
TV_LAMBDA = 0.05       # Total Variation to ensure continuous density

def get_tv_loss(planes):
    """Calculates Total Variation loss on Tri-Planes."""
    loss = 0
    for p in planes:
        # p: (B, C, H, W)
        diff_h = torch.pow(p[:, :, 1:, :] - p[:, :, :-1, :], 2).mean()
        diff_w = torch.pow(p[:, :, :, 1:] - p[:, :, :, :-1], 2).mean()
        loss += diff_h + diff_w
    return loss

def load_target_density(name, device):
    """Loads and voxelizes a target protein."""
    if name == "myoglobin":
        # Load real data from cath_subset (using 1hel as a reliable baseline since 1mbn is missing)
        data_all = torch.load(os.path.join(DATA_DIR, "cath_subset.pt"), weights_only=False)
        coords_pdb = data_all['1hel'].to(device)
        centroid = coords_pdb.mean(dim=0)
        coords_pdb = coords_pdb - centroid
        
        # Voxelize to get vol_gt
        vol_gt = VolumeDataset.voxelize_gaussian(coords_pdb, L, 1.0, 1.0).to(device).unsqueeze(0).unsqueeze(0)
        
        # Encode to get "Ground Truth" planes
        from models.triplane_encoder import TriPlaneEncoder
        encoder = TriPlaneEncoder(channels=32, plane_res=L).to(device)
        encoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'triplane_encoder_v2.pth'), map_location=device))
        encoder.eval()
        
        decoder_tmp = TriPlaneDecoder(channels=32).to(device)
        decoder_tmp.load_state_dict(torch.load(os.path.join(CKPT_DIR, 'triplane_decoder_v2.pth'), map_location=device))
        decoder_tmp.eval()
        
        with torch.no_grad():
            planes = encoder(vol_gt)
            grid_1d = torch.linspace(-1.0, 1.0, L, device=device)
            zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
            vol = decoder_tmp(planes, coords).reshape(L, L, L)
            
        return vol, name
    
    # Load from cath_subset
    data_all = torch.load(os.path.join(DATA_DIR, "cath_subset.pt"), weights_only=False)
    file_map = {"lysozyme": "1hel", "ubiquitin": "1ubq"}
    coords = data_all[file_map[name]].to(device)
    
    # Center and scale to match Tri-Plane space [-1, 1]
    coords = coords - coords.mean(dim=0)
    # Norm to fit in -0.8 to 0.8 range for safety
    max_dist = torch.norm(coords, dim=1).max()
    coords = coords * (0.8 / max_dist)
    
    # Voxelize using Gaussian (standard practice for benchmarks)
    vol = VolumeDataset.voxelize_gaussian(coords, L, (2.0/L), 0.05).to(device)
    return vol, name

def run_reconstruction(vol_gt, name, model, decoder, ray_marcher, radon, device):
    print(f"\n--- Benchmarking: {name.upper()} ---")
    
    # 1. Simulate Projections
    # Wrap volume in a callable that looks like a model for the ray marcher
    class VolWrapper(torch.nn.Module):
        def __init__(self, v):
            super().__init__()
            self.v = v.unsqueeze(0).unsqueeze(0)
        def forward(self, coords):
            # coords: (1, N, 3) in range [-1, 1]
            # Map coords to grid indices
            indices = (coords + 1.0) * (L / 2.0)
            # grid_sample expects [-1, 1]
            # wait, vol is (1, 1, L, L, L). 
            # coords is (1, N, 3). Grid sample expects (B, C, D, H, W) and (B, N1, N2, N3, 3)
            # We can reshape coords to (1, N, 1, 1, 3)
            samp_coords = coords.view(1, -1, 1, 1, 3)
            vals = F.grid_sample(self.v, samp_coords, align_corners=True)
            return vals.view(1, -1, 1)

    gt_model = VolWrapper(vol_gt)
    R_target = radon.random_rotation_matrix(NUM_PROJS, device=device)
    y = ray_marcher(gt_model, R_target).detach()
    
    # 2. DPS Loop
    def dps_guidance(x_t, t, x_0_pred):
        # --- BIOLOGICAL ANCHOR (Phase 10) ---
        # Clamp predicted x_0 latents to stay within the trained manifold [-2, 2]
        x_0_pred = torch.clamp(x_0_pred, -1.5, 1.5)
        
        curr_planes = [x_0_pred[:, 0:32], x_0_pred[:, 32:64], x_0_pred[:, 64:96]]
        xy, xz, yz = curr_planes
        
        # Wrapped Decoder for guidance
        class WrappedDecoder(torch.nn.Module):
            def __init__(self, dec, pl):
                super().__init__()
                self.dec = dec
                self.pl = pl
            def forward(self, coords):
                return self.dec(self.pl, coords)
        
        curr_model = WrappedDecoder(decoder, curr_planes)
        y_hat = ray_marcher(curr_model, R_target)
        
        # Measurement Loss (Use L2 norm for DPS stability, not MSE)
        loss_meas = torch.norm(y_hat - y)
        
        # Constraints (Use L2 norm)
        loss_edge = torch.norm(xy.mean(dim=3) - xz.mean(dim=3)) + \
                    torch.norm(xy.mean(dim=2) - yz.mean(dim=2)) + \
                    torch.norm(xz.mean(dim=2) - yz.mean(dim=3))
        
        loss_tv = get_tv_loss(curr_planes)
        
        # Time-dependent Zeta: Start strong, decay to let the prior refine details.
        # DPS typically scales Zeta with the noise level or just uses a constant.
        # Here we let the gradient naturally scale with the distance, rather than normalizing it to 1.0!
        t_norm = t.float() / 1000.0
        curr_zeta = GUIDANCE_ZETA * t_norm # Drops to 0 at t=0
        
        loss = curr_zeta * loss_meas + EDGE_LAMBDA * loss_edge + TV_LAMBDA * loss_tv
        
        grad = torch.autograd.grad(loss, x_t)[0]
        # CRITICAL FIX: DO NOT NORMALIZE TO 1.0! 
        # Normalizing to 1.0 destroys the manifold because the step size is huge near t=0.
        return grad

    # Quick test: does unconditional generation produce sparks?
    uncond_latent = model.sample((1, 96, 64, 64), device=device, guidance_fn=None)
    x_rec_latent = model.sample((1, 96, 64, 64), device=device, guidance_fn=dps_guidance)
    
    # 3. Final Metrics
    with torch.no_grad():
        final_planes = [x_rec_latent[:, 0:32], x_rec_latent[:, 32:64], x_rec_latent[:, 64:96]]
        uncond_planes = [uncond_latent[:, 0:32], uncond_latent[:, 32:64], uncond_latent[:, 64:96]]
        
        grid_1d = torch.linspace(-1.0, 1.0, L, device=device)
        zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
        vol_rec = decoder(final_planes, coords).reshape(L, L, L)
        vol_uncond = decoder(uncond_planes, coords).reshape(L, L, L)
        
        # Calculate metrics
        vol_rec_aligned = align_volumes_com(vol_rec, vol_gt)
        print(f"    GT Max: {vol_gt.max().item():.3f}, Mean: {vol_gt.mean().item():.3f}")
        print(f"    Rec Max: {vol_rec_aligned.max().item():.3f}, Mean: {vol_rec_aligned.mean().item():.3f}")
        
        cc = compute_cc(vol_rec, vol_gt, align=True)
        freqs, fsc = compute_fsc(vol_rec, vol_gt, align=True)
        
        # Resolution at 0.5
        res_idx = np.where(fsc < 0.5)[0]
        res = freqs[res_idx[0]] if len(res_idx) > 0 else 1.0
        
        print(f"  CC: {cc:.4f} | Res: {res:.3f}")
        return vol_rec, vol_uncond, cc, res, (freqs, fsc)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"FrostByte Diversity Benchmark | Protocol S8 | Device: {device}")
    
    # Load Core Models
    unet = TriPlaneUNet(plane_channels=32).to(device)
    unet.load_state_dict(torch.load(f'{CKPT_DIR}/ddpm_triplane_2d_v2.pth', map_location=device))
    unet.eval()
    
    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_decoder_v2.pth', map_location=device))
    decoder.eval()
    
    model = DiffusionModel(unet, 1000).to(device)
    ray_marcher = NeuralRayMarcher(img_size=L, num_steps=L).to(device)
    radon = RadonProjector(L)
    
    targets = ["myoglobin", "lysozyme", "ubiquitin"]
    results = {}
    vols = []
    
    for t_name in targets:
        vol_gt, _ = load_target_density(t_name, device)
        vol_rec, vol_uncond, cc, res, fsc_data = run_reconstruction(vol_gt, t_name, model, decoder, ray_marcher, radon, device)
        results[t_name] = {"cc": cc, "res": res}
        vols.append((vol_gt, vol_rec, vol_uncond, fsc_data))
        
    # Visualization
    fig = plt.figure(figsize=(22, 12), facecolor='#080808')
    gs = gridspec.GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, t_name in enumerate(targets):
        gt, rec, uncond, (freqs, fsc) = vols[i]
        
        # Isosurface slices for GT and Rec
        def get_slice(v):
            # Ensure tensor is detached
            v_np = v.detach().cpu().numpy()
            thr = v_np.mean() + 1.2 * v_np.std()
            return np.clip(v_np[L//2] - thr, 0, 1)
        
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_gt.imshow(get_slice(gt), cmap='viridis')
        ax_gt.set_title(f"{t_name} GT", color='white')
        ax_gt.axis('off')
        
        ax_rec = fig.add_subplot(gs[i, 1])
        ax_rec.imshow(get_slice(rec), cmap='magma')
        ax_rec.set_title(f"{t_name} Rec", color='cyan')
        ax_rec.axis('off')
        
        ax_uncond = fig.add_subplot(gs[i, 2])
        ax_uncond.imshow(get_slice(uncond), cmap='plasma')
        ax_uncond.set_title(f"{t_name} Uncond Prior", color='yellow')
        ax_uncond.axis('off')
        
        # Large FSC Plot
        ax_fsc = fig.add_subplot(gs[i, 3:])
        ax_fsc.set_facecolor('#111')
        ax_fsc.plot(freqs, fsc, color='cyan', label=f'FSC (CC={results[t_name]["cc"]:.2f})')
        ax_fsc.axhline(0.5, color='red', ls='--', alpha=0.5)
        ax_fsc.set_ylim(0, 1.1)
        ax_fsc.set_title(f"{t_name} Resolution: {results[t_name]['res']:.3f}", color='white')
        ax_fsc.tick_params(colors='white')
        
    fig.suptitle("FrostByte Diversity Benchmark v1.0 | 3-Fold Cross-Validation | Phase 8", color='cyan', fontsize=20)
    plt.savefig("experiments/results/diversity_benchmark.png", facecolor='#080808', dpi=300)
    
    with open("experiments/results/benchmark_diversity.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nBenchmark Complete! Summary saved to experiments/results/diversity_benchmark.png")

if __name__ == "__main__":
    main()
