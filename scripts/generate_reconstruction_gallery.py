"""
Phase 6: Tri-Plane NeRF - Structural Reconstruction Gallery
Demonstrates the conditional prior's ability to recover high-fidelity 
3D electron density from noisy latent observations using Tri-Plane INRs.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.triplane_encoder import TriPlaneEncoder
from models.triplane import TriPlaneDecoder
from models.unet_2d import TriPlaneUNet
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset

# Publication dark style
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial'],
    'axes.facecolor': '#0d1117',
    'figure.facecolor': '#0d1117',
    'text.color': '#c9d1d9',
})

CKPT_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cath_subset.pt")

PROTEINS = {
    '1hel': 'Lysozyme\n(simple fold)',
    '1ubq': 'Ubiquitin\n(beta-grasp)',
    '2vii': 'Villin HP36\n(3-helix bundle)',
    '2cro': 'Cro Repressor\n(helix-turn-helix)',
    '1igd': 'Protein G\n(β1 domain)',
    '1r69': 'Engrailed HD\n(homeodomain)',
}

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Structural Reconstruction Gallery on {device}")

    # Load v4 encoder + decoder (Stable Baseline)
    encoder = TriPlaneEncoder(channels=32, plane_res=128, signal_scale=4.0).to(device)
    encoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_encoder_v4.pth', map_location=device))
    encoder.eval()

    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_decoder_v4.pth', map_location=device))
    decoder.eval()

    # Load Stage-1 diffusion prior (v2 stable)
    unet = TriPlaneUNet(plane_channels=32).to(device)
    unet.load_state_dict(torch.load(f'{CKPT_DIR}/ddpm_triplane_2d_v2.pth', map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, 1000).to(device)

    data_all = torch.load(DATA_PATH, weights_only=False)
    L = 128
    L_lr = 64

    def decode_vol(planes, res):
        grid_1d = torch.linspace(-1.0, 1.0, res, device=device)
        zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        q = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
        with torch.no_grad():
            vol = decoder(planes, q).reshape(res, res, res)
        return vol

    def mip(vol):
        return vol.max(axis=2).values.cpu().float().numpy()

    def normalize(img):
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8: return img
        return (img - mn) / (mx - mn)

    # ── Figure layout: 6 proteins × 3 columns (GT | AE Recon | Diff-Denoised)
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    fig.patch.set_facecolor('#0d1117')
    outer = gridspec.GridSpec(3, 6, figure=fig, hspace=0.08, wspace=0.06)

    col_labels = ["Ground Truth", "AE Reconstruction", "Diffusion-Denoised"]
    col_colors = ['#8b949e', '#58a6ff', '#ff7b54']

    keys = list(PROTEINS.keys())

    for idx, pdb_id in enumerate(keys):
        row = idx // 2
        col_base = (idx % 2) * 3

        coords = data_all[pdb_id].to(device)
        coords -= coords.mean(dim=0)
        vol_gt = VolumeDataset.voxelize_gaussian(coords, L, 0.6, 0.6).to(device)
        gt_mip = normalize(mip(vol_gt))

        # Autoencoder Reconstruction (direct encode → decode)
        with torch.no_grad():
            planes_hr = encoder(vol_gt.unsqueeze(0).unsqueeze(0))
            vol_ae = decode_vol(planes_hr, L)
            ae_mip = normalize(mip(vol_ae))

            # Diffusion-denoised: encode, add noise at t=500, denoise with prior
            x_0 = torch.cat(planes_hr, dim=1)  # (1, 96, 128, 128)
            # Downsample to 64 for Stage-1 prior
            x_0_lr = F.interpolate(x_0, size=(L_lr, L_lr), mode='bilinear', align_corners=True)
            t_noise = torch.tensor([500], device=device).long()
            noise = torch.randn_like(x_0_lr)
            sqrt_alpha = diffusion.sqrt_alphas_cumprod[t_noise].view(1,1,1,1)
            sqrt_1m_alpha = diffusion.sqrt_one_minus_alphas_cumprod[t_noise].view(1,1,1,1)
            x_t = sqrt_alpha * x_0_lr + sqrt_1m_alpha * noise

            # Denoise from t=500 → 0 using the diffusion prior
            for i in reversed(range(0, 500)):
                t_step = torch.full((1,), i, device=device, dtype=torch.long)
                eps = unet(x_t, t_step)
                eps = torch.cat(eps, dim=1)
                beta_t = diffusion.betas[i]
                coeff1 = 1 / torch.sqrt(1 - beta_t)
                coeff2 = beta_t / torch.sqrt(1 - diffusion.alphas_cumprod[i])
                x_t = coeff1 * (x_t - coeff2 * eps)
                if i > 0:
                    x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)
                x_t.clamp_(-6.0, 6.0)

            # Upsample denoised latent back to 128, decode
            x_denoised_hr = F.interpolate(x_t, size=(L, L), mode='bilinear', align_corners=True)
            planes_dn = [x_denoised_hr[:,0:32], x_denoised_hr[:,32:64], x_denoised_hr[:,64:96]]
            vol_dn = decode_vol(planes_dn, L)
            dn_mip = normalize(mip(vol_dn))

        imgs   = [gt_mip, ae_mip, dn_mip]
        cmaps  = ['Greys_r', 'inferno', 'magma']

        for ci, (img, cmap, label, clr) in enumerate(zip(imgs, cmaps, col_labels, col_colors)):
            ax = fig.add_subplot(outer[row, col_base + ci])
            
            # Zoom logic: crop to center 64x64 if zoom is high
            h, w = img.shape
            ch, cw = h // 2, w // 2
            s = 32 # size to keep
            img_zoomed = img[ch-s:ch+s, cw-s:cw+s]
            
            ax.imshow(np.clip(img_zoomed, 0, 1) ** 0.6, cmap=cmap, interpolation='bilinear')
            ax.axis('off')
            if row == 0 and idx % 2 == 0:
                ax.set_title(label, color=clr, fontsize=11, pad=8, fontweight='bold')
            if ci == 0:
                ax.text(-0.05, 0.5, f"{pdb_id.upper()}\n{PROTEINS[pdb_id]}",
                        transform=ax.transAxes, color='#8b949e',
                        fontsize=8, va='center', ha='right', rotation=0)

    plt.suptitle(
        "Tri-Plane Prior: Structural Reconstruction at 128³ Atomic Resolution",
        color='white', fontsize=16, y=1.01, fontweight='light'
    )

    out = os.path.join(BASE_DIR, "experiments", "results", "reconstruction_gallery.png")
    plt.savefig(out, dpi=280, facecolor='#0d1117', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    run()
