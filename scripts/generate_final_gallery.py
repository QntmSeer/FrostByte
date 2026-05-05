"""
Phase 20: Final Gallery — Structural Reconstruction with Proper Scale (coordinate_scale=20)
GT | AE Reconstruction | Diffusion-Sampled Prior
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

plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica', 'Arial'],
    'axes.facecolor': '#0a0e17',
    'figure.facecolor': '#0a0e17',
    'text.color': '#c9d1d9',
})

CKPT_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cath_subset.pt")

# Coord scale used during training
COORD_SCALE = 20.0
SIGMA = 2.0
L = 128
VS = 0.6

PROTEINS = {
    '1hel': 'Lysozyme (129 aa)',
    '1ubq': 'Ubiquitin (76 aa)',
    '2vii': 'Villin HP36 (36 aa)',
    '2cro': 'Cro Repressor (65 aa)',
    '1igd': 'Protein G B1 (56 aa)',
    '1r69': 'Engrailed HD (61 aa)',
}

def decode_vol(decoder, planes, device, res=128):
    """Decode tri-plane latent to a density MIP image."""
    grid_1d = torch.linspace(-1.0, 1.0, res, device=device)
    zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    q = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
    with torch.no_grad():
        vol = decoder(planes, q).reshape(res, res, res)
    mip = vol.max(dim=2).values.cpu().float().numpy()
    return mip

def norm_gamma(img, gamma=0.75):
    img = np.clip(img, 0, None)
    mx = img.max()
    if mx < 1e-8: return img
    return (img / mx) ** gamma

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating v5 (Scale-Fixed) Gallery on {device}")

    # --- Models (v5 = retrained with coordinate_scale=20) ---
    encoder = TriPlaneEncoder(channels=32, plane_res=128, signal_scale=4.0).to(device)
    encoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_encoder_v5.pth', map_location=device))
    encoder.eval()

    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_decoder_v5.pth', map_location=device))
    decoder.eval()

    # Diffusion prior (Stage-1, trained on v5 latents)
    unet = TriPlaneUNet(plane_channels=32).to(device)
    unet.load_state_dict(torch.load(f'{CKPT_DIR}/ddpm_triplane_2d_v5.pth', map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, 1000).to(device)

    data_all = torch.load(DATA_PATH, weights_only=False)
    keys = list(PROTEINS.keys())

    # 3 columns: Ground Truth | AE Reconstruction | Diffusion Sampled
    fig = plt.figure(figsize=(15, 13), facecolor='#0a0e17')
    gs_outer = gridspec.GridSpec(3, 6, figure=fig, hspace=0.12, wspace=0.05,
                                  left=0.08, right=0.97, top=0.92, bottom=0.04)

    col_titles  = ['Ground Truth', 'AE Reconstruction', 'Diffusion Prior']
    col_cmaps   = ['magma',         'inferno',            'plasma']
    col_colors  = ['#8b949e',       '#58a6ff',            '#ff7b54']

    for p_idx, pdb_id in enumerate(keys):
        row      = p_idx // 2
        col_base = (p_idx % 2) * 3

        # --- Ground Truth ---
        coords = data_all[pdb_id].to(device) * COORD_SCALE
        coords -= coords.mean(0)
        vol_gt = VolumeDataset.voxelize_gaussian(coords, L, VS, SIGMA)
        gt_mip = norm_gamma(vol_gt.max(2).values.cpu().numpy())

        # --- AE Reconstruction (encode → decode) ---
        with torch.no_grad():
            vol_in = vol_gt.to(device).unsqueeze(0).unsqueeze(0)
            planes_ae = encoder(vol_in)
            ae_mip    = norm_gamma(decode_vol(decoder, planes_ae, device))

        # --- Diffusion Prior Sample (unconditional generation) ---
        with torch.no_grad():
            x_sample = diffusion.sample((1, 96, 128, 128), device=device)
            planes_s  = [x_sample[:, 0:32], x_sample[:, 32:64], x_sample[:, 64:96]]
            diff_mip  = norm_gamma(decode_vol(decoder, planes_s, device))

        imgs = [gt_mip, ae_mip, diff_mip]

        for ci, (img, cmap, ttl, clr) in enumerate(zip(imgs, col_cmaps, col_titles, col_colors)):
            ax = fig.add_subplot(gs_outer[row, col_base + ci])
            ax.imshow(img, cmap=cmap, interpolation='bilinear', vmin=0, vmax=1)
            ax.axis('off')

            # Column headers for first protein in each pair
            if row == 0 and p_idx == 0:
                ax.set_title(ttl, color=clr, fontsize=11, pad=7, fontweight='bold')

            # Protein label on GT column only
            if ci == 0:
                ax.text(0.02, 0.97, f"{pdb_id.upper()}",
                        transform=ax.transAxes, color='white',
                        fontsize=9, va='top', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0e17', alpha=0.7, edgecolor='none'))
                ax.text(0.02, 0.84, PROTEINS[pdb_id],
                        transform=ax.transAxes, color='#8b949e',
                        fontsize=7.5, va='top')

    fig.suptitle(
        'Tri-Plane Structural Prior  ·  128³ Resolution  ·  Root-Cause-Fixed (v5)',
        color='white', fontsize=14, fontweight='light', y=0.97
    )

    out = os.path.join(BASE_DIR, "experiments", "results", "gallery_v5_final.png")
    plt.savefig(out, dpi=280, facecolor='#0a0e17', bbox_inches='tight')
    print(f"Saved: {out}")

if __name__ == "__main__":
    run()
