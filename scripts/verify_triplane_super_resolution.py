import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_2d import TriPlaneUNet
from models.triplane import TriPlaneDecoder
from models.triplane_encoder import TriPlaneEncoder
from models.diffusion import DiffusionModel

def isosurface_threshold(vol, sigma_level=1.5):
    """
    Hard isosurface threshold — the standard approach used in UCSF ChimeraX and RELION.
    Zeroes out anything below mean + sigma * std. 
    Then normalizes the surviving density to [0, 1].
    This is how all gold-standard cryo-EM software presents density maps.
    """
    mean = vol.mean()
    std  = vol.std()
    threshold = mean + sigma_level * std
    out = np.where(vol >= threshold, vol - threshold, 0.0)
    if out.max() > 0:
        out = out / out.max()
    return np.clip(out, 0, 1)

def verify_triplane():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Phase 6: Tri-Plane NeRF Density Models...")

    unet = TriPlaneUNet(plane_channels=32, time_dim=64).to(device)
    unet.load_state_dict(torch.load('experiments/checkpoints/ddpm_triplane_2d_v2.pth', map_location=device))
    unet.eval()
    diffusion = DiffusionModel(unet, 1000).to(device)

    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load('experiments/checkpoints/triplane_decoder_v2.pth', map_location=device))
    decoder.eval()

    with torch.no_grad():
        print("Sampling Latent Tri-Planes from diffusion model...")
        planes_latent = diffusion.sample((1, 96, 64, 64), device=device)
        planes = [planes_latent[:, 0:32], planes_latent[:, 32:64], planes_latent[:, 64:96]]
        print(f"  Planes XY: mean={planes[0].mean():.3f}, std={planes[0].std():.3f}")

        # Decode at 96^3 — fine enough to show continuous structure
        L = 96
        print(f"Decoding continuous {L}^3 volume...")
        grid_1d = torch.linspace(-1.0, 1.0, L, device=device)
        zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
        queries_flat = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)

        chunk_size = L * L
        dens_chunks = []
        for i in range(0, queries_flat.shape[1], chunk_size):
            c = decoder(planes, queries_flat[:, i:i + chunk_size])
            dens_chunks.append(c.cpu())

        vol = torch.cat(dens_chunks, dim=1).reshape(L, L, L).numpy()
        print(f"  Raw: max={vol.max():.3f}  mean={vol.mean():.4f}  std={vol.std():.4f}")

        # Hard isosurface — ChimeraX/RELION style
        vol_iso = isosurface_threshold(vol, sigma_level=1.5)
        n_active = (vol_iso > 0).sum()
        print(f"  Iso (sigma=1.5): {n_active} active voxels / {L**3} total ({100*n_active/L**3:.1f}%)")

        mid = L // 2
        # Canonical orthogonal cross-sections (same plane as RELION slice viewer)
        sl_xy = vol_iso[mid, :, :]    # mid-Z
        sl_xz = vol_iso[:, mid, :]    # mid-Y
        sl_yz = vol_iso[:, :, mid]    # mid-X

        # ---- Publication-style 3-panel orthogonal layout (ChimeraX / RELION) ----
        fig = plt.figure(figsize=(15, 5.2), facecolor='#080808')
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.035, left=0.02, right=0.98,
                               top=0.88, bottom=0.02)

        panels = [
            (sl_xy, f'XY  ·  Z={mid}'),
            (sl_xz, f'XZ  ·  Y={mid}'),
            (sl_yz, f'YZ  ·  X={mid}'),
        ]

        for col, (sl, label) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            ax.set_facecolor('#080808')

            ax.imshow(sl, cmap='inferno', origin='lower',
                      vmin=0, vmax=1,
                      extent=[0, L, 0, L],
                      aspect='equal',
                      interpolation='bilinear')   # smooth, no pixelation

            # RELION-style thin grey border
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')
                spine.set_linewidth(0.7)
            ax.set_xlim(0, L)
            ax.set_ylim(0, L)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_title(label, color='#aaaaaa', fontsize=9, pad=5, fontfamily='monospace')

        fig.suptitle(
            'Phase 6 · Tri-Plane NeRF · Continuous $96^3$ Volume · $\\sigma=1.5$ Isosurface',
            color='#dddddd', fontsize=10, y=0.975, fontfamily='monospace'
        )

        out = 'experiments/sandbox/triplane_super_resolution.png'
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='#080808', edgecolor='none')
        print(f"Saved -> {out}")

if __name__ == "__main__":
    verify_triplane()
