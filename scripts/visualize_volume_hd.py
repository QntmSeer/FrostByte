import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset
from projection.radon import RadonProjector

from scripts.verify_volume_reconstruction import reconstruct_volume_dps

def generate_animated_reconstruction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating HD Animation on {device}")
    
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # 1. Load Overfitted Model
    net = UNet3D(in_ch=1, out_ch=1, time_dim=64)
    model = DiffusionModel(net, timesteps=1000).to(device)
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_volume_overfit.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Get Ground Truth (Lysozyme 1HEL)
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    data_dict = torch.load(data_path, weights_only=False)
    coords = data_dict['1hel'] * 10.0
    coords = coords - coords.mean(dim=0, keepdim=True)
    vol_gt = VolumeDataset.voxelize_gaussian(coords, 64, 1.0, 1.0).to(device)
    vol_gt = vol_gt.unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64, 64)
    
    # 3. Simulate Projection
    projector = RadonProjector(64).to(device)
    R = projector.random_rotation_matrix(3, device=device)
    with torch.no_grad():
        x_rep = vol_gt.repeat(3, 1, 1, 1, 1)
        y_meas = projector(x_rep, R)
        
    print("Reconstructing...")
    # Using step_scale 2.0 to balance sharpness while preventing background noise explosion
    vol_rec = reconstruct_volume_dps(model, projector, y_meas, R, device, step_scale=2.0)
    
    # 4. Process Volumes for HD Visualization
    v_gt = vol_gt[0, 0].cpu().numpy()
    v_rec = vol_rec[0, 0].cpu().numpy()
    proj = y_meas[0, 0].cpu().numpy()
    
    def threshold(v):
        # 1. Apply a spherical mask to kill all boundary noise
        z, y, x = np.ogrid[0:64, 0:64, 0:64]
        mask = (x - 32)**2 + (y - 32)**2 + (z - 32)**2 <= 28**2
        v = v * mask

        # 2. Thresholding logic
        v = np.clip(v, 0, None)
        p99 = np.percentile(v, 99.5)
        if p99 > 0:
            v = v / p99
            
        v = np.clip(v, 0, 1.0)
        v[v < 0.15] = 0.0 # Slightly harsher cutoff for remaining noise
        return v
        
    v_gt = threshold(v_gt)
    v_rec = threshold(v_rec)
    
    # 5. Animate Z-Slices
    print("Creating GIF...")
    fig, ax = plt.subplots(1, 3, figsize=(20, 7), facecolor='black')
    
    for a in ax:
        a.axis('off')
        a.set_facecolor('black')
        
    # Colormap 'magma' looks great on black background
    cmap = 'magma'
    
    # Left panel: Static Projection
    ax[0].imshow(proj, cmap='gray')
    ax[0].set_title("Input 2D Projection", color='white', fontsize=16, pad=15)
    
    # Middle: GT Slice
    im_gt = ax[1].imshow(v_gt[32], cmap=cmap, vmin=0, vmax=1)
    title_gt = ax[1].set_title("Ground Truth 3D Volume (Z=32)", color='white', fontsize=16, pad=15)
    
    # Right: Rec Slice
    im_rec = ax[2].imshow(v_rec[32], cmap=cmap, vmin=0, vmax=1)
    title_rec = ax[2].set_title("ML Reconstruction 3D (Z=32)", color='white', fontsize=16, pad=15)
    
    fig.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98, wspace=0.1)
    
    def update(z):
        im_gt.set_array(v_gt[z])
        title_gt.set_text(f"Ground Truth 3D Volume (Z={z:02d})")
        
        im_rec.set_array(v_rec[z])
        title_rec.set_text(f"ML Reconstruction 3D (Z={z:02d})")
        return [im_gt, title_gt, im_rec, title_rec]
        
    anim = animation.FuncAnimation(fig, update, frames=range(10, 54), interval=100, blit=True)
    
    save_path_gif = os.path.join(base, "experiments", "sandbox", "volume_reconstruction_hd.gif")
    anim.save(save_path_gif, writer='pillow', fps=10)
    print(f"Saved HD GIF to {save_path_gif}")
    
    # Save as MP4 for optimal LinkedIn playback (GIFs get heavily compressed by LinkedIn)
    try:
        import imageio
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        frames = []
        canvas = FigureCanvas(fig)
        
        for z in range(10, 54):
            update(z)
            canvas.draw()
            # Convert canvas to RGB array (drop alpha channel)
            rgba = np.asarray(canvas.buffer_rgba())
            img = rgba[:, :, :3]
            frames.append(img)
            
        save_path_mp4 = os.path.join(base, "experiments", "sandbox", "volume_reconstruction_hd.mp4")
        imageio.mimsave(save_path_mp4, frames, fps=10, macro_block_size=None)
        print(f"Saved HD MP4 to {save_path_mp4}")
    except Exception as e:
        print(f"Could not save MP4 via imageio: {e}")
    
    # Also save a killer static frame (Z=32) for the post thumbnail
    update(32)
    static_path = os.path.join(base, "experiments", "sandbox", "volume_reconstruction_hd.png")
    fig.savefig(static_path, facecolor='black', dpi=300)
    print(f"Saved HD Static to {static_path}")

if __name__ == "__main__":
    generate_animated_reconstruction()
