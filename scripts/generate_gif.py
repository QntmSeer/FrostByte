import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.unet_2d import TriPlaneUNet
from models.triplane import TriPlaneDecoder
from models.diffusion import DiffusionModel

def generate_evolution_gif():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Generating Diffusion Evolution GIF...")
    
    CKPT_DIR = os.path.join(BASE_DIR, "experiments", "checkpoints")
    
    unet = TriPlaneUNet(plane_channels=32).to(device)
    unet.load_state_dict(torch.load(f'{CKPT_DIR}/ddpm_triplane_2d_v2.pth', map_location=device))
    unet.eval()
    
    decoder = TriPlaneDecoder(channels=32).to(device)
    decoder.load_state_dict(torch.load(f'{CKPT_DIR}/triplane_decoder_v2.pth', map_location=device))
    decoder.eval()
    
    diffusion = DiffusionModel(unet, 1000).to(device)
    
    # We will sample unconditionally, tracking x_0 predictions
    B = 1
    shape = (1, 96, 64, 64)
    x = torch.randn(shape, device=device)
    
    L = 64
    grid_1d = torch.linspace(-1.0, 1.0, L, device=device)
    zz, yy, xx = torch.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(1, -1, 3)
    
    frames = []
    
    with torch.no_grad():
        for i in reversed(range(0, diffusion.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            
            epsilon_theta = diffusion.model(x, t)
            if isinstance(epsilon_theta, list):
                epsilon_theta = torch.cat(epsilon_theta, dim=1)
                
            beta_t = diffusion.betas[i]
            alpha_bar_t = diffusion.alphas_cumprod[i]
            sqrt_alpha_bar_t = diffusion.sqrt_alphas_cumprod[i]
            sqrt_one_minus_alpha_bar_t = diffusion.sqrt_one_minus_alphas_cumprod[i]
            
            # Posterior mean estimate scaled back to Decoder's domain
            x_0_pred = (x - sqrt_one_minus_alpha_bar_t * epsilon_theta) / sqrt_alpha_bar_t
            
            if i % 25 == 0 or i == 0:
                # Decode the volume
                planes = [x_0_pred[:, 0:32], x_0_pred[:, 32:64], x_0_pred[:, 64:96]]
                vol = decoder(planes, coords).reshape(L, L, L).cpu().numpy()
                
                # 2D Projection Rendering (Summed Density along Z axis)
                # This reveals the silhouetted 3D fold of the protein
                proj_img = np.sum(vol, axis=2)
                
                # Soft threshold and normalization
                vmax = np.percentile(proj_img, 99) if np.max(proj_img) > 0 else 1.0
                proj_img = np.clip(proj_img / vmax, 0, 1)
                frames.append(proj_img)
            
            # standard step
            alpha_t = 1 - beta_t
            coeff1 = 1 / torch.sqrt(alpha_t)
            coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = coeff1 * (x - coeff2 * epsilon_theta)
            
            if i > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
                
    # Save GIF
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    
    img = ax.imshow(frames[0], cmap='magma', vmin=0, vmax=1)
    
    def animate(i):
        img.set_array(frames[i])
        ax.set_title(f"Timestep {1000 - i * 25}", color='white', pad=20)
        return [img]
        
    fig.patch.set_facecolor('black')
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100, blit=True)
    out_path = os.path.join(BASE_DIR, "experiments", "results", "evolution.gif")
    anim.save(out_path, writer='pillow', fps=10)
    print(f"GIF saved to {out_path}")

if __name__ == "__main__":
    generate_evolution_gif()
