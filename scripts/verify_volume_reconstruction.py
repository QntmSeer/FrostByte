
import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset
from projection.radon import RadonProjector

def reconstruct_volume_dps(model, projector, y_meas, rot_matrix, device, step_scale=100.0):
    # Initialize from noise
    # Shape: (1, 1, 64, 64, 64)
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    
    K = y_meas.shape[0] # Number of views
    
    pbar = tqdm(reversed(range(0, model.timesteps)), total=model.timesteps)
    
    for i in pbar:
        t_tensor = torch.full((1,), i, device=device, dtype=torch.long)
        
        # 1. Diffusion Reverse Step
        x = x.detach().requires_grad_(True)
        epsilon_theta = model.model(x, t_tensor)
        
        # Estimate x_0
        alpha_bar_t = model.alphas_cumprod[i]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        
        x_0_hat = (x - sqrt_one_minus_alpha_bar * epsilon_theta) / sqrt_alpha_bar
        
        # 2. Measurement Consistency (Gradient)
        # Project x_0_hat
        # We need to broadcast x_0_hat for K views?
        # x_0_hat: (1, 1, L, L, L)
        # rots: (K, 3, 3)
        # We can repeat x_0_hat K times -> (K, 1, L, L, L)
        x_0_rep = x_0_hat.repeat(K, 1, 1, 1, 1)
        y_hat = projector(x_0_rep, rot_matrix) # (K, 1, L, L)
        
        loss = F.mse_loss(y_hat, y_meas)
        
        grad = torch.autograd.grad(loss, x)[0]
        
        # 3. Update x
        # Standard DDPM update + gradient step
        beta_t = model.betas[i]
        alpha_t = 1 - beta_t
        sigma_t = torch.sqrt(beta_t)
        
        coeff1 = 1 / torch.sqrt(alpha_t)
        coeff2 = beta_t / sqrt_one_minus_alpha_bar
        
        mean = coeff1 * (x.detach() - coeff2 * epsilon_theta.detach())
        
        # Gradient descent guidance
        # Scale step size by 1/grad_norm
        norm = grad.norm()
        if norm > 0: grad = grad / norm
        
        # DPS typically uses fixed scale
        mean = mean - step_scale * grad
        
        if i > 0:
            noise = torch.randn_like(x).detach()
            x = mean + sigma_t * noise
        else:
            x = mean
            
        pbar.set_postfix({'meas_loss': loss.item()})
        
    return x.detach()

def run_verification():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Volumetric Verification on {device}")
    
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # 1. Load Model
    net = UNet3D(in_ch=1, out_ch=1, time_dim=64)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_volume_overfit.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
    else:
        print("Model checkpoint not found!")
        return

    # 2. Prepare Ground Truth (Lysozyme)
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    # Quick hack: fetch one coords from dict
    data_dict = torch.load(data_path, weights_only=False)
    if '1hel' in data_dict:
        coords = data_dict['1hel']
    else:
        coords = list(data_dict.values())[0]
        
    # Voxelize
    print("Voxelizing Ground Truth...")
    # Manually call static method
    # Center coords first
    coords = coords - coords.mean(dim=0, keepdim=True)
    # Scale by 10.0 (match training)
    coords = coords * 10.0
    vol_gt = VolumeDataset.voxelize_gaussian(coords, 64, 1.0, 1.0).to(device)
    vol_gt = vol_gt.unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64, 64)
    
    # 3. Simulate Projections
    projector = RadonProjector(64).to(device)
    # 3 Orthogonal views
    R = torch.eye(3, device=device).unsqueeze(0).repeat(3, 1, 1)
    # View 1: Identity (Z-axis)
    # View 2: Rot 90 X
    # View 3: Rot 90 Y
    # ... simple random views might be better to avoid symmetry issues
    # 10 views for higher resolution
    R = projector.random_rotation_matrix(10, device=device)
    
    print("Simulating Projections...")
    with torch.no_grad():
        x_rep = vol_gt.repeat(10, 1, 1, 1, 1)
        y_meas = projector(x_rep, R)
        # Add noise?
        y_meas = y_meas + torch.randn_like(y_meas) * 0.0 # Clean for now
        
    # 4. Reconstruct
    print("Reconstructing Volume...")
    # step_scale=100.0 was exploding, but 1.0 might be too soft for high resolution.
    # Increasing to 5.0 to strictly enforce measurement consistency.
    vol_rec = reconstruct_volume_dps(model, projector, y_meas, R, device, step_scale=5.0)
    
    print(f"Rec Range: [{vol_rec.min():.2f}, {vol_rec.max():.2f}] Mean: {vol_rec.mean():.2f}")
    
    # 5. Visualize (Central Slice)
    print("Saving visualization...")
    mid = 32
    slice_gt = vol_gt[0, 0, mid].cpu().numpy()
    slice_rec = vol_rec[0, 0, mid].cpu().numpy()
    
    proj_gt = y_meas[0, 0].cpu().numpy()
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(proj_gt, cmap='gray')
    ax[0].set_title("Input Projection (View 1)")
    
    ax[1].imshow(slice_gt, cmap='viridis')
    ax[1].set_title("GT Mid-Slice (Z=32)")
    
    ax[2].imshow(slice_rec, cmap='viridis')
    ax[2].set_title("Reconstructed Mid-Slice")
    
    save_path = os.path.join(base, "experiments", "sandbox", "volume_reconstruction.png")
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    run_verification()
