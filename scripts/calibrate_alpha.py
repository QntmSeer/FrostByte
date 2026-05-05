
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector
from inference.reconstruction import reconstruct_dps, compute_radius_of_gyration

def calibrate_alpha():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running alpha calibration on {device}")
    
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # 1. Load Model
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    except:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Load Ground Truth
    data_path = os.path.join(base, "data", "processed", "1hel_ca.pt")
    try:
        data = torch.load(data_path, weights_only=False)
    except:
        data = torch.load(data_path)
    x_gt = data['coords'].to(device).unsqueeze(0)
    n_atoms = x_gt.shape[1]
    rg_gt = compute_radius_of_gyration(x_gt)
    print(f"Target Rg: {rg_gt:.4f}")
    
    # 3. Setup Projector (1 View, low noise to isolate prior effect)
    sigma = 0.05
    projector = CryoProjector(output_size=(64, 64), sigma_noise=sigma).to(device)
    rot = projector.random_rotation_matrix(1, device=device)
    with torch.no_grad():
        y = projector.project(x_gt, rot) + torch.randn(1, 64, 64, device=device) * sigma
        
    # 4. Alpha Sweep with Coordinate Scale
    # Training Rg ~ 0.38, Target Rg ~ 0.607
    # Scale factor = 0.607 / 0.382 = 1.59
    coord_scale = 1.59
    print(f"Using Coordinate Scale: {coord_scale}")
    
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] 
    rgs = []
    
    print("\nStarting Alpha Sweep (Guidance Strength)...")
    for alpha in alphas:
        print(f"Testing step_size = {alpha}")
        x_rec, _, _ = reconstruct_dps(model, projector, y, rot, device, 
                                      step_size=alpha, n_atoms=n_atoms, known_pose=True,
                                      coordinate_scale=coord_scale)
        
        rg = compute_radius_of_gyration(x_rec)
        rgs.append(rg)
        print(f"  Rg: {rg:.4f}")
        
    # 5. Plot
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, rgs, 'o-', linewidth=2, color='#2196F3', label='Reconstructed Rg')
    plt.axhline(y=rg_gt, color='k', linestyle='--', label='Ground Truth')
    plt.axhline(y=0.3818, color='g', linestyle=':', label='Training Mean Rg')
    plt.xscale('log')
    plt.xlabel('Guidance Step Size (alpha)')
    plt.ylabel('Radius of Gyration (Rg)')
    plt.title('Calibration: Scale Recovery vs Guidance Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    res_path = os.path.join(base, "experiments", "results", "calibration_plot.png")
    plt.savefig(res_path, dpi=200)
    print(f"Saved calibration plot to {res_path}")

if __name__ == "__main__":
    calibrate_alpha()
