
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector
from inference.reconstruction import reconstruct_dps

def kabsch_align(P, Q):
    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)
    H = torch.matmul(P_centered.T, Q_centered)
    U, S, V = torch.svd(H)
    d = torch.sign(torch.det(torch.matmul(V, U.T)))
    diag = torch.ones(3, device=P.device)
    diag[2] = d
    R = torch.matmul(torch.matmul(V, torch.diag(diag)), U.T)
    P_rotated = torch.matmul(P_centered, R.T)
    return P_rotated + Q.mean(dim=0, keepdim=True)

def run_ubq_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    print(f"Running SAFE Ubiquitin Experiment on {device}")
    
    # 1. Load SAME Model (Generalization Test)
    model = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    dm = DiffusionModel(model, timesteps=1000).to(device)
    ckpt = os.path.join(base, "experiments", "checkpoints", "ddpm_multi_protein.pth")
    try:
        dm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    except:
        dm.load_state_dict(torch.load(ckpt, map_location=device))
    dm.eval()

    # 2. Load Ubiquitin Data
    data_path = os.path.join(base, "data", "processed", "ubiquitin_ca.pt")
    if not os.path.exists(data_path):
        print("Ubiquitin data not found!")
        return

    try:
        data = torch.load(data_path, weights_only=False)
    except:
        data = torch.load(data_path)
    
    # Handle both 'coords' and raw tensor formats
    if isinstance(data, dict) and 'coords' in data:
        x_gt = data['coords']
    else:
        x_gt = data # Phase 1 might have saved raw tensor directly
        
    x_gt = x_gt.to(device)
    if x_gt.ndim == 2:
        x_gt = x_gt.unsqueeze(0)
        
    n_atoms = x_gt.shape[1]
    print(f"Loaded Ubiquitin: {n_atoms} atoms")

    # 3. Project (Same noise level)
    projector = CryoProjector(output_size=(64, 64), sigma_noise=0.05).to(device)
    rot = projector.random_rotation_matrix(1, device=device)
    y = projector.project(x_gt, rot) + torch.randn(1, 64, 64, device=device) * 0.05

    # 4. Reconstruct (Using Lysozyme-tuned params: Alpha=1.0, Scale=1.59)
    print("Reconstructing with Alpha=1.0, Scale=1.59...")
    x_rec, _, _ = reconstruct_dps(dm, projector, y, rot, device, 
                                  step_size=1.0, 
                                  n_atoms=n_atoms, 
                                  known_pose=True,
                                  coordinate_scale=1.59)

    # 5. Visualization
    x_gt_np = x_gt.squeeze().cpu().detach().numpy()
    x_rec_aligned = kabsch_align(x_rec.squeeze(), x_gt.squeeze()).cpu().detach().numpy()
    
    # RMSD check
    diff = x_rec_aligned - x_gt_np
    rmsd = np.sqrt((diff**2).sum(axis=1).mean())
    print(f"Ubiquitin Generalization RMSD: {rmsd:.4f} Angstroms")

    fig = plt.figure(figsize=(15, 5))
    views = [('XY Plane', 0, 1), ('XZ Plane', 0, 2), ('YZ Plane', 1, 2)]
    
    for i, (title, dim1, dim2) in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1)
        ax.plot(x_gt_np[:, dim1], x_gt_np[:, dim2], '-', color='gray', alpha=0.3, linewidth=1)
        ax.scatter(x_gt_np[:, dim1], x_gt_np[:, dim2], c='red', s=20, alpha=0.6, label='Ubiquitin GT')
        ax.scatter(x_rec_aligned[:, dim1], x_rec_aligned[:, dim2], c='green', s=20, alpha=0.6, label='Reconstruction') # Green for Ubq
        
        # Connect errors
        for j in range(len(x_gt_np)):
             ax.plot([x_gt_np[j, dim1], x_rec_aligned[j, dim1]],
                     [x_gt_np[j, dim2], x_rec_aligned[j, dim2]],
                     '-', color='black', alpha=0.2, linewidth=0.5)
                     
        ax.set_title(title)
        if i == 0: ax.legend()

    # Consistent bounds
    all_coords = np.concatenate([x_gt_np, x_rec_aligned], axis=0)
    min_val = all_coords.min() - 0.5
    max_val = all_coords.max() + 0.5
    for ax in fig.axes:
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Generalization Test: Ubiquitin (1UBQ)\nRMSD: {rmsd:.2f} Å (Zero-Shot Transfer)", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(base, "experiments", "sandbox", "ubq_result.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved sandbox result to {save_path}")

if __name__ == "__main__":
    run_ubq_experiment()
