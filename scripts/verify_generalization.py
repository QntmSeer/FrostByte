
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from projection.projector import CryoProjector
from inference.reconstruction import reconstruct_dps, compute_radius_of_gyration

def kabsch_rmsd(P, Q):
    P_centered = P - P.mean(dim=0, keepdim=True)
    Q_centered = Q - Q.mean(dim=0, keepdim=True)
    H = torch.matmul(P_centered.T, Q_centered)
    U, S, V = torch.svd(H)
    d = torch.sign(torch.det(torch.matmul(V, U.T)))
    diag = torch.ones(3, device=P.device)
    diag[2] = d
    R = torch.matmul(torch.matmul(V, torch.diag(diag)), U.T)
    P_rotated = torch.matmul(P_centered, R.T)
    diff = P_rotated - Q_centered
    rmsd = torch.sqrt(torch.mean(torch.sum(diff**2, dim=-1)))
    return rmsd.item()

def verify_generalist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    
    # 1. Load Generalist Model
    print("Loading Generalist Model...")
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    ckpt_path = os.path.join(base, "experiments", "checkpoints", "ddpm_cath_generalist.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Load Myoglobin (OOD Target)
    # Using the sandbox file generated previously
    mbn_path = os.path.join(base, "experiments", "sandbox", "1mbn_ca.pt")
    if not os.path.exists(mbn_path):
         # Fallback to fetching it if missing (should be there from previous step)
         pass 

    data = torch.load(mbn_path, weights_only=False)
    print(f"Loaded data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {data.keys()}")
        if 'coords' in data:
            print(f"Coords shape: {data['coords'].shape}")
            x_gt = data['coords'].to(device)
            if x_gt.dim() == 2:
                x_gt = x_gt.unsqueeze(0)
        else:
            # Maybe it's under a different key or it's just the tensor
            print("No 'coords' key found.")
            return
    else:
        # Assume data is the tensor
        x_gt = data.to(device)
        if x_gt.dim() == 2:
            x_gt = x_gt.unsqueeze(0)

    rg_gt = compute_radius_of_gyration(x_gt)
    n_atoms = x_gt.shape[1]
    print(f"Target: Myoglobin (1MBN) - {n_atoms} atoms. Rg: {rg_gt:.2f} A")
    
    # Determine scale
    # If Rg < 5, assume normalized. We need physical for projection.
    # Training data was normalized by /10.0.
    # So if input is normalized, we treat it as latent.
    # Physical = Latent * 10.0
    
    if rg_gt < 5.0:
        print("Detected Normalized Data. Scaling by 10.0 for physical projection.")
        x_gt_phys = x_gt * 10.0
        coordinate_scale = 10.0
    else:
        print("Detected Physical Data (Angstroms).")
        x_gt_phys = x_gt
        coordinate_scale = 10.0 # Model trained on /10 data, so it outputs /10 scale.
        
    # 3. Simulate Projection from Physical Structure
    projector = CryoProjector(output_size=(64, 64), sigma_noise=0.05).to(device)
    rot_gt = projector.random_rotation_matrix(1, device=device)
    with torch.no_grad():
        y = projector.project(x_gt_phys, rot_gt) + torch.randn(1, 64, 64, device=device) * 0.05
        
    # 4. Reconstruct
    print(f"Reconstructing with Generalist Prior (Scale={coordinate_scale})...")
    # Use alpha=0.1 for stability check
    x_rec, _, _ = reconstruct_dps(model, projector, y, rot_gt, device, 
                                  step_size=0.1, n_atoms=n_atoms, known_pose=True,
                                  coordinate_scale=coordinate_scale)
    
    # 5. Evaluate
    # x_rec is aligned to x_gt_phys? No, reconstruct_dps returns PHYSICAL scale if coordinate_scale is passed?
    # No, reconstruct_dps returns: x.detach() * coordinate_scale
    # So x_rec is PHYSICAL scale.
    
    rmsd = kabsch_rmsd(x_rec.squeeze(), x_gt_phys.squeeze())
    print(f"\nResult:")
    print(f"Generalist RMSD on Myoglobin: {rmsd:.4f} Angstroms")
    
    # Compare with baseline (Phase 3 result ~15A)
    if rmsd < 3.0:
        print("SUCCESS: Model has generalized!")
    else:
        print("FAILURE: Model still struggles with OOD.")
        
    # Plot
    x_rec_np = x_rec.squeeze().cpu().numpy()
    x_gt_np = x_gt_phys.squeeze().cpu().numpy()
    
    # Simple alignment for plot
    # Re-using kabsch just for alignment
    def align_for_plot(P, Q):
        P_centered = P - P.mean(axis=0, keepdims=True)
        Q_centered = Q - Q.mean(axis=0, keepdims=True)
        H = np.dot(P_centered.T, Q_centered)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.dot(Vt.T, U.T)))
        diag = np.diag([1, 1, d])
        R = np.dot(np.dot(Vt.T, diag), U.T)
        return np.dot(P_centered, R.T) + Q.mean(axis=0, keepdims=True)

    x_rec_aligned = align_for_plot(x_rec_np, x_gt_np)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_gt_np[:, 0], x_gt_np[:, 1], x_gt_np[:, 2], c='red', label='Ground Truth (1MBN)', alpha=0.5, s=20)
    ax.scatter(x_rec_aligned[:, 0], x_rec_aligned[:, 1], x_rec_aligned[:, 2], c='blue', label=f'Generalist Pred (RMSD={rmsd:.1f}A)', alpha=0.5, s=20)
    
    # Draw connections
    for i in range(len(x_gt_np) - 1):
        ax.plot([x_gt_np[i,0], x_gt_np[i+1,0]], [x_gt_np[i,1], x_gt_np[i+1,1]], [x_gt_np[i,2], x_gt_np[i+1,2]], c='red', alpha=0.3)
    
    for i in range(len(x_rec_aligned) - 1):
        ax.plot([x_rec_aligned[i,0], x_rec_aligned[i+1,0]], [x_rec_aligned[i,1], x_rec_aligned[i+1,1]], [x_rec_aligned[i,2], x_rec_aligned[i+1,2]], c='blue', alpha=0.3)

    plt.legend()
    plt.title(f"Generalist Model on Unseen Myoglobin (10 Epochs)\nNeeds more training to resolve alpha-helices")
    
    save_path = os.path.join(base, "experiments", "sandbox", "generalist_result.png")
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    
    with open(os.path.join(base, "experiments", "sandbox", "generalist_result.txt"), "w") as f:
        f.write(f"RMSD: {rmsd:.4f}\n")

if __name__ == "__main__":
    verify_generalist()
