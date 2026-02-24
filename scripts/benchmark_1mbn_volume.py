
import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset
from projection.radon import RadonProjector

def fetch_1mbn_volume(device):
    """Load Myoglobin (1MBN) Cα coordinates and voxelise into a 64^3 density grid."""
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    pdb_path = os.path.join(base, "data", "raw", "1mbn.pdb")

    if not os.path.exists(pdb_path):
        import biotite.database.rcsb as rcsb
        rcsb.fetch("1mbn", "pdb", os.path.dirname(pdb_path))

    pdb_file = pdb.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)
    ca = structure[structure.atom_name == "CA"]
    coords = torch.tensor(ca.coord, dtype=torch.float32)

    # Centre and voxelise using physical Angstrom coordinates (1A/voxel, 64^3 grid)
    coords = coords - coords.mean(dim=0, keepdim=True)
    vol = VolumeDataset.voxelize_gaussian(coords, 64, 1.0, 1.0)
    return vol.unsqueeze(0).unsqueeze(0).to(device)

def reconstruct_volume(model, projector, y_meas, rot_matrix, device):
    """DPS-guided reverse diffusion reconstruction from measured projections."""
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    step_scale = 1.0
    
    for i in tqdm(reversed(range(0, model.timesteps))):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = x.detach().requires_grad_(True)
        epsilon = model.model(x, t)
        
        alpha_bar = model.alphas_cumprod[i]
        x_0_hat = (x - torch.sqrt(1-alpha_bar)*epsilon) / torch.sqrt(alpha_bar)
        
        # Loss
        x_rep = x_0_hat.repeat(y_meas.shape[0], 1, 1, 1, 1)
        y_hat = projector(x_rep, rot_matrix)
        loss = F.mse_loss(y_hat, y_meas)
        
        grad = torch.autograd.grad(loss, x)[0]
        grad = grad / (grad.norm() + 1e-8)
        
        # Step
        beta = model.betas[i]
        alpha = 1 - beta
        mean = (1/torch.sqrt(alpha)) * (x.detach() - (beta/torch.sqrt(1-alpha_bar))*epsilon.detach())
        mean = mean - step_scale * grad
        
        if i > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
            
    return x.detach()

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking 1MBN (Myoglobin) on {device}")
    
    # 1. Load Model
    net = UNet3D(in_ch=1, out_ch=1, time_dim=64)
    model = DiffusionModel(net, timesteps=1000).to(device)
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    model.load_state_dict(torch.load(os.path.join(base, "experiments/checkpoints/ddpm_volume_unet.pth"), map_location=device))
    model.eval()
    
    # 2. Prepare Data (1MBN)
    # We simulate projections from the Ground Truth Volume
    vol_gt = fetch_1mbn_volume(device)
    print(f"GT Volume Mass: {vol_gt.sum().item():.2f}")
    
    projector = RadonProjector(64).to(device)
    R = projector.random_rotation_matrix(3, device=device)  # 3 random views
    y_meas = projector(vol_gt.repeat(3, 1, 1, 1, 1), R)
    
    # 3. Reconstruct
    vol_rec = reconstruct_volume(model, projector, y_meas, R, device)
    
    # 4. Compare
    # Cross Correlation
    flat_gt = vol_gt.flatten()
    flat_rec = vol_rec.flatten()
    cc = torch.dot(flat_gt, flat_rec) / (flat_gt.norm() * flat_rec.norm())
    print(f"Volumetric CC: {cc.item():.4f}")
    
    # 5. Viz
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    mid = 32
    ax[0].imshow(y_meas[0,0].cpu().numpy(), cmap='gray')
    ax[0].set_title("Input Projection")
    ax[1].imshow(vol_gt[0,0,mid].cpu().numpy(), cmap='viridis')
    ax[1].set_title("GT Slice (1MBN)")
    ax[2].imshow(vol_rec[0,0,mid].cpu().numpy(), cmap='viridis')
    ax[2].set_title(f"Rec Slice (CC={cc.item():.2f})")
    
    plt.suptitle("Volumetric vs Point Cloud Comparison\nMethod: 3D U-Net (Voxel) vs PointTransformer (Coord)")
    plt.savefig(os.path.join(base, "experiments/sandbox/benchmark_1mbn.png"))
    print("Saved benchmark plot.")

if __name__ == "__main__":
    run_benchmark()
