
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class VolumeDataset(Dataset):
    """
    Volumetric Dataset for Cryo-EM.
    Converts atomic coordinates -> 3D Density Grid (Voxelization).
    """
    def __init__(self, data_path, grid_size=64, voxel_size=1.0, sigma=1.0, coordinate_scale=1.0):
        """
        Args:
            data_path: Path to .pt file containing {pdb_id: coords}
            grid_size: Number of voxels per dimension (L)
            voxel_size: Angstroms per voxel (Resolution)
            sigma: Width of Gaussian atom blob (in Angstroms)
            coordinate_scale: Factor to scale input coordinates (e.g. 10.0 if normalized)
        """
        try:
            self.data_dict = torch.load(data_path, weights_only=False)
        except:
            self.data_dict = torch.load(data_path)
            
        self.pdb_ids = list(self.data_dict.keys())
        self.coords_list = [self.data_dict[k] for k in self.pdb_ids]
        
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.coordinate_scale = coordinate_scale
        
        print(f"Loaded Volume Dataset: {len(self.pdb_ids)} structures.")
        print(f"Volume: {grid_size}^3 voxels @ {voxel_size} A/pix. Scale: {coordinate_scale}")

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        # 1. Get coords (N, 3)
        coords = self.coords_list[idx].clone() * self.coordinate_scale
        
        # 2. Center coords
        centroid = coords.mean(dim=0, keepdim=True)
        coords = coords - centroid
        
        # 3. Voxelize
        volume = self.voxelize_gaussian(coords, self.grid_size, self.voxel_size, self.sigma)
        
        # Add channel dim: (1, L, L, L)
        return volume.unsqueeze(0)

    @staticmethod
    def voxelize_gaussian(coords, grid_size, voxel_size, sigma):
        """
        Splat atoms into a 3D grid using Gaussian kernels.
        """
        device = coords.device
        L = grid_size
        
        # Create grid coordinates (centered at 0)
        # range: [-L/2 * vs, L/2 * vs]
        r = torch.linspace(-L/2 * voxel_size, L/2 * voxel_size, L)
        grid_x, grid_y, grid_z = torch.meshgrid(r, r, r, indexing='ij')
        
        # shape: (L, L, L)
        grid_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(device) # (L, L, L, 3)
        
        # We can't iterate over grid for every atom (too slow).
        # We can't iterate over atoms for every voxel (too slow).
        # Optimization: Only splat atoms into local neighborhood?
        # For simplicity in Phase 5 prototype: Brute force over atoms, but optimized.
        
        # Actually, simpler approach:
        # 1. Map atoms to grid indices.
        # 2. Scatter add density.
        # 3. Convolve with Gaussian kernel.
        
        # Step 1: Discretize
        # grid index = (coord / voxel_size) + L/2
        indices = (coords / voxel_size) + (L / 2)
        indices = indices.long()
        
        # Filter out of bounds
        mask = (indices[:, 0] >= 0) & (indices[:, 0] < L) & \
               (indices[:, 1] >= 0) & (indices[:, 1] < L) & \
               (indices[:, 2] >= 0) & (indices[:, 2] < L)
        valid_indices = indices[mask]
        
        # Step 2: Scatter to grid
        # Create empty grid
        # We rely on indices being unique? No, atoms can overlap.
        # Helper to flatten indices
        flat_indices = valid_indices[:, 0] * L * L + valid_indices[:, 1] * L + valid_indices[:, 2]
        
        grid_flat = torch.zeros(L * L * L, device=device)
        ones = torch.ones_like(flat_indices, dtype=torch.float32)
        
        grid_flat.scatter_add_(0, flat_indices, ones)
        grid = grid_flat.view(1, 1, L, L, L) # (B, C, D, H, W) for conv3d
        
        # Step 3: Gaussian Smooth (Convolution)
        # Sigma in voxels
        sigma_vox = sigma / voxel_size
        k_size = int(6 * sigma_vox) + 1
        if k_size % 2 == 0: k_size += 1
        
        # Create Gaussian Kernel
        k_r = torch.linspace(-k_size//2, k_size//2, k_size)
        kx, ky, kz = torch.meshgrid(k_r, k_r, k_r, indexing='ij')
        k_dist_sq = kx**2 + ky**2 + kz**2
        kernel = torch.exp(-k_dist_sq / (2 * sigma_vox**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, k_size, k_size, k_size).to(device)
        
        # Convolve
        # Padding
        pad = k_size // 2
        density = F.conv3d(grid, kernel, padding=pad)
        
        return density.squeeze(0).squeeze(0) # (L, L, L)

if __name__ == "__main__":
    # Test
    import matplotlib.pyplot as plt
    
    # Mock data
    coords = torch.randn(100, 3) * 10
    vol = VolumeDataset.voxelize_gaussian(coords, 64, 1.0, 1.0)
    
    print(f"Volume shape: {vol.shape}")
    print(f"Total mass: {vol.sum().item()}")
    
    # Project (Sum Z)
    proj = vol.sum(dim=2)
    plt.imshow(proj.cpu().numpy())
    plt.colorbar()
    plt.title("Projected Test Volume")
    plt.savefig("test_volume.png")
