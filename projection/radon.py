
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RadonProjector(nn.Module):
    """
    Differentiable Forward Projector for Volumetric Data.
    Operation: Rotate Volume -> Integrate along Z.
    """
    def __init__(self, resolution=64):
        super().__init__()
        self.resolution = resolution
        
        # Create static identity grid
        # Range [-1, 1] for grid_sample
        d = torch.linspace(-1, 1, resolution)
        meshz, meshy, meshx = torch.meshgrid(d, d, d, indexing='ij')
        self.register_buffer('grid', torch.stack((meshx, meshy, meshz), dim=-1)) # (L, L, L, 3)

    def forward(self, volume, rot_matrix):
        """
        Args:
            volume: (B, 1, L, L, L) density grid
            rot_matrix: (B, 3, 3) rotation matrices
            
        Returns:
            projection: (B, 1, L, L) 2D image
        """
        B = volume.shape[0]
        L = self.resolution
        
        # 1. Rotate Grid
        # We need to sample the volume at rotated coordinates.
        # grid_sample(input, grid) samples input at grid locations.
        # If we want to rotate volume by R, we need to query at R^T * grid.
        # R maps object -> camera.
        # So we want pixel x to look up value at R^T * x.
        
        # Expand grid for batch
        grid = self.grid.unsqueeze(0).repeat(B, 1, 1, 1, 1) # (B, L, L, L, 3)
        grid_flat = grid.view(B, -1, 3) # (B, N, 3)
        
        # Apply inverse rotation (transpose)
        # grid_rot = (R^T @ grid^T)^T = grid @ R
        rot_grid = torch.bmm(grid_flat, rot_matrix) # (B, N, 3)
        
        # Reshape back to grid
        rot_grid = rot_grid.view(B, L, L, L, 3)
        
        # 2. Sample Volume
        # align_corners=True matches standard definition usually
        rotated_volume = F.grid_sample(volume, rot_grid, align_corners=True, mode='bilinear')
        
        # 3. Integrate along Z (dim=2)
        # rotated_volume: (B, 1, L, L, L) -> D, H, W (Z, Y, X)
        # Sum along Z
        projection = rotated_volume.sum(dim=2) # (B, 1, L, L)
        
        return projection

    def random_rotation_matrix(self, batch_size, device='cpu'):
        """Generates random rotation matrices (Haar measure)."""
        # Helper copied from CryoProjector
        rand = torch.randn(batch_size, 3, 3, device=device)
        u, s, v = torch.svd(rand)
        r = torch.bmm(u, v.transpose(1, 2))
        det = torch.det(r)
        r[det < 0] = -r[det < 0]
        return r

if __name__ == "__main__":
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    projector = RadonProjector(64).to(device)
    
    # Create sphere volume
    L = 64
    x = torch.linspace(-1, 1, L)
    grid_z, grid_y, grid_x = torch.meshgrid(x, x, x, indexing='ij')
    dist_sq = grid_x**2 + grid_y**2 + grid_z**2
    sphere = (dist_sq < 0.5**2).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Rotate 45 deg
    R = torch.eye(3, device=device).unsqueeze(0)
    # rot z
    theta = np.pi/4
    R[0, 0, 0] = np.cos(theta)
    R[0, 0, 1] = -np.sin(theta)
    R[0, 1, 0] = np.sin(theta)
    R[0, 1, 1] = np.cos(theta)
    
    proj = projector(sphere, R)
    
    print(f"Volume shape: {sphere.shape}")
    print(f"Proj shape: {proj.shape}") # Should be (1, 1, 64, 64)
    
    import matplotlib.pyplot as plt
    plt.imshow(proj[0, 0].cpu().numpy(), cmap='gray')
    plt.title("Radon Projection of Sphere")
    plt.savefig("test_radon.png")
