import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from data.volume_dataset import VolumeDataset

class TriPlaneDataset(Dataset):
    """
    Dataset for training Tri-Plane Implicit Neural Representations (INRs).
    Instead of outputting a full 64x64x64 voxel grid, this dataset outputs a batch of 
    continuous (x, y, z) spatial coordinates and their corresponding ground-truth electron density.
    """
    def __init__(self, data_path, num_samples=2048, grid_size=128, voxel_size=0.6, sigma=0.6, coordinate_scale=1.0, augment=False):
        try:
            self.data_dict = torch.load(data_path, weights_only=False)
        except:
            self.data_dict = torch.load(data_path)
            
        self.pdb_ids = list(self.data_dict.keys())
        self.coords_list = [self.data_dict[k] for k in self.pdb_ids]
        
        self.num_samples = num_samples # Number of (x,y,z) points to sample per protein per iteration
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.coordinate_scale = coordinate_scale
        self.augment = augment
        
        print(f"Loaded Tri-Plane Dataset: {len(self.pdb_ids)} structures.")
        print(f"Sampling {num_samples} continuous points per crop. Augment={augment}")

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        # 1. Get Physical Atomic Coords (N, 3)
        coords = self.coords_list[idx].clone() * self.coordinate_scale
        centroid = coords.mean(dim=0, keepdim=True)
        coords = coords - centroid
        
        # 1.5 Optional Augmentation (Random Rotation)
        if self.augment:
            import math
            angles = torch.rand(3) * 2 * math.pi
            ca, cb, cg = torch.cos(angles)
            sa, sb, sg = torch.sin(angles)
            
            # Rotation matrices
            Rx = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
            Ry = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            Rz = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            coords = coords @ R.T.to(coords.device)
        
        # 2. Compute the Ground Truth Voxel Grid (for density lookup)
        # In a fully continuous implementation, we would compute exact Gaussian sums
        # at the continuous (x,y,z) points. For speed during training, we compute the 
        # grid once and trilinearly interpolate the continuous points from it.
        vol_gt = VolumeDataset.voxelize_gaussian(coords, self.grid_size, self.voxel_size, self.sigma)
        vol_gt = vol_gt.unsqueeze(0).unsqueeze(0) # (1, 1, D, H, W)
        
        # 3. Sample random continuous coordinates within the bounding box
        # Bounding box is [-L/2 * vs, L/2 * vs]
        L = self.grid_size
        limit = (L / 2) * self.voxel_size
        
        # We need to sample continuous points. To fix class imbalance (mostly empty space),
        # sample 50% of points by adding small Gaussian noise to the actual ground truth atomic coordinates.
        # Sample 50% points randomly in the bounding box.
        num_atom_pts = self.num_samples // 2
        num_rand_pts = self.num_samples - num_atom_pts
        
        # Ensure we don't index out of bounds if num_atom_pts > len(coords)
        idx_atoms = torch.randint(0, coords.shape[0], (num_atom_pts,))
        atom_queries = coords[idx_atoms] + torch.randn(num_atom_pts, 3, device=coords.device) * self.sigma * 0.5
        
        rand_queries = torch.randn(num_rand_pts, 3, device=coords.device) * (limit / 2.0)
        
        query_coords = torch.cat([atom_queries, rand_queries], dim=0)
        query_coords = torch.clamp(query_coords, -limit, limit)
        
        # 4. Extract Ground Truth Density via Trilinear Interpolation
        # grid_sample expects coordinates in the range [-1, 1]
        # PyTorch grid_sample expects (x, y, z) corresponding to (W, H, D)
        # To avoid flipping issues, we'll keep the space tightly symmetric.
        grid_query = (query_coords / limit).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, N, 3)
        
        # Sample: Output shape (1, 1, 1, 1, N) -> squeeze to (N,)
        density_gt = F.grid_sample(vol_gt, grid_query, mode='bilinear', padding_mode='zeros', align_corners=True)
        density_gt = density_gt.squeeze()
        
        # Return the continuous points and their ground truth density targets
        # Return normalized coordinates in [-1, 1] so grid_sample works!
        return vol_gt.squeeze(0), query_coords / limit, density_gt

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ds = TriPlaneDataset('data/processed/cath_subset.pt', num_samples=100)
    vol, queries, densities = ds[0]
    
    print(f"GT Volume Shape: {vol.shape}")
    print(f"Query Coords Shape: {queries.shape}")
    print(f"Sampled Densities Shape: {densities.shape}")
    print(f"Max sampled density: {densities.max().item():.4f}")
