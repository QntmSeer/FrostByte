import torch
import torch.nn as nn
import torch.nn.functional as F

class TriPlaneDecoder(nn.Module):
    """
    Implicit Neural Representation (INR) Decoder for Tri-Plane Architectures.
    A continuous density function: f_theta(x, y, z | planes) -> density
    """
    def __init__(self, channels=32, hidden_dim=128, num_layers=4, pos_dim=32):
        super().__init__()
        self.pos_dim = pos_dim
        # Fourier Features: log-spaced frequencies
        freqs = 2.0 ** torch.linspace(0, 5, pos_dim // 2)
        self.register_buffer('freqs', freqs)
        
        # Input: SUM(XY, XZ, YZ) + Fourier(3D Coords)
        in_features = channels + (3 * pos_dim) 
        
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def embed_pos(self, coords):
        # coords: (B, N, 3)
        # Output: (B, N, 3 * pos_dim)
        x = coords.unsqueeze(-1) * self.freqs # (B, N, 3, half_pos)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) # (B, N, 3, pos_dim)
        return emb.reshape(coords.shape[0], coords.shape[1], -1)
        
    def sample_plane(self, plane, coords_u, coords_v):
        """
        Samples a 2D feature plane at continuous (u, v) coordinates.
        plane: (B, C, H, W)
        """
        B, N = coords_u.shape
        grid = torch.stack([coords_u, coords_v], dim=-1).unsqueeze(1) # (B, 1, N, 2)
        features = F.grid_sample(plane, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return features.squeeze(2).transpose(1, 2) # (B, N, C)

    def forward(self, planes, coords):
        """
        planes: List of 3 tensors [XY, XZ, YZ]
        coords: (B, N, 3) XYZ coordinates in range [-1, 1]
        """
        plane_xy, plane_xz, plane_yz = planes
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        
        # 1. Plane features
        feat_xy = self.sample_plane(plane_xy, x, y)
        feat_xz = self.sample_plane(plane_xz, x, z)
        feat_yz = self.sample_plane(plane_yz, y, z)
        
        feat_sum = feat_xy + feat_xz + feat_yz # (B, N, C)
        
        # 2. Position features (Fourier)
        feat_pos = self.embed_pos(coords) # (B, N, 3 * pos_dim)
        
        # 3. Concatenate and Decode
        feat_full = torch.cat([feat_sum, feat_pos], dim=-1)
        density = self.mlp(feat_full) 
        
        return F.softplus(density).squeeze(-1)

if __name__ == "__main__":
    B, N = 2, 1000
    decoder = TriPlaneDecoder(channels=32)
    planes = [torch.randn(B, 32, 128, 128) for _ in range(3)]
    coords = (torch.rand(B, N, 3) * 2 - 1)
    
    densities = decoder(planes, coords)
    print(f"Queried {N} points. Output density shape: {densities.shape}")
