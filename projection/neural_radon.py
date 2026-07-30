import torch
import torch.nn as nn
from projection.radon import RadonProjector

class NeuralRayMarcher(nn.Module):
    """
    A Differentiable DDC (Direct Density Rendering) Ray-Marcher for Tri-Plane INRs.
    Instead of summing a dense 64^3 voxel grid like RadonProjector,
    this module analytically computes ray trajectories, queries the continuous INR at N steps along the ray,
    and integrates the predicted density to form the 2D pixel intensity.
    """
    def __init__(self, img_size=64, num_steps=64):
        """
        img_size: Final resolution of the 2D projection
        num_steps: Number of integration steps along each ray
        """
        super().__init__()
        self.img_size = img_size
        self.num_steps = num_steps

    def forward(self, triplane_model, R):
        """
        Computes 2D projections from the continuous TriPlane model.
        triplane_model: nn.Module returning density given (B, N, 3) coords
        R: (B, 3, 3) Rotation matrices defining the camera pose
        Returns: (B, 1, img_size, img_size) projection images
        """
        B = R.shape[0]
        device = R.device
        L = self.img_size
        
        # 1. Define the 2D Image Grid (Detector Plane)
        # range extends from -1.0 to 1.0 (coordinate scale of the triplane)
        x = torch.linspace(-1.0, 1.0, L, device=device)
        y = torch.linspace(-1.0, 1.0, L, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        
        # Base rays originating from the image plane (Z=0), marching along the Z axis
        # Shape: (L, L, 3)
        base_origins = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)-1.0], dim=-1)
        base_directions = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 1, 3).expand(L, L, -1)
        
        # 2. Setup Integration steps
        # Sample points along the z-axis from -1.0 to 1.0
        z_vals = torch.linspace(0.0, 2.0, self.num_steps, device=device) # (N_steps,)
        dz = 2.0 / self.num_steps
        
        # Flat shapes for vectorization
        base_origins = base_origins.reshape(-1, 3) # (L^2, 3)
        base_directions = base_directions.reshape(-1, 3) # (L^2, 3)
        num_rays = L * L
        
        projections = []
        
        # Process each batch item (different rotation)
        for i in range(B):
            Rot = R[i] # (3, 3)
            # The object rotates by R, which is mathematically equivalent to rotating the rays by R^T
            # However, in cryo-em, standard convention is the object rotates by R.
            # So if object rotated by R, the coordinates the ray queries at space x_ray
            # correspond to the object's body frame at R^T * x_ray
            # Actually, standard Radon formulation: P(R x)
            # We want to sample object bound to [-1, 1]^3.
            
            # Rotate ray origins and directions inversely to stay in object frame
            origins = base_origins @ Rot
            directions = base_directions @ Rot
            
            # Expand to steps: origin + direction * t
            # shape: (num_rays, num_steps, 3)
            pts = origins.unsqueeze(1) + directions.unsqueeze(1) * z_vals.unsqueeze(0).unsqueeze(-1)
            
            # Mask points outside the unit bounding sphere/cube to save compute
            # For simplicity, query everything but clamp bounds
            pts_flat = pts.reshape(1, num_rays * self.num_steps, 3) # (1, TotalPts, 3)
            
            # Query the Tri-Plane continuous model
            with torch.amp.autocast(device_type='cuda', enabled=False): # Avoid autocast issues with INRs
                # Filter out of bounds points (-1, 1) safely
                densities_flat = triplane_model(pts_flat.clamp(-1.0, 1.0)) # (1, TotalPts)
            
            densities = densities_flat.reshape(num_rays, self.num_steps)
            
            # Ray Integration (Riemann Sum)
            # integral = sum(density * dz)
            pixel_intensities = densities.sum(dim=-1) * dz # (num_rays,)
            
            # Reshape back to image
            proj_img = pixel_intensities.reshape(1, L, L)
            projections.append(proj_img)
            
        return torch.stack(projections, dim=0) # (B, 1, L, L)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.triplane import TriPlaneVolume
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TriPlaneVolume(resolution=64, channels=32).to(device)
    projector = NeuralRayMarcher(img_size=64, num_steps=64).to(device)
    
    # Generate random rotations
    radon_tools = RadonProjector(64)
    R = radon_tools.random_rotation_matrix(2, device=device)
    
    projections = projector(model, R)
    print(f"Projected shape: {projections.shape}")
    
    import matplotlib.pyplot as plt
    plt.imshow(projections[0, 0].detach().cpu().numpy())
    plt.colorbar()
    plt.savefig('test_raymarch.png')
    print("Saved test_raymarch.png")
