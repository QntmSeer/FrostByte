import torch
import torch.nn as nn
import torch.nn.functional as F

class TriPlaneEncoder(nn.Module):
    """
    Encodes a 3D density volume into three orthogonal 2D feature planes.
    Upgraded for 128x128 with Spatial Conv3d Aggregation.
    """
    def __init__(self, channels=32, plane_res=128, signal_scale=2.0):
        super().__init__()
        self.plane_res = plane_res
        self.signal_scale = signal_scale
        inner = 32

        # Step 1: 3D backbone (L^3 -> (L/2)^3)
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, inner, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, inner),
            nn.SiLU(),
            nn.Conv3d(inner, inner, kernel_size=3, padding=1),
            nn.GroupNorm(8, inner),
            nn.SiLU(),
        )

        # Step 2: Spatially-Aware Aggregation (Fixed for 128^3 -> 64^3 bottleneck)
        mid = 64 # Bottleneck res for 128 input
        self.agg_xy = nn.Conv3d(inner, channels, kernel_size=(mid, 1, 1))
        self.agg_xz = nn.Conv3d(inner, channels, kernel_size=(1, mid, 1))
        self.agg_yz = nn.Conv3d(inner, channels, kernel_size=(1, 1, mid))

        def refine_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.Tanh()
            )
            
        self.refine_xy = refine_block(channels, channels)
        self.refine_xz = refine_block(channels, channels)
        self.refine_yz = refine_block(channels, channels)

    def forward(self, v):
        # v: (B, 1, 128, 128, 128)
        feat = self.backbone(v) # (B, 32, 64, 64, 64)
        
        # Collapse one axis via learned convolution
        xy = self.agg_xy(feat).squeeze(2) # (B, C, 64, 64)
        xz = self.agg_xz(feat).squeeze(3) # (B, C, 64, 64)
        yz = self.agg_yz(feat).squeeze(4) # (B, C, 64, 64)

        # Refine and upsample to 128x128
        xy = F.interpolate(self.refine_xy(xy), size=self.plane_res, mode='bilinear', align_corners=True) * self.signal_scale
        xz = F.interpolate(self.refine_xz(xz), size=self.plane_res, mode='bilinear', align_corners=True) * self.signal_scale
        yz = F.interpolate(self.refine_yz(yz), size=self.plane_res, mode='bilinear', align_corners=True) * self.signal_scale

        return [xy, xz, yz]

if __name__ == "__main__":
    B = 1
    enc = TriPlaneEncoder(channels=32, plane_res=128)
    vol = torch.randn(B, 1, 128, 128, 128)
    planes = enc(vol)
    print(f"Output XY shape: {planes[0].shape}")
