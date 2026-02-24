import torch
import torch.nn as nn
import torch.nn.functional as F

class TriPlaneEncoder(nn.Module):
    """
    Encodes a 3D density volume into three orthogonal 2D feature planes (XY, XZ, YZ).

    Architecture:
        1. 3D CNN backbone  : (B, 1, D, H, W) -> (B, 32, D/2, H/2, W/2)
        2. Learned axis aggregation via Conv3d (preserves spatial layout per plane)
        3. 2D refinement + bilinear upsample -> each plane is (B, channels, plane_res, plane_res)

    Args:
        channels  : Feature channels per plane.
        plane_res : Spatial resolution of output planes (default 64).
    """


    def __init__(self, channels=32, plane_res=64):
        super().__init__()
        self.plane_res = plane_res
        inner = 32

        # Step 1: Shared 3D backbone (64^3 -> 32^3 feature volume)
        self.backbone = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv3d(16, inner, kernel_size=3, stride=2, padding=1),  # 32^3
            nn.GroupNorm(8, inner),
            nn.SiLU(),
            nn.Conv3d(inner, inner, kernel_size=3, padding=1),
            nn.GroupNorm(8, inner),
            nn.SiLU(),
        )

        mid = plane_res // 2  # 32 after stride-2

        # Step 2: Learned axis-aggregation (collapses one spatial dim)
        self.agg_xy = nn.Conv3d(inner, channels, kernel_size=(mid, 1, 1))  # collapse Z
        self.agg_xz = nn.Conv3d(inner, channels, kernel_size=(1, mid, 1))  # collapse Y
        self.agg_yz = nn.Conv3d(inner, channels, kernel_size=(1, 1, mid))  # collapse X

        # Step 3: 2D refinement + upsample to plane_res
        def refine_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(8, out_c),
                nn.SiLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
            )
        self.refine_xy = refine_block(channels, channels)
        self.refine_xz = refine_block(channels, channels)
        self.refine_yz = refine_block(channels, channels)

    def forward(self, v):
        # v: (B, 1, D, H, W) where D=H=W=64
        feat = self.backbone(v)   # (B, 32, 32, 32, 32)
        mid = feat.shape[2]       # 32

        # Learned aggregation: collapse one axis via conv
        # agg_xy collapses axis-2 (Z) -> (B, C, 1, H, W) -> squeeze -> (B, C, H, W)
        xy = self.agg_xy(feat).squeeze(2)                   # (B, C, 32, 32)
        xz = self.agg_xz(feat).squeeze(3)                   # (B, C, 32, 32)
        yz = self.agg_yz(feat).squeeze(4)                   # (B, C, 32, 32)

        # Refine and upsample to full plane resolution (64x64)
        xy = F.interpolate(self.refine_xy(xy), size=self.plane_res, mode='bilinear', align_corners=True)
        xz = F.interpolate(self.refine_xz(xz), size=self.plane_res, mode='bilinear', align_corners=True)
        yz = F.interpolate(self.refine_yz(yz), size=self.plane_res, mode='bilinear', align_corners=True)

        return [xy, xz, yz]


if __name__ == "__main__":
    B = 2
    enc = TriPlaneEncoder(channels=32, plane_res=64)
    vol = torch.randn(B, 1, 64, 64, 64)
    planes = enc(vol)
    print("Encoder test PASSED")
    print(f"  XY: {planes[0].shape}  XZ: {planes[1].shape}  YZ: {planes[2].shape}")
    total = sum(p.numel() for p in enc.parameters())
    print(f"  Params: {total/1e3:.1f}k")
