import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_2d import TriPlaneUNet

class CascadedTriPlaneUNet(nn.Module):
    """
    A Super-Resolution UNet for Tri-Planes.
    Takes noisy 128x128 planes and conditions on 64x64 base planes.
    """
    def __init__(self, plane_channels=32, time_dim=64):
        super().__init__()
        # Each internal UNet handles ONE plane, conditioned on ONE low-res plane
        self.unet_xy = TriPlaneUNet(plane_channels=32, time_dim=time_dim, in_channels=64, out_channels=32)
        self.unet_xz = TriPlaneUNet(plane_channels=32, time_dim=time_dim, in_channels=64, out_channels=32)
        self.unet_yz = TriPlaneUNet(plane_channels=32, time_dim=time_dim, in_channels=64, out_channels=32)

    def forward(self, x_t, t, low_res):
        # x_t: (B, 96, 128, 128)
        # low_res: (B, 96, 64, 64)
        
        # 1. Upsample low_res guidance to 128x128
        lr_up = F.interpolate(low_res, size=(128, 128), mode='bilinear', align_corners=True)
        
        # 2. Split into planes
        p_t = [x_t[:, 0:32], x_t[:, 32:64], x_t[:, 64:96]]
        p_lr = [lr_up[:, 0:32], lr_up[:, 32:64], lr_up[:, 64:96]]
        
        # 3. Concatenate and Solve (Concatenate noisy plane with guidance)
        out_xy = self.unet_xy(torch.cat([p_t[0], p_lr[0]], dim=1), t)
        out_xz = self.unet_xz(torch.cat([p_t[1], p_lr[1]], dim=1), t)
        out_yz = self.unet_yz(torch.cat([p_t[2], p_lr[2]], dim=1), t)
        
        return [out_xy, out_xz, out_yz]
