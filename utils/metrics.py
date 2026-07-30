import torch
import torch.fft
import numpy as np

def align_volumes_com(vol1, vol2):
    """
    Aligns vol1 to vol2 by their centers of mass (CoM).
    Returns aligned vol1.
    """
    def get_com(v):
        v = v.detach()
        # Threshold to ignore background noise (SOTA practice)
        v_thresh = torch.where(v > v.mean() + v.std(), v, torch.zeros_like(v))
        
        coords = torch.stack(torch.meshgrid(
            torch.arange(v.shape[0], device=v.device),
            torch.arange(v.shape[1], device=v.device),
            torch.arange(v.shape[2], device=v.device),
            indexing='ij'
        ), dim=-1).float()
        mass = v_thresh.sum() + 1e-8
        com = (v_thresh.unsqueeze(-1) * coords).sum(dim=(0,1,2)) / mass
        return com

    com1 = get_com(vol1)
    com2 = get_com(vol2)
    shift = (com2 - com1).round().long()
    
    # Simple integer shift using roll (assumes periodic boundaries for robustness, 
    # but practically we just want alignment)
    aligned = torch.roll(vol1, shifts=tuple(shift.tolist()), dims=(0, 1, 2))
    return aligned

def get_soft_mask(shape, radius, device, sigma=2.0):
    """Creates a soft spherical mask."""
    D, H, W = shape
    coords = torch.linspace(-1, 1, D, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    r = torch.sqrt(xx**2 + yy**2 + zz**2)
    mask = torch.sigmoid((radius - r) / (sigma / D))
    return mask

def compute_cc(vol1, vol2, align=True):
    """Computes 3D Pearson Cross-Correlation."""
    if align:
        vol1 = align_volumes_com(vol1, vol2)
        
    v1 = vol1.flatten()
    v2 = vol2.flatten()
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    cc = (v1 * v2).sum() / (torch.sqrt((v1**2).sum()) * torch.sqrt((v2**2).sum()) + 1e-8)
    return cc.item()

def compute_fsc(vol1, vol2, align=True, mask_radius=0.8):
    """Computes Fourier Shell Correlation (FSC) with alignment and masking."""
    device = vol1.device
    if align:
        vol1 = align_volumes_com(vol1, vol2)
    
    mask = get_soft_mask(vol1.shape, mask_radius, device)
    vol1 = vol1 * mask
    vol2 = vol2 * mask
        
    D, H, W = vol1.shape
    assert D == H == W, "FSC requires cubic volumes"
    L = D
    
    # Fourier Transform
    f1 = torch.fft.fftshift(torch.fft.fftn(vol1))
    f2 = torch.fft.fftshift(torch.fft.fftn(vol2))
    
    # 2. Create Radius Grid
    coords = torch.linspace(-L//2, L//2 - 1, L, device=device)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    r_grid = torch.sqrt(xx**2 + yy**2 + zz**2)
    
    # Max radius is L/2
    r_bins = torch.arange(0, L//2 + 1, 1, device=device)
    fsc_vals = []
    
    # 3. Compute correlation per shell
    # Numerator: sum(F1 * conj(F2))
    # Denominator: sqrt(sum(|F1|^2) * sum(|F2|^2))
    num_full = f1 * torch.conj(f2)
    den1_full = torch.abs(f1)**2
    den2_full = torch.abs(f2)**2
    
    for i in range(len(r_bins) - 1):
        r_start = r_bins[i]
        r_end = r_bins[i+1]
        
        mask_shell = (r_grid >= r_start) & (r_grid < r_end)
        
        if mask_shell.sum() == 0:
            fsc_vals.append(0.0)
            continue
            
        num = num_full[mask_shell].sum().real
        den1 = den1_full[mask_shell].sum()
        den2 = den2_full[mask_shell].sum()
        
        fsc = num / (torch.sqrt(den1 * den2) + 1e-8)
        fsc_vals.append(fsc.item())
        
    freqs = r_bins[:-1].cpu().numpy() / (L / 2) # Normalized frequency [0, 1]
    return freqs, np.array(fsc_vals)

if __name__ == "__main__":
    # Test
    L = 32
    v1 = torch.randn(L, L, L)
    v2 = v1 + torch.randn(L, L, L) * 0.1 # High correlation
    
    freqs, fsc = compute_fsc(v1, v2)
    print("FSC (Low Freq):", fsc[0:5])
    print("CC:", compute_cc(v1, v2))
