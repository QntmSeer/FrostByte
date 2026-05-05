
import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.volume_dataset import VolumeDataset

def check_scaling():
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    
    # 1. Load OLD dataset (Scale=1.0)
    ds_old = VolumeDataset(data_path, grid_size=64, coordinate_scale=1.0)
    vol_old = ds_old[0] # (1, 64, 64, 64)
    
    # 2. Load NEW dataset (Scale=10.0)
    ds_new = VolumeDataset(data_path, grid_size=64, coordinate_scale=10.0)
    vol_new = ds_new[0]
    
    # Viz middle slice
    mid = 32
    slice_old = vol_old[0, mid].cpu().numpy()
    slice_new = vol_new[0, mid].cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(slice_old, cmap='viridis')
    ax[0].set_title(f"Old Input (Scale=1.0)\nMax Density: {slice_old.max():.2f}")
    
    ax[1].imshow(slice_new, cmap='viridis')
    ax[1].set_title(f"New Input (Scale=10.0)\nMax Density: {slice_new.max():.2f}")
    
    plt.suptitle("Training Data Input Comparison\nWhat the model actually sees")
    save_path = os.path.join(base, "experiments", "sandbox", "training_data_check.png")
    plt.savefig(save_path)
    print(f"Saved comparison to {save_path}")

if __name__ == "__main__":
    check_scaling()
