import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset

class SingleProteinVolume(Dataset):
    def __init__(self, pdb_id, data_path, grid_size=64, voxel_size=1.0, sigma=1.0, coordinate_scale=10.0):
        # Load specific protein
        data_dict = torch.load(data_path, weights_only=False)
        self.coords = data_dict[pdb_id] * coordinate_scale
        
        # Center
        self.coords = self.coords - self.coords.mean(dim=0, keepdim=True)
        
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.sigma = sigma
        
        # Precompute
        self.volume = VolumeDataset.voxelize_gaussian(self.coords, grid_size, voxel_size, sigma)
        self.volume = self.volume.unsqueeze(0) # (1, L, L, L)
        
    def __len__(self):
        return 100 # Repeat identical sample 100 times per epoch for faster loading
        
    def __getitem__(self, idx):
        return self.volume

def train_overfit_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Overfitting Volumetric Model on {device} for Visualization")
    
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    
    # 1. Data (Just 1MBN)
    # We'll use 1hel since it's definitely in cath_subset.pt, but the benchmark uses 1mbn.
    # Let's check if 1mbn is in there. If not, we'll use 1hel and change the benchmark title.
    # Actually benchmark fetches 1mbn from torchvision. Let's use 1hel which is in our dataset.
    dataset = SingleProteinVolume('1hel', data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Model
    net = UNet3D(in_ch=1, out_ch=1, time_dim=64)
    model = DiffusionModel(net, timesteps=1000).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # Higher LR for overfitting
    
    # 3. Train
    model.train()
    epochs = 50 # Increased to get a perfectly sharp image for LinkedIn
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch.to(device)
            optimizer.zero_grad()
            loss = model.get_loss(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    # 4. Save specifically for this demo
    save_path = os.path.join(base, "experiments", "checkpoints", "ddpm_volume_overfit.pth")
    torch.save(model.state_dict(), save_path)
    print("Saved overfit model.")

if __name__ == "__main__":
    train_overfit_model()
