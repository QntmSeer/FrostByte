
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_3d import UNet3D
from models.diffusion import DiffusionModel
from data.volume_dataset import VolumeDataset

def train_volume_model():
    # Force GPU if available, else standard
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Volumetric Model on {device}")
    
    # Hyperparams
    batch_size = 4 # 64^3 is heavy, keep batch size small
    lr = 2e-4
    epochs = 10 
    
    # 1. Data
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    
    # Voxelize to 64^3 with 1A resolution (approx 64A box)
    # Scale by 10.0 because CATH data was normalized by /10.0
    dataset = VolumeDataset(data_path, grid_size=64, voxel_size=1.0, sigma=1.0, coordinate_scale=10.0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Model
    # U-Net 3D
    net = UNet3D(in_ch=1, out_ch=1, time_dim=64)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 3. Training Loop
    model.train()
    save_dir = os.path.join(base, "experiments", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting volumetric training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # batch is (B, 1, 64, 64, 64)
            x = batch.to(device)
            
            optimizer.zero_grad()
            
            # Loss
            # DiffusionModel.get_loss expects (B, N, 3) for points or (B, C, D, H, W) for grids?
            # get_loss calls self.model(x_t, t) which returns noise.
            # Then MSE between predicted and actual noise.
            # This works for any shape as long as model outputs same shape as input.
            loss = model.get_loss(x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    # 4. Save
    save_path = os.path.join(save_dir, "ddpm_volume_unet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved Volumetric Model to {save_path}")

if __name__ == "__main__":
    train_volume_model()
