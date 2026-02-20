
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion import PointDiffusionTransformer, DiffusionModel
from data.dataset import CATHDataset, collate_fn

def train_generalist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Generalist Model on {device}")
    
    # Hyperparams
    batch_size = 4 # Moderate batch size
    lr = 1e-4
    epochs = 10 
    
    # 1. Data
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    data_path = os.path.join(base, "data", "processed", "cath_subset.pt")
    
    dataset = CATHDataset(data_path, augment_noise=0.01, max_len=256, max_atoms_limit=500)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 2. Model
    # We can either start from scratch or finetune. 
    # Let's start from scratch to prove "Generalist" capability from diverse data.
    # Or finetune from the 2-protein model. 
    # Given the small dataset (20 proteins), finetuning might be safer, 
    # but starting from scratch is cleaner for "Phase 4".
    # I'll initialize from scratch but keep the same architecture.
    
    net = PointDiffusionTransformer(hidden_dim=128, num_layers=4)
    model = DiffusionModel(net, timesteps=1000).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 3. Training Loop
    model.train()
    save_dir = os.path.join(base, "experiments", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, mask in pbar:
            x = x.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            
            # Loss = Simple MSE output of the model
            loss = model.get_loss(x, mask=mask)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
    # 4. Save
    save_path = os.path.join(save_dir, "ddpm_cath_generalist.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved Generalist Model to {save_path}")

if __name__ == "__main__":
    train_generalist()
