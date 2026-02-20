"""
Dataset classes for 3D protein structure diffusion model.

Supports:
1. Single structure (with noise augmentation) — original mode
2. Conformational ensemble — research-grade mode (Upgrade 1)
"""

import torch
from torch.utils.data import Dataset
import os


class ProteinStructureDataset(Dataset):
    """Original dataset: single structure with noise augmentation."""
    
    def __init__(self, data_path, augment_noise=0.0):
        try:
            self.data = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data = torch.load(data_path)

        self.coords = self.data['coords']  # (N, 3)
        self.augment_noise = augment_noise
        self.epoch_len = 1000

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        x = self.coords.clone()
        if self.augment_noise > 0:
            x = x + torch.randn_like(x) * self.augment_noise
        return x


class ConformationalEnsembleDataset(Dataset):
    """
    Ensemble dataset: samples from a distribution of conformers.
    
    This is the key upgrade — instead of memorizing one structure,
    the diffusion model learns p(x_3D) over a manifold of conformations.
    """
    
    def __init__(self, data_path, augment_noise=0.005):
        """
        Args:
            data_path: path to ensemble .pt file
            augment_noise: additional noise on top of ensemble variation
        """
        try:
            self.data = torch.load(data_path, weights_only=False)
        except TypeError:
            self.data = torch.load(data_path)
        
        self.ensemble = self.data['ensemble']  # (M, N, 3)
        self.reference = self.data.get('reference', None)
        self.n_conformers = self.ensemble.shape[0]
        self.augment_noise = augment_noise
        
        print(f"Loaded ensemble: {self.n_conformers} conformers, "
              f"{self.ensemble.shape[1]} atoms each")

    def __len__(self):
        return self.n_conformers

    def __getitem__(self, idx):
        x = self.ensemble[idx].clone()
        if self.augment_noise > 0:
            x = x + torch.randn_like(x) * self.augment_noise
        return x


class MultiProteinDataset(Dataset):
    """
    Research Upgrade: Dataset that combines multiple protein ensembles.
    Handles varying atom counts via padding.
    """
    def __init__(self, data_paths, augment_noise=0.005):
        self.datasets = [ConformationalEnsembleDataset(p, augment_noise) for p in data_paths]
        self.lengths = [len(d) for d in self.datasets]
        self.total_len = sum(self.lengths)
        self.max_atoms = max([d.ensemble.shape[1] for d in self.datasets])
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # find which dataset
        curr = 0
        for d in self.datasets:
            if idx < curr + len(d):
                x = d[idx - curr]
                n_atoms = x.shape[0]
                # Pad to max_atoms
                # (N, 3) -> (max_N, 3)
                padded_x = torch.zeros((self.max_atoms, 3))
                padded_x[:n_atoms] = x
                mask = torch.zeros(self.max_atoms, dtype=torch.bool)
                mask[:n_atoms] = True
                return padded_x, mask
            curr += len(d)

class CATHDataset(Dataset):
    """
    Generalist Dataset: Loads a dictionary of {pdb_id: coords} from a single .pt file.
    Pads all proteins to the size of the largest protein in the batch/dataset.
    """
    def __init__(self, data_path, augment_noise=0.005, max_len=256, max_atoms_limit=500):
        try:
            self.data_dict = torch.load(data_path, weights_only=False)
        except:
            self.data_dict = torch.load(data_path)
            
        all_ids = list(self.data_dict.keys())
        
        # Filter out large proteins to prevent OOM
        self.pdb_ids = []
        self.coords_list = []
        
        for k in all_ids:
            c = self.data_dict[k]
            if c.shape[0] <= max_atoms_limit:
                self.pdb_ids.append(k)
                self.coords_list.append(c)
            else:
                print(f"Skipping {k}: {c.shape[0]} atoms > {max_atoms_limit}")
        
        if not self.pdb_ids:
            raise ValueError(f"No proteins found with < {max_atoms_limit} atoms")

        # Determine max atoms for padding
        self.max_atoms_in_data = max([c.shape[0] for c in self.coords_list])
        self.max_atoms = max(max_len, self.max_atoms_in_data)
        
        self.augment_noise = augment_noise
        print(f"Loaded CATH Dataset: {len(self.pdb_ids)} structures. Max atoms: {self.max_atoms_in_data} (Padding to {self.max_atoms})")

    def __len__(self):
        # We can artificially increase epoch length if needed, or just iterate once per epoch
        return len(self.pdb_ids) * 100 # Augment 100x per epoch

    def __getitem__(self, idx):
        # Map augmented index to real index
        real_idx = idx % len(self.pdb_ids)
        
        x = self.coords_list[real_idx].clone()
        n_atoms = x.shape[0]
        
        # Apply noise augmentation
        if self.augment_noise > 0:
            x = x + torch.randn_like(x) * self.augment_noise
            
        # Pad to max_atoms
        padded_x = torch.zeros((self.max_atoms, 3))
        padded_x[:n_atoms] = x
        
        # Create mask (1 for real atom, 0 for pad)
        mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        mask[:n_atoms] = True
        
        return padded_x, mask

def collate_fn(batch):
    """Custom collate for padded proteins."""
    # filtering out None/Errors if any
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None, None
    
    coords = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    return coords, masks
