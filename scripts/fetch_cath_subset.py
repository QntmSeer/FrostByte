
import os
import torch
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np
from tqdm import tqdm

# A manually curated subset of CATH domains representing diverse folds
# 1. Mainly Alpha
# 2. Mainly Beta
# 3. Alpha Beta
CATH_SUBSET = [
    "1hel", "1ubq", # The classics (Lysozyme, Ubiquitin) - Myoglobin (1mbn) removed for OOD test
    "2vii", # Villin Headpiece (Alpha)
    "1r69", # 434 Repressor (Alpha)
    "1bxz", # Bungarotoxin (Beta)
    "2cro", # Cro Repressor (Alpha)
    "1igd", # IgG-binding domain (Beta)
    "1fca", # Ferritin (Alpha)
    "1hrc", # Horse heart cytochrome c (Alpha)
    "2ptl", # Protein L (Beta)
    "1shf", # SH3 domain (Beta)
    "1ten", # Fibronectin Type III (Beta)
    "3icb", # Calbindin (Alpha)
    "1pht", # Phosphotransferase (Alpha/Beta)
    "5pti", # BPTI (Alpha/Beta)
    "1a3n", # Hemoglobin (Alpha)
    "1tit", # Titin (Beta)
    "1vii", # Villin (Alpha)
    "1yrn"  # Rubredoxin (Beta)
]

def fetch_and_process(pdb_id, output_dir):
    try:
        file_path = rcsb.fetch(pdb_id, "pdb", target_path=output_dir, verbose=False)
    except Exception as e:
        print(f"Failed to fetch {pdb_id}: {e}")
        return None

    if file_path is None: return None

    try:
        pdb_file = pdb.PDBFile.read(file_path)
        structure = pdb_file.get_structure(model=1)
        
        # Filter C-alpha
        ca = structure[structure.atom_name == "CA"]
        ca = ca[struc.filter_amino_acids(ca)]
        
        if len(ca) < 10: # Minimum size check
            return None

        coords = torch.tensor(ca.coord, dtype=torch.float32)
        
        # Center at origin
        coords = coords - coords.mean(dim=0, keepdim=True)
        
        # Normalize to approx unit variance (Angstroms -> Latent)
        # Typical protein radius is ~15-30A. Dividing by 10 puts it in ~1.5-3 range.
        coords = coords / 10.0
        
        return coords
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None

def main():
    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    raw_dir = os.path.join(base, "data", "raw", "cath_subset")
    proc_dir = os.path.join(base, "data", "processed")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    print(f"Fetching {len(CATH_SUBSET)} diverse structures...")
    
    val_set = {} # Dictionary to store all structures
    
    count = 0
    for pdb_id in tqdm(CATH_SUBSET):
        coords = fetch_and_process(pdb_id, raw_dir)
        if coords is not None:
            val_set[pdb_id] = coords
            count += 1
            
    # Save the consolidated dataset
    save_path = os.path.join(proc_dir, "cath_subset.pt")
    torch.save(val_set, save_path)
    print(f" Successfully processed {count} proteins.")
    print(f" Saved consolidated dataset to {save_path}")

if __name__ == "__main__":
    main()
