
import sys
import os
import torch
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np

def fetch_and_process(pdb_id, output_dir):
    print(f"Fetching {pdb_id}...")
    try:
        file_path = rcsb.fetch(pdb_id, "pdb", target_path=output_dir)
    except Exception as e:
        print(f"Failed to fetch: {e}")
        return None

    print(f"Processing {file_path}...")
    pdb_file = pdb.PDBFile.read(file_path)
    structure = pdb_file.get_structure(model=1)
    
    # Filter C-alpha
    ca = structure[structure.atom_name == "CA"]
    
    # Remove heteroatoms/water if any (biotite usually handles canonical)
    ca = ca[struc.filter_amino_acids(ca)]
    
    coords = torch.tensor(ca.coord, dtype=torch.float32)
    
    # Center
    coords = coords - coords.mean(dim=0, keepdim=True)
    
    output_path = os.path.join(output_dir, f"{pdb_id}_ca.pt")
    torch.save(coords, output_path)
    print(f"Saved processed data to {output_path} (Attributes: {coords.shape})")
    return output_path

if __name__ == "__main__":
    sandbox = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior\experiments\sandbox"
    os.makedirs(sandbox, exist_ok=True)
    fetch_and_process("1mbn", sandbox)
