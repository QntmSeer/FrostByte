import os
import torch
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
from tqdm import tqdm
import argparse
import requests
import numpy as np

def get_diverse_pdb_ids(limit=1000):
    """
    Query RCSB PDB for a diverse set of protein structures.
    We want X-ray structures, resolution < 2.5A, and representative at 30% sequence identity.
    Using the RCSB Search API.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 2.5
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                }
            ]
        },
        "request_options": {
            "return_all_hits": True
        },
        "return_type": "polymer_entity"
    }

    print("Querying RCSB PDB for high-quality single-chain structures...")
    response = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    
    if response.status_code != 200:
        print(f"Error querying RCSB API: {response.text}")
        return []

    data = response.json()
    all_hits = data.get('result_set', [])
    
    # Extract unique PDB IDs (hits are in format '1ABC_1')
    pdb_ids = list(set([hit['identifier'].split('_')[0] for hit in all_hits]))
    
    # Note: To ensure true 30% sequence diversity, we can use the PDB sequence clusters,
    # but the above query at least gets us high-quality single chain X-ray structures.
    # We will pick the top 'limit' ones. To add a bit of randomness:
    import random
    random.seed(42)
    random.shuffle(pdb_ids)
    
    return pdb_ids[:limit]

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
        
        # Must be single chain
        chains = np.unique(ca.chain_id)
        if len(chains) > 1:
            return None # Ignore multi-chain complexes
            
        if len(ca) < 50 or len(ca) > 500: # Filter by size to keep reasonable domain sizes
            return None

        coords = torch.tensor(ca.coord, dtype=torch.float32)
        
        # Center at origin
        coords = coords - coords.mean(dim=0, keepdim=True)
        
        # Normalize to approx unit variance (Angstroms -> Latent)
        coords = coords / 10.0
        
        # Delete raw file to save space if running on HPC
        os.remove(file_path)
        
        return coords
    except Exception as e:
        print(f"Failed to process {pdb_id}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000, help="Target number of proteins")
    parser.add_argument("--output", type=str, default="data/processed/cath_s40.pt", help="Output .pt path")
    args = parser.parse_args()

    base = r"c:\Users\Gebruiker\Documents\Computational Bio\diffusion-cryoem-prior"
    # Or current working dir if on HPC
    if not os.path.exists(os.path.join(base, "data")):
        base = os.getcwd()

    raw_dir = os.path.join(base, "data", "raw", "temp_fetch")
    proc_dir = os.path.dirname(os.path.join(base, args.output))
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    candidates = get_diverse_pdb_ids(limit=args.limit * 3) # Fetch extra since some will fail size filters
    print(f"Retrieved {len(candidates)} candidate IDs.")
    
    val_set = {}
    
    pbar = tqdm(candidates, desc=f"Processing up to {args.limit}")
    for pdb_id in pbar:
        coords = fetch_and_process(pdb_id, raw_dir)
        if coords is not None:
            val_set[pdb_id] = coords
            pbar.set_postfix(success=len(val_set))
            
        if len(val_set) >= args.limit:
            break
            
    # Save the consolidated dataset
    save_path = os.path.join(base, args.output)
    torch.save(val_set, save_path)
    print(f"\nSuccessfully processed {len(val_set)} proteins.")
    print(f"Saved consolidated dataset to {save_path}")

if __name__ == "__main__":
    main()
