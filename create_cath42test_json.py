import json
import os
import numpy as np
from tqdm import tqdm

def create_test_set_json():
    """
    Extracts, cleans, and reformats the CATH 4.2 test set from source files
    and saves it as a new JSON file.
    """
    # Define file paths
    splits_file = './data/cath4.2/chain_set_splits.json'
    data_file = './data/cath4.2/chain_set.jsonl'
    output_dir = './data/cath4.2test'
    output_file = os.path.join(output_dir, 'cath4.2test.json')

    print(f"Reading splits file from: {splits_file}")
    print(f"Reading data file from: {data_file}")
    print(f"Will write output to: {output_file}")

    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load the list of test set identifiers
    try:
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        test_ids = set(splits['test'])
        print(f"Successfully loaded {len(test_ids)} test set IDs.")
    except FileNotFoundError:
        print(f"Error: Splits file not found at {splits_file}")
        return
    
    # 3. Initialize a list to hold the processed test data
    processed_test_data = []

    # 4. Process the main data file
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        print(f"Processing {len(lines)} entries from the source JSONL file...")
        for line in tqdm(lines, desc="Filtering and processing test set"):
            entry = json.loads(line)

            # 5. Filter for entries that are in the test set
            if entry['name'] in test_ids:
                
                # --- Data Cleaning (replicates logic from _load_metadata) ---
                
                # Convert coordinate lists to NumPy arrays
                coords_dict = entry['coords']
                n_coords = np.array(coords_dict['N'])
                ca_coords = np.array(coords_dict['CA'])
                c_coords = np.array(coords_dict['C'])
                o_coords = np.array(coords_dict['O'])
                
                # Stack coordinates to check for invalid values per residue
                # The order of stacking here doesn't matter for the validity check
                coords_stack = np.stack([ca_coords, c_coords, o_coords, n_coords], axis=1)

                # Create a mask for residues containing NaN or infinite values
                nan_mask = np.isnan(coords_stack).sum(axis=(1, 2)) > 0
                inf_mask = np.isinf(coords_stack).sum(axis=(1, 2)) > 0
                invalid_residue_mask = nan_mask | inf_mask
                
                # Keep only the valid residues
                valid_mask = ~invalid_residue_mask
                
                clean_n = n_coords[valid_mask]
                clean_ca = ca_coords[valid_mask]
                clean_c = c_coords[valid_mask]
                clean_o = o_coords[valid_mask]
                
                # Filter the sequence string accordingly
                original_seq = entry['seq']
                clean_seq = "".join([original_seq[i] for i, is_valid in enumerate(valid_mask) if is_valid])

                # --- Coordinate Restructuring ---

                # Reformat coordinates to match the target structure:
                # A list of residues, where each residue is a list of [N, CA, C, O] atom coordinates.
                restructured_coords = [
                    list(res_atoms) for res_atoms in zip(
                        clean_n.tolist(), 
                        clean_ca.tolist(), 
                        clean_c.tolist(), 
                        clean_o.tolist()
                    )
                ]

                # 6. Assemble the final dictionary for this entry
                processed_entry = {
                    'name': entry['name'],
                    'seq': clean_seq,
                    'coords': restructured_coords
                }
                processed_test_data.append(processed_entry)

    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        return

    # 7. Write the final list of processed data to the output JSON file
    print(f"\nProcessed {len(processed_test_data)} test entries.")
    print(f"Writing data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(processed_test_data, f, indent=4)

    print("Done!")

# Run the function
if __name__ == '__main__':
    create_test_set_json()