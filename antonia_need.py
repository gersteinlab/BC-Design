import os
import torch
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
import json

def sample_and_splice(args):
    """
    Main function to sample sequences from logits, splice them with ground truth,
    and save the results.
    """
    # --- 1. INITIALIZE MODEL COMPONENTS AND DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # --- 2. LOAD GROUND TRUTH SEQUENCES ---
    try:
        with open(args.json_path, 'r') as f:
            ground_truth_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json_path}")
        return

    # Create a dictionary for quick lookup of ground truth sequences by name
    ground_truth_map = {item['name']: item['seq'] for item in ground_truth_data}

    # --- 3. MAIN LOOP: SAMPLE, PROCESS, AND SPLICE ---
    spliced_sequences = []
    print("Starting sequence generation...")

    if not os.path.isdir(args.logits_dir):
        print(f"Error: Directory not found at {args.logits_dir}")
        return

    for file_name in tqdm(sorted(os.listdir(args.logits_dir))):
        if not file_name.endswith(".pt"):
            continue

        logits_path = os.path.join(args.logits_dir, file_name)
        logits = torch.load(logits_path).to(device)
        sequence_length = logits.shape[0]

        # Determine the base name for lookup (e.g., "16pk_designed_8")
        base_name = file_name.replace('.pt', '')
        
        # Check if the corresponding ground truth sequence exists
        if base_name not in ground_truth_map:
            print(f"Warning: No ground truth sequence found for {base_name}. Skipping.")
            continue
        
        ground_truth_seq = ground_truth_map[base_name]

        # --- Sampling Step ---
        for i in range(args.num_samples):
            # Apply temperature and sample
            scaled_logits = logits / args.temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)

            # Decode tokens
            raw_tokens = tokenizer.convert_ids_to_tokens(
                sampled_indices.tolist(), skip_special_tokens=True
            )
            sampled_seq = "".join(raw_tokens)

            # --- Validation Step ---
            if len(ground_truth_seq) != len(sampled_seq):
                print(f"Warning: Length mismatch for {file_name}.")
                print(f"Ground truth length: {len(ground_truth_seq)}, Sampled length: {len(sampled_seq)}")
                continue

            # --- Splicing Step ---
            gt_list = list(ground_truth_seq)
            sampled_list = list(sampled_seq)
            
            # Define splicing positions based on file version
            if "_v2" in file_name:
                # Positions for v2 files
                positions = [
                    (37, 45), (83, 91), (137, 146), (160, 160),
                    (171, 179), (193, 210), (260, 260)
                ]
            else:
                # Positions for v1 files
                positions = [
                    (41, 49), (87, 95), (141, 150), (164, 164),
                    (175, 183), (197, 214), (264, 264)
                ]

            for start, end in positions:
                # Adjust for 0-based indexing
                start_idx = start - 1
                end_idx = end
                gt_list[start_idx:end_idx] = sampled_list[start_idx:end_idx]

            final_sequence = "".join(gt_list)
            
            # Store the result
            spliced_sequences.append({
                "name": f"{base_name}_sample_{i}",
                "sequence": final_sequence
            })

    # --- 4. SAVE FINAL SEQUENCES ---
    with open(args.output_file, 'w') as f:
        json.dump(spliced_sequences, f, indent=4)
    
    print(f"\nProcessing complete.")
    print(f"{len(spliced_sequences)} sequences were generated and saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample, splice, and save protein sequences.")
    
    parser.add_argument(
        "--logits_dir", 
        type=str, 
        default="./logits/UBC2Model-bcmask1.01/antonia0824",
        help="Directory containing the .pt logit files."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="./data/antonia0824/antonia0824.json",
        help="Path to the JSON file with ground truth sequences."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5,
        help="Number of sequences to sample from each logit file."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.1,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="antonia0824bakboneonly-inference.json",
        help="Name of the output JSON file."
    )
    
    args = parser.parse_args()
    sample_and_splice(args)
