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

        # Determine the base name for lookup (e.g., "16pk_designed_8")
        base_name = file_name.replace('.pt', '')
        
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

            # Store the result
            spliced_sequences.append({
                "name": f"{base_name}_sample_{i}",
                "sequence": sampled_seq
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
        default="./logits/UBC2Model-antonia0824-partial/antonia0824",
        help="Directory containing the .pt logit files."
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
        default="antonia0824partial-design.json",
        help="Name of the output JSON file."
    )
    
    args = parser.parse_args()
    sample_and_splice(args)
