import os
import torch
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm

def hamming_distance_batch(seqs, device):
    """
    Calculates the average pairwise Hamming distance for a batch of sequences.

    Args:
        seqs (list): A list of strings, all of which must be of equal length.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.

    Returns:
        float: The average pairwise Hamming distance.
    """
    if not seqs:
        return 0.0
    
    n = len(seqs)
    L = len(seqs[0])

    # Convert list of strings to a numerical tensor for efficient processing
    seq_tensor = torch.tensor([[ord(c) for c in s] for s in seqs], dtype=torch.int16, device=device)

    # Expand dimensions to enable broadcasting for pairwise comparison
    # seq1 becomes [n, 1, L], seq2 becomes [1, n, L]
    seq1 = seq_tensor.unsqueeze(1)
    seq2 = seq_tensor.unsqueeze(0)

    # Compare all sequences against all others element-wise and sum the differences (mismatches)
    # The result is a matrix where diff_matrix[i, j] is the Hamming distance between seq i and seq j
    diff_matrix = (seq1 != seq2).sum(dim=2)  # Shape: [n, n]

    # To avoid duplicate comparisons (i,j vs j,i) and self-comparisons (i,i),
    # we only take the upper triangle of the matrix.
    triu_indices = torch.triu_indices(n, n, offset=1)
    pairwise_distances = diff_matrix[triu_indices[0], triu_indices[1]]
    
    # Return the average of all unique pairwise distances
    return pairwise_distances.float().mean().item()

def main(args):
    """Main function to run the sampling and diversity calculation."""
    
    # --- 1. INITIALIZE MODEL COMPONENTS AND DEVICE ---
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Logits directory: {args.logits_dir}")
    print(f"Number of samples per file: {args.num_samples}")
    print(f"Sampling temperature: {args.temperature}\n")

    # Load the tokenizer for converting token IDs back to amino acids
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    results = []

    # --- 2. MAIN LOOP: SAMPLE SEQUENCES AND CALCULATE DIVERSITY ---
    print("Starting sampling and diversity calculation...")
    if not os.path.isdir(args.logits_dir):
        print(f"Error: Directory not found at {args.logits_dir}")
        return

    for file_name in tqdm(sorted(os.listdir(args.logits_dir))): # Sorted for consistent order
        if not file_name.endswith(".pt"):
            continue

        logits_path = os.path.join(args.logits_dir, file_name)
        
        # Load logits and move to the selected device
        log_probs_masked = torch.load(logits_path).to(device)
        sequence_length = log_probs_masked.shape[0]

        # --- Sampling Step ---
        sampled_sequences = []
        for _ in range(args.num_samples):
            # Apply temperature to logits to control randomness
            scaled_logits = log_probs_masked / args.temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Sample token indices based on the probability distribution
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)

            # Convert token IDs back to string tokens
            raw_tokens = tokenizer.convert_ids_to_tokens(
                sampled_indices.tolist(), skip_special_tokens=False
            )

            # --- Token Processing Step ---
            try:
                eos_index = raw_tokens.index("<eos>")
            except ValueError:
                eos_index = None

            tokens = ["X" if t.startswith("<") and t.endswith(">") else t for t in raw_tokens]

            if eos_index is not None:
                final_tokens = tokens[:eos_index]
                final_tokens.extend(["X"] * (sequence_length - len(final_tokens)))
            else:
                final_tokens = tokens
            
            final_tokens = final_tokens[:sequence_length]
            
            sequence = "".join(final_tokens)
            sampled_sequences.append(sequence)
        
        # --- Diversity Calculation Step ---
        if sampled_sequences:
            avg_hd = hamming_distance_batch(sampled_sequences, device)
            normalized_diversity = avg_hd / sequence_length
            results.append((file_name, normalized_diversity))
            # print(f"{file_name}: diversity = {normalized_diversity:.4f}")

    # --- 3. FINAL RESULTS ---
    if results:
        overall_avg = sum(v for _, v in results) / len(results)
        print(f"\n-----------------------------------------")
        print(f"Overall Average Diversity: {overall_avg:.4f}")
        print(f"-----------------------------------------")

if __name__ == "__main__":
    # --- COMMAND-LINE ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Sample sequences from logits and calculate diversity.")
    
    parser.add_argument(
        "--logits_dir", 
        type=str, 
        default="./logits/UBC2Model/CATH4.2",
        help="Directory containing the .pt logit files."
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=64,
        help="Number of sequences to sample from each logit file."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=1.0,
        help="Sampling temperature. <1 sharpens, >1 flattens the distribution."
    )
    
    args = parser.parse_args()
    main(args)
