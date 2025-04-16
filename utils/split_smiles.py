import argparse
import os
import math
from pathlib import Path

def count_lines(file_path):
    """Count the number of lines in a plain text file."""
    with open(file_path, "r") as f:
        return sum(1 for _ in f)

def split_smiles(input_file, output_dir, min_chunk_size=10000, target_num_chunks=1000):
    """
    Split SMILES file into chunks based on size.

    Args:
        input_file (str): Path to input SMILES file (plain text).
        output_dir (str): Directory to store output chunk files.
        min_chunk_size (int): Minimum number of SMILES per chunk (default: 10,000).
        target_num_chunks (int): Target number of chunks for large datasets (default: 1000).
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Count total SMILES
    print(f"Counting SMILES in {input_file}...")
    total_smiles = count_lines(input_file)
    print(f"Total SMILES: {total_smiles}")

    # Determine chunking strategy
    if total_smiles < min_chunk_size:
        # No splitting needed, copy to a single file
        output_file = os.path.join(output_dir, "smiles_0.smi")
        print(f"Dataset too small ({total_smiles} < {min_chunk_size}). Copying to {output_file}")
        with open(input_file, "r") as f_in:
            with open(output_file, "w") as f_out:
                f_out.writelines(f_in)
        return

    # Calculate chunk size
    # For large datasets, aim for ~target_num_chunks, but ensure at least min_chunk_size
    chunk_size = max(min_chunk_size, math.ceil(total_smiles / target_num_chunks))
    num_chunks = math.ceil(total_smiles / chunk_size)
    print(f"Splitting into {num_chunks} chunks of ~{chunk_size} SMILES each")

    # Open input file
    with open(input_file, "r") as f_in:
        chunk_idx = 0
        smiles_count = 0
        chunk_smiles = []

        for line in f_in:
            line = line.strip()
            if line:  # Skip empty lines
                chunk_smiles.append(line)
                smiles_count += 1

                if len(chunk_smiles) >= chunk_size:
                    # Write chunk to file
                    output_file = os.path.join(output_dir, f"smiles_{chunk_idx}.smi")
                    with open(output_file, "w") as f_out:
                        f_out.write("\n".join(chunk_smiles) + "\n")
                    print(f"Wrote {len(chunk_smiles)} SMILES to {output_file}")
                    chunk_idx += 1
                    chunk_smiles = []

        # Write remaining SMILES (if any)
        if chunk_smiles:
            output_file = os.path.join(output_dir, f"smiles_{chunk_idx}.smi")
            with open(output_file, "w") as f_out:
                f_out.write("\n".join(chunk_smiles) + "\n")
            print(f"Wrote {len(chunk_smiles)} SMILES to {output_file}")

    print(f"Finished splitting. Created {chunk_idx + 1} chunk(s) in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Split a SMILES file into chunks.")
    parser.add_argument("--input", required=True, help="Input SMILES file (plain text, e.g., .smi)")
    parser.add_argument("--output", required=True, help="Output directory for chunked files")
    parser.add_argument("--target_num", type=int, default=1000, help="Target number of chunks (default: 1000)")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist")

    # Validate target_num
    if args.target_num <= 0:
        raise ValueError("target_num must be positive")

    # Run splitting
    split_smiles(args.input, args.output, target_num_chunks=args.target_num)

if __name__ == "__main__":
    main()
