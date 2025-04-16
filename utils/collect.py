import argparse
import os
import numpy as np
from glob import glob
from pathlib import Path
import random
from joblib import Parallel, delayed
import linecache

def count_lines(file):
    """Count non-empty lines in a file."""
    with open(file, "r") as f:
        return sum(1 for line in f if line.strip())

def get_file_pairs(smiles_dir, fingerprint_dir):
    """Get paired SMILES and fingerprint files, sorted by index."""
    smiles_files = glob(os.path.join(smiles_dir, "smiles_*.smi"))
    smiles_indices = []
    for f in smiles_files:
        try:
            index = int(os.path.basename(f).replace("smiles_", "").replace(".smi", ""))
            smiles_indices.append((f, index))
        except ValueError:
            continue
    smiles_indices.sort(key=lambda x: x[1])
    smiles_files = [f for f, _ in smiles_indices]

    fingerprint_files = []
    for smiles_file, index in smiles_indices:
        fp_file = os.path.join(fingerprint_dir, f"smiles_{index}_fingerprints.txt")
        if not os.path.exists(fp_file):
            raise FileNotFoundError(f"Fingerprint file {fp_file} not found for {smiles_file}")
        fingerprint_files.append(fp_file)

    if len(smiles_files) != len(fingerprint_files):
        raise ValueError(f"Mismatch: {len(smiles_files)} SMILES files vs {len(fingerprint_files)} fingerprint files")

    return list(zip(smiles_files, fingerprint_files))

def collect_random_compounds(smiles_files, fingerprint_files, num_compounds, total_smiles, counts):
    """
    Randomly select num_compounds from SMILES and fingerprints efficiently.

    Args:
        smiles_files (list): List of SMILES file paths.
        fingerprint_files (list): List of corresponding fingerprint file paths.
        num_compounds (int): Number of compounds to select.
        total_smiles (int): Total number of SMILES across all files.
        counts (list): Number of SMILES per file.

    Returns:
        Tuple of (selected_smiles, selected_fingerprints).
    """
    if num_compounds > total_smiles:
        raise ValueError(f"Requested {num_compounds} compounds, but only {total_smiles} available")

    print(f"Selecting {num_compounds} compounds from {total_smiles} total SMILES")

    # Estimate number of files to sample (oversample to ensure enough compounds)
    avg_smiles_per_file = total_smiles / len(smiles_files)
    est_files_needed = max(1, int(num_compounds / avg_smiles_per_file * 2))  # Oversample by 2x
    est_files_needed = min(est_files_needed, len(smiles_files))

    # Randomly select files
    file_indices = random.sample(range(len(smiles_files)), est_files_needed)
    selected_smiles_files = [smiles_files[i] for i in file_indices]
    selected_fp_files = [fingerprint_files[i] for i in file_indices]
    selected_counts = [counts[i] for i in file_indices]

    # Compute cumulative counts for selected files
    cum_counts = np.cumsum([0] + selected_counts)
    total_selected_smiles = cum_counts[-1]

    if num_compounds > total_selected_smiles:
        raise ValueError(f"Selected files have only {total_selected_smiles} SMILES, need {num_compounds}")

    # Sample indices from selected files
    selected_indices = random.sample(range(total_selected_smiles), num_compounds)
    selected_indices.sort()

    selected_smiles = []
    selected_fingerprints = []
    idx_pos = 0  # Position in selected_indices

    # Process each selected file
    for file_idx, (s_file, f_file, count) in enumerate(zip(selected_smiles_files, selected_fp_files, selected_counts)):
        file_indices = []
        # Collect indices for this file
        while idx_pos < len(selected_indices) and cum_counts[file_idx] <= selected_indices[idx_pos] < cum_counts[file_idx + 1]:
            file_indices.append(selected_indices[idx_pos] - cum_counts[file_idx] + 1)  # 1-based for linecache
            idx_pos += 1

        if file_indices:
            # Read only required lines
            smiles = [linecache.getline(s_file, i).strip() for i in file_indices]
            fingerprints = [linecache.getline(f_file, i).strip() for i in file_indices]

            if len(smiles) != len(fingerprints):
                raise ValueError(f"Mismatch in {s_file} and {f_file} for selected lines")

            selected_smiles.extend(smiles)
            selected_fingerprints.extend(fingerprints)

    return selected_smiles, selected_fingerprints

def main():
    parser = argparse.ArgumentParser(description="Collect random SMILES and fingerprints for training.")
    parser.add_argument("--smiles_dir", required=True, help="Directory containing split SMILES files (smiles_*.smi)")
    parser.add_argument("--fingerprint_dir", required=True, help="Directory containing fingerprint files (fingerprints_*.txt)")
    parser.add_argument("--train_dir", required=True, help="Output directory for training data")
    parser.add_argument("--num_compounds", type=int, default=1000, help="Number of compounds to collect (default: 1000)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.smiles_dir):
        raise FileNotFoundError(f"SMILES directory {args.smiles_dir} does not exist")
    if not os.path.exists(args.fingerprint_dir):
        raise FileNotFoundError(f"Fingerprint directory {args.fingerprint_dir} does not exist")
    if args.num_compounds <= 0:
        raise ValueError("num_compounds must be positive")

    # Create train_dir
    Path(args.train_dir).mkdir(parents=True, exist_ok=True)

    # Get paired SMILES and fingerprint files
    file_pairs = get_file_pairs(args.smiles_dir, args.fingerprint_dir)
    smiles_files, fingerprint_files = zip(*file_pairs)

    # Count SMILES in parallel
    print("Counting SMILES across files...")
    counts = Parallel(n_jobs=-1)(delayed(count_lines)(f) for f in smiles_files)
    total_smiles = sum(counts)
    print(f"Total SMILES: {total_smiles}")

    # Collect random compounds
    selected_smiles, selected_fingerprints = collect_random_compounds(
        smiles_files, fingerprint_files, args.num_compounds, total_smiles, counts
    )

    # Save outputs
    smiles_output = os.path.join(args.train_dir, "train_smiles.smi")
    with open(smiles_output, "w") as f:
        f.write("\n".join(selected_smiles) + "\n")
    print(f"Saved {len(selected_smiles)} SMILES to {smiles_output}")

    fp_output = os.path.join(args.train_dir, "train_fingerprints.txt")
    with open(fp_output, "w") as f:
        f.write("\n".join(selected_fingerprints) + "\n")
    print(f"Saved {len(selected_fingerprints)} fingerprints to {fp_output}")

if __name__ == "__main__":
    main()
