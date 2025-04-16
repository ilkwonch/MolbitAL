
"""
Parallel fingerprint generator using RDKit to create Morgan fingerprints from an SMI file.
Generates sparse fingerprints (on-bit indices) with chirality, saving to a condensed text file.


Example usage (one-liner):

python isomer_enumerator.py --input_smi smiles/smiles_all_01.smi --output_smi isomers.smi --max_centers 10 --enum_nitrogen --enum_bridgehead

 python fingerprint_generator.py --input_smi smiles_all/smiles_all_00_tautomers.smi --output_file fingerprints.txt --processes 4 --quiet

"""

import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple
from multiprocessing import Pool
from tqdm import tqdm
import os
#from rdkit.Chem import rdFingerprintGenerator

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate sparse Morgan fingerprints from an SMI file using RDKit with parallel processing.")
    parser.add_argument(
        "--input_smi",
        type=str,
        required=True,
        help="Path to input SMI file containing SMILES and IDs (format: SMILES ID)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="fingerprints.txt",
        help="Path to output text file for sparse fingerprints (default: fingerprints.txt)."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Radius for Morgan fingerprints (default: 2, equivalent to ECFP4)."
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=1024,
        help="Number of bits for the fingerprint vector (default: 1024)."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of parallel processes (default: 4)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress status messages for quieter operation."
    )
    return parser.parse_args()

def read_smi_file(input_smi: str, quiet: bool = False) -> List[Tuple[str, str]]:
    """Read SMILES and IDs from an SMI file."""
    smiles_data = []
    with open(input_smi, 'r') as f:
        for line in f:
            if line.strip():
                smi, idx = line.strip().split(maxsplit=1)
                smiles_data.append((smi, idx))
    if not quiet:
        print(f"Found {len(smiles_data)} SMILES to process.")
    return smiles_data

def generate_fingerprint(args: Tuple[str, str, int, int]) -> Tuple[str, List[int]]:
    """Generate sparse Morgan fingerprint (on-bit indices) for a single SMILES string."""
    smi, idx, radius, n_bits = args
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, []  
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, useChirality=True)
   
    #mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
    #fp = mfpgen.GetFingerprint(mol)

    on_bits = [i for i in range(n_bits) if fp.GetBit(i)]
    return idx, on_bits

def write_fingerprints(output_file: str, fingerprints: List[Tuple[str, List[int]]], quiet: bool = False):
    """Write sparse fingerprints to a condensed text file."""
    with open(output_file, 'w') as f:
        for idx, on_bits in fingerprints:
            if on_bits:  
                f.write(f"{idx},{','.join(map(str, on_bits))}\n")
            else:
                f.write(f"{idx}\n")  
    if not quiet:
        print(f"Wrote {len(fingerprints)} fingerprints to {output_file}")

def main():
    """Main function to run the fingerprint generator."""
    args = parse_arguments()

    if not args.quiet:
        print(f"Reading input SMILES from {args.input_smi}...")
    smiles_data = read_smi_file(args.input_smi, args.quiet)

    fp_args = [(smi, idx, args.radius, args.n_bits) for smi, idx in smiles_data]

    if not args.quiet:
        print(f"Generating fingerprints with {args.processes} processes...")
    with Pool(processes=args.processes) as pool:
        fingerprints = list(tqdm(pool.imap(generate_fingerprint, fp_args), total=len(fp_args), disable=args.quiet))

    write_fingerprints(args.output_file, fingerprints, args.quiet)
    if not args.quiet:
        print("Fingerprint generation completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
