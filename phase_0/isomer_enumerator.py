
"""
Flipper-style stereoisomer generator using OpenEye's OEFlipper from an SMI file.
Enumerates possible stereoisomers for input SMILES and writes them to an output SMI file.
Supports enumeration of nitrogen and bridgehead stereocenters.


Example usage (one-liner):

python isomer_enumerator.py --input_smi smiles/smiles_all_01.smi --output_smi isomers.smi --max_centers 10 --enum_nitrogen --enum_bridgehead


"""

import os
import argparse
from openeye import oechem
from openeye import oeomega
from rdkit import Chem
from typing import List, Tuple
from tqdm import tqdm

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate stereoisomers from an SMI file using OpenEye Flipper.")
    parser.add_argument(
        "--input_smi",
        type=str,
        required=True,
        help="Path to input SMI file containing SMILES and IDs (format: SMILES ID)."
    )
    parser.add_argument(
        "--output_smi",
        type=str,
        default="stereoisomers.smi",
        help="Path to output SMI file for enumerated stereoisomers (default: stereoisomers.smi)."
    )
    parser.add_argument(
        "--max_centers",
        type=int,
        default=12,
        help="Maximum number of stereocenters to enumerate (default: 12). Higher values increase computation time."
    )
    parser.add_argument(
        "--enum_nitrogen",
        action="store_true",
        help=(
            "Enable enumeration of nonterminal nitrogen stereocenters. "
            "Nonterminal nitrogens with pyramidal geometry (e.g., in amines) and no more than two ring bonds "
            "are considered 'invertible'. If enabled, all possible puckers (inversions) are enumerated. "
            "Useful for molecules where nitrogen stereochemistry is relevant, but may increase output size."
        )
    )
    parser.add_argument(
        "--enum_bridgehead",
        action="store_true",
        help=(
            "Enable enumeration of bridgehead stereocenters. "
            "Bridgehead atoms are those at the junction of fused rings (e.g., in bicyclic systems). "
            "If enabled, stereoisomers involving bridgehead chirality are generated. "
            "Typically relevant for complex polycyclic molecules; use cautiously as it may generate additional isomers."
        )
    )
    return parser.parse_args()

def read_smi_file(input_smi: str) -> List[Tuple[str, str]]:
    """Read SMILES and IDs from an SMI file."""
    smiles_data = []
    with open(input_smi, 'r') as f:
        for line in f:
            if line.strip():
                smi, idx = line.strip().split(maxsplit=1)
                smiles_data.append((smi, idx))
    return smiles_data

def enumerate_stereoisomers(
    smi: str, idx: str, max_centers: int, enum_nitrogen: bool, enum_bridgehead: bool
) -> List[Tuple[str, str]]:
    """Enumerate stereoisomers for a single SMILES using OEFlipper."""
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smi):
        print(f"Failed to parse SMILES: {smi} ({idx})")
        return [(smi, idx)]  # Return original if parsing fails

    # Set up Flipper options
    flipper_opts = oeomega.OEFlipperOptions()
    flipper_opts.SetMaxCenters(max_centers)  # Max number of stereocenters to enumerate
    flipper_opts.SetEnumNitrogen(enum_nitrogen)  # Enumerate invertible nitrogens
    flipper_opts.SetEnumBridgehead(enum_bridgehead)  # Enumerate bridgehead stereocenters
    flipper_opts.SetEnumEZ(True)  # Enumerate E/Z isomers for double bonds
    flipper_opts.SetEnumRS(True)  # Enumerate R/S isomers for chiral centers

    stereoisomers = []
    for i, enantiomer in enumerate(oeomega.OEFlipper(mol.GetActive(), flipper_opts)):
        enantiomer = oechem.OEMol(enantiomer)
        new_smi = oechem.OEMolToSmiles(enantiomer)
        new_idx = f"{idx}_iso{i}"  # Use '_iso' suffix for stereoisomer IDs
        stereoisomers.append((new_smi, new_idx))
    
    if not stereoisomers:  # If no stereoisomers generated, return original
        stereoisomers.append((smi, idx))
    
    return stereoisomers

def write_smi_file(output_smi: str, stereoisomers: List[Tuple[str, str]]):
    """Write enumerated stereoisomers to an output SMI file."""
    with open(output_smi, 'w') as f:
        for smi, idx in stereoisomers:
            f.write(f"{smi} {idx}\n")

def main():
    """Main function to run the stereoisomer generator."""
    args = parse_arguments()

    # Verify OpenEye license
    if "OE_LICENSE" not in os.environ:
        raise EnvironmentError("OpenEye license (OE_LICENSE) not detected. Please set up the license to use this tool.")

    # Read input SMI file
    print(f"Reading input SMILES from {args.input_smi}...")
    smiles_data = read_smi_file(args.input_smi)
    print(f"Found {len(smiles_data)} SMILES to process.")

    # Enumerate stereoisomers
    all_stereoisomers = []
    for smi, idx in tqdm(smiles_data, desc="Enumerating stereoisomers"):
        stereoisomers = enumerate_stereoisomers(
            smi, idx, args.max_centers, args.enum_nitrogen, args.enum_bridgehead
        )
        all_stereoisomers.extend(stereoisomers)

    # Write output SMI file
    print(f"Writing {len(all_stereoisomers)} stereoisomers to {args.output_smi}...")
    write_smi_file(args.output_smi, all_stereoisomers)
    print("Stereoisomer generation completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
