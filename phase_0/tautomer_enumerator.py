"""
Tautomer/protomer generator using OpenEye's OETautomerOptions and OEGetReasonableTautomers.
Enumerates tautomers and protomers from an input SMI file and writes them to an output SMI file.

Example usage (one-liner):

python tautomer_enumerator.py --input_smi smiles_isomers/smiles_all_00_isomers.smi --output_smi tautomers2.smi --max_tautomers 10 --pka_norm --max_search_time 30.0

"""

import os
import argparse
from openeye import oechem
from openeye import oequacpac
from typing import List, Tuple
from tqdm import tqdm
import sys

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate tautomers/protomers from an SMI file using OpenEye.")
    parser.add_argument(
        "--input_smi",
        type=str,
        required=True,
        help="Path to input SMI file containing SMILES and IDs (format: SMILES ID)."
    )
    parser.add_argument(
        "--output_smi",
        type=str,
        default="tautomers.smi",
        help="Path to output SMI file for enumerated tautomers/protomers (default: tautomers.smi)."
    )
    parser.add_argument(
        "--max_tautomers",
        type=int,
        default=16,
        help="Maximum number of tautomers/protomers to generate per molecule (default: 16)."
    )
    parser.add_argument(
        "--pka_norm",
        action="store_true",
        help=(
            "Enable pKa normalization to favor chemically reasonable tautomers/protomers. "
            "May include protonation state changes (protomeric forms)."
        )
    )
    parser.add_argument(
        "--max_search_time",
        type=float,
        default=60.0,
        help="Maximum search time (in seconds) per molecule (default: 60.0)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all status messages and warnings, including OpenEye's internal warnings."
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

def enumerate_tautomers(
    smi: str, idx: str, max_tautomers: int, pka_norm: bool, max_search_time: float, quiet: bool
) -> List[Tuple[str, str]]:
    """Enumerate tautomers/protomers for a single SMILES using OEGetReasonableTautomers."""
    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smi):
        if not quiet:
            print(f"Failed to parse SMILES: {smi} ({idx})")
        return [(smi, idx)]  # Return original if parsing fails

    # Set up tautomer options
    tautomer_opts = oequacpac.OETautomerOptions()
    tautomer_opts.SetMaxTautomersGenerated(max_tautomers)
    tautomer_opts.SetMaxSearchTime(max_search_time)
    tautomer_opts.SetCarbonHybridization(True)
    tautomer_opts.SetLevel(5)  # Default strictness level
    tautomer_opts.SetSaveStereo(True)

    # Suppress OpenEye warnings in quiet mode
    if quiet:
        null_stream = oechem.oeostream()
        null_stream.open(os.devnull)  # Redirect to /dev/null (Unix) or nul (Windows)
        oechem.OEThrow.SetOutputStream(null_stream)

    tautomers = []
    for i, tautomer in enumerate(oequacpac.OEGetReasonableTautomers(mol, tautomer_opts, pka_norm)):
        new_smi = oechem.OEMolToSmiles(tautomer)
        new_idx = f"{idx}_taut{i}"
        tautomers.append((new_smi, new_idx))
    
    # Restore default output stream after enumeration
    if quiet:
        oechem.OEThrow.SetOutputStream(oechem.oeout)  # Reset to stdout

    if not tautomers:  # If no tautomers/protomers generated, return original
        if not quiet:
            print(f"No tautomers/protomers generated for {smi} ({idx}), using original.")
        tautomers.append((smi, idx))
    
    return tautomers

def write_smi_file(output_smi: str, tautomers: List[Tuple[str, str]], quiet: bool = False):
    """Write enumerated tautomers/protomers to an output SMI file."""
    with open(output_smi, 'w') as f:
        for smi, idx in tautomers:
            f.write(f"{smi} {idx}\n")
    if not quiet:
        print(f"Wrote {len(tautomers)} tautomers/protomers to {output_smi}")

def main():
    """Main function to run the tautomer/protomer generator."""
    args = parse_arguments()

    # Verify OpenEye license
    if "OE_LICENSE" not in os.environ:
        raise EnvironmentError("OpenEye license (OE_LICENSE) not detected. Please set up the license to use this tool.")

    # Read input SMI file
    if not args.quiet:
        print(f"Reading input SMILES from {args.input_smi}...")
    smiles_data = read_smi_file(args.input_smi, args.quiet)

    # Enumerate tautomers
    all_tautomers = []
    for smi, idx in tqdm(smiles_data, desc="Enumerating tautomers/protomers", disable=args.quiet):
        tautomers = enumerate_tautomers(
            smi, idx, args.max_tautomers, args.pka_norm, args.max_search_time, args.quiet
        )
        all_tautomers.extend(tautomers)

    # Write output SMI file
    write_smi_file(args.output_smi, all_tautomers, args.quiet)
    if not args.quiet:
        print("Tautomer/protomer generation completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Even in quiet mode, fatal errors should be reported
        print(f"Error: {str(e)}", file=sys.stderr)
        exit(1)
