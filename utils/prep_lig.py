import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
import meeko
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="meeko")

def process_ligand(idx, mol_data, output_dir, fmt):
    try:
        meeko_prep = meeko.MoleculePreparation()
        if fmt == "smi":
            smiles = mol_data.strip()
            lig = Chem.MolFromSmiles(smiles)
        else:
            lig = mol_data
        if not lig:
            return f"Invalid molecule at index {idx}. Skipping."
        lig_h = Chem.AddHs(lig)
        success = AllChem.EmbedMolecule(lig_h, AllChem.ETKDGv3())
        if success == -1:
            return f"3D embedding failed for molecule at index {idx}. Skipping."
        meeko_prep.prepare(lig_h)
        lig_pdbqt = meeko_prep.write_pdbqt_string()
        ligand_name = f"ligand_{idx}.pdbqt"
        output_path = os.path.join(output_dir, ligand_name)
        with open(output_path, 'w') as out_file:
            out_file.write(lig_pdbqt)
        # Only return a message on error, otherwise return None
        return None
    except Exception as e:
        return f"Error processing molecule at index {idx}: {str(e)}"

def parallel_prepare_ligand(input_file, output_dir, fmt="smi", num_workers=4, end=None, silent=False):
    os.makedirs(output_dir, exist_ok=True)
    
    if fmt == "smi":
        with open(input_file, 'r') as file:
            if end is not None:
                molecules = [next(file).strip() for _ in range(min(end, sum(1 for _ in open(input_file))))]
            else:
                molecules = file.readlines()
    elif fmt == "sdf":
        molecules = list(Chem.SDMolSupplier(input_file))
        if end is not None:
            molecules = molecules[:end]
    elif fmt == "mol2":
        molecules = [Chem.MolFromMol2File(input_file)]
        if end is not None:
            molecules = molecules[:end]
    
    if not silent:
        print(f"Processing {len(molecules)} molecules from {input_file}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_ligand, idx, mol_data, output_dir, fmt): idx
            for idx, mol_data in enumerate(molecules)
        }
        for future in as_completed(futures):
            result = future.result()
            # Only print non-None results (errors)
            if result is not None and not silent:
                print(result)
    
    if not silent:
        print(f"All PDBQT files are saved in {output_dir}.")

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ligands in parallel for docking workflows.",
        epilog="Example usage:\n"
               "  python prep_lig.py -i ligands.smi -o ligand_pdbqt --format smi --workers 8 --end 1000",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Input file path (SMILES, SDF, or MOL2).")
    parser.add_argument("-o", "--output", type=str, default="ligand_pdbqt",
                        help="Output directory for PDBQT files.")
    parser.add_argument("--format", type=str, choices=["smi", "sdf", "mol2"], default="smi",
                        help="Input file format. Choices: smi, sdf, mol2.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes.")
    parser.add_argument("--end", type=int, default=None,
                        help="Number of molecule lines to process (1-based indexing).")
    parser.add_argument("--verbose", action="store_false", dest="silent", default=True,
                        help="Run in verbose mode (default is silent mode)")
    
    args = parser.parse_args()
    
    if args.end is not None and args.end < 1:
        print("Error: --end must be a positive integer")
        sys.exit(1)
    
    parallel_prepare_ligand(
        args.input,
        args.output,
        fmt=args.format,
        num_workers=args.workers,
        end=args.end,
        silent=args.silent
    )

if __name__ == "__main__":
    main()
