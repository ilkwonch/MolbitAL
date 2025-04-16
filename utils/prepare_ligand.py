import argparse
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import meeko


def prepare_ligand(input_file, output_dir, format="smi"):
    """Prepares ligands from input file and saves them as PDBQT."""
    supported_formats = ["smi", "sdf", "mol2"]
    if format not in supported_formats:
        raise ValueError(f"Unsupported file format: {format}. Supported formats are {', '.join(supported_formats)}")

    meeko_prep = meeko.MoleculePreparation()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if format == "smi":
        with open(input_file, 'r') as file:
            molecules = file.readlines()
    elif format == "sdf":
        molecules = Chem.SDMolSupplier(input_file)
    elif format == "mol2":
        molecules = [Chem.MolFromMol2File(input_file)]

    for idx, mol_data in enumerate(molecules):
        try:
            # Parse the molecule
            if format == "smi":
                smiles = mol_data.strip()
                lig = Chem.MolFromSmiles(smiles)
            else:
                lig = mol_data

            if not lig:
                print(f"Invalid molecule at index {idx}. Skipping.")
                continue

            # Add hydrogens
            lig_h = Chem.AddHs(lig)

            # Generate 3D coordinates
            success = AllChem.EmbedMolecule(lig_h, AllChem.ETKDGv3())
            if success == -1:
                print(f"3D embedding failed for molecule at index {idx}. Skipping.")
                continue

            # Prepare PDBQT with Meeko
            meeko_prep.prepare(lig_h)
            lig_pdbqt = meeko_prep.write_pdbqt_string()

            # Save as individual PDBQT
            ligand_name = f"ligand_{idx}.pdbqt"
            output_path = os.path.join(output_dir, ligand_name)
            with open(output_path, 'w') as out_file:
                out_file.write(lig_pdbqt)
            print(f"Saved PDBQT: {output_path}")
        except Exception as e:
            print(f"Error processing molecule at index {idx}: {str(e)}")

    print(f"All PDBQT files are saved in {output_dir}.")


def main():
    parser = argparse.ArgumentParser(
        description="Automate ligand preparation for docking (vina/smina/qvina).",
        epilog="Example usage:\n"
               "  python prep_ligand.py -i ligands.smi -o ligand_pdbqt --format smi\n"
               "  python prep_ligand.py --input ligands.sdf --output ligand_pdbqt --format sdf",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input file path (SMILES, SDF, or MOL2)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="ligand_pdbqt",
        help="Output directory where PDBQT files will be saved.\n"
             "Default is 'ligand_pdbqt/'."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["smi", "sdf", "mol2"],
        default="smi",
        help="Input file format. Choose one of: smi, sdf, mol2. Default is 'smi'."
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function to prepare ligands
    try:
        prepare_ligand(args.input, args.output, format=args.format)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()


