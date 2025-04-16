from rdkit import Chem
import numpy as np
import argparse

def create_autobox(sdf_file, output_file, margin=3.0):
    # Read the SDF file
    supplier = Chem.SDMolSupplier(sdf_file)
    
    # Lists to store all coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    # Iterate through all molecules in the SDF
    for mol in supplier:
        if mol is not None:  # Check if molecule was parsed successfully
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                x_coords.append(pos.x)
                y_coords.append(pos.y)
                z_coords.append(pos.z)
    
    if not x_coords:  # Check if we have any coordinates
        raise ValueError("No valid molecules found in the SDF file")
    
    # Calculate min and max for each dimension
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    min_z, max_z = min(z_coords), max(z_coords)
    
    # Calculate center
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    # Calculate size with margin
    size_x = max_x - min_x + 2 * margin
    size_y = max_y - min_y + 2 * margin
    size_z = max_z - min_z + 2 * margin
    
    # Format the output
    output_content = f"center_x = {center_x:.3f}\n"
    output_content += f"center_y = {center_y:.3f}\n"
    output_content += f"center_z = {center_z:.3f}\n"
    output_content += f"size_x = {size_x:.3f}\n"
    output_content += f"size_y = {size_y:.3f}\n"
    output_content += f"size_z = {size_z:.3f}\n"
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(output_content)
    
    print(f"Autobox parameters saved to {output_file}")
    print(output_content)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create autobox parameters from SDF file')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input SDF file containing molecules')
    parser.add_argument('--output', type=str, default='autobox_.txt',
                       help='Output text file for autobox parameters (default: autobox_.txt)')
    parser.add_argument('--margin', type=float, default=3.0,
                       help='Margin size in Angstroms to add to each dimension (default: 5.0)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        create_autobox(args.input, args.output, args.margin)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
