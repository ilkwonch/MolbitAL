import argparse
import os
import sys
import subprocess
import csv
import re
import glob
from pathlib import Path
import time
from tqdm import tqdm
from extract_scores import extract_vina_score, extract_ligand_id, process_docking_results, process_unidock_results


def parse_smiles_file(smiles_file):
    smiles_data = {}
    line_to_molecule_id = {}  # Maps 0-indexed line numbers to molecule IDs
    
    try:
        with open(smiles_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if parts:
                    # First column is SMILES
                    smiles = parts[0]
                    # Second column is molecule ID if available, otherwise use line number as ID
                    molecule_id = parts[1] if len(parts) > 1 else f"mol_{i+1}"
                    smiles_data[i] = (smiles, molecule_id)
                    line_to_molecule_id[i] = molecule_id 
        return smiles_data, line_to_molecule_id
    except Exception as e:
        print(f"Error parsing SMILES file {smiles_file}: {e}")
        return {}, {}


def count_ligands(pattern):
    """Count the number of ligands matching the pattern."""
    return len(glob.glob(pattern))


def run_command(cmd):
    """Run a shell command and return True if successful."""
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def initialize_scores_csv(smiles_file, csv_path):
    smiles_data, line_to_molecule_id = parse_smiles_file(smiles_file)
    if not smiles_data:
        print(f"Error: Could not parse SMILES file {smiles_file}")
        return False, {}
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['molecule_id', 'smiles', 'docking'])
            
            for i, (smiles, molecule_id) in sorted(smiles_data.items()):
                # Write the row with NaN for docking score
                writer.writerow([molecule_id, smiles, 'NaN'])
        
        print(f"Initialized {csv_path} with {len(smiles_data)} molecules")
        return True, line_to_molecule_id
    except Exception as e:
        print(f"Error writing CSV file {csv_path}: {e}")
        return False, {}


def merge_previous_scores(current_csv, prev_csv, output_csv):
    """
    Merge docking scores from the previous cycle's CSV into the current CSV based on molecule_id.
    
    Args:
        current_csv: Path to the current cycle's scores CSV
        prev_csv: Path to the previous cycle's scores CSV
        output_csv: Path to save the merged CSV
    """
    try:
        # Read the current CSV
        with open(current_csv, 'r', newline='') as f:
            reader = csv.reader(f)
            current_rows = list(reader)
        
        if not current_rows:
            print(f"Current CSV file {current_csv} is empty.")
            return False
        
        header = current_rows[0]
        try:
            docking_idx = header.index('docking')
            molecule_id_idx = header.index('molecule_id')
        except ValueError:
            print("Current CSV header does not contain required columns 'molecule_id' and 'docking'.")
            return False
        
        # Read the previous CSV
        prev_scores = {}
        if os.path.exists(prev_csv):
            with open(prev_csv, 'r', newline='') as f:
                reader = csv.reader(f)
                prev_rows = list(reader)
                if prev_rows:
                    prev_header = prev_rows[0]
                    try:
                        prev_docking_idx = prev_header.index('docking')
                        prev_molecule_id_idx = prev_header.index('molecule_id')
                        for row in prev_rows[1:]:
                            if len(row) > max(prev_molecule_id_idx, prev_docking_idx):
                                molecule_id = row[prev_molecule_id_idx]
                                docking_score = row[prev_docking_idx]
                                if docking_score.lower() != 'nan' and docking_score:
                                    prev_scores[molecule_id] = docking_score
                    except ValueError:
                        print("Previous CSV header does not contain required columns 'molecule_id' and 'docking'.")
                        return False
        else:
            print(f"Previous CSV file {prev_csv} does not exist. Proceeding without merging.")
        
        # Update current rows with previous scores
        updated_rows = [header]
        matches = 0
        for row in current_rows[1:]:
            if len(row) > max(molecule_id_idx, docking_idx):
                molecule_id = row[molecule_id_idx]
                if molecule_id in prev_scores and row[docking_idx].lower() == 'nan':
                    row[docking_idx] = prev_scores[molecule_id]
                    matches += 1
                updated_rows.append(row)
            else:
                updated_rows.append(row)
        
        # Write the merged CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)
        
        print(f"Merged scores from previous cycle: {matches} docking scores updated in {output_csv}")
        return True
    except Exception as e:
        print(f"Error merging scores from {prev_csv} into {current_csv}: {e}")
        return False


def update_scores_csv_in_place(scores, csv_path):
    """
    Update an existing CSV file with docking scores based on the ligand id.
    
    It assumes the CSV file has header ['molecule_id', 'smiles', 'docking'].
    For each score in scores, the function tries to match the ligand_id with molecule_id
    in the CSV and updates the corresponding docking value.
    
    Args:
        scores: List of tuples (ligand_id, score)
        csv_path: Path to the CSV file to update
    """
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            print(f"CSV file {csv_path} is empty.")
            return False
        header = rows[0]
        try:
            docking_idx = header.index('docking')
            molecule_id_idx = header.index('molecule_id')
        except ValueError:
            print("CSV header does not contain required columns 'molecule_id' and 'docking'.")
            return False
        
        # Data rows start from index 1
        data_rows = rows[1:]
        
        # Create a mapping of molecule_id values to row indices
        id_to_row_idx = {}
        for i, row in enumerate(data_rows):
            id_to_row_idx[row[molecule_id_idx]] = i
            
            # Also map numeric part if molecule_id contains numbers
            if any(c.isdigit() for c in row[molecule_id_idx]):
                # Extract numeric part from molecule_id
                match = re.search(r'(\d+)', row[molecule_id_idx])
                if match:
                    num_id = match.group(1)
                    id_to_row_idx[num_id] = i
        
        # Track matches and misses
        matches = 0
        misses = 0
        
        # Update docking value for each score
        for ligand_id, score in scores:
            try:
                # First try direct match
                if ligand_id in id_to_row_idx:
                    row_idx = id_to_row_idx[ligand_id]
                    data_rows[row_idx][docking_idx] = str(score)
                    matches += 1
                    continue
                
                # If direct match fails, try extracting numeric part from ligand_id
                if isinstance(ligand_id, str):
                    match = re.search(r'(\d+)', ligand_id)
                    if match:
                        # Extract just the numeric part
                        num_id = match.group(1)
                        if num_id in id_to_row_idx:
                            row_idx = id_to_row_idx[num_id]
                            data_rows[row_idx][docking_idx] = str(score)
                            matches += 1
                            continue
                        
                        # Try standard "mol_X" format
                        mol_id = f"mol_{num_id}"
                        if mol_id in id_to_row_idx:
                            row_idx = id_to_row_idx[mol_id]
                            data_rows[row_idx][docking_idx] = str(score)
                            matches += 1
                            continue
                
                # If all matching attempts failed
                misses += 1
                print(f"Warning: Could not find matching molecule_id for ligand id {ligand_id} in CSV")
            
            except Exception as e:
                print(f"Error processing ligand id {ligand_id}: {e}")
                misses += 1
                continue
        
        # Write back the updated CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        
        print(f"Updated docking scores in {csv_path}: {matches} matches, {misses} misses")
        return True
    except Exception as e:
        print(f"Error updating CSV file {csv_path}: {e}")
        return False


def create_ligand_index_file(ligand_files, ligand_dir):
    """
    Create a ligand index file for Unidock in the same directory as the ligand files.
    
    Args:
        ligand_files: List of ligand file paths
        ligand_dir: Directory where ligands are located and where to save the index file
    
    Returns:
        Path to the created index file
    """
    index_path = os.path.join(ligand_dir, "index.txt")
    
    try:
        with open(index_path, 'w') as f:
            # Write absolute paths to each ligand file, one per line
            for ligand_file in ligand_files:
                abs_path = os.path.abspath(ligand_file)
                f.write(f"{abs_path}\n")
        
        print(f"Created ligand index file at {index_path} with {len(ligand_files)} ligands")
        return index_path
    except Exception as e:
        print(f"Error creating ligand index file: {e}")
        return None


def run_unidock(engine, receptor, config, ligand_dir, ligand_files, output_dir, extra_args=None):
    """
    Run Unidock with the appropriate options.
    
    Args:
        engine: Should be "unidock"
        receptor: Path to receptor file
        config: Path to configuration file
        ligand_dir: Directory containing ligand files
        ligand_files: List of ligand files to process
        output_dir: Directory for output files
        extra_args: Additional arguments for the docking engine
    
    Returns:
        True if successful, False otherwise
    """
    if engine != "unidock":
        print(f"Error: Expected engine 'unidock', got '{engine}'")
        return False
    
    index_file = create_ligand_index_file(ligand_files, ligand_dir)
    if not index_file:
        return False
    
    cmd = [
        "unidock",
        "--receptor", receptor,
        "--ligand_index", index_file,
        "--dir", output_dir
    ]
    
    if config:
        cmd.extend(["--config", config])
    
    extra_args_str = " ".join(extra_args) if extra_args else ""
    
    if "--search_mode" not in extra_args_str:
        cmd.extend(["--search_mode", "balance"])
    
    if "--scoring" not in extra_args_str:
        cmd.extend(["--scoring", "vina"])
    
    if "--num_modes" not in extra_args_str:
        cmd.extend(["--num_modes", "1"])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running Unidock with {len(ligand_files)} ligands...")
    return run_command(cmd)


def extract_scores_and_update_csv(output_dir, engine, results_csv, line_to_molecule_id=None):
    """
    Extract docking scores from output files and update the CSV with the results.
    
    Args:
        output_dir: Directory containing the docking results
        engine: Docking engine used
        results_csv: Path to output CSV file
        line_to_molecule_id: Dictionary mapping from line numbers (0-indexed) to molecule IDs
    
    Returns:
        A list of tuples (ligand_id, score) with the extracted scores
    """
    print("Extracting docking scores...")
    all_scores = []
    
    try:
        output_dir_path = Path(output_dir)
        
        if engine == "unidock":
            # For unidock, we'll first try to find output PDBQT files in the output directory
            result_files = sorted(output_dir_path.glob("*.pdbqt"))
            
            if not result_files:
                # If no PDBQT files found, try to check other output formats
                print("No PDBQT output files found for unidock, checking other formats...")
                unidock_scores_file = output_dir_path / "scores.csv"
                
                if unidock_scores_file.exists():
                    # Parse unidock scores.csv if it exists
                    with open(unidock_scores_file, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)  # Skip header
                        for row in reader:
                            if len(row) >= 2:
                                ligand_path = row[0]
                                # Extract ligand ID from filename
                                match = re.search(r'ligand_(\d+)', Path(ligand_path).name)
                                if match:
                                    ligand_id = match.group(1)
                                    score = float(row[1])
                                    all_scores.append((ligand_id, score))
                else:
                    # Process all unidock output files using the existing function
                    temp_csv = os.path.join(output_dir, "temp_scores.csv")
                    process_unidock_results(str(output_dir), temp_csv)
                    
                    if os.path.exists(temp_csv):
                        with open(temp_csv, 'r', newline='') as f:
                            reader = csv.reader(f)
                            next(reader)  # Skip header
                            for row in reader:
                                if len(row) >= 2:
                                    ligand_id = row[0]
                                    score = float(row[1])
                                    all_scores.append((ligand_id, score))
                        os.remove(temp_csv)  # Clean up temp file
            else:
                # Process PDBQT files
                print(f"Processing {len(result_files)} unidock output files...")
                for result_file in tqdm(result_files, desc="Extracting scores"):
                    # Extract ligand ID from filename
                    match = re.search(r'ligand_(\d+)', result_file.stem)
                    if match:
                        ligand_id = match.group(1)
                        score = extract_vina_score(result_file)
                        if score is not None:
                            all_scores.append((ligand_id, score))
        else:
            # For other docking engines
            result_files = sorted(output_dir_path.glob("*.pdbqt"))
            print(f"Processing {len(result_files)} output files...")
            
            for result_file in tqdm(result_files, desc="Extracting scores"):
                match = re.search(r'ligand_(\d+)', result_file.stem)
                if match:
                    ligand_id = match.group(1)
                    score = extract_vina_score(result_file)
                    if score is not None:
                        all_scores.append((ligand_id, score))
        
        # Check for and add any ligands from index.txt that might have been used
        index_file = os.path.join(os.path.dirname(output_dir), "index.txt")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    for line in f:
                        ligand_path = line.strip()
                        if ligand_path:
                            # Extract ligand ID from the path
                            match = re.search(r'ligand_(\d+)', os.path.basename(ligand_path))
                            if match:
                                ligand_id = match.group(1)
                                # Check if we already have a score for this ligand
                                if not any(lid == ligand_id for lid, _ in all_scores):
                                    # Try to find score for this ligand
                                    result_path = os.path.join(output_dir, f"ligand_{ligand_id}_out.pdbqt")
                                    if os.path.exists(result_path):
                                        score = extract_vina_score(result_path)
                                        if score is not None:
                                            all_scores.append((ligand_id, score))
            except Exception as e:
                print(f"Warning: Error processing index.txt: {e}")
        
        # Convert numeric ligand IDs to molecule IDs using the mapping
        if line_to_molecule_id:
            mapped_scores = []
            for ligand_id, score in all_scores:
                try:
                    # Try to convert the ligand_id to an integer for mapping
                    line_num = int(ligand_id)
                    if line_num in line_to_molecule_id:
                        molecule_id = line_to_molecule_id[line_num]
                        mapped_scores.append((molecule_id, score))
                    else:
                        # If no mapping found, keep original ligand_id
                        print(f"Warning: No molecule ID mapping found for ligand ID {ligand_id}")
                        mapped_scores.append((ligand_id, score))
                except ValueError:
                    # If ligand_id is not an integer, keep it as is
                    mapped_scores.append((ligand_id, score))
            all_scores = mapped_scores
        
        # Sort by numeric ligand ID if possible
        try:
            all_scores.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
        except (ValueError, TypeError):
            # If sorting by integer fails, sort as strings
            all_scores.sort(key=lambda x: str(x[0]))
    except Exception as e:
        print(f"Error extracting scores: {e}")
        return []
    
    if all_scores:
        update_scores_csv_in_place(all_scores, results_csv)
    else:
        print("No scores were extracted.")
    
    return all_scores


def main():
    parser = argparse.ArgumentParser(
        description="""Run molecular docking and extract scores.

    - needs to initialize docking command first by installing setup.py
    - prepare pdbqt format for receptor and ligands, and specifiy box.txt to assign binding pockets (use --config or explicitly assign box_size and box_center)
    - Autodock-Vina and other vina variant's software (e.g. qvina,smina, unidock)
    - Extracting docking scores and its output (pdbqt format)

    Typical usage:
    python run_docking.py --engine qvina2.1 --receptor receptor.pdbqt --ligands-dir ligands/ --config box.txt --smiles_file al_{num}/train_smiles.smi --output-dir results
    
    if you use unidock engine, you need to specifiy ligand-index or gpu-batch to use batch mode
    python run_docking.py --engine unidock --receptor receptor.pdbqt --ligand-index ligands/ --config box.txt --smiles_file al_{num}/train_smiles.smi --output-dir results --extra-args --max_gpu_memory {gpu memory} 

    """,
    formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--engine", required=True, 
                        choices=["vina_gpu2.1", "qvina_w", "qvina2.1", "smina", "vina", "unidock"],
                        help="Docking engine to use")
    parser.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDBQT)")
    parser.add_argument("--config", help="Path to configuration file (e.g., autobox.txt)")
    
    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument("--ligands-dir", help="Directory containing prepared ligand files")
    ligand_group.add_argument("--ligand-pattern", help="Glob pattern to match ligand files (e.g., 'ligands/*.pdbqt')")
    ligand_group.add_argument("--ligand-index", help="Directory path. For unidock, index.txt will be created in this directory")
    ligand_group.add_argument("--gpu-batch", help="Path to a single ligand for GPU batch (Unidock)")
    ligand_group.add_argument("--batch", help="Path to a single ligand for batch processing (Unidock)")
    
    parser.add_argument("--smiles-file", help="Path to the original SMILES file with molecule IDs")
    
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--results-csv", required=True, help="Path to output CSV file with docking scores")
    
    parser.add_argument("--active", action="store_true", help="Enable active learning mode to merge scores from previous cycle")
    
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional arguments for the docking engine")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results_csv = args.results_csv
    if not os.path.isabs(results_csv):
        results_csv = os.path.join(args.output_dir, results_csv)
    
    line_to_molecule_id = {}
    if args.smiles_file:
        print(f"Parsing SMILES file: {args.smiles_file}")
        success, line_to_molecule_id = initialize_scores_csv(args.smiles_file, results_csv)
        if not success:
            print("Failed to initialize scores CSV.")
            sys.exit(1)
    
    ligand_files = []
    ligand_dir = None
    
    if args.engine == "unidock" and (args.gpu_batch or args.batch):
        if args.gpu_batch:
            source_file = args.gpu_batch
            batch_type = "--gpu-batch"
        else:
            source_file = args.batch
            batch_type = "--batch"
        
        source_dir = os.path.dirname(source_file)
        if not source_dir:
            source_dir = "."
        
        ligand_dir = source_dir
        pattern = os.path.join(source_dir, "*.pdbqt")
        ligand_files = sorted(glob.glob(pattern))
        print(f"Using {batch_type} with {len(ligand_files)} ligands from {source_dir}")
    
    elif args.engine == "unidock" and args.ligand_index:
        ligand_dir = args.ligand_index
        if not os.path.isdir(ligand_dir):
            print(f"Error: --ligand-index for Unidock should be a directory, got: {ligand_dir}")
            sys.exit(1)
        
        pattern = os.path.join(ligand_dir, "*.pdbqt")
        ligand_files = sorted(glob.glob(pattern))
        print(f"Using --ligand-index with {len(ligand_files)} ligands from {ligand_dir}")
    
    elif args.ligands_dir:
        ligand_dir = args.ligands_dir
        ligand_pattern = str(Path(args.ligands_dir) / "*.pdbqt")
        ligand_files = sorted(glob.glob(ligand_pattern))
    elif args.ligand_pattern:
        ligand_pattern = args.ligand_pattern
        ligand_dir = os.path.dirname(ligand_pattern)
        if not ligand_dir:
            ligand_dir = "."
        ligand_files = sorted(glob.glob(ligand_pattern))
    elif args.ligand_index and args.engine != "unidock":
        try:
            with open(args.ligand_index, 'r') as f:
                ligand_files = sorted([line.strip() for line in f if line.strip()])
            if ligand_files:
                ligand_dir = os.path.dirname(ligand_files[0])
                if not ligand_dir:
                    ligand_dir = "."
        except Exception as e:
            print(f"Error reading ligand index file {args.ligand_index}: {e}")
            sys.exit(1)
    
    total_ligands = len(ligand_files)
    if total_ligands == 0:
        print("No ligand files found to process.")
        sys.exit(1)
    print(f"Found {total_ligands} ligand files to process")
    
    start_time = time.time()
    print("Starting docking process...")
    
    docking_success = False
    
    if args.engine == "unidock":
        docking_success = run_unidock(
            engine=args.engine,
            receptor=args.receptor,
            config=args.config,
            ligand_dir=ligand_dir,
            ligand_files=ligand_files,
            output_dir=str(args.output_dir),
            extra_args=args.extra_args
        )
    else:
        docking_success = True
        failed_ligands = []
        
        for i, ligand_file in enumerate(tqdm(ligand_files, desc="Docking ligands", unit="ligand")):
            ligand_basename = os.path.basename(ligand_file)
            output_file = os.path.join(args.output_dir, f"{os.path.splitext(ligand_basename)[0]}_out.pdbqt")
            
            docking_cmd = [
                "docking",
                "--engine", args.engine,
                "--receptor", args.receptor,
                "--ligand", ligand_file,
                "--output", output_file
            ]
            
            if args.config:
                docking_cmd.extend(["--config", args.config])
            
            if args.extra_args:
                docking_cmd.extend(["--extra-args"] + args.extra_args)
            
            if not run_command(docking_cmd):
                print(f"Docking failed for ligand: {ligand_file}")
                docking_success = False
                failed_ligands.append(ligand_file)
            
            if (i+1) % 5 == 0 or i+1 == total_ligands:
                print(f"Progress: {i+1}/{total_ligands} ligands completed")
        
        if failed_ligands:
            print(f"Docking failed for {len(failed_ligands)} ligands: {failed_ligands}")
    
    if not docking_success:
        print("Docking failed. Check the error messages above.")
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    print(f"Docking completed in {elapsed_time:.2f} seconds")
    
    scores = extract_scores_and_update_csv(
        output_dir=args.output_dir,
        engine=args.engine,
        results_csv=results_csv,
        line_to_molecule_id=line_to_molecule_id
    )
    
    if not scores:
        print("No docking scores were found.")
        sys.exit(1)
    
    if args.active:
        current_cycle_dir = os.path.dirname(args.output_dir)
        match = re.search(r'al_(\d+)', current_cycle_dir)
        if match:
            current_cycle_num = int(match.group(1))
            if current_cycle_num > 1:
                prev_cycle_dir = os.path.join(os.path.dirname(current_cycle_dir), f"al_{current_cycle_num-1}")
                prev_scores_csv = os.path.join(prev_cycle_dir, "train_docking", "scores.csv")
                print(f"Merging scores from previous cycle: {prev_scores_csv}")
                merge_previous_scores(results_csv, prev_scores_csv, results_csv)
            else:
                print("No previous cycle to merge from (current cycle is al_1).")
        else:
            print("Could not determine cycle number from output directory. Skipping merge.")
    
    total_elapsed = time.time() - start_time
    print(f"Entire process completed in {total_elapsed:.2f} seconds")
    print(f"Docking scores saved to {results_csv}")
    print(f"Processed {len(scores)} ligands successfully")


if __name__ == "__main__":
    main()
