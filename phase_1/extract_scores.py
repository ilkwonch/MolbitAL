from tqdm import tqdm
import argparse
import os
import re
import csv
import glob
from pathlib import Path

def extract_vina_score(pdbqt_file):
    """
    Extract the docking score from a PDBQT file.
    Looks for the line formats:
    - REMARK VINA RESULT:    -9.653      0.000      0.000
    - REMARK minimizedAffinity -10.1285791
    Returns the VINA RESULT score if present, minimizedAffinity score for smina.
    """
    try:
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if "REMARK VINA RESULT:" in line:
                    match = re.search(r'RESULT:\s+([-\d.]+)', line)
                    if match:
                        return float(match.group(1))
                elif "REMARK minimizedAffinity" in line:
                    match = re.search(r'minimizedAffinity\s+([-\d.]+)', line)
                    if match:
                        return float(match.group(1))
        print(f"Warning: No docking score found in {pdbqt_file}")
        return None
    except Exception as e:
        print(f"Error reading {pdbqt_file}: {e}")
        return None

def extract_ligand_id(filename):
    match = re.search(r'ligand_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return Path(filename).stem

def process_docking_results(results_dir, output_csv, pattern="*.pdbqt", id_extractor=extract_ligand_id):
    """
    Process all docking result files in the specified directory and save scores to CSV.
    
    Args:
        results_dir: Directory containing the docking results
        output_csv: Path to the output CSV file
        pattern: Glob pattern to match result files
        id_extractor: Function to extract ligand ID from filename
    """
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob(pattern))
    
    if not result_files:
        print(f"No result files found in {results_dir} matching pattern '{pattern}'")
        return False
    
    scores = []
    
    try:
        result_iterator = tqdm(result_files, desc="Processing output files")
    except ImportError:
        result_iterator = result_files
        print(f"Processing {len(result_files)} output files...")
    
    for result_file in result_iterator:
        ligand_id = id_extractor(result_file.name)
        score = extract_vina_score(result_file)
        if score is not None:
            scores.append((ligand_id, score))
    
    try:
        scores.sort(key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0])
    except (ValueError, TypeError):
        scores.sort(key=lambda x: x[0])
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ligand_id', 'docking'])
        writer.writerows(scores)
    
    print(f"Extracted {len(scores)} scores and saved to {output_csv}")
    return True

def process_unidock_results(results_dir, output_csv):
    """
    Process UniDock results which may have a different file structure.
    UniDock may save scores in its own format or in PDBQT files.
    """
    pdbqt_success = process_docking_results(results_dir, output_csv)
    
    if not pdbqt_success:
        score_files = list(Path(results_dir).glob("*.score")) + list(Path(results_dir).glob("*_scores.txt"))
        
        if score_files:
            score_file = score_files[0]
            scores = []
            
            with open(score_file, 'r') as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            ligand_name = parts[0]
                            score = float(parts[1])
                            
                            ligand_id = extract_ligand_id(ligand_name)
                            scores.append((ligand_id, score))
                        except (ValueError, IndexError):
                            print(f"Could not parse line: {line}")
            
            if scores:
                try:
                    scores.sort(key=lambda x: int(x[0]) if isinstance(x[0], str) else x[0])
                except (ValueError, TypeError):
                    scores.sort(key=lambda x: x[0])
                
                with open(output_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ligand_id', 'docking'])
                    writer.writerows(scores)
                
                print(f"Extracted {len(scores)} scores from {score_file} and saved to {output_csv}")
                return True
            
            return False
        
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract docking scores from output files")
    parser.add_argument("--input-dir", required=True, help="Directory containing docking results")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV file")
    parser.add_argument("--engine", default="vina", 
                        choices=["vina_gpu2.1", "qvina_w", "qvina2.1", "smina", "vina", "unidock"],
                        help="Docking engine used (affects file parsing)")
    parser.add_argument("--pattern", default="*.pdbqt", 
                        help="Glob pattern to match result files")
    
    args = parser.parse_args()
    
    if args.engine == "unidock":
        process_unidock_results(args.input_dir, args.output_csv)
    else:
        process_docking_results(args.input_dir, args.output_csv, pattern=args.pattern)

if __name__ == "__main__":
    main()
