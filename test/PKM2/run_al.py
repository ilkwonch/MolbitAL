import json
import os
import subprocess
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run active learning pipeline for molecular docking.")
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")
    return parser.parse_args()

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} does not exist")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {config_path}")
        sys.exit(1)

    required_fields = [
        "project_name", "smiles_dir", "fingerprint_dir", "receptor", "config",
        "docking_engine", "ml_model", "num_compounds", "script_dir"
    ]
    missing_fields = [field for field in required_fields if field not in config or config[field] is None]
    if missing_fields:
        print(f"Error: Missing required configuration fields: {', '.join(missing_fields)}")
        sys.exit(1)

    config.setdefault("max_gpu_memory", 2)
    config.setdefault("workers", 8)
    config.setdefault("start_cycle", 1)
    config.setdefault("end_cycle", 5)
    config.setdefault("acquisition", "greedy")  

    return config

def run_command(command, step_name):
    print(f"Running {step_name}...")
    #print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {step_name}: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Exception in {step_name}: {str(e)}")
        sys.exit(1)

def read_fingerprints(fp_path, n_bits=1024):
    fingerprints = {}
    #print(f"Reading fingerprints from {fp_path}...")
    
    try:
        if os.path.isdir(fp_path):
            fp_files = [os.path.join(fp_path, f) for f in os.listdir(fp_path) 
                      if f.endswith('_fingerprints.txt') or f.endswith('.txt')]
        else:
            fp_files = [fp_path]
        
        total_fingerprints = 0
        
        for fp_file in fp_files:
            #print(f"Processing fingerprint file: {fp_file}")
            with open(fp_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    parts = line.strip().split(',')
                    if len(parts) < 1:
                        continue
                        
                    mol_id = parts[0].strip()
                    
                    fp_array = np.zeros(n_bits, dtype=np.int8)
                    
                    if len(parts) > 1:
                        try:
                            on_bits = [int(bit) for bit in parts[1:] if bit.strip().isdigit()]
                            valid_bits = [bit for bit in on_bits if 0 <= bit < n_bits]
                            if valid_bits:
                                fp_array[valid_bits] = 1
                        except Exception as e:
                            print(f"Warning: Could not parse fingerprint for {mol_id}: {e}")
                            continue
                    
                    fingerprints[mol_id] = fp_array
                    total_fingerprints += 1
        
    except Exception as e:
        print(f"Error reading fingerprints: {e}")
        sys.exit(1)
    
    return fingerprints

def create_next_cycle_data(cycle, project_dir, num_compounds, prev_cycle_dir, curr_cycle_dir, acquisition, fingerprint_dir=None):
    """
    Create train_smiles.smi and train_fingerprints.txt for the current cycle when resuming
    """
    prev_inference_dir = os.path.join(prev_cycle_dir, "train_inference")
    prev_predictions_csv = os.path.join(prev_inference_dir, f"predictions_{acquisition}_cycle_{cycle-1}.csv")
    prev_smiles_file = os.path.join(prev_cycle_dir, "train_smiles.smi")
    prev_fp_file = os.path.join(prev_cycle_dir, "train_fingerprints.txt")

    curr_smiles_path = os.path.join(curr_cycle_dir, "train_smiles.smi")
    curr_fp_path = os.path.join(curr_cycle_dir, "train_fingerprints.txt")

    if not os.path.exists(prev_predictions_csv):
        print(f"Error: Predictions file {prev_predictions_csv} does not exist")
        sys.exit(1)
    
    df = pd.read_csv(prev_predictions_csv)
    
    # Read ALL fingerprints from fingerprint directory if provided, otherwise use previous cycle fingerprints
    if fingerprint_dir and os.path.exists(fingerprint_dir):
        #print(f"Reading fingerprints from directory: {fingerprint_dir}")
        all_fingerprints = read_fingerprints(fingerprint_dir)
    else:
        #print(f"Reading fingerprints from previous cycle: {prev_fp_file}")
        all_fingerprints = read_fingerprints(prev_fp_file)
    
    prev_smiles_lines = []
    if os.path.exists(prev_smiles_file):
        with open(prev_smiles_file, 'r') as f:
            prev_smiles_lines = [line.strip() for line in f if line.strip()]
    
    selected_df = df[df["compound_label"] == 1].copy()
    selected_ids = selected_df["molecule_id"].tolist()
    selected_id_set = set(selected_ids)  # For faster lookups
    
    new_smiles = []
    for mol_id in selected_ids:
        row = df[df["molecule_id"] == mol_id]
        if not row.empty:
            smiles = row["smiles"].iloc[0]
            new_smiles.append(f"{smiles}\t{mol_id}")
    
    new_fingerprints = []
    for mol_id in selected_ids:
        if str(mol_id) in all_fingerprints:
            fp_array = all_fingerprints[str(mol_id)]
            on_bits = np.where(fp_array == 1)[0]
            fp_str = f"{mol_id},{','.join(map(str, on_bits))}"
            new_fingerprints.append(fp_str)
    
    prev_mol_ids = set()
    for line in prev_smiles_lines:
        parts = line.split('\t')
        if len(parts) > 1:
            prev_mol_ids.add(parts[1])
    
    all_smiles = new_smiles + prev_smiles_lines
    
    all_fp_lines = new_fingerprints.copy()
    
    if os.path.exists(prev_fp_file):
        with open(prev_fp_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',', 1)
                    if parts and parts[0] not in selected_id_set:
                        all_fp_lines.append(line.strip())
    
    os.makedirs(curr_cycle_dir, exist_ok=True)
    
    with open(curr_smiles_path, 'w') as f:
        f.write("\n".join(all_smiles) + "\n")
    print(f"Saved {len(all_smiles)} SMILES to {curr_smiles_path}")
    
    with open(curr_fp_path, 'w') as f:
        f.write("\n".join(all_fp_lines) + "\n")
    print(f"Saved {len(all_fp_lines)} fingerprints to {curr_fp_path}")
    
    print(f"\nCycle {cycle} training data breakdown:")
    print(f"  Newly selected compounds: {len(new_smiles)}")
    print(f"  Previous training compounds: {len(prev_smiles_lines)}")
    print(f"  Total compounds: {len(all_smiles)}")
    print(f"  Total fingerprints: {len(all_fp_lines)}")

def main():
    args = parse_arguments()
    config = load_config(args.config)

    project_name = config["project_name"]
    smiles_dir = config["smiles_dir"]
    fingerprint_dir = config["fingerprint_dir"]
    receptor = config["receptor"]
    config_file = config["config"]
    docking_engine = config["docking_engine"]
    ml_model = config["ml_model"]
    num_compounds = config["num_compounds"]
    start_cycle = config["start_cycle"]
    end_cycle = config["end_cycle"]
    script_dir = config["script_dir"]
    max_gpu_memory = config["max_gpu_memory"]
    workers = config["workers"]
    acquisition = config["acquisition"]

    project_dir = os.path.join(os.getcwd(), project_name)
    os.makedirs(project_dir, exist_ok=True)
    print(f"Project directory: {project_dir}")

    if start_cycle > 1:
        prev_cycle_dir = os.path.join(project_dir, f"al_{start_cycle-1}")
        if not os.path.exists(prev_cycle_dir):
            print(f"Error: Previous cycle directory {prev_cycle_dir} does not exist")
            sys.exit(1)
        curr_cycle_dir = os.path.join(project_dir, f"al_{start_cycle}")
        print(f"Resuming from cycle {start_cycle}, generating training data...")
        create_next_cycle_data(start_cycle, project_dir, num_compounds, prev_cycle_dir, curr_cycle_dir, acquisition, fingerprint_dir)

    else:
        print("Running initialization...")
        first_cycle_dir = os.path.join(project_dir, "al_1")
        os.makedirs(first_cycle_dir, exist_ok=True)
        collect_script = os.path.join(script_dir, "utils", "collect.py")
        run_command([
            sys.executable, collect_script,
            "--smiles_dir", smiles_dir,
            "--fingerprint_dir", fingerprint_dir,
            "--train_dir", first_cycle_dir,
            "--num_compounds", str(num_compounds)
        ], "data collection")

    for cycle in range(start_cycle, end_cycle + 1):
        print(f"Starting active learning cycle {cycle}...")

        cycle_dir = os.path.join(project_dir, f"al_{cycle}")
        os.makedirs(cycle_dir, exist_ok=True)

        smiles_file = os.path.join(cycle_dir, "train_smiles.smi")
        pdbqt_dir = os.path.join(cycle_dir, "train_pdbqt")
        docking_dir = os.path.join(cycle_dir, "train_docking")
        scores_csv = os.path.join(docking_dir, "scores.csv")
        ml_dir = os.path.join(cycle_dir, "train_ml")
        inference_dir = os.path.join(cycle_dir, "train_inference")
        fp_file = os.path.join(cycle_dir, "train_fingerprints.txt")
        
        # Step 1: lig prep
        prep_lig_script = os.path.join(script_dir, "utils", "prep_lig.py")
        print(f"Preparing ligands for cycle {cycle}...")
        run_command([
            sys.executable, prep_lig_script,
            "--input", smiles_file,
            "--output", pdbqt_dir,
            "--format", "smi",
            "--workers", str(workers),
            "--end", str(num_compounds),
            "--verbose"  
        ], "ligand preparation")

        # Step 2: Run docking
        print(f"Running docking for cycle {cycle}...")
        os.makedirs(docking_dir, exist_ok=True)
        docking_script = os.path.join(script_dir, "phase_1", "run_docking.py")
        docking_cmd = [
            sys.executable, docking_script,
            "--engine", docking_engine,
            "--receptor", receptor,
            "--config", config_file,
            "--smiles-file", smiles_file,
            "--output-dir", docking_dir,
            "--results-csv", "scores.csv"
        ]
        if start_cycle < cycle <= end_cycle:
            docking_cmd.append("--active")
        if docking_engine == "unidock":
            docking_cmd.extend(["--ligand-index", f"{pdbqt_dir}/"])
            docking_cmd.append("--extra-args")
            docking_cmd.append(f"--max_gpu_memory")
            docking_cmd.append(f"{max_gpu_memory}")
        else:
            docking_cmd.extend(["--ligands-dir", f"{pdbqt_dir}/"])
        run_command(docking_cmd, "docking")

        # Step 3: Train ML model
        print(f"Training ML model for cycle {cycle}...")
        train_ml_script = os.path.join(script_dir, "phase_2", "train_ml.py")
        run_command([
            sys.executable, train_ml_script,
            "--input-csv", scores_csv,
            "--input-fp", fp_file,
            "--output-dir", ml_dir,
            "--save-predictions",
            "--model", ml_model
        ], "ML training")

        # Step 4: Run inference
        print(f"Running inference for cycle {cycle}...")
        inference_script = os.path.join(script_dir, "phase_2", "inference.py")
        model_path = os.path.join(ml_dir, f"ml_models_{ml_model}", f"{ml_model}_model.joblib")
        scaler_path = os.path.join(ml_dir, f"ml_models_{ml_model}", "scaler.joblib")
        inference_cmd = [
            sys.executable, inference_script,
            "--model", model_path,
            "--scaler", scaler_path,
            "--input-smi-dir", smiles_dir,
            "--input-fp-dir", fingerprint_dir,
            "--output-dir", inference_dir,
            "--cycle", str(cycle),
            "--k", "2.0",
            "--top-compounds", str(num_compounds),
            "--train-smi", smiles_file,
            "--train-fp", fp_file,
            "--project-dir", project_dir,
            "--acquisition", acquisition
        ]
        if cycle < end_cycle:
            inference_cmd.append("--active")
        run_command(inference_cmd, "inference")

        print(f"Completed cycle {cycle}")

    print("Active learning process completed!")
    print(f"Results saved in {project_dir}")

if __name__ == "__main__":
    main()
