import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from tqdm import tqdm
from glob import glob
import shutil
from sklearn.ensemble import RandomForestRegressor
from xgboost_distribution import XGBDistribution

def read_fingerprints(fp_path, n_bits=1024):
    """
    Read pre-generated sparse fingerprints
    """
    start_time = time.time()
    fingerprints = {}
    
    if os.path.isdir(fp_path):
        fp_files = [os.path.join(fp_path, f) for f in os.listdir(fp_path) 
                   if f.endswith('_fingerprints.txt') or f.endswith('.txt')]
    else:
        fp_files = [fp_path]
    
    total_loaded = 0
    for fp_file in fp_files:
        with open(fp_file, 'r') as f:
#            for i, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(fp_file)}")):
            for i, line in enumerate(f):

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
                        continue
                
                fingerprints[mol_id] = fp_array
                total_loaded += 1
    
    elapsed = time.time() - start_time
    print(f"Loaded {total_loaded} fingerprints in {elapsed:.2f} seconds")
    
    return fingerprints

def setup_argparse():
    """Set up argument parser"""
    parser = argparse.ArgumentParser(description="Predict docking scores using trained ML model and pre-generated fingerprints")
    
    parser.add_argument("--model", required=True, help="Path to trained model file (.joblib)")
    parser.add_argument("--input-smi-dir", required=True, help="Directory containing SMILES files")
    parser.add_argument("--input-fp-dir", required=True, help="Directory containing fingerprint files")
    parser.add_argument("--output-dir", default=".", help="Directory to save output files")
    parser.add_argument("--cycle", type=int, default=1, help="Active learning cycle number")
    parser.add_argument("--scaler", help="Path to scaler file")
    parser.add_argument("--n-bits", type=int, default=1024, help="Number of bits in the fingerprint")
    parser.add_argument("--id-col", default="molecule_id", help="Name of column containing molecule IDs")
    parser.add_argument("--smiles-col", default="smiles", help="Name of column containing SMILES strings")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--k", type=float, default=2.0, help="LCB exploration coefficient")
    parser.add_argument("--top-compounds", type=int, default=1000, help="Number of top compounds to select")
    parser.add_argument("--train-smi", default="train_smiles.smi", help="Path to training SMILES file")
    parser.add_argument("--train-fp", default=None, help="Path to training fingerprints file")
    parser.add_argument("--active", action="store_true", help="Enable active learning cycle")
    parser.add_argument("--project-dir", default=None, help="Project directory for active learning")
    parser.add_argument("--acquisition", default="greedy", choices=["greedy", "lcb", "unc"], 
                       help="Acquisition function to use (greedy: y(x), lcb: y(x) - k*σ(x), unc: σ(x))")

    return parser.parse_args()

def load_model_and_scaler(model_path, scaler_path=None):
    """
    Load trained model and optional scaler
    """
    start_time = time.time()
    
    try:
        model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        
        if isinstance(model, RandomForestRegressor):
            print(f"Random Forest model detected with {len(model.estimators_)} trees")
        elif isinstance(model, XGBDistribution):
            print("XGBoost Distribution model detected")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    scaler = None
    if scaler_path:
        try:
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"Model loading completed in {elapsed:.2f} seconds")
    
    return model, scaler

def read_smiles_directory(smi_dir, id_col="molecule_id", smiles_col="smiles"):
    """
    Read all SMILES files from a directory and create a DataFrame
    """
    smi_files = sorted(glob(os.path.join(smi_dir, "*.smi")))
    if not smi_files:
        print(f"No SMILES files found in {smi_dir}")
        sys.exit(1)
    
    data = []
    #print(f"Reading {len(smi_files)} SMILES files...")
    for smi_file in tqdm(smi_files, desc="Reading SMILES"):
        try:
            with open(smi_file, 'r') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    if parts:
                        smiles = parts[0]
                        mol_id = parts[1] if len(parts) > 1 else f"mol_{os.path.basename(smi_file)}_{i}"
                        data.append({id_col: mol_id, smiles_col: smiles})
        except Exception as e:
            print(f"Error reading {smi_file}: {e}")
            continue
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} SMILES from directory")
    return df

def process_in_batches(df, fingerprints, model, scaler, id_col, batch_size):
    """
    Process data in batches to avoid memory issues
    """
    df[id_col] = df[id_col].astype(str)
    num_batches = (len(df) + batch_size - 1) // batch_size
    batches = np.array_split(df, num_batches)
    
    all_predictions = {}
    all_uncertainties = {}
    print(f"Processing {len(df)} molecules in {num_batches} batches...")
    
    for i, batch_df in enumerate(tqdm(batches, desc="Processing batches")):
        batch_ids = batch_df[id_col].values
        batch_fps = []
        valid_ids = []
        
        for mol_id in batch_ids:
            if mol_id in fingerprints:
                batch_fps.append(fingerprints[mol_id])
                valid_ids.append(mol_id)
        
        if not batch_fps:
            continue
        
        X_batch = np.vstack(batch_fps)
        
        if scaler is not None:
            X_batch = scaler.transform(X_batch)
        
        try:
            if isinstance(model, RandomForestRegressor):
                all_preds = np.stack([tree.predict(X_batch) for tree in model.estimators_], axis=0)
                y_pred = np.mean(all_preds, axis=0)
                y_std = np.std(all_preds, axis=0)
            else:  # Assume XGBDistribution
                predictions = model.predict(X_batch)
                y_pred = predictions.loc
                y_std = predictions.scale
            
            for j, mol_id in enumerate(valid_ids):
                all_predictions[mol_id] = y_pred[j]
                all_uncertainties[mol_id] = y_std[j]
        except Exception as e:
            print(f"Error making predictions for batch {i+1}: {e}")
    
    return all_predictions, all_uncertainties

def read_train_smiles(train_smiles_path):
    train_smiles_list = []
    train_ids_list = []
    original_lines = []
    try:
        with open(train_smiles_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    smiles = parts[0]
                    mol_id = parts[1] if len(parts) > 1 else None
                    train_smiles_list.append(smiles)
                    original_lines.append(line.strip())
                    if mol_id:
                        train_ids_list.append(mol_id)
        print(f"Loaded {len(train_smiles_list)} SMILES from {train_smiles_path}")
        if train_ids_list:
            print(f"Found {len(set(train_ids_list))} molecule IDs in training file")
    except Exception as e:
        print(f"Error reading training SMILES file: {e}")
        sys.exit(1)
    return set(train_smiles_list), set(train_ids_list), original_lines

def read_train_fingerprints(train_fp_path):
    """
    Read molecule IDs from training fingerprint file
    """
    train_ids_list = []
    original_lines = []
    
    if not train_fp_path or not os.path.exists(train_fp_path):
        print(f"No training fingerprints file provided or file does not exist")
        return set(), []
    
    try:
        with open(train_fp_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if parts:
                        mol_id = parts[0].strip()
                        train_ids_list.append(mol_id)
                        original_lines.append(line.strip())
        print(f"Loaded {len(train_ids_list)} molecule IDs from {train_fp_path}")
    except Exception as e:
        print(f"Error reading training fingerprints file: {e}")
        
    return set(train_ids_list), original_lines

def calculate_acquisition_scores(predictions, uncertainties, acquisition_type="greedy", k=2.0):
    """
    Calculate acquisition scores based on the specified strategy
    """
    if acquisition_type == "greedy":
        return predictions
    elif acquisition_type == "lcb":
        return predictions - k * uncertainties
    elif acquisition_type == "unc":
        return -1 * uncertainties
    else:
        print(f"Unknown acquisition type: {acquisition_type}, defaulting to greedy")
        return predictions

def plot_distribution_with_selected(pred_values, selected_indices, title, output_path, acquisition): 
    """ 
    Create histogram of predicted docking scores with selected compounds 
    """
    acquisition_display = acquisition.upper() if acquisition in ["lcb", "unc"] else "Greedy"
    title = f"Predicted Docking Scores ({acquisition_display}) - Cycle {title}"
    
    sns.histplot(pred_values, kde=True, bins=50, alpha=0.7, label='All Compounds')
    
    if len(selected_indices) > 0: 
        selected_values = pred_values[selected_indices] 
        sns.histplot(selected_values, bins=50, color='green', alpha=0.7, 
                    label=f'Selected Compounds ({len(selected_indices)})', edgecolor='black')
        threshold = np.max(selected_values) 
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Selection Threshold: {threshold:.2f}')
    
    plt.yscale('function', functions=(lambda x: x**0.5, lambda x: x**2))
    plt.title(title, fontsize=12, pad=10) 
    plt.xlabel("Predicted Docking Score (kcal/mol)", fontsize=10) 
    plt.ylabel("Frequency", fontsize=10) 
    plt.xlim(-20, 10) 
    plt.grid(alpha=0.3) 
    plt.legend(fontsize=6) 
    plt.savefig(output_path, dpi=300, bbox_inches="tight") 
    plt.close()

def plot_uncertainty_scatter(pred_values, uncertainty_values, selected_indices, top_pred_indices, 
                           thresh_pred, thresh_acq, title, output_path, acquisition_type="greedy"):
    acquisition_display = acquisition_type.upper() if acquisition_type in ["lcb", "unc"] else "Greedy"
    title = f"Acquisition Strategy : {acquisition_display} - Cycle {title}"
    
    plt.figure(figsize=(10, 6))
    plt.scatter(pred_values, uncertainty_values, color='gray', alpha=0.5, label='All Data Points')
    
    acquisition_label = {
        "greedy": "Selected by Greedy",
        "lcb": "Selected by LCB",
        "unc": "Selected by UNC"
    }.get(acquisition_type, "Selected")
    
    plt.scatter(pred_values[selected_indices], uncertainty_values[selected_indices], 
                color='red', label=acquisition_label, edgecolors='k')
    
    if not np.array_equal(selected_indices, top_pred_indices):
        plt.scatter(pred_values[top_pred_indices], uncertainty_values[top_pred_indices], 
                    color='blue', alpha=0.3, label='Selected by Prediction', edgecolors='k')
    
    plt.axvline(thresh_pred, linestyle='--', color='blue',
               label=f"Prediction Threshold = {thresh_pred:.2f}")
    
    if acquisition_type != "greedy" and abs(thresh_acq - thresh_pred) > 1e-6:
        if acquisition_type == "lcb":
            plt.axhline(abs(thresh_acq - pred_values[selected_indices[-1]])/args.k, 
                       linestyle='--', color='red',
                       label=f"Min Uncertainty for Selection = {abs(thresh_acq - pred_values[selected_indices[-1]])/args.k:.2f}")
        elif acquisition_type == "unc":
            plt.axhline(uncertainty_values[selected_indices[-1]], 
                       linestyle='--', color='red',
                       label=f"Min Uncertainty for Selection = {uncertainty_values[selected_indices[-1]]:.2f}")
    
    plt.xlabel("Predicted Docking Score (kcal/mol)")
    plt.ylabel("Uncertainty")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def get_project_root_from_train_path(train_path):
    """
    Extract the project root directory from a training file path
    """
    parts = train_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("al_") and part[3:].isdigit():
            return os.path.join(*parts[:i])
    return None

def create_next_cycle_data(args, df, fingerprints, cycle, project_dir=None):
    """
    Create train_smiles.smi and train_fingerprints.txt for the next active learning cycle
    """
    next_cycle = cycle + 1
    
    if project_dir is None:
        if args.train_smi and os.path.exists(args.train_smi):
            possible_project_dir = get_project_root_from_train_path(args.train_smi)
            if possible_project_dir:
                project_dir = possible_project_dir
                print(f"Inferred project directory: {project_dir}")
            else:
                project_dir = os.path.dirname(os.path.dirname(args.output_dir))
                print(f"Using default project directory: {project_dir}")
        else:
            project_dir = os.path.dirname(os.path.dirname(args.output_dir))
            print(f"Using default project directory: {project_dir}")
    
    al_dir = f"al_{next_cycle}"
    al_path = os.path.join(project_dir, al_dir)
    os.makedirs(al_path, exist_ok=True)
    print(f"Creating next cycle directory: {al_path}")
    
    selected_df = df[df["compound_label"] == 1].copy()
    selected_ids = selected_df[args.id_col].tolist()
    
    new_smiles = []
    for mol_id in selected_ids:
        row = df[df[args.id_col] == mol_id]
        if not row.empty:
            smiles = row[args.smiles_col].iloc[0]
            new_smiles.append(f"{smiles}\t{mol_id}")
    
    new_fingerprints = []
    for mol_id in selected_ids:
        if mol_id in fingerprints:
            fp_array = fingerprints[mol_id]
            on_bits = np.where(fp_array == 1)[0]
            fp_str = f"{mol_id},{','.join(map(str, on_bits))}"
            new_fingerprints.append(fp_str)
    
    prev_smiles_lines = []
    prev_fp_lines = []
    
    if args.train_smi and os.path.exists(args.train_smi):
        _, _, prev_smiles_lines = read_train_smiles(args.train_smi)
    
    if args.train_fp and os.path.exists(args.train_fp):
        _, prev_fp_lines = read_train_fingerprints(args.train_fp)
    
    all_smiles = new_smiles + prev_smiles_lines
    all_fingerprints = new_fingerprints + prev_fp_lines
    
    new_smiles_path = os.path.join(al_path, "train_smiles.smi")
    with open(new_smiles_path, 'w') as f:
        f.write("\n".join(all_smiles) + "\n")
    print(f"Saved {len(all_smiles)} SMILES to {new_smiles_path}")
    
    new_fp_path = os.path.join(al_path, "train_fingerprints.txt")
    with open(new_fp_path, 'w') as f:
        f.write("\n".join(all_fingerprints) + "\n")
    print(f"Saved {len(all_fingerprints)} fingerprints to {new_fp_path}")
    
    print(f"\nNext cycle training data breakdown:")
    print(f"  Newly selected compounds: {len(selected_ids)}")
    print(f"  Previous training compounds: {len(prev_smiles_lines)}")
    print(f"  Total compounds for cycle {next_cycle}: {len(all_smiles)}")

def main():
    """Main function"""
    global args
    args = setup_argparse()
    total_start_time = time.time()
    
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    model_name = os.path.basename(args.model).replace("_model.joblib", "").replace(".joblib", "")
    
    df = read_smiles_directory(args.input_smi_dir, args.id_col, args.smiles_col)
    
    fingerprints = read_fingerprints(args.input_fp_dir, args.n_bits)
    if not fingerprints:
        print("Error: No fingerprints could be loaded")
        sys.exit(1)
    
    predictions, uncertainties = process_in_batches(df, fingerprints, model, scaler, args.id_col, args.batch_size)
    
    pred_col = f"predicted_{model_name}"
    uncertainty_col = f"uncertainty_{model_name}"
    acq_col = f"acquisition_{args.acquisition}_{model_name}"
    df[pred_col] = np.nan
    df[uncertainty_col] = np.nan
    df[acq_col] = np.nan
    df["compound_label"] = 0
    
    matched_count = 0
    prediction_values = []
    uncertainty_values = []
    matched_indices = []
    
    for i, mol_id in enumerate(df[args.id_col].astype(str)):
        if mol_id in predictions:
            pred_value = predictions[mol_id]
            unc_value = uncertainties[mol_id]
            df.loc[i, pred_col] = pred_value
            df.loc[i, uncertainty_col] = unc_value
            prediction_values.append(pred_value)
            uncertainty_values.append(unc_value)
            matched_indices.append(i)
            matched_count += 1
    
    print(f"Made predictions for {matched_count} out of {len(df)} molecules ({matched_count/len(df)*100:.1f}%)")
    
    if matched_count > 0:
        prediction_values = np.array(prediction_values)
        uncertainty_values = np.array(uncertainty_values)
        
        # Calculate acquisition scores
        acquisition_scores = calculate_acquisition_scores(
            prediction_values, uncertainty_values, args.acquisition, args.k)
        
        for idx, i in enumerate(matched_indices):
            df.loc[i, acq_col] = acquisition_scores[idx]
        
        train_smiles = set()
        train_ids_from_smiles = set()
        train_ids_from_fp = set()
        
        if args.train_smi and os.path.exists(args.train_smi):
            train_smiles, train_ids_from_smiles, _ = read_train_smiles(args.train_smi)
        
        if args.train_fp and os.path.exists(args.train_fp):
            train_ids_from_fp, _ = read_train_fingerprints(args.train_fp)
        
        for i in df.index:
            mol_id = df.loc[i, args.id_col]
            if pd.notna(mol_id) and str(mol_id) in train_ids_from_smiles:
                df.loc[i, "compound_label"] = 2
        
        acq_order = np.argsort(acquisition_scores)
        
        selected_count = 0
        selected_indices = []
        
        for idx in acq_order:
            df_idx = matched_indices[idx]
            if df.loc[df_idx, "compound_label"] == 2:
                continue
            df.loc[df_idx, "compound_label"] = 1
            selected_indices.append(idx)
            selected_count += 1
            if selected_count >= args.top_compounds:
                break
        
        if selected_count < args.top_compounds:
            print(f"\nWARNING: Could only select {selected_count} new compounds based on {args.acquisition} scores")
        
        top_pred_indices = []
        pred_order = np.argsort(prediction_values)
        selected_count = 0
        
        for idx in pred_order:
            df_idx = matched_indices[idx]
            if df.loc[df_idx, "compound_label"] == 2:
                continue
            if selected_count >= args.top_compounds:
                break
            top_pred_indices.append(idx)
            selected_count += 1
        
        if selected_indices:
            thresh_acq = acquisition_scores[selected_indices[-1]]
        else:
            thresh_acq = np.inf
            
        if top_pred_indices:
            thresh_pred = prediction_values[top_pred_indices[-1]]
        else:
            thresh_pred = np.inf
        
        overlap = np.intersect1d(selected_indices, top_pred_indices)
        
        print(f"\nAcquisition Analysis for {args.acquisition}:")
        print(f"Number of compounds selected by prediction (greedy): {len(top_pred_indices)}")
        print(f"Number of compounds selected by {args.acquisition}: {len(selected_indices)}")
        print(f"Overlap: {len(overlap)} ({len(overlap)/max(1, len(selected_indices))*100:.1f}%)")
        
        if selected_indices:
            print(f"Prediction range for {args.acquisition}-selected: {prediction_values[selected_indices].min():.4f} to {prediction_values[selected_indices].max():.4f}")
            print(f"Uncertainty range for {args.acquisition}-selected: {uncertainty_values[selected_indices].min():.4f} to {uncertainty_values[selected_indices].max():.4f}")
        
        label_counts = df["compound_label"].value_counts()
        print(f"\nLabel Distribution:")
        print(f"  {args.acquisition}-selected (label 1): {label_counts.get(1, 0)}")
        print(f"  Train SMILES/FP (label 2): {label_counts.get(2, 0)}")
        print(f"  Others (label 0): {label_counts.get(0, 0)}")
    
    cycle_dir = os.path.join(args.output_dir)
    os.makedirs(cycle_dir, exist_ok=True)
    
    pred_plot_path = os.path.join(cycle_dir, f"{model_name}_{args.acquisition}_predictions_cycle{args.cycle}.png")
    scatter_plot_path = os.path.join(cycle_dir, f"{model_name}_{args.acquisition}_uncertainty_scatter_cycle{args.cycle}.png")
    
    if matched_count > 0 and selected_indices:
        plot_distribution_with_selected(
            prediction_values, selected_indices, 
            args.cycle, pred_plot_path, args.acquisition)
        plot_uncertainty_scatter(
            prediction_values, uncertainty_values, 
            selected_indices, top_pred_indices,
            thresh_pred, thresh_acq, 
            args.cycle, scatter_plot_path, args.acquisition)
        print(f"Saved plots to {pred_plot_path} and {scatter_plot_path}")
    
    output_csv = os.path.join(cycle_dir, f"predictions_{args.acquisition}_cycle_{args.cycle}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")
    
    if args.active and matched_count > 0:
        if args.project_dir:
            project_dir = os.path.abspath(args.project_dir)
        else:
            project_dir = os.getcwd()
        print(f"Using project directory: {project_dir}")
        create_next_cycle_data(args, df, fingerprints, args.cycle, project_dir)
    
    valid_preds = ~np.isnan(df[pred_col])
    if valid_preds.any():
        valid_values = df.loc[valid_preds, pred_col]
        print("\nPrediction summary:")
        print(f"  Count: {valid_values.count()}")
        print(f"  Mean: {valid_values.mean():.4f}")
        print(f"  Min: {valid_values.min():.4f}")
        print(f"  Max: {valid_values.max():.4f}")
        print(f"  Std Dev: {valid_values.std():.4f}")
        
        selected_df = df[df["compound_label"] == 1]
        print(f"\nSelected {len(selected_df)} compounds for docking (label 1)")
        if len(selected_df) > 0:
            selected_scores = selected_df[pred_col].dropna()
            print(f"  Mean predicted score: {selected_scores.mean():.4f}")
            print(f"  Min predicted score: {selected_scores.min():.4f}")
            print(f"  Max predicted score: {selected_scores.max():.4f}")
    
    total_elapsed = time.time() - total_start_time
    print(f"Total runtime: {total_elapsed:.2f} seconds")
    print(f"Inference complete. Results saved to {cycle_dir}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()
