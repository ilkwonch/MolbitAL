import matplotlib
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm

from regressor import create_regressor


def setup_argparse():
    """Set up argument parser"""
    parser = argparse.ArgumentParser(description="Train ML models on docking scores using pre-generated fingerprints")
    
    parser.add_argument("--input-csv", required=True, help="Path to input CSV file with docking scores")
    parser.add_argument("--input-fp", required=True, help="Path to input fingerprint file or directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save trained models and results")
    
    parser.add_argument("--model", default="rf", 
                        choices=["rf", "xgbd"],
                        help="Regression model to use (default: rf)")
    parser.add_argument("--n-bits", type=int, default=1024,
                        help="Number of bits in the fingerprint (default: 1024)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save-predictions", action="store_true", 
                        help="Save predictions to CSV")
    parser.add_argument("--no-plots", action="store_true", 
                        help="Disable generation of plots")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with detailed output")
    
    return parser.parse_args()


def read_fingerprints(fp_path, n_bits=1024, debug=False):
    start_time = time.time()
    fingerprints = {}
    
    print(f"Reading fingerprints from {fp_path}...")
    
    if os.path.isdir(fp_path):
        fp_files = [os.path.join(fp_path, f) for f in os.listdir(fp_path) 
                   if f.endswith('_fingerprints.txt') or f.endswith('.txt')]
    else:
        fp_files = [fp_path]
    
    if debug:
        print(f"Found {len(fp_files)} fingerprint files")
        print("Sample raw fingerprint entries:")
        try:
            with open(fp_files[0], 'r') as f:
                for i, line in enumerate(f):
                    if i >= 5: break
                    print(f"  {line.strip()}")
            print("...")
        except Exception as e:
            print(f"Error reading sample fingerprints: {e}")
    
    total_loaded = 0
    for fp_file in fp_files:
        with open(fp_file, 'r') as f:
            for i, line in enumerate(tqdm(f, desc=f"Reading {os.path.basename(fp_file)}", disable=not debug)):
                if not line.strip():
                    continue
                
                parts = line.strip().split(',')
                
                if len(parts) < 1:
                    if debug:
                        print(f"Warning: Line {i+1} has invalid format: {line}")
                    continue
                    
                mol_id = parts[0].strip()
                
                fp_array = np.zeros(n_bits, dtype=np.int8)
                
                if len(parts) > 1:
                    try:
                        on_bits = [int(bit) for bit in parts[1:] if bit.strip().isdigit()]
                        valid_bits = [bit for bit in on_bits if 0 <= bit < n_bits]
                        if len(valid_bits) != len(on_bits) and debug:
                            print(f"Warning: Line {i+1}, molecule {mol_id} has {len(on_bits) - len(valid_bits)} invalid bit positions")
                        if valid_bits:
                            fp_array[valid_bits] = 1
                    except Exception as e:
                        if debug:
                            print(f"Error parsing bits for molecule {mol_id}: {e}")
                        continue
                
                fingerprints[mol_id] = fp_array
                total_loaded += 1
    
    elapsed = time.time() - start_time
    print(f"Loaded {total_loaded} fingerprints in {elapsed:.2f} seconds")
    
    return fingerprints


def filter_and_clean_data(df, id_col='molecule_id', debug=False):
    df = df.copy()
    
    initial_rows = len(df)
    print(f"Initial data: {initial_rows} rows")
    
    df[id_col] = df[id_col].astype(str)
    
    if debug:
        print("\nSample raw docking scores (first 5 rows):")
        print(df[['molecule_id', 'docking']].head())
    
    df = df.dropna(subset=['docking'])
    print(f"Rows after removing NaN values: {len(df)}")
    
    df['docking'] = pd.to_numeric(df['docking'], errors='coerce')
    df = df.dropna(subset=['docking'])
   
    #custom outlier applied , truncated <-20, >10 kcal/mol
    outliers = df[(df['docking'] > 10) | (df['docking'] < -20)]
    if not outliers.empty:
        print(f"\nRemoving {len(outliers)} outlier values outside range -20 to 10 kcal/mol")
        if debug:
            print("Sample outlier values:")
            print(outliers['docking'].head())
    
    df = df[(df['docking'] <= 10) & (df['docking'] >= -20)]
    print(f"Rows after removing outliers: {len(df)}")
    
    dup_ids = df[id_col].duplicated()
    if dup_ids.any():
        print(f"Warning: {dup_ids.sum()} duplicate molecule IDs found")
        df = df[~df[id_col].duplicated()]
    
    df = df.reset_index(drop=True)
    
    if debug:
        print("\nDocking score statistics after cleaning:")
        print(df['docking'].describe())
    
    print(f"Final cleaned data: {len(df)} rows ({len(df)/initial_rows:.1%} of original)")
    return df


def match_fingerprints_with_scores(fingerprints, scores_df, id_col='molecule_id', debug=False):
    start_time = time.time()
    print("Matching fingerprints with docking scores...")
    
    scores_df[id_col] = scores_df[id_col].astype(str)
    
    matched_data = []
    matched_ids = []
    matched_scores = []
    
    for idx, row in tqdm(scores_df.iterrows(), total=len(scores_df), desc="Matching data"):
        mol_id = row[id_col]
        if mol_id in fingerprints:
            matched_data.append(fingerprints[mol_id])
            matched_ids.append(mol_id)
            matched_scores.append(row['docking'])
    
    if matched_data:
        X = np.vstack(matched_data)
        y = np.array(matched_scores)
    else:
        X = np.array([])
        y = np.array([])
    
    elapsed = time.time() - start_time
    print(f"Matched {len(matched_ids)} molecules in {elapsed:.2f} seconds")
    
    if debug:
        if len(y) > 0:
            print("\nTarget (docking) value statistics after matching:")
            print(f"  Min: {np.min(y):.4f}")
            print(f"  Max: {np.max(y):.4f}")
            print(f"  Mean: {np.mean(y):.4f}")
            print(f"  Std: {np.std(y):.4f}")
            
    return X, y, matched_ids


def plot_distribution(y, output_dir, title="Docking Score Distribution"):
    """Plot the distribution of docking scores"""
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=30)
    plt.title(title)
    plt.xlabel("Docking Score (kcal/mol)")
    plt.ylabel("Frequency")
    plt.xlim(-20, 10)  
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "docking_score_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_and_plot(model, X_train, X_test, y_train, y_test, output_dir, skip_plots=False, debug=False):
    """Evaluate the model and create performance plots"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\nModel: {model.name}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    if skip_plots:
        return {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2
        }
    
    plt.figure(figsize=(12, 5))
    
    # Training set
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    plt.title(f'Training Set')
    plt.xlabel('Docking score')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)

    textstr = f'R² = {train_r2:.3f}\nRMSE = {train_rmse:.3f}\nMAE = {train_mae:.3f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))   


    # Test set
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f'Test set')
    plt.xlabel('Docking score')
    plt.ylabel('Predicted')
    plt.grid(alpha=0.3)


    textstr = f'R² = {test_r2:.3f}\nRMSE = {test_rmse:.3f}\nMAE = {test_mae:.3f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model.name}_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    residuals_train = y_train - y_train_pred
    plt.scatter(y_train_pred, residuals_train, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Training Set Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.grid(alpha=0.3)
    
    # Test residuals
    plt.subplot(1, 2, 2)
    residuals_test = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals_test, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', lw=2)
    plt.title('Test Set Residuals')
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.grid(alpha=0.3)
   

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model.name}_residuals.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2
    }


def make_predictions(model, X, molecule_ids, scores_df, id_col='molecule_id'):
    """Make predictions and add them to the scores DataFrame"""
    df = scores_df.copy()
    
    df[id_col] = df[id_col].astype(str)
    
    predictions, uncertainties = model.predict_with_uncertainty(X)
    
    col_name = f"predicted_{model.name}"
    unc_col_name = f"uncertainty_{model.name}"
    
    df[col_name] = np.nan
    df[unc_col_name] = np.nan
    
    id_to_idx = {id_val: idx for idx, id_val in enumerate(molecule_ids)}
    
    for i, mol_id in enumerate(df[id_col]):
        if mol_id in id_to_idx:
            idx = id_to_idx[mol_id]
            df.loc[i, col_name] = predictions[idx]
            df.loc[i, unc_col_name] = uncertainties[idx]
    
    return df


def main():
    """Main function"""
    args = setup_argparse()
    
    total_start_time = time.time()
    
    output_dir = os.path.join(args.output_dir, f"ml_models_{args.model}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    debug = args.debug
    
    if debug:
        print("DEBUG MODE ENABLED")
        print(f"Arguments: {args}")
    
    print(f"Loading docking scores from {args.input_csv}")
    try:
        scores_df = pd.read_csv(args.input_csv)
        if debug:
            print(f"Loaded CSV with {len(scores_df)} rows and {len(scores_df.columns)} columns")
            print(f"Column names: {scores_df.columns.tolist()}")
            print(f"First 5 rows:\n{scores_df.head()}")
    except Exception as e:
        print(f"Error loading input CSV file: {e}")
        sys.exit(1)
    
    required_cols = ['molecule_id', 'docking']
    missing_cols = [col for col in required_cols if col not in scores_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    scores_df = filter_and_clean_data(scores_df, debug=debug)
    
    if len(scores_df) < 10:
        print("Error: Not enough valid data points for training")
        sys.exit(1)
    
    fingerprints = read_fingerprints(args.input_fp, args.n_bits, debug=debug)
    
    if not fingerprints:
        print("Error: No fingerprints could be loaded")
        sys.exit(1)
    
    X, y, matched_ids = match_fingerprints_with_scores(fingerprints, scores_df, debug=debug)
    
    if len(X) == 0:
        print("Error: No matching data between fingerprints and docking scores")
        sys.exit(1)
    
    if debug:
        print(f"Final dataset: {len(X)} samples with {X.shape[1]} features")
        print(f"Sample fingerprint (bits set to 1): {np.sum(X[0])}")
        print(f"Sample targets (first 5): {y[:5]}")
    
    if not args.no_plots:
        plot_distribution(y, output_dir, "Distribution of Docking Scores")
    
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if debug:
        print(f"Scaled features: mean={scaler.mean_[:5]}..., std={scaler.scale_[:5]}...")
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_scaled, y, matched_ids, test_size=args.test_size, random_state=args.random_state)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    
    print(f"Training {args.model} model...")
    model = create_regressor(args.model, random_state=args.random_state)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{args.model}_model.joblib")
    joblib.dump(model.model, model_path)
    print(f"Model saved to {model_path}")
    
    metrics = evaluate_and_plot(model, X_train, X_test, y_train, y_test, output_dir, args.no_plots, debug=debug)
    
    metrics_path = os.path.join(output_dir, f"{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    if args.save_predictions:
        predictions_df = make_predictions(model, X_scaled, matched_ids, scores_df)
        predictions_path = os.path.join(output_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Saved predictions to {predictions_path}")
    
    with open(os.path.join(output_dir, "training_config.txt"), 'w') as f:
        f.write(f"Input CSV: {args.input_csv}\n")
        f.write(f"Input fingerprints: {args.input_fp}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Fingerprint bits: {args.n_bits}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Random state: {args.random_state}\n")
    
    total_elapsed = time.time() - total_start_time
    print(f"Total runtime: {total_elapsed:.2f} seconds")
    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    matplotlib.use('Agg')  
    main()
