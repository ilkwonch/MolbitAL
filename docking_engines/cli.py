# docking_engines/cli.py
import argparse
import subprocess
import os
import sys
from pathlib import Path
import glob

def get_engine_path(engine_name):
    """Return the path to the engine executable."""
    # Get the path to the installed package
    package_dir = Path(__file__).parent
    engine_paths = {
        "vina_gpu2.1": package_dir / "engines/AutoDock-Vina-GPU-2-1",
        "qvina_w": package_dir / "engines/qvina-w",
        "qvina2.1": package_dir / "engines/qvina2.1",
        "smina": package_dir / "engines/smina",
        "vina": package_dir / "engines/vina"
    }
    
    # Handle unidock separately as it's installed via conda
    if engine_name == "unidock":
        return Path("unidock")  # Use the command directly, assuming it's in PATH
    
    if engine_name not in engine_paths:
        raise ValueError(f"Unknown engine: {engine_name}. Available: {list(engine_paths.keys())} and unidock")
    return engine_paths[engine_name]

def run_vina_type_engine(engine, receptor, ligand, output, config=None, extra_args=None):
    """Run Vina, QVina, or Smina docking engine."""
    engine_path = get_engine_path(engine)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine binary not found at {engine_path}")
    
    cmd = [str(engine_path), "--receptor", receptor, "--ligand", ligand]
    
    if config:
        cmd.extend(["--config", config])
    
    # Default parameters
    if "--num_modes" not in " ".join(extra_args or []):
        cmd.extend(["--num_modes", "1"])
    
    if output:
        cmd.extend(["--out", output])
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully ran {engine} for {ligand}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {engine} for {ligand}: {e}")
        return False

def run_unidock(receptor, output_dir, config=None, ligand=None, ligand_index=None, batch=None, gpu_batch=None, extra_args=None):
    """Run UniDock docking engine with its specific parameters."""
    engine_path = get_engine_path("unidock")
    # For unidock, we don't check if the path exists since it's assumed to be in PATH via conda
    
    cmd = [str(engine_path), "--receptor", receptor]
    
    if config:
        cmd.extend(["--config", config])
    
    # Handle ligand inputs (one of these must be specified)
    if ligand:
        cmd.extend(["--ligand", ligand])
    elif ligand_index:
        cmd.extend(["--ligand_index", ligand_index])
    elif batch:
        cmd.extend(["--batch", batch])
    elif gpu_batch:
        cmd.extend(["--gpu_batch", gpu_batch])
    else:
        raise ValueError("Must specify one of: --ligand, --ligand_index, --batch, or --gpu_batch")
    
    # Add output directory
    if output_dir:
        cmd.extend(["--dir", output_dir])
    
    # Default parameters if not in extra_args
    if extra_args is None:
        extra_args = []
    
    if "--search_mode" not in " ".join(extra_args):
        cmd.extend(["--search_mode", "balance"])
    
    if "--scoring" not in " ".join(extra_args):
        cmd.extend(["--scoring", "vina"])
    
    if "--num_modes" not in " ".join(extra_args):
        cmd.extend(["--num_modes", "1"])
    
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully ran unidock")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running unidock: {e}")
        return False

def batch_process_ligands(engine, receptor, ligands_pattern, output_dir, config=None, extra_args=None):
    """Process multiple ligands with the specified engine."""
    ligand_files = glob.glob(ligands_pattern)
    
    if not ligand_files:
        print(f"No ligand files found matching pattern: {ligands_pattern}")
        return False
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    success_count = 0
    total_count = len(ligand_files)
    
    # For unidock, we can process all ligands at once
    if engine == "unidock":
        # Create a ligand index file
        index_file = output_dir / "ligand_index.txt"
        with open(index_file, "w") as f:
            f.write(" ".join([str(p) for p in ligand_files]))
        
        result = run_unidock(
            receptor=receptor,
            output_dir=str(output_dir),
            config=config,
            ligand_index=str(index_file),
            extra_args=extra_args
        )
        if result:
            success_count = total_count
    else:
        # For vina-like engines, process one by one
        for ligand_file in ligand_files:
            ligand_name = Path(ligand_file).stem
            output_file = output_dir / f"{ligand_name}_out.pdbqt"
            
            result = run_vina_type_engine(
                engine=engine,
                receptor=receptor,
                ligand=ligand_file,
                output=str(output_file),
                config=config,
                extra_args=extra_args
            )
            
            if result:
                success_count += 1
            
            print(f"Progress: {success_count}/{total_count}")
    
    print(f"Completed {success_count} out of {total_count} docking runs")
    return success_count == total_count

def main():
    parser = argparse.ArgumentParser(description="Run molecular docking with various engines")
    parser.add_argument("--engine", required=True, 
                        choices=["vina_gpu2.1", "qvina_w", "qvina2.1", "smina", "vina", "unidock"],
                        help="Docking engine to use")
    parser.add_argument("--receptor", required=True, help="Path to receptor file (e.g., PDBQT)")
    
    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument("--ligand", help="Path to a single ligand file (e.g., PDBQT)")
    ligand_group.add_argument("--ligand-pattern", help="Glob pattern to match multiple ligand files (e.g., 'ligands/*.pdbqt')")
    ligand_group.add_argument("--ligand-index", help="Path to file containing list of ligand paths (supports unidock)")
    ligand_group.add_argument("--gpu-batch", help="Pattern for GPU batch processing (supports unidock)")
    ligand_group.add_argument("--batch", help="Pattern for batch processing (supports unidock)")
    
    # Output options
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Path to output file (for single ligand)")
    output_group.add_argument("--output-dir", help="Directory for output files (for multiple ligands, supports unidock)")
    
    # Optional common parameters
    parser.add_argument("--config", help="Path to configuration file (e.g., autobox.txt)")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                        help="Additional arguments for the engine")
    
    args = parser.parse_args()
    
    # Process based on input type
    if args.engine == "unidock":
        if args.ligand:
            run_unidock(
                receptor=args.receptor,
                output_dir=args.output_dir or os.path.dirname(args.output or "."),
                config=args.config,
                ligand=args.ligand,
                extra_args=args.extra_args
            )
        elif args.ligand_index:
            run_unidock(
                receptor=args.receptor,
                output_dir=args.output_dir or os.path.dirname(args.output or "."),
                config=args.config,
                ligand_index=args.ligand_index,
                extra_args=args.extra_args
            )
        elif args.gpu_batch:
            run_unidock(
                receptor=args.receptor,
                output_dir=args.output_dir or os.path.dirname(args.output or "."),
                config=args.config,
                gpu_batch=args.gpu_batch,
                extra_args=args.extra_args
            )
        elif args.batch:
            run_unidock(
                receptor=args.receptor,
                output_dir=args.output_dir or os.path.dirname(args.output or "."),
                config=args.config,
                batch=args.batch,
                extra_args=args.extra_args
            )
        elif args.ligand_pattern:
            batch_process_ligands(
                engine=args.engine,
                receptor=args.receptor,
                ligands_pattern=args.ligand_pattern,
                output_dir=args.output_dir,
                config=args.config,
                extra_args=args.extra_args
            )
    else:
        # For other engines (vina-like)
        if args.ligand:
            run_vina_type_engine(
                engine=args.engine,
                receptor=args.receptor,
                ligand=args.ligand,
                output=args.output,
                config=args.config,
                extra_args=args.extra_args
            )
        elif args.ligand_pattern:
            batch_process_ligands(
                engine=args.engine,
                receptor=args.receptor,
                ligands_pattern=args.ligand_pattern,
                output_dir=args.output_dir,
                config=args.config,
                extra_args=args.extra_args
            )

if __name__ == "__main__":
    main()
