"""
extract_fold_checkpoints.py

Extracts folds from best or selected epoch

Usage:
    # Default (Best loss):
    python extract_fold_checkpoints.py --optuna_db sqlite:////srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/afib_hpo_parallel_2.5.1.db --study afib_hpo_parallel_2.5.1 --output ./results
    
    # Specific Epoch (e.g., Epoch 5):
    python extract_fold_checkpoints.py --optuna_db sqlite:////srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/afib_hpo_parallel_2.5.1.db \
        --study afib_hpo_parallel_2.5.1 \
        --output /srv/home/jhyl/Afib_recurrence/diplomka/results \
        --epoch 5 \
        --trial 42    
"""

import optuna
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import argparse
import shutil
import os

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_OPTUNA_DB    = "sqlite:////srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/afib_hpo_parallel_2.5.1.db"
DEFAULT_STUDY_NAME   = "afib_hpo_parallel_2.5.1"
DEFAULT_MLFLOW_DB    = "sqlite:////srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/mlflow.db"
DEFAULT_OUTPUT_DIR   = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model"
# ─────────────────────────────────────────────────────────────────────────────

def get_artifact_fs_path(artifact_uri: str, relative_path: str) -> str:
    """
    Given an artifact_uri like 'file:///some/path/artifacts' or
    '/some/path/artifacts', return the absolute filesystem path
    to the requested relative artifact.
    """
    base = artifact_uri
    if base.startswith("file://"):
        base = base[len("file://"):]
    return os.path.join(base, relative_path)

def extract_checkpoints():
    parser = argparse.ArgumentParser(description="Extract fold checkpoints using Optuna as truth")
    parser.add_argument("--optuna_db", default=DEFAULT_OPTUNA_DB)
    parser.add_argument("--study", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--mlflow_db", default=DEFAULT_MLFLOW_DB)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--trial", type=int, default=None, help="Specific trial number to extract (defaults to best)")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to extract from all folds (e.g., 5). Defaults to best_loss_weights.")
    args = parser.parse_args()

    # 1. Connect to Optuna and find the trial
    print(f"Connecting to Optuna DB: {args.optuna_db}")
    try:
        study = optuna.load_study(study_name=args.study, storage=args.optuna_db)
    except Exception as e:
        print(f"Error loading Optuna study: {e}")
        return

    if args.trial is not None:
        try:
            target_trial = study.trials[args.trial]
            print(f"Targeting specific Trial #{args.trial}")
        except IndexError:
            print(f"Trial #{args.trial} not found in study.")
            return
    else:
        target_trial = study.best_trial
        print(f"Targeting Best Trial #{target_trial.number} (Value: {target_trial.value:.4f})")

    trial_number = target_trial.number
    trial_params = target_trial.params

    # 2. Connect to MLflow
    print(f"Connecting to MLflow at {args.mlflow_db}")
    mlflow.set_tracking_uri(args.mlflow_db)
    client = MlflowClient()

    # Search for the MLflow parent run that matches "Trial_N"
    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments]

    # Search for runs with the name "Trial_N"
    runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=f"tags.`mlflow.runName` = 'Trial_{trial_number}'",
        order_by=["start_time DESC"]
    )

    if not runs:
        print(f"Could not find MLflow run 'Trial_{trial_number}'")
        return

    # To resolve conflicts, find the run whose params match the Optuna trial params
    selected_run = None
    for run in runs:
        match = True
        for p_name, p_val in trial_params.items():
            ml_val = run.data.params.get(p_name)
            if ml_val is not None:
                # MLflow stores params as strings
                try:
                    if abs(float(ml_val) - float(p_val)) > 1e-6:
                        match = False
                        break
                except ValueError:
                    if ml_val != str(p_val):
                        match = False
                        break
        if match:
            selected_run = run
            break

    if not selected_run:
        print(f"Found runs named 'Trial_{trial_number}' but none matched the Optuna parameters.")
        selected_run = runs[0]
        print(f"Falling back to the most recent run with that name: {selected_run.info.run_id}")
    else:
        print(f"Matched MLflow run: {selected_run.info.run_id}")

    parent_run_id = selected_run.info.run_id

    # 3. Find fold runs (children of this parent)
    child_runs = client.search_runs(
        experiment_ids=exp_ids,
        filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'",
        order_by=["start_time ASC"]
    )

    if not child_runs:
        print(f"No child fold runs found for MLflow parent run {parent_run_id}")
        return

    print(f"\nFound {len(child_runs)} fold runs:")
    for r in child_runs:
        print(f"  {r.data.tags.get('mlflow.runName', '?')} (run_id: {r.info.run_id})")

    # 4. Output directory
    if args.epoch is not None:
        out_dir = os.path.join(args.output, f"Trial_{trial_number}_epoch_{args.epoch}")
    else:
        out_dir = os.path.join(args.output, f"Trial_{trial_number}")
        
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutputting checkpoints to: {out_dir}")

    # 5. Extract checkpoints from each fold
    for fold_num, run in enumerate(child_runs, start=1):
        fold_name = run.data.tags.get("mlflow.runName", f"Fold_{fold_num}")
        artifact_uri = run.info.artifact_uri

        # --- NEW LOGIC: Check for specific epoch vs best loss ---
        if args.epoch is not None:
            checkpoint_dir = get_artifact_fs_path(artifact_uri, "checkpoints")
            expected_filename = f"checkpoint_epoch_{args.epoch}.pth"
            src = os.path.join(checkpoint_dir, expected_filename)
            
            if not os.path.exists(src):
                print(f"  [{fold_name}] WARNING: {expected_filename} not found. Skipping.")
                continue
                
        else:
            # Original Logic: Prefer the explicitly saved best-loss weights
            best_path_relative = "model/best_loss_weights.pth"
            src = get_artifact_fs_path(artifact_uri, best_path_relative)

            if not os.path.exists(src):
                print(f"  [{fold_name}] best_loss_weights.pth not found, scanning checkpoints/...")
                checkpoint_dir = get_artifact_fs_path(artifact_uri, "checkpoints")
                if os.path.isdir(checkpoint_dir):
                    ckpts = sorted(
                        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
                        key=lambda f: int(f.replace("checkpoint_epoch_", "").replace(".pth", ""))
                    )
                    if ckpts:
                        src = os.path.join(checkpoint_dir, ckpts[-1])
                        print(f"  [{fold_name}] Using last checkpoint: {ckpts[-1]}")
                    else:
                        print(f"  [{fold_name}] WARNING: No checkpoints found. Skipping.")
                        continue
                else:
                    print(f"  [{fold_name}] WARNING: No model found. Skipping.")
                    continue
        # --------------------------------------------------------

        dst = os.path.join(out_dir, f"MODEL_{fold_num}.pth")
        shutil.copy2(src, dst)
        
        if args.epoch is not None:
            print(f"  [{fold_name}] ✓ Copied Epoch {args.epoch} → {dst}")
        else:
            print(f"  [{fold_name}] ✓ Copied → {dst}")

    print(f"\nDone! All fold checkpoints are in:\n  {out_dir}")

if __name__ == "__main__":
    extract_checkpoints()