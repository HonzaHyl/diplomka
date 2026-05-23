import os
import argparse
import torch
import optuna
from model_structure import EnsembleNN
"""
python3 create_ensemble.py --study_name afib_hpo_parallel_2.5.1 --db_path afib_hpo_parallel_2.5.1.db --trial_number 42 --target_epoch 5

"""

def create_ensemble(trial_dir, output_path, device, target_epoch=None):
    """
    Loads weights from the folds of a cross-validation trial
    and saves them as a single EnsembleNN model.
    """
    print(f"Scanning trial directory: {trial_dir}")
    
    # Find fold directories
    folds = [d for d in os.listdir(trial_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(trial_dir, d))]
    folds.sort() # Ensure consistent ordering
    
    if len(folds) == 0:
        raise ValueError(f"No fold directories found in {trial_dir}")
        
    print(f"Found {len(folds)} folds: {folds}")
    
    all_weight_paths = []
    
    if target_epoch is not None:
        print(f"Collecting checkpoints for epoch {target_epoch}...")
        for fold in folds:
            fold_dir = os.path.join(trial_dir, fold)
            checkpoints = [f for f in os.listdir(fold_dir) if f.startswith("checkpoint_epoch")]
            for ckpt in checkpoints:
                try:
                    epoch = int(ckpt.split('_')[-1].split('.')[0])
                    if epoch == target_epoch:
                        all_weight_paths.append(os.path.join(fold_dir, ckpt))
                except ValueError:
                    continue # Skip if it doesn't match expected format
        if not all_weight_paths:
            raise FileNotFoundError(f"No checkpoints found for epoch {target_epoch}.")
        
        # Sort paths to be deterministic
        all_weight_paths.sort()
    else:
        for i, fold in enumerate(folds):
            fold_dir = os.path.join(trial_dir, fold)
            
            # Prefer best_loss_weights.pth as the end model
            weight_path = os.path.join(fold_dir, "best_loss_weights.pth")
            
            if not os.path.exists(weight_path):
                print(f"  best_loss_weights.pth not found in {fold}, looking for latest checkpoint...")
                checkpoints = [f for f in os.listdir(fold_dir) if f.startswith("checkpoint_epoch")]
                if checkpoints:
                    # Sort by epoch number to get the latest
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    weight_path = os.path.join(fold_dir, checkpoints[-1])
                else:
                    raise FileNotFoundError(f"No valid weights or checkpoints found in {fold_dir}")
                    
            all_weight_paths.append(weight_path)
            
    num_models = len(all_weight_paths)
    print(f"Total models to include in ensemble: {num_models}")
            
    # Initialize the ensemble model
    ensemble = EnsembleNN(nOUT=2, num_models=num_models).to(device)
    
    for i, weight_path in enumerate(all_weight_paths):
        print(f"  Loading model {i+1}/{num_models} from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Load weights into the corresponding sub-model
        ensemble.models[i].load_state_dict(state_dict)
        ensemble.models[i].eval()
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    print(f"Saving ensemble model to {output_path}")
    # Save the whole ensemble state dict so it can be loaded with ensemble.load_state_dict()
    torch.save({
        'model_state_dict': ensemble.state_dict(),
        'num_models': num_models,
        'is_ensemble': True
    }, output_path)
    
    print("Done!")
    print(f"\nTo use this model in testing:")
    print(f"  1. Initialize: model = EnsembleNN(nOUT=2, num_models={num_models})")
    print(f"  2. Load weights: model.load_state_dict(torch.load('{output_path}')['model_state_dict'])")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an ensemble model from a cross-validation trial.")
    parser.add_argument("--trial_dir", type=str, help="Direct path to the trial directory containing fold_1, fold_2, etc.")
    parser.add_argument("--study_name", type=str, help="Optuna study name to automatically find the best trial (e.g., afib_hpo_parallel_2.4).")
    parser.add_argument("--db_path", type=str, help="Path to the Optuna SQLite DB (e.g., /srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/afib_hpo_parallel_2.4.db).")
    parser.add_argument("--trial_number", type=int, help="Specific trial number to use from the DB (optional, defaults to best trial).")
    parser.add_argument("--model_dir", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/hpo_runs", help="Base directory where Optuna trials are saved.")
    parser.add_argument("--output_file", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/ensemble_model.pth", help="Output path for the ensemble .pth file.")
    parser.add_argument("--target_epoch", type=int, default=None, help="Epoch number to create the ensemble from. If provided, models from this specific epoch across all folds will be used.")
    
    args = parser.parse_args()
    
    if args.study_name and args.db_path:
        print(f"Loading Optuna study '{args.study_name}' from {args.db_path}...")
        storage_name = f"sqlite:///{os.path.abspath(args.db_path)}"
        study = optuna.load_study(study_name=args.study_name, storage=storage_name)
        
        if args.trial_number is not None:
            trial_number = args.trial_number
            print(f"Using explicitly requested trial: #{trial_number}")
        else:
            best_trial = study.best_trial
            trial_number = best_trial.number
            print(f"Found best trial: #{trial_number} (Value: {best_trial.value:.4f})")
            
        trial_dir = os.path.join(args.model_dir, f"trial_{trial_number}")
    elif args.trial_dir:
        trial_dir = args.trial_dir
    else:
        raise ValueError("You must provide either --trial_dir OR both --study_name and --db_path")
    
    # Use CPU to just load and re-save the weights to avoid GPU memory overhead
    device = torch.device('cpu')
    
    create_ensemble(trial_dir, args.output_file, device, target_epoch=args.target_epoch)
