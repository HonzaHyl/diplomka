import os
import torch
import argparse
from model_structure import EnsembleNN, NN
from bcos_utils import bcosify_model
from helper_code import finetune_model_prep

def create_ensemble_bcos(trial_dir, output_path, device):
    """
    Creates an EnsembleNN from B-cosified fold checkpoints.
    """
    # 1. Find all folds in the trial directory
    folds = [d for d in os.listdir(trial_dir) if d.startswith('fold_') and os.path.isdir(os.path.join(trial_dir, d))]
    folds.sort()
    
    if not folds:
        raise ValueError(f"No fold directories found in {trial_dir}")
    
    print(f"Scanning trial directory: {trial_dir}")
    print(f"Found {len(folds)} folds: {folds}")
    
    # 2. Initialize the Ensemble model
    ensemble = EnsembleNN(nOUT=2, num_models=len(folds)).to(device)
    
    # 3. Load each fold's best weights into the sub-models
    for i, fold_name in enumerate(folds):
        fold_path = os.path.join(trial_dir, fold_name)
        
        # Try finding 'best_loss_weights.pth' or 'checkpoint_epoch_X.pth'
        weight_file = os.path.join(fold_path, "best_loss_weights.pth")
        if not os.path.exists(weight_file):
            # Fallback to the last epoch checkpoint if best_loss doesn't exist
            checkpoints = [f for f in os.listdir(fold_path) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                weight_file = os.path.join(fold_path, checkpoints[-1])
        
        if not os.path.exists(weight_file):
            print(f"  [WARNING] No weights found for {fold_name}. Skipping.")
            continue
            
        print(f"  Loading {fold_name} from {weight_file}")
        checkpoint = torch.load(weight_file, map_location=device)
        
        # Extract the state dict (handle both full checkpoints and raw weights)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # IMPORTANT: Expand to 24 channels and B-cosify before loading weights!
        finetune_model_prep(ensemble.models[i])
        
        # Load weights
        ensemble.models[i].load_state_dict(state_dict)
        
    # 4. Save the final ensemble
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': ensemble.state_dict(),
        'num_models': len(folds),
        'is_ensemble': True,
        'is_bcos': True
    }, output_path)
    
    print("\nSuccessfully created B-cos Ensemble!")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a B-cos Ensemble model.")
    parser.add_argument("--trial_dir", type=str, required=True, help="Path to directory containing fold_1, fold_2, etc.")
    parser.add_argument("--output_file", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/ensemble_bcos_model.pth", help="Output path.")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    create_ensemble_bcos(args.trial_dir, args.output_file, device)
