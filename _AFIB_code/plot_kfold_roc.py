import os
import argparse
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from scipy import signal
from scipy.stats import zscore
import optuna

from helper_code import load_header, get_leads, get_frequency, get_labels, load_recording, expand_leads
from device_selector import DeviceSelector
from model_structure import NN
"""
python3 plot_kfold_roc.py --study_name afib_hpo_parallel_2.5.1 --db_path /srv/home/jhyl/Afib_recurrence/diplomka/_AFIB_code/afib_hpo_parallel_2.5.1.db --data_dir /srv/home/jhyl/Afib_recurrence/finetune_data_all/train --trial_number 42 --target_epoch 5
"""

plt.rcParams.update({
    'mathtext.fontset': 'cm',         # Uses Matplotlib's built-in Computer Modern font
    'font.family': 'serif',           # Defaults to the serif Computer Modern style
    'axes.unicode_minus': False,      # Prevents warning bugs with minus signs
    'figure.figsize': (6.27, 8.0),    # Exact width of A4 text block, taller for 2 plots
    'figure.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def preprocessing(recording, fs):
    b,a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    if fs==1000:
        recording = signal.resample_poly(recording, up=1, down=2, axis=-1)
        fs = 500
    elif fs==500:
        pass
    else:
        recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
        fs = 500

    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)

    tensor_data = torch.tensor(recording, dtype=torch.float32)
    tensor_data = tensor_data.unsqueeze(0)
    tensor_data = tensor_data.unsqueeze(2)

    return tensor_data

def run_inference(model, header_files, DEVICE):
    outputs = []
    targets = []

    features_csv_path = '/srv/home/jhyl/Afib_recurrence/features.csv'
    if os.path.exists(features_csv_path):
        features_df = pd.read_csv(features_csv_path)
        rhythm_map = dict(zip(features_df['ID'].astype(str), features_df['is_AFIB_before']))
    else:
        rhythm_map = {}

    for header_path in header_files:
        recording_path_mat = header_path.replace(".hea", ".mat")
        recording_path_npy = header_path.replace(".hea", ".npy")
        filename = os.path.basename(header_path).replace(".hea", "")

        try:
            header = load_header(header_path)
        except Exception as e:
            print(f"Could not load {header_path}: {e}")
            continue
            
        leads = get_leads(header)
        fs = get_frequency(header)
        label = int(get_labels(header)[0])

        if os.path.exists(recording_path_npy):
            recording = np.load(recording_path_npy)
            _, lead_indicator = expand_leads(np.zeros((12, 1)), leads)
            lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32).unsqueeze(0)
            
            tensor_data = torch.tensor(recording, dtype=torch.float32)
            if len(tensor_data.shape) == 2:
                tensor_data = tensor_data.unsqueeze(0).unsqueeze(2)
            elif len(tensor_data.shape) == 3:
                tensor_data = tensor_data.unsqueeze(2)
            recording = tensor_data
        else:
            recording = load_recording(recording_path_mat)
            recording, lead_indicator = expand_leads(recording, leads)
            lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32).unsqueeze(0)
            recording = preprocessing(recording, fs)

        recording = recording.to(DEVICE)
        sig_len = recording.shape[-1]
        window_size = 4992
        step_size = 2496
        
        if sig_len < window_size:
            pad_len = window_size - sig_len
            recording = torch.nn.functional.pad(recording, (0, pad_len), "constant", 0)
            sig_len = window_size
            
        windows = []
        for start in range(0, sig_len - window_size + 1, step_size):
            windows.append(recording[:, :, :, start : start + window_size])
        if len(windows) == 0:
            windows.append(recording[:, :, :, -window_size:])
            
        windows_tensor = torch.cat(windows, dim=0).to(DEVICE)
        lead_indicator_expanded = lead_indicator.expand(windows_tensor.shape[0], -1).to(DEVICE)
        
        rhythm = rhythm_map.get(filename, 0)
        if rhythm == 1:
            rhythm_vec = [1, -1]
        else:
            rhythm_vec = [-1, 1]
        r_tensor = torch.tensor([rhythm_vec], dtype=torch.float32).to(DEVICE)
        r_expanded = r_tensor.expand(windows_tensor.shape[0], -1)
        
        with torch.no_grad():
            y = model(windows_tensor, lead_indicator_expanded, r_expanded)
            p = torch.softmax(y, dim=1)
        
        prob_recurrence = p[:, 1].mean().item()
        
        outputs.append(prob_recurrence)
        targets.append(label)

    return np.array(targets), np.array(outputs)

def plot_kfold_roc(trial_dir, data_dir, output_plot, target_epoch=None):
    selector = DeviceSelector()
    DEVICE = selector.select(1)[0]
    
    print(f"Scanning trial directory: {trial_dir}")
    folds = [d for d in os.listdir(trial_dir) if d.startswith("fold_") and os.path.isdir(os.path.join(trial_dir, d))]
    folds.sort()
    
    if len(folds) == 0:
        raise ValueError(f"No fold directories found in {trial_dir}")
        
    print(f"Found {len(folds)} folds: {folds}")

    mean_fpr = np.linspace(0, 1, 100)
    tprs_train = []
    tprs_valid = []
    
    aucs_train = []
    aucs_valid = []

    all_v_targets = []
    all_v_probs = []

    model = NN(nOUT=2).to(DEVICE)

    for i, fold in enumerate(folds):
        fold_dir = os.path.join(trial_dir, fold)
        
        # 1. Load Model Weights
        if target_epoch is not None:
            checkpoints = [f for f in os.listdir(fold_dir) if f.startswith("checkpoint_epoch")]
            weight_path = None
            for ckpt in checkpoints:
                try:
                    epoch = int(ckpt.split('_')[-1].split('.')[0])
                    if epoch == target_epoch:
                        weight_path = os.path.join(fold_dir, ckpt)
                        break
                except ValueError:
                    continue
            if weight_path is None:
                raise FileNotFoundError(f"No checkpoint found for epoch {target_epoch} in {fold_dir}")
        else:
            weight_path = os.path.join(fold_dir, "best_loss_weights.pth")
            if not os.path.exists(weight_path):
                checkpoints = [f for f in os.listdir(fold_dir) if f.startswith("checkpoint_epoch")]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    weight_path = os.path.join(fold_dir, checkpoints[-1])
                else:
                    raise FileNotFoundError(f"No valid weights or checkpoints found in {fold_dir}")
                
        print(f"\n--- Loading {fold} ---")
        print(f"Weights: {weight_path}")
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Dynamically adjust head if model was trained with extra rhythm features
        if 'head.weight' in state_dict and state_dict['head.weight'].shape[1] != model.head.in_features:
            model.head = torch.nn.Linear(state_dict['head.weight'].shape[1], 2).to(DEVICE)
            
        model.load_state_dict(state_dict)
        model.eval()

        # 2. Load Split Data
        pickle_path = os.path.join(fold_dir, f"PROGRESS_{fold}.pickle")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Could not find {pickle_path} to get the data splits!")
            
        with open(pickle_path, 'rb') as handle:
            _ = pickle.load(handle) # OUTPUT
            train_files_base = pickle.load(handle)
            valid_files_base = pickle.load(handle)
            
        train_files = [os.path.join(data_dir, f) for f in train_files_base]
        valid_files = [os.path.join(data_dir, f) for f in valid_files_base]
        
        print(f"Train samples: {len(train_files)} | Valid samples: {len(valid_files)}")
        
        # 3. Run Inference Train
        print("Running inference on Train Set...")
        t_targets, t_probs = run_inference(model, train_files, DEVICE)
        fpr_t, tpr_t, _ = roc_curve(t_targets, t_probs)
        roc_auc_t = auc(fpr_t, tpr_t)
        aucs_train.append(roc_auc_t)
        
        interp_tpr_t = np.interp(mean_fpr, fpr_t, tpr_t)
        interp_tpr_t[0] = 0.0
        tprs_train.append(interp_tpr_t)
        
        # 4. Run Inference Valid
        print("Running inference on Valid Set...")
        v_targets, v_probs = run_inference(model, valid_files, DEVICE)
        fpr_v, tpr_v, _ = roc_curve(v_targets, v_probs)
        roc_auc_v = auc(fpr_v, tpr_v)
        aucs_valid.append(roc_auc_v)

        all_v_targets.extend(v_targets)
        all_v_probs.extend(v_probs)
        
        interp_tpr_v = np.interp(mean_fpr, fpr_v, tpr_v)
        interp_tpr_v[0] = 0.0
        tprs_valid.append(interp_tpr_v)

    print("\n--- Calculating Global Validation Threshold ---")
    fpr_pool, tpr_pool, thresh_pool = roc_curve(all_v_targets, all_v_probs)
    J_pool = tpr_pool - fpr_pool
    opt_idx = np.argmax(J_pool)
    optimal_val_threshold = thresh_pool[opt_idx]
    
    print(f"POOLED OPTIMAL VALIDATION THRESHOLD: {optimal_val_threshold:.4f}")
    
    # Save it to a text file in the trial directory so you don't lose it
    with open(os.path.join(trial_dir, "optimal_val_threshold.txt"), "w") as f:
        f.write(str(optimal_val_threshold))

    # --- Calculate Means and Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.27, 8.0))

    # --- Train Plot ---
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Chance', alpha=.8)
    
    for i, interp_tpr in enumerate(tprs_train):
        ax1.plot(mean_fpr, interp_tpr, lw=1, alpha=0.3, label=f'{folds[i]} ROC (AUC = {aucs_train[i]:.3f})')
        
    mean_tpr_t = np.mean(tprs_train, axis=0)
    mean_tpr_t[-1] = 1.0
    mean_auc_t = auc(mean_fpr, mean_tpr_t)
    std_auc_t = np.std(aucs_train)
    ax1.plot(mean_fpr, mean_tpr_t, color='b', label=f'Mean ROC (AUC = {mean_auc_t:.3f} $\\pm$ {std_auc_t:.3f})', lw=2, alpha=.8)
    
    std_tpr_t = np.std(tprs_train, axis=0)
    tprs_upper_t = np.minimum(mean_tpr_t + std_tpr_t, 1)
    tprs_lower_t = np.maximum(mean_tpr_t - std_tpr_t, 0)
    ax1.fill_between(mean_fpr, tprs_lower_t, tprs_upper_t, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Cross-Validation ROC (Training Set)')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # --- Valid Plot ---
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Chance', alpha=.8)
    
    for i, interp_tpr in enumerate(tprs_valid):
        ax2.plot(mean_fpr, interp_tpr, lw=1, alpha=0.3, label=f'{folds[i]} ROC (AUC = {aucs_valid[i]:.3f})')
        
    mean_tpr_v = np.mean(tprs_valid, axis=0)
    mean_tpr_v[-1] = 1.0
    mean_auc_v = auc(mean_fpr, mean_tpr_v)
    std_auc_v = np.std(aucs_valid)
    ax2.plot(mean_fpr, mean_tpr_v, color='darkorange', label=f'Mean ROC (AUC = {mean_auc_v:.3f} $\\pm$ {std_auc_v:.3f})', lw=2, alpha=.8)
    
    std_tpr_v = np.std(tprs_valid, axis=0)
    tprs_upper_v = np.minimum(mean_tpr_v + std_tpr_v, 1)
    tprs_lower_v = np.maximum(mean_tpr_v - std_tpr_v, 0)
    ax2.fill_between(mean_fpr, tprs_lower_v, tprs_upper_v, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Cross-Validation ROC (Validation Set)')
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='bottom')

    plt.tight_layout()
    
    # Save the plot
    out_dir = os.path.dirname(output_plot)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"\nROC curves saved to {output_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot K-Fold Cross Validation ROC curves.")
    parser.add_argument("--trial_dir", type=str, help="Direct path to the trial directory containing fold_1, fold_2, etc.")
    parser.add_argument("--study_name", type=str, help="Optuna study name to automatically find the best trial.")
    parser.add_argument("--db_path", type=str, help="Path to the Optuna SQLite DB.")
    parser.add_argument("--trial_number", type=int, help="Specific trial number to use from the DB.")
    parser.add_argument("--model_dir", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/hpo_runs", help="Base directory where Optuna trials are saved.")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing the original data (.hea files) so it can reconstruct the splits.")
    parser.add_argument("--output_plot", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_42_2.5.1/kfold_roc_curves.png", help="Path to save the output plot.")
    parser.add_argument("--target_epoch", type=int, default=None, help="Epoch number to use for ROC plotting. If provided, models from this specific epoch across all folds will be used.")
    
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
    
    plot_kfold_roc(trial_dir, args.data_dir, args.output_plot, target_epoch=args.target_epoch)
