import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import zscore
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             precision_recall_curve, average_precision_score, 
                             brier_score_loss)

from helper_code import find_header_files, load_header, get_leads, get_frequency, get_labels, load_recording, expand_leads
from device_selector import DeviceSelector
from model_structure import EnsembleNN, NN

"""
python3 evaluate_model.py --model_path /srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_42_2.5.1/epoch_5/ensemble_model_5.pth \
--data_dir /srv/home/jhyl/Afib_recurrence/finetune_data_all/test \
--output_plot /srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_42_2.5.1/test_results_roc.png \
--threshold 0.5110
"""


# --- GLOBAL PLOT SETTINGS FOR A4 PRINTING (Vertical Stack) ---
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
# -------------------------------------------------------------

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

def evaluate(model_path, data_dir, output_plot, optimal_threshold):
    selector = DeviceSelector()
    DEVICE = selector.select(1)[0]
    
    print('Finding header and recording files...')
    header_files = find_header_files(data_dir)

    print(f'Loading model from {model_path}...')
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if checkpoint.get('is_ensemble', False):
        num_models = checkpoint.get('num_models', 4)
        print(f"Loading Ensemble model with {num_models} sub-models...")
        model = EnsembleNN(nOUT=2, num_models=num_models).to(DEVICE)
    else:
        print("Loading standard single NN model...")
        model = NN(nOUT=2).to(DEVICE)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    outputs = []
    targets = []

    features_csv_path = '/srv/home/jhyl/Afib_recurrence/features.csv'
    if os.path.exists(features_csv_path):
        features_df = pd.read_csv(features_csv_path)
        rhythm_map = dict(zip(features_df['ID'].astype(str), features_df['is_AFIB_before']))
    else:
        rhythm_map = {}

    print("Running inference on Test Set...")
    for header_path in header_files:
        recording_path_mat = header_path.replace(".hea", ".mat")
        recording_path_npy = header_path.replace(".hea", ".npy")
        filename = os.path.basename(header_path).replace(".hea", "")

        header = load_header(header_path)
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
        
        prob_healthy = p[:, 0].mean().item()
        prob_recurrence = p[:, 1].mean().item()
        patient_p = np.array([[prob_healthy, prob_recurrence]])
        
        outputs.append(patient_p)
        targets.append(label)

    outputs = np.concatenate(outputs, axis=0)
    positive_class_probs = outputs[:, 1]
    targets_np = np.array(targets)

    # --- 1. ROC AUC & Bootstrapped Confidence Intervals ---
    fpr, tpr, thresholds = roc_curve(targets_np, positive_class_probs)
    roc_auc = auc(fpr, tpr)

    n_bootstraps = 1000
    rng_seed = 42
    bootstrapped_aucs = []
    rng = np.random.RandomState(rng_seed)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(positive_class_probs), len(positive_class_probs))
        if len(np.unique(targets_np[indices])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(targets_np[indices], positive_class_probs[indices])
        bootstrapped_aucs.append(auc(fpr_b, tpr_b))
        
    sorted_scores = np.array(bootstrapped_aucs)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    print(f"\n================ TEST SET RESULTS ================")
    print(f"AUROC: {roc_auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")

    # --- 2. Calculate Clinical Metrics using the LOCKED Validation Threshold ---
    print(f"Applying locked validation threshold: {optimal_threshold:.4f}")
    binary_predictions = (positive_class_probs >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(targets_np, binary_predictions).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"\nMetrics at Fixed Threshold ({optimal_threshold:.4f}):")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"NPV: {npv:.4f}")

    # --- 3. Advanced Metrics ---
    prc_auc = average_precision_score(targets_np, positive_class_probs)
    brier_score = brier_score_loss(targets_np, positive_class_probs)
    print(f"\nAUPRC: {prc_auc:.4f}")
    print(f"Brier Score: {brier_score:.4f}")
    print("==================================================\n")

    # --- 4. Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1) # Stacked vertically

    # Panel 1: ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Map the locked threshold to the closest point on the Test ROC curve for visualization
    closest_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    thresh_fpr = fpr[closest_idx]
    thresh_tpr = tpr[closest_idx]
    
    ax1.scatter(thresh_fpr, thresh_tpr, marker='o', color='red', s=50, zorder=5,
                label=f'Fixed Threshold: {optimal_threshold:.2f}\n(Sens: {thresh_tpr:.2f}, Spec: {1-thresh_fpr:.2f})')

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)')
    ax1.set_ylabel('True Positive Rate (Sensitivity)')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")

    # Panel 2: Probability Distribution
    probs_healthy = positive_class_probs[targets_np == 0]
    probs_afib = positive_class_probs[targets_np == 1]
    
    ax2.hist(probs_healthy, bins=np.linspace(0, 1, 51), alpha=0.5, color='green', label='Healthy (0)', density=True)
    ax2.hist(probs_afib, bins=np.linspace(0, 1, 51), alpha=0.5, color='red', label='AFib Recurrence (1)', density=True)
    
    ax2.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2, 
                label=f'Fixed Threshold ({optimal_threshold:.2f})')

    ax2.set_xlim([-0.05, 1.05])
    ax2.set_xlabel('Predicted Probability of AFib Recurrence')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distributions by Class')
    ax2.legend(loc="upper right")
    ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='bottom')
    ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='bottom')

    plt.tight_layout()
    
    # Save the plot
    output_dir = os.path.dirname(output_plot)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_plot, bbox_inches='tight')
    print(f"Test evaluation plots saved to {output_plot}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model and plot ROC curve')
    parser.add_argument('--model_path', type=str, required=True, help='Path to ensemble model (.pth)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with test data')
    parser.add_argument('--output_plot', type=str, required=True, help='Path to save ROC plot (e.g. plot.png)')
    
    # NEW ARGUMENT: Required threshold from the validation phase
    parser.add_argument('--threshold', type=float, required=True, help='The locked optimal threshold calculated from the validation set.')
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.output_plot, args.threshold)