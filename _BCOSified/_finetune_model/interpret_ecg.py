import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.signal as signal
from scipy.stats import zscore

from model_structure import EnsembleNN, NN
from helper_code import load_header, load_recording, expand_leads, get_frequency, get_leads, finetune_model_prep
from bcos_utils import bcosify_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES_CSV = "/srv/home/jhyl/Afib_recurrence/features.csv"
WINDOW_SIZE = 4992

def get_bcos_explanation(model, input_tensor, lead_ind, rhythm_vec, target_class=0):
    """
    Extracts the inherent B-cos explanation (Gradient) for a target class.
    Default target_class=0 (Recurrence).
    """
    model.to(DEVICE)
    model.eval()
    
    # Ensure gradients can be computed
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad = True
    
    lead_ind = lead_ind.to(DEVICE)
    rhythm_vec = rhythm_vec.to(DEVICE)

    # Forward pass
    # EnsembleNN returns log_probs, but for attribution we want the sum of logits 
    # to get the pure W(x) alignment.
    output = model(input_tensor, lead_ind, rhythm_vec)
    
    # Target class logit (Recurrence is class 0)
    target_logit = output[:, target_class].sum()
    
    model.zero_grad()
    target_logit.backward()
    
    # Gradient is the Dynamic Linear Mapping W(x)
    grad = input_tensor.grad.detach().cpu().numpy() # [Batch, 24, 1, Seq]
    
    # The attribution is W(x) * x. Since our input was split into [pos, neg], 
    # the 24-channel gradient already represents the contribution of each side.
    # We combine them back to 12-lead space:
    explanation = grad[0, :12, 0, :] - grad[0, 12:, 0, :]
    
    return explanation

def preprocess_for_viz(record_path, rhythm_val):
    """Matches the exact preprocessing used in main_code.py"""
    header = load_header(record_path)
    fs = get_frequency(header)
    leads = get_leads(header)
    
    recording_path = record_path.replace(".hea", ".mat")
    recording = load_recording(recording_path)
    
    # 1. Resample to 500Hz
    if fs != 500:
        recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
        
    # 2. Filter & Z-score
    b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')
    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)
    
    # 3. Expand to 12 leads
    recording_12, lead_ind = expand_leads(recording, leads)
    
    # 4. Rhythm vector mapping
    if rhythm_val == 1:
        rv = np.array([1, -1], dtype=np.float32)
    else:
        rv = np.array([-1, 1], dtype=np.float32)
    rv_tensor = torch.tensor(rv).unsqueeze(0)
    lead_ind_tensor = torch.from_numpy(lead_ind).float().unsqueeze(0)

    # Return the full recording (no truncation) along with metadata tensors.
    # The caller is responsible for windowing if needed.
    return lead_ind_tensor, rv_tensor, recording_12

def create_interpretability_plot(patient_id, signal_12, attribution_12, output_path):
    """Creates a 12-lead interactive plot with colored background sectors"""
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    fig = make_subplots(rows=12, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03,
                        subplot_titles=lead_names)
    
    time = np.arange(signal_12.shape[1]) / 500.0 # seconds
    
    # Normalize attribution for coloring (scale -1 to 1)
    # We use a robust normalization to avoid single peaks washing out the plot
    max_attr = np.percentile(np.abs(attribution_12), 99.5) + 1e-8
    norm_attr = np.clip(attribution_12 / max_attr, -1, 1)
    
    for i in range(12):
        # 1. Background Attribution Heatmap
        # This creates the "colored sectors" effect
        # We set the Y range of the heatmap to cover the signal range
        y_min, y_max = np.min(signal_12[i]), np.max(signal_12[i])
        y_padding = (y_max - y_min) * 0.1
        
        fig.add_trace(
            go.Heatmap(
                x=time,
                y=[y_min - y_padding, y_max + y_padding],
                z=[norm_attr[i]],
                colorscale='RdBu_r', # Red for positive (Recurrence), Blue for negative (Healthy)
                zmid=0,
                showscale=(i == 0),
                opacity=0.35,
                name=f"Importance {lead_names[i]}",
                hoverinfo='skip',
                colorbar=dict(
                    title="Influence<br>Red: Recurrence<br>Blue: Healthy", 
                    x=1.02, 
                    thickness=15
                ) if i == 0 else None
            ),
            row=i+1, col=1
        )
        
        # 2. Solid ECG Signal Line (Black)
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_12[i],
                mode='lines',
                line=dict(color='black', width=1.2),
                name=f"Lead {lead_names[i]}",
                hoverinfo='x+y'
            ),
            row=i+1, col=1
        )
        
    fig.update_layout(
        height=2200, # Taller for better lead visibility
        width=1200,
        title_text=f"B-cos Interpretability Report: Patient {patient_id}",
        showlegend=False,
        template="plotly_white", # Light background
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    # Sync Y-axes to the signal data
    for i in range(12):
        y_min, y_max = np.min(signal_12[i]), np.max(signal_12[i])
        y_padding = (y_max - y_min) * 0.15
        fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding], row=i+1, col=1)

    fig.write_html(output_path)
    print(f"Explanation saved to: {output_path}")

def bcos_attribution_full(model, recording_12, lead_ind_tensor, rv_tensor, target_class=0):
    """
    Slide WINDOW_SIZE windows across the full recording, compute B-cos attributions
    for each window, and stitch them into a single attribution array that matches
    the full signal length.

    A half-window stride is used so adjacent windows overlap.  Overlapping
    regions are averaged, giving a smooth transition between windows.
    """
    n_leads, total_len = recording_12.shape
    stride = WINDOW_SIZE // 2

    # Accumulator arrays for averaging overlapping regions
    attr_sum = np.zeros_like(recording_12, dtype=np.float64)
    attr_count = np.zeros(total_len, dtype=np.float64)

    starts = list(range(0, total_len - WINDOW_SIZE + 1, stride))
    # Always include a window that covers the very end of the signal
    if not starts or starts[-1] + WINDOW_SIZE < total_len:
        starts.append(max(0, total_len - WINDOW_SIZE))

    for idx, start in enumerate(starts):
        end = start + WINDOW_SIZE
        win = recording_12[:, start:end]  # [12, WINDOW_SIZE]

        # Pad if the recording is shorter than WINDOW_SIZE
        if win.shape[1] < WINDOW_SIZE:
            pad = WINDOW_SIZE - win.shape[1]
            win = np.pad(win, ((0, 0), (0, pad)))

        # B-cosify this window
        pos = np.maximum(win, 0)
        neg = np.maximum(-win, 0)
        data_bcos = np.concatenate([pos, neg], axis=0)  # [24, WINDOW_SIZE]
        input_t = torch.from_numpy(data_bcos).float().unsqueeze(0).unsqueeze(2)

        attr_win = get_bcos_explanation(model, input_t, lead_ind_tensor, rv_tensor, target_class)
        # attr_win: [12, WINDOW_SIZE]

        actual_end = min(end, total_len)
        actual_len = actual_end - start
        attr_sum[:, start:actual_end] += attr_win[:, :actual_len]
        attr_count[start:actual_end] += 1.0

        print(f"  Window {idx + 1}/{len(starts)}: samples {start}–{actual_end}")

    # Average overlapping contributions
    attr_count = np.maximum(attr_count, 1.0)  # avoid divide-by-zero
    attribution_full = attr_sum / attr_count[np.newaxis, :]
    return attribution_full


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to ensemble_model.pth")
    parser.add_argument("--record_path", type=str, required=True, help="Path to a .hea file")
    parser.add_argument("--output_dir", type=str, default="/srv/home/jhyl/Afib_recurrence/diplomka/results/explanations", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model
    print("Loading model...")
    checkpoint = torch.load(args.model_path, map_location=DEVICE)

    # Determine if this is an ensemble or a single fold
    is_ensemble = checkpoint.get('is_ensemble', False)

    if is_ensemble:
        num_models = checkpoint.get('num_models', 4)
        model = EnsembleNN(nOUT=2, num_models=num_models)
        # Expand and B-cosify each sub-model in the ensemble
        for m in model.models:
            finetune_model_prep(m)
        state_dict = checkpoint['model_state_dict']
    else:
        # Single model (from a checkpoint_epoch_X.pth)
        model = NN(nOUT=2)
        finetune_model_prep(model)
        # Check if it's a full checkpoint or just weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

    model.to(DEVICE)
    model.load_state_dict(state_dict)

    # Load Patient Rhythm Info
    pid = os.path.basename(args.record_path).replace(".hea", "")
    fdf = pd.read_csv(FEATURES_CSV)
    rhythm_val = fdf[fdf['ID'].astype(str) == pid]['is_AFIB_before'].values[0]

    # Preprocess — returns the full-length recording
    print(f"Preprocessing patient {pid}...")
    lead_t, rv_t, raw_12 = preprocess_for_viz(args.record_path, rhythm_val)

    # Slide windows across the full recording and stitch attributions
    print("Generating B-cos explanation across full signal...")
    attr = bcos_attribution_full(model, raw_12, lead_t, rv_t, target_class=0)

    # Visualize
    out_file = os.path.join(args.output_dir, f"explanation_{pid}.html")
    create_interpretability_plot(pid, raw_12, attr, out_file)
