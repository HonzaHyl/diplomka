#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy import signal
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d

import neurokit2 as nk

# Custom codebase imports
from helper_code import (
    find_header_files, 
    load_header, 
    get_leads, 
    get_frequency, 
    get_labels, 
    load_recording, 
    expand_leads
)
from device_selector import DeviceSelector
from model_structure import EnsembleNN, NN
from captum.attr import IntegratedGradients, DeepLift, NoiseTunnel

"""
Usage Examples:

  # Classic two-pass mode (default)
  python3 explain_model.py \
      --model_path /srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_42_2.5.1/epoch_5/ensemble_model_5.pth \
      --data_dir   /srv/home/jhyl/Afib_recurrence/finetune_data_all/test \
      --output_dir /srv/home/jhyl/Afib_recurrence/diplomka/results/explainability \
      --method deeplift --leads I,II,V2 --record_id 313

  # Attribution Spectrogram mode
  python3 explain_model.py \
      --model_path /srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_42_2.5.1/epoch_5/ensemble_model_5.pth \
      --data_dir   /srv/home/jhyl/Afib_recurrence/finetune_data_all/test \
      --output_dir /srv/home/jhyl/Afib_recurrence/diplomka/results/explainability \
      --leads I,II,V2 --record_id 495 \
      --mode spectrogram --spectrogram_window 4.0,6.0
"""

# Set default premium plotting style settings
plt.rcParams.update({
    'mathtext.fontset': 'cm',         # Uses Matplotlib's built-in Computer Modern font
    'font.family': 'serif',           # Defaults to the serif Computer Modern style
    'axes.unicode_minus': False,      # Prevents warning bugs with minus signs
    'figure.figsize': (6.27, 4.0),    # Exact width of A4 text block
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

# ---------------------------------------------------------------------------
# Attribution Spectrogram helpers
# ---------------------------------------------------------------------------

def build_filter_bank(upper_hz=47):
    """Return list of (label, low_hz, high_hz) for 4-Hz-step bands up to upper_hz, starting with 1–4 Hz."""
    bands = []
    # First band is 1 to 4 Hz
    bands.append(("1–4 Hz", 1.0, 4.0))
    # Remaining bands are 4–8, 8–12, ..., up to upper_hz
    low = 4.0
    step = 4.0
    while low < upper_hz:
        high = min(low + step, float(upper_hz))
        label = f"{int(low)}–{int(high)} Hz"
        bands.append((label, low, high))
        low = high
    return bands


def bandpass_filter_np(sig_np, low_hz, high_hz, fs, order=4):
    """
    FFT-based zero-phase band-pass filter with raised-cosine transition windows.
    Ensures perfect mathematical complementarity, no phase distortion, and sharp transitions without Gibbs ringing.
    Works on last axis.
    """
    n = sig_np.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    
    # Smooth transition band (e.g. 1.0 Hz wide)
    trans_hz = 1.0
    
    mask = np.zeros(len(freqs), dtype=np.float64)
    for i, f in enumerate(freqs):
        # Inside the passband
        if low_hz <= f <= high_hz:
            mask[i] = 1.0
        # Lower transition band [low_hz - trans_hz, low_hz]
        elif low_hz - trans_hz < f < low_hz:
            t = (f - (low_hz - trans_hz)) / trans_hz
            mask[i] = 0.5 * (1.0 - np.cos(np.pi * t))
        # Upper transition band [high_hz, high_hz + trans_hz]
        elif high_hz < f < high_hz + trans_hz:
            t = (f - high_hz) / trans_hz
            mask[i] = 0.5 * (1.0 + np.cos(np.pi * t))
            
    # Apply filter in the frequency domain
    fft_data = np.fft.rfft(sig_np, axis=-1)
    filtered = np.fft.irfft(fft_data * mask, n=n, axis=-1)
    return filtered.copy()


def make_band_baseline(x_np, low_hz, high_hz, fs):
    """baseline = original − band component  →  DeepLIFT measures that band's unique value."""
    band_component = bandpass_filter_np(x_np, low_hz, high_hz, fs)
    return x_np - band_component


def run_spectrogram_attribution(
    x_input, l_input, r_input,
    sub_models, target_class,
    bands, fs, device,
    nt_samples=10, stdevs=0.05
    ):
    """
    For each frequency band run DeepLIFT+SmoothGrad with a band-subtracted baseline.
    Returns dict {lead_idx -> np.ndarray shape (n_bands, T)}.
    """
    patient_np = x_input.cpu().detach().numpy()   # (1, 12, 1, T)
    n_leads = x_input.shape[1]
    T = x_input.shape[-1]
    n_bands = len(bands)

    band_attrs = []   # list of n_bands arrays, each shape (n_leads, T)

    for b_idx, (band_label, low_hz, high_hz) in enumerate(bands):
        print(f"  Band {b_idx+1}/{n_bands}: {band_label} ...")

        baseline_np = make_band_baseline(patient_np, low_hz, high_hz, fs)
        baseline_tensor = torch.tensor(baseline_np, dtype=torch.float32).to(device)

        weighted_attrs = []
        for sub_model in sub_models:
            sub_model.eval()
            with torch.no_grad():
                logits = sub_model(x_input, l_input, r_input)
                prob = torch.softmax(logits, dim=1)[0, target_class].item()
            try:
                explainer = NoiseTunnel(DeepLift(sub_model))
                attr_t = explainer.attribute(
                    inputs=x_input,
                    nt_type='smoothgrad', nt_samples=nt_samples, stdevs=stdevs,
                    baselines=baseline_tensor, target=target_class,
                    additional_forward_args=(l_input, r_input)
                )
                # attr_t shape: (1, 12, 1, T) → squeeze → (12, T)
                attr_np = attr_t.squeeze(0).squeeze(1).detach().cpu().numpy()
                weighted_attrs.append((prob, attr_np))
            except Exception as e:
                print(f"    [Warning] DeepLIFT failed for band {band_label}: {e}")

        if weighted_attrs:
            total_w = sum(p for p, _ in weighted_attrs)
            if total_w > 1e-6:
                agg = sum(p * a for p, a in weighted_attrs) / total_w
            else:
                agg = np.mean([a for _, a in weighted_attrs], axis=0)
        else:
            agg = np.zeros((n_leads, T))

        band_attrs.append(agg)   # (n_leads, T)

    # Build per-lead (n_bands, T) matrix
    per_lead = {}
    for lead_idx in range(n_leads):
        matrix = np.stack([band_attrs[b][lead_idx, :] for b in range(n_bands)], axis=0)
        per_lead[lead_idx] = matrix

    return per_lead


def plot_attribution_spectrogram(
    attr_matrix, ecg_sig, band_labels, time_axis,
    lead_name, filename, patient_out_dir,
    gt_text, pred_text, best_prob,
    t_start=4.0, t_end=6.0
 ):
    """
    Plot a 2D attribution spectrogram (heatmap) with ECG waveform overlaid.
    attr_matrix: (n_bands, T)  —  rows=low→high freq, cols=time
    ecg_sig:     (T,)  —  single lead ECG
    Always computed w.r.t. Recurrence (class 1): Red = evidence for Recurrence,
    Blue = evidence for Healthy.
    """
    mask = (time_axis >= t_start) & (time_axis <= t_end)
    t_zoom = time_axis[mask]
    attr_zoom = attr_matrix[:, mask]          # (n_bands, T_zoom)
    ecg_zoom  = ecg_sig[mask]                 # (T_zoom,)
    n_bands = attr_matrix.shape[0]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # --- Heatmap ---
    vmax = max(np.percentile(np.abs(attr_zoom), 98), 1e-8)
    im = ax.imshow(
        attr_zoom,
        cmap='RdBu_r',
        aspect='auto',
        extent=[t_zoom[0], t_zoom[-1], -0.5, n_bands - 0.5],
        vmin=-vmax, vmax=vmax,
        interpolation='bilinear',
        origin='lower',
        zorder=1
    )

    # --- ECG overlay (normalize to [0.15, n_bands-1-0.15]) ---
    lo, hi = ecg_zoom.min(), ecg_zoom.max()
    if hi - lo > 1e-8:
        ecg_norm = (ecg_zoom - lo) / (hi - lo) * (n_bands - 1 - 0.3) + 0.15
    else:
        ecg_norm = np.full_like(ecg_zoom, (n_bands - 1) / 2.0)
    ax.plot(t_zoom, ecg_norm, color='black', linewidth=1.1, alpha=0.82, zorder=5)

    # --- Y-axis: show low-Hz value of each band (1, 8, 15, 22, ...), top→bottom ---
    import re as _re
    low_hz_labels = [_re.split(r'[\u2013\-]', lbl)[0].strip() for lbl in band_labels]
    last_high = _re.split(r'[\u2013\-]', band_labels[-1])[1].replace('Hz', '').strip()
    tick_positions = [i - 0.5 for i in range(n_bands)] + [n_bands - 0.5]
    tick_labels    = low_hz_labels + [last_high]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, color='#222222')
    ax.set_ylabel("Frequency Band (Hz)", color='#222222')
    ax.set_xlabel("Time (seconds)", color='#222222')
    ax.tick_params(colors='#333333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#888888')

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cbar.set_label(
        "DeepLIFT Score  (\u25b2 Red = Recurrence,  \u25bc Blue = Healthy)",
        color='#222222'
    )
    cbar.ax.yaxis.set_tick_params(color='#333333')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#333333')
    cbar.outline.set_visible(False)

    # --- Title ---
    title = (
        f"Attribution Spectrogram  |  Lead {lead_name}  |  Patient: {filename}\n"
        f"GT: {gt_text}  |  Pred: {pred_text} ({best_prob:.4f})"
    )
    ax.set_title(title, weight='bold', color='#111111', pad=10)

    plt.tight_layout()
    plot_path = os.path.join(patient_out_dir, f"{filename}_spectrogram_Lead_{lead_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {os.path.basename(plot_path)}")
    return plot_path


def preprocess_recording(recording, fs):
    b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    if fs == 1000:
        recording = signal.resample_poly(recording, up=1, down=2, axis=-1) # to 500Hz
        fs = 500
    elif fs == 500:
        pass
    else:
        recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
        fs = 500

    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)

    tensor_data = torch.tensor(recording, dtype=torch.float32)
    tensor_data = tensor_data.unsqueeze(0).unsqueeze(2)

    return tensor_data

def load_patient_data(header_path, rhythm_map, device):
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
        recording = preprocess_recording(recording, fs)

    recording = recording.to(device)
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
        
    windows_tensor = torch.cat(windows, dim=0).to(device)
    lead_indicator_expanded = lead_indicator.expand(windows_tensor.shape[0], -1).to(device)
    
    rhythm = rhythm_map.get(filename, 0)
    rhythm_vec = [1, -1] if rhythm == 1 else [-1, 1]
    r_tensor = torch.tensor([rhythm_vec], dtype=torch.float32).to(device)
    r_expanded = r_tensor.expand(windows_tensor.shape[0], -1)

    return windows_tensor, lead_indicator_expanded, r_expanded, label, filename, leads



def run_explainability(args):
    # Setup Device
    if torch.cuda.is_available():
        try:
            selector = DeviceSelector()
            device = selector.select(1)[0]
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load Model (Ensure EnsembleNN/NN classes are in scope)
    checkpoint = torch.load(args.model_path, map_location=device)
    if checkpoint.get('is_ensemble', False):
        num_models = checkpoint.get('num_models', 4)
        model = EnsembleNN(nOUT=2, num_models=num_models).to(device)
    else:
        model = NN(nOUT=2).to(device)
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load rhythm features
    rhythm_map = {}
    if os.path.exists(args.features_csv):
        features_df = pd.read_csv(args.features_csv)
        rhythm_map = dict(zip(features_df['ID'].astype(str), features_df['is_AFIB_before']))

    # Find patient header
    header_files = find_header_files(args.data_dir)
    if not header_files:
        raise FileNotFoundError(f"No header (.hea) files found in data directory: {args.data_dir}")

    selected_header = None
    if args.record_id:
        for h_file in header_files:
            if args.record_id in os.path.basename(h_file):
                selected_header = h_file
                break
        if selected_header is None:
            raise FileNotFoundError(
                f"Could not find any header (.hea) file containing '{args.record_id}' in '{args.data_dir}'.\n"
                f"Available files in this folder start with: {[os.path.basename(f) for f in header_files[:5]]}..."
            )
    else:
        selected_header = header_files[0]

    print(f"\nAnalyzing patient: {os.path.basename(selected_header)}")
    
    windows_tensor, lead_indicator_expanded, r_expanded, label, filename, present_leads = load_patient_data(
        selected_header, rhythm_map, device
    )

    # --- ENSEMBLE PREDICTION ---
    with torch.no_grad():
        y_ensemble = model(windows_tensor, lead_indicator_expanded, r_expanded)
        probs_ensemble = torch.softmax(y_ensemble, dim=1)

    # Determine predicted class from global mean across all windows
    mean_probs = probs_ensemble.mean(dim=0)
    predicted_class_global = 1 if mean_probs[1].item() >= 0.5 else 0

    # Select best window by predicted class (direct model output, no arithmetic)
    target_probs = probs_ensemble[:, predicted_class_global]
    best_win_idx = torch.argmax(target_probs).item()
    best_prob = target_probs[best_win_idx].item()   # confidence of predicted class

    prob_afib = probs_ensemble[best_win_idx, 1].item()
    gt_text   = "AFib Recurrence" if label == 1 else "Healthy"
    pred_text = "AFib Recurrence" if predicted_class_global == 1 else "Healthy"

    print(f"\n--- Patient Classification (Window {best_win_idx}) ---")
    print(f"Ensemble Prediction: Healthy={probs_ensemble[best_win_idx, 0]:.4f}, AFib Recurrence={probs_ensemble[best_win_idx, 1]:.4f}")
    print(f"Ground Truth Label: {label} ({gt_text})")

    # Isolate best window and enable gradients for input
    x_input = windows_tensor[best_win_idx:best_win_idx+1].clone().detach().requires_grad_(True)
    l_input = lead_indicator_expanded[best_win_idx:best_win_idx+1]
    r_input = r_expanded[best_win_idx:best_win_idx+1]

    sub_models = list(model.models) if isinstance(model, EnsembleNN) else [model]

    # -----------------------------------------------------------------------
    # SPECTROGRAM MODE
    # -----------------------------------------------------------------------
    if args.mode == 'spectrogram':
        print("\n--- Attribution Spectrogram Mode ---")
        bands = build_filter_bank(upper_hz=47)
        band_labels = [b[0] for b in bands]
        print(f"Filter bank: {len(bands)} bands → {', '.join(band_labels)}")

        t_bounds = [float(v) for v in args.spectrogram_window.split(",")]
        t_start, t_end = t_bounds[0], t_bounds[1]

        per_lead_matrices = run_spectrogram_attribution(
            x_input=x_input, l_input=l_input, r_input=r_input,
            sub_models=sub_models, target_class=1,
            bands=bands, fs=500, device=device
        )

        requested_leads = [lead.strip().upper() for lead in args.leads.split(",")]
        twelve_leads_upper = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        ecg_signal_np = x_input.squeeze(0).squeeze(1).detach().cpu().numpy()  # (12, T)
        time_axis = np.arange(4992) / 500.0

        patient_out_dir = os.path.join(args.output_dir, f"patient_{filename}")
        os.makedirs(patient_out_dir, exist_ok=True)

        for lead_name in requested_leads:
            if lead_name not in twelve_leads_upper:
                continue
            lead_idx = twelve_leads_upper.index(lead_name)
            if l_input[0, lead_idx].item() <= 0:
                print(f"  [Skip] Lead {lead_name} not present in recording.")
                continue

            plot_attribution_spectrogram(
                attr_matrix=per_lead_matrices[lead_idx],
                ecg_sig=ecg_signal_np[lead_idx, :],
                band_labels=band_labels,
                time_axis=time_axis,
                lead_name=lead_name,
                filename=filename,
                patient_out_dir=patient_out_dir,
                gt_text=gt_text, pred_text=pred_text,
                best_prob=best_prob,
                t_start=t_start, t_end=t_end
            )

        # Save raw matrices
        raw_path = os.path.join(patient_out_dir, f"{filename}_spectrogram_attributions.npy")
        np.save(raw_path, {
            "patient_id": filename, "true_label": label,
            "best_window_idx": best_win_idx, "ensemble_prob": best_prob,
            "band_labels": band_labels, "leads": present_leads,
            "ecg_signal": ecg_signal_np,
            "per_lead_matrices": {str(k): v for k, v in per_lead_matrices.items()}
        }, allow_pickle=True)
        print(f"Raw data saved to: {raw_path}\nDone!")
        return

    # -----------------------------------------------------------------------
    # CLASSIC TWO-PASS MODE
    # -----------------------------------------------------------------------
    # Dynamic spectral baselines
    patient_np = x_input.cpu().detach().numpy()

    # Split frequency: 16 Hz  (fs=500, Nyquist=250)
    # SLOW band  (<16 Hz): T-wave, ST segment, P-wave, baseline wander
    # FAST band  (>16 Hz): QRS main body, notches, late potentials
    #
    # FFT-based raised-cosine splitter:
    #   - Truly zero-phase (no phase distortion)
    #   - Perfectly complementary: SLOW + FAST = original (to machine precision)
    #   - Raised-cosine transition avoids Gibbs ringing of a brick-wall FFT cutoff
    _cutoff_hz   = 20.0
    _fs          = 500
    _n           = patient_np.shape[-1]
    _freqs       = np.fft.rfftfreq(_n, d=1.0 / _fs)        # shape (n//2+1,)
    _trans_hz    = 2.0                                      # ±2 Hz transition band

    # Build the smooth low-pass mask in the frequency domain
    _lp_mask = np.ones(len(_freqs), dtype=np.float64)
    for _i, _f in enumerate(_freqs):
        if _f > _cutoff_hz + _trans_hz:
            _lp_mask[_i] = 0.0
        elif _f > _cutoff_hz:
            _t = (_f - _cutoff_hz) / _trans_hz          # 0 → 1 across transition
            _lp_mask[_i] = 0.5 * (1.0 + np.cos(np.pi * _t))   # raised cosine

    _fft_data   = np.fft.rfft(patient_np, axis=-1)
    low_pass_np  = np.fft.irfft(_fft_data * _lp_mask,      n=_n, axis=-1).copy()
    high_pass_np = np.fft.irfft(_fft_data * (1.0 - _lp_mask), n=_n, axis=-1).copy()
    # Guarantee: low_pass_np + high_pass_np == patient_np (to float64 precision)

    low_pass_baseline  = torch.tensor(low_pass_np,  dtype=torch.float32).to(device)
    high_pass_baseline = torch.tensor(high_pass_np, dtype=torch.float32).to(device)

    requested_methods = [m.strip().lower() for m in args.method.split(",")]
    
    # Separate attribution trackers for the two passes
    attributions_fast = {m: [] for m in requested_methods}
    attributions_slow = {m: [] for m in requested_methods}
    sub_model_probs = []

    print("\n--- Computing Two-Pass Attributions (SmoothGrad) ---")
    for m_idx, sub_model in enumerate(sub_models):
        sub_model.eval() # Model params do not need requires_grad=True, saves memory

        with torch.no_grad():
            logits_m = sub_model(x_input, l_input, r_input)
            # Always weight by Recurrence confidence (class 1)
            prob_m = torch.softmax(logits_m, dim=1)[0, 1].item()
        sub_model_probs.append(prob_m)

        for method in requested_methods:
            try:
                # Initialize Explainer
                if method == "ig":
                    explainer = NoiseTunnel(IntegratedGradients(sub_model))
                    kwargs = {'n_steps': 20, 'internal_batch_size': 5}
                else:
                    explainer = NoiseTunnel(DeepLift(sub_model))
                    kwargs = {}

                # PASS 1: FAST FEATURES (Using Low-Pass Baseline)
                # Always target Recurrence (class 1): positive=red=Recurrence, negative=blue=Healthy
                attr_tensor_fast = explainer.attribute(
                    inputs=x_input, nt_type='smoothgrad', nt_samples=10, stdevs=0.05,
                    baselines=low_pass_baseline, target=1,
                    additional_forward_args=(l_input, r_input), **kwargs
                )
                
                # PASS 2: SLOW FEATURES (Using High-Pass Baseline)
                attr_tensor_slow = explainer.attribute(
                    inputs=x_input, nt_type='smoothgrad', nt_samples=10, stdevs=0.05,
                    baselines=high_pass_baseline, target=1,
                    additional_forward_args=(l_input, r_input), **kwargs
                )
                
                # Extract IG tuple if necessary, else just use the tensor
                map_fast = attr_tensor_fast[0] if isinstance(attr_tensor_fast, tuple) else attr_tensor_fast
                map_slow = attr_tensor_slow[0] if isinstance(attr_tensor_slow, tuple) else attr_tensor_slow

                attributions_fast[method].append((prob_m, map_fast.squeeze(0).squeeze(1).detach().cpu().numpy()))
                attributions_slow[method].append((prob_m, map_slow.squeeze(0).squeeze(1).detach().cpu().numpy()))
                
            except Exception as e:
                print(f"    [Warning] {method.upper()} failed on sub-model {m_idx+1}: {e}")

    print("\n--- Confidence-Weighted Aggregation ---")
    def aggregate_attrs(attr_dict):
        aggregated = {}
        for method, results in attr_dict.items():
            if len(results) == 0: continue
            total_weight = sum([prob for prob, _ in results])
            if total_weight > 1e-6:
                weighted_sum = np.zeros_like(results[0][1])
                for prob, attr_map in results: weighted_sum += prob * attr_map
                aggregated[method] = weighted_sum / total_weight
            else:
                aggregated[method] = np.mean([attr_map for _, attr_map in results], axis=0)
        return aggregated

    agg_fast = aggregate_attrs(attributions_fast)
    agg_slow = aggregate_attrs(attributions_slow)

    # --- CREATE PATIENT DIRECTORY ---
    patient_out_dir = os.path.join(args.output_dir, f"patient_{filename}")
    os.makedirs(patient_out_dir, exist_ok=True)

    requested_leads = [lead.strip().upper() for lead in args.leads.split(",")]
    twelve_leads_upper = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    ecg_signal_np = x_input.squeeze(0).squeeze(1).detach().cpu().numpy()
    time_axis = np.arange(4992) / 500.0

    # Two rows per figure: FAST (>16 Hz) and SLOW (<16 Hz)
    # Both computed with target=1 (Recurrence): red=Recurrence, blue=Healthy
    band_configs = [
        ("FAST", agg_fast),
        ("SLOW", agg_slow),
    ]

    valid_leads = [
        (lead, twelve_leads_upper.index(lead))
        for lead in requested_leads
        if lead in twelve_leads_upper and l_input[0, twelve_leads_upper.index(lead)].item() > 0
    ]

    for method in requested_methods:
        if not any(method in d for _, d in band_configs):
            continue
        for lead_name, lead_idx in valid_leads:
            sig = ecg_signal_np[lead_idx, :]
            fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
            fig.patch.set_facecolor('white')

            for ax, (band_name, attr_dict) in zip(axes, band_configs):
                if method not in attr_dict:
                    ax.set_visible(False)
                    continue
                plot_attr = gaussian_filter1d(attr_dict[method][lead_idx, :], sigma=20)
                vmax = max(np.percentile(np.abs(plot_attr), 98), 1e-8)
                norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

                ax.set_facecolor('white')
                ax.plot(time_axis, sig, color='black', linewidth=1.0, zorder=5)
                extent = [time_axis[0], time_axis[-1], sig.min() - 0.5, sig.max() + 0.5]
                im = ax.imshow(
                    plot_attr[np.newaxis, :], cmap='RdBu_r', norm=norm,
                    aspect='auto', extent=extent, alpha=0.7, zorder=1, interpolation='bilinear'
                )
                ax.set_xlim(0.0, 10.0)
                ax.set_ylim(sig.min() - 0.4, sig.max() + 0.4)
                ax.set_ylabel(band_name, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cccccc')
                ax.spines['bottom'].set_color('#cccccc')
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.01, shrink=0.8)
                cbar.set_label("▲ Recurrence  ▼ Healthy", color='#333333')
                cbar.set_ticks([-vmax, -vmax/2, 0, vmax/2, vmax])
                cbar.set_ticklabels([
                    f"{-vmax:.2e}", f"{-vmax/2:.2e}", "0",
                    f"{vmax/2:.2e}", f"{vmax:.2e}"
                ], color='#333333')
                cbar.outline.set_visible(False)

            axes[-1].set_xlabel("Time (seconds)")
            # Show sub-model confidence for the same class as the ensemble display
            if predicted_class_global == 0:  # predicted Healthy
                sub_probs_str = ", ".join([f"M{i+1}:{1-p:.2f}" for i, p in enumerate(sub_model_probs)])
            else:  # predicted Recurrence
                sub_probs_str = ", ".join([f"M{i+1}:{p:.2f}" for i, p in enumerate(sub_model_probs)])
            main_title = (
                f"Lead {lead_name}  |  {method.upper()}  |  Patient: {filename}\n"
                f"GT: {gt_text}  |  Pred: {pred_text}  |  "
                f"Max-Window Conf ({pred_text}): {best_prob:.4f} ({sub_probs_str})"
            )
            fig.suptitle(main_title, weight='bold', y=1.01)
            plt.tight_layout()
            plot_path = os.path.join(patient_out_dir, f"{filename}_{method}_Lead_{lead_name}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  Saved plot: {os.path.basename(plot_path)}")

    # Save Raw Data
    raw_data_path = os.path.join(patient_out_dir, f"{filename}_raw_attributions.npy")
    np.save(raw_data_path, {
        "patient_id": filename, "true_label": label, "best_window_idx": best_win_idx,
        "ensemble_prob": best_prob, "sub_model_probs": sub_model_probs, "leads": present_leads,
        "ecg_signal": ecg_signal_np, "agg_fast": agg_fast, "agg_slow": agg_slow
    }, allow_pickle=True)
    print(f"Data exported to: {raw_data_path}\nDone!")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--record_id", type=str, default=None)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--method", type=str, default="ig,deeplift")
    parser.add_argument("--leads", type=str, default="I,II,V2")
    parser.add_argument("--features_csv", type=str, default="/srv/home/jhyl/Afib_recurrence/features.csv")
    # Spectrogram mode
    parser.add_argument(
        "--mode", type=str, default="twopass",
        choices=["twopass", "spectrogram"],
        help="'twopass' = classic FAST/SLOW attribution (default). "
             "'spectrogram' = attribution spectrogram across frequency bands."
    )
    parser.add_argument(
        "--spectrogram_window", type=str, default="4.0,6.0",
        help="Time window to display in the spectrogram plot, e.g. '4.0,6.0' (seconds)."
    )
    args = parser.parse_args()
    run_explainability(args)