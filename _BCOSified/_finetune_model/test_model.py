import os
import torch
import yaml
import sys
import pickle
import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.stats import zscore
import mlflow

from helper_code import (
    find_header_files, _load_model, load_header, get_leads, 
    get_frequency, get_labels, load_recording, expand_leads, 
    finetune_model_prep
)
from model_structure import NN, EnsembleNN
from device_selector import DeviceSelector

selector = DeviceSelector()
DEVICE = selector.select(1)[0]

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, 
    recall_score, accuracy_score, f1_score, confusion_matrix
)

MODEL_ID = "finetuned"
FEATURES_CSV = "/srv/home/jhyl/Afib_recurrence/features.csv"
WINDOW_SIZE = 4992
STEP_SIZE   = 2496

def test_model(model_dir, test_data_dir, output_dir):
    # Load patient rhythm info for context vector r
    print(f'[{datetime.datetime.now()}] Loading rhythm features...', flush=True)
    if os.path.exists(FEATURES_CSV):
        features_df = pd.read_csv(FEATURES_CSV)
        rhythm_map = dict(zip(features_df['ID'].astype(str), features_df['is_AFIB_before']))
    else:
        raise FileNotFoundError(f"Features file not found at {FEATURES_CSV}")

    # Load .hea of test data
    print(f'[{datetime.datetime.now()}] Finding header and recording files...', flush=True)
    header_files = find_header_files(test_data_dir)
    print(f'[{datetime.datetime.now()}] Found {len(header_files)} header files.', flush=True)

    # 1. Load model using the new BN-folding pipeline
    # Note: _load_model now handles folding and remapping automatically.
    print(f'[{datetime.datetime.now()}] Loading B-cosified model from {model_dir}...', flush=True)
    
    # Check if we are loading an ensemble or a single model
    ensemble_path = os.path.join(model_dir, "ensemble_bcos_model.pth")
    if os.path.exists(ensemble_path):
        print(f"[INFO] Detected Ensemble model at {ensemble_path}")
        checkpoint = torch.load(ensemble_path, map_location=DEVICE)
        num_models = checkpoint.get('num_models', 4)
        model = EnsembleNN(nOUT=2, num_models=num_models)
        for m in model.models:
            finetune_model_prep(m)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback to single model loading (Fold 1 default for testing)
        loaded_data = _load_model(model_dir, 1, nOUT=2)
        model = loaded_data["classifier"]
        model = finetune_model_prep(model)
        
    model.to(DEVICE)
    model.eval()

    print(f'[{datetime.datetime.now()}] Starting inference loop...', flush=True)

    outputs = []
    targets = []
    softmax_probas = {}

    num_files = len(header_files)
    for i, header_path in enumerate(header_files):
        if i % 10 == 0:
            print(f'[{datetime.datetime.now()}] Processing file {i+1}/{num_files}: {header_path}', flush=True)
        
        record_id = os.path.basename(header_path).replace(".hea", "")
        
        # Load recording and header
        recording_path = header_path.replace(".hea", ".mat")
        recording = load_recording(recording_path)
        header = load_header(header_path)
        leads = get_leads(header)
        fs = get_frequency(header)
        label = int(get_labels(header)[0])

        # 1. Preprocess: Resample, Filter, Z-score, 12-lead expansion
        # (Using logic from preprocess_finetune_data.py)
        if fs != 500:
            recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
        
        b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')
        recording = signal.filtfilt(b, a, recording)
        
        recording, lead_indicator = expand_leads(recording, leads)
        
        mu = np.nanmean(recording, axis=-1, keepdims=True)
        std = np.nanstd(recording, axis=-1, keepdims=True) + 1e-8
        recording = (recording - mu) / std
        recording = np.nan_to_num(recording).astype(np.float32)

        # 2. Rhythm vector r
        is_afib = rhythm_map.get(record_id, None)
        if is_afib is None:
            print(f"[WARNING] Patient {record_id} not in features.csv. Defaulting to Healthy.")
            is_afib = 0
        
        if is_afib == 1:
            r_vec = np.array([1, -1], dtype=np.float32)
        else:
            r_vec = np.array([-1, 1], dtype=np.float32)
        
        r_tensor = torch.from_numpy(r_vec).unsqueeze(0).to(DEVICE)
        l_tensor = torch.from_numpy(lead_indicator).float().unsqueeze(0).to(DEVICE)

        # 3. Windowed Inference (matches main_code.py validation)
        seq_len = recording.shape[1]
        
        # Pad to multiple of 64
        remainder = seq_len % 64
        if remainder != 0:
            pad_len = 64 - remainder
            recording = np.pad(recording, ((0, 0), (0, pad_len)), mode='constant')
            seq_len = recording.shape[1]

        windows = []
        for start in range(0, seq_len - WINDOW_SIZE + 1, STEP_SIZE):
            windows.append(recording[:, start : start + WINDOW_SIZE])
        if len(windows) == 0:
            windows.append(recording[:, -WINDOW_SIZE:])
        
        windows_np = np.stack(windows, axis=0) # [NumWindows, 12, WINDOW_SIZE]
        
        # B-cosification (24 channels)
        pos = np.maximum(windows_np, 0)
        neg = np.maximum(-windows_np, 0)
        windows_bcos = np.concatenate([pos, neg], axis=1) # [NumWindows, 24, WINDOW_SIZE]
        
        windows_t = torch.from_numpy(windows_bcos).float().unsqueeze(2).to(DEVICE) # [NumWindows, 24, 1, WINDOW_SIZE]
        
        l_expanded = l_tensor.expand(windows_t.size(0), -1)
        r_expanded = r_tensor.expand(windows_t.size(0), -1)

        with torch.no_grad():
            y = model(windows_t, l_expanded, r_expanded)
            # Average (AVG) Pooling for patient-level prediction
            p = torch.softmax(y, dim=1).mean(dim=0, keepdim=True)

        softmax_probas[record_id] = p.tolist()
        outputs.append(p.data.cpu().numpy())
        targets.append(label)

    outputs = np.concatenate(outputs, axis=0)
    targets = np.array(targets)

    # Evaluation Metrics
    # Remap for sklearn: internally 0=recurrence, 1=healthy.
    # Sklearn assumes 1=positive, so flip: 1=recurrence, 0=healthy.
    targets_sk = 1 - targets
    pos_probs = outputs[:, 0] # P(recurrence)
    predictions = (1 - np.argmax(outputs, axis=1))

    results = {
        "auprc": float(average_precision_score(y_true=targets_sk, y_score=pos_probs)),
        "auroc": float(roc_auc_score(y_true=targets_sk, y_score=pos_probs)),
        "f1": float(f1_score(y_true=targets_sk, y_pred=predictions)),
        "precision": float(precision_score(y_true=targets_sk, y_pred=predictions)),
        "recall": float(recall_score(y_true=targets_sk, y_pred=predictions)),
        "accuracy": float(accuracy_score(y_true=targets_sk, y_pred=predictions)),
        "confusion_matrix": confusion_matrix(y_true=targets_sk, y_pred=predictions).tolist(),
        "softmax_probas": softmax_probas
    }

    mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, (int, float))})

    output_file_path = os.path.join(output_dir, "model_evaluation_results.yaml")
    with open(output_file_path, "w") as f:
        yaml.dump(results, f, default_flow_style=None, sort_keys=False)
    
    print(f'[{datetime.datetime.now()}] Evaluation complete. Results saved to {output_file_path}')

if __name__ == '__main__':
    # MLflow Setup
    db_url = "sqlite:////mnt/mdpm/d03/jhyl/deepstem_results/mlflow_runs.db"
    experiment_name = "BCOSified_testing"
    artifact_path = "file:///mnt/mdpm/d03/jhyl/Afib_recurrence/diplomka/results/mlruns"

    mlflow.set_tracking_uri(db_url)
    try:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_path)
    except:
        pass
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Testing_{timestamp}"
    mlflow.start_run(run_name=run_name)

    model_dir = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model/Trial_34/"
    test_data_dir = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/finetune_run/test_data" # Update if needed
    output_dir = "/srv/home/jhyl/Afib_recurrence/diplomka/results/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    test_model(model_dir, test_data_dir, output_dir)
    mlflow.end_run()