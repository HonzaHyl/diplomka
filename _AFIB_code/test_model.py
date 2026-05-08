from operator import pos
from helper_code import find_header_files, _load_model, load_header, get_leads, get_frequency, get_labels, load_recording, expand_leads, finetune_model_prep
import scipy.signal as signal
from scipy.stats import zscore
import numpy as np
import torch
import os
import yaml
import sys
import pickle
import torch
import mlflow
import datetime

from device_selector import DeviceSelector

from model_structure import NN, EnsembleNN

selector = DeviceSelector()
DEVICE = selector.select(1)[0]

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, balanced_accuracy_score

MODEL_ID = "finetuned"

def test_model(model_path, test_data_dir, output_dir):

    # Load .hea of test data
    print('Finding header and recording files...')
    header_files = find_header_files(test_data_dir)

    # Load model
    print(f'Loading model from {model_path}...')
    # Check if the checkpoint is an ensemble
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

    temp_dict = {"precision":0,
                 "recall":0,
                 "confusion_matrix":0,
                 "f1":0,
                 "accuracy":0,
                 "auprc":0,
                 "auroc":0,
                 "softmax_probas":{}}
 
    # Preprocess recording and run inference
    for header_path in header_files:
        recording_path_mat = header_path.replace(".hea", ".mat")
        recording_path_npy = header_path.replace(".hea", ".npy")
        filename = os.path.basename(header_path).replace(".hea", "")

        # Get info from header
        header = load_header(header_path)
        leads = get_leads(header)
        fs = get_frequency(header)
        
        # The dataset has Healthy = 1, Recurrence = 0.
        # Keeping it exactly as it is.
        label = int(get_labels(header)[0])

        # Check if preprocessed .npy file exists
        if os.path.exists(recording_path_npy):
            recording = np.load(recording_path_npy)
            
            # Expand leads to get the lead_indicator 
            # (assuming the .npy itself already has 12 rows, we just need the indicator)
            _, lead_indicator = expand_leads(np.zeros((12, 1)), leads)
            lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32).unsqueeze(0)
            
            # .npy is already preprocessed. Ensure it's the right tensor shape.
            tensor_data = torch.tensor(recording, dtype=torch.float32)
            if len(tensor_data.shape) == 2:
                tensor_data = tensor_data.unsqueeze(0).unsqueeze(2)
            elif len(tensor_data.shape) == 3:
                tensor_data = tensor_data.unsqueeze(2)
            recording = tensor_data
        else:
            # Load .mat recording
            recording = load_recording(recording_path_mat)
            
            # Expand recording to 12 leads
            recording, lead_indicator = expand_leads(recording, leads)
            lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32).unsqueeze(0)
            
            # Preprocess recording (filters, downsampling, standardization)
            recording = preprocessing(recording, fs)

        # Infer recordings using sliding windows (matching validation!)
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
        
        with torch.no_grad():
            y = model(windows_tensor, lead_indicator_expanded)
            p = torch.softmax(y, dim=1)
        
        # Patient-level prediction (Mean across windows)
        # The model was trained with Healthy = 1, Recurrence = 0.
        # So p[:, 1] is the probability of Healthy (the positive class).
        prob_recurrence = p[:, 0].mean().item()
        prob_healthy = p[:, 1].mean().item()
        patient_p = np.array([[prob_recurrence, prob_healthy]])
        
        temp_dict["softmax_probas"][filename] = {
            "probabilities": patient_p.tolist()[0],
            "ground_truth": label
        }

        outputs.append(patient_p)
        targets.append(label)

    # Concatenate all batch results into single numpy arrays
    # targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)


    positive_class_probs = outputs[:, 1]
    targets_np = np.array(targets)

    # AUPRC is highly sensitive to class imbalance. 
    # We calculate it for both the Healthy class (1) and Recurrence class (0).
    temp_dict["auprc_healthy"] = float(average_precision_score(y_true=targets_np, y_score=outputs[:, 1]))
    temp_dict["auprc_recurrence"] = float(average_precision_score(y_true=1-targets_np, y_score=outputs[:, 0]))
    
    # AUROC is symmetric, so one is enough
    temp_dict["auroc"] = float(roc_auc_score(y_true=targets_np, y_score=positive_class_probs))

    y_pred = (positive_class_probs >= 0.5).astype(int)

    temp_dict["confusion_matrix"] = confusion_matrix(targets_np, y_pred).tolist()

    # Macro averages treat both classes equally, preventing the majority class from dominating the score.
    temp_dict["f1_macro"] = float(f1_score(targets_np, y_pred, average="macro"))
    temp_dict["precision_macro"] = float(precision_score(targets_np, y_pred, average="macro", zero_division=0))
    temp_dict["recall_macro"] = float(recall_score(targets_np, y_pred, average="macro"))
    
    temp_dict["accuracy"] = float(accuracy_score(targets_np, y_pred))
    temp_dict["balanced_accuracy"] = float(balanced_accuracy_score(targets_np, y_pred))

    output_file_path = os.path.join(output_dir, "model_"+MODEL_ID+".yaml")

    with open(output_file_path, "w") as f:
        yaml.dump(temp_dict, f, default_flow_style=None, sort_keys=False)
        

    

def preprocessing(recording, fs):
    b,a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    if fs==1000:
        recording = signal.resample_poly(recording, up=1, down=2, axis=-1) # to 500Hz
        fs = 500
    elif fs==500:
        pass
    else:
        recording = signal.resample(recording, int(recording.shape[1] * 500 / fs), axis=1)
        print(f'RESAMPLING FROM {fs} TO 500')
        fs = 500

    recording = signal.filtfilt(b, a, recording)
    recording = zscore(recording, axis=-1)
    recording = np.nan_to_num(recording)

    # We do NOT pad to 149952 here. We keep its true length to avoid feeding blank zeroes into the model windows!
    tensor_data = torch.tensor(recording, dtype=torch.float32)
    tensor_data = tensor_data.unsqueeze(0)
    tensor_data = tensor_data.unsqueeze(2)

    return tensor_data

if __name__ == '__main__':

    #################### MlFlow Setup ####################
    db_url = "sqlite:////mnt/mdpm/d03/jhyl/deepstem_results/mlflow_runs.db"
    experiment_name = "BCOSified_finetuning_focal"
    artifact_path = "file:///mnt/mdpm/d03/jhyl/Afib_recurrence/diplomka/results/mlruns"

    mlflow.set_tracking_uri(db_url)

    try:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_path)
    except mlflow.exceptions.MlflowException:
        pass # Experiment already exists
    mlflow.set_experiment(experiment_name)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Testing_{timestamp}"
    run_dir = os.path.join("/srv/home/jhyl/Afib_recurrence/diplomka/results", run_name)
    os.makedirs(os.path.join(run_dir, "test_outputs"), exist_ok=True)
    print(f"[INFO] Run directory created: {run_dir}")
    mlflow.start_run(run_name=run_name)

    print(f"[INFO] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[INFO] Experiment ID: {mlflow.get_experiment_by_name(experiment_name).experiment_id}")

    USE_ARG = False

    if USE_ARG == True:
        # Parse arguments.
        if len(sys.argv) != 4:
            raise Exception('Include the model path, data folder, and output folder as arguments, e.g., python test_model.py model.pth data outputs.')

        model_path = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
    else:
        model_path = "/srv/home/jhyl/Afib_recurrence/diplomka/results/Trial_45/ensemble_model.pth"
        data_directory = "/srv/home/jhyl/Afib_recurrence/diplomka/finetune_data/SR_before/test"
        output_directory = os.path.join(run_dir, "test_outputs")

    print('Starting main script...', flush=True)
    test_model(model_path, data_directory, output_directory)
    print('Done.', flush=True)