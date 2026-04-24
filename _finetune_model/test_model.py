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

from model_structure import NN

selector = DeviceSelector()
DEVICE = selector.select(1)[0]

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

MODEL_ID = "finetuned"

def test_model(model_dir, test_data_dir, output_dir):

    # Load .hea of test data
    print('Finding header and recording files...')
    header_files = find_header_files(test_data_dir)

    # Load model
    # Load the best checkpoint from your resumed training run
    checkpoint_path = os.path.join(model_dir, "checkpoint_epoch_106.pth")  # You can adjust this file name

    model = NN(nOUT=2).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
        # Path to recording file
        recording_path = header_path.replace(".hea", ".mat")
        filename, extension = os.path.splitext(recording_path)
        filename = filename.split("/")[-1]

        # Load recording
        recording = load_recording(recording_path)
        
        # Get info from header
        header = load_header(header_path)
        leads = get_leads(header)
        fs = get_frequency(header)
        label = int(get_labels(header)[0])

        # Expand recording to 12 leads
        recording, lead_indicator = expand_leads(recording, leads)
        lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32)
        lead_indicator = lead_indicator.unsqueeze(0)
        # Preprocess recording (filters, downsampling, standardization)
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
        mean_prob_positive = p[:, 1].mean().item()
        prob_negative = 1.0 - mean_prob_positive
        patient_p = np.array([[prob_negative, mean_prob_positive]])
        
        temp_dict["softmax_probas"][filename] = patient_p.tolist()

        outputs.append(patient_p)
        targets.append(label)

    # Concatenate all batch results into single numpy arrays
    # targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)


    positive_class_probs = outputs[:, 1]

    temp_dict["auprc"] = float(average_precision_score(y_true=targets, y_score=positive_class_probs))
    temp_dict["auroc"] = float(roc_auc_score(y_true=targets, y_score=positive_class_probs))

    y_pred = (positive_class_probs >= 0.6).astype(int)

    temp_dict["confusion_matrix"] = confusion_matrix(targets, y_pred).tolist()

    temp_dict["f1"] = float(f1_score(targets, y_pred))
    temp_dict["precision"] = float(precision_score(targets, y_pred))
    temp_dict["recall"] = float(recall_score(targets, y_pred))
    temp_dict["accuracy"] = float(accuracy_score(targets, y_pred))

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
    experiment_name = "BCOSified_finetuning"
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
            raise Exception('Include the model, data, and output folders as arguments, e.g., python test_model.py model data outputs.')

        model_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
    else:
        model_directory = "/srv/home/jhyl/Afib_recurrence/diplomka/results/Training_20260324_134204/model"
        data_directory = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/finetune_run/test_data"
        output_directory = os.path.join(run_dir, "test_outputs")

    print('Starting main script...', flush=True)
    test_model(model_directory, data_directory, output_directory)
    print('Done.', flush=True)