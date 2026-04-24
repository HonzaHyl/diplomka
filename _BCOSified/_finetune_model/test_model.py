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
    print(f'[{datetime.datetime.now()}] Finding header and recording files...', flush=True)
    header_files = find_header_files(test_data_dir)
    print(f'[{datetime.datetime.now()}] Found {len(header_files)} header files.', flush=True)

    # Load model
    PTH_PATH = os.path.join(model_dir, f"FINAL_MODEL_{MODEL_ID}.pth")
    PROGRESS_PATH = os.path.join(model_dir, f"PROGRESS_{MODEL_ID}.pickle")
    
    model = NN(nOUT=26).to(DEVICE)
    model = finetune_model_prep(model)

    if os.path.exists(PTH_PATH):
        print(f'[{datetime.datetime.now()}] Loading weights from {PTH_PATH}...', flush=True)
        model.load_state_dict(torch.load(PTH_PATH, map_location=DEVICE))
    elif os.path.exists(PROGRESS_PATH):
        print(f'[{datetime.datetime.now()}] Loading model history from {PROGRESS_PATH}... (This might take a while if the file is large)', flush=True)
        with open(PROGRESS_PATH, "rb") as handle:
            models = pickle.load(handle)
        print(f'[{datetime.datetime.now()}] Pickle loaded successfully.', flush=True)
        model.load_state_dict(models[-1]["model"]) # Use the last epoch's model
    else:
        raise FileNotFoundError(f"Neither {PTH_PATH} nor {PROGRESS_PATH} found in {model_dir}")

    model.eval()

    print(f'[{datetime.datetime.now()}] Starting inference loop...', flush=True)

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
    num_files = len(header_files)
    for i, header_path in enumerate(header_files):
        if i % 10 == 0:
            print(f'[{datetime.datetime.now()}] Processing file {i+1}/{num_files}: {header_path}', flush=True)
        # Path to recording file
        recording_path = header_path.replace(".hea", ".mat")
        filename, extension = os.path.splitext(recording_path)
        filename = filename.split("/")[-1]

        # Load recording
        recording = load_recording(recording_path)
        
        # Get info from header
        header = load_header(header_path)
        leads = get_leads(header)
        fs = get_frequency(header)
        label = int(get_labels(header)[0])

        # Expand recording to 12 leads
        recording, lead_indicator = expand_leads(recording, leads)
        
        # B-cosification 6-channel equivalent:
        lead_indicator = np.concatenate([lead_indicator, lead_indicator], axis=0)
        
        lead_indicator = torch.tensor(lead_indicator, dtype=torch.float32).to(DEVICE)
        lead_indicator = lead_indicator.unsqueeze(0)
        # Preprocess recording (filters, downsampling, standardization)
        recording = preprocessing(recording, fs).to(DEVICE)
        # Infer recordings
        y = model(recording, lead_indicator)
        # Gain probabilities
        p = torch.softmax(y, dim=1)
        temp_dict["softmax_probas"][filename] = p.tolist()

        outputs.append(p.data.cpu().numpy())
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
    mlflow.log_metrics(temp_dict)

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

    # B-cosification 6-channel equivalent:
    recording = np.concatenate([recording, 1.0 - recording], axis=0)

    # Signal padding
    maxL = 149952
    padded_data = np.zeros((recording.shape[0], maxL))

    if recording.shape[1] > maxL:
        padded_data = recording[:, :maxL].copy()
    else:
        padded_data[:, :recording.shape[1]] = recording
    
    padded_tensor = torch.tensor(padded_data, dtype=torch.float32)
    padded_tensor = padded_tensor.unsqueeze(0)
    padded_tensor = padded_tensor.unsqueeze(2)

    return padded_tensor

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

    #################### MlFlow Setup ####################
    USE_ARG = False

    if USE_ARG == True:
        # Parse arguments.
        if len(sys.argv) != 4:
            raise Exception('Include the model, data, and output folders as arguments, e.g., python test_model.py model data outputs.')

        model_directory = sys.argv[1]
        data_directory = sys.argv[2]
        output_directory = sys.argv[3]
    else:
        model_directory = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/finetune_run/model"
        data_directory = "/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/finetune_run/test_data"
        output_directory = os.path.join(run_dir, "test_outputs")

    print('Starting main script...', flush=True)
    test_model(model_directory, data_directory, output_directory)
    print('Done.', flush=True)