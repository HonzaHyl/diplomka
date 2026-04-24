import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn


import numpy as np
import pandas as pd
import os
import time
import copy
import pickle
import warnings
import mlflow


from scipy import signal
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix
from pathlib import Path

# Imports from local scripts
from helper_code import load_header, get_nsamp, get_leads, get_sex, get_frequency, find_header_files
from helper_code import get_labels, lead_exctractor, load_recording, expand_leads, _load_model, finetune_model_prep

from device_selector import DeviceSelector

selector = DeviceSelector()
DEVICE = selector.select(1)[0]

# --- Global Window Parameters (500Hz) ---
WINDOW_SIZE = 15040  # 30.08 seconds (Multiple of 64)
STEP_SIZE   = 10048  # 20.096 second stride (9.984 second overlap)

CONFIG = {"learning_rate": 1e-5,
          "LR_scheduler": "StepLR",
          "step_size": 20,
          "gamma": 0.75,
          "optimizer": "AdamW",
          "weight_decay": 1e-2,
          "epochs": 30,
          "layer_configs": {
              "conv": {"trainable": False, "lr": 1e-7},
              "bn":   {"trainable": False, "lr": 1e-7},
              "rb_0": {"trainable": False, "lr": 1e-7},
              "rb_1": {"trainable": False, "lr": 1e-7},
              "rb_2": {"trainable": False, "lr": 1e-7},
              "rb_3": {"trainable": True,  "lr": 1e-6},
              "rb_4": {"trainable": True,  "lr": 1e-6},
              "fc_1": {"trainable": True,  "lr": 1e-4},
          }
          }

# Class that creates custom dataset from given data
class CustomDataset(Dataset):
    def __init__(self, header_paths, window_size=WINDOW_SIZE, step=STEP_SIZE):
        super().__init__()
        self.files = list()
        self.is_train = True
        self.window_size = window_size
        self.step_size = step
        self.num_leads = 12 

        for path in header_paths:
            temp_dict = dict()
            temp_dict["header"] = path
            
            # Record is now the .npy file
            npy_path = path.replace(".hea", ".npy")
            temp_dict["npy_path"] = npy_path

            # Load target from header
            header = load_header(path)
            label = int(get_labels(header)[0])
            target_vector = np.zeros(2, dtype=int)
            target_vector[label] = 1
            temp_dict['target'] = target_vector
            
            # Map windows using the pre-processed .npy shape
            if os.path.exists(npy_path):
                data_mmap = np.load(npy_path, mmap_mode='r')
                sig_len = data_mmap.shape[1]
                start_indices = np.array(range(0, sig_len - self.window_size + 1, self.step_size))
                temp_dict["start_indices"] = start_indices
                self.files.append(temp_dict)
            else:
                print(f"Warning: {npy_path} not found.")

        self.files_df = pd.DataFrame(self.files)
        self.window_map = [(i, start) for i, row in self.files_df.iterrows() for start in row['start_indices']]

    def summary(self, output):
        targets = np.stack(self.files_df['target'].to_list(), axis=0)
        if output == 'pandas':
            return pd.Series(targets.sum(axis=0))
        if output == 'numpy':
            return targets.sum(axis=0)

    def train_valid_split(self, test_size):
        files = self.files_df['header'].to_numpy().reshape(-1,1)
        targets = np.stack(self.files_df['target'].to_list(), axis=0)

        x_train, y_train, x_valid, y_valid = iterative_train_test_split(files, targets, test_size=test_size)

        train = CustomDataset(header_paths=x_train[:,0].tolist(), window_size=self.window_size, step=self.step_size)
        train.is_train = True

        valid = CustomDataset(header_paths=x_valid[:,0].tolist(), window_size=self.window_size, step=self.step_size)
        valid.is_train = False

        return train, valid

    def __len__(self):
        return len(self.window_map) if self.is_train else len(self.files_df)

    def __getitem__(self, index):
        if self.is_train:
            # --- TRAINING: 10-second slices ---
            sig_idx, window_start = self.window_map[index]
            row = self.files_df.iloc[sig_idx]
            
            # Fast load and slice
            data = np.load(row['npy_path'], mmap_mode='r')
            window = data[:, window_start : window_start + self.window_size].copy()
            
            # B-cosification
            pos = np.maximum(window, 0)
            neg = np.maximum(-window, 0)
            window_bcos = np.concatenate([pos, neg], axis=0)

            return torch.from_numpy(window_bcos).float(), torch.from_numpy(row['target']).float(), torch.ones(24)

        else:
            # --- VALIDATION: Full length signal ---
            row = self.files_df.iloc[index]
            data = np.load(row['npy_path']).astype(np.float32)

            # Pad length to be a perfect multiple of 64
            seq_len = data.shape[1]
            remainder = seq_len % 64
            if remainder != 0:
                pad_len = 64 - remainder
                data = np.pad(data, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
            
            # B-cosification
            pos = np.maximum(data, 0)
            neg = np.maximum(-data, 0)
            data_bcos = np.concatenate([pos, neg], axis=0)

            return torch.from_numpy(data_bcos).float(), torch.from_numpy(row['target']).float(), torch.ones(24)

# ==========================================
# DATALOADER COLLATION
# ==========================================
def collate_fn(batch):
    # If training (b[0] is 2D: [C, L]), stack them
    X = torch.stack([b[0] for b in batch], dim=0)
    t = torch.stack([b[1] for b in batch], dim=0)
    l = torch.stack([b[2] for b in batch], dim=0)
    
    return X, t, l

# ==========================================
# TRAINING LOOP
# ==========================================
def training_code(data_directory, model_directory):
    _training_code(data_directory, model_directory, "finetuned")

def _training_code(data_directory, model_directory, ensamble_ID):

    print('Finding header and recording files...')
    header_files = find_header_files(data_directory)

    full_dataset = CustomDataset(header_files)
    train, valid = full_dataset.train_valid_split(test_size=0.1)
    print("Successfully created train and valid dataset...")

    train_files = [k.split('/')[-1] for k in train.files_df['header'].to_list()]
    valid_files = [k.split('/')[-1] for k in valid.files_df['header'].to_list()]

    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    # Validation MUST have batch_size=1 to process one patient's windows at a time
    valid_loader = DataLoader(dataset=valid, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    loaded_model = _load_model("/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model/", 1, nOUT=26)
    classifier = loaded_model["classifier"]
    
    from helper_code import set_requires_grad
    
    model = finetune_model_prep(classifier)
    
    # 1. Apply layer-wise freezing configuration
    layers_trainable = {name: config["trainable"] for name, config in CONFIG["layer_configs"].items()}
    set_requires_grad(model, layers_trainable)
    
    model = model.to(DEVICE)

    # Verify trainable parameters
    print("\n--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"UNCAGED: {name}")
    print("---------------------------\n")

    # --- NEW: Calculate Class Weights ---
    # Get the total number of samples per class in the training set
    class_counts = train.summary('numpy') 
    print(class_counts)
    
    # Calculate inverse frequencies (e.g., if counts are [200, 50], weights become [0.005, 0.02])
    weights = 1.0 / class_counts 
    
    # Normalize weights so they sum to 1 (optional, but keeps learning rate stable)
    weights = weights / weights.sum() 
    
    # Convert to tensor and move to device
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    
    # Initialize the loss function here, so we can pass it down
    # Using label smoothing to help prevent overconfidence in the majority class
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    # ------------------------------------

    # Note: Using AdamW with flexible differential learning rates
    param_groups = []
    for layer_name, config in CONFIG["layer_configs"].items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            # Only add to optimizer if it contains trainable parameters
            if any(p.requires_grad for p in layer.parameters()):
                param_groups.append({'params': layer.parameters(), 'lr': config['lr']})
        else:
            print(f"Warning: Layer '{layer_name}' not found in model.")

    if not param_groups:
        print("Warning: No trainable parameters found! Everything is frozen.")
        # Fallback to all parameters if everything is frozen, to avoid optimizer error 
        # (though training won't do much)
        param_groups = [{'params': model.parameters(), 'lr': 1e-7}]

    opt = optim.AdamW(param_groups, weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=CONFIG["step_size"], gamma=CONFIG["gamma"])
    
    # Prepare and log parameters to MLflow
    # Flatten layer_configs to ensure they are visible and searchable in the MLflow UI
    mlflow_params = {k: v for k, v in CONFIG.items() if k != "layer_configs"}
    for layer_name, config in CONFIG["layer_configs"].items():
        mlflow_params[f"layer_{layer_name}_trainable"] = config["trainable"]
        mlflow_params[f"layer_{layer_name}_lr"] = config["lr"]
    
    mlflow.log_params(mlflow_params)

    OUTPUT = []
    EPOCHS = CONFIG["epochs"]
    for epoch in range(EPOCHS):
        print(f"============================[{epoch}]============================")
        
        # Train
        train_auprc, train_auroc, train_f1, train_cm = train_part(model, train_loader, opt, loss_fn)
        print(f"Train | AUPRC: {train_auprc:.4f} | AUROC: {train_auroc:.4f} | F1: {train_f1:.4f}")
        
        # Validate
        valid_auprc, valid_auroc, valid_f1, valid_cm, valid_targets, valid_outputs, best_threshold = valid_part(model, valid_loader)
        print(f"Valid | AUPRC: {valid_auprc:.4f} | AUROC: {valid_auroc:.4f} | F1: {valid_f1:.4f}")
        print(f"Valid Confusion Matrix:\n{valid_cm}")
        print(f"Used Threshold: {best_threshold}")

        # Unpack Validation Confusion Matrix for MLflow logging
        tn, fp, fn, tp = valid_cm.ravel()
        current_lr = scheduler.get_last_lr()[0]

        # Output Dict
        OUTPUT.append({'epoch': epoch,
                       'train_auprc': train_auprc,
                       'valid_auprc': valid_auprc,
                       'valid_auroc': valid_auroc,
                       'valid_f1': valid_f1,
                       'valid_cm': valid_cm, 
                       'valid_targets': valid_targets,
                       'valid_outputs': valid_outputs})

        # Save Checkpoint every epoch
        checkpoint_name = f"checkpoint_epoch_{epoch:03d}.pth"
        checkpoint_path = os.path.join(model_directory, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)
        
        # Log to MLflow as an artifact
        try:
            mlflow.log_artifact(checkpoint_path)
        except Exception as e:
            print(f"Warning: Failed to log artifact to MLflow: {e}")
                       
        # MLflow Logging
        mlflow.log_metrics({
            'train_auprc': train_auprc,
            'train_auroc': train_auroc,
            'train_f1': train_f1,
            'valid_auprc': valid_auprc, 
            'valid_auroc': valid_auroc,
            'valid_f1': valid_f1,
            'valid_tn': tn,
            'valid_fp': fp,
            'valid_fn': fn,
            'valid_tp': tp,
            'learning_rate': current_lr
        }, step=epoch)
        
        scheduler.step()
        
    # Before saving the final pickle, add the last model state to the LAST item in OUTPUT 
    # to maintain compatibility with scripts that expect a model state at the end.
    if OUTPUT:
        OUTPUT[-1]['model'] = copy.deepcopy(model).cpu().state_dict()
    
    name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
    with open(name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_part(model, dataset, opt, loss_fn):
    targets = []
    outputs = []
    model.train()

    for (x, t, l) in dataset:
        opt.zero_grad()

        x = x.unsqueeze(2).float().to(DEVICE)
        t = t.to(DEVICE)
        l = l.float().to(DEVICE)

        y = model(x, l)
        t_indices = torch.argmax(t, dim=1)

        J = loss_fn(input=y, target=t_indices)
        J.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        opt.step()

        p = torch.softmax(y, dim=1)
       
        targets.append(t_indices.data.cpu().numpy())
        outputs.append(p.data.cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    
    positive_class_probs = outputs[:, 1]
    predictions = np.argmax(outputs, axis=1) # Convert probabilities to hard predictions (0 or 1)
    
    # Calculate metrics
    auprc = average_precision_score(y_true=targets, y_score=positive_class_probs)
    auroc = roc_auc_score(y_true=targets, y_score=positive_class_probs)
    f1 = f1_score(y_true=targets, y_pred=predictions)
    cm = confusion_matrix(y_true=targets, y_pred=predictions)

    return auprc, auroc, f1, cm

def valid_part(model, dataset):
    targets = []
    outputs = []
    model.eval() 

    with torch.no_grad():  
        for (x, t, l) in dataset:
            # x is shape [1, 24, Length] initially
            sig_len = x.shape[-1]
            window_size = WINDOW_SIZE
            step_size = STEP_SIZE
            
            # Pad if shorter than a single window
            if sig_len < window_size:
                pad_len = window_size - sig_len
                x = torch.nn.functional.pad(x, (0, pad_len), "constant", 0)
                sig_len = window_size
                
            # Slice into windows
            windows = []
            for start in range(0, sig_len - window_size + 1, step_size):
                windows.append(x[:, :, start : start + window_size])
            if len(windows) == 0:
                windows.append(x[:, :, -window_size:])
                
            windows_tensor = torch.cat(windows, dim=0).to(DEVICE)  # Shape: [N_windows, 24, 4992]
            windows_tensor = windows_tensor.unsqueeze(2)           # Shape: [N_windows, 24, 1, 4992]

            t = t.to(DEVICE)
            l = l.float().to(DEVICE)
            l_expanded = l.expand(windows_tensor.shape[0], -1)   # Expand l to match N_windows
            
            y = model(windows_tensor, l_expanded)
            p = torch.softmax(y, dim=1) # Shape: [N_windows, 2]

            # The MAX probability acted as an "OR" gate across all windows, which proved 
            # too sensitive to false-positive noise in 5-minute recordings!
            # Let's use the MEAN probability instead. The model now requires a consistent 
            # positive signal (or one massive 99% confident spike) to predict positive.
            mean_prob_positive = p[:, 1].mean().unsqueeze(0)
            prob_negative = 1.0 - mean_prob_positive
            patient_p = torch.stack([prob_negative, mean_prob_positive], dim=1) # Shape: [1, 2]

            t_indices = torch.argmax(t, dim=1)

            targets.append(t_indices.data.cpu().numpy())
            outputs.append(patient_p.data.cpu().numpy())
            
    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    
    positive_class_probs = outputs[:, 1]
    
    # Static Threshold (Standard Argmax)
    predictions = np.argmax(outputs, axis=1)
    best_threshold = 0.5
    
    auprc = average_precision_score(y_true=targets, y_score=positive_class_probs)
    auroc = roc_auc_score(y_true=targets, y_score=positive_class_probs)
    f1 = f1_score(y_true=targets, y_pred=predictions)
    cm = confusion_matrix(y_true=targets, y_pred=predictions)
    
    return auprc, auroc, f1, cm, targets, outputs, best_threshold