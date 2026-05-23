import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
try:
    import optuna
except ImportError:
    optuna = None

from model_structure import NN

from torch.utils.data import WeightedRandomSampler


from scipy import signal
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, confusion_matrix
from pathlib import Path

# Imports from local scripts
from helper_code import load_header, get_nsamp, get_leads, get_sex, get_frequency, find_header_files
from helper_code import get_labels, lead_exctractor, load_recording, expand_leads, _load_model, finetune_model_prep

# Global DEVICE is now handled by passing it as an argument
# selector = DeviceSelector()
# DEVICE = selector.select(1)[0]

# ── Reproducibility seed ────────────────────────────────────────────────────
SEED = 42
random_module = __import__('random')
random_module.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 4992 # 10 seconds (Multiple of 64) 15040 4992
STEP_SIZE   = 2496  # 5 second stride 10048 2496

# ── Optuna Trial Pruning ─────────────────────────────────────────────────────
# If valid_loss / train_loss exceeds this ratio at ANY epoch (incl. warmup),
# the trial's hyperparameters are structurally bad → prune the entire trial.
# Keep at ~3.0: a ratio of 2x is expected in small datasets, 3x+ means
# the LR or weight-decay combination is actively blowing up generalisation.
PRUNE_LOSS_RATIO = 3.0
# ─────────────────────────────────────────────────────────────────────────────


# ── High-Regularization B-cos Config ────────────────────────────────────────
# Hit AUROC 0.83 on Train in 4 epochs but Valid stayed at 0.54 (memorization).
# We are increasing Weight Decay to 0.1 and lowering LR to force generalization.
# ─────────────────────────────────────────────────────────────────────────────
_HEAD_LR     = 1e-4
_DECAY       = 0.25
_WD          = 0.01

CONFIG = {
    "LR_scheduler": "OneCycleLR",
    "pct_start": 0.2,   # Slightly longer warmup
    "anneal_strategy": "cos",
    "optimizer": "AdamW",
    "epochs": 20,
    "batch_size": 256,

    # Layers to train — High 'friction' to prevent memorization
    "layer_tuning": {
        "conv": {"trainable": False, "lr": 0,                              "weight_decay": 0},
        "bn":   {"trainable": True,  "lr": 1e-6,                           "weight_decay": 0},
        "rb_0": {"trainable": True,  "lr": _HEAD_LR * _DECAY ** 3,        "weight_decay": _WD},
        "rb_1": {"trainable": True,  "lr": _HEAD_LR * _DECAY ** 3,        "weight_decay": _WD},
        "rb_2": {"trainable": True,  "lr": _HEAD_LR * _DECAY ** 3,        "weight_decay": _WD},
        "rb_3": {"trainable": True,  "lr": _HEAD_LR * _DECAY ** 2,        "weight_decay": _WD},
        "rb_4": {"trainable": True,  "lr": _HEAD_LR * _DECAY,             "weight_decay": _WD},
        "head": {"trainable": True,  "lr": _HEAD_LR,                      "weight_decay": _WD * 10},
    }
}




def build_flexible_optimizer(model, config):
    param_groups = []
    layer_config = config.get("layer_tuning", {})
    default_lr = 1e-5
    default_wd = 1e-3
    
    # Iterate through the top-level blocks of the model (conv, rb_0, head, etc.)
    for name, child in model.named_children():
        # Skip layers that have no parameters (like max pool, avg pool, dropout)
        if len(list(child.parameters())) == 0:
            continue
        
        # Get settings for this layer, or use defaults if not specified
        settings = layer_config.get(name, {"trainable": True, "lr": default_lr})
        is_trainable = settings.get("trainable", True)
        lr = settings.get("lr", default_lr)
        wd = settings.get("weight_decay", default_wd)
        
        # 1. Turn the layer on or off
        for param in child.parameters():
            param.requires_grad = is_trainable
            
        # 2. If it's on, add it to the optimizer with its specific learning rate
        if is_trainable:
            param_groups.append({
                'params': filter(lambda p: p.requires_grad, child.parameters()),
                'lr': lr,
                'weight_decay': wd,
                'name': name # Helpful for debugging later if needed
            })
            
    # Build and return the optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=default_wd)
    
    # Optional: Print a summary so you know exactly what is happening
    print("\n--- Layer Tuning Summary ---")
    for name, child in model.named_children():
        if len(list(child.parameters())) == 0:
            continue
        settings = layer_config.get(name, {})
        status = "🟢 Trainable" if settings.get("trainable", True) else "🔴 Frozen"
        print(f"{name.ljust(10)}: {status} | LR: {settings.get('lr', default_lr)} | WD: {settings.get('weight_decay', default_wd)}")
    print("----------------------------\n")
            
    return optimizer


import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # 1. Get raw log probabilities (unweighted!)
        log_p = F.log_softmax(input, dim=-1)
        
        # 2. Extract the log_prob of the correct target class
        log_pt = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        
        # 3. pt is the unweighted probability of the correct class
        pt = torch.exp(log_pt)
        
        # 4. Calculate unweighted Cross Entropy
        ce_loss = -log_pt
        
        # 5. Apply class weights if provided
        if self.weight is not None:
            at = self.weight[target]
            ce_loss = ce_loss * at
            
        # 6. Apply the focal modulation factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ── ECG Augmentation constants (train-only) ──────────────────────────────────
AUG_AMP_MIN   = 0.85   # minimum amplitude scale factor
AUG_AMP_MAX   = 1.15   # maximum amplitude scale factor
AUG_NOISE_STD = 0.003  # Gaussian noise σ — chosen well below P-wave amplitude
AUG_CUTOUT_PROB = 0.5  # Probability to apply time cutout
# ─────────────────────────────────────────────────────────────────────────────

# Class that creates custom dataset from given data
class CustomDataset(Dataset):

    def __init__(self, header_paths, window_size=WINDOW_SIZE, step=STEP_SIZE):
        """Initialize dataset

        Args:
            header_paths (list): List of paths to header files
            window_size (int, optional): Size of window. Defaults to 1000.
            step (int, optional): Posun okna. Defaults to 500.
        """
        super().__init__()
        self.files = list()
        self.is_train = True
        self.window_size = window_size
        self.step_size = step
        self.num_leads = 12 

        features_csv_path = '/srv/home/jhyl/Afib_recurrence/features.csv'
        if os.path.exists(features_csv_path):
            features_df = pd.read_csv(features_csv_path)
            self.rhythm_map = dict(zip(features_df['ID'].astype(str), features_df['is_AFIB_before']))
        else:
            raise FileNotFoundError(f"Features file not found at {features_csv_path}")

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
            
            # Load rhythm info from features.csv
            record_id = os.path.basename(path).replace(".hea", "")
            is_afib = self.rhythm_map.get(record_id, None)
            if is_afib is None:
                raise ValueError(f"Patient ID {record_id} not found in features.csv!")
            if is_afib == 1:
                rhythm_vector = np.array([1, -1], dtype=np.float32)
            else:
                rhythm_vector = np.array([-1, 1], dtype=np.float32)
            temp_dict['rhythm'] = rhythm_vector
            
            # Map windows using the pre-processed .npy shape
            if os.path.exists(npy_path):
                data_mmap = np.load(npy_path, mmap_mode='r')
                sig_len = data_mmap.shape[1]
                start_indices = np.array(range(0, sig_len - self.window_size + 1, self.step_size))
                temp_dict["start_indices"] = start_indices
                self.files.append(temp_dict)
            else:
                print(f"Warning: {npy_path} not found.")

        self.files_df = pd.DataFrame(self.files).sort_values('header').reset_index(drop=True)
        # Each entry: (file_index, window_start, flipped)
        # flipped=False → original; flipped=True → polarity-inverted copy (train only)
        self.window_map = [(i, start, False) for i, row in self.files_df.iterrows() for start in row['start_indices']]

    def _add_flipped_windows(self):
        """Append polarity-inverted copies of every window to window_map.
        Call ONLY after the train/valid split to ensure zero leakage."""
        flipped = [(i, start, True) for (i, start, _) in self.window_map]
        self.window_map = self.window_map + flipped

    def train_valid_split(self, test_size):
        files = self.files_df['header'].to_numpy().reshape(-1,1)
        targets = np.stack(self.files_df['target'].to_list(),axis=0)

        x_train, y_train, x_valid, y_valid = iterative_train_test_split(files, targets, test_size=test_size)

        train = CustomDataset(header_paths=x_train[:,0].tolist())
        train.is_train=True
        train.num_leads=None
        #train._add_flipped_windows()  # doubles the training set via polarity flip

        valid = CustomDataset(header_paths=x_valid[:,0].tolist())
        valid.is_train=False
        valid.num_leads=12
        # NOTE: no flip added to valid — zero leakage

        return train, valid
    
    def get_kfold_splits(self, n_splits=4):
        """Generates train and valid dataset pairs for K-Fold Cross Validation"""
        # We need the filenames and targets to split on
        files = self.files_df['header'].to_numpy()
        targets = np.stack(self.files_df['target'].to_list(), axis=0)
        
        # Targets are one-hot encoded (e.g. [1,0] or [0,1]), so we convert them 
        # to class indices (0 or 1) so StratifiedKFold can balance them correctly.
        labels = np.argmax(targets, axis=1)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        splits = []
        for train_idx, valid_idx in skf.split(files, labels):
            train_headers = files[train_idx].tolist()
            valid_headers = files[valid_idx].tolist()
            
            # Create training dataset instance
            train_dataset = CustomDataset(header_paths=train_headers)
            train_dataset.is_train = True
            train_dataset.num_leads = None
            #źtrain_dataset._add_flipped_windows()  # doubles the training set via polarity flip

            # Create validation dataset instance
            valid_dataset = CustomDataset(header_paths=valid_headers)
            valid_dataset.is_train = False
            valid_dataset.num_leads = 12
            # NOTE: no flip added to valid — zero leakage
            
            splits.append((train_dataset, valid_dataset))
            
        return splits

    def summary(self,output):
        if output=='pandas':
            return pd.Series(np.stack(self.files_df['target'].to_list(),axis=0).sum(axis=0))
        if output=='numpy':
            return np.stack(self.files_df['target'].to_list(),axis=0).sum(axis=0)
    
    def show_problematic_samples(self):
        problematic_list = list()
        for index, row in self.files_df.iterrows():
            data = load_recording(row["record"])
            if data.shape[1] != row["nsamp"]:
                problematic_list.append((row["record"], data.shape[1], row["nsamp"]))
        return problematic_list
    
    def __len__(self):
        if self.is_train == True:
            return len(self.window_map)
        elif self.is_train == False:
            return len(self.files_df)

    def __getitem__(self, index):
        if self.is_train:
            # --- TRAINING: fixed-size window slices + augmentation ---
            sig_idx, start, flipped = self.window_map[index]
            row = self.files_df.iloc[sig_idx]
            data = np.load(row['npy_path'], mmap_mode='r').astype(np.float32)
            window = data[:, start : start + self.window_size].copy()  # writable copy

            # 1. Polarity flip (horizontal-axis inversion of all leads)
            if flipped:
                window = -window

            # 2. Mild amplitude augmentation (random scale ±15%) - DISABLED to preserve RMS voltage
            # amp_scale = np.random.uniform(AUG_AMP_MIN, AUG_AMP_MAX)
            # window = window * amp_scale

            # 3. Very mild Gaussian noise — σ kept well below P-wave amplitude
            noise = np.random.normal(0.0, AUG_NOISE_STD, size=window.shape).astype(np.float32)
            window = window + noise

            # 4. Spatial Lead Dropout (random choice of lead configurations)
            # Simulates 2, 3, 4, 6, 8, or 12-lead machines dynamically
            lead_indicator = np.ones(12)
            window, lead_indicator = lead_exctractor.get(window, self.num_leads, lead_indicator)

            # 5. Time Cutout (Regularization)
            if np.random.rand() < AUG_CUTOUT_PROB:
                # drop 10% to 20% of the signal in time
                cutout_len = int(self.window_size * np.random.uniform(0.20, 0.30))
                start_idx = np.random.randint(0, self.window_size - cutout_len)
                window[:, start_idx : start_idx + cutout_len] = 0.0

            # --- B-COSIFICATION ---
            pos = np.maximum(window, 0)
            neg = np.maximum(-window, 0)
            window_bcos = np.concatenate([pos, neg], axis=0)

            return torch.from_numpy(window_bcos).float(), torch.from_numpy(row['target']).float(), torch.from_numpy(lead_indicator).float(), torch.from_numpy(row['rhythm']).float()
        else:
            # --- VALIDATION: Full length signal — NO augmentation ---
            row = self.files_df.iloc[index]
            data = np.load(row['npy_path']).astype(np.float32)

            # Pad length to be a perfect multiple of 64
            seq_len = data.shape[1]
            remainder = seq_len % 64
            if remainder != 0:
                pad_len = 64 - remainder
                data = np.pad(data, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)

            lead_indicator = np.ones(12)
            data, lead_indicator = lead_exctractor.get(data, self.num_leads, lead_indicator)

            # --- B-COSIFICATION ---
            pos = np.maximum(data, 0)
            neg = np.maximum(-data, 0)
            data_bcos = np.concatenate([pos, neg], axis=0)

            return torch.from_numpy(data_bcos).float(), torch.from_numpy(row['target']).float(), torch.from_numpy(lead_indicator).float(), torch.from_numpy(row['rhythm']).float()
        

def training_code(data_directory, model_directory, resume_checkpoint=None, config=None, device=None):
    return _training_code(data_directory, model_directory, "finetuned", resume_checkpoint, config=config, device=device)

def training_code_kfold(data_directory, model_directory, k_folds=4, resume_checkpoint=None, config=None, device=None, trial=None):
    print(f"Finding header and recording files for {k_folds}-Fold Cross-Validation...")
    header_files = find_header_files(data_directory)
    full_dataset = CustomDataset(header_files)
    
    # Get K folds
    folds = full_dataset.get_kfold_splits(n_splits=k_folds)
    
    fold_aurocs = []
    # Train separate models iteratively
    for fold_idx, (train_ds, valid_ds) in enumerate(folds):
        print(f"\n" + "="*20 + f" FOLD {fold_idx + 1}/{k_folds} " + "="*20)
        
        fold_model_dir = os.path.join(model_directory, f"fold_{fold_idx+1}")
        os.makedirs(fold_model_dir, exist_ok=True)
        
        # Use nested MLflow runs to keep the dashboard organized
        with mlflow.start_run(run_name=f"Fold_{fold_idx+1}", nested=True):
            # Pass the pre-split datasets instead of data_directory
            best_auroc, train_at_best = _training_code((train_ds, valid_ds), fold_model_dir, f"fold_{fold_idx+1}", resume_checkpoint, config=config, device=device, trial=trial, fold_idx=fold_idx)
            fold_aurocs.append((best_auroc, train_at_best))
    
    return fold_aurocs

def _training_code(data_directory_or_datasets, model_directory, ensamble_ID, resume_checkpoint=None, config=None, device=None, trial=None, fold_idx=0):
    if config is None:
        config = CONFIG
    
    if device is None:
        # Fallback if no device is passed, though we should always pass it
        from device_selector import DeviceSelector
        selector = DeviceSelector()
        device = selector.select(1)[0]
    # === TensorBoard setup ===
    log_dir = os.path.join(model_directory, "runs", f"ensamble_{ensamble_ID}_{int(time.time())}")
    writer = SummaryWriter(log_dir=log_dir)

    # Allow data_directory_or_datasets to be a pair of datasets or a path to find files
    if isinstance(data_directory_or_datasets, str):
        # Find header and recording files.
        print('Finding header and recording files...')
        header_files = find_header_files(data_directory_or_datasets)

        full_dataset = CustomDataset(header_files)
        # print(full_dataset.show_problematic_samples())
        train,valid = full_dataset.train_valid_split(test_size=0.2)
        print("Succesfully created train and valid dataset...")
    else:
        # Pre-split datasets are passed for K-fold
        train, valid = data_directory_or_datasets
        print("Successfully loaded pre-split train and valid datasets for fold...")

    # negative to positive ratio
    loss_weight = (len(train) - train.summary(output='numpy'))/train.summary(output='numpy')


    # to be saved in resulting model pickle
    train_files = train.files_df['header'].to_list()
    train_files = [k.split('/')[-1] for k in train_files]

    valid_files = valid.files_df['header'].to_list()
    valid_files = [k.split('/')[-1] for k in valid_files]

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    print("Configuring WeightedRandomSampler...")
    
    # 1. Get total counts for each class
    class_counts = train.summary(output='numpy') # e.g., [50000, 150000] windows
    class_weights = 1.0 / class_counts           # Inverse frequency
    
    # 2. Assign the appropriate weight to every single window in the dataset
    sample_weights = []
    for sig_idx, _start, _flipped in train.window_map:
        # Grab the target vector (e.g., [1, 0] or [0, 1]) and find the class index
        target_vector = train.files_df.iloc[sig_idx]['target']
        class_idx = np.argmax(target_vector)
        
        # Append the weight for that specific class
        sample_weights.append(class_weights[class_idx])
        
    sample_weights = torch.DoubleTensor(sample_weights)
    
    # 3. Create the sampler
    # DISABLED FOR FOCAL LOSS: Oversampling duplicates minority data, ruining Focal Loss dynamics
    # sampler = WeightedRandomSampler(
    #     weights=sample_weights, 
    #     num_samples=len(sample_weights), 
    #     replacement=True
    # )

    # Seeded generator so DataLoader shuffle is reproducible
    _train_gen = torch.Generator()
    _train_gen.manual_seed(SEED)

    train = DataLoader(dataset=train,
                       batch_size=config.get("batch_size", 128),
                       shuffle=True,  # Added shuffle=True since sampler is disabled
                       num_workers=8,
                       collate_fn=collate_fn,
                       pin_memory=True,
                       drop_last=False,
                       generator=_train_gen)


    valid = DataLoader(dataset=valid,
                       batch_size=1,
                       shuffle=False,
                       num_workers=8,
                       collate_fn=collate_fn,
                       pin_memory=True,
                       drop_last=False)

    dropout_rate = config.get("dropout_rate", 0.5)
    loaded_model = _load_model("/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model/Trial_42_epoch_5/", fold_idx + 1, nOUT=2, device=device, dropout_rate=dropout_rate)
    classifier = loaded_model["classifier"]
    prep_m = finetune_model_prep(classifier, device)
    model = prep_m.to(device)
    
    # Log the exact model architecture for this run
    mlflow.log_text(str(model), "model_architecture.txt")

    # Use .dataset since train is now a DataLoader
    class_counts = train.dataset.summary(output='numpy')
    weights = 1.0 / class_counts 
    weights = weights / weights.sum() 
    
    # Setup information about the class weights (class imbalance) for the loss
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        
        opt = build_flexible_optimizer(model, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        # opt.load_state_dict(checkpoint['optimizer_state_dict']) # <-- TEMPORARILY DISABLED: We want a fresh Learning Rate!
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # <-- TEMPORARILY DISABLED: Restart the schedule!
        print(f"Loaded checkpoint '{resume_checkpoint}' (resuming from epoch {start_epoch})")
    else:
        opt = build_flexible_optimizer(model, config)
        # Use ReduceLROnPlateau for the main training phase
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, 
            mode='min', 
            factor=0.5, 
            patience=2
        )
        
    # Store initial target LRs for warmup
    target_lrs = [group['lr'] for group in opt.param_groups]
    warmup_epochs = 3
    
    mlflow.log_params(config)

    OUTPUT = []
    EPOCHS = config["epochs"]
    best_valid_auroc = 0.0
    train_auroc_at_best = 0.0
    min_valid_loss = float('inf')

    min_valid_loss = float('inf')
    patience = 5  # Stop if loss doesn't improve for 5 epochs
    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        print(f"============================[{epoch}]============================")
        
        # --- HYBRID SCHEDULER: Manual Linear Warmup ---
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for i, group in enumerate(opt.param_groups):
                group['lr'] = target_lrs[i] * warmup_factor
            print(f"Warm-up Phase: Scaling LR to {warmup_factor*100:.1f}% of target.")
        
        # Unpack the new train loss (Removed batch-level scheduler step)
        train_loss, train_auprc, train_auroc, train_f1, train_cm = train_part(model, train, opt, loss_fn, device, scheduler=None)
        print(f"Train | Loss: {train_loss:.4f} | AUPRC: {train_auprc:.4f} | AUROC: {train_auroc:.4f} | F1: {train_f1:.4f}")
        
        # Pass loss_fn to validation and unpack the new valid loss
        valid_loss, valid_auprc, valid_auroc, valid_f1, valid_cm, valid_targets, valid_outputs, best_threshold = valid_part(model, valid, loss_fn, device)
        
        if valid_auroc > best_valid_auroc:
            best_valid_auroc = valid_auroc
            train_auroc_at_best = train_auroc
            
        print(f"Valid | Loss: {valid_loss:.4f} | AUPRC: {valid_auprc:.4f} | AUROC: {valid_auroc:.4f} | F1: {valid_f1:.4f}")
        print(f"Valid Confusion Matrix:\n{valid_cm}")
        
        tn, fp, fn, tp = valid_cm.ravel()
        loss_ratio = valid_loss / (train_loss + 1e-8)
        print(f"Valid/Train Loss Ratio: {loss_ratio:.2f}")

        # ── TIER-1: TRIAL PRUNING (bad hyperparameters — kills entire Optuna trial) ──
        # Active at EVERY epoch, including warmup, so extreme early overfitting is caught.
        # A high valid/train loss ratio means the LR / weight-decay combination is
        # structurally wrong, not just fold-level noise → no point running remaining folds.
        if trial is not None and optuna is not None and loss_ratio > PRUNE_LOSS_RATIO:
            phase = "warm-up" if epoch < warmup_epochs else "post-warmup"
            print(f"[TRIAL PRUNED] ({phase} epoch {epoch}) "
                  f"Valid/Train loss ratio {loss_ratio:.2f} > {PRUNE_LOSS_RATIO}. "
                  f"Hyperparameters are structurally bad — skipping remaining folds.")
            mlflow.set_tag("pruned_reason", f"loss_ratio={loss_ratio:.2f} at epoch {epoch} ({phase})")
            raise optuna.exceptions.TrialPruned()
        # ─────────────────────────────────────────────────────────────────────────────

        # MLflow Logging
        mlflow.log_metrics({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'loss_ratio': loss_ratio,
            'train_auprc': train_auprc,
            'train_auroc': train_auroc,
            'train_f1': train_f1,
            'valid_auprc': valid_auprc, 
            'valid_auroc': valid_auroc,
            'valid_f1': valid_f1,
            'valid_tn': tn,
            'valid_fp': fp,
            'valid_fn': fn,
            'valid_tp': tp
        }, step=epoch)

        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(model_directory, checkpoint_filename)
        
        # Save full state to allow for resuming training
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'valid_auroc': valid_auroc
        }, checkpoint_path)
        
        # Log the checkpoint to MLflow
        mlflow.log_artifact(local_path=checkpoint_path, artifact_path="checkpoints")
        
        # ── TIER-2: FOLD EARLY STOPPING (overfitting on this fold — break only) ──
        # Fires only when BOTH conditions hold simultaneously (post-warmup):
        #   1. valid_loss > train_loss  (model has crossed into overfitting territory)
        #   2. valid_loss is not improving  (it's plateaued / climbing, not recovering)
        # If valid_loss is still below train_loss the model is fine; reset the counter.
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            patience_counter = 0
            
            # Save the 'best' model weights right now, before overfitting starts
            best_model_path = os.path.join(model_directory, "best_loss_weights.pth")
            torch.save(model.state_dict(), best_model_path)
            mlflow.log_artifact(local_path=best_model_path, artifact_path="model")
        else:
            # Only count patience post-warmup AND when valid has crossed above train
            if epoch >= warmup_epochs and valid_loss > train_loss:
                patience_counter += 1
                print(f"Valid loss above train loss and not improving. "
                      f"Patience: {patience_counter}/{patience}")
            elif epoch >= warmup_epochs:
                # Valid loss is not improving but still below train — reset, keep training
                print(f"Valid loss not improving but still below train loss "
                      f"({valid_loss:.4f} <= {train_loss:.4f}). Patience counter reset.")
                patience_counter = 0
            else:
                print("Warm-up phase: Patience counter paused.")
        # ─────────────────────────────────────────────────────────────────────────────
            
        # --- HYBRID SCHEDULER: Plateau Step ---
        if epoch >= warmup_epochs:
            scheduler.step(valid_loss)

        # CHECK: Has patience run out? (fold early stop — not a trial prune)
        if patience_counter >= patience:
            print(f"FOLD EARLY STOPPED: Validation loss has not improved for {patience} epochs. "
                  f"(Remaining folds will still run.)")
            break

        
    writer.close()
    
    name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
    with open(name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(class_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_valid_auroc, train_auroc_at_best



def train_part(model, dataset, opt, loss_fn, device, scheduler=None):
    from torchmetrics.classification import (
        BinaryAUROC, BinaryAveragePrecision, BinaryF1Score, BinaryConfusionMatrix
    )

    # Initialize torchmetrics metrics and move them to device
    metric_auroc = BinaryAUROC().to(device)
    metric_auprc = BinaryAveragePrecision().to(device)
    metric_f1    = BinaryF1Score().to(device)
    metric_cm    = BinaryConfusionMatrix().to(device)

    total_loss = 0.0
    num_batches = 0

    # Set the entire model to training mode
    model.train()

    #PYTORCH GOTCHA FIX:
    # `model.train()` will reactivate BatchNorm running stats for the ENTIRE network.
    # If a layer is supposed to be frozen (requires_grad=False), the BN layers inside it
    # will STILL track new statistics, destroying your pre-trained weights!
    # We must loop through and force frozen BN layers back into `.eval()` mode.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            if hasattr(m, 'weight') and m.weight is not None and not m.weight.requires_grad:
                m.eval()

    for (x, t, l, r) in dataset:
        opt.zero_grad()

        x = x.unsqueeze(2).float().to(device)
        t = t.to(device)
        l = l.float().to(device)
        r = r.float().to(device)

        y = model(x, l, r)
        t_indices = torch.argmax(t, dim=1)
        J = loss_fn(input=y, target=t_indices)

        J.backward()

        total_loss += J.item()
        num_batches += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        opt.step()
        
        # Update torchmetrics: remap so that 1 = recurrence (positive class)
        with torch.no_grad():
            p = torch.softmax(y, dim=1)
            pos_probs   = p[:, 0]                  # P(recurrence): higher = more likely recurrence
            targets_sk  = 1 - t_indices            # 0→1 (recurrence), 1→0 (healthy)

            metric_auroc.update(pos_probs, targets_sk)
            metric_auprc.update(pos_probs, targets_sk)
            metric_f1.update(pos_probs, targets_sk)
            metric_cm.update(pos_probs, targets_sk)

    # Compute epoch-level aggregates
    auroc = metric_auroc.compute().item()
    auprc = metric_auprc.compute().item()
    f1    = metric_f1.compute().item()
    cm    = metric_cm.compute().cpu().numpy()

    # Reset all metrics for the next epoch
    metric_auroc.reset()
    metric_auprc.reset()
    metric_f1.reset()
    metric_cm.reset()

    avg_train_loss = total_loss / num_batches
    return avg_train_loss, auprc, auroc, f1, cm


def valid_part(model, dataset, loss_fn, device): # <-- Added loss_fn
    targets = []
    outputs = []
    total_loss = 0.0 # <-- Initialize loss tracking
    num_batches = 0
    
    model.eval() 

    with torch.no_grad():  
        for (x, t, l, r) in dataset:
            sig_len = x.shape[-1]
            window_size = WINDOW_SIZE
            step_size = STEP_SIZE
            
            if sig_len < window_size:
                pad_len = window_size - sig_len
                x = torch.nn.functional.pad(x, (0, pad_len), "constant", 0)
                sig_len = window_size
                
            windows = []
            for start in range(0, sig_len - window_size + 1, step_size):
                windows.append(x[:, :, start : start + window_size])
            if len(windows) == 0:
                windows.append(x[:, :, -window_size:])
                
            windows_tensor = torch.cat(windows, dim=0).to(device)
            windows_tensor = windows_tensor.unsqueeze(2)

            t = t.to(device)
            l = l.float().to(device)
            r = r.float().to(device)
            l_expanded = l.expand(windows_tensor.shape[0], -1)
            r_expanded = r.expand(windows_tensor.shape[0], -1)
            
            # Get raw logits for all windows: [num_windows, 2]
            y = model(windows_tensor, l_expanded, r_expanded)
            
            # Patient-level prediction: Average (AVG) Pooling on probabilities
            # Simple arithmetic mean of window-level predictions.
            patient_p = torch.softmax(y, dim=1).mean(dim=0, keepdim=True)
            
            # True validation loss: FocalLoss expects logits. We can safely pass log(p)
            # because FocalLoss internally applies log_softmax, and log_softmax(log(p)) = log(p).
            pseudo_logits = torch.log(patient_p + 1e-8)
            
            t_indices = torch.argmax(t, dim=1)
            batch_loss = loss_fn(input=pseudo_logits, target=t_indices)
            total_loss += batch_loss.item()
            num_batches += 1

            targets.append(t_indices.data.cpu().numpy())
            outputs.append(patient_p.data.cpu().numpy())
            
    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    
    # Remap for sklearn: internally 0=recurrence, 1=healthy.
    # Sklearn assumes 1=positive, so flip: 1=recurrence, 0=healthy.
    targets_sk  = 1 - targets                       # 0→1 (recurrence), 1→0 (healthy)
    pos_probs   = outputs[:, 0]                     # P(recurrence): higher = more likely recurrence
    predictions = (1 - np.argmax(outputs, axis=1))  # argmax 0→pred 1 (recurrence), 1→pred 0 (healthy)
    best_threshold = 0.5
    
    auprc = average_precision_score(y_true=targets_sk, y_score=pos_probs)
    auroc = roc_auc_score(y_true=targets_sk,           y_score=pos_probs)
    f1    = f1_score(y_true=targets_sk,                y_pred=predictions)
    cm    = confusion_matrix(y_true=targets_sk,        y_pred=predictions)
    
    avg_valid_loss = total_loss / num_batches
    
    # Return the loss alongside the metrics
    return avg_valid_loss, auprc, auroc, f1, cm, targets_sk, outputs, best_threshold


def collate_fn(batch):
    # batch: list of tuples (x, t, l, r)
    
    # Stack inputs along batch dimension
    X = torch.stack([b[0] for b in batch], dim=0)
    t = torch.stack([b[1] for b in batch], dim=0)
    l = torch.stack([b[2] for b in batch], dim=0)
    r = torch.stack([b[3] for b in batch], dim=0)
    
    return X, t, l, r