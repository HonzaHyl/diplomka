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

from device_selector import DeviceSelector

selector = DeviceSelector()
DEVICE = selector.select(1)[0]

WINDOW_SIZE = 4992 # 30.08 seconds (Multiple of 64) 15040 4992
STEP_SIZE   = 2496  # 20.096 second stride 10048 2496

CONFIG = {
    "learning_rate": 1e-4,
    "LR_scheduler": "OneCycleLR",
    "pct_start": 0.3, 
    "anneal_strategy": "cos",
    "optimizer": "AdamW",
    "weight_decay": 1e-3,
    "epochs": 30,
    
    # Layers to train (unfrozen from the start)
    "layer_tuning": {
        "conv": {"trainable": True, "lr": 1e-7},
        "bn":   {"trainable": True, "lr": 1e-7},
        "rb_0": {"trainable": True, "lr": 1e-6},
        "rb_1": {"trainable": True, "lr": 1e-6},
        "rb_2": {"trainable": True, "lr": 1e-6},
        "rb_3": {"trainable": True,  "lr": 5e-6}, 
        "rb_4": {"trainable": True,  "lr": 5e-6}, 
        "fc_1": {"trainable": True,  "lr": 1e-4}
    }
}

def build_flexible_optimizer(model, config):
    param_groups = []
    layer_config = config.get("layer_tuning", {})
    default_lr = config.get("learning_rate", 1e-5)
    
    # Iterate through the top-level blocks of the model (conv, rb_0, fc_1, etc.)
    for name, child in model.named_children():
        
        # Get settings for this layer, or use defaults if not specified
        settings = layer_config.get(name, {"trainable": True, "lr": default_lr})
        is_trainable = settings["trainable"]
        lr = settings["lr"]
        
        # 1. Turn the layer on or off
        for param in child.parameters():
            param.requires_grad = is_trainable
            
        # 2. If it's on, add it to the optimizer with its specific learning rate
        if is_trainable:
            param_groups.append({
                'params': filter(lambda p: p.requires_grad, child.parameters()),
                'lr': lr,
                'name': name # Helpful for debugging later if needed
            })
            
    # Build and return the optimizer
    optimizer = optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    
    # Optional: Print a summary so you know exactly what is happening
    print("\n--- Layer Tuning Summary ---")
    for name, child in model.named_children():
        status = "🟢 Trainable" if layer_config.get(name, {}).get("trainable", True) else "🔴 Frozen"
        print(f"{name.ljust(10)}: {status} | LR: {layer_config.get(name, {}).get('lr', default_lr)}")
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

    def train_valid_split(self, test_size):
        files = self.files_df['header'].to_numpy().reshape(-1,1)
        targets = np.stack(self.files_df['target'].to_list(),axis=0)

        x_train, y_train, x_valid, y_valid = iterative_train_test_split(files, targets, test_size=test_size)

        train = CustomDataset(header_paths=x_train[:,0].tolist())
        train.is_train=True
        train.num_leads=None

        valid = CustomDataset(header_paths=x_valid[:,0].tolist())
        valid.is_train=False
        valid.num_leads=12

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

            # Create validation dataset instance
            valid_dataset = CustomDataset(header_paths=valid_headers)
            valid_dataset.is_train = False
            valid_dataset.num_leads = 12
            
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
            # --- TRAINING: 10-second slices ---
            sig_idx, window_start = self.window_map[index]
            row = self.files_df.iloc[sig_idx]
            
            # Fast load
            data = np.load(row['npy_path'], mmap_mode='r')
            sig_len = data.shape[1]
            
            # 1. Random Shift Augmentation (Jitter)
            # Instead of grabbing the exact same fixed window every epoch, we shift it randomly 
            # by up to +/- 1 second (500 samples), forcing the model to learn translation invariance.
            max_shift = 500
            min_start = max(0, window_start - max_shift)
            max_start = min(sig_len - self.window_size, window_start + max_shift)
            
            if min_start < max_start:
                actual_start = np.random.randint(min_start, max_start + 1)
            else:
                actual_start = window_start # Fallback if signal is short
                
            window = data[:, actual_start : actual_start + self.window_size].copy()
            
            # --- NEW DATA AUGMENTATIONS ---
            # 1. Amplitude scaling (between 80% and 120%)
            scale = np.random.uniform(0.8, 1.2)
            window = window * scale
            
            # 2. Gaussian noise (mu=0, sigma=0.05) to simulate sensor noise
            noise = np.random.normal(0, 0.05, window.shape)
            window = window + noise
            
            # 3. Random Polarity Inversion (20% chance)
            # TEMPORARILY DISABLED: Destroying the electrical axis on a 10-second strip is too aggressive.
            # if np.random.rand() > 0.8:
            #     window = -window
                
            # 4. Cutout / Time Masking (50% chance)
            # TEMPORARILY DISABLED: Erasing 1 whole second out of a 10s strip deletes a full P-QRS complex!
            # if np.random.rand() > 0.5:
            #     mask_len = np.random.randint(1, int(self.window_size * 0.10) + 1)
            #     mask_start = np.random.randint(0, self.window_size - mask_len)
            #     window[:, mask_start : mask_start + mask_len] = 0.0
            # ------------------------------

            return torch.from_numpy(window).float(), torch.from_numpy(row['target']).float(), torch.ones(12)
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

            return torch.from_numpy(data).float(), torch.from_numpy(row['target']).float(), torch.ones(12)
        

def training_code(data_directory, model_directory, resume_checkpoint=None):
    _training_code(data_directory, model_directory, "finetuned", resume_checkpoint)

def training_code_kfold(data_directory, model_directory, k_folds=4, resume_checkpoint=None):
    print(f"Finding header and recording files for {k_folds}-Fold Cross-Validation...")
    header_files = find_header_files(data_directory)
    full_dataset = CustomDataset(header_files)
    
    # Get K folds
    folds = full_dataset.get_kfold_splits(n_splits=k_folds)
    
    # Train separate models iteratively
    for fold_idx, (train_ds, valid_ds) in enumerate(folds):
        print(f"\n" + "="*20 + f" FOLD {fold_idx + 1}/{k_folds} " + "="*20)
        
        fold_model_dir = os.path.join(model_directory, f"fold_{fold_idx+1}")
        os.makedirs(fold_model_dir, exist_ok=True)
        
        # Use nested MLflow runs to keep the dashboard organized
        with mlflow.start_run(run_name=f"Fold_{fold_idx+1}", nested=True):
            # Pass the pre-split datasets instead of data_directory
            _training_code((train_ds, valid_ds), fold_model_dir, f"fold_{fold_idx+1}", resume_checkpoint)

def _training_code(data_directory_or_datasets, model_directory, ensamble_ID, resume_checkpoint=None):
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
    for sig_idx, _ in train.window_map:
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

    train = DataLoader(dataset=train,
                       batch_size=128,
                       shuffle=True,  # Added shuffle=True since sampler is disabled
                       num_workers=8,
                       collate_fn=collate_fn,
                       pin_memory=True,
                       drop_last=False)


    valid = DataLoader(dataset=valid,
                       batch_size=1,
                       shuffle=False,
                       num_workers=8,
                       collate_fn=collate_fn,
                       pin_memory=True,
                       drop_last=False)

    loaded_model = _load_model(".", 1, nOUT=26)
    classifier = loaded_model["classifier"]
    prep_m = finetune_model_prep(classifier)
    model = prep_m.to(DEVICE)

    # Use .dataset since train is now a DataLoader
    class_counts = train.dataset.summary(output='numpy')
    weights = 1.0 / class_counts 
    weights = weights / weights.sum() 
    
    # Setup information about the class weights (class imbalace) for the focal loss
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    #soft_weights = torch.tensor([0.70, 0.30], dtype=torch.float).to(DEVICE)
    #loss_fn = FocalLoss(weight=class_weights, gamma=2.0)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint '{resume_checkpoint}'")
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        start_epoch = checkpoint['epoch'] + 1
        
        opt = build_flexible_optimizer(model, CONFIG)
        model.load_state_dict(checkpoint['model_state_dict'])
        # opt.load_state_dict(checkpoint['optimizer_state_dict']) # <-- TEMPORARILY DISABLED: We want a fresh Learning Rate!
        
        steps_per_epoch = len(train)
        max_lrs = [group['lr'] for group in opt.param_groups]
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lrs,
            steps_per_epoch=steps_per_epoch,
            epochs=CONFIG["epochs"] - start_epoch,
            pct_start=CONFIG.get("pct_start", 0.1),
            anneal_strategy=CONFIG.get("anneal_strategy", "cos"),
            cycle_momentum=False
        )
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict']) # <-- TEMPORARILY DISABLED: Restart the schedule!
        print(f"Loaded checkpoint '{resume_checkpoint}' (resuming from epoch {start_epoch})")
    else:
        opt = build_flexible_optimizer(model, CONFIG)
        steps_per_epoch = len(train)
        max_lrs = [group['lr'] for group in opt.param_groups]
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=max_lrs,
            steps_per_epoch=steps_per_epoch,
            epochs=CONFIG["epochs"],
            pct_start=CONFIG.get("pct_start", 0.1),
            anneal_strategy=CONFIG.get("anneal_strategy", "cos"),
            cycle_momentum=False
        )
    
    mlflow.log_params(CONFIG)

    OUTPUT = []
    EPOCHS = CONFIG["epochs"]
    for epoch in range(start_epoch, EPOCHS):
        print(f"============================[{epoch}]============================")
        
        # Unpack the new train loss
        train_loss, train_auprc, train_auroc, train_f1, train_cm = train_part(model, train, opt, loss_fn, scheduler=scheduler)
        print(f"Train | Loss: {train_loss:.4f} | AUPRC: {train_auprc:.4f} | AUROC: {train_auroc:.4f} | F1: {train_f1:.4f}")
        
        # Pass loss_fn to validation and unpack the new valid loss
        valid_loss, valid_auprc, valid_auroc, valid_f1, valid_cm, valid_targets, valid_outputs, best_threshold = valid_part(model, valid, loss_fn)
        print(f"Valid | Loss: {valid_loss:.4f} | AUPRC: {valid_auprc:.4f} | AUROC: {valid_auroc:.4f} | F1: {valid_f1:.4f}")
        print(f"Valid Confusion Matrix:\n{valid_cm}")

        tn, fp, fn, tp = valid_cm.ravel()
        current_lr = scheduler.get_last_lr()[0]

        # MLflow Logging (Add the losses here)
        mlflow.log_metrics({
            'train_loss': train_loss,   # <-- NEW
            'valid_loss': valid_loss,   # <-- NEW
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

        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(model_directory, checkpoint_filename)
        
        # 2. Save full state to allow for resuming training
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'valid_auroc': valid_auroc # Optional: handy to know how good this epoch was
        }, checkpoint_path)
        
        # 3. Log the file to MLflow under a "checkpoints" folder
        mlflow.log_artifact(local_path=checkpoint_path, artifact_path="checkpoints")
        
    writer.close()
    
    name = Path(model_directory, f'PROGRESS_{ensamble_ID}.pickle')
    with open(name, 'wb') as handle:
        pickle.dump(OUTPUT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(class_weights, handle, protocol=pickle.HIGHEST_PROTOCOL)



def train_part(model, dataset, opt, loss_fn, scheduler=None):
    targets = []
    outputs = []
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

    for (x, t, l) in dataset:
        opt.zero_grad()

        x = x.unsqueeze(2).float().to(DEVICE)
        t = t.to(DEVICE)
        l = l.float().to(DEVICE)

        # --- Mixup Data Augmentation ---
        # Turning Mixup back ON to combat the massive overfitting seen in the logs 
        apply_mixup = np.random.rand() > 0.5 
   
        if apply_mixup:
            # Beta distribution for combining samples
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            
            # Shuffle indices to mix within the batch
            index = torch.randperm(x.size(0)).to(DEVICE)
            
            x = lam * x + (1 - lam) * x[index]
            t_a, t_b = t, t[index]
            
            y = model(x, l)
            
            t_indices_a = torch.argmax(t_a, dim=1)
            t_indices_b = torch.argmax(t_b, dim=1)
            
            J = lam * loss_fn(input=y, target=t_indices_a) + (1 - lam) * loss_fn(input=y, target=t_indices_b)
            t_indices = t_indices_a # For metrics approximation
        else:
            y = model(x, l)
            t_indices = torch.argmax(t, dim=1)
            J = loss_fn(input=y, target=t_indices)
        # -------------------------------

        J.backward()
        
        total_loss += J.item() # <-- Track loss
        num_batches += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        p = torch.softmax(y, dim=1)
        targets.append(t_indices.data.cpu().numpy())
        outputs.append(p.data.cpu().numpy())

    targets = np.concatenate(targets, axis=0)
    outputs = np.concatenate(outputs, axis=0)
    
    # Remap for sklearn: internally 0=recurrence, 1=healthy.
    # Sklearn assumes 1=positive, so flip: 1=recurrence, 0=healthy.
    targets_sk  = 1 - targets                       # 0→1 (recurrence), 1→0 (healthy)
    pos_probs   = outputs[:, 0]                     # P(recurrence): higher = more likely recurrence
    predictions = (1 - np.argmax(outputs, axis=1))  # argmax 0→pred 1 (recurrence), 1→pred 0 (healthy)
    
    auprc = average_precision_score(y_true=targets_sk, y_score=pos_probs)
    auroc = roc_auc_score(y_true=targets_sk,           y_score=pos_probs)
    f1    = f1_score(y_true=targets_sk,                y_pred=predictions)
    cm    = confusion_matrix(y_true=targets_sk,        y_pred=predictions)

    avg_train_loss = total_loss / num_batches
    return avg_train_loss, auprc, auroc, f1, cm


def valid_part(model, dataset, loss_fn): # <-- Added loss_fn
    targets = []
    outputs = []
    total_loss = 0.0 # <-- Initialize loss tracking
    num_batches = 0
    
    model.eval() 

    with torch.no_grad():  
        for (x, t, l) in dataset:
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
                
            windows_tensor = torch.cat(windows, dim=0).to(DEVICE)
            windows_tensor = windows_tensor.unsqueeze(2)

            t = t.to(DEVICE)
            l = l.float().to(DEVICE)
            l_expanded = l.expand(windows_tensor.shape[0], -1)
            
            # Get raw logits for all windows: [num_windows, 2]
            y = model(windows_tensor, l_expanded)
            
            # Patient-level prediction: Top-K Average Pooling across windows.
            # Averages the top K logits to isolate alarm signals without being
            # too sensitive to single-window noise artifacts (which Max pooling suffers from).
            k = min(3, y.shape[0])
            topk_vals = torch.topk(y, k=k, dim=0)[0]           # [k, 2]
            agg_logits = topk_vals.mean(dim=0, keepdim=True)   # [1, 2]
            
            patient_p = torch.softmax(agg_logits, dim=1)  # [1, 2]
            
            # True validation loss: feed aggregated logits into loss function
            t_indices = torch.argmax(t, dim=1)
            batch_loss = loss_fn(input=agg_logits, target=t_indices)
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
    # batch: list of tuples (x, t, l)
    
    # Stack inputs along batch dimension
    X = torch.stack([b[0] for b in batch], dim=0)
    t = torch.stack([b[1] for b in batch], dim=0)
    l = torch.stack([b[2] for b in batch], dim=0)
    
    return X, t, l