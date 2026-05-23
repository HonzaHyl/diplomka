import numpy as np
import random
import pickle
import os
import torch
import torch.nn as nn

from scipy.io import loadmat
from pathlib import Path

from model_structure import NN
from bcos_utils import bcosify_model

# Remove global DEVICE selection to avoid conflicts in multiprocessing
# DEVICE = selector.select(1)[0]


# Load header file as a string.
def load_header(header_path):
    """Reads header files

    Args:
        header_path (string): Path to header file

    Returns:
        string: Whole content of header file
    """
    with open(header_path, 'r') as f:
        header = f.read()
    return header

############################################################ 
# Read stuff from header file 
############################################################

# Find header and recording files.
def find_header_files(data_directory):
    header_files = list()
    for f in os.listdir(data_directory):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.hea':
            header_file = os.path.join(data_directory, root + '.hea')
            if os.path.isfile(header_file):
                header_files.append(header_file)
    return header_files

# Get number of samples for given subject
def get_nsamp(header):
    return int(header.split('\n')[0].split(' ')[3])


# Get leads from header.
def get_leads(header):
    """Get list(tuple) of leads present in the recording

    Args:
        header (string): Content of header file

    Returns:
        tuple: Tuple of leads
    """
    leads = list()
    for i, l in enumerate(header.split('\n')):
        entries = l.split(' ')
        if i==0:
            # How many leads are in the record
            num_leads = int(entries[1])
        elif i<=num_leads:
            # Append name of lead
            leads.append(entries[-1])
        else:
            break
        # Return tuple with all names of leads found 
    return tuple(leads)


# Get sex from header.
def get_sex(header):
    sex = None
    for l in header.split('\n'):
        if l.startswith('#Sex'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex


# Get frequency from header.
def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get labels from header.
def get_labels(header):
    labels = list()
    for l in header.split('\n'):
        if l.startswith('#Dx'):
            try:
                entries = l.split(': ')[1].split(',')
                for entry in entries:
                    labels.append(entry.strip())
            except:
                pass
    return labels

############################################################ 
# Read stuff from recording file
############################################################

# Load recording file as an array.
def load_recording(recording_file, header=None, leads=None, key='val'):
    recording = loadmat(recording_file)[key]
    return recording


############################################################ 
# Lead modification and manipulation
############################################################

class lead_exctractor():
    """
    used to select specific leads or random choice of configurations

    Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Six leads: I, II, III, aVR, aVL, aVF
    Four leads: I, II, III, V2
    Three leads: I, II, V2
    Two leads: I, II

    """
    L2 = np.array([1,1,0,0,0,0,0,0,0,0,0,0])
    L3 = np.array([1,1,0,0,0,0,0,1,0,0,0,0])
    L4 = np.array([1,1,1,0,0,0,0,1,0,0,0,0])
    L6 = np.array([1,1,1,1,1,1,0,0,0,0,0,0])
    L8 = np.array([1,1,0,0,0,0,1,1,1,1,1,1])
    L12 = np.array([1,1,1,1,1,1,1,1,1,1,1,1])

    @staticmethod
    def get (x,num_leads,lead_indicator):
        if num_leads==None:
            # 100% Stochastic: force the model to learn from any combination of leads.
            num_leads = "stochastic"

        if num_leads == "stochastic":
            # Pick a random number of leads to keep (between 1 and 12)
            n_keep = random.randint(1, 12)
            # Randomly select which leads to keep
            all_leads = list(range(12))
            keep_indices = random.sample(all_leads, n_keep)
            mask = np.zeros(12)
            mask[keep_indices] = 1
            x = x * mask.reshape(12, 1)
            return x, lead_indicator * mask

        if num_leads==12:
            # Twelve leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
            return x,lead_indicator * lead_exctractor.L12

        if num_leads==8:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L8.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L8

        if num_leads==6:
            # Six leads: I, II, III, aVL, aVR, aVF
            x = x * lead_exctractor.L6.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L6

        if num_leads==4:
            # Six leads: I, II, III, V2
            x = x * lead_exctractor.L4.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L4

        if num_leads==3:
            # Three leads: I, II, V2
            x = x * lead_exctractor.L3.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L3

        if num_leads==2:
            # Two leads: II, V5
            x = x * lead_exctractor.L2.reshape(12,1)
            return x,lead_indicator * lead_exctractor.L2
        raise Exception("invalid-leads-number")
    

def expand_leads(recording,input_leads):
    # Final twelve lead signal
    output = np.zeros((12, recording.shape[1]))
    twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    # Conver items in list to lower case    
    twelve_leads = [k.lower() for k in twelve_leads] 
    input_leads = [k.lower() for k in input_leads]
    # Array to track which leads are originally present
    output_leads = np.zeros((12,)) 

    # Fill in the signal for recording with leads that are present and mark them with 1
    # Other rows of the signal stay zero and are marked as 0
    for i,k in enumerate(input_leads):
        if k in twelve_leads:
            idx = twelve_leads.index(k)
            output[idx,:] = recording[i,:]
            output_leads[idx] = 1
        else:
            print(f"Unknown lead {k}. Ignoring")
    return output,output_leads

############################################################ 
# Model loading
############################################################

def fold_batchnorm_in_state_dict(state_dict):
    """
    Mathematically folds BatchNorm2d parameters into their preceding Conv2d layers.

    For each Conv + BN pair, the BN is absorbed into the conv weights and a bias is
    produced so that: BN(Conv(x)) == FoldedConv(x) + bias, exactly.

    Formula:
        scale = gamma / sqrt(running_var + eps)
        W_new  = W * scale.view(-1,1,1,1)        (scale each output filter)
        b_new  = beta - scale * running_mean

    The BN keys are then removed from the returned state_dict.
    Supports the pairing structure:
        conv   -> bn
        rb_X.conv1 -> rb_X.bn1
        rb_X.conv2 -> rb_X.bn2

    Args:
        state_dict (dict): Raw checkpoint state dict from the legacy (BN-containing) model.

    Returns:
        dict: Cleaned state dict compatible with the new BN-free architecture.
    """
    sd = {k: v.clone() if hasattr(v, 'clone') else v for k, v in state_dict.items()}

    # Map each conv weight key to its paired bn prefix
    bn_pairs = {
        "conv.weight": "bn",
    }
    for block in ["rb_0", "rb_1", "rb_2", "rb_3", "rb_4"]:
        bn_pairs[f"{block}.conv1.weight"] = f"{block}.bn1"
        bn_pairs[f"{block}.conv2.weight"] = f"{block}.bn2"

    eps = 1e-5  # PyTorch BatchNorm default

    for conv_key, bn_prefix in bn_pairs.items():
        if conv_key not in sd:
            continue
        # Retrieve BN tensors
        gamma = sd[f"{bn_prefix}.weight"]          # shape [C_out]
        beta  = sd[f"{bn_prefix}.bias"]            # shape [C_out]
        mu    = sd[f"{bn_prefix}.running_mean"]    # shape [C_out]
        var   = sd[f"{bn_prefix}.running_var"]     # shape [C_out]

        # Compute scale per output channel
        scale = gamma / (var + eps).sqrt()          # shape [C_out]

        # Fold into conv weight: multiply each output filter by its scale
        W = sd[conv_key]                            # shape [C_out, C_in, kH, kW]
        sd[conv_key] = W * scale.view(-1, 1, 1, 1)

        # Compute folded bias: b_new = beta - scale * mu
        bias_key = conv_key.replace(".weight", ".bias")
        sd[bias_key] = beta - scale * mu

        # Remove all BN keys
        for suffix in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            sd.pop(f"{bn_prefix}.{suffix}", None)

    print("[INFO] BatchNorm folding complete. Remaining keys:")
    for k in sd.keys():
        print(f"  {k}")
    return sd


# Generic function for loading a model.
# Supports two formats:
#   - MODEL_{id}.pth  : raw state_dict saved by torch.save(model.state_dict(), ...)
#                       (produced by extract_fold_checkpoints.py from MLflow)
#   - MODEL_{id}.pickle: legacy pickle wrapper with {'state_dict':..., 'thresholds':..., 'classes':...}
def _load_model(model_directory, id, nOUT, device, dropout_rate=0.5):
    pth_path    = Path(model_directory, f'MODEL_{id}.pth')
    pickle_path = Path(model_directory, f'MODEL_{id}.pickle')

    model = {}
    model['classifier'] = NN(nOUT=nOUT, dropout_rate=dropout_rate).to(device)

    if pth_path.exists():
        # Raw state_dict or checkpoint from torch.save
        print(f"[INFO] Loading raw .pth checkpoint: {pth_path}")
        checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
        
        # If it's a full checkpoint dict, extract the state_dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                raw_state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                raw_state_dict = checkpoint['model_state_dict']
            else:
                raw_state_dict = checkpoint
        else:
            raw_state_dict = checkpoint
            
        model['thresholds'] = None
        model['classes']    = None
    elif pickle_path.exists():
        # Legacy pickle wrapper
        print(f"[INFO] Loading legacy .pickle model: {pickle_path}")
        with open(pickle_path, 'rb') as handle:
            input_data = pickle.load(handle)
        raw_state_dict = input_data['state_dict']
        model['thresholds'] = input_data.get('thresholds', None)
        model['classes']    = input_data.get('classes', None)
    else:
        raise FileNotFoundError(
            f"No model file found for id={id} in {model_directory}.\n"
            f"Looked for: {pth_path} and {pickle_path}"
        )

    # ── Step 1: Fold BatchNorm parameters into their preceding Conv2d layers ──
    # This produces a clean state_dict that matches the new BN-free architecture.
    folded_state_dict = fold_batchnorm_in_state_dict(raw_state_dict)

    # ── Step 2: Map the legacy linear head weights → new Conv1d head ─────────
    # Old head: Linear(526, 2) with weight [2, 526]
    #   indices 0:256   → x_avg  features (we keep)
    #   indices 256:512 → x_max  features (discarded — max pool removed)
    #   indices 512:524 → l      (12 lead indicators)
    #   indices 524:526 → r      (2 rhythm vector)
    # New head: Conv1d(270, 2, kernel_size=1) with weight [2, 270, 1]
    #   indices 0:256   → latent features
    #   indices 256:268 → l
    #   indices 268:270 → r
    if "head.weight" in folded_state_dict:
        old_head_w = folded_state_dict["head.weight"]  # [2, 526]
        old_head_b = folded_state_dict.get("head.bias", None)  # [2]

        new_head_w = torch.zeros(nOUT, 270, 1, device=old_head_w.device)
        new_head_w[:, :256,   0] = old_head_w[:, :256]    # x_avg → latent
        new_head_w[:, 256:268, 0] = old_head_w[:, 512:524] # l
        new_head_w[:, 268:270, 0] = old_head_w[:, 524:526] # r

        folded_state_dict["head.weight"] = new_head_w
        if old_head_b is not None:
            folded_state_dict["head.bias"] = old_head_b
        print("[INFO] Head weights mapped: x_avg→latent, l, r preserved; x_max discarded.")

    # ── Step 3: Load folded weights into the clean (no-BN/no-ReLU) model ─────
    model_state_dict = model['classifier'].state_dict()
    filtered_state_dict = {}
    for key, value in folded_state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
            else:
                print(f"[INFO] Skipping {key} due to size mismatch: {value.shape} vs {model_state_dict[key].shape}")
        else:
            print(f"[INFO] Skipping {key} as it's not in the current model structure.")

    model['classifier'].load_state_dict(filtered_state_dict, strict=False)
    model['classifier'].eval()
    return model


def set_requires_grad(model, layers_config):
    """
    layers_config: dict matching layer names (e.g., 'conv', 'rb_0') to boolean 
    (True = trainable, False = frozen)
    """
    for name, param in model.named_parameters():
        # Check if any key in layers_config is a prefix of the parameter name
        for layer_prefix, is_trainable in layers_config.items():
            if name.startswith(layer_prefix):
                param.requires_grad = is_trainable
                break

def finetune_model_prep(model, device):
    """
    Prepares the B-cos model for fine-tuning:
      1. Expands model.conv from 12→24 input channels.
         - First 12 channels: copy pre-trained weights exactly.
         - Next  12 channels: initialize with -W (negative of pre-trained weights).
         This preserves the mathematical equivalence: W·x = W·pos + (-W)·neg
         since pos - neg = x (the original signal).
      2. Bcosifies all Conv2d and Linear layers.
    """
    # 1. Modify the first layer to accept 24 channels (12 original + 12 inverted)
    old_conv = model.conv
    new_conv = nn.Conv2d(in_channels=24,
                         out_channels=old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=(old_conv.bias is not None)).to(device)

    with torch.no_grad():
        # First 12 channels: copy original pre-trained weights
        new_conv.weight[:, :12, :, :] = old_conv.weight
        # Next 12 channels: NEGATIVE of pre-trained weights.
        # Ensures W·pos + (-W)·neg = W·(pos - neg) = W·x (mathematically equivalent)
        new_conv.weight[:, 12:, :, :] = -old_conv.weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv = new_conv

    # 2. Convert Conv2d/Linear layers to Bcosified versions
    bcosify_model(model)

    # Note: freezing/unfreezing is handled by build_flexible_optimizer in the main training loop.
    return model