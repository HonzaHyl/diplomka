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

from device_selector import DeviceSelector

selector = DeviceSelector()
DEVICE = selector.select(1)[0]


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
            # random choice output
            num_leads = random.choice([12,8,6,4,3,2])

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

# Generic function for loading a model.
def _load_model(model_directory,id, nOUT):
    filename = Path(model_directory,f'MODEL_{id}.pickle')
    model = {}
    with open(filename, 'rb') as handle:
        input_data = pickle.load(handle)

    model['classifier'] = NN(nOUT=nOUT).to(DEVICE)
    
    # Filter state_dict for size mismatches (e.g., when pooling changes the head size)
    checkpoint_state_dict = input_data['state_dict']
    model_state_dict = model['classifier'].state_dict()
    
    filtered_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                filtered_state_dict[key] = value
            else:
                print(f"[INFO] Skipping {key} due to size mismatch: {value.shape} vs {model_state_dict[key].shape}")
        else:
            print(f"[INFO] Skipping {key} as it's not in the current model structure.")

    model['classifier'].load_state_dict(filtered_state_dict, strict=False)
    model['classifier'].eval()
    
    model['thresholds'] = input_data['thresholds']
    model['classes'] = input_data['classes']
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

def finetune_model_prep(model):
    # 1. Modify the first layer to accept 24 channels (12 + 12 inverted)
    old_conv = model.conv
    new_conv = nn.Conv2d(in_channels=24,
                         out_channels=old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False).to(DEVICE)
    
    # Initialize weights: first 12 channels get original weights, next 12 get zeros
    with torch.no_grad():
        new_conv.weight[:, :12, :, :] = old_conv.weight
        new_conv.weight[:, 12:, :, :] = 0.0
    
    model.conv = new_conv

    # 2. Create new last layer with 2 output features
    pool_out_features = model.fc_1.in_features - 12  # = 256
    in_features = pool_out_features + 24             # = 280
    model.fc_1 = nn.Linear(in_features, 2).to(DEVICE)
    
    # 3. CRITICAL FIX: Convert Conv2d/Linear layers to Bcosified versions FIRST
    bcosify_model(model)

    # Note: We no longer unfreeze everything here. 
    # That is handled by set_requires_grad in the main training loop.
    return model