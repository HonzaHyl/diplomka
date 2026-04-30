import numpy as np
import random
import pickle
import os
import torch
import torch.nn as nn

from scipy.io import loadmat
from pathlib import Path

from device_selector import DeviceSelector
from model_structure import NN

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
    for f in sorted(os.listdir(data_directory)):   # sorted() → deterministic on Linux
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
    def get (x,num_leads,lead_indicator, rng=None):
        if num_leads==None:
            # Deterministic random choice: use a seeded numpy RNG when provided.
            _rng = rng if rng is not None else np.random.default_rng()
            num_leads = int(_rng.choice([12, 8, 6, 4, 3, 2]))

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
        input = pickle.load(handle)

    model['classifier'] = NN(nOUT=nOUT).to(DEVICE)
    try:
        model['classifier'].load_state_dict(input['state_dict'])
    except RuntimeError:
        print("Trying to load stat dict into model with strict=False")
        state_dict = input['state_dict']
    
        # Manually delete the incompatible keys so PyTorch doesn't even see them
        state_dict.pop('fc_1.weight', None)
        state_dict.pop('fc_1.bias', None)
        model['classifier'].load_state_dict(state_dict, strict=False)
    model['classifier'].eval()
    model['thresholds'] = input['thresholds']
    model['classes'] = input['classes']
    return model


def finetune_model_prep(model):
    # Unfreeze all layers for full architecture finetuning
    for param in model.parameters():
        param.requires_grad = True

    # Create new last layer with 2 output features
    in_features = model.head[-1].in_features
    model.head[-1] = nn.Linear(in_features, 2)


    # Verify which parameters are trainable
    print("\n--- Trainable Parameters ---")
    for name, param in model.named_parameters():
            print(param.requires_grad)

    print(model.head[-1].out_features)

    return model