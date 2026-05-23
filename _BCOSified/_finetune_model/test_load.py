import sys
import torch
import warnings
warnings.filterwarnings('ignore')

sys.path.append("/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model")
from model_structure import NN
from helper_code import _load_model, finetune_model_prep

try:
    print("Loading base model...")
    loaded_model = _load_model("/srv/home/jhyl/Afib_recurrence/diplomka/_BCOSified/_finetune_model/", 1, nOUT=26)
    classifier = loaded_model["classifier"]
    
    print("Applying B-cosification prep...")
    prep_m = finetune_model_prep(classifier)
    
    print("Success! The architecture is synced and weights are loaded correctly.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
