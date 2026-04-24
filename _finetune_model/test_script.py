from pathlib import Path
import pickle
from model_structure import NN
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = "/srv/home/jhyl/Afib_recurrence/diplomka/results/Training_20260324_140945/model"

def _load_model(model_directory,id,nOUT):
    filename = Path(model_directory,f'MODEL_{id}.pickle')
    model = {}
    with open(filename, 'rb') as handle:
        input = pickle.load(handle)

    model['classifier'] = NN(nOUT=nOUT).to(DEVICE)
    model['classifier'].load_state_dict(input['state_dict'], strict=False)
    model['classifier'].eval()
    model['thresholds'] = input['thresholds']
    model['classes'] = input['classes']
    return model

if __name__=="__main__":
    model = _load_model(MODEL_DIR, 1, 26)