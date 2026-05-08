import os, glob, numpy as np
from scipy.io import loadmat
from ipywidgets import interact, fixed
import plotly.graph_objects as go

DATA_DIR = "/srv/home/jhyl/Afib_recurrence/finetune_data_all"
LEADS    = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def parse_header(path):
    with open(path) as f:
        lines = f.read().splitlines()
    p = lines[0].split()
    return float(p[2]), [lines[i+1].split()[-1] for i in range(int(p[1]))]

_cache = {}
def load_ecg(base):
    if base not in _cache:
        _cache.clear()
        fs, leads = parse_header(os.path.join(DATA_DIR, base + ".hea"))
        sig = loadmat(os.path.join(DATA_DIR, base + ".mat"))["val"].astype(np.float32)
        _cache[base] = (sig, leads, fs)
    return _cache[base]

bases = sorted(
    [os.path.splitext(os.path.basename(p))[0]
     for p in glob.glob(os.path.join(DATA_DIR, "*.hea"))],
    key=lambda x: int(x) if x.isdigit() else x
)

@interact(file=bases, lead=LEADS)
def show(file, lead):
    sig, leads, fs = load_ecg(file)
    if lead not in leads:
        print(f"Lead {lead} not in this file"); return
    y = sig[leads.index(lead)]
    t = np.arange(len(y)) / fs
    go.Figure(
        go.Scatter(x=t, y=y, mode="lines", line=dict(width=0.8)),
        layout=dict(title=f"{file} — {lead}", xaxis_title="Time (s)",
                    yaxis_title="Amplitude", height=400)
    ).show()
