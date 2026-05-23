import os
import torch
import numpy as np
from scipy import signal
from tqdm import tqdm
from helper_code import load_header, load_recording, get_frequency, find_header_files, expand_leads

def preprocess_folder(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    header_files = find_header_files(input_dir)
    
    # Filter parameters (same as your original code)
    b, a = signal.butter(3, [1 / 250, 47 / 250], 'bandpass')

    print(f"Processing {len(header_files)} files from {input_dir}...")

    for h_file in tqdm(header_files):
        file_id = os.path.basename(h_file).replace(".hea", ".npy")
        output_path = os.path.join(output_dir, file_id)
        
        # Resume option: skip if already processed
        if os.path.exists(output_path):
            continue
            
        # 1. Load raw data
        header = load_header(h_file)
        try:
            fs = get_frequency(header)
            recording = load_recording(h_file.replace(".hea", ".mat"))
        except Exception as e:
            print(f"Failed to load {h_file}: {e}. Skipping...")
            continue
        
        # 2. Standardize leads to 12 leads
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        data, _ = expand_leads(recording, input_leads=leads)
        data = np.nan_to_num(data)

        # 3. Resample to 500Hz
        if fs != 500:
            num_samples = int(data.shape[1] * 500 / fs)
            data = signal.resample(data, num_samples, axis=1)

        # 4. Filter
        data = signal.filtfilt(b, a, data)

        # --- NEW: 5. Global Z-score Normalization (Per Lead) ---
        mu = np.nanmean(data, axis=-1, keepdims=True)
        std = np.nanstd(data, axis=-1, keepdims=True) + 1e-8
        data = (data - mu) / std
        data = np.nan_to_num(data) # Safety net for any weird zero-variance leads
        # -------------------------------------------------------

        # Convert to float32 numpy array
        data_np = data.astype(np.float32)

        # 6. Save as .npy
        np.save(output_path, data_np)

if __name__=="__main__":
    # Run for both folders
    preprocess_folder("/srv/home/jhyl/Afib_recurrence/", "/srv/home/jhyl/Afib_recurrence/")