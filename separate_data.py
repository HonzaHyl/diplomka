import os
import shutil
import pandas as pd

def main():
    base_dir = '/srv/home/jhyl/Afib_recurrence/diplomka'
    data_dir = os.path.join(base_dir, 'finetune_data')
    features_file = os.path.join(base_dir, 'features.csv')
    sr_dir = os.path.join(data_dir, 'SR_before')
    fs_dir = os.path.join(data_dir, 'pathology_before')

    # Create target directories
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(fs_dir, exist_ok=True)

    # Read features
    df = pd.read_csv(features_file)

    # Count how many files copied
    copied_count = 0

    # Copy files
    for index, row in df.iterrows():
        patient_id = str(row['ID'])
        has_pathology = int(row['patology_before'])
        
        target_dir = fs_dir if has_pathology == 1 else sr_dir
        
        # Extensions to look for
        extensions = ['.hea', '.mat', '.npy']
        
        for ext in extensions:
            src_file = os.path.join(data_dir, f"{patient_id}{ext}")
            if os.path.exists(src_file):
                dst_file = os.path.join(target_dir, f"{patient_id}{ext}")
                shutil.copy2(src_file, dst_file)
                copied_count += 1
                # print(f"Copied {patient_id}{ext} to {os.path.basename(target_dir)}")

    print(f"Done separating data. Copied {copied_count} files in total.")

if __name__ == "__main__":
    main()
