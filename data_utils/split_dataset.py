import os
import shutil
import argparse
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Deterministic Train/Test Splitter for Multi-extension Files")
    parser.add_argument('--input_folder', type=str, help="Path to the folder containing the files")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for deterministic splitting")
    parser.add_argument('--action', choices=['copy', 'move'], default='copy', help="Whether to copy or move files to the new folders")

    args = parser.parse_args()

    input_path = Path(args.input_folder)
    
    if not input_path.is_dir():
        print(f"Error: {input_path} is not a valid directory.")
        return

    # Target extensions that form a single data sample
    extensions = ['.hea', '.mat', '.npy']

    # Find all unique basenames. We look at all files and group by filename without extension.
    basenames = set()
    for f in input_path.iterdir():
        if f.is_file() and f.suffix in extensions:
            basenames.add(f.stem)

    # Sort basenames to ensure absolute determinism regardless of OS file iteration order
    basenames = sorted(list(basenames))
    
    if not basenames:
        print(f"No files with extensions {extensions} found in {input_path}.")
        return

    # Deterministic split
    random.seed(args.seed)
    # Shuffle in place
    random.shuffle(basenames)

    split_idx = int(len(basenames) * (1 - args.test_size))
    train_basenames = basenames[:split_idx]
    test_basenames = basenames[split_idx:]

    print(f"Found {len(basenames)} unique sample basenames.")
    print(f"Splitting into {len(train_basenames)} train and {len(test_basenames)} test samples.")

    # Create directories inside the input folder
    train_dir = input_path / 'train'
    test_dir = input_path / 'test'
    
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    def process_files(basename_list, target_dir):
        for basename in basename_list:
            for ext in extensions:
                src_file = input_path / f"{basename}{ext}"
                if src_file.exists():
                    dst_file = target_dir / f"{basename}{ext}"
                    if args.action == 'copy':
                        shutil.copy2(src_file, dst_file)
                    else:
                        shutil.move(src_file, dst_file)

    print(f"{args.action.capitalize()}ing files to {train_dir} ...")
    process_files(train_basenames, train_dir)
    
    print(f"{args.action.capitalize()}ing files to {test_dir} ...")
    process_files(test_basenames, test_dir)

    print("Done!")

if __name__ == "__main__":
    main()
