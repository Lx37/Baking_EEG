
import os
import glob
import numpy as np

BASE_RESULTS_DIR = "/Users/tombalay/Desktop/ENSC/Stage 2A CAP/resultats_publi_baking_and_code/Baking_EEG_results_V17/"
all_files = glob.glob(os.path.join(BASE_RESULTS_DIR, '**', '*decoding_results_full.npz'), recursive=True)

if all_files:
    file_path = all_files[0]
    print(f"Inspecting file: {file_path}")
    try:
        with np.load(file_path, allow_pickle=True) as data:
            print("Keys:", list(data.keys()))
            for key in data.keys():
                if 'scores' in key and 'mean' not in key:
                    print(f"Found potential raw scores: {key}, shape: {data[key].shape}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No files found.")
