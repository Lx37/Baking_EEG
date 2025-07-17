import os
import glob
import numpy as np

BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V17"


def scan_npz_keys(base_path):
    pattern = os.path.join(base_path, "**", "decoding_results_full.npz")
    files = glob.glob(pattern, recursive=True)
    print(f"Trouvé {len(files)} fichiers NPZ.")
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as data:
                print(f"\nFichier: {f}")
                print("Clés:", list(data.keys()))
        except Exception as e:
            print(f"Erreur lors du chargement de {f}: {e}")

if __name__ == "__main__":
    scan_npz_keys(BASE_RESULTS_DIR)
