#for connectivity

import os
import pandas as pd

# === CONFIG ===
subjects = [
   

    "DA75",
    "GT50",
    "JA71",
    "MC58"
    
]


band = "gamma"  # ← CHOOSE the band to process
protocols = ["Resting", "LG", "PP"]
roi_keys_order = [
    ("ROI_Frontal_Droit", "ROI_Frontal_Gauche"),  # FD-FG
    ("ROI_Frontal_Droit", "ROI_Caudal_Droit"),    # FD-CD
    ("ROI_Caudal_Droit", "ROI_Caudal_Gauche"),    # CD-CG
    ("ROI_Frontal_Gauche", "ROI_Caudal_Gauche"),  # FG-CG
]

root_path = r"C:\Users\adminlocal\Desktop\Connect_new_ROI\mcs"
output = r'C:\Users\adminlocal\Desktop'
output_path = os.path.join(output, f"FFT_mcs_{band}_summary.xlsx")
output_file = os.path.join(output, f"FFT_mcs_{band}_summary.xlsx")


# === GATHER DATA ===
all_data = []

for subject in subjects:
    row = [subject]
    for proto in protocols:
        file = f"{subject}_{proto}_wpli2_debiased_{band}_ROI.xlsx"
        path = os.path.join(root_path, subject, file)

        if not os.path.exists(path):
            print(f"❌ Missing file: {file}")
            row.extend([None] * 4)
            continue

        try:
            df = pd.read_excel(path, index_col=0)
        except Exception as e:
            print(f"⚠️ Could not load {file} —", e)
            row.extend([None] * 4)
            continue

        for roi1, roi2 in roi_keys_order:
            val = df.loc[roi1, roi2] if roi1 in df.index and roi2 in df.columns else None
            row.append(val)

    all_data.append(row)

# === HEADER ROW ===
header = ["Subject"]
for proto in protocols:
    header += [f"{proto}_FD-FG", f"{proto}_FD-CD", f"{proto}_CD-CG", f"{proto}_FG-CG"]

# === SAVE TO EXCEL ===
pd.DataFrame(all_data, columns=header).to_excel(output_file, index=False)
print(f"✅ Export complete: {output_file}")

# # to see number of epochs

# import os
# import sys
# import mne

# # === Define your folder path here ===
# data_folder = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_connectivity"  # <-- Change this only once!

# def count_epochs_for_patient(patient_file):
#     """
#     Load a specific patient's .fif epoched file and print the number of epochs.
#     """
#     file_path = os.path.join(data_folder, patient_file)

#     if not os.path.exists(file_path):
#         print(f"File not found: {file_path}")
#         return

#     try:
#         epochs = mne.read_epochs(file_path, preload=False)
#         print(f"{patient_file}: {len(epochs)} epochs")
#     except Exception as e:
#         print(f"Error loading {patient_file}: {e}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python count_epochs.py <patient_file>")
#         print("Example: python count_epochs.py Patient001_rest-epo.fif")
#         sys.exit(1)

#     patient_file = sys.argv[1]
#     count_epochs_for_patient(patient_file)
