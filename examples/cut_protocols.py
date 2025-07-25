# import os
# import mne

# # Main directories
# data_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\All_Raw_Data\data_EEG_battery_2019-"
# save_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\All_Raw_Data\data_EEG_battery_2019-\cut"

# # Create the output directory if it doesn't exist.
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#     print("Created output directory:", save_dir)

# # Define sample index mappings for each patient (only TWB1 and TPC2).
# # Adjust these values as needed.
# index_mapping = {
#     "TWB1": {
#          "LG": (1300, 4000),
#          "PP": (4750, 5700),
#          "RS": (5700, None)  # RS (Resting) from 5700 to end
#     },
#     "TPC2": {
#          "PP": (2050, 3000),
#          "LG": (3000, 5700),
#          "RS": (5700, None)
#     }
# }

# # Expected protocol tokens (in uppercase) for filenames.
# expected_protocols = {"LG", "PP", "RS"}

# # Process each subfolder in the main directory.
# for patient_folder in os.listdir(data_dir):
#     full_patient_path = os.path.join(data_dir, patient_folder)
#     if not os.path.isdir(full_patient_path):
#         continue  # Skip files in the root directory

#     # Use uppercase to match keys in index_mapping.
#     patient_key = patient_folder.upper()
#     if patient_key not in index_mapping:
#         print(f"Skipping folder {patient_folder}: no index mapping available.")
#         continue

#     print(f"\nProcessing patient folder: {patient_folder}")
#     mapping = index_mapping[patient_key]

#     # Process each .mff file in the patient folder.
#     for filename in os.listdir(full_patient_path):
#         if not filename.lower().endswith('.mff'):
#             continue

#         file_path = os.path.join(full_patient_path, filename)
#         print(f"\nLoading file: {file_path}")
#         try:
#             # Use preload=False to avoid loading the entire file into memory immediately
#             raw = mne.io.read_raw_egi(file_path, preload=False)
#         except Exception as e:
#             print(f"Error loading {filename}: {e}")
#             continue

#         sfreq = raw.info['sfreq']
#         print(f"Sampling frequency: {sfreq} Hz")

#         # Split the filename by underscores.
#         tokens = filename.split('_')

#         # Check if a protocol token is present as the second token.
#         if len(tokens) > 1 and tokens[1].upper() in expected_protocols:
#             protocol = tokens[1].upper()
#             if protocol not in mapping:
#                 print(f"No index mapping for protocol {protocol} for patient {patient_folder}. Skipping file {filename}.")
#                 continue

#             start_idx, stop_idx = mapping[protocol]
#             tmin = start_idx / sfreq
#             tmax = stop_idx / sfreq if stop_idx is not None else None

#             try:
#                 # Crop the data; note: no "copy" keyword here.
#                 raw.crop(tmin=tmin, tmax=tmax)
#             except Exception as e:
#                 print(f"Error cropping {filename} for protocol {protocol}: {e}")
#                 continue

#             outname = f"{patient_key}_{protocol}_raw.fif"
#             outpath = os.path.join(save_dir, outname)
#             try:
#                 # Save using float32 to reduce memory usage
#                 raw.save(outpath, fmt='single', overwrite=True)
#                 print(f"Saved segment: {outpath}")
#             except Exception as e:
#                 print(f"Error saving {outpath}: {e}")

#         else:
#             # If no valid protocol token is found in the filename,
#             # assume the file is a continuous recording and extract all defined protocols.
#             for protocol, (start_idx, stop_idx) in mapping.items():
#                 tmin = start_idx / sfreq
#                 tmax = stop_idx / sfreq if stop_idx is not None else None

#                 # Re-read the file for each segment to avoid holding large data in memory.
#                 try:
#                     raw_segment = mne.io.read_raw_egi(file_path, preload=False)
#                 except Exception as e:
#                     print(f"Error loading {filename} for protocol {protocol}: {e}")
#                     continue

#                 try:
#                     raw_segment.crop(tmin=tmin, tmax=tmax)
#                 except Exception as e:
#                     print(f"Error cropping {filename} for protocol {protocol}: {e}")
#                     continue

#                 outname = f"{patient_key}_{protocol}_raw.fif"
#                 outpath = os.path.join(save_dir, outname)
#                 try:
#                     raw_segment.save(outpath, fmt='single', overwrite=True)
#                     print(f"Saved segment: {outpath}")
#                 except Exception as e:
#                     print(f"Error saving {outpath}: {e}")


import os
import glob
import mne

# Directory containing the .mff file
mff_dir = r"C:\Users\adminlocal\Desktop\new_conscious"

# Search for .mff files in the directory
mff_files = glob.glob(os.path.join(mff_dir, "*.mff"))
if not mff_files:
    raise FileNotFoundError(f"No .mff files found in {mff_dir}")

# For this example, we'll process the first .mff file found.
mff_file = mff_files[0]
print("Processing file:", mff_file)

# Load the .mff file using MNE's EGI reader (adjust parameters as needed)
raw = mne.io.read_raw_egi(mff_file, preload=False)

# Crop the data: cut from 5700 seconds until the end.
raw.crop(tmin=5700)  # tmax defaults to the end of the recording

# Build the output filename by appending "_cut.fif" to the base name
base_name = os.path.splitext(os.path.basename(mff_file))[0]
output_file = os.path.join(mff_dir, base_name + "_cut.fif")
print("Saving cropped file as:", output_file)

# Save the cropped data to a .fif file
raw.save(output_file, overwrite=True)

print("Cropping and saving complete.")

