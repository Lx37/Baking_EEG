# # CODE TO KNOW HOW MANY AND WHICH MISSSING CHANNELS THERE ARE

import os
import mne

def load_specific_file(directory, filename):
    """
    Loads a specific preprocessed EEG file from the given directory.
    
    Parameters
    ----------
    directory : str
        The directory where the preprocessed EEG files are stored.
    filename : str
        The specific file name to load (e.g., 'patient01_Resting_preproc.fif').
    
    Returns
    -------
    data : instance of mne.io.Raw
        The loaded EEG data.
    """
    file_path = os.path.join(directory, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Assumes file is in .fif format
    if file_path.lower().endswith('.fif'):
        data = mne.io.read_raw_fif(file_path, preload=True, verbose=True)
    else:
        raise ValueError("Unsupported file format. Only .fif files are supported.")
    
    print("Loaded file:", file_path)
    print("Data info:", data.info)
    
    return data

def list_missing_channels_from_EGI(data):
    """
    Compares the standard full EGI montage channels (GSN-HydroCel-129) 
    with the channels present in the loaded data and returns a list of missing channels.
    
    Parameters
    ----------
    data : instance of mne.io.Raw
        The EEG data after applying the appropriate montage.
    
    Returns
    -------
    missing_channels : list
        List of channel names missing from the standard 129-channel EGI montage.
    """
    # Get the standard EGI full montage channel names (GSN-HydroCel-129)
    egi_montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    full_egi_channels = egi_montage.ch_names
    
    # Extract the EEG channel names from the data
    actual_channels = data.pick_types(eeg=True).ch_names
    
    # Compute missing channels by subtracting the actual channels from the full EGI channels
    missing_channels = [ch for ch in full_egi_channels if ch not in actual_channels]
    
    print("Missing channels from the EGI full montage:", missing_channels)
    print("Actual channels:", actual_channels)
    return missing_channels

# ----- Example usage -----
directory = r'C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_preproc'
filename = 'FP102_Resting_preproc.fif'  # Replace with the actual file name
data = load_specific_file(directory, filename)
missing_channels = list_missing_channels_from_EGI(data)


# #CODE TO KNOW THE SHAPE OF DATA

# import os
# import mne
# import numpy as np

# # Configuration
# data_epochs_path = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_connectivity"
# save_log_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\spectral_power\psd_vs"
# prefix_epo_conn = "_epo_conn.fif"

# # Define your protocols and subjects (adjust as needed)
# protocols = ['LG']
# subjects =  [
#     'DR92','AG42','FM60'
# ]



# # Create a list to store shape information
# log_lines = []
# log_lines.append("Subject,Protocol,Average PSD Shape (n_channels, n_freqs)")

# # Loop over each protocol and subject
# for proto in protocols:
#     for sub in subjects:
#         # Build the filename
#         fname = os.path.join(data_epochs_path, f"{sub}_{proto}{prefix_epo_conn}")
#         if not os.path.exists(fname):
#             log_lines.append(f"{sub},{proto},File not found")
#             continue
#         try:
#             print(f"Processing file: {fname}")
#             epochs = mne.read_epochs(fname, proj=False, preload=True, verbose=False)
#             # Compute PSD (0 to 45 Hz) using all channels
#             epo_spectrum = epochs.compute_psd(fmin=0, fmax=45, picks='all')
#             # Get PSD data and frequency vector
#             psds, freqs = epo_spectrum.get_data(return_freqs=True)
#             # Average the PSD over epochs (resulting shape: (n_channels, n_freqs))
#             avg_psd = np.average(psds, axis=0)
#             shape_info = avg_psd.shape  # a tuple (n_channels, n_freqs)
#             log_lines.append(f"{sub},{proto},{shape_info}")
#         except Exception as e:
#             log_lines.append(f"{sub},{proto},Error: {e}")

# # Ensure the save directory exists
# os.makedirs(save_log_dir, exist_ok=True)
# log_file = os.path.join(save_log_dir, "psd_shapes_log.txt")

# # Write the log lines to the text file
# with open(log_file, "w") as f:
#     for line in log_lines:
#         f.write(line + "\n")

# print(f"Saved log file with PSD shapes to: {log_file}")
