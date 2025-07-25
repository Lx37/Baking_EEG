from getpass import getuser
user = getuser()  # Username of the user running the scripts
print('User is:', user)
import sys
if user == 'adminlocal':    
    sys.path.append('C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera\\Baking_EEG')
import os
import mne
from Baking_EEG import config as cfg
from Baking_EEG import _3_epoch as epoch

######################################
############ Your input ##############
######################################

# Input: Full path to the preprocessed .fif file
fif_file_path = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_preproc\FP102_Resting_preproc.fif"

# Subject ID and protocol must match the ones in the file name
subject_id = "FP102"
protocol = "Resting"

# Where to save epoch data (same as your usual path)
data_save_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera"

# Set parameters
save = True
verbose = True
plot = True

######################################
############ Load + Epoch ############
######################################

print(f"######## Loading preprocessed data: {fif_file_path}")
if not os.path.exists(fif_file_path):
    raise FileNotFoundError(f"Could not find the .fif file at: {fif_file_path}")

# Load raw data
data = mne.io.read_raw_fif(fif_file_path, preload=True)

if data.info['sfreq'] != cfg.sfreq:
    print(f"Resampling from {data.info['sfreq']} Hz to {cfg.sfreq} Hz...")
    data.resample(cfg.sfreq, npad="auto")


# Run epoching (as in your original script)
print(f"######## Epoching data {subject_id}_{protocol}")
data = epoch.get_epochs_connectivity(
    data,
    subject_id,
    protocol,
    data_save_dir,
    cfg,
    save=save,
    verbose=verbose,
    plot=plot
)
