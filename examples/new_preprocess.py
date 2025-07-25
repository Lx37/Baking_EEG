import os
import mne
import numpy as np

# Provide the absolute path to your already cropped .fif file.
# (Even if the file has an EGI header, if it's saved as a FIF file, use read_raw_fif.)
fname = r"C:\Users\adminlocal\Desktop\new_conscious\TWB1_RS_20200605_141122_raw.fif"

print("Loading file:", fname)
raw = mne.io.read_raw_fif(fname, preload=True)
print(raw)

# # Apply a 50 Hz notch filter to remove power line noise
# raw.notch_filter(freqs=50)

# # Apply a bandpass filter from 1 to 40 Hz
# raw.filter(1., 40.)

# # Downsample the data to 250 Hz
# raw.resample(250, npad="auto")

# # Open an interactive plot so you can mark bad channels.
# # In the plot window, click on the channel names to toggle their status as "bad".
# print("Please mark bad channels interactively by clicking on channel names, then close the window.")
# raw.plot(block=True)

# # After closing the plot window, the bad channels will be stored in raw.info['bads']
# print("Selected bad channels:", raw.info['bads'])

# # Build the output filename by appending '_preproc.fif' to the base name
# base_name = os.path.splitext(os.path.basename(fname))[0]
# output_file = os.path.join(os.path.dirname(fname), base_name + "_preproc.fif")
# print("Saving preprocessed data to:", output_file)

# # Save the preprocessed data (including the bad channels list) as a .fif file
# raw.save(output_file, overwrite=True)

# print("Preprocessing complete. File saved as:", output_file)


import os
import numpy as np
import mne

# --- Configuration (adjust these values as needed) ---
class Config:
    sfreq = 250  # sampling frequency (Hz)
    epochs_reject_con = dict(eeg=200e-6, eog=100e-6)  # rejection thresholds
    data_con_path = "Connectivity_Epochs/"  # subfolder or path string (include trailing slash if needed)
    prefix_epo_conn = "_epo_conn.fif"  # suffix to be added to the epochs file

cfg = Config()

# --- Epoching Function (as provided) ---
def get_epochs_connectivity(data, sub, proto, data_save_dir, cfg, save=True, verbose=True, plot=True):
    # Ensure the sampling frequency is as expected.
    assert data.info['sfreq'] == cfg.sfreq, 'Sampling frequency mismatch!'

    # Define new triggers: 16 events of 10 s = 160 s (for shorter recordings)
    event_id = 303
    nb_event = 60   # for patients with less available time
    size_epoch_s = 10

    # Determine the onset of the first epoch
    if proto in ['PP', 'LG']:
        data_events = mne.find_events(data, stim_channel='STI 014')
        if len(data_events) == 0:
            print("No trigger events found. Cannot epoch data.")
            return
        onset_init = data_events[0][0]
    elif proto == 'Resting':
        onset_init = int(cfg.sfreq * 60)  # start after 60 s for resting
    else:
        print('Error: Protocol name not recognized.')
        return

    # Create event onsets and event array
    onsets = np.arange(start=onset_init, 
                       stop=onset_init + (nb_event * cfg.sfreq * size_epoch_s),
                       step=cfg.sfreq * size_epoch_s, dtype='int64')
    events = np.vstack((onsets, np.ones(nb_event, int), event_id * np.ones(nb_event, int))).T
    print('New Events: ', events)
    print(data)

    # Define epoch time window (each epoch is 10 s)
    tmin, tmax = 0, size_epoch_s

    # Pick EEG channels (adjust exclusions as needed)
    picks_eeg = mne.pick_types(data.info, eeg=True, exclude=[])
    epochs = mne.Epochs(data, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=(None, None), picks=picks_eeg, reject=cfg.epochs_reject_con,
                        preload=True, detrend=1, reject_by_annotation=False)
    if verbose:
        print('Epochs info: ', epochs.info)
        print('Number of epochs: ', len(epochs))

    if plot:
        epochs.plot(title=sub + " " + proto + ' - Click to manually reject epochs if needed', 
                    show=True, block=True, scalings=dict(eeg=200e-6, eog=100e-6))

    if save:
        # Construct the save path: concatenate the data_save_dir, data_con_path, subject, protocol, and suffix.
        # Ensure that directories exist.
        save_dir = os.path.join(data_save_dir, cfg.data_con_path)
        os.makedirs(save_dir, exist_ok=True)
        epochs_name = os.path.join(save_dir, sub + '_' + proto + cfg.prefix_epo_conn)
        print("Saving epochs data to: " + epochs_name)
        epochs.save(epochs_name, overwrite=True)

    return epochs

# --- Main Script: Load Preprocessed File and Epoch It ---
# Set the directory where your preprocessed file is located.
# Replace 'X' with your actual directory.
preproc_dir = r"C:\Users\adminlocal\Desktop\new_conscious\preprocess"

# Set the full filename of your preprocessed file.
# For example, if your preprocessed file is named "TPC2_Resting_preproc.fif":
fname = os.path.join(preproc_dir, "TPC2_Resting_preproc.fif")
print("Loading preprocessed file:", fname)
data = mne.io.read_raw_fif(fname, preload=True)
data.plot(block=True)  # Optional: view the raw data

# Set subject and protocol (modify if needed)
sub = "TPC2"
proto = "Resting"

# Use the same preprocessed directory as the save directory.
data_save_dir = preproc_dir

# Get epochs (this will also save the epochs file)
data.plot(block=True)
epochs = get_epochs_connectivity(data, sub, proto, data_save_dir, cfg, save=True, verbose=True, plot=True)
