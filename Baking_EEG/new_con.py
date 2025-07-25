import numpy as np
import pandas as pd
import os
import mne
from mne_connectivity import spectral_connectivity_epochs
import os
import sys

# compute the path of the *outer* Baking_EEG folder
this_file = os.path.abspath(__file__)
inner_pkg_dir = os.path.dirname(this_file)
outer_pkg_dir = os.path.dirname(inner_pkg_dir)

# put the outer folder on sys.path
if outer_pkg_dir not in sys.path:
    sys.path.insert(0, outer_pkg_dir)


from Baking_EEG import config as cfg

# # logging info
# import logging
from datetime import datetime

# Ensure the correct QT backend
os.environ["QT_API"] = "pyside6"

def connectivity_overSubs(subs, data_save_dir, selected_chans, proto, cfg, save=True):
    """
    Compute spectral connectivity for each subject and ROI, then save ROI connectivity excels per subject and averaged across subjects.

    Parameters
    ----------
    subs : list of str
        List of subject identifiers.
    data_save_dir : str
        Base directory for input/output data.
    selected_chans : list or 'All'
        List of channels to use, or 'All' for all channels.
    proto : str
        Protocol or task identifier used in filenames.
    cfg : object
        Configuration object with attributes:
        - con_event_ids: list of event IDs
        - con_freq_bands: dict mapping band names to (fmin, fmax)
        - con_tmin: float, start time for connectivity window
        - con_method: string, connectivity method
        - data_con_path: path relative to data_save_dir for epoched files
        - prefix_epo_conn: file suffix for epochs
        - result_con_path: path relative to data_save_dir for saving results
        - EGI_con_chan: list of channel names (for consistency check)
        - con_all_ROI_chan: dict mapping ROI names to lists of channel names
        - con_vmin, con_vmax: plotting limits (unused)
    save : bool
        Whether to save the results to disk.

    Returns
    -------
    None
    """
    event_ids = cfg.con_event_ids
    freq_bands = cfg.con_freq_bands
    tmin = cfg.con_tmin

    # Determine channel set
    if selected_chans != 'All':
        chans_names = selected_chans.copy()
    else:
        # Load first subject to get full channel list
        first_file = os.path.join(data_save_dir, cfg.data_con_path,
                                  f"{subs[0]}_{proto}{cfg.prefix_epo_conn}")
        epochs = mne.read_epochs(first_file, proj=False, verbose=True, preload=True)
        chans_names = epochs.ch_names
        # Consistency check
        assert chans_names == cfg.EGI_con_chan, \
            "Channel names mismatch across subjects."

    # Map ROI channel names to indices
    All_ROI = {}
    for roi, chan_list in cfg.new_ROI_chan.items():
        All_ROI[roi] = [chans_names.index(ch) for ch in chan_list]

    n_subs = len(subs)

    # Loop over frequency bands
    for band_name, (fmin, fmax) in freq_bands.items():
        # Initialize list to accumulate ROI matrices per subject
        roi_matrices = []

        # Loop over subjects
        for sub in subs:
            # Load subject epochs
            epo_file = os.path.join(data_save_dir, cfg.data_con_path,
                                    f"{sub}_{proto}{cfg.prefix_epo_conn}")
            epochs = mne.read_epochs(epo_file, proj=False, verbose=False, preload=True)

            # Loop over event IDs (assuming a single event for ROI averaging)
            # If multiple events needed, extend accordingly
            event_id = event_ids[0]
            sel_epochs = epochs[event_id]

            # Compute connectivity
            con = spectral_connectivity_epochs(
                sel_epochs, method=cfg.con_method, mode='multitaper',
                sfreq=epochs.info['sfreq'], fmin=fmin, fmax=fmax,
                faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)
            con_mat = con.get_data(output='dense')[:, :, 0]

            # Compute ROI-ROI connectivity
            roi_data = np.zeros((len(All_ROI), len(All_ROI)))
            roi_labels = list(All_ROI.keys())
            for i, roi_i in enumerate(roi_labels):
                for j, roi_j in enumerate(roi_labels):
                    vals = con_mat[np.ix_(All_ROI[roi_i], All_ROI[roi_j])]
                    roi_data[i, j] = vals.mean()

            # Save per-subject ROI connectivity
            if save:
                out_dir = os.path.join(data_save_dir, cfg.result_con_path, sub)
                os.makedirs(out_dir, exist_ok=True)
                df_roi = pd.DataFrame(roi_data, index=roi_labels, columns=roi_labels)
                fname = os.path.join(out_dir,
                                     f"{sub}_{proto}_{cfg.con_method}_{band_name}_ROI.xlsx")
                df_roi.to_excel(fname)

            roi_matrices.append(roi_data)

        # After looping subjects: average across subjects
        avg_roi = np.mean(np.stack(roi_matrices, axis=2), axis=2)
        df_avg = pd.DataFrame(avg_roi, index=roi_labels, columns=roi_labels)
        if save:
            avg_fname = os.path.join(data_save_dir, cfg.result_con_path,
                                     f"{proto}_{cfg.con_method}_{band_name}_allSub_ROI.xlsx")
            # Ensure result directory exists
            os.makedirs(os.path.dirname(avg_fname), exist_ok=True)
            df_avg.to_excel(avg_fname)

    # End of function


#_______________________________________________________________________________________________________


import os
from getpass import getuser
from Baking_EEG import config as cfg
from Baking_EEG import _4_connectivity as connectivity

# === PARAMETERS ===
save    = True
verbose = True

user = getuser()
if user == 'tkz':
    sys.path.append('/home/tkz/…/Baking_EEG')
elif user == 'adminlocal':
    sys.path.append('C:\\Users\\adminlocal\\…\\Baking_EEG')


# Subjects list
sujets = [
    "TF53", "CA55", "ME64", "MP68", "JA61", "SV62",
    "TpAT19J1", "TpCF24J1", "TpEM13J1", "TpEP16J1",
    "TT45", "YG72",
]

# Input & output directories (absolute paths)
data_save_dir = (
    r"C:\Users\adminlocal\Desktop\ConnectDoc"
    r"\EEG_2025_CAP_FPerrin_Vera"
    r"\Analysis_Baking_EEG_Vera"
    r"\data_connectivity"
)
output_dir = (
    r"C:\Users\adminlocal\Desktop\Connect_new_ROI\excel\coma"
)
os.makedirs(output_dir, exist_ok=True)

# Override cfg.result_con_path so connectivity_overSubs writes to `output_dir`
# (it does os.path.join(data_save_dir, cfg.result_con_path, …) internally)
cfg.result_con_path = output_dir

# Channels to use
selected_chans = cfg.EGI_con_chan

# Protocols to run
protocols = ['LG', 'PP', 'Resting']

print("################## Connectivity for subjects", sujets, "##################\n")

for proto in protocols:
    print(f"=== Processing protocol: {proto} ===")
    connectivity.connectivity_overSubs(
        subs=sujets,
        data_save_dir=data_save_dir,      # where to read SUBJECT_PROTO_epo_conn.fif
        selected_chans=selected_chans,
        proto=proto,
        cfg=cfg,
        save=save
    )
    print(f"--> Done {proto}\n")

print("All ROI Excels written to:", output_dir)
