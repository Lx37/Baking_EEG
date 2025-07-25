import os
import mne
import numpy as np
import matplotlib.pyplot as plt

def preprocess_fif(
    fif_path,
    output_dir,
    highpass=1.0,
    lowpass=25.0,
    target_sfreq=250,
    mark_bad_channels=True,
    plot=True,
    save=True
):
    if not os.path.exists(fif_path):
        raise FileNotFoundError(f"EEG file not found at {fif_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🔹 Loading raw data: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=True)
    print(f"Original sfreq: {raw.info['sfreq']}")

    # Plot sensors before anything (optional)
    if plot:
        raw.plot_sensors(kind="3d", title="Initial Sensor Layout")

    # Notch + Bandpass filtering
    print("\n🔹 Applying filters...")
    raw.notch_filter(np.arange(50, 126, 50), picks='eeg')  # Only 50 and 100 Hz
    raw.filter(highpass, lowpass, method='fir', phase='zero-double', fir_design='firwin2')

    if plot:
        raw.plot_psd(tmax=10, fmax=60)

    # Downsample
    if raw.info['sfreq'] != target_sfreq:
        print(f"\n🔹 Resampling to {target_sfreq} Hz...")
        raw.resample(target_sfreq)

    # Optional manual bad channel marking
    if mark_bad_channels:
        print("\n🔹 Mark bad channels manually:")
        raw.plot(title="Mark bads by clicking → then close window")

    # Interpolate bads for average reference
    raw.interpolate_bads(reset_bads=False)

    # Re-reference
    print("\n🔹 Applying average reference...")
    raw.set_eeg_reference('average')

    # Try to crop from first to last event if present
    print("\n🔹 Cropping around events if any...")
    try:
        events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
        if len(events) > 0:
            tmin = events[0, 0] / raw.info['sfreq'] - 3
            tmax = events[-1, 0] / raw.info['sfreq'] + 3
            raw.crop(tmin=max(tmin, 0), tmax=tmax)
            print(f"  → Cropped from {tmin:.2f}s to {tmax:.2f}s")
    except Exception as e:
        print(f"  → Could not crop: {e}")

    # Save
    if save:
        basename = os.path.basename(fif_path).replace('.fif', '_preproc.fif')
        save_path = os.path.join(output_dir, basename)
        print(f"\n💾 Saving preprocessed data to:\n{save_path}")
        raw.save(save_path, overwrite=True)

    print("\n✅ Done.")
    return raw


fif_path = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_preproc\c:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_preproc_clean\TWB1_LG_preproc.fif"
output_dir = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\data_preproc_clean"

preprocess_fif(fif_path, output_dir)