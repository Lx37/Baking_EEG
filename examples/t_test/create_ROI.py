import os
import numpy as np
import mne

# ROI definition dictionary
con_all_ROI_chan = {
    'ROI_Frontal' : ['E22', 'E15', 'E9', 'E18','E14', 'E16', 'E10','E17','E21' 'E19', 'E11', 'E4', 'E12', 'E5'],  # E17, E21, E14
    'ROI_Frontal_droit' : ['E1','E2', 'E3', 'E123','E8', 'E124', 'E122', 'E118', 'E121', 'E117', 'E116'],  # 121, E8, E1
    'ROI_Frontal_gauche' : ['E26', 'E23', 'E27', 'E24', 'E33', 'E34', 'E28', 'E20','E38'],  # E38
    'ROI_Central' :  ['E6', 'E13', 'E112', 'E30', 'E7', 'E106', 'E105', 'E31', 'E37', 'E80', 'E87', 'E79', 'E54', 'E55','E128'],  # Remis CZ ('VREF', position 128) E128
    'ROI_Temporal_droit' : ['E110', 'E111', 'E115', 'E109', 'E104', 'E103', 'E108', 'E93', 'E98', 'E102','E114'],  # E114
    'ROI_Temporal_gauche' :  ['E39', 'E35', 'E29', 'E40', 'E41', 'E36', 'E45', 'E46', 'E47', 'E42','E44'],  # E44
    'ROI_Parietal' : ['E61', 'E62', 'E78', 'E67', 'E72', 'E77', 'E71', 'E76', 'E70', 'E75', 'E83','E74','E82'],  # E74, E82
    'ROI_Occipito_temporal_droit' : ['E50', 'E51', 'E52', 'E53', 'E58', 'E59', 'E60', 'E65', 'E66','E57','E64','E69'],  # E57, E64, E69
    'ROI_Occipito_temporal_gauche' : ['E86', 'E92', 'E97', 'E101', 'E85', 'E91', 'E96', 'E84', 'E90','E100','E95','E89']  # E100, E95, E89
}

def convert_electrode_to_ROI(connectivity_array, electrode_to_roi):
    """
    Converts an electrode-level connectivity array (n_channels x n_channels) into an ROI-level
    connectivity matrix by averaging connectivity values between all electrodes in each ROI pair.
    
    Parameters:
    -----------
    connectivity_array : np.ndarray
        Electrode-level connectivity matrix (n_channels x n_channels).
    electrode_to_roi : dict
        Dictionary mapping each ROI name to a list of electrode indices (integers).
        
    Returns:
    --------
    roi_matrix : np.ndarray
        ROI-level connectivity matrix (n_ROIs x n_ROIs).
    """
    roi_names = list(electrode_to_roi.keys())
    n_ROIs = len(roi_names)
    roi_matrix = np.empty((n_ROIs, n_ROIs))
    roi_matrix.fill(np.nan)
    
    for i, roi_i in enumerate(roi_names):
        idx_i = electrode_to_roi[roi_i]
        for j, roi_j in enumerate(roi_names):
            idx_j = electrode_to_roi[roi_j]
            # For within-ROI connectivity, exclude self-connections if ROI has more than one electrode
            if roi_i == roi_j and len(idx_i) > 1:
                submatrix = connectivity_array[np.ix_(idx_i, idx_j)]
                # Exclude diagonal elements
                mask = ~np.eye(len(idx_i), dtype=bool)
                values = submatrix[mask]
            else:
                submatrix = connectivity_array[np.ix_(idx_i, idx_j)]
                values = submatrix.flatten()
            # Average the values if there are any; otherwise, set as nan.
            if values.size > 0:
                roi_matrix[i, j] = np.mean(values)
            else:
                roi_matrix[i, j] = np.nan
    return roi_matrix

# Main directories and conditions
base_dir = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
conditions = ['vs']
#['coma', 'mcs', 'del+', 'del-', 'conscious']

# Loop over each condition folder
for cond in conditions:
    cond_path = os.path.join(base_dir, cond)
    if not os.path.exists(cond_path):
        print(f"Condition folder not found: {cond_path}")
        continue

    # List files matching the pattern "3D_*_allSubConArray.npy"
    files = [f for f in os.listdir(cond_path) if f.startswith("3D_") and f.endswith("_allSubConArray.npy")]
    if not files:
        print(f"No electrode-level connectivity files found in {cond_path}")
        continue

    for file_name in files:
        file_path = os.path.join(cond_path, file_name)
        print(f"Processing file: {file_path}")
        
        # Load the electrode-level connectivity array
        # Expected shape: (n_channels, n_channels, n_subjects)
        electrode_array = np.load(file_path)
        n_channels, _, n_subjects = electrode_array.shape
        print(f"Array shape: {electrode_array.shape}")

        # Get a standard montage and extract the electrode names in the order used in the array.
        # We assume that the connectivity arrays use the first n_channels from the montage.
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        elec_names = montage.ch_names[:n_channels]
        # Create a mapping: electrode name (e.g., 'E22') -> index in connectivity array
        electrode_mapping = {name: idx for idx, name in enumerate(elec_names)}
        
        # Build a dictionary mapping each ROI to the list of electrode indices
        roi_to_indices = {}
        for roi, elec_list in con_all_ROI_chan.items():
            indices = []
            for elec in elec_list:
                if elec in electrode_mapping:
                    indices.append(electrode_mapping[elec])
                else:
                    # If electrode not found, warn but continue
                    print(f"Warning: Electrode {elec} not found in montage for condition {cond}.")
            if indices:
                roi_to_indices[roi] = indices
            else:
                print(f"Warning: No electrodes found for ROI {roi} in condition {cond}.")
        
        # Number of ROIs
        n_ROIs = len(roi_to_indices)
        if n_ROIs == 0:
            print("No valid ROIs found; skipping file.")
            continue

        # Prepare an array to hold the ROI-level connectivity for all subjects.
        roi_array = np.empty((n_ROIs, n_ROIs, n_subjects))
        roi_array.fill(np.nan)

        # Process each subject's electrode-level connectivity matrix
        for s in range(n_subjects):
            elec_conn = electrode_array[:, :, s]
            roi_conn = convert_electrode_to_ROI(elec_conn, roi_to_indices)
            roi_array[:, :, s] = roi_conn

        # Build output filename.
        # For example, if processing "3D_LG_alpha_allSubConArray.npy" in condition "coma",
        # then output file is "ROI_coma_3D_LG_alpha_allSubConArray.npy"
        out_filename = "ROI_" + cond + "_" + file_name
        out_path = os.path.join(cond_path, out_filename)
        np.save(out_path, roi_array)
        print(f"Saved ROI-level connectivity array to: {out_path}")
