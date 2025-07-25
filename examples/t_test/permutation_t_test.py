
import os
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_t_test

def perform_permutation_ttest(input_dir, output_dir, condition, bandwidth, protocol1, protocol2):
    """
    Performs a point-wise permutation t-test comparing connectivity between
    two protocols (e.g., protocol1 vs protocol2) for a given patient condition and bandwidth.
    
    For each subject, the difference (protocol1 - protocol2) is computed for each electrode pair.
    A permutation t-test is then performed to test if these differences are significantly
    different from zero. The resulting T-statistics and p-values are saved to CSV files.
    
    Note: permutation_t_test does not support clustering or an adjacency parameter.
    
    Parameters:
    -----------    
    input_dir : str
        The root input directory where the condition subdirectories are located.
    output_dir : str
        The directory where the result files will be saved.
    condition : str
        The condition subdirectory name (e.g., "doc").
    bandwidth : str
        The bandwidth identifier (e.g., "alpha").
    protocol1 : str
        The first protocol for the comparison (e.g., "PP").
    protocol2 : str
        The second protocol for the comparison (e.g., "LG").
    """
    condition_path = os.path.join(input_dir, condition)
    if not os.path.exists(condition_path):
        print(f"Directory {condition_path} not found.")
        return

    # Load data files for all protocols
    protocols = ["PP", "LG", "Resting"]
    data_files = {}
    for protocol in protocols:
        filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
        file_path = os.path.join(condition_path, filename)
        if os.path.exists(file_path):
            data = np.load(file_path)
            print(f"{protocol} data shape: {data.shape}")
            data_files[protocol] = data
        else:
            print(f"File not found: {file_path}")
            return

    # Check that required protocols for the comparison are available
    if protocol1 not in data_files or protocol2 not in data_files:
        print(f"Required protocols ({protocol1} and/or {protocol2}) not found.")
        return

    data_1 = data_files[protocol1]
    data_2 = data_files[protocol2]
    
    # Adjust for different subject numbers if necessary
    if data_1.shape[2] != data_2.shape[2]:
        min_subjects = min(data_1.shape[2], data_2.shape[2])
        data_1 = data_1[:, :, :min_subjects]
        data_2 = data_2[:, :, :min_subjects]
        print(f"Adjusted data shapes to match subjects: {data_1.shape}, {data_2.shape}")
    
    # Compute the difference (protocol1 - protocol2) for each subject
    diff_data = data_1 - data_2  # shape: (channels, channels, subjects)
    print(f"Comparing {protocol1} vs {protocol2}. Difference data shape: {diff_data.shape}")
    
    n_channels = diff_data.shape[0]
    n_subjects = diff_data.shape[2]
    
    # Rearrange the data: flatten each subject's connectivity difference matrix into a vector.
    X = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
    X = X.reshape(n_subjects, n_channels * n_channels)

    # Create indices for the lower triangle (excluding the diagonal)
    lower_idx = np.tril_indices(n_channels, k=-1)

    # Rearrange data: For each subject, extract only the lower triangle of the connectivity matrix.
    # This gives a matrix of shape (subjects, number_of_unique_connections),
    # where number_of_unique_connections = n_channels*(n_channels-1)/2.
    X_full = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
    X = np.array([subject_matrix[lower_idx] for subject_matrix in X_full])

    
    # Run a point-wise permutation t-test (note: no adjacency support here)\n   
    T_obs, p_values, H0 = permutation_t_test(X, n_permutations=1000, tail=0)

    # Create indices for the lower triangle (excluding the diagonal)
    lower_idx = np.tril_indices(n_channels, k=-1)

    # Instead of reshaping t_obs directly, create a full matrix filled with NaN
    T_obs_full = np.full((n_channels, n_channels), np.nan)
    p_values_full = np.full((n_channels, n_channels), np.nan)

    # Assign the t_obs and p_values to the lower triangle of the full matrix
    T_obs_full[lower_idx] = T_obs
    p_values_full[lower_idx] = p_values

    # Now T_obs_full and p_values_full have shape (124,124)

    
    # Create electrode labels for clarity
    electrodes = [f"Ch_{i+1}" for i in range(n_channels)]
    
    # Convert matrices to DataFrames and save as CSV files.
    df_T = pd.DataFrame(T_obs_full, index=electrodes, columns=electrodes)
    df_p = pd.DataFrame(p_values_full, index=electrodes, columns=electrodes)
    
    t_csv_filename = os.path.join(output_dir, f"PermT_T_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")
    p_csv_filename = os.path.join(output_dir, f"PermT_p_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")
    
    df_T.to_csv(t_csv_filename)
    df_p.to_csv(p_csv_filename)
    
    print("Permutation t-test results saved:")
    print(f"  T-values CSV: {t_csv_filename}")
    print(f"  P-values CSV: {p_csv_filename}")

# Define input and output directories
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\t_test_permutation_values"

patient_categories = ['vs', 'coma', 'conscious', 'mcs', 'del-', 'del+']
# ['vs', 'coma', 'conscious', 'mcs', 'del-', 'del+']

# Define the bandwidths to analyze.
bandwidths = ['alpha', 'beta', 'theta', 'delta', 'gamma']
#['alpha', 'beta', 'theta', 'delta', 'gamma']

# Define the protocol comparisons as tuples: (protocol1, protocol2)
comparisons = [('PP', 'LG'), ('LG', 'Resting'), ('PP', 'Resting')]

# Loop through each combination of patient category, bandwidth, and protocol comparison.
for condition in patient_categories:
    for bandwidth in bandwidths:
        for protocol1, protocol2 in comparisons:
            print(f"\nRunning permutation t-test for condition: {condition}, bandwidth: {bandwidth}, comparison: {protocol1} vs {protocol2}")
            perform_permutation_ttest(input_directory, output_directory, condition, bandwidth, protocol1, protocol2)