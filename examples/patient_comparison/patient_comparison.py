import os
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_t_test

def perform_permutation_ttest_patient_comparison(input_dir, output_dir, group1, group2, bandwidth, protocol):
    """
    Performs a point-wise permutation t-test comparing connectivity between
    two patient groups (e.g., group1 vs group2) for a given protocol and bandwidth.
    
    For each subject in each group, the connectivity arrays are loaded from 
    '3D_{protocol}_{bandwidth}_allSubConArray.npy' located in each group's directory.
    The difference (group1 - group2) is computed for each electrode pair.
    A permutation t-test is then performed to test if these differences are significantly
    different from zero. The resulting T-statistics and p-values are saved to CSV files.
    
    Parameters:
    -----------    
    input_dir : str
        The root input directory where the patient group subdirectories are located.
    output_dir : str
        The directory where the result files will be saved.
    group1 : str
        The first patient group for the comparison (e.g., "coma").
    group2 : str
        The second patient group for the comparison (e.g., "conscious").
    bandwidth : str
        The bandwidth identifier (e.g., "theta").
    protocol : str
        The protocol to use (e.g., "PP").
    """
    group1_path = os.path.join(input_dir, group1)
    group2_path = os.path.join(input_dir, group2)
    
    if not os.path.exists(group1_path):
        print(f"Directory {group1_path} not found.")
        return
    if not os.path.exists(group2_path):
        print(f"Directory {group2_path} not found.")
        return
    
    filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
    file1 = os.path.join(group1_path, filename)
    file2 = os.path.join(group2_path, filename)
    
    if not os.path.exists(file1):
        print(f"File not found: {file1}")
        return
    if not os.path.exists(file2):
        print(f"File not found: {file2}")
        return
    
    data_1 = np.load(file1)
    data_2 = np.load(file2)
    
    print(f"{group1} data shape: {data_1.shape}")
    print(f"{group2} data shape: {data_2.shape}")
    
    # Adjust for different subject numbers if necessary
    if data_1.shape[2] != data_2.shape[2]:
        min_subjects = min(data_1.shape[2], data_2.shape[2])
        data_1 = data_1[:, :, :min_subjects]
        data_2 = data_2[:, :, :min_subjects]
        print(f"Adjusted data shapes to match subjects: {data_1.shape}, {data_2.shape}")
    
    # Compute the difference (group1 - group2) for each subject
    diff_data = data_1 - data_2  # shape: (channels, channels, subjects)
    print(f"Comparing {group1} vs {group2}. Difference data shape: {diff_data.shape}")
    
    n_channels = diff_data.shape[0]
    n_subjects = diff_data.shape[2]
    
    # Rearrange the data: extract the lower triangle for each subject.
    lower_idx = np.tril_indices(n_channels, k=-1)
    X_full = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
    X = np.array([subject_matrix[lower_idx] for subject_matrix in X_full])
    
    # Run a point-wise permutation t-test (no adjacency support)
    T_obs, p_values, H0 = permutation_t_test(X, n_permutations=1000, tail=0)
    
    # Build full matrices (channels x channels) and fill with NaN initially.
    T_obs_full = np.full((n_channels, n_channels), np.nan)
    p_values_full = np.full((n_channels, n_channels), np.nan)
    
    # Place the computed t-values and p-values into the lower triangle of the full matrices.
    T_obs_full[lower_idx] = T_obs
    p_values_full[lower_idx] = p_values
    
    # Create electrode labels for clarity.
    electrodes = [f"Ch_{i+1}" for i in range(n_channels)]
    
    # Convert matrices to DataFrames.
    df_T = pd.DataFrame(T_obs_full, index=electrodes, columns=electrodes)
    df_p = pd.DataFrame(p_values_full, index=electrodes, columns=electrodes)
    
    # Save the results to CSV files.
    t_csv_filename = os.path.join(output_dir, f"PermT_T_{group1}_vs_{group2}_{bandwidth}_{protocol}.csv")
    p_csv_filename = os.path.join(output_dir, f"PermT_p_{group1}_vs_{group2}_{bandwidth}_{protocol}.csv")
    
    df_T.to_csv(t_csv_filename)
    df_p.to_csv(p_csv_filename)
    
    print("Permutation t-test results saved:")
    print(f"  T-values CSV: {t_csv_filename}")
    print(f"  P-values CSV: {p_csv_filename}")

# --- Example Usage ---
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory =r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
