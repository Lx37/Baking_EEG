import os
import numpy as np
from mne.stats import permutation_cluster_1samp_test

import os
import numpy as np
from mne.stats import permutation_cluster_1samp_test

def perform_comparison(input_dir, output_dir, condition, bandwidth, protocol1, protocol2):
    """
    Performs a cluster-based permutation test comparing connectivity between
    two protocols (e.g., protocol1 vs protocol2) for a given patient condition and bandwidth.
    
    For each subject, the difference (protocol1 - protocol2) is computed for each electrode pair.
    A one-sample cluster permutation test is then performed to test if the differences
    are significantly different from zero. The resulting T-statistics and cluster-level
    p-values are saved to text files.
    
    Parameters:
    -----------
    input_dir : str
        The root input directory where the condition subdirectories are located.
    output_dir : str
        The directory where the result files will be saved.
    condition : str
        The condition subdirectory name (e.g., "coma").
    bandwidth : str
        The bandwidth identifier (e.g., "alpha").
    protocol1 : str
        The first protocol for the comparison (e.g., "LG").
    protocol2 : str
        The second protocol for the comparison (e.g., "Resting").
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

    # Extract data and compute the difference (protocol1 - protocol2) for each subject.
    data_1 = data_files[protocol1]
    data_2 = data_files[protocol2]
    diff_data = data_1 - data_2  # shape: (channels, channels, subjects)
    
    print(f"Comparing {protocol1} vs {protocol2}. Difference data shape: {diff_data.shape}")
    
    # Get dimensions
    n_channels = diff_data.shape[0]
    n_subjects = diff_data.shape[2]
    
    # Rearrange data: permutation_cluster_1samp_test expects shape (subjects, n_tests)
    # Flatten each subject's connectivity difference matrix into a vector.
    X = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
    X = X.reshape(n_subjects, n_channels * n_channels)
    
    # Run a one-sample cluster permutation test.
    # tail=0 tests for any deviation from zero (two-sided).
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X, n_permutations=1000, tail=0, out_type='mask', verbose=True
    )
    
    # Reshape the observed T-statistics back to the connectivity matrix shape.
    T_obs_matrix = T_obs.reshape(n_channels, n_channels)
    
    # Create a matrix to store the cluster p-values.
    cluster_p_matrix = np.full((n_channels, n_channels), np.nan)
    for c, p_val in zip(clusters, cluster_p_values):
        # Check if c is a tuple (as sometimes clusters are returned as a tuple)
        if isinstance(c, tuple):
            c = c[0]
        # If c is a slice, convert it to a boolean mask
        if isinstance(c, slice):
            flat_mask = np.zeros(n_channels * n_channels, dtype=bool)
            flat_mask[c] = True
            mask = flat_mask.reshape(n_channels, n_channels)
        else:
            mask = c.reshape(n_channels, n_channels)
        # Assign the cluster's p-value to all positions in the mask
        cluster_p_matrix[mask] = p_val
    
    # Save the results to text files.
    t_filename = os.path.join(output_dir, f"Cluster_T_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.txt")
    p_filename = os.path.join(output_dir, f"Cluster_p_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.txt")
    np.savetxt(t_filename, T_obs_matrix, fmt="%.4e")
    np.savetxt(p_filename, cluster_p_matrix, fmt="%.4e")
    print("Cluster-based permutation test results saved:")
    print(f"  T values: {t_filename}")
    print(f"  Cluster p-values: {p_filename}")

# Define input and output directories
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"

# Example usage: Choose a condition, bandwidth, and protocols to compare
selected_condition = "del-"   # Change as needed
selected_bandwidth = "delta"   # Change as needed

# For comparing LG vs Resting:
#perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "LG", "Resting")

# For comparing PP vs Resting, you would call:
#perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "PP", "Resting")

# For comparing PP vs LG, you would call:
perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "PP", "LG")
