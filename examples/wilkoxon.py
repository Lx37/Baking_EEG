import os
import numpy as np
from scipy.stats import wilcoxon

import os
import numpy as np
from scipy.stats import wilcoxon


def perform_wilcoxon_tests(input_dir, output_dir, condition, bandwidth):
    """
    Performs Wilcoxon tests between the three protocols
    for a given patient condition and bandwidth. This function computes a half matrix of 
    p-values (and test statistics) per electrode pair (only the upper triangular part, 
    excluding the diagonal).
    """
    protocols = ["PP", "LG", "Resting"]
    condition_path = os.path.join(input_dir, condition)

    if not os.path.exists(condition_path):
        print(f"Directory {condition_path} not found.")
        return

    # Load data files (keeping the 3D structure: channels x channels x patients)
    data_files = {}
    for protocol in protocols:
        filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
        file_path = os.path.join(condition_path, filename)
        if os.path.exists(file_path):
            data_files[protocol] = np.load(file_path)
        else:
            print(f"File not found: {file_path}")
            return

    data1 = data_files["PP"]
    data2 = data_files["LG"]
    i, j = 39, 9  # Example electrode pair; adjust indices as needed.
    values1 = data1[i, j, :]
    values2 = data2[i, j, :]
    diff = values1 - values2
    print(f"Electrode pair ({i}, {j}) differences: {diff}")
    print("Unique differences:", np.unique(diff))

    if len(data_files) != 3:
        print("Not all protocol files found. Check filenames.")
        return

    # Extract data for each protocol
    pp_data = data_files["PP"]
    lg_data = data_files["LG"]
    resting_data = data_files["Resting"]

    # Assuming the shape is (channels, channels, patients)
    n_channels = pp_data.shape[0]

    # Define protocol comparisons: each key will store a dictionary with two matrices:
    # one for the Wilcoxon test statistic and one for the p-values.
    comparisons = {
        "PP_vs_LG": (pp_data, lg_data),
        "PP_vs_Resting": (pp_data, resting_data),
        "LG_vs_Resting": (lg_data, resting_data)
    }

    results = {}
    for comp, (data1, data2) in comparisons.items():
        # Initialize matrices to hold the test statistic and p-value for each electrode pair.
        # We'll fill only the upper triangular part; the rest remains np.nan.
        stat_matrix = np.full((n_channels, n_channels), np.nan)
        p_matrix = np.full((n_channels, n_channels), np.nan)
        
        # Iterate only over the upper triangular part (excluding the diagonal)
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Extract connectivity values across patients for electrode pair (i, j)
                values1 = data1[i, j, :]
                values2 = data2[i, j, :]
                try:
                    stat, p = wilcoxon(values1, values2)
                except Exception as e:
                    stat, p = np.nan, np.nan
                stat_matrix[i, j] = stat
                p_matrix[i, j] = p

        results[comp] = {"stat_matrix": stat_matrix, "p_matrix": p_matrix}

    # Save the results to text files
    for comp, matrices in results.items():
        p_filename = os.path.join(output_dir, f"Wilcoxon_pvalues_{condition}_{bandwidth}_{comp}.txt")
        stat_filename = os.path.join(output_dir, f"Wilcoxon_stats_{condition}_{bandwidth}_{comp}.txt")
        np.savetxt(p_filename, matrices["p_matrix"], fmt="%.4e")
        np.savetxt(stat_filename, matrices["stat_matrix"], fmt="%.4e")
        print(f"Results for {comp} saved to:")
        print(f"  p-values: {p_filename}")
        print(f"  statistics: {stat_filename}")

# Define input and output directories
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"

# Example usage: Choose a condition and bandwidth
selected_condition = "coma"   # Change as needed
selected_bandwidth = "alpha"   # Change as needed

perform_wilcoxon_tests(input_directory, output_directory, selected_condition, selected_bandwidth)
