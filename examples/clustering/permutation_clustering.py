import os
import numpy as np
from mne.stats import permutation_cluster_1samp_test

import os
import numpy as np
from mne.stats import permutation_cluster_1samp_test

#CLUSTERING, T-TESTS V1

# def perform_comparison(input_dir, output_dir, condition, bandwidth, protocol1, protocol2):
#     """
#     Performs a cluster-based permutation test comparing connectivity between
#     two protocols (e.g., protocol1 vs protocol2) for a given patient condition and bandwidth.
    
#     For each subject, the difference (protocol1 - protocol2) is computed for each electrode pair.
#     A one-sample cluster permutation test is then performed to test if the differences
#     are significantly different from zero. The resulting T-statistics and cluster-level
#     p-values are saved to text files.
    
#     Parameters:
#     -----------
#     input_dir : str
#         The root input directory where the condition subdirectories are located.
#     output_dir : str
#         The directory where the result files will be saved.
#     condition : str
#         The condition subdirectory name (e.g., "coma").
#     bandwidth : str
#         The bandwidth identifier (e.g., "alpha").
#     protocol1 : str
#         The first protocol for the comparison (e.g., "LG").
#     protocol2 : str
#         The second protocol for the comparison (e.g., "Resting").
#     """
#     condition_path = os.path.join(input_dir, condition)
#     if not os.path.exists(condition_path):
#         print(f"Directory {condition_path} not found.")
#         return
    
#     # Load data files for all protocols
#     protocols = ["PP", "LG", "Resting"]
#     data_files = {}
#     for protocol in protocols:
#         filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
#         file_path = os.path.join(condition_path, filename)
#         if os.path.exists(file_path):
#             data = np.load(file_path)
#             print(f"{protocol} data shape: {data.shape}")
#             data_files[protocol] = data
#         else:
#             print(f"File not found: {file_path}")
#             return
    
#     # Check that required protocols for the comparison are available
#     if protocol1 not in data_files or protocol2 not in data_files:
#         print(f"Required protocols ({protocol1} and/or {protocol2}) not found.")
#         return

#     # Extract data and compute the difference (protocol1 - protocol2) for each subject.
#     data_1 = data_files[protocol1]
#     data_2 = data_files[protocol2]
#     diff_data = data_1 - data_2  # shape: (channels, channels, subjects)
    
#     print(f"Comparing {protocol1} vs {protocol2}. Difference data shape: {diff_data.shape}")
    
#     # Get dimensions
#     n_channels = diff_data.shape[0]
#     n_subjects = diff_data.shape[2]
    
#     # Rearrange data: permutation_cluster_1samp_test expects shape (subjects, n_tests)
#     # Flatten each subject's connectivity difference matrix into a vector.
#     X = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
#     X = X.reshape(n_subjects, n_channels * n_channels)
    
#     # Run a one-sample cluster permutation test.
#     # tail=0 tests for any deviation from zero (two-sided).
#     T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
#         X, n_permutations=1000, tail=0, out_type='mask', verbose=True
#     )
    
#     # Reshape the observed T-statistics back to the connectivity matrix shape.
#     T_obs_matrix = T_obs.reshape(n_channels, n_channels)
    
#     # Create a matrix to store the cluster p-values.
#     cluster_p_matrix = np.full((n_channels, n_channels), np.nan)
#     for c, p_val in zip(clusters, cluster_p_values):
#         # Check if c is a tuple (as sometimes clusters are returned as a tuple)
#         if isinstance(c, tuple):
#             c = c[0]
#         # If c is a slice, convert it to a boolean mask
#         if isinstance(c, slice):
#             flat_mask = np.zeros(n_channels * n_channels, dtype=bool)
#             flat_mask[c] = True
#             mask = flat_mask.reshape(n_channels, n_channels)
#         else:
#             mask = c.reshape(n_channels, n_channels)
#         # Assign the cluster's p-value to all positions in the mask
#         cluster_p_matrix[mask] = p_val
    
#     # Save the results to text files.
#     t_filename = os.path.join(output_dir, f"Cluster_T_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.txt")
#     p_filename = os.path.join(output_dir, f"Cluster_p_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.txt")
#     np.savetxt(t_filename, T_obs_matrix, fmt="%.4e")
#     np.savetxt(p_filename, cluster_p_matrix, fmt="%.4e")
#     print("Cluster-based permutation test results saved:")
#     print(f"  T values: {t_filename}")
#     print(f"  Cluster p-values: {p_filename}")

# # Define input and output directories
# input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
# output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"

# # Example usage: Choose a condition, bandwidth, and protocols to compare
# selected_condition = "vs"   # Change as needed
# selected_bandwidth = "alpha"   # Change as needed

# # For comparing LG vs Resting:
# # perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "LG", "Resting")

# #For comparing PP vs Resting, you would call:
# #perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "PP", "Resting")

# # For comparing PP vs LG, you would call:
# #perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "PP", "LG")


# cluster + PERMUTATION V2

# import os
# import numpy as np
# import pandas as pd
# import mne
# from mne.stats import permutation_cluster_1samp_test
# from mne.channels import find_ch_adjacency

# def perform_comparison(input_dir, output_dir, condition, bandwidth, protocol1, protocol2):
#     condition_path = os.path.join(input_dir, condition)
#     if not os.path.exists(condition_path):
#         print(f"Directory {condition_path} not found.")
#         return

#     protocols = ["PP", "LG", "Resting"]
#     data_files = {}
#     for protocol in protocols:
#         filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
#         file_path = os.path.join(condition_path, filename)
#         if os.path.exists(file_path):
#             data = np.load(file_path)
#             data_files[protocol] = data
#         else:
#             print(f"File not found: {file_path}")
#             return

#     if protocol1 not in data_files or protocol2 not in data_files:
#         print(f"Required protocols ({protocol1} and/or {protocol2}) not found.")
#         return

#     data_1 = data_files[protocol1]
#     data_2 = data_files[protocol2]
#     diff_data = data_1 - data_2

#     n_channels = diff_data.shape[0]
#     n_subjects = diff_data.shape[2]

#     X = np.transpose(diff_data, (2, 0, 1)).reshape(n_subjects, n_channels * n_channels)

#     # Load EGI HydroCel montage and create adjacency
#     montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
#     info = mne.create_info(ch_names=montage.ch_names[:n_channels], sfreq=100, ch_types='eeg')
#     info.set_montage(montage)

#     adjacency, _ = find_ch_adjacency(info, ch_type='eeg')

#     T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
#         X, n_permutations=1000, tail=0, adjacency=adjacency, out_type='mask', verbose=True
#     )

#     T_obs_matrix = T_obs.reshape(n_channels, n_channels)
#     cluster_p_matrix = np.full((n_channels, n_channels), np.nan)

#     for c, p_val in zip(clusters, cluster_p_values):
#         if isinstance(c, tuple):
#             c = c[0]
#         if isinstance(c, slice):
#             flat_mask = np.zeros(n_channels * n_channels, dtype=bool)
#             flat_mask[c] = True
#             mask = flat_mask.reshape(n_channels, n_channels)
#         else:
#             mask = c.reshape(n_channels, n_channels)
#         cluster_p_matrix[mask] = p_val

#     electrodes = [f"Ch_{i+1}" for i in range(n_channels)]

#     df_t_values = pd.DataFrame(T_obs_matrix, index=electrodes, columns=electrodes)
#     df_p_values = pd.DataFrame(cluster_p_matrix, index=electrodes, columns=electrodes)

#     t_csv_filename = os.path.join(output_dir, f"Cluster_T_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")
#     p_csv_filename = os.path.join(output_dir, f"Cluster_p_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")

#     df_t_values.to_csv(t_csv_filename)
#     df_p_values.to_csv(p_csv_filename)

#     print("Cluster-based permutation test results saved:")
#     print(f"  T-values CSV: {t_csv_filename}")
#     print(f"  Cluster p-values CSV: {p_csv_filename}")

# # Define input and output directories
# input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
# output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"

# # Example usage:
# selected_condition = "coma"
# selected_bandwidth = "alpha"

# # Perform the comparison (example PP vs LG)
# perform_comparison(input_directory, output_directory, selected_condition, selected_bandwidth, "PP", "LG")

#VERSION 3 AVEC PLOT


import os
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

def perform_comparison(input_dir, output_dir, condition, bandwidth, protocol1, protocol2):
    condition_path = os.path.join(input_dir, condition)
    if not os.path.exists(condition_path):
        print(f"Directory {condition_path} not found.")
        return

    protocols = ["PP", "LG", "Resting"]
    data_files = {}
    for protocol in protocols:
        filename = f"3D_{protocol}_{bandwidth}_allSubConArray.npy"
        file_path = os.path.join(condition_path, filename)
        if os.path.exists(file_path):
            data = np.load(file_path)
            data_files[protocol] = data
        else:
            print(f"File not found: {file_path}")
            return

    if protocol1 not in data_files or protocol2 not in data_files:
        print(f"Required protocols ({protocol1} and/or {protocol2}) not found.")
        return

    data_1 = data_files[protocol1]
    data_2 = data_files[protocol2]
    diff_data = data_1 - data_2  # shape: (channels, channels, subjects)
    print(f"Comparing {protocol1} vs {protocol2}. Difference data shape: {diff_data.shape}")
    
    n_channels = diff_data.shape[0]
    n_subjects = diff_data.shape[2]

    # Rearrange data: flatten each subject's connectivity difference matrix into a vector.
    X = np.transpose(diff_data, (2, 0, 1))  # shape: (subjects, channels, channels)
    X = X.reshape(n_subjects, n_channels * n_channels)

    # Load EGI HydroCel montage and create adjacency for channels
    montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
    info = mne.create_info(ch_names=montage.ch_names[:n_channels], sfreq=100, ch_types='eeg')
    info.set_montage(montage)
    from scipy.sparse import csr_matrix

    ch_adjacency, _ = find_ch_adjacency(info, ch_type='eeg')
    # Ensure the channel adjacency is in dense format
    if hasattr(ch_adjacency, "toarray"):
        ch_adj = ch_adjacency.toarray()
    else:
        ch_adj = ch_adjacency
    # Create the combined dense adjacency matrix for the connectivity space
    adjacency_dense = np.kron(ch_adj, ch_adj)
    # Convert to a SciPy sparse matrix (CSR format)
    adjacency = csr_matrix(adjacency_dense)


    # Run cluster-based permutation test
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
        X, n_permutations=1000, tail=0, adjacency=adjacency, out_type='mask', verbose=True
    )

    # Reshape observed T-statistics into full connectivity matrix
    T_obs_matrix = T_obs.reshape(n_channels, n_channels)
    # Create a full matrix for cluster p-values, initializing with NaN
    cluster_p_matrix = np.full((n_channels, n_channels), np.nan)
    for c, p_val in zip(clusters, cluster_p_values):
        if isinstance(c, tuple):
            c = c[0]
        if isinstance(c, slice):
            flat_mask = np.zeros(n_channels * n_channels, dtype=bool)
            flat_mask[c] = True
            mask = flat_mask.reshape(n_channels, n_channels)
        else:
            mask = c.reshape(n_channels, n_channels)
        cluster_p_matrix[mask] = p_val

    # Create electrode labels
    electrodes = [f"Ch_{i+1}" for i in range(n_channels)]
    df_T = pd.DataFrame(T_obs_matrix, index=electrodes, columns=electrodes)
    df_p = pd.DataFrame(cluster_p_matrix, index=electrodes, columns=electrodes)

    t_csv_filename = os.path.join(output_dir, f"Cluster_T_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")
    p_csv_filename = os.path.join(output_dir, f"Cluster_p_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.csv")

    df_T.to_csv(t_csv_filename)
    df_p.to_csv(p_csv_filename)

    print("Cluster-based permutation test results saved:")
    print(f"  T-values CSV: {t_csv_filename}")
    print(f"  Cluster p-values CSV: {p_csv_filename}")
    print(clusters)

    # --- Save clusters information to clusters.txt ---
    clusters_txt_path = os.path.join(output_dir, "clusters.txt")
    with open(clusters_txt_path, 'a') as f:
        header = f"Comparison {protocol1} vs {protocol2}, {condition}, {bandwidth}:\n"
        f.write(header)
        # For each cluster, extract the unique electrodes involved.
        for idx, c in enumerate(clusters):
            # Ensure c is a boolean mask of shape (n_channels*n_channels,)
            if isinstance(c, slice):
                flat_mask = np.zeros(n_channels * n_channels, dtype=bool)
                flat_mask[c] = True
            else:
                flat_mask = c
            cluster_indices = np.where(flat_mask)[0]
            # Map flat indices to 2D indices (rows, columns)
            rows, cols = np.unravel_index(cluster_indices, (n_channels, n_channels))
            unique_electrodes = np.unique(np.concatenate([rows, cols]))
            electrode_names = [f"Ch_{i+1}" for i in unique_electrodes]
            f.write(f"  Cluster {idx+1}: electrodes {electrode_names}\n")
        f.write("\n")
    print(f"Clusters information appended to: {clusters_txt_path}")


    # OPTIONAL PLOTTING SECTION:
    # Uncomment the block below to generate and save plots.
    #"""
    # # 2D Heatmap of Mean Connectivity Difference (averaged over subjects)
    # conn_mean = np.mean(diff_data, axis=2)
    # fig2d, ax2d = plt.subplots(figsize=(8, 6))
    # im_conn = ax2d.imshow(conn_mean, cmap='coolwarm', origin='lower',
    #                   vmin=-0.5, vmax=0.5)  # Fixed color scale from -1 to 1
    # ax2d.set_title('Mean Connectivity Difference')
    # ax2d.set_xlabel('Channel')
    # ax2d.set_ylabel('Channel')
    # ax2d.set_xticks(np.arange(n_channels))
    # ax2d.set_yticks(np.arange(n_channels))
    # ax2d.set_xticklabels(electrodes, rotation=90)
    # ax2d.set_yticklabels(electrodes)
    # fig2d.colorbar(im_conn, ax=ax2d, label='Connectivity Diff')
    # fig2d.set_size_inches(20, 20)
    
    # # Save 2D connectivity heatmap
    # plot2d_conn_filename = os.path.join(output_dir, f"2D_connectivity_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.png")
    # fig2d.savefig(plot2d_conn_filename, dpi=300, bbox_inches='tight')
    # plt.close(fig2d)


    # 2D Heatmap of Cluster p-values
    fig2d_p, ax2d_p = plt.subplots(figsize=(8, 6))
    im_p = ax2d_p.imshow(cluster_p_matrix, cmap='coolwarm', origin='lower')
    ax2d_p.set_title('Cluster p-values Heatmap')
    ax2d_p.set_xlabel('Channel')
    ax2d_p.set_ylabel('Channel')
    ax2d_p.set_xticks(np.arange(n_channels))
    ax2d_p.set_yticks(np.arange(n_channels))
    ax2d_p.set_xticklabels(electrodes, rotation=90)
    ax2d_p.set_yticklabels(electrodes)
    fig2d_p.colorbar(im_p, ax=ax2d_p, label='p-value')

    # Save 2D p-values heatmap
    plot2d_p_filename = os.path.join(output_dir, f"2D_pvalues_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.png")
    fig2d_p.savefig(plot2d_p_filename, dpi=300, bbox_inches='tight')
    plt.close(fig2d_p)
    
    # 3D Surface Plot of T-test Values
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    X_grid, Y_grid = np.meshgrid(np.arange(n_channels), np.arange(n_channels))
    surf = ax3d.plot_surface(X_grid, Y_grid, T_obs_matrix, cmap='viridis')
    ax3d.set_title('T-test Values (3D Surface)')
    ax3d.set_xlabel('Channel')
    ax3d.set_ylabel('Channel')
    ax3d.set_zlabel('T-value')
    fig3d.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
    
    # Save 3D T-test surface plot
    plot3d_filename = os.path.join(output_dir, f"3D_Ttest_{condition}_{bandwidth}_{protocol1}_vs_{protocol2}.png")
    fig3d.savefig(plot3d_filename, dpi=300, bbox_inches='tight')
    plt.close(fig3d)
    
    print(f"2D connectivity heatmap saved: {plot2d_p_filename}")
    print(f"3D T-test plot saved: {plot3d_filename}")
    #"""

# Define input and output directories
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\cluster_permutation_values"

import os

# Base directories
base_input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\cluster_permutation_values"

# Option 1: Automatically get patient folders (subdirectories) from the base input directory.
# patient_categories = [folder for folder in os.listdir(base_input_directory) 
#                       if os.path.isdir(os.path.join(base_input_directory, folder))]

# Option 2: Use a predefined list (uncomment if you prefer this approach)
patient_categories = ['vs', 'coma', 'conscious', 'mcs', 'del-', 'del+']
#['vs', 'coma', 'conscious', 'mcs', 'del-']

# Define the bandwidths to analyze.
bandwidths = ['alpha', 'beta', 'theta', 'delta','gamma']  # Adjust this list to include the bandwidths you need
#['alpha', 'beta', 'theta', 'delta','gamma']

# Define the protocol comparisons: (protocol1, protocol2)
comparisons = [('PP', 'LG'), ('LG', 'Resting'), ('PP', 'Resting')]

# Loop over each patient category (each subfolder in base_input_directory)
for condition in patient_categories:
    # perform_comparison will look for files in:
    # os.path.join(base_input_directory, condition)
    for bw in bandwidths:
        for prot1, prot2 in comparisons:
            print(f"Processing: Condition='{condition}', Bandwidth='{bw}', Comparison='{prot1}' vs '{prot2}'")
            # Since perform_comparison expects the base input directory and the condition name,
            # we pass base_input_directory as the first argument.
            perform_comparison(base_input_directory, output_directory, condition, bw, prot1, prot2)
