import os
import numpy as np
import mne
from mne.stats import permutation_t_test

def perform_roi_permutation_ttest(condition_dir, condition, bandwidth, protocol1, protocol2, output_dir):
    """
    Loads ROI connectivity arrays for two protocols from a given condition folder,
    performs a permutation t-test comparing the connectivity (using the lower triangle)
    between protocols, and saves the resulting t-statistic and p-value matrices.
    
    The ROI connectivity arrays are expected to have shape (n_ROIs, n_ROIs, n_subjects)
    and be named as:
      ROI_<condition>_3D_<protocol>_<bandwidth>_allSubConArray.npy
      
    The t-test is performed on the vectorized lower-triangle (excluding diagonal)
    for each subject. The resulting full (n_ROIs x n_ROIs) t and p matrices, with NaN 
    in the upper triangle, are saved as .npy files in the output directory.
    
    Output filenames:
      ROI_<condition>_<protocol1>_vs_<protocol2>_<bandwidth>_T.npy
      ROI_<condition>_<protocol1>_vs_<protocol2>_<bandwidth>_p.npy
    """
    # Build file paths for each protocol file
    file1 = os.path.join(condition_dir, f"ROI_{condition}_3D_{protocol1}_{bandwidth}_allSubConArray.npy")
    file2 = os.path.join(condition_dir, f"ROI_{condition}_3D_{protocol2}_{bandwidth}_allSubConArray.npy")
    
    if not os.path.exists(file1):
        print(f"File not found: {file1}")
        return
    if not os.path.exists(file2):
        print(f"File not found: {file2}")
        return
        
    data1 = np.load(file1)  # shape: (n_ROIs, n_ROIs, n_subjects)
    data2 = np.load(file2)
    
    # Adjust subject count if needed
    if data1.shape[2] != data2.shape[2]:
        n_subjects = min(data1.shape[2], data2.shape[2])
        data1 = data1[:, :, :n_subjects]
        data2 = data2[:, :, :n_subjects]
        print(f"Adjusted subject count to {n_subjects}")
    else:
        n_subjects = data1.shape[2]
    
    # Compute difference for each subject (protocol1 - protocol2)
    diff_data = data1 - data2  # shape: (n_ROIs, n_ROIs, n_subjects)
    print(f"Processing {condition} {bandwidth}: Comparing {protocol1} vs {protocol2} for {n_subjects} subjects.")
    
    n_ROIs = diff_data.shape[0]
    
    # Extract lower triangle indices (excluding the diagonal)
    lower_idx = np.tril_indices(n_ROIs, k=-1)
    
    # Create a data matrix X where each row corresponds to a subject's vectorized lower-triangle
    X = np.array([diff_data[:, :, s][lower_idx] for s in range(n_subjects)])
    
    # Run the point-wise permutation t-test
    T_obs, p_values, H0 = permutation_t_test(X, n_permutations=1000, tail=0)
    
    # Create full matrices for t-values and p-values (n_ROIs x n_ROIs), fill upper triangle with NaN
    T_matrix = np.full((n_ROIs, n_ROIs), np.nan)
    p_matrix = np.full((n_ROIs, n_ROIs), np.nan)
    T_matrix[lower_idx] = T_obs
    p_matrix[lower_idx] = p_values
    
    # Build output filenames
    out_t_file = os.path.join(output_dir, f"T_ROI_{condition}_{protocol1}_vs_{protocol2}_{bandwidth}.npy")
    out_p_file = os.path.join(output_dir, f"P_ROI_{condition}_{protocol1}_vs_{protocol2}_{bandwidth}.npy")
    
    np.save(out_t_file, T_matrix)
    np.save(out_p_file, p_matrix)
    
    print(f"Saved T-values to: {out_t_file}")
    print(f"Saved p-values to: {out_p_file}")

# Main settings

# Conditions: folder names within the base directory (e.g., coma, mcs, del+, del-, conscious)
conditions = ['coma', 'mcs', 'del+', 'del-', 'conscious','vs']

# Frequency bands (update as needed)
bandwidths = ['alpha', 'beta', 'theta', 'delta', 'gamma']

# Protocol comparisons: list of tuples (protocol1, protocol2)
comparisons = [('LG', 'PP'), ('PP', 'Resting'), ('LG', 'Resting')]

# Base directory where the ROI connectivity arrays are saved (per condition)
base_dir = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"

# Output directory for the ROI connectivity comparisons
output_dir = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\ROI_connectivity"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop over conditions, frequency bands, and protocol comparisons
for cond in conditions:
    condition_dir = os.path.join(base_dir, cond)
    if not os.path.exists(condition_dir):
        print(f"Condition folder not found: {condition_dir}")
        continue
    for bw in bandwidths:
        for prot1, prot2 in comparisons:
            print(f"\nRunning ROI permutation t-test for condition: {cond}, bandwidth: {bw}, comparison: {prot1} vs {prot2}")
            perform_roi_permutation_ttest(condition_dir, cond, bw, prot1, prot2, output_dir)
