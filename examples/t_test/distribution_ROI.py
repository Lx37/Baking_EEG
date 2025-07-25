import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def process_npy_distributions(input_dir, output_dir, conditions, bandwidths, comparisons):
    """
    For each condition, bandwidth, and protocol comparison, this function:
      - Loads the corresponding .npy files for p-values and t-test values.
      - Flattens the ROI matrices into vectors.
      - Plots a histogram (density=True) with a KDE overlay if possible.
      - Saves the plots in the output directory.
      - Logs any files skipped or any errors encountered.
    
    Expected file naming (inside each condition folder under input_dir):
      - P-values: P_ROI_<condition>_<protocol1>_vs_<protocol2>_<bandwidth>.npy
      - T-test:   T_ROI_<condition>_<protocol1>_vs_<protocol2>_<bandwidth>.npy
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    skipped_log = []
    var_threshold = 1e-12  # variance threshold below which we skip KDE
    
    # Loop over each condition folder.
    for cond in conditions:
        cond_folder = os.path.join(input_dir, cond)
        if not os.path.exists(cond_folder):
            print(f"Condition folder not found: {cond_folder}")
            skipped_log.append(f"{cond_folder}: Folder not found.")
            continue
        
        for bw in bandwidths:
            for protocol1, protocol2 in comparisons:
                # Build filenames for p-values and t-test values.
                p_filename = f"P_ROI_{cond}_{protocol1}_vs_{protocol2}_{bw}.npy"
                t_filename = f"T_ROI_{cond}_{protocol1}_vs_{protocol2}_{bw}.npy"
                p_path = os.path.join(cond_folder, p_filename)
                t_path = os.path.join(cond_folder, t_filename)
                
                # Process p-values file.
                if os.path.isfile(p_path):
                    try:
                        p_values_arr = np.load(p_path)
                        p_values_arr = p_values_arr.flatten()
                    except Exception as e:
                        skipped_log.append(f"{p_path}: Error loading file: {e}")
                        continue
                    
                    if p_values_arr.size == 0:
                        print(f"No valid p-values found in {p_path}. Skipping.")
                        skipped_log.append(f"{p_path}: No valid p-values found.")
                    else:
                        plt.figure(figsize=(10, 6))
                        plt.hist(p_values_arr, bins=50, density=True, color="green", alpha=0.6, edgecolor="black")
                        if p_values_arr.size > 1:
                            if np.var(p_values_arr) < var_threshold:
                                skipped_log.append(f"Skipped KDE for p-value file {p_path}: variance too low.")
                            else:
                                try:
                                    kde = gaussian_kde(p_values_arr)
                                    x = np.linspace(p_values_arr.min(), p_values_arr.max(), 100)
                                    plt.plot(x, kde(x), color="black", linewidth=2)
                                except Exception as e:
                                    skipped_log.append(f"Error computing KDE for p-value file {p_path}: {e}")
                        plt.title(f"P-value Distribution\nCondition: {cond}, Bandwidth: {bw}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                        plt.xlabel("P-values", fontsize=12)
                        plt.ylabel("Density", fontsize=12)
                        plt.grid(True)
                        output_filename = f"{cond}_{bw}_{protocol1}_vs_{protocol2}_pvalues.png"
                        plt.savefig(os.path.join(output_dir, output_filename))
                        plt.close()
                        print(f"Saved p-value histogram: {output_filename}")
                else:
                    print(f"P-value file not found: {p_path}")
                    skipped_log.append(f"{p_path}: File not found.")
                
                # Process t-test file.
                if os.path.isfile(t_path):
                    try:
                        t_values_arr = np.load(t_path)
                        t_values_arr = t_values_arr.flatten()
                    except Exception as e:
                        skipped_log.append(f"{t_path}: Error loading file: {e}")
                        continue
                    
                    if t_values_arr.size == 0:
                        print(f"No valid t-test values found in {t_path}. Skipping.")
                        skipped_log.append(f"{t_path}: No valid t-test values found.")
                    else:
                        plt.figure(figsize=(10, 6))
                        plt.hist(t_values_arr, bins=50, density=True, color="red", alpha=0.6, edgecolor="black")
                        if t_values_arr.size > 1:
                            if np.var(t_values_arr) < var_threshold:
                                skipped_log.append(f"Skipped KDE for t-test file {t_path}: variance too low.")
                            else:
                                try:
                                    kde = gaussian_kde(t_values_arr)
                                    x = np.linspace(t_values_arr.min(), t_values_arr.max(), 100)
                                    plt.plot(x, kde(x), color="black", linewidth=2)
                                except Exception as e:
                                    skipped_log.append(f"Error computing KDE for t-test file {t_path}: {e}")
                        plt.title(f"T-test Distribution\nCondition: {cond}, Bandwidth: {bw}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                        plt.xlabel("T-test values", fontsize=12)
                        plt.ylabel("Density", fontsize=12)
                        plt.grid(True)
                        output_filename = f"{cond}_{bw}_{protocol1}_vs_{protocol2}_ttest.png"
                        plt.savefig(os.path.join(output_dir, output_filename))
                        plt.close()
                        print(f"Saved t-test histogram: {output_filename}")
                else:
                    print(f"T-test file not found: {t_path}")
                    skipped_log.append(f"{t_path}: File not found.")
    
    # Write the skipped log to a file in the output directory.
    log_file_path = os.path.join(output_dir, "skipped_files_log.txt")
    with open(log_file_path, 'w') as log_file:
        for entry in skipped_log:
            log_file.write(entry + "\n")
    print(f"Skipped files log saved to: {log_file_path}")


# --- Example Usage ---
conditions = ['vs', 'coma', 'conscious', 'mcs', 'del-','del+']  # List of condition folder names
bandwidths = ['alpha', 'beta', 'theta', 'delta', 'gamma']
comparisons = [('PP', 'LG'), ('LG', 'Resting'), ('PP', 'Resting')]

# Input directory: Each condition folder is inside ROI_connectivity folder.
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\ROI_connectivity"
# Output directory for distribution plots.
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\ROI_connectivity\distribution"

process_npy_distributions(input_directory, output_directory, conditions, bandwidths, comparisons)
