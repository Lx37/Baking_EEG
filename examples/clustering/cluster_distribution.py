import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Version 1 : distribution de p value and t test par comparaison protocole, par bande et par condition


def read_all_numeric_values(file_path):
    """
    Reads a CSV file and returns a NumPy array of all numeric values found in the file.
    Non-numeric cells (such as headers) are skipped.
    """
    numeric_values = []
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                try:
                    value = float(cell)
                    if np.isfinite(value):
                        numeric_values.append(value)
                except ValueError:
                    # Skip non-numeric cells.
                    continue
    return np.array(numeric_values, dtype=np.float64)

def process_csv_distributions(input_dir, output_dir, patient_categories, bandwidths, comparisons):
    """
    For each patient category, bandwidth, and protocol comparison:
      - Loads the corresponding CSV files for p-values and t-test values.
      - Extracts all numeric values from the files.
      - Plots a histogram (density=True) of the data.
      - Computes a KDE using gaussian_kde and overlays it on the histogram if the data
        has enough variability.
      - Saves the plots in the output directory.
      - Logs any files skipped for KDE reasons (low variance or KDE errors) in a log file.
    
    Expected file naming:
      - P-values: Cluster_p_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv
      - T-test:   Cluster_T_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv
    """
    # Create output directory if it doesn't exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List to keep track of files skipped for KDE reasons.
    skipped_log = []
    # Define a variance threshold below which we consider the data too degenerate.
    var_threshold = 1e-12

    for patient in patient_categories:
        for bandwidth in bandwidths:
            for protocol1, protocol2 in comparisons:
                # Build filenames.
                p_filename = f"Cluster_p_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv"
                t_filename = f"Cluster_T_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv"
                p_path = os.path.join(input_dir, p_filename)
                t_path = os.path.join(input_dir, t_filename)
                
                # Process p-value file.
                if os.path.isfile(p_path):
                    p_values_arr = read_all_numeric_values(p_path)
                    if p_values_arr.size == 0:
                        print(f"No valid p-values found in {p_path}. Skipping.")
                        skipped_log.append(f"{p_path}: No valid p-values found.")
                    else:
                        plt.figure(figsize=(10, 6))
                        plt.hist(p_values_arr, bins=50, density=True, color="green", alpha=0.6, edgecolor="black")
                        
                        # Only attempt KDE if there is more than one value.
                        if p_values_arr.size > 1:
                            if np.var(p_values_arr) < var_threshold:
                                skipped_log.append(f"Skipped KDE for p-value file {p_path}: variance too low.")
                            else:
                                try:
                                    kde = gaussian_kde(p_values_arr)
                                    x = np.linspace(float(p_values_arr.min()), float(p_values_arr.max()), 100)
                                    plt.plot(x, kde(x), color="black", linewidth=2)
                                except Exception as e:
                                    skipped_log.append(f"Error computing KDE for p-value file {p_path}: {e}")
                        
                        plt.title(f"P-value Distribution\nPatient: {patient}, Bandwidth: {bandwidth}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                        plt.xlabel("P-values", fontsize=12)
                        plt.ylabel("Density", fontsize=12)
                        plt.grid(True)
                        
                        output_filename = f"{patient}_{bandwidth}_{protocol1}_vs_{protocol2}_pvalues.png"
                        plt.savefig(os.path.join(output_dir, output_filename))
                        plt.close()
                        print(f"Saved p-value histogram: {output_filename}")
                else:
                    print(f"P-value file not found: {p_path}")
                    skipped_log.append(f"{p_path}: File not found.")
                
                # Process t-test file.
                if os.path.isfile(t_path):
                    t_values_arr = read_all_numeric_values(t_path)
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
                                    x = np.linspace(float(t_values_arr.min()), float(t_values_arr.max()), 100)
                                    plt.plot(x, kde(x), color="black", linewidth=2)
                                except Exception as e:
                                    skipped_log.append(f"Error computing KDE for t-test file {t_path}: {e}")
                        
                        plt.title(f"T-test Distribution\nPatient: {patient}, Bandwidth: {bandwidth}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                        plt.xlabel("T-test values", fontsize=12)
                        plt.ylabel("Density", fontsize=12)
                        plt.grid(True)
                        
                        output_filename = f"{patient}_{bandwidth}_{protocol1}_vs_{protocol2}_ttest.png"
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
patient_categories = ['vs', 'coma', 'conscious', 'mcs', 'del-']  # Your list of patient categories
bandwidths = ['alpha', 'beta', 'theta', 'delta', 'gamma']          # Your list of bandwidths
comparisons = [('PP', 'LG'), ('LG', 'Resting'), ('PP', 'Resting')]   # Protocol comparisons


# Define the input directory where the CSV files are located.
input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\cluster_permutation_values"

# Define the output directory to save the plots.
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\cluster_permutation_values\distribution"

# Process the CSV files and generate the distribution plots.
process_csv_distributions(input_directory, output_directory, patient_categories, bandwidths, comparisons)


#_______________________________________________________________________________________________________________________________

# Version 2 : comparer toutes les conditions, même bande de frequence et même comparaison

#'''

def read_all_numeric_values(file_path):
    """
    Reads a CSV file and returns a NumPy array of all numeric values found in the file.
    Non-numeric cells (such as headers) are skipped.
    """
    numeric_values = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    try:
                        value = float(cell)
                        if np.isfinite(value):
                            numeric_values.append(value)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return np.array(numeric_values, dtype=np.float64)

def process_csv_distributions_all_conditions(input_dir, output_dir, patient_categories, bandwidths, comparisons):
    """
    For each protocol comparison and each bandwidth, this function aggregates the numeric values 
    from all patient condition files and then:
      - Plots a histogram (density=True) of the aggregated p-values and t-test values.
      - Computes a KDE using gaussian_kde and overlays it on the histogram if possible.
      - Saves the plots in the output directory.
      - Logs any files that were skipped or had issues.
    
    Expected file naming conventions:
      - P-values: Cluster_p_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv
      - T-test:   Cluster_T_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    skipped_log = []
    var_threshold = 1e-12  # Variance threshold below which we skip KDE
    
    # Loop over each comparison and each bandwidth.
    for protocol1, protocol2 in comparisons:
        for bandwidth in bandwidths:
            aggregated_p = []  # list to collect p-values across all conditions.
            aggregated_t = []  # list to collect t-test values across all conditions.
            
            # Loop over all patient categories to aggregate values.
            for patient in patient_categories:
                p_filename = f"Cluster_p_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv"
                t_filename = f"Cluster_T_{patient}_{bandwidth}_{protocol1}_vs_{protocol2}.csv"
                p_path = os.path.join(input_dir, p_filename)
                t_path = os.path.join(input_dir, t_filename)
                
                if os.path.isfile(p_path):
                    p_values = read_all_numeric_values(p_path)
                    if p_values.size == 0:
                        skipped_log.append(f"{p_path}: No valid p-values found.")
                    else:
                        aggregated_p.extend(p_values.tolist())
                else:
                    skipped_log.append(f"{p_path}: File not found.")
                
                if os.path.isfile(t_path):
                    t_values = read_all_numeric_values(t_path)
                    if t_values.size == 0:
                        skipped_log.append(f"{t_path}: No valid t-test values found.")
                    else:
                        aggregated_t.extend(t_values.tolist())
                else:
                    skipped_log.append(f"{t_path}: File not found.")
            
            # Process aggregated p-values for this comparison and bandwidth.
            if aggregated_p:
                p_values_arr = np.array(aggregated_p, dtype=np.float64)
                plt.figure(figsize=(10, 6))
                plt.hist(p_values_arr, bins=50, density=True, color="green", alpha=0.6, edgecolor="black")
                if p_values_arr.size > 1:
                    if np.var(p_values_arr) < var_threshold:
                        skipped_log.append(f"Skipped KDE for aggregated p-values for {bandwidth} {protocol1}_vs_{protocol2}: variance too low.")
                    else:
                        try:
                            kde = gaussian_kde(p_values_arr)
                            x = np.linspace(float(p_values_arr.min()), float(p_values_arr.max()), 100)
                            plt.plot(x, kde(x), color="black", linewidth=2)
                        except Exception as e:
                            skipped_log.append(f"Error computing KDE for aggregated p-values for {bandwidth} {protocol1}_vs_{protocol2}: {e}")
                plt.title(f"Aggregated P-value Distribution\nBandwidth: {bandwidth}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                plt.xlabel("P-values", fontsize=12)
                plt.ylabel("Density", fontsize=12)
                plt.grid(True)
                output_filename = f"all_con_{bandwidth}_{protocol1}_vs_{protocol2}_pvalues.png"
                plt.savefig(os.path.join(output_dir, output_filename))
                plt.close()
                print(f"Saved p-value histogram: {output_filename}")
            else:
                print(f"No aggregated p-values for {bandwidth} {protocol1}_vs_{protocol2}.")
            
            # Process aggregated t-test values for this comparison and bandwidth.
            if aggregated_t:
                t_values_arr = np.array(aggregated_t, dtype=np.float64)
                plt.figure(figsize=(10, 6))
                plt.hist(t_values_arr, bins=50, density=True, color="red", alpha=0.6, edgecolor="black")
                if t_values_arr.size > 1:
                    if np.var(t_values_arr) < var_threshold:
                        skipped_log.append(f"Skipped KDE for aggregated t-test for {bandwidth} {protocol1}_vs_{protocol2}: variance too low.")
                    else:
                        try:
                            kde = gaussian_kde(t_values_arr)
                            x = np.linspace(float(t_values_arr.min()), float(t_values_arr.max()), 100)
                            plt.plot(x, kde(x), color="black", linewidth=2)
                        except Exception as e:
                            skipped_log.append(f"Error computing KDE for aggregated t-test for {bandwidth} {protocol1}_vs_{protocol2}: {e}")
                plt.title(f"Aggregated T-test Distribution\nBandwidth: {bandwidth}\nComparison: {protocol1} vs {protocol2}", fontsize=14)
                plt.xlabel("T-test values", fontsize=12)
                plt.ylabel("Density", fontsize=12)
                plt.grid(True)
                output_filename = f"all_con_{bandwidth}_{protocol1}_vs_{protocol2}_ttest.png"
                plt.savefig(os.path.join(output_dir, output_filename))
                plt.close()
                print(f"Saved t-test histogram: {output_filename}")
            else:
                print(f"No aggregated t-test values for {bandwidth} {protocol1}_vs_{protocol2}.")
    
    # Write the skipped log to a file in the output directory.
    log_file_path = os.path.join(output_dir, "all_con_skipped_files_log.txt")
    with open(log_file_path, 'w') as log_file:
        for entry in skipped_log:
            log_file.write(entry + "\n")
    print(f"Skipped files log saved to: {log_file_path}")

# --- Example Usage ---
patient_categories = ['vs', 'coma', 'conscious', 'mcs', 'del-']
bandwidths = ['alpha', 'beta', 'theta', 'delta', 'gamma']
comparisons = [('PP', 'LG'), ('LG', 'Resting'), ('PP', 'Resting')]

input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\ROI_connectivity"
output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\ROI_connectivity\distribution"

process_csv_distributions_all_conditions(input_directory, output_directory, patient_categories, bandwidths, comparisons)


#'''



