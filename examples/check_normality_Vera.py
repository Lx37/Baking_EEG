import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.stats as stats
from itertools import combinations

#this code check the type of distribution of the data
'''
def load_dwpli_data(file_path):
    """Loads the dwPLI data from an Excel file."""
    df = pd.read_excel(file_path, index_col=0)  # Assuming first column is ROI names
    return df

def plot_distribution(data, title):
    """Plots histogram and Q-Q plot for normality check."""
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Histogram of {title}")
    plt.xlabel("dwPLI Values")
    plt.ylabel("Frequency")
    
    # Q-Q Plot
    plt.subplot(1, 2, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {title}")
    
    plt.tight_layout()
    plt.show()

def check_normality(data):
    """Performs normality tests on the given data."""
    print("\n--- Normality Tests ---\n")
    
    # Shapiro-Wilk Test (best for small samples, <5000)
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: W={shapiro_stat:.3f}, p={shapiro_p:.5f}")
    
    # Kolmogorov-Smirnov Test (good for large samples)
    ks_stat, ks_p = stats.kstest(data, 'norm')
    print(f"Kolmogorov-Smirnov Test: D={ks_stat:.3f}, p={ks_p:.5f}")
    
    # Anderson-Darling Test
    ad_result = stats.anderson(data, dist='norm')
    print(f"Anderson-Darling Test Statistic: {ad_result.statistic:.3f}")
    for i, crit in enumerate(ad_result.critical_values):
        print(f"Significance Level {ad_result.significance_level[i]}%: {crit:.3f}")
    
    if shapiro_p < 0.05 or ks_p < 0.05 or ad_result.statistic > ad_result.critical_values[2]:
        print("\nConclusion: Data is NOT normally distributed. Consider using non-parametric tests.")
    else:
        print("\nConclusion: Data appears to be normally distributed.")

# Define file path
base_path = "C:\\Users\\adminlocal\\Desktop\\Connectivity\\connectivity_VS"
protocol = "PP"  # Change to "PP" or "Resting" as needed
band = "delta"  # Change to "theta" or "delta" as needed

# Construct full file path
file_name = f"{protocol}_wpli2_debiased_{band}_allSub_ROI.xlsx"
file_path = os.path.join(base_path, file_name)

# Load the data
df = load_dwpli_data(file_path)

# Flatten matrix to get all dwPLI values
dwpli_values = df.values.flatten()
dwpli_values = dwpli_values[~np.isnan(dwpli_values)]  # Remove NaN values

# Plot distribution
plot_distribution(dwpli_values, f"{protocol} - {band}")

# Perform normality tests
check_normality(dwpli_values)

'''

# #écraser la dimension des epochs

# import os
# import numpy as np

# def process_connectivity_files(input_dir, output_dir):
#     """Processes EEG connectivity .npy files by averaging over the event dimension (i_event)
#     so that the resulting array has shape (channels, channels, subjects)."""
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # List all .npy files in the input directory
#     files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    
#     for file in files:
#         file_path = os.path.join(input_dir, file)
#         print(f"Processing file: {file_path}")
        
#         # Load the 4D connectivity array (channels * channels * event * subject)
#         data = np.load(file_path)
        
#         if data.ndim != 4:
#             print(f"Skipping {file} - Expected 4D array but got {data.ndim}D array")
#             continue
        
#         # Average over the event dimension (axis=2)
#         averaged_data = np.mean(data, axis=2, keepdims=False)
#         print("Shape after averaging:", averaged_data.shape)
        
#         # If a singleton dimension still remains (for example, if shape is (124, 124, 1, 21)),
#         # remove it using squeeze. This should yield a shape of (124, 124, 21).
#         averaged_data = np.squeeze(averaged_data)
#         print("Shape after squeeze:", averaged_data.shape)
        
#         # Generate new file name
#         new_filename = "3D_" + file.replace("_wpli2_debiased_", "_")
#         output_file_path = os.path.join(output_dir, new_filename)
        
#         # Save the processed array to the output directory
#         np.save(output_file_path, averaged_data)
#         print(f"Saved processed file to: {output_file_path}")

# # Define input and output directories
# input_directory = r"C:\Users\adminlocal\Desktop\Connectivity\connectivity_vs"
# output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\vs"

# # Run the processing function
# process_connectivity_files(input_directory, output_directory)



# # verify normality
# #  testing whether the overall distribution is normal across all conditions, 
# # but not checking normality for each protocol and patient separately

# def check_normality(data):
#     """Performs normality tests on the given data and returns True if normal, False otherwise."""
#     if len(data) < 3:
#         return False  # Not enough data to check normality
    
#     # Shapiro-Wilk Test (best for small samples, <5000)
#     shapiro_p = stats.shapiro(data)[1]
    
#     # Kolmogorov-Smirnov Test (good for large samples)
#     ks_p = stats.kstest(data, 'norm')[1]
    
#     # Anderson-Darling Test
#     ad_result = stats.anderson(data, dist='norm')
#     ad_normal = ad_result.statistic <= ad_result.critical_values[2]  # 5% significance level
    
#     # If any test suggests non-normality, return False
#     return shapiro_p >= 0.05 and ks_p >= 0.05 and ad_normal

# def process_connectivity_files(input_dir, output_file):
#     """Checks normality of connectivity distributions for each electrode pair across all protocols and patient conditions."""
#     patient_conditions = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
#     # Initialize electrode pairs list
#     electrode_pairs = None
#     normality_results = {}
    
#     for condition in patient_conditions:
#         condition_path = os.path.join(input_dir, condition)
#         files = [f for f in os.listdir(condition_path) if f.endswith(".npy")]
        
#         for file in files:
#             file_path = os.path.join(condition_path, file)
#             print(f"Processing file: {file_path}")
            
#             data = np.load(file_path)  # Load 3D array (channel * channel * subject)
#             num_channels = data.shape[0]
            
#             if electrode_pairs is None:
#                 electrode_pairs = list(combinations(range(num_channels), 2))
            
#             for (ch1, ch2) in electrode_pairs:
#                 pair_key = f"E{ch1+1}-E{ch2+1}"
#                 connectivity_values = data[ch1, ch2, :].flatten()  # Extract across subjects
                
#                 if pair_key not in normality_results:
#                     normality_results[pair_key] = []
                
#                 normality_results[pair_key].extend(connectivity_values.tolist())
    
#     # Perform normality check on each electrode pair
#     with open(output_file, "w") as f:
#         for pair, values in normality_results.items():
#             is_normal = check_normality(np.array(values))
#             f.write(f"{pair}: {'Yes' if is_normal else 'No'}\n")
#             print(f"{pair}: {'Yes' if is_normal else 'No'}")

# # Define input directory and output file path
# input_directory = "C:\\Users\\adminlocal\\Desktop\\Connectivity\\averaged_epochs"
# output_file = os.path.join(input_directory, "electrode_normality_results.txt")

# # Run the processing function
# process_connectivity_files(input_directory, output_file)


# # plotting data in histogram to check distribution visually

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_histogram_npy(directory, filename):
#     """Loads EEG connectivity data from an .npy file and plots its histogram."""
#     file_path = os.path.join(directory, filename)

#     # Load the .npy file
#     try:
#         data = np.load(file_path)  # Load the NumPy array
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         return
    
#     # Flatten the matrix to 1D array
#     values = data.flatten()

#     # Plot the histogram
#     plt.figure(figsize=(10, 6))
#     sns.histplot(values, bins=50, kde=True, color="blue")  # KDE adds density curve

#     # Formatting the plot
#     plt.title(f"Histogram of {filename}", fontsize=14)
#     plt.xlabel("Connectivity Values", fontsize=12)
#     plt.ylabel("Frequency", fontsize=12)
#     plt.grid(True)

#     # Show the plot
#     plt.show()

# directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\coma"
# filename = "3D_LG_alpha_allSubConArray.npy"
# plot_histogram_npy(directory, filename)


#VERSION 4 - plotting distribution per protocol and bandwidth, across all conditions

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict

# def load_and_aggregate_data(root_dir):
#     """
#     Loops over patient condition folders, loads .npy files with names formatted as
#     '3D_{protocol}_{bandwidth}_allSubConArray.npy', and aggregates connectivity values
#     by protocol and bandwidth.
#     """
#     # List of patient condition folders
#     patient_conditions = ["coma", "vs", "mcs", "del+", "del-", "conscious"]
    
#     # Nested dictionary: data_dict[protocol][bandwidth] will store a list of connectivity values.
#     data_dict = defaultdict(lambda: defaultdict(list))
    
#     for condition in patient_conditions:
#         condition_path = os.path.join(root_dir, condition)
#         if not os.path.isdir(condition_path):
#             print(f"Directory not found: {condition_path}")
#             continue
        
#         # Look for files starting with "3D_" and ending with ".npy"
#         files = [f for f in os.listdir(condition_path) if f.endswith(".npy") and f.startswith("3D_")]
#         for file in files:
#             file_path = os.path.join(condition_path, file)
#             # Expected filename format: 3D_{protocol}_{bandwidth}_allSubConArray.npy
#             parts = file.split('_')
#             if len(parts) < 4:
#                 print(f"Filename format unexpected: {file}")
#                 continue
#             protocol = parts[1]
#             bandwidth = parts[2]
            
#             try:
#                 data = np.load(file_path)
#                 # Flatten the 3D connectivity array to a 1D array of connectivity values
#                 values = data.flatten()
#                 data_dict[protocol][bandwidth].extend(values.tolist())
#                 print(f"Aggregated {len(values)} values from {file_path} for protocol '{protocol}', bandwidth '{bandwidth}'")
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")
    
#     return data_dict

# def plot_and_save_histograms(data_dict, output_dir):
#     """
#     For each protocol and bandwidth combination in data_dict, plot a histogram with a KDE,
#     and save the figure to the specified output directory.
#     """
#     # Create the output directory if it does not exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     for protocol, bw_dict in data_dict.items():
#         for bandwidth, values in bw_dict.items():
#             plt.figure(figsize=(10, 6))
#             sns.histplot(values, bins=50, kde=True, color="blue")
#             plt.title(f"Connectivity Distribution\nProtocol: {protocol}, Bandwidth: {bandwidth}", fontsize=14)
#             plt.xlabel("Connectivity Values", fontsize=12)
#             plt.ylabel("Frequency", fontsize=12)
#             plt.grid(True)
            
#             # Save the plot as an image file.
#             file_name = f"{protocol}_{bandwidth}_histogram.png"
#             file_path = os.path.join(output_dir, file_name)
#             plt.savefig(file_path)
#             print(f"Saved histogram for Protocol: {protocol}, Bandwidth: {bandwidth} at {file_path}")
#             plt.close()

# # Define the root directory where patient condition folders are located.
# root_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs"
# # Define the output directory for saving histogram images.
# output_directory = r"C:\Users\adminlocal\Desktop\Connectivity\averaged_epochs\distribution"

# # Load and aggregate connectivity data by protocol and bandwidth.
# data_dict = load_and_aggregate_data(root_directory)

# # Plot histogram distributions for each protocol and bandwidth, and save them.
# plot_and_save_histograms(data_dict, output_directory)




