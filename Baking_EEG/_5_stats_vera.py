import os
import pandas as pd
from scipy.stats import shapiro

# Set the target directory
directory = r"C:\Users\adminlocal\Desktop\ConnectDoc\EEG_2025_CAP_FPerrin_Vera\Analysis_Baking_EEG_Vera\spectral_power_single\stats"  # ← Change this to your path

# Output file path
output_path = os.path.join(directory, "normality_test_results.txt")

# Open the results file
with open(output_path, "w") as results_file:
    results_file.write("Normality Test Results (Shapiro-Wilk)\n")
    results_file.write("="*50 + "\n\n")

    for filename in os.listdir(directory):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_excel(file_path)

                # Remove the first column (patient names)
                data = df.iloc[:, 1:]

                results_file.write(f"File: {filename}\n")
                for col in data.columns:
                    try:
                        # Drop NaN and run Shapiro-Wilk test
                        clean_data = data[col].dropna()
                        stat, p = shapiro(clean_data)

                        if p > 0.05:
                            result = "Normal"
                        else:
                            result = "Not Normal"

                        results_file.write(f"  Column '{col}': {result} (p = {p:.4f})\n")
                    except Exception as e:
                        results_file.write(f"  Column '{col}': Error - {e}\n")

                results_file.write("\n")
            except Exception as e:
                results_file.write(f"Could not process {filename}: {e}\n\n")

print("Done. Results saved to:", output_path)