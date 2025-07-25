import os
import pandas as pd

# REPLACE THIS with your actual root folder
root_dir = r"C:\Users\adminlocal\Desktop\Connect_new_ROI\conscious"

subject = "FG104"
protocol = "LG"
band = "alpha"
metric = "FD-FG"

# ROI key map
roi_keys = {
    "FD-FG": ("ROI_Frontal_Droit", "ROI_Frontal_Gauche"),
    "FD-CD": ("ROI_Frontal_Droit", "ROI_Caudal_Droit"),
    "CD-CG": ("ROI_Caudal_Droit", "ROI_Caudal_Gauche"),
    "FG-CG": ("ROI_Frontal_Gauche", "ROI_Caudal_Gauche")
}

roi1, roi2 = roi_keys[metric]
file = f"{subject}_{protocol}_wpli2_debiased_{band}_ROI.xlsx"
path = os.path.join(root_dir, subject.strip(), file)

print("Looking for file:", path)
if os.path.exists(path):
    df = pd.read_excel(path, index_col=0)
    if roi1 in df.index and roi2 in df.columns:
        print("Value:", df.loc[roi1, roi2])
    else:
        print("ROI pair not found.")
else:
    print("❌ File not found.")
