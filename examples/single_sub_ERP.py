from getpass import getuser
user = getuser()  # Username of the user running the scripts
print('User is:', user)
import sys
if user == 'tkz':
    sys.path.append('/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG')
if user == 'adminlocal':    
    sys.path.append('C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera\\Baking_EEG')
import os
import mne
import numpy as np

from config import config as cfg
from utils import utils
from Baking_EEG import _1_preprocess as prepro
from Baking_EEG import _2_cleaning as cleaning
from Baking_EEG import _3_epoch as epoch

######################################
############ Your part ! #############
######################################
# Indicate the protocol and subject you're working on + data directory and excel file with patients info
protocol = 'LG' # 'PP' or 'LG' or 'Resting' (TODO: 'Words' or 'Arythmetic')
sujet = 'AD94'#'AD94' #LC97 #AG42
# Set the parameters for the preprocessing : save data or not, verbose or not, plot or not (True or False)
save = True
verbose = True
plot = True

if user == 'tkz':
    # where the data are stored
    raw_data_dir = '/home/tkz/Projets/data/data_EEG_battery_2019-/'
    # excel file with all patients info
    xls_patients_info = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/ConnectDoc_patients_df.csv'
    # path to save the analyzed data
    data_save_dir = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG_data/'
if user == 'adminlocal':
    # where the data are stored
    raw_data_dir = 'C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\Data\\data_EEG_battery_2019-\\'
    # excel file with all patients info
    xls_patients_info = 'C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera\\ConnectDoc_patients_df.csv'
    # path to save the analyzed data
    data_save_dir = 'C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera\\Analysis_Baking_EEG_Vera\\'


############################################################################

## Start of the script

print('MNE VERSION : ', mne.__version__)

# create the patient_info object (with names, config, protocol, file name, bad channels, etc.)
patient_info = utils.create_patient_info(sujet, xls_patients_info, protocol, raw_data_dir, data_save_dir)
print('patient_info : ', patient_info)

data = []
epochs = []
epochs_TtP = []

# create the arborescence for required analysis
utils.create_arbo(protocol, patient_info, cfg)

'''
print("################## Preprocessing data " + sujet + " ##################")

data = prepro.preprocess(patient_info, cfg, save, verbose, plot)

print("################## End of Preprocess ##################")

'''

'''
print("################## Cleaning data " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed

data = mne.io.read_raw_fif(data_name, preload=True)
data = cleaning.correct_blink_ICA(data, patient_info, cfg, save=save, verbose=verbose, plot=plot) # to test, work, adjust threshold,..
'''

#'''
print("################## Epoching data " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
data_name_preproc = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed #prefix_ICA  # cfg.prefix_processed
data_name_preproc_ICA = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_ICA

if os.path.exists(data_name_preproc_ICA):
    fif_name = data_name_preproc_ICA
elif os.path.exists(data_name_preproc):
    fif_name = data_name_preproc

data = mne.io.read_raw_fif(fif_name, preload=True)

events = mne.find_events(data, stim_channel='STI 014')
eventplot = mne.viz.plot_events(events, data.info['sfreq'])

data = epoch.get_ERP_epochs(data, patient_info, cfg, save=True, verbose=True, plot=True)

print(data.info)