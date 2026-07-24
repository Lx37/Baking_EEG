import sys
from getpass import getuser
user = getuser()  
print('user is : ', user)
if user == 'tkz':
    sys.path.append('/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP_JB/Baking_EEG')
if user == 'jeanbaptistechevallier':
    sys.path.append('/Users/jeanbaptistechevallier/Documents/TRAVAIL/EEG data/LYON24/Baking_EEG')
import os
import mne

from Baking_EEG import config as cfg
from Baking_EEG import utils
from Baking_EEG import _1_preprocess as prepro
from Baking_EEG import _2_cleaning as cleaning
from Baking_EEG import _3_epoch as epoch
from Baking_EEG import _4_evoked as erp

######################################
############ Your part ! #############
######################################
# Indicate the protocol and subject you're working on + data directory and excel file with patients info
protocol = 'PP' # 'PP' or 'LG' or 'Words' or 'Arythmetic' or 'Resting'
sujet = 'JA01' #'JAM'
# Set the parameters for the preprocessing : save data or not, verbose or not, plot or not (True or False)
save = True
verbose = True
plot = True

user = getuser()  # Username of the user running the scripts

if user == 'tkz':
    # where the data are stored
    raw_data_dir = '/home/tkz/Projets/data/data_JB-montpellier/'
    # excel file with all patients info
    xls_patients_info = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP_JB/JB-Montpellier_patients_df.csv'
    # path to save the analyzed data
    data_save_dir = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP_JB/Baking_EEG_data_JB/'

if user == 'jeanbaptistechevallier':
    # where the data are stored
    raw_data_dir = '/Users/jeanbaptistechevallier/Documents/TRAVAIL/EEG data/LYON24/sujets/'
    # excel file with all patients info
    xls_patients_info = '/Users/jeanbaptistechevallier/Documents/TRAVAIL/EEG data/LYON24/ConnectDoc_patient_df_26.csv'
    # path to save the analyzed data
    data_save_dir = '/Users/jeanbaptistechevallier/Documents/TRAVAIL/EEG data/LYON24/Baking_EEG_analysed_data/'


######################################
######################################

## Start of the script

print('MNE VERSION : ', mne.__version__)

# create the patient_info object (with names, config, protocol, file name, bad channels, etc.)
patient_info = utils.create_patient_info(sujet, xls_patients_info, protocol, raw_data_dir, data_save_dir)

data = []
epochs = []
epochs_TtP = []

# create the arborescence for required analysis
utils.create_arbo(protocol, patient_info, cfg)

#print("################## Preprocessing data " + sujet + " ##################")

if patient_info['data_fname'].endswith('.set'): # EEGlab raw data format (exported)
    data = prepro.preprocess(patient_info, cfg, save, verbose, plot)


print("################## Epoching data " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed

data = mne.io.read_raw_fif(data_name, preload=True)
data_epoch = epoch.get_ERP_epochs(data, patient_info, cfg, save=True, verbose=True, plot=True)


print("################## Create Evocked " + sujet + " ##################")

if protocol == 'PP':
    epochs_name = patient_info['data_save_dir'] + cfg.all_folders_PP['data_epochs_path']
    epochs_name = epochs_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_epochs_PPAP

epochsar = mne.read_epochs(epochs_name, proj=False, verbose=True, preload=True)
print(epochsar.info)
evoked_dict = erp.create_evoked(epochsar, patient_info, cfg, save=True, verbose=True, plot=True)
