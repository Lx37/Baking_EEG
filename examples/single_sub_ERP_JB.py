import sys
sys.path.append('/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG')
import os
import mne
from getpass import getuser

from config import config as cfg
from utils import utils
from Baking_EEG import _1_preprocess as prepro
from Baking_EEG import _2_cleaning as cleaning
from Baking_EEG import _3_epoch as epoch

######################################
############ Your part ! #############
######################################
# Indicate the protocol and subject you're working on + data directory and excel file with patients info
protocol = 'Resting' # 'PP' or 'LG' or 'Words' or 'Arythmetic' or 'Resting'
sujet = 'SS25' #'VS91'
# Set the parameters for the preprocessing : save data or not, verbose or not, plot or not (True or False)
save = True
verbose = True
plot = True

user = getuser()  # Username of the user running the scripts

if user == 'tkz':
    # where the data are stored
    raw_data_dir = '/home/tkz/Projets/data/data_JB-montpellier/'
    # excel file with all patients info
    xls_patients_info = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/JB-Montpellier_patients_df.csv'
    # path to save the analyzed data
    data_save_dir = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG_data_JB/'


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

#'''
print("################## Preprocessing data " + sujet + " ##################")

if patient_info['data_fname'].endswith('.set'): # EEGlab raw data format (exported)
    data = prepro.preprocess(patient_info, cfg, save, verbose, plot)
#else:
#   mircromed #TODO
#   GTec #TODO
#'''

'''
print("################## Cleaning data " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed

data = mne.io.read_raw_fif(data_name, preload=True)
data = cleaning.correct_blink_ICA(data, patient_info, cfg, save=save, verbose=verbose, plot=plot) # to test, work, adjust threshold,..
'''

'''
print("################## Epoching data " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_ICA

data = mne.io.read_raw_fif(data_name, preload=True)
data = epoch.get_ERP_epochs(data, patient_info, cfg, save=True, verbose=True, plot=True)

'''