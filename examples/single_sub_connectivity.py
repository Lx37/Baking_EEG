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
from getpass import getuser

from Baking_EEG import config as cfg
from Baking_EEG import utils
from Baking_EEG import _1_preprocess as preprocess 
from Baking_EEG import _2_cleaning as cleaning
from Baking_EEG import _3_epoch as epoch
from Baking_EEG import _4_connectivity as connectivity

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

data = []
epochs = []
epochs_TtP = []

# create the arborescence for required analysis
utils.create_arbo(protocol, patient_info, cfg)

'''
print("################## Preprocessing data " + sujet + " ##################")

if patient_info['data_fname'].endswith('.mff'): # EGI .mff raw data format
    data = preprocess.preprocess_mff(patient_info, cfg, save, verbose, plot)
#else:
#   mircromed #TODO
#   GTec #TODO
'''

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
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed # cfg.prefix_ICA

data = mne.io.read_raw_fif(data_name, preload=True)
data = epoch.get_epochs_connectivity(data, patient_info, cfg, save=True, verbose=True, plot=True)

'''

#'''
print("################## Connectivity " + sujet + " ##################")

data_name = patient_info['data_save_dir'] + cfg.data_con_path
data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_epo_conn # cfg.prefix_ICA

con_data = connectivity.connectivity_1sub(data_name, patient_info, cfg, save=True, verbose=True, plot=True)


print('con_data : ', con_data)
#print_infos(con_data)

'''
con_matrix = con_data.get_data(output="dense")[:, :, 0]
nb_chan = con_matrix.shape[0]
#print( 'XXXXXXXXXXXXXXXXXXXXXX nb_chan : ',  nb_chan)
#df_chan = pd.DataFrame(np.nan, index=nb_chan, columns=nb_chan)
#df_chan.loc[:, :] = con_matrix
df_con_sub_name =  f'{cfg.data_conn_path}/{sujet}/{sujet}_{protocol}_{method}_conData.csv'
#df_chan.to_excel(df_con_sub_name)
np.savetxt(df_con_sub_name, con_matrix, delimiter=",", fmt="%.2f")


Result_ROI, df_ROI = connectivity.get_ROI(con_matrix)
df_Roi_sub_name =  f'{cfg.data_conn_path}/{sujet}/{sujet}_{protocol}_{method}_ROI.xlsx'
df_ROI.to_excel(df_Roi_sub_name)
'''
