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

from config import config as cfg
from utils import utils
from Baking_EEG import _4_connectivity as connectivity

######################################
############ Your part ! #############
######################################

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

# It is supposed here that you've computed epoching on your data, ready for connectivity

save = True
verbose = True
plot = True

#sujets = ['CS38', 'JC39', 'TT45', 'MV48', 'TF53', 'CA55', 'JA61', 'SV62', 'ME63', 'ME64',
#        'SB67', 'MP68', 'YG72', 'MB73', 'KS76', 'BS81', 'TpAT19J1', 'TpCF24J1', 'TpDC22J1', 'TpEM13J1', 'TpEP16J1',
#        'TpEP20J1', 'TpFM25J1', 'TpLC21J1', 'TpMB18J1', 'TpMG17J1', 'TpPC23J1']
sujets = ['JC39']#'AD94' #LC97 #AG42
protocol = 'LG' # 'PP' or 'LG' or 'Resting' (TODO: 'Words' or 'Arythmetic')
selected_chans = cfg.EGI_con_chan  #['E32', 'E25', 'E26']


print("################## Connectivity " + str(sujets) + " ##################")

all_conn_aray = connectivity.connectivity_overSubs(sujets, data_save_dir, selected_chans, protocol, cfg, save=save, plot=plot, show_plot=plot)

print('all_conn_aray : ', all_conn_aray)

