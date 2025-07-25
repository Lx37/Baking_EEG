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
from Baking_EEG import _4_connectivity as connectivity

######################################
############ Your part ! #############
######################################

####MODIF BRU : ajout de ces quelques lignes ci-dessous pour appeler la fonction
subs = ['CHE', 'DES', 'EDC', 'GIC', 'GRL', 'HAA', 'MAL', 'MIL', 'MOP', 'PAP', 'POA', 'PRP', 'REL', 'ROL', 'SAE', 'TAI']
data_save_dir = "C:/Users\Bruno\Documents\PycharmProjects\scripts_exp\Analyse_EEG\Analysis/"
selected_chans = 'All'
proto = 'preproc_100Hz_ICA'
all_conn_data = connectivity.connectivity_overSubs_BRU(subs, data_save_dir, selected_chans, proto, cfg, save=True, plot=False, show_plot=False)
print("Données de connectivité collectées :", all_conn_data.shape)