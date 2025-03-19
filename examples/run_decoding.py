from getpass import getuser
user = getuser()  # Username of the user running the scripts
print('User is:', user)
import sys
if user == 'tkz':
    sys.path.append('/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG')
if user == 'adminlocal':    
    sys.path.append('C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera\\Baking_EEG')
if user ==  'Tom':
    print('TODO')
import os
import mne
import numpy as np
import mne

# Set the parameters for the preprocessing : save data or not, verbose or not, plot or not (True or False)
save = True
verbose = True
plot = True

# Controls
data_path = '/home/tkz/Projets/data/data_Tom_PPDel/PP_CONTROLS_0.5/data_epochs/'
sub_fif = 'TJR7_PP_preproc_noICA_PP-epo_ar.fif'
fif_fname = data_path + sub_fif

all_conditions  = { 'PP' : ["PP"],
                    'AP' : ["AP"],
                    }

epochs = mne.read_epochs(fif_fname, proj=False, verbose=True, preload=True)#.pick_types(eeg=True) 
print('epochs info : ', epochs.info)
print('epochs event_id : ', epochs.event_id)
print('epochs events : ', epochs.events)

#epochs.pick(picks="eeg", exclude="bads")  

print(epochs["PP"])
print(epochs["AP"])
        
#Attention pas le même nompre d'epochs 'PP' / 'AP' -> à normaliser

XPP = epochs["PP"].pick(picks="eeg").get_data(copy=False) #n_epochs, n_eeg_channels, n_times
print('XPP.shape : ', XPP.shape)

XAP = epochs["AP"].pick(picks="eeg").get_data(copy=False) #n_epochs, n_eeg_channels, n_times
print('XAP.shape : ', XAP.shape)