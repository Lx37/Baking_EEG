


###################################################
###### File names organisation and prefixes ######

# folder to store the data for preprocessing
data_preproc_path = 'data_preproc/'
stimDict_path = 'data_stimdict/'
data_con_path = 'data_connectivity/'
result_con_path = 'connectivity/'
data_epochs_path = data_con_path #'data_epochs/' # Patch for Vera - should be updated #TODO
result_con_path_BRU = 'connectivity/'
data_psd_path = 'data_spectrum/'

##folders to store the data for PP analysis
all_folders_PP = {
    #'data_Subject_Dir' : 'data_EEG/',
    #'data_preproc_path' : 'data_preproc/',
    #'stimDict_path' : 'PP/data_stimdict/',
    'data_epochs_path' : 'PP/data_epochs/',
    'data_evoked_path' : 'PP/data_evoked/',
    'plot_topo_path' : 'PP/plots/Topomap/',
    'plot_erp_path' : 'PP/plots/ERP/',
    'plot_stats_path' : 'PP/plots/stats/',
    'data_stats_path' : 'PP/stats/',
    'text_stats_path' : 'PP/stats/',
    'cleaning_path' : 'PP/plots/cleaning/',
    'data_grandaverage_path' : 'PP/data_grandaverage/',
    'plot_grandaverage_path' : 'PP/plots/grandaverage/',
    'plot_GFP_path' : 'PP/plots/GFP/',
    'data_evoked_group_path' : 'PP/data_evoked_group/',
    'plot_evoked_group_path' : 'PP/evoked_group/'
}

all_folders_LG = {key: value.replace('PP', 'LG') for key, value in all_folders_PP.items()}
all_folders_Resting = {key: value.replace('PP', 'Resting') for key, value in all_folders_PP.items()}


##Prefixes - how to name the data after each step
prefix_stimDict = '_event_id.npy'
prefix_processed = '_preproc.fif'
prefix_epochs_PPAP = '_PP-epo.fif'
prefix_epochs_BRU = '-epo10s_new.fif'
prefix_ICA = '_preproc_ICA.fif'
prefix_noICA = '_preproc_noICA.fif'
prefix_epo_conn = '_epo_conn.fif'
#prefix_autoreject = '_ar.fif'
#prefix_ave = '-erp.fif'
#prefix_grandaverage = 'GrandAverage'
#prefix_evoked_group = '_evoked_group.fif'

###################################################
##################################################
######### Configuration for preprocessing ########

# EGI system : chan coordinates are loaded with name 'EEG 001', 'EEG 002',...
# so MNE 1.0.0 doesn't found them in the digmontage
# this list is use to set same names
EGI_chan_names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
                'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
                'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
                'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E48', 'E49', 'E50',
                'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60',
                'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70',
                'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
                'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90',
                'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100',
                'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110',
                'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120',
                'E121', 'E122', 'E123', 'E124', 'E125', 'E126', 'E127', 'E128', 'VREF'] #E129

EGI_chan_names_BRU = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
                'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20',
                'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30',
                'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40',
                'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47', 'E49', 'E50',
                'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60',
                'E61', 'E62', 'E63', 'E64', 'E65', 'E66', 'E67', 'E68', 'E69', 'E70',
                'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80',
                'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E87', 'E88', 'E89', 'E90',
                'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100',
                'E101', 'E102', 'E103', 'E104', 'E105', 'E106', 'E107', 'E108', 'E109', 'E110',
                'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E120',
                'E121', 'E122', 'E123', 'E124', 'E129', 'ECG']

EGI_misc_dict = {'E8':'misc','E25':'misc','E17':'misc','E126':'misc','E127':'misc'}
EGI_misc_dict_BRU = {'E125':'misc','E126':'misc','E127':'misc','E128':'misc', 'E119':'misc', 'E48':'misc', 'ECG':'misc'}

#VEOG = veog1 - veog
occular_EGI = {'veog1':'E25', 'veog2':'E127', 'veogr1':'E8', 'veogr2':'E126', 'heog1':'E32', 'heog2':'E1'}
#occular_Gtec = ?? {'veog1':'E25', 'veog2':'E127', 'veogr1':'E8', 'veogr2':'E126', 'heog1':'E32', 'heog2':'E1'}

GTec_mapping = {'EEG 001':'FP1', 'EEG 002':'FP2', 'EEG 003':'AF7', 'EEG 004':'AF8', 'EEG 005':'F3',
                'EEG 006':'Fz', 'EEG 007':'F4', 'EEG 008':'FT7', 'EEG 009':'FT8', 'EEG 010':'FC5', 
                'EEG 011':'C3', 'EEG 012':'C4', 'EEG 013':'FC6', 'EEG 014':'C5', 'EEG 015':'C1',
                'EEG 016':'Cz', 'EEG 017':'C2', 'EEG 018':'C6', 'EEG 019':'CP3', 'EEG 020':'CP4',
                'EEG 021':'CP1', 'EEG 022':'CP2', 'EEG 023':'P7', 'EEG 024':'P3', 'EEG 025':'Pz',
                'EEG 026':'P4', 'EEG 027':'P8', 'EEG 028':'PO7', 'EEG 029':'PO3', 'EEG 030':'PO4',
                'EEG 031':'PO8', 'EEG 032':'Oz'}

###MAPPING FOR BRU###

mapping_type_che = {'D255': 'misc', 'DIN4': 'misc', 'DI75': 'misc', 'DI77': 'misc', 'DI79': 'misc',
  'DI65': 'misc', 'DI67': 'misc', 'DI69': 'misc', 'DI66': 'misc', 'DI68': 'misc',
  'DI70': 'misc', 'STI 014': 'stim', 'ECG': 'ecg', 'EMG': 'emg'}

mapping_type_tai = {
  'DI65': 'misc', 'DI67': 'misc', 'DI69': 'misc', 'DI66': 'misc', 'DI68': 'misc',
  'DI70': 'misc', 'STI 014': 'stim', 'ECG': 'ecg', 'EMG': 'emg'}

mapping_type = {'DI75': 'misc', 'DI77': 'misc', 'DI79': 'misc',
  'DI65': 'misc', 'DI67': 'misc', 'DI69': 'misc', 'DI66': 'misc', 'DI68': 'misc',
  'DI70': 'misc', 'STI 014': 'stim', 'ECG': 'ecg', 'EMG': 'emg'}

highpass = 0.1 ## TODOLX was 0.5 for Fabrice
highcut = 45 # TODOLX was 25 for Fabrice

highpass_BRU = 1
highcut_BRU = 30

# DownSampling to
sfreq = 250

###################################################
######### Configuration for cleaning (ICA) ########

minBlinksICA = 100  #300 for protocol PP Lizette
eog_threshold = 3
# For ICA
n_components = 0.95 #15 was ok #0.99 from Fabrice  # (instead of 0.95) if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 2  # we need sufficient statistics, not all time points -> saves time
random_state = 23
ica_reject = None

######## For ICA - BRU #########
# For blink detection (could be a dict if different params by subjects)
eeg_for_eog = ['E25'] # No EOG for Bruno so we take some EEG
blink_detect_th =0.00005  #seuil pour détection des blinks

minBlinksICA = 150
eog_threshold = 3

n_components = 0.95  # if float, select n_components by explained variance of PCA
method = 'fastica'  # for comparison with EEGLAB try "extended-infomax" here
decim = 2  # we need sufficient statistics, not all time points -> saves time
random_state = 23
ica_reject = None
erp_reject = {'eeg': 100e-6, 'eog': 200e-6}

###################################################
##############  Configuration for ERP  ############

#erp_reject = {'eeg': 100e-6, 'eog': 200e-6}

#for epoching [Fabrice PP / Word]
erp_window_tmin = -0.2
erp_window_tmax = 1
erp_baseline = (None, 0) # from first instance to t=0
erp_detrend = None # Either 0 or 1, the order of the detrending. 0 is a constant (DC) detrend, 1 is a linear detrend.
erp_topo = True

# Specific PP
#events_id_PP = { 'PP/10': 110,'PP/20': 120,'PP/30': 130, 'AP/11': 111, 'AP/12': 112, 'AP/13': 113, 'AP/14': 114, 'AP/15': 115, 'AP/16': 116,
#'AP/21': 121, 'AP/22': 122, 'AP/23': 123, 'AP/24': 124, 'AP/25': 125, 'AP/26': 126,
#'AP/31': 131, 'AP/32': 132, 'AP/33': 133, 'AP/34': 134, 'AP/35': 135, 'AP/36': 136 } #'AP/22': 108, 'AP/12': 4 pour TpJC5 qui a 2 triggers en trop (22 et 23 devenus 4 et 108)

epochs_reject_PP = dict(eeg= 150e-6, eog=150e-6)
epochs_reject_LG = dict(eeg= 150e-6, eog=150e-6)

############# PP
#All second numbers correspond to a particular song
#For Conv:  1-3=Convoluted Right, 4-6=convoluted Left
MusicDio = {'M21': 21, 'M22': 22, 'M23': 23, 'M24': 24, 'M25': 25, 'M26': 26}
MusicConvG = {'M11': 11, 'M12': 12, 'M13': 13}
MusicConvD = { 'M14': 14, 'M15': 15, 'M16': 16}

TtP = { 'PP4': 140, 'PP5': 150, 'PP6': 160, 'PP7': 170, 'PP8': 180, 'PP9': 190,
        'AP41': 141, 'AP42': 142, 'AP43': 143, 'AP44': 144, 'AP45': 145, 'AP46': 146,
        'AP51': 151, 'AP52': 152, 'AP53': 153, 'AP54': 154, 'AP55': 155, 'AP56': 156,
        'AP61': 161, 'AP62': 162, 'AP63': 163, 'AP64': 164, 'AP65': 165, 'AP66': 166,
        'AP71': 171, 'AP72': 172, 'AP73': 173, 'AP74': 174, 'AP75': 175, 'AP76': 176,
        'AP81': 181, 'AP82': 182, 'AP83': 183, 'AP84': 184, 'AP85': 185, 'AP86': 186,
        'AP91': 191, 'AP92': 192, 'AP93': 193, 'AP94': 194, 'AP95': 195, 'AP96': 196}

# Redifinition of AP triggers that come just before the PP, to take only those in further analysis
Pp = { 'PP4': 140, 'PP5': 150, 'PP6': 160,
      'PP7': 170, 'PP8': 180, 'PP9': 190,
      'PP14': 1140, 'PP15': 1150, 'PP16': 1160,
      'PP17': 1170, 'PP18': 1180, 'PP19': 1190}

Ap = {'AP41': 141, 'AP42': 142, 'AP43': 143, 'AP44': 144, 'AP45': 145, 'AP46': 146,
	'AP51': 151, 'AP52': 152, 'AP53': 153, 'AP54': 154, 'AP55': 155, 'AP56': 156,'AP61': 161, 'AP62': 162, 'AP63': 163, 'AP64': 164, 'AP65': 165, 'AP66': 166,
	'AP71': 171, 'AP72': 172, 'AP73': 173, 'AP74': 174, 'AP75': 175, 'AP76': 176,'AP81': 181, 'AP82': 182, 'AP83': 183, 'AP84': 184, 'AP85': 185, 'AP86': 186,
	'AP91': 191, 'AP92': 192, 'AP93': 193, 'AP94': 194, 'AP95': 195, 'AP96': 196,
	'AP141': 1141, 'AP142': 1142, 'AP143': 1143, 'AP144': 1144, 'AP145': 1145, 'AP146': 1146,
	'AP151': 1151, 'AP152': 1152, 'AP153': 1153, 'AP154': 1154, 'AP155': 1155, 'AP156': 1156,'AP161': 1161, 'AP162': 1162, 'AP163': 1163, 'AP164': 1164, 'AP165': 1165, 'AP166': 1166,
	'AP171': 1171, 'AP172': 1172, 'AP173': 1173, 'AP174': 1174, 'AP175': 1175, 'AP176': 1176,'AP181': 1181, 'AP182': 1182, 'AP183': 1183, 'AP184': 1184, 'AP185': 1185, 'AP186': 1186,
	'AP191': 1191, 'AP192': 1192, 'AP193': 1193, 'AP194': 1194, 'AP195': 1195, 'AP196': 1196}

########### Definition of stimuli ###########
events_id_PP = {'PP/Music/Conv/G/V1': 1140, 'PP/Music/Conv/G/V2': 1150,  'PP/Music/Conv/G/V3': 1160, 'PP/Music/Conv/D/V1': 1170, 'PP/Music/Conv/D/V2': 1180, 'PP/Music/Conv/D/V3': 1190,
            'AP/1/Music/Conv/G/V1': 3141, 'AP/1/Music/Conv/G/V2': 3151, 'AP/1/Music/Conv/G/V3': 3161, 'AP/1/Music/Conv/D/V1': 3171, 'AP/1/Music/Conv/D/V2': 3181, 'AP/1/Music/Conv/D/V3': 3191,
			'AP/2/Music/Conv/G/V1': 3142, 'AP/2/Music/Conv/G/V2': 3152, 'AP/2/Music/Conv/G/V3': 3162, 'AP/2/Music/Conv/D/V1': 3172, 'AP/2/Music/Conv/D/V2': 3182, 'AP/2/Music/Conv/D/V3': 3192,
			'AP/3/Music/Conv/G/V1': 3143, 'AP/3/Music/Conv/G/V2': 3153, 'AP/3/Music/Conv/G/V3': 3163, 'AP/3/Music/Conv/D/V1': 3173, 'AP/3/Music/Conv/D/V2': 3183, 'AP/3/Music/Conv/D/V3': 3193,
			'AP/4/Music/Conv/G/V1': 3144, 'AP/4/Music/Conv/G/V2': 3154, 'AP/4/Music/Conv/G/V3': 3164, 'AP/4/Music/Conv/D/V1': 3174, 'AP/4/Music/Conv/D/V2': 3184, 'AP/4/Music/Conv/D/V3': 3194,
			'AP/5/Music/Conv/G/V1': 3145, 'AP/5/Music/Conv/G/V2': 3155, 'AP/5/Music/Conv/G/V3': 3165, 'AP/5/Music/Conv/D/V1': 3175, 'AP/5/Music/Conv/D/V2': 3185, 'AP/5/Music/Conv/D/V3': 3195,
			'AP/6/Music/Conv/G/V1': 3146, 'AP/6/Music/Conv/G/V2': 3156, 'AP/6/Music/Conv/G/V3': 3166, 'AP/6/Music/Conv/D/V1': 3176, 'AP/6/Music/Conv/D/V2': 3186, 'AP/6/Music/Conv/D/V3': 3196}

############### Averaging ###############
all_conditions_PP  = {  'PPmusic' : ["PP/Music"],'APmusic' : ["AP/Music"],}

###############  LG
events_id_LG = {'LS/GS': 11, 'LS/GD':12, 'LD/GS':21, 'LD/GD':22}


############### Averaging ###############
all_conditions_LG  = {  'LSGS' : ["LS/GS"],'LSGD' : ["LS/GD"],'LDGS' : ["LD/GS"],'LDGD' : ["LD/GD"], 'LS' : ["LS"],
                    'LD' : ["LD"], 'GS' : ["GS"], 'GD' : ["GD"]
                                }

####EPOCHING BRU####

goodttrigs = [65, 66, 67, 68, 69, 70]
nb_trig2add = 5             #59
size_win = 10 # en secondes  #1

#for epoching
# Initial rejection setting
epochs_reject=dict(eeg= 300e-6)
erp_window_tmin = 0
erp_window_tmax = size_win
erp_baseline = None # from first instance to t=0
erp_detrend = None # Either 0 or 1, the order of the detrending. 0 is a constant (DC) detrend, 1 is a linear detrend.
erp_topo = True

events_id_BRU = {'Music': 65, 'Noise':66, 'Rest after Music':67, 'Rest after noise':68,
                 'Interact music': 69, 'Interact noise':70}

###################################################
#########  Configuration for connectivity  ########

# Connectivity epoching
epochs_reject_con = dict(eeg=1000e-6)      # unit: V (EEG channels)
#epochs_reject_con = None

# Connectivity parameters for computation                  
con_freq_bands = {"delta": [0.5, 4.0],
                "theta": [4.0, 8.0],
                "alpha": [8.0, 13.0],
                "beta": [13.0, 30.0],
                "gamma": [30.0, 40.0],}
#fmin = 4
#fmax = 8
con_tmin = 0.0  # exclude the baseline period
con_method = 'wpli2_debiased'

con_vmin = 0
con_vmax = 1

con_event_ids_BRU = ['Music', 'Noise', 'Rest after Music', 'Rest after noise', 'Interact music', 'Interact noise']
con_event_ids = ['303']   

con_all_ROI_chan = {
    'ROI_Frontal' : ['E22', 'E15', 'E9', 'E18', 'E16', 'E10', 'E19', 'E11', 'E4', 'E12', 'E5'], #E17, E21, E14
    'ROI_Frontal_droit' : ['E2', 'E3', 'E123', 'E124', 'E122', 'E118', 'E117', 'E116'],  #121 E8, E1
    'ROI_Frontal_gauche' : ['E26', 'E23', 'E27', 'E24', 'E33', 'E34', 'E28', 'E20'], #E38
    'ROI_Central' :  ['E6', 'E13', 'E112', 'E30', 'E7', 'E106', 'E105', 'E31', 'E37', 'E80', 'E87', 'E79', 'E54', 'E55'], #Remis CZ (nommé 'VREF', position 128) E128
    'ROI_Temporal_droit' : ['E110', 'E111', 'E115', 'E109', 'E104', 'E103', 'E108', 'E93', 'E98', 'E102'], #E114
    'ROI_Temporal_gauche' :  ['E39', 'E35', 'E29', 'E40', 'E41', 'E36', 'E45', 'E46', 'E47', 'E42'], #E44
    'ROI_Parietal' : ['E61', 'E62', 'E78', 'E67', 'E72', 'E77', 'E71', 'E76', 'E70', 'E75', 'E83'], #E74, E82
    'ROI_Occipito_temporal_droit' : ['E50', 'E51', 'E52', 'E53', 'E58', 'E59', 'E60', 'E65', 'E66'], #E57, E64, E69
    'ROI_Occipito_temporal_gauche' : ['E86', 'E92', 'E97', 'E101', 'E85', 'E91', 'E96', 'E84', 'E90'] #E100, E95, E89
}

con_all_ROI_chan_BRU = {
    'lFP': ['E32', 'E25', 'E26', 'E22', 'E21', 'E17'],
    'rFP': ['E15', 'E14', 'E9', 'E8', 'E1', 'E2'],
    'lF': ['E38', 'E39', 'E33', 'E34', 'E35', 'E27', 'E28', 'E29', 'E23', 'E24', 'E20', 'E13', 'E18', 'E19', 'E12',
           'E16'],
    'rF': ['E11', 'E10', 'E4', 'E5', 'E112', 'E118', 'E3', 'E124', 'E123', 'E117', 'E111', 'E110', 'E116', 'E122',
           'E121', 'E115'],
    'Ce': ['E40', 'E41', 'E36', 'E30', 'E37', 'E31', 'E7', 'E6', 'E106', 'E55', 'E80', 'E105', 'E87', 'E104',
           'E103', 'E109'],
    'lT': ['E43', 'E44', 'E49', 'E45', 'E50', 'E56', 'E57', 'E58', 'E63', 'E64'],
    'rT': ['E120', 'E114', 'E113', 'E108', 'E101', 'E100', 'E107', 'E96', 'E99', 'E95'],
    'lP': ['E46', 'E51', 'E47', 'E42', 'E52', 'E59', 'E60', 'E53', 'E54', 'E61', 'E67', 'E71', 'E66', 'E62'],
    'rP': ['E72', 'E76', 'E77', 'E78', 'E79', 'E84', 'E85', 'E86', 'E93', 'E92', 'E91', 'E98', 'E97', 'E102'],
    'Oc': ['E68', 'E65', 'E69', 'E73', 'E70', 'E74', 'E75', 'E81', 'E82', 'E83', 'E90', 'E89', 'E88', 'E94']
}

EGI_con_chan =  [x for x in EGI_chan_names if x not in EGI_misc_dict.keys()] # All chan - those declared as misc
EGI_con_chan_BRU =  [x for x in EGI_chan_names_BRU if x not in EGI_misc_dict_BRU.keys()] # All chan - those declared as misc
