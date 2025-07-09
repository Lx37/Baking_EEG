import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import mne


## logging info ###
import logging
from datetime import datetime


logname = './logs/' + datetime.now().strftime('log_%Y-%m-%d.log')
logging.basicConfig(filename=logname, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# Correspond to protocol "Battery" in old scripts
def redefine_event_PP(new_events, cfg, verbose=True, plot=True):
    # Redefinition des triggers pour nom apres Music (+1000) et PP apres Noise (non change)
    events_redef = new_events.copy()
    nb_name = 42
    add_trig_PMusic = 1000
    for idx, val in enumerate(events_redef):
        if (val[2] in cfg.MusicDio.values() or val[2] in cfg.MusicConvG.values() or val[2] in cfg.MusicConvD.values()):
            for idx2, val2 in enumerate(events_redef[idx:idx+nb_name+1]):
                if val2[2] in cfg.TtP.values():
                    events_redef[idx+idx2][2] += add_trig_PMusic

    # Redifinition of AP triggers that come just before the PP, to take only those in further analysis
    events_redef_equal = events_redef.copy()
    add_trig_AP = 2000
    for idx, val in enumerate(events_redef_equal):
        if (val[2] in cfg.Pp.values()):
            if (events_redef_equal[idx-1][2] in cfg.Ap.values()):
                events_redef_equal[idx-1][2] += add_trig_AP
            elif not (events_redef_equal[idx-1][2] in cfg.Ap.values()):
                if (events_redef_equal[idx+1][2] in cfg.Ap.values()):
                    events_redef_equal[idx+1][2] += add_trig_AP
                else:
                    print('Error no AP before or after PP ??')
    if verbose:
        print('Events AFTER changes : ', events_redef_equal)

    return events_redef_equal


def redefine_event_LG(data, patient_info, cfg, verbose=True, plot=True):
    ########### Renaming of triggers  ###########
    # Loading csv
    # csvname = cfg.csvpath + patient_info['ID_patient'] + '.csv'
    csvname = patient_info['raw_data_dir'] + patient_info['ID_patient'] + \
        '/Stimulations/local-global/playframe.csv'
    try:
        ord = pd.read_csv(csvname, sep=",")
    except:
        print("Can't find playframe excel file Local Global protocol for patient : ",
              patient_info['ID_patient'])
        print('CSV file : ', csvname)
        exit()

    number = 0
    ord["block"] = 0
    ord["LS"] = 0

    for count, x in enumerate(ord["Unnamed: 0"]):
        # ord["block"][count] = str(number)
        if x == 1:
            print(x)
            print(count)
            print(number)
            number = number + 1
        ord["block"][count] = str(number)

    GS = 0

    for i in range(4, len(ord), 5):
        # print(ord["ISI"][i])
        # print(ord["Trigger"][i])
        if ord["Trigger"][i-1] == ord["Trigger"][i]:
            ord.at[i, "LS"] = 10
        else:
            ord.at[i, "LS"] = 20
        BLnr = ord["block"][i]
        print(BLnr)
        # print(ord["Unnamed: 0"]==4)
        # print(ord["block"]==BLnr)
        GS = ord[(ord["Unnamed: 0"] == 5) & (
            ord["block"] == BLnr)]["Trigger"].item()
        print(GS)
        # ord["Trigger"][i]=ord["Trigger"][i]+"_GS"
        if GS == ord["Trigger"][i]:
            ord["LS"][i] = ord["LS"][i]+1
        else:
            ord["LS"][i] = ord["LS"][i]+2

    string = ord["LS"]

    # Finding events
    events = mne.find_events(data, stim_channel='STI 014')

    new_events = events.copy()
    for (x, y), value in np.ndenumerate(events):
        if y == 2:
            new_value = string[x]
            print(new_value)
            if new_value != '':
                new_events[x, y] = new_value

    if plot:
        eventplot = mne.viz.plot_events(new_events, data.info['sfreq'])

    return new_events


def get_ERP_epochs(data, patient_info, cfg, save=True, verbose=True, plot=True):

    # if patient_info['protocol'] == 'WORDS':
    #     import config_WORDS as cfg

    # if protocol == 'PP':
    #     import config_PP as cfg

    SubName = patient_info['ID_patient']

    ########### Renaming of EGI-loaded triggers  ###########

    if patient_info['EEG_system'] == 'EGI' and patient_info['protocol'] == 'PP':

        StimNpy = patient_info['data_save_dir'] + cfg.stimDict_path + \
            SubName + '_' + patient_info['protocol'] + cfg.prefix_stimDict
        print(StimNpy)
        translate_dict = np.load(StimNpy, allow_pickle=True).item()

        events = mne.find_events(data, stim_channel='STI 014')
        print('Events BEFORE first changes : ', events)
        print('translate_dict : ', translate_dict)
        # eventplot = mne.viz.plot_events(events, data.info['sfreq'])

        new_dict = {}
        new_events = events.copy()
        for key, value in translate_dict.items():
            if key != 'Rest' and key != 'Code' and key != 'rest' and key != 'star' and key != 'IEND':
                new_key = ''.join(x for x in key if x.isdigit())
                # new_key = int(new_key) # retire or ...
            else:
                new_key = key
            new_dict[new_key] = value
        print('new_dict : ', new_dict)

        for (x, y), value in np.ndenumerate(events):
            if y == 2:
                # corect here because one key is associated to one value. better to use inverse dict ?
                new_value = [k for k, v in new_dict.items() if v == value][0]
                if new_value != 'IEND' and new_value != 'rest' and new_value != 'Code' and new_value != 'Rest' and new_value != 'star':
                    new_events[x, y] = new_value

        events_redef = redefine_event_PP(
            new_events, cfg, verbose=True, plot=True)

        if plot:
            eventplot = mne.viz.plot_events(events_redef, data.info['sfreq'])

        events_id = cfg.events_id_PP
        epochs_reject = None  # cfg.epochs_reject_PP

    elif patient_info['protocol'] == 'LG':
        events_redef = redefine_event_LG(
            data, patient_info, cfg, verbose=True, plot=True)
        events_id = cfg.events_id_LG
        epochs_reject = cfg.epochs_reject_LG

    ########### Segmetation and rejection ###########
    tmin, tmax = cfg.erp_window_tmin, cfg.erp_window_tmax
    baseline = cfg.erp_baseline
    # Either 0 or 1, the order of the detrending. 0 is a constant (DC) detrend, 1 is a linear detrend.
    detrend = cfg.erp_detrend

    # eeg and eog chan for rejection
    picks_eeg_eog = mne.pick_types(data.info, eeg=True, eog=True, exclude=[])
    # picks_eeg_eog = mne.pick_types(data.info, eeg=True, eog=True, selection =['Cz', 'E55', 'E62', 'E106', 'E7', 'E80', 'E31', 'E79', 'E54', 'E61', 'E78']) #FOR PATIENT TpTP
    epochs = mne.Epochs(data, events=events_redef, event_id=events_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, detrend=detrend, picks=picks_eeg_eog, on_missing='warn',
                        reject=epochs_reject, preload=True)
    if verbose:
        print('epochs : ', epochs.info)
        print('Number of events :', events_redef.size/3)
        # print('Number of redefined events :', events_redef_equal.size)
        print('Number of epochs :', len(epochs))

    if plot:
        epochs.plot_drop_log(subject=SubName)
        epochs.plot(title=SubName, show=True, block=True, scalings=dict(
            eeg=50e-6, eog=100e-6))  # Here it's possible to click on event to reject
        # epochs["TW/Congurent"].average().plot()
        # epochs["TW/Incongruent"].average().plot()
        # epochs["TPW"].average().plot()

    logger.info('Number of events : ,%s', events.size)
    logger.info('Number of events : ,%s', events.size)
    # gger.info('Number of redefined events :,%s', events_redef_equal.size)
    logger.info('Number of epochs :,%s', len(epochs))

    if save:
        if patient_info['protocol'] == 'PP':
            epochs_name = patient_info['data_save_dir'] + \
                cfg.all_folders_PP['data_epochs_path']
        elif patient_info['protocol'] == 'LG':
            epochs_name = patient_info['data_save_dir'] + \
                cfg.all_folders_LG['data_epochs_path']
        elif patient_info['protocol'] == 'Resting':
            epochs_name = patient_info['data_save_dir'] + \
                cfg.all_folders_Resting['data_epochs_path']
        epochs_name = epochs_name + \
            patient_info['ID_patient'] + '_' + \
            patient_info['protocol'] + cfg.prefix_epochs_PPAP
        print("Saving data : " + epochs_name)

        # epochs_name = cfg.data_epochs_path + data.info['subject_info']['his_id'] + '_' + proto + cfg.prefix_epoched
        # print("Saving data : " + epochs_name)
        epochs.save(epochs_name, overwrite=True)

    return epochs


# From Riham epoching code -> connectivity epochs
def get_epochs_connectivity(data, sub, proto, data_save_dir, cfg, save=True, verbose=True, plot=True):

    # SubName = patient_info['ID_patient']
    # proto = patient_info['protocol']

    assert data.info['sfreq'] == cfg.sfreq, 'Pbl Fréquences d"echantillonnage !!'

    # Define new trigs: 60 event of 10s = 10 min, from first trig or first 1min
    event_id = 303
    nb_event = 60
    size_epoch_s = 10

    if proto == 'PP' or proto == 'LG':
        data_events = mne.find_events(data, stim_channel='STI 014')
        onset_init = data_events[0][0]
    elif proto == 'Resting':  # pas de trig dans ce proto
        onset_init = cfg.sfreq*60
    else:
        print('Error on the protocol name - can"t find it')
        return
    onsets = np.arange(start=onset_init, stop=onset_init+(nb_event *
                       cfg.sfreq*size_epoch_s), step=cfg.sfreq*size_epoch_s, dtype='int64')
    events = np.vstack((onsets, np.ones(nb_event, int), event_id *
                       np.ones(nb_event, int))).T   # event : (times, durations, codes)

    print('New Events: ', events)
    print(data)
    # eventplot = mne.viz.plot_events(events, data.info['sfreq'])
    # plt.show()

    ########### Segmetation and rejection ###########

    tmin, tmax = 0, size_epoch_s

    # eeg and eog chan for rejection
    picks_eeg = mne.pick_types(data.info, eeg=True, exclude=[])
    epochs = mne.Epochs(data, events=events, event_id=event_id, tmin=tmin, tmax=tmax,  # new_events
                        baseline=(None, None), picks=picks_eeg, reject=cfg.epochs_reject_con, preload=True,
                        detrend=1, reject_by_annotation=False)
    if verbose:
        print('epochs : ', epochs.info)
        # print('Number of events :', new_events.size)
        print('Number of epochs :', len(epochs))

    if plot:
        # epochs.plot_drop_log(subject=SubName)
        epochs.plot(title=sub + " " + proto + ' - Click to manually reject event if needed', show=True,
                    block=True, scalings=dict(eeg=200e-6, eog=100e-6))  # Here it's possible to click on event to reject

    logger.info('Number of events : ,%s', events.size)
    logger.info('Number of events : ,%s', events.size)
    logger.info('Number of epochs :,%s', len(epochs))

    if save:
        epochs_name = data_save_dir + cfg.data_con_path
        epochs_name = epochs_name + sub + '_' + proto + cfg.prefix_epo_conn
        print("Saving data : " + epochs_name)

        # epochs_name = cfg.data_epochs_path + data.info['subject_info']['his_id'] + '_' + proto + cfg.prefix_epoched
        # print("Saving data : " + epochs_name)
        epochs.save(epochs_name, overwrite=True)

    return epochs
