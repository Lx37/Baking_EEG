import numpy as np
import pandas as pd
import platform
import mne
import os
import sys



def load_from_csv(csv_path):
    df = pd.read_csv(csv_path, sep=",", keep_default_na=False) # was sep=","
    return df


def create_patient_info(sujet, xls_patients_info, protocol, raw_data_dir, data_save_dir):
        
    print('###### Creating Patient Info ######')
    #load all patients info from the excel file and get specific patient info
    all_patients_info = load_from_csv(xls_patients_info)
    ID_patient = sujet
    
    print('Patient: ', ID_patient)
    print('Protocol : ', protocol)
    
    if protocol not in ['PP', 'LG', 'Resting']: #TODO : other protocols : LG' / 'Words' / 'Arythmetic'
        print("Protocol not recognized. Please choose between 'PP', 'LG', 'Resting'")
        sys.exit()
    
    if sujet not in all_patients_info['ID_patient'].values:
        print(f"Patient {sujet} not found in the provided XLS patient information.")
        sys.exit()
    
    Name_File = all_patients_info[all_patients_info['ID_patient'] == sujet]['Name_File_' + protocol].values[0]
    data_fname = raw_data_dir + sujet + '/EEG/' + Name_File 
    Bad_Chans =  all_patients_info[all_patients_info['ID_patient'] == sujet]['Bad_Chans_' + protocol].values[0]
    # Bad chan should be marqued as E23,E125 in the correspondig excel file (no space!)
    # We need to convert it to a list of strings
    bad_sub_chan = []
    if len(Bad_Chans)==0:
        bad_sub_chan = Bad_Chans
    else:
        chanstring = Bad_Chans.split(",")
        for i in range (len(chanstring)):
            bad_sub_chan.append(chanstring[i])
    
    #print('bad_sub_chan : ', bad_sub_chan)
    print('data_fname : ', data_fname)
    print('ici : ', data_fname.endswith('.mff'))

    if data_fname.endswith('.mff'): # EGI .mff raw data format
        EEG_system = 'EGI'
    elif data_fname.endswith('.set'):
        EEG_system = 'Gtec_EEGlab'
    else:
        print('Data format not recognized. Please check path and data file name in excel file.')
        sys.exit()

  
    #create patient_info dictionary
    patient_info = {
        'xls_patients_info': xls_patients_info, 
        'ID_patient': ID_patient, 
        'protocol': protocol, 
        'raw_data_dir': raw_data_dir, 
        'data_save_dir': data_save_dir,
        'data_fname': data_fname, 
        'bad_sub_chan': bad_sub_chan, 
        'EEG_system': EEG_system
        }

    return patient_info


def create_arbo(protocol, patient_info, cfg):
    """
    Create the arborescence for the required analysis
    """
    folder = patient_info['data_save_dir'] + cfg.data_preproc_path
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    folder = patient_info['data_save_dir'] + cfg.stimDict_path
    if not os.path.exists(folder):
        os.makedirs(folder)   
    
    # Create the folders for the preprocessing
    if protocol == 'PP':
        print('###### Creating the PP arborescence folders ######')
        for key in cfg.all_folders_PP:
            folder = patient_info['data_save_dir'] + cfg.all_folders_PP[key]
            if not os.path.exists(folder):
                os.makedirs(folder)
    elif protocol == 'LG':
        print('###### Creating the LG arborescence folders ######')
        for key in cfg.all_folders_LG:
            folder = patient_info['data_save_dir'] + cfg.all_folders_LG[key]
            if not os.path.exists(folder):
                os.makedirs(folder)
    elif protocol == 'Resting':
        print('###### Creating the Resting arborescence folders ######')
        for key in cfg.all_folders_Resting:
            folder = patient_info['data_save_dir'] + cfg.all_folders_Resting[key]
            if not os.path.exists(folder):
                os.makedirs(folder)
        
    else:
        print('Protocol not recognized. Please choose between PP, LG, Resting')
        return


def update_excel_bad_chan(patient_info, bad_chans):
    """
    Update the excel file with the bad channels given for the subject
    """
    df = pd.read_csv(patient_info['xls_patients_info'])
    
    print("df['ID_patient'] : ", df['ID_patient'])
    print("patient_info.ID_patient : ", patient_info['ID_patient'])
    print('bad_chans : ', bad_chans)
    print('[str(i) for i in bad_chans] : ', ''.join(str(i) + ',' for i in bad_chans)[:-1])
    
    # Convert bad_chans to a single string with a comma separator (no space!)
    bad_chans_str = ','.join(str(i) for i in bad_chans)
    
    # Update the DataFrame
    if patient_info['protocol'] == 'PP':
        df.loc[df['ID_patient'] == patient_info['ID_patient'], 'Bad_Chans_PP'] = bad_chans_str
    elif  patient_info['protocol'] == 'LG':
        df.loc[df['ID_patient'] == patient_info['ID_patient'], 'Bad_Chans_LG'] = bad_chans_str
    elif  patient_info['protocol'] == 'Resting':
        df.loc[df['ID_patient'] == patient_info['ID_patient'], 'Bad_Chans_Resting'] = bad_chans_str
    else:
        print('Protocol not recognized when updating the excel file for bab_chans.')
        return 
    #print('df : ', df[df['ID_patient'] == patient_info['ID_patient']]['Bad_Chans_PP'])
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(patient_info['xls_patients_info'], index=False)
    print('csv saved')
   
    return df


def cut_preprocessed_sig(data, patient_info, cfg):

    # Patch for data that have not been cutted around events [Riham Analysis]

    if patient_info['protocol'] != 'Resting':
        events = mne.find_events(data, stim_channel='STI 014')
        event_names = np.zeros((events.shape[0],), dtype='S10') # new array one column of zero's with max lenght of 10 caracters
        print('events : ', events)
        print('events type : ', type(events))
        print('events shape : ', events.shape)
        print('event_names : ', event_names)
        event_id = list(np.unique(events[:, 2]))
        print('event_id : ', event_id)
        
        for x in range(events.shape[0]): # loop over rows
            value = events[x, 2] # take each 3th column
            new_value = [k for k in event_id if k==value][0]
            event_names[x] = new_value
            good_events = events[(event_names!=b'Rest') & (event_names!=b'Code') & (event_names!=b'star') & (event_names!=b'rest'), :] # all events where names is not 'rest'

        i_start = int(good_events[0][0]/data.info['sfreq']-3)
        i_stop =  int(good_events[-1][0]/data.info['sfreq']+3)
            

    data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
    data_name = data_name + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_processed
    print("Saving data : " + data_name)
    if patient_info['protocol'] != 'LG':
        data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)
        if patient_info['EEG_system'] == 'EGI': 
            ######For EGI subjects, save stimulation name dictionary #######
            nameStimDict =  patient_info['data_save_dir'] + cfg.stimDict_path
            nameStimDict = nameStimDict + patient_info['ID_patient'] + '_' + patient_info['protocol'] + cfg.prefix_stimDict ## For the stimuli dictionary (names of stimuli given automatically vs ones we gave the stimuli)
            #np.save(nameStimDict, event_id)
    else:
        #cas particuliers du 'resting'
        if patient_info['ID_patient'] == 'TpDC22J1': 
            i_start = 0
            i_stop =  1050
            data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)
        
        if patient_info['ID_patient'] == 'FP102': 
            i_start = 0
            i_stop =  2204
            data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)
        


