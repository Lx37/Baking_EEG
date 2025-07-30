#tom
import os
import logging
import numpy as np
import mne
from pathlib import Path
import sys
from getpass import getuser
import pandas as pd

logger_path_utils = logging.getLogger(__name__)


def configure_project_paths(current_user_login):
    """Configure input data and output results paths based on current user."""
    if not isinstance(current_user_login, str):
        logger_path_utils.error(
            "Invalid current_user_login type. Must be string."
        )
        current_user_login = "unknown_user"  # Fallback

    logger_path_utils.info(
        "Python executable: %s, Version: %s",
        sys.executable, sys.version.split()[0]
    )
    logger_path_utils.info("Current Working Directory: %s", os.getcwd())

    # User-specific input data paths
    user_input_data_paths = {
        "tom.balay": "/mnt/data/tom.balay/data/Baking_EEG_data",
        "tkz": ("/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/"
                "Baking_EEG_data"),
        "adminlocal": ("C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\"
                       "EEG_2025_CAP_FPerrin_Vera"),
        "tom": "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG_data",
        # Add other users and their paths here
    }
    base_input_data_path = user_input_data_paths.get(
        current_user_login,
        os.path.join(os.path.expanduser("~"),
                     "Baking_EEG_data_fallback"),  # Fallback path
    )
    if current_user_login not in user_input_data_paths:
        logger_path_utils.warning(
            "User '%s' not in input path config, using fallback: %s",
            current_user_login, base_input_data_path
        )
    logger_path_utils.info(
        "Input data path for user '%s': %s",
        current_user_login, base_input_data_path
    )

    output_version_folder_name = "V17"  # Versioning for results
    user_output_results_paths = {
        "tom.balay": (f"/home/tom.balay/results/"
                      f"Baking_EEG_results_{output_version_folder_name}"),
        "tom": (f"/Users/tom/Desktop/ENSC/Stage CAP/"
                f"Baking_EEG_results_{output_version_folder_name}"),
        # Add other users and their output paths here
    }
    base_output_results_path = user_output_results_paths.get(
        current_user_login,
        os.path.join(base_input_data_path,  # Fallback output path
                     f"decoding_results_{output_version_folder_name}"),
    )
    if current_user_login not in user_output_results_paths:
        logger_path_utils.warning(
            "User '%s' not in output path config, using fallback: %s",
            current_user_login, base_output_results_path
        )
    logger_path_utils.info(
        "Output results path for user '%s': %s",
        current_user_login, base_output_results_path
    )
    try:
        os.makedirs(base_output_results_path, exist_ok=True)
    except OSError as e:
        logger_path_utils.error(
            "Failed to create base output directory '%s': %s",
            base_output_results_path, e, exc_info=True
        )
        raise
    return base_input_data_path, base_output_results_path


def setup_logging(logger_name: str, log_level: str = 'INFO') -> logging.Logger:
    """Configure le logging.

    Args:
        logger_name: Nom du logger
        log_level: Niveau de logging

    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def create_patient_info(sujet, xls_patients_info, protocol, raw_data_dir, data_save_dir):

    print('###### Creating Patient Info ######')
    # load all patients info from the excel file and get specific patient info
    all_patients_info = load_from_csv(xls_patients_info)
    ID_patient = sujet

    print('Patient: ', ID_patient)
    print('Protocol : ', protocol)

    # TODO : other protocols : LG' / 'Words' / 'Arythmetic'
    if protocol not in ['PP', 'LG', 'Resting']:
        print("Protocol not recognized. Please choose between 'PP', 'LG', 'Resting'")
        sys.exit()

    if sujet not in all_patients_info['ID_patient'].values:
        print(
            f"Patient {sujet} not found in the provided XLS patient information.")
        sys.exit()

    Name_File = all_patients_info[all_patients_info['ID_patient']
                                  == sujet]['Name_File_' + protocol].values[0]
    data_fname = raw_data_dir + sujet + '/EEG/' + Name_File
    Bad_Chans = all_patients_info[all_patients_info['ID_patient']
                                  == sujet]['Bad_Chans_' + protocol].values[0]
    # Bad chan should be marqued as E23,E125 in the correspondig excel file (no space!)
    # We need to convert it to a list of strings
    bad_sub_chan = []
    if len(Bad_Chans) == 0:
        bad_sub_chan = Bad_Chans
    else:
        chanstring = Bad_Chans.split(",")
        for i in range(len(chanstring)):
            bad_sub_chan.append(chanstring[i])

    # print('bad_sub_chan : ', bad_sub_chan)
    print('data_fname : ', data_fname)
    print('ici : ', data_fname.endswith('.mff'))

    if data_fname.endswith('.mff'):  # EGI .mff raw data format
        EEG_system = 'EGI'
    elif data_fname.endswith('.set'):
        EEG_system = 'Gtec_EEGlab'
    else:
        print('Data format not recognized. Please check path and data file name in excel file.')
        sys.exit()

    # create patient_info dictionary
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
            folder = patient_info['data_save_dir'] + \
                cfg.all_folders_Resting[key]
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
    print('[str(i) for i in bad_chans] : ', ''.join(
        str(i) + ',' for i in bad_chans)[:-1])

    # Convert bad_chans to a single string with a comma separator (no space!)
    bad_chans_str = ','.join(str(i) for i in bad_chans)

    # Update the DataFrame
    if patient_info['protocol'] == 'PP':
        df.loc[df['ID_patient'] == patient_info['ID_patient'],
               'Bad_Chans_PP'] = bad_chans_str
    elif patient_info['protocol'] == 'LG':
        df.loc[df['ID_patient'] == patient_info['ID_patient'],
               'Bad_Chans_LG'] = bad_chans_str
    elif patient_info['protocol'] == 'Resting':
        df.loc[df['ID_patient'] == patient_info['ID_patient'],
               'Bad_Chans_Resting'] = bad_chans_str
    else:
        print('Protocol not recognized when updating the excel file for bab_chans.')
        return
    # print('df : ', df[df['ID_patient'] == patient_info['ID_patient']]['Bad_Chans_PP'])

    # Save the updated DataFrame back to the CSV file
    df.to_csv(patient_info['xls_patients_info'], index=False)
    print('csv saved')

    return df


def cut_preprocessed_sig(data, patient_info, cfg):

    # Patch for data that have not been cutted around events [Riham Analysis]

    if patient_info['protocol'] != 'Resting':
        events = mne.find_events(data, stim_channel='STI 014')
        # new array one column of zero's with max lenght of 10 caracters
        event_names = np.zeros((events.shape[0],), dtype='S10')
        print('events : ', events)
        print('events type : ', type(events))
        print('events shape : ', events.shape)
        print('event_names : ', event_names)
        event_id = list(np.unique(events[:, 2]))
        print('event_id : ', event_id)

        for x in range(events.shape[0]):  # loop over rows
            value = events[x, 2]  # take each 3th column
            new_value = [k for k in event_id if k == value][0]
            event_names[x] = new_value
            good_events = events[(event_names != b'Rest') & (event_names != b'Code') & (
                # all events where names is not 'rest'
                event_names != b'star') & (event_names != b'rest'), :]

        i_start = int(good_events[0][0]/data.info['sfreq']-3)
        i_stop = int(good_events[-1][0]/data.info['sfreq']+3)

    data_name = patient_info['data_save_dir'] + cfg.data_preproc_path
    data_name = data_name + patient_info['ID_patient'] + \
        '_' + patient_info['protocol'] + cfg.prefix_processed
    print("Saving data : " + data_name)
    if patient_info['protocol'] != 'Resting':
        data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)
        if patient_info['EEG_system'] == 'EGI':
            ###### For EGI subjects, save stimulation name dictionary #######
            nameStimDict = patient_info['data_save_dir'] + cfg.stimDict_path
            # For the stimuli dictionary (names of stimuli given automatically vs ones we gave the stimuli)
            nameStimDict = nameStimDict + \
                patient_info['ID_patient'] + '_' + \
                patient_info['protocol'] + cfg.prefix_stimDict
            # np.save(nameStimDict, event_id)
    else:
        # cas particuliers du 'resting'
        if patient_info['ID_patient'] == 'TpDC22J1':
            i_start = 0
            i_stop = 1050
            data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)

        if patient_info['ID_patient'] == 'XL89':
            i_start = 0
            i_stop = 2170
            data.save(data_name, tmin=i_start, tmax=i_stop, overwrite=True)


def setup_analysis_results_directory(base_output_path, analysis_type, subject_or_group_id,
                                     method_suffix="", create_timestamp_subdir=True):
    """
    Create and return a directory path for storing analysis results.

    Args:
        base_output_path (str): Base output directory path
        analysis_type (str): Type of analysis (e.g., 'single_subject', 'group_summary_intra_subject')
        subject_or_group_id (str): Subject ID or group identifier (can include group_protocol format)
        method_suffix (str): Suffix describing the method used
        create_timestamp_subdir (bool): Whether to create timestamped subdirectory

    Returns:
        str: Created directory path
    """
    from datetime import datetime

    # Create main analysis directory with hierarchical structure
    analysis_dir = os.path.join(
        base_output_path, analysis_type, subject_or_group_id)

    if method_suffix:
        analysis_dir = os.path.join(analysis_dir, method_suffix)

    if create_timestamp_subdir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_dir = os.path.join(analysis_dir, timestamp)

    try:
        os.makedirs(analysis_dir, exist_ok=True)
        logger_path_utils.info(
            f"Created analysis results directory: {analysis_dir}")
        
        # Log the hierarchical structure for clarity
        if '_' in subject_or_group_id and len(subject_or_group_id.split('_')) >= 2:
            parts = subject_or_group_id.split('_')
            group_part = parts[0]
            protocol_part = '_'.join(parts[1:])
            logger_path_utils.info(
                f"Organized by Group: {group_part}, Protocol: {protocol_part}")
                
    except OSError as e:
        logger_path_utils.error(
            f"Failed to create analysis directory '{analysis_dir}': {e}")
        raise

    return analysis_dir


def load_from_csv(csv_file_path):
    """
    Load data from CSV file.

    Args:
        csv_file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        return pd.read_csv(csv_file_path)
    except Exception as e:
        logger_path_utils.error(
            f"Failed to load CSV file '{csv_file_path}': {e}")
        raise
