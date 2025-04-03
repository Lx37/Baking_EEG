from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import numpy as np
import mne
import pandas as pd
import os
import logging
from datetime import datetime
import sys
import time
import argparse
from getpass import getuser
from sklearn.feature_selection import SelectPercentile, f_classif

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Baking_EEG'))

from _4_decoding import (
    decode_window,
    plot_decoding_results_dashboard,
    plot_group_results,
    decode_cross_subject_fold
)

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
logname = os.path.join(log_dir, datetime.now().strftime('log_%Y-%m-%d.log'))
logging.basicConfig(
    filename=logname,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)



# ---------------------------
# Global Constants
# ---------------------------
# Define the subject IDs for each experimental group.
# This provides a centralized and easily modifiable list of participants for others experiments
SUBJECTS = {
    'controls': ['LAB1', 'LAG6', 'LAT3', 'LBM4', 'LCM2',
                 'LPO5', 'TJL3', 'TJR7', 'TLP8', 'TPC2',
                 'TPLV4', 'TTDV5', 'TYS6'],
    'del': ['TpAB19', 'TpAG51', 'TpAK24', 'TpAK27', 'TpCB15',
            'TpCF1', 'TpDRL3', 'TpJB25', 'TpJB26', 'TpJC5',
            'TpJLR17', 'TpJPS55', 'TpKT33', 'TpLA28', 'TpMB45',
            'TpMM4', 'TpMN42', 'TpPC21', 'TpPM14', 'TpPM31',
            'TpRD38', 'TpSM49'],
    'nodel': ['TpAC23', 'TpAM43', 'TpBD16', 'TpBL47', 'TpCG36',
              'TpFF34', 'TpFL53', 'TpGB8', 'TpGT32', 'TpJA20',
              'TpJPG7', 'TpJPL10', 'TpKS6', 'TpLP11', 'TpMD13',
              'TpMD52', 'TpME22', 'TpPA35', 'TpPI46', 'TpPL48',
              'TpRB50', 'TpRK39', 'TpSD30', 'TpYB41']
}

# ---------------------------
# Configuration functions
# ---------------------------


def configure_paths(user):
    """configure data paths based on the current username in order to be load on different machine

    This function allows the script to adapt to different file system
    structures on various machines by checking the username. It provides
    flexibility for different collaborators or environments.

    Parameters
    ----------
    user : str
        The username obtained via getpass.getuser().

    Returns
    -------
    str
        The base path where the EEG data is expected to be located.
    """
    # Set the base path according to the identified user.
    if user == 'tkz':
        base_path = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG_data'
    elif user == 'adminlocal':
        base_path = 'C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera'
    elif user.lower() == 'tom':
        base_path = '/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_data'
    else:
        # Default path configuration if the user is not explicitly listed.
        # Assumes data is in a 'Baking_EEG_data' folder in the user's home directory.
        base_path = os.path.join(os.path.expanduser('~'), 'Baking_EEG_data')
    logger.info(f'Configured paths for user {user}')
    return base_path


def get_patient_info(subject_id, protocol, base_path):
    """Create a dictionary containing essential information for a subject.

    This centralizes specific metadata on subject required by various processing
    and analysis functions.

    Parameters
    ----------
    subject_id : str
        The unique identifier for the subject.
    protocol : str
        The experimental protocol being analyzed (e.g., 'PP').
    base_path : str
        The root directory containing the subject's data.

    Returns
    -------
    dict
        A dictionary with keys 'ID_patient', 'protocol', 'data_dir',
        and 'data_save_dir'.
    """
    return {
        'ID_patient': subject_id,
        'protocol': protocol,
        'data_dir': base_path,
        'data_save_dir': base_path
    }


def setup_results_dir(patient_info, group_name):
    """Configure and create the directory for saving decoding results.

    This function ensures a consistent and organized directory structure
    for storing results, categorized by group, subject, and protocol.

    Parameters
    ----------
    patient_info : dict
        Dictionary containing subject information (from get_patient_info).
    group_name : str
        The name of the experimental group the subject belongs to.

    Returns
    -------
    str
        The full path to the directory where results for this subject
        should be saved.
    """
    # Construct the path for saving results.
    # Structure: base_path / decoding_results / group_name / subject_protocol
    results_dir = os.path.join(
        patient_info['data_save_dir'],  # Use the designated save directory
        'decoding_results',          # Main folder for all decoding results
        group_name,                  # Subfolder for the specific group
        # Subfolder for this subject/protocol
        f"{patient_info['ID_patient']}_{patient_info['protocol']}"
    )
    # Create the directory if it doesn't exist.
    # exist_ok=True prevents errors if the directory already exists.
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# ---------------------------
# Data Loading Functions
# ---------------------------


def load_epochs_for_decoding(subject_id, group, base_path, verbose=True):
    """Load preprocessed MNE Epochs object for a given subject and group.

    This function handles locating the correct preprocessed file (ICA or NOICA) and loading the EEG data. 
    it also extracts the data corresponding to 'PP' and 'AP' conditions, which are
    the classes to be decoded. It ensures data is loaded as 3D  (n_epochs, n_channels, n_times).

    Parameters
    ----------
    subject_id : str
        The identifier of the subject whose data is to be loaded.
    group : str
        The group the subject belongs to ('controls', 'del', 'nodel').
        This determines the subdirectory structure to search within.
    base_path : str
        The base directory where data folders are located.
    verbose : bool, optional
        If True, print status messages during loading. Default is True.

    Returns
    -------
    epochs : mne.Epochs
        The loaded MNE Epochs object containing metadata and data.
    XPP : np.ndarray
        EEG data for the 'PP' condition (n_epochs_pp, n_channels, n_times).
    XAP : np.ndarray
        EEG data for the 'AP' condition (n_epochs_ap, n_channels, n_times).

    Raises
    ------
    ValueError
        If the specified group is unknown.
    FileNotFoundError
        If no suitable preprocessed file (_noICA or _ICA) is found for the subject.
    """
    start_time = time.time()
    # Determine the path to the data based on the subject's group.
    if group == 'controls':
        data_path = os.path.join(base_path, 'PP_CONTROLS_0.5', 'data_epochs')
    elif group in ['del', 'nodel']:
        data_path = os.path.join(
            base_path, f'PP_PATIENTS_{group.upper()}_0.5', 'data_epochs')
    else:
        raise ValueError(f"Unknown group: {group}")

    # Construct filenames for the preprocessed data.
    file_noICA = os.path.join(
        data_path, subject_id + '_PP_preproc_noICA_PP-epo_ar.fif')
    file_ICA = os.path.join(data_path, subject_id +
                            '_PP_preproc_ICA_PP-epo_ar.fif')

    # Check which file exists and select it for loading.
    if os.path.exists(file_noICA):
        fif_fname = file_noICA
    elif os.path.exists(file_ICA):
        fif_fname = file_ICA
    else:
        raise FileNotFoundError(
            f"No preprocessed epoch file found for subject {subject_id} in {data_path}")

    if verbose:
        print(f"\nProcessing subject {subject_id} from group {group}")
        print(f"Loading file: {fif_fname}")

    # Load the epochs data from the selected .fif file.
    # Use a warning log level context to suppress verbose MNE loading messages unless necessary.
    # preload=True loads data into memory immediately. proj=False ignores projector operations.
    with mne.utils.use_log_level('warning'):
        epochs = mne.read_epochs(
            fif_fname, proj=False, verbose=verbose, preload=True)

    # Extract the EEG data specifically for the 'PP' and 'AP' conditions.
    # pick(picks="eeg") selects only EEG channels, excluding others (like EOG).
    # get_data(copy=False) avoids making an unnecessary copy of the data for efficiency.
    XPP = epochs["PP"].pick(picks="eeg").get_data(copy=False)
    XAP = epochs["AP"].pick(picks="eeg").get_data(copy=False)

    if verbose:
        print(f'XPP shape: {XPP.shape}')  # (n_epochs_pp, n_channels, n_times)
        print(f'XAP shape: {XAP.shape}')  # (n_epochs_ap, n_channels, n_times)
        print(f"Number of total epochs loaded: {len(epochs)}")
        print(f"Time taken for loading: {time.time() - start_time:.2f} sec")

    return epochs, XPP, XAP

# ---------------------------
# Decoding Execution Functions
# ---------------------------


def run_decoding_on_subject(subject_id, protocol='PP',
                            save=True, verbose=True, plot=True, group=None, base_path=None):
    """Perform intra-subject decoding for a single participant.

    This function orchestrates the decoding process for one subject:
    1. Configures paths and subject information.
    2. Loads the necessary EEG data.
    3. Prepares the data (X) and labels (y) for classification.
    4. Defines the machine learning pipeline (preprocessing + classifier).
    5. Sets up the cross-validation strategy.
    6. Runs the decoding using the 'decode_window' function.
    7. Logs, saves and plots the results.

    Parameters
    ----------
    subject_id : str
        Identifier of the subject to process.
    protocol : str, optional
        Experimental protocol identifier. Default is 'PP'. We can add others.
    save : bool, optional
        Whether to save the decoding results to disk. Default is True.
    verbose : bool, optional
        Whether to print detailed status messages. Default is True.
    plot : bool, optional
        Whether to generate and save plots of the results. Default is True.
    group : str, optional
        The group the subject belongs to. Required for loading data correctly.
        If None, an error might occur in load_epochs_for_decoding. Default is None.
    base_path : str, optional
        Base path for data. If None, it's configured automatically based on user.
        Default is None.

    Returns
    -------
    float or None
        The mean cross-validation score (AUC) for the subject, or None if
        an error occurred during processing.
    """
    total_start = time.time()  # Time the entire function execution.
    try:
        # Configure base path if not provided.
        if base_path is None:
            base_path = configure_paths(getuser())
        # Get subject-specific information.
        patient_info = get_patient_info(subject_id, protocol, base_path)
        if verbose:
            logger.info(
                f"Starting decoding for subject {subject_id} (group: {group}, protocol: {protocol})")

        # Load the epoched data for the specified subject and group.
        epochs, XPP, XAP = load_epochs_for_decoding(
            subject_id, group, base_path, verbose)

        # Prepare the data matrix (X) and target vector (y) for the classifier.
        # Concatenate data from the two conditions ('AP' and 'PP').
        # Shape: (n_total_epochs, n_channels, n_times)
        X = np.concatenate([XAP, XPP], axis=0)
        # Create labels: 0 for 'AP' condition, 1 for 'PP' condition.
        # Shape: (n_total_epochs,)
        y = np.concatenate([np.zeros(XAP.shape[0]), np.ones(XPP.shape[0])])
        # Ensure the number of samples matches between X and y.
        assert X.shape[0] == y.shape[0], f"Mismatch in X and y shapes: X={X.shape}, y={y.shape}"

        if verbose:
            print(f'Combined X shape: {X.shape}')
            print(f'Combined y shape: {y.shape}')

        # Define the machine learning pipeline.
        # This sequences the steps applied to the data.
        clf = make_pipeline(
            StandardScaler(),  # Standardize features: important for SVM
            # It's possible not to use feature selection based on ANOVA F-test to reduce dimensionnality
            SelectPercentile(f_classif, percentile=15),
            # Linear SVM classifier.
            SVC(kernel='linear', probability=True, class_weight='balanced')
            # probability=True: Needed for ROC AUC calculation.
            # class_weight='balanced': Adjusts weights inversely proportional to class frequencies, useful for imbalanced datasets (our case)
        )

        # StratifiedKFold ensures that the proportion of classes is maintained in each fold.
        # shuffle=True randomizes the data before splitting. random_state ensures reproducibility.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Run the decoding process using the imported function.
        # This performs cross-validation, trains/tests the classifier, and calculates metrics.
        # n_jobs='auto' attempts to use all available CPU cores for parallelization.
        probas, predictions, cv_scores, scores_time, metrics = decode_window(
            X, y, clf=clf, cv=cv, n_jobs='auto'
        )

        if verbose:
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            print(f"Subject {subject_id} decoding complete.")
            print(f"Mean CV AUC: {mean_score:.3f} +/- {std_score:.3f}")
            print(f"Total time elapsed: {time.time() - total_start:.2f} sec")
            logger.info(
                f"Subject {subject_id} - Cross-validation AUC scores: {cv_scores}")
            logger.info(
                f"Subject {subject_id} - Mean CV AUC: {mean_score:.3f}, Std: {std_score:.3f}")
            for metric_name, metric_value in metrics.items():
                logger.info(
                    f"Subject {subject_id} - Overall {metric_name}: {metric_value:.3f}")

        if save:

            results_dir = setup_results_dir(patient_info, group)

            results_to_save = {
                'subject_id': subject_id,
                'group': group,
                'protocol': protocol,
                # Predicted probabilities for each trial (example : condition 0 vs 1 ; we have an array (0.3,0.7))
                'probas': probas,
                # Predicted class (0 or 1) for each trial
                'predictions': predictions,
                'cv_scores': cv_scores,  # AUC score for each CV fold
                'scores_time': scores_time,  # Time-resolved AUC scores
                'metrics': metrics,  # Dictionary of overall performance metrics
            }

            np.save(os.path.join(
                results_dir, f"decoding_{subject_id}_{group}_results.npy"), results_to_save)

            csv_results = {
                'subject_id': subject_id,
                'group': group,
                'protocol': protocol,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
            }
            csv_results.update(metrics)

            df = pd.DataFrame([csv_results])

            csv_path = os.path.join(
                results_dir, f"decoding_{subject_id}_{group}_results.csv")
            df.to_csv(csv_path, index=False)
            if verbose:
                logger.info(f"Results for {subject_id} saved in {results_dir}")
                print(f"Results saved to {results_dir}")

        if plot:
            plot_decoding_results_dashboard(
                # Data and results needed for plotting
                epochs, X, y, probas, predictions, cv_scores, scores_time,
                subject_id, group,  # Identifiers for titles/filenames
                results_dir if save else None  # Directory to save plots
            )
            if verbose:
                print(f"Plots saved for {subject_id}")

        return np.mean(cv_scores)

    except Exception as e:

        logger.error(
            f"Error decoding subject {subject_id}: {e}", exc_info=True)
        print(f"Error decoding subject {subject_id}: {e}")
        return None


def run_on_group(subjects, group_name, protocol='PP',
                 save=True, verbose=True, plot=True, base_path=None):
    """Run intra-subject decoding for all subjects within a specified group.

    This function iterates through a list of subject IDs, calling
    'run_decoding_on_subject' for each one. It collects the results,
    calculates group statistics (mean and std deviation of scores),
    and optionally saves and plots the aggregated group results.

    Parameters
    ----------
    subjects : list of str
        A list containing the identifiers of all subjects in the group.
    group_name : str
        The name of the group being processed (e.g., 'controls', 'del', 'nodel').
    protocol : str, optional
        Experimental protocol identifier. Default is 'PP'.
    save : bool, optional
        Whether to save individual and group results. Default is True.
    verbose : bool, optional
        Whether to print detailed status messages. Default is True.
    plot : bool, optional
        Whether to generate plots for individuals and the group. Default is True.
    base_path : str, optional
        Base path for data. If None, it's configured automatically. Default is None.

    Returns
    -------
    dict
        A dictionary where keys are subject IDs and values are their mean
        CV scores (or None if decoding failed for that subject).
    """
    total_start = time.time()  # Time the entire group processing.
    # Configure base path if not provided.
    if base_path is None:
        base_path = configure_paths(getuser())

    if verbose:
        logger.info(
            f"Starting intra-subject decoding for group {group_name} ({len(subjects)} subjects)")
        print(
            f"\n=== Decoding Group: {group_name} ({len(subjects)} subjects) ===")

    results = {}

    for i, subject_id in enumerate(subjects, 1):
        if verbose:
            print(
                f"\n--- Processing Subject {i}/{len(subjects)}: {subject_id} ---")

        score = run_decoding_on_subject(subject_id, protocol,
                                        save, verbose, plot, group_name, base_path)
        # Store the returned score (or None if failed) in the results dictionary.
        results[subject_id] = score

    # Filter out None values (subjects for whom decoding failed).
    valid_scores = [score for score in results.values() if score is not None]

    # Calculate and report group statistics if there are valid scores.
    if valid_scores:
        mean_group_score = np.mean(valid_scores)
        std_group_score = np.std(valid_scores)
        if verbose:
            logger.info(f"Group {group_name} intra-subject decoding results:")
            logger.info(
                f"Mean AUC: {mean_group_score:.3f} ± {std_group_score:.3f} (based on {len(valid_scores)} subjects)")
            print(f"\n=== Group {group_name} Summary ===")
            print(
                f"Mean AUC: {mean_group_score:.3f} ± {std_group_score:.3f} (from {len(valid_scores)} subjects)")
            print(f"Total time for group: {time.time() - total_start:.2f} sec")

        if save:
            # Define the directory to save group-level summary files.
            group_results_dir = os.path.join(
                base_path, 'decoding_results', f'group_{group_name}')

            os.makedirs(group_results_dir, exist_ok=True)

            group_results_summary = {
                'group': group_name,
                'protocol': protocol,
                'subject_scores': results,
                'mean_score': mean_group_score,
                'std_score': std_group_score,
                'n_subjects_processed': len(subjects),
                'n_subjects_successful': len(valid_scores)
            }

            np.save(os.path.join(group_results_dir,
                    f"group_{group_name}_intra_subject_decoding_summary.npy"), group_results_summary)

            df = pd.DataFrame([
                {'subject_id': subject_id, 'score': score, 'group': group_name}
                for subject_id, score in results.items()
            ])

            df.to_csv(os.path.join(
                group_results_dir, f'group_{group_name}_intra_subject_scores.csv'), index=False)
            if verbose:
                logger.info(
                    f"Group {group_name} summary results saved in {group_results_dir}")
                print(f"Group summary results saved to {group_results_dir}")

        if plot:
            plot_group_results(results,
                               f"{group_name} (Intra-Subject)",
                               save_dir=group_results_dir if save else None)
            if verbose:
                print(f"Group plot generated for {group_name}")

    elif verbose:
        print(f"No valid decoding results obtained for group {group_name}.")
        logger.warning(
            f"No valid scores to calculate group statistics for {group_name}.")

    return results  # Return the dictionary of individual subject scores.


def run_cross_subject(subjects, group_name, protocol='PP', save=True, verbose=True, plot=True, base_path=None):
    """Perform cross-subject decoding using Leave-One-Subject-Out (LOSO).

    This function implements a cross-subject decoding scheme where, for each
    subject, a model is trained on data from all other subjects in the list
    and then tested on the held-out subject. This assesses the generalizability
    of the decoding model across individuals within the specified group.

    Parameters
    ----------
    subjects : list of str
        A list containing the identifiers of all subjects to include in the
        cross-subject analysis.
    group_name : str
        The name associated with this set of subjects (used for saving/logging).
    protocol : str, optional
        Experimental protocol identifier. Default is 'PP'.
    save : bool, optional
        Whether to save the results (scores, metrics, models). Default is True.
    verbose : bool, optional
        Whether to print detailed status messages. Default is True.
    plot : bool, optional
        Whether to generate plots for each fold's test subject. Default is True.
    base_path : str, optional
        Base path for data. If None, it's configured automatically. Default is None.

    Returns
    -------
    mean_score : float or None
        The average AUC score across all test subjects (folds), or None if no
        valid results were obtained.
    std_score : float or None
        The standard deviation of AUC scores across all test subjects, or None.
    models : dict or None
        A dictionary where keys are test subject IDs and values are the trained
        classifier models used to test them, or None.
    """
    total_start = time.time()

    if base_path is None:
        base_path = configure_paths(getuser())

    logger.info(f"Starting cross-subject decoding for group/set: {group_name}")
    print(
        f"\n=== Cross-Subject Decoding ({group_name}, {len(subjects)} subjects) ===")

    # Ensure subjects list is a numpy array for easier indexing.
    subjects = np.array(subjects)

    # Load data for all subjects first to avoid repeated loading within the loop.
    # Dictionary to store loaded data {subject_id: {'epochs':..., 'XAP':..., 'XPP':...}}
    subject_data = {}
    print("Loading data for all subjects...")
    for subject_id in subjects:
        try:
            # Assume all subjects belong to the 'group_name' context for loading path determination.
            epochs, XPP, XAP = load_epochs_for_decoding(
                subject_id, group_name, base_path, verbose=False)
            subject_data[subject_id] = {
                'epochs': epochs,  # Keep epochs for potential plotting later
                'XAP': XAP,
                'XPP': XPP
            }
            if verbose:
                print(
                    f"  Loaded {subject_id}: AP shape {XAP.shape}, PP shape {XPP.shape}")
        except Exception as e:

            logger.error(
                f"Failed to load data for subject {subject_id} in cross-subject setup: {str(e)}")
            print(
                f"  Warning: Could not load data for {subject_id} - {str(e)}. Skipping this subject.")
            # This subject will be excluded from the analysis.
            continue  # We continue to the next subject ID

    # Check if any valid data was loaded.
    valid_subjects = list(subject_data.keys())
    if not valid_subjects:
        logger.error(
            "No valid subject data could be loaded for cross-subject decoding. Aborting.")
        print("Error: No valid subject data loaded. Cannot proceed with cross-subject decoding.")
        return None, None, None

    if len(valid_subjects) < len(subjects):
        print(
            f"Proceeding with {len(valid_subjects)} out of {len(subjects)} subjects due to loading errors.")
        # Update the subjects array to only include those successfully loaded.
        subjects = np.array(valid_subjects)

    if save:
        results_dir = os.path.join(
            base_path, 'decoding_results', f'cross_subject_{group_name}')
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")
    else:
        results_dir = None

    # Initialize LeaveOneGroupOut cross-validator. Here, we consider that each 'group' is one subject.
    logo = LeaveOneGroupOut()
    # 0 à n-1 subjects ==> {0,1,2.., number of subject -1}
    indices = np.arange(len(subjects))

    # Dictionaries to store results from each fold (where a fold tests one subject).
    scores = {}  # {test_subject_id: score}
    metrics_all = {}  # {test_subject_id: metrics_dict}
    models = {}  # {test_subject_id: trained_model}

    # Iterate through each fold defined by LeaveOneGroupOut.

    # Each iteration leaves one subject out for testing.
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(indices, groups=indices), 1):
        # Identify the subject held out for testing in this fold.
        test_subject_index = test_idx[0]
        test_subject_id = subjects[test_subject_index]

        # Identify the subjects used for training in this fold.
        train_subject_ids = subjects[train_idx]

        if verbose:
            print(f"\n--- Fold {fold_idx}/{len(subjects)} ---")
            print(f"  Test Subject: {test_subject_id}")
            print(f"  Train Subjects: {', '.join(train_subject_ids)}")

        # Aggregate training data from all subjects in the training set for this fold.
        X_train_list, y_train_list = [], []
        for train_subj_id in train_subject_ids:
            # Retrieve pre-loaded data for the training subject.
            data = subject_data.get(train_subj_id)
            if data:
                # Combine AP and PP conditions for this training subject.
                X_subj = np.concatenate([data['XAP'], data['XPP']], axis=0)
                y_subj = np.concatenate([np.zeros(data['XAP'].shape[0]), np.ones(data['XPP'].shape[0])])
                X_train_list.append(X_subj)
                y_train_list.append(y_subj)
            else:
                logger.warning(
                    f"Training data for {train_subj_id} not found during fold {fold_idx}. Skipping subject in train set.")

     
        if not X_train_list:
            print("  Error: No valid training data found for this fold. Skipping fold.")
            logger.error(
                f"No valid training data could be aggregated for fold {fold_idx} (test subject {test_subject_id}). Skipping.")
            continue

        # Concatenate data from all training subjects into single arrays.
        # Shape: (n_total_train_epochs, n_channels, n_times)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0).astype(
            int)

        # Prepare the test data (from the single held-out subject).
        test_data = subject_data[test_subject_id]
        # Shape: (n_test_epochs, n_channels, n_times)
        X_test = np.concatenate([test_data['XAP'], test_data['XPP']], axis=0)
        y_test = np.concatenate([
            np.zeros(test_data['XAP'].shape[0]),
            np.ones(test_data['XPP'].shape[0])
        ]).astype(int)  

        if verbose:
            print(f"  Train data shape: X={X_train.shape}, y={y_train.shape}")
            print(f"  Test data shape: X={X_test.shape}, y={y_test.shape}")

        # Perform the decoding for this specific train/test split (fold).
        try:
            # Call the dedicated function for handling one cross-subject fold.
            # 'save' is set to False here because we save aggregated results later.
            result = decode_cross_subject_fold(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_subject=test_subject_id,
                group_name=group_name,
                protocol=protocol,
                base_path=base_path,  
                save=False)  
        except Exception as e:
        
            logger.error(
                f"Error during cross-subject decoding fold {fold_idx} (test: {test_subject_id}): {str(e)}", exc_info=True)
            print(f"  Error decoding fold {fold_idx}: {str(e)}")
            continue  # Skip to the next fold

       
        if result is not None:
            score, metrics, model, probas, predictions, scores_time = result
            scores[test_subject_id] = score  # Store the main score (e.g., AUC)
          
            metrics_all[test_subject_id] = metric
            models[test_subject_id] = model

            if verbose:
                print(
                    f"  Fold {fold_idx} - Test Subject {test_subject_id} AUC: {score:.3f}")
                for metric_name, metric_value in metrics.items():
                    print(f"    {metric_name}: {metric_value:.3f}")

          
            if plot:
                subject_plot_dir = None
                if save and results_dir:
                    subject_plot_dir = os.path.join(
                        results_dir, f'fold_{fold_idx}_{test_subject_id}_test_results')
                    os.makedirs(subject_plot_dir, exist_ok=True)
                #Here we only have one fold bcs one subject is tested
                plot_decoding_results_dashboard(
                    subject_data[test_subject_id]['epochs'],
                    X_test, y_test, probas, predictions,
                    [score],
                    scores_time,  #time-resolved scores 
                    test_subject_id,
                    f"{group_name}_cross_subject_test",
                    subject_plot_dir  
                )
                if verbose and subject_plot_dir:
                    print(
                        f"  Plots saved for test subject {test_subject_id} in {subject_plot_dir}")

    # After iterating through all folds, process the aggregated results.
    if not scores:
        logger.error(
            f"Cross-subject decoding for {group_name} failed to produce any valid results.")
        print("\nError: No valid results obtained from any cross-subject fold.")
        return None, None, None  

    # Calculate overall mean and standard deviation of the scores across folds.
    mean_score = np.mean(list(scores.values()))
    std_score = np.std(list(scores.values()))

    logger.info(f"Cross-subject decoding for {group_name} complete.")
    logger.info(
        f"Mean test AUC across subjects: {mean_score:.3f} ± {std_score:.3f}")

   
    if save and results_dir:
        scores_df = pd.DataFrame({
            'test_subject_id': list(scores.keys()),
            'auc_score': list(scores.values()),
            'group': group_name
        })
        scores_df.to_csv(os.path.join(
            results_dir, f'cross_subject_{group_name}_scores.csv'), index=False)

        metrics_df = pd.DataFrame(metrics_all).T
        metrics_df.index.name = 'test_subject_id'
        metrics_df.to_csv(os.path.join(
            results_dir, f'cross_subject_{group_name}_metrics.csv'))

        # Save each trained model (takes places). Remind that each model was trained on N-1 subjects and tested on the key subject.
        models_dir = os.path.join(results_dir, 'trained_models')
        os.makedirs(models_dir, exist_ok=True)
        for subj_id, model in models.items():
            joblib.dump(model, os.path.join(
                models_dir, f'model_tested_on_{subj_id}.joblib'))

        if verbose:
            print(
                f"Aggregated cross-subject results (scores, metrics, models) saved in {results_dir}")

    # Plot aggregated group results (distribution of scores when each subject was tested).
    if plot and scores:
        try:
            plot_group_results(
                scores,  
                f"{group_name} (Cross-Subject)",
                save_dir=results_dir if save else None 
            )
            if verbose:
                print(
                    f"Aggregated group plot generated for {group_name} cross-subject results.")
        except Exception as e:
            logger.error(
                f"Error during final group visualization for cross-subject: {str(e)}")
            print(f"  Warning: Could not generate final group plot: {str(e)}")

  
    total_time = time.time() - total_start
    logger.info(
        f"Total time for cross-subject decoding ({group_name}): {total_time / 60:.1f} min")
    print("\n=== Cross-Subject Final Summary ===")
    print(f"Group/Set: {group_name}")
    print(
        f"Mean Test AUC: {mean_score:.3f} ± {std_score:.3f} (across {len(scores)} subjects)")
    print(f"Total execution time: {total_time / 60:.1f} minutes")

    return mean_score, std_score, models


# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == '__main__':
    # Set up the argument parser to handle command-line options.
    # This allows running the script in different modes without code modification.
    parser = argparse.ArgumentParser(
        description='EEG Decoding Module for Intra and Cross-Subject Analysis')
    # Define the '--mode' argument to choose the type of analysis.
    parser.add_argument('--mode', type=str, choices=['one', 'group', 'cross'], required=True,
                        help='Execution mode: "one" (single subject intra-subject), '
                             '"group" (all subjects in a group, intra-subject), '
                             '"cross" (cross-subject decoding within a specified group)')

    # Parse the arguments provided by the user when running the script.
    args = parser.parse_args()

    user = getuser()
    base_path = configure_paths(user)
    logger.info(f"Script started in mode: {args.mode} by user: {user}")

    # Execute the corresponding function based on the selected mode.
    if args.mode == 'one':
        # Run intra-subject decoding for a single subject specified by the user.
        print("\n--- Mode: Single Subject Intra-Subject Decoding ---")
        print("Available subject IDs:")
        for group, subjects_list in SUBJECTS.items():
            print(f"  {group}: {', '.join(subjects_list)}")
        # Prompt user for subject ID and group.
        subject_id = input("\nEnter the Subject ID (e.g., TJR7): ").strip()
        # Ask for the group, suggesting 'controls' as default.
        group = input(
            f"Enter the group for {subject_id} ('controls', 'del', 'nodel') [default based on list or 'controls']: ").strip()
    
        found_group = None
        for g, s_list in SUBJECTS.items():
            if subject_id in s_list:
                found_group = g
                break
        if not group:
            group = found_group if found_group else 'controls'  # Default logic
            print(f"Using group: {group}")
        elif group not in SUBJECTS:
            print(
                f"Warning: Provided group '{group}' not in predefined lists. Ensure data paths are correct.")

        run_decoding_on_subject(subject_id, protocol='PP', group=group,
                                base_path=base_path, save=True, plot=True, verbose=True)

    elif args.mode == 'group':
        # Run intra-subject decoding for all subjects in a specified group.
        print("\n--- Mode: Group Intra-Subject Decoding ---")
        print("Available groups:")
        for group in SUBJECTS.keys():
            print(f" {group}")
        # Prompt user for the group name.
        group = input(
            "Enter the group name to process ('controls', 'del', or 'nodel'): ").strip()
        if group in SUBJECTS:
            subjects_list = SUBJECTS[group]
            run_on_group(subjects_list, group, protocol='PP',
                         base_path=base_path, save=True, plot=True, verbose=True)
        else:
            print(
                f"Error: Unknown group name '{group}'. Please choose from 'controls', 'del', or 'nodel'.")
            logger.error(
                f"Invalid group name entered in 'group' mode: {group}")

    elif args.mode == 'cross':
        # Run cross-subject decoding for a specified list of subjects and group name.
        print("\n--- Mode: Cross-Subject Decoding ---")
        print("Define the subjects and a group name for this cross-subject analysis.")
        print("Available subject IDs:")
        for group, subjects_list in SUBJECTS.items():
            print(f"  {group}: {', '.join(subjects_list)}")
        # Prompt user for the list of subjects and a group name.
        cross_input = input(
            "\nEnter comma-separated subject IDs followed by a group name (e.g. TpAB19,TpAG51,TpAK27 del) (we need at least two subjects): ").strip()
        try:
            parts = cross_input.split()
            if len(parts) < 2:  # Need at least one subject and one group name
                raise ValueError(
                    "Invalid format. Must provide subject IDs and a group name.")
            # Group name is the last part.
            group_name = parts[-1]
            subjects_str = "".join(parts[:-1])
            subjects_list = [s.strip()
                             for s in subjects_str.split(',') if s.strip()]
            if not subjects_list:
                raise ValueError("No subject IDs were provided.")

            print(
                f"Running cross-subject decoding for group '{group_name}' with subjects: {', '.join(subjects_list)}")
            run_cross_subject(subjects_list, group_name, protocol='PP',
                              base_path=base_path, save=True, plot=True, verbose=True)

        except Exception as e:
            print(
                f"Error parsing input or running cross-subject analysis: {e}")
            logger.error(
                f"Error in 'cross' mode setup or execution: {e}", exc_info=True)

    # we can add this option later : loading a pre-trained model 
    # if args.load_model:
    #     model_path = args.load_model
    #     print(f"Loading model from {model_path}...")
    #     try:
    #         loaded_model = joblib.load(model_path)
    #        
    logger.info("Script execution finished.")
    print("\nScript finished.")
