# python run_decoding.py --mode single_subject
# python run_decoding.py --mode group_intra
# python run_decoding.py --mode all_groups_intra
# python run_decoding.py --mode cross_subject_set
# python run_decoding.py --mode all_groups_cross
# python run_decoding.py --mode single_subject --clf_type_override svc
# python run_decoding.py --mode single_subject --clf_type_override logreg

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from scipy import stats as scipy_stats
from sklearn.base import clone
import matplotlib.pyplot as plt
import mne
import pandas as pd
import numpy as np
from getpass import getuser
import argparse
import time
from datetime import datetime
import logging
import sys
import os


package_parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
if package_parent_dir not in sys.path:
    sys.path.insert(0, package_parent_dir)
from Baking_EEG.stats import decoding_stats as bEEG_stats
from Baking_EEG._4_decoding import (
    run_temporal_decoding_analysis,
    create_subject_decoding_dashboard_plots,
    plot_group_mean_scores_barplot,
    run_cross_subject_decoding_for_fold,
    _build_standard_classifier_pipeline
)
# --- Logger configuration ---
log_dir_run_decoding = './logs_run_decoding'
os.makedirs(log_dir_run_decoding, exist_ok=True)
logname_run_decoding = os.path.join(log_dir_run_decoding, datetime.now(
).strftime('log_run_decoding_%Y-%m-%d_%H%M%S.log'))
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s:%(lineno)d] - %(message)s',

    handlers=[logging.FileHandler(
        logname_run_decoding), logging.StreamHandler(sys.stdout)]
)
logger_run_decoding = logging.getLogger(__name__)
logging.getLogger('Baking_EEG._4_decoding').setLevel(logging.INFO)
logging.getLogger('Baking_EEG.stats.decoding_stats').setLevel(logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --- Event code definitions for specific plots ---
# These define specific event codes or families of event codes used for targeted decoding comparisons.
PP_CODES_FOR_SPECIFIC_COMPARISON = ["PP/10", "PP/20", "PP/30"]
# AP_FAMILIES_FOR_SPECIFIC_COMPARISON creates a dictionary where keys are "AP_FAMILY_X"
# and values are lists of more specific AP codes (e.g., "AP/1X", "AP/2X", "AP/3X").
# This corresponds to actions related to different units (e.g., unit 1, unit 2, ... unit 6)
AP_FAMILIES_FOR_SPECIFIC_COMPARISON = {
    f"AP_FAMILY_{unit_digit}": [f"AP/{decade}{unit_digit}" for decade in [1, 2, 3]]
    for unit_digit in range(1, 7)  # 6 families, corresponding to 6 different AP
}
# Example event_id mapping from original data:
# {'PP/10': 110, 'PP/20': 120, 'PP/30': 130, ...
#  'AP/11': 111, 'AP/12': 112, ..., 'AP/36': 136}

# --- Configuration for data loading ---
# Defines how to load data for main decoding (all Prepared Postures vs all Action Potentials).
CONFIG_LOAD_MAIN_DECODING = {
    "XPP_ALL": "PP/",  # Loads all events starting with "PP/"
    "XAP_ALL": "AP/",  # Loads all events starting with "AP/"
}
# Merges main decoding config with specific comparison configs for single subject analysis.
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT = {
    **CONFIG_LOAD_MAIN_DECODING,
    "PP_FOR_SPECIFIC_COMPARISON": PP_CODES_FOR_SPECIFIC_COMPARISON,
    **AP_FAMILIES_FOR_SPECIFIC_COMPARISON,
}

# --- Global constants and configuration ---
N_JOBS_PROCESSING = "auto"  # Default for n_jobs, can be overridden by CLI
CLASSIFIER_MODEL_TYPE = "svc"  # Default classifier type
# For stats on CV folds within a single subject's analysis.
N_PERMUTATIONS_INTRA_SUBJECT = 1024 # Number of permutations for intra-subject cluster tests
# For stats across subjects at the group level.
N_PERMUTATIONS_GROUP_LEVEL = 1024 # Number of permutations for group-level cluster tests

# Configuration for cluster thresholding in statistical tests
# 'tfce' (Threshold-Free Cluster Enhancement) is often more sensitive.
# 'stat' would use a fixed t-value threshold.
GROUP_LEVEL_STAT_THRESHOLD_TYPE = "tfce"
# T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING: Only used if GROUP_LEVEL_STAT_THRESHOLD_TYPE is 'stat'.
T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING = None # 2.0 or calculated based on df
# INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG: Configuration for TFCE when doing stats on CV folds for one subject.
INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG = {"start": 0.1, "step": 0.1} # Standard TFCE parameters

# Flags to control parts of the analysis
COMPUTE_INTRA_SUBJECT_STATISTICS = True # If True, computes FDR & Cluster on CV folds for *each subject's* curves
COMPUTE_GROUP_LEVEL_STATISTICS = True   # If True, computes FDR & Cluster on *group-aggregated* curves/TGMs
SAVE_ANALYSIS_RESULTS = True
GENERATE_PLOTS = True
COMPUTE_TEMPORAL_GENERALIZATION_MATRICES = True # Compute TGM for the main PP/all vs AP/all decoding.
CHANCE_LEVEL_AUC_SCORE = 0.5 # Standard chance level for ROC AUC

# Mapping of subject IDs to their respective groups
SUBJECT_GROUPS_MAPPING = {
    "controls": [
        "LAB1", "LAG6", "LAT3", "LBM4", "LCM2", "LPO5", "TJL3",
        "TJR7", "TLP8", "TPC2", "TPLV4", "TTDV5", "TYS6",
    ],
    "del": [
        "TpAB19", "TpAG51", "TpAK24", "TpAK27", "TpCB15", "TpCF1", "TpDRL3",
        "TpJB25", "TpJB26", "TpJC5", "TpJLR17", "TpJPS55", "TpKT33",
        "TpLA28", "TpMB45", "TpMM4", "TpMN42", "TpPC21", "TpPM14",
        "TpPM31", "TpRD38", "TpSM49",
    ],
    "nodel":[ 
        "TpAC23", "TpAM43", "TpBD16", "TpBL47", "TpCG36", "TpFF34", "TpFL53",
        "TpGB8", "TpGT32", "TpJA20", "TpJPG7", "TpJPL10", "TpKS6",
        "TpLP11", "TpMD13", "TpMD52", "TpME22", "TpPA35", "TpPI46",
        "TpPL48", "TpRB50", "TpRK39", "TpSD30", "TpYB41",
    ],
}

# --- Path and directory management ---

def configure_project_paths(current_user_login):
    """Configures input data and output results paths based on the current user."""
    logger_run_decoding.info(
        f"Python executable: {sys.executable}, Version: {sys.version.split()[0]}"
    )
    logger_run_decoding.info(f"Current Working Directory: {os.getcwd()}")

    # User-specific input data paths
    user_input_data_paths = {
        "tom.balay": "/mnt/data/tom.balay/Baking_EEG_data",
        "tkz": "/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG_data",
        "adminlocal": "C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera",
        "tom": "/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_data",
        # Add other users and their paths here
    }
    base_input_data_path = user_input_data_paths.get(
        current_user_login,
        os.path.join(os.path.expanduser("~"), "Baking_EEG_data_fallback"),
    )
    if current_user_login not in user_input_data_paths:
        logger_run_decoding.warning(
            f"User '{current_user_login}' not in input path config, using fallback: {base_input_data_path}"
        )
    logger_run_decoding.info(
        f"Input data path for {current_user_login}: {base_input_data_path}"
    )

    output_version_folder_name = "V5_finish_RF" # Versioning for results
    user_output_results_paths = {
        "tom.balay": f"/home/tom.balay/results/Baking_EEG_results_{output_version_folder_name}",
        "tom": f"/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_results_{output_version_folder_name}",
        # Add other users and their output paths here
    }
    base_output_results_path = user_output_results_paths.get(
        current_user_login,
        os.path.join(base_input_data_path, f"decoding_results_{output_version_folder_name}"), 
    )
    if current_user_login not in user_output_results_paths:
        logger_run_decoding.warning(
            f"User '{current_user_login}' not in output path config, using fallback: {base_output_results_path}"
        )
    logger_run_decoding.info(
        f"Output results path for {current_user_login}: {base_output_results_path}"
    )
    os.makedirs(base_output_results_path, exist_ok=True)
    return base_input_data_path, base_output_results_path


def setup_analysis_results_directory(
    base_output_path, analysis_type_folder_name, group_identifier, subfolder_name=None
):
    """Helper function to create nested directories for storing analysis results."""
    results_directory_path = os.path.join(
        base_output_path, analysis_type_folder_name, group_identifier
    )
    if subfolder_name:
        results_directory_path = os.path.join(results_directory_path, subfolder_name)
    os.makedirs(results_directory_path, exist_ok=True)
    return results_directory_path


# --- Data loading function ---

def load_epochs_data_for_decoding(
    subject_identifier,
    group_affiliation, # Group name used for path determination
    base_input_data_path,
    conditions_to_load=None, # Dict: {condition_name: event_specifier}
    verbose_logging=True,
):
    """
    Loads preprocessed MNE Epochs data for a subject and extracts data for specified conditions.

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation of the subject (e.g., 'controls', 'del').
                                 Used to determine the data subfolder.
        base_input_data_path (str): The root directory for input data.
        conditions_to_load (dict, optional): A dictionary specifying conditions to extract.
            Keys are custom names for the extracted data arrays, values are MNE event
            specifiers (e.g., 'Event/Type', ['Event/1', 'Event/2'], or 'Event/Prefix/').
            If None, defaults to `CONFIG_LOAD_MAIN_DECODING`.
        verbose_logging (bool): If True, enables detailed logging of the loading process.

    Returns:
        tuple: (mne.Epochs or None, dict)
            - mne.Epochs object if loading was successful, else None.
            - Dictionary where keys are `condition_name` from `conditions_to_load` and
              values are NumPy arrays of extracted epoch data (n_epochs, n_channels, n_times).
              If a condition has no data, the value is an empty array.
    """
    start_time = time.time()
    group_affiliation_lower = group_affiliation.lower()
    data_root_path = None

    # --- Path determination logic for different groups ---
    # This section tries to find the correct data directory based on group affiliation.
    if group_affiliation_lower == "controls":
        potential_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["del", "nodel"]:
        potential_path = os.path.join(
            base_input_data_path, f"PP_PATIENTS_{group_affiliation.upper()}_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path

    # Fallback if primary path logic fails
    if not data_root_path:
        detected_group = next(
            (
                g
                for g, s_list in SUBJECT_GROUPS_MAPPING.items()
                if subject_identifier in s_list
            ),
            None,
        )
        if detected_group:
            logger_run_decoding.warning(
                f"Original group path for '{group_affiliation}' not found for '{subject_identifier}'. "
                f"Found subject in '{detected_group}'. Using its path convention."
            )
            group_affiliation_lower = detected_group.lower() # Update group based on detection
            if group_affiliation_lower == "controls":
                data_root_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
            elif group_affiliation_lower in ["del", "nodel"]:
                data_root_path = os.path.join(
                    base_input_data_path, f"PP_PATIENTS_{detected_group.upper()}_0.5"
                )
        else: # Generic fallback if subject not in mapping
            potential_path_generic_group = os.path.join(
                base_input_data_path, f"PP_{group_affiliation.upper()}_0.5"
            )
            if os.path.isdir(potential_path_generic_group):
                data_root_path = potential_path_generic_group
            else:
                data_root_path = base_input_data_path 

    if not data_root_path or not os.path.isdir(data_root_path):
        raise FileNotFoundError(
            f"Data directory for subject '{subject_identifier}' (group: '{group_affiliation}') "
            f"not found. Attempted path: '{data_root_path}'. Base input: '{base_input_data_path}'."
        )

    epochs_file_path_base = os.path.join(data_root_path, "data_epochs")
    # Handle potential variations in subject ID naming (e.g., with/without 'Tp')
    possible_subject_ids = [subject_identifier, subject_identifier.replace("Tp", "")]
    # Account for different preprocessing stages/suffixes in filenames
    possible_suffixes = ["noICA_PP", "ICA_PP", ""] # Order matters: try more specific first
    fname_candidates = []
    for s_id_cand in possible_subject_ids:
        for suffix_cand in possible_suffixes:
            base_name = f"{s_id_cand}_PP_preproc"
            if suffix_cand: # Add suffix if it's not empty
                base_name += f"_{suffix_cand}"
            fname_candidates.append(
                os.path.join(epochs_file_path_base, f"{base_name}-epo_ar.fif")
            )
    # Find the first existing file from candidates
    epochs_fif_filename = next(
        (f for f in fname_candidates if os.path.exists(f)), None
    )

    if not epochs_fif_filename:
        raise FileNotFoundError(
            f"No preprocessed epoch FIF file found for '{subject_identifier}' in '{epochs_file_path_base}'. "
            f"Checked candidates (first 5 of {len(fname_candidates)}): {fname_candidates[:5]}"
        )

    if verbose_logging:
        logger_run_decoding.info(f"Loading epoch data from: {epochs_fif_filename}")
    try:
        # Suppress MNE's own verbose logging during read if not needed
        with mne.utils.use_log_level("WARNING"):
            epochs_object = mne.read_epochs(
                epochs_fif_filename, proj=False, verbose=False, preload=True
            )
    except Exception as e:
        logger_run_decoding.error(
            f"Failed to read epochs file {epochs_fif_filename} for subject {subject_identifier}: {e}",
            exc_info=True,
        )
        return None, {}

    # Get dimensions from loaded EEG data
    num_eeg_channels = len(epochs_object.copy().pick(picks="eeg").ch_names)
    num_time_points = len(epochs_object.times)
    extracted_data = {}

    # Use provided conditions_to_load or default to main decoding config.
    actual_conditions_to_process = (
        conditions_to_load
        if conditions_to_load is not None
        else CONFIG_LOAD_MAIN_DECODING
    )
    if not isinstance(actual_conditions_to_process, dict):
        logger_run_decoding.error(
            f"'conditions_to_load' must be a dict or None. Received {type(actual_conditions_to_process)}. "
            "Using default for main decoding."
        )
        actual_conditions_to_process = CONFIG_LOAD_MAIN_DECODING

    if verbose_logging:
        logger_run_decoding.info(
            f"Processing conditions: {list(actual_conditions_to_process.keys())}"
        )

    # --- Data extraction per condition ---
    for condition_name, specifier in actual_conditions_to_process.items():
        event_keys_to_select = []
        # Default to empty array if no data found for a condition
        empty_data_array = np.empty((0, num_eeg_channels, num_time_points))

        if isinstance(specifier, str): # Single string specifier
            if specifier.endswith("/") or specifier.endswith("*"): # Prefix matching
                prefix = specifier.rstrip("/*")
                event_keys_to_select = [
                    k for k in epochs_object.event_id if k.startswith(prefix)
                ]
            elif specifier in epochs_object.event_id: # Exact match
                event_keys_to_select = [specifier]
        elif isinstance(specifier, list): # List of exact event keys
            event_keys_to_select = [
                k for k in specifier if k in epochs_object.event_id
            ]

        if not event_keys_to_select and specifier: # If specifier was given but no keys matched
            if verbose_logging:
                logger_run_decoding.debug(
                    f"No event keys found for specifier '{specifier}' (condition: {condition_name}) "
                    f"in subject {subject_identifier}'s epochs. Event IDs available: {list(epochs_object.event_id.keys())}"
                )
            extracted_data[condition_name] = empty_data_array
            continue

        try:
            if not event_keys_to_select: # If still no keys (e.g. empty specifier list)
                extracted_data[condition_name] = empty_data_array
                continue

            selected_epochs = epochs_object[event_keys_to_select]
            if len(selected_epochs) > 0:
                extracted_data[condition_name] = selected_epochs.pick(
                    picks="eeg" # Select only EEG channels
                ).get_data(copy=False) # Avoid unnecessary data copying
                if verbose_logging:
                    logger_run_decoding.debug(
                        f"  Extracted {extracted_data[condition_name].shape[0]} epochs for '{condition_name}'."
                    )
            else:
                extracted_data[condition_name] = empty_data_array
        except KeyError: # If MNE selection `epochs_object[...]` fails
            extracted_data[condition_name] = empty_data_array
            if verbose_logging:
                logger_run_decoding.debug(
                    f"KeyError (no events found) for specifier '{specifier}' (condition: {condition_name}) "
                    f"in subject {subject_identifier}."
                )
        except Exception as e_ex: # Catch other potential errors
            logger_run_decoding.error(
                f"Error extracting data for condition '{condition_name}' (subject {subject_identifier}): {e_ex}",
                exc_info=True,
            )
            extracted_data[condition_name] = empty_data_array

    total_loaded_epochs = sum(
        arr.shape[0] for arr in extracted_data.values() if hasattr(arr, "ndim") and arr.ndim == 3
    )
    if total_loaded_epochs == 0 and any(actual_conditions_to_process.values()):
        logger_run_decoding.error(
            f"CRITICAL: No data loaded for ANY specified conditions for subject {subject_identifier}."
        )

    # Log if any condition resulted in empty data despite being requested
    for cn, da in extracted_data.items():
        if da.size == 0 and actual_conditions_to_process.get(cn):
            logger_run_decoding.warning(
                f"Empty data array for condition '{cn}' (specifier: '{actual_conditions_to_process.get(cn)}') "
                f"for subject {subject_identifier}."
            )

    if verbose_logging:
        shapes_log = ", ".join(
            [f"{name}={arr.shape}" for name, arr in extracted_data.items()]
        )
        logger_run_decoding.info(
            f"  Data shapes for '{subject_identifier}': {shapes_log}. Loaded in {time.time() - start_time:.2f}s"
        )
    return epochs_object, extracted_data


# --- Plotting functions for group stats ---
def plot_group_temporal_decoding_statistics(
    time_points_array,
    mean_group_temporal_scores,
    group_identifier_for_plot,
    output_directory_path,
    std_error_group_temporal_scores=None, # SEM for shading
    cluster_p_value_map_1d=None, # P-value map from cluster test for significance bars
    fdr_significance_mask_1d=None, # Boolean mask from FDR for significance bars
    chance_level=CHANCE_LEVEL_AUC_SCORE,
):
    """Plots group-level temporal decoding statistics with significance overlays."""
    logger_run_decoding.info(
        f"Plotting group temporal decoding statistics for: {group_identifier_for_plot}"
    )
    plt.switch_backend("Agg") # Ensure non-interactive backend (for cluster)
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot mean group score
    ax.plot(
        time_points_array,
        mean_group_temporal_scores,
        label="Mean Group AUC",
        color="black",
        linewidth=2.5,
    )
    # Plot Standard Error of the Mean (SEM) as a shaded area
    if std_error_group_temporal_scores is not None:
        ax.fill_between(
            time_points_array,
            mean_group_temporal_scores - std_error_group_temporal_scores,
            mean_group_temporal_scores + std_error_group_temporal_scores,
            color="gray",
            alpha=0.25,
            label="SEM",
        )

    ax.axhline(
        chance_level, color="dimgray", linestyle="--", linewidth=1.2, label=f"Chance ({chance_level})"
    )
    # Plot stimulus onset line if time points are available
    if time_points_array is not None and time_points_array.size > 0:
        ax.axvline(0, color="firebrick", linestyle=":", linewidth=1.2, label="Stimulus Onset")

    # --- Significance bar plotting logic ---
    # Position significance bars below the data and chance level to avoid overlap.
    finite_mean_scores = mean_group_temporal_scores[np.isfinite(mean_group_temporal_scores)]
    min_score_for_sig_bar = (
        np.min(finite_mean_scores) if finite_mean_scores.size > 0 else chance_level
    )
    sig_y_base = min(min_score_for_sig_bar, chance_level) # Base y for first bar
    sig_bar_offset = 0.03  # Offset below data/chance
    sig_bar_height = 0.01  # Height of the significance bar itself
    current_y_pos = sig_y_base - sig_bar_offset # Y-pos for top of current bar

    # Plot FDR significance bar
    if fdr_significance_mask_1d is not None and \
       np.any(fdr_significance_mask_1d) and \
       time_points_array is not None and \
       time_points_array.size == fdr_significance_mask_1d.size:
        ax.fill_between(
            time_points_array,
            current_y_pos, # Bottom of bar
            current_y_pos + sig_bar_height, # Top of bar
            where=fdr_significance_mask_1d,
            color="deepskyblue",
            alpha=0.7,
            step="mid", # Step-like fill for time windows
            label="FDR p<0.05",
        )
        current_y_pos -= (sig_bar_height + 0.005) # Move down for next potential bar

    # Plot cluster permutation significance bar
    if cluster_p_value_map_1d is not None and \
       np.any(cluster_p_value_map_1d < 0.05) and \
       time_points_array is not None and \
       time_points_array.size == cluster_p_value_map_1d.size:
        ax.fill_between(
            time_points_array,
            current_y_pos,
            current_y_pos + sig_bar_height,
            where=(cluster_p_value_map_1d < 0.05), # Where p-values from map are significant
            color="orangered",
            alpha=0.7,
            step="mid",
            label="Cluster Perm. p<0.05",
        )
        # current_y_pos -= (sig_bar_height + 0.005) # If more bars were to be added

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding Score (ROC AUC)")
    ax.set_title(
        f"Group Temporal Decoding: {group_identifier_for_plot}",
        fontsize=15,
        fontweight="bold",
    )
    # Adjust y-limits to ensure significance bars and data are visible
    ymin_data_plot = np.min(finite_mean_scores) if finite_mean_scores.size > 0 else chance_level - 0.1
    ymax_data_plot = np.max(finite_mean_scores) if finite_mean_scores.size > 0 else chance_level + 0.1
    # Ensure y-axis accommodates the lowest significance bar and data range
    ax.set_ylim(
        min(current_y_pos - sig_bar_height * 0.5, ymin_data_plot - 0.05),
        max(1.01, ymax_data_plot + 0.05),
    )
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout() # Adjust plot to prevent labels from overlapping

    # Sanitize filename for saving
    safe_fname = "".join(
        c if c.isalnum() else "_" for c in group_identifier_for_plot.replace(" ", "_")
    )
    save_path = os.path.join(
        output_directory_path, f"group_temporal_decoding_stats_{safe_fname}.png"
    )
    try:
        fig.savefig(save_path, dpi=200, bbox_inches="tight") # Ensure everything fits
        logger_run_decoding.info(f"Group temporal plot saved: {save_path}")
    except Exception as e:
        logger_run_decoding.error(
            f"Failed to save group temporal plot '{save_path}': {e}", exc_info=True
        )
    plt.close(fig) # Close figure to free memory


def plot_group_tgm_statistics(
    mean_group_tgm_scores,
    time_points_tgm_array,
    # For TGM, cluster permutation is NOT applied, so these will be None or empty
    significant_cluster_masks_tgm, # Should be None or empty list
    cluster_p_values_tgm, # Should be None
    group_identifier_for_plot,
    output_directory_path,
    observed_t_values_map_tgm=None, # Optional: plot t-values instead of AUCs
    fdr_significance_mask_tgm=None, # 2D boolean mask for FDR significance
    chance_level=0.5,
    plot_vmin=None, # Custom vmin for colormap
    plot_vmax=None, # Custom vmax for colormap
):
    """Plots group-level TGM statistics with FDR significance overlay."""
    logger_run_decoding.info(
        f"Plotting group TGM statistics for: {group_identifier_for_plot}"
    )
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Decide whether to plot raw AUC scores or t-values (if provided)
    data_to_plot = (
        observed_t_values_map_tgm
        if observed_t_values_map_tgm is not None
        else mean_group_tgm_scores
    )

    vmin_current, vmax_current = plot_vmin, plot_vmax
    cmap_to_use = "RdBu_r" # Diverging colormap, good for scores/t-values around a central point
    cbar_label_text = (
        "t-value vs chance" if observed_t_values_map_tgm is not None else "Mean AUC"
    )

    # Determine color limits (vmin, vmax) for the plot
    if observed_t_values_map_tgm is not None:  # If plotting t-values
        if vmin_current is None or vmax_current is None:  # Auto-scale if not specified
            finite_t_values_tgm = observed_t_values_map_tgm[np.isfinite(observed_t_values_map_tgm)]
            abs_max_tval_tgm = (
                np.max(np.abs(finite_t_values_tgm)) if finite_t_values_tgm.size > 0 else 3.0 # Default if no finite values
            )
            vmin_current, vmax_current = -abs_max_tval_tgm, abs_max_tval_tgm # Symmetric around 0
    else:  # If plotting AUC scores
        if vmin_current is None:
            vmin_current = 0.35  # Default min AUC for plotting
        if vmax_current is None:
            vmax_current = 0.90  # Default max AUC for plotting

    if time_points_tgm_array is None or time_points_tgm_array.size == 0:
        logger_run_decoding.error(
            f"Time points array is missing for TGM plot of {group_identifier_for_plot}. Cannot plot."
        )
        plt.close(fig)
        return

    im = ax.imshow(
        data_to_plot,
        interpolation="lanczos", # Smoother interpolation
        origin="lower", # Standard for MNE GAT plots
        cmap=cmap_to_use,
        extent=time_points_tgm_array[[0, -1, 0, -1]], # [xmin, xmax, ymin, ymax]
        vmin=vmin_current,
        vmax=vmax_current,
        aspect="auto", # Adjust aspect ratio automatically
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    plot_title_tgm = f"Group TGM: {group_identifier_for_plot}"

    significance_info_parts_tgm = []
    # Check for FDR significance (Cluster significance is not applied to TGM)
    if fdr_significance_mask_tgm is not None and np.any(fdr_significance_mask_tgm):
        significance_info_parts_tgm.append("FDR sig. hatched")

    if significance_info_parts_tgm:
        plot_title_tgm += f"\n({', '.join(significance_info_parts_tgm)}, p<0.05)"
    ax.set_title(plot_title_tgm)

    ax.axvline(0, color="k", linestyle=":", lw=0.8) # Stimulus onset line (vertical)
    ax.axhline(0, color="k", linestyle=":", lw=0.8) # Stimulus onset line (horizontal)

    # Plot FDR significance using hatching
    if fdr_significance_mask_tgm is not None and \
       fdr_significance_mask_tgm.shape == data_to_plot.shape and \
       np.any(fdr_significance_mask_tgm):
        X_coords_fdr_tgm, Y_coords_fdr_tgm = np.meshgrid(time_points_tgm_array, time_points_tgm_array)
        ax.contourf(
            X_coords_fdr_tgm,
            Y_coords_fdr_tgm,
            fdr_significance_mask_tgm,
            levels=[0.5, 1.5], # Levels to define hatched region
            colors="none", # No fill color for contourf
            hatches=["///"], # Hatch pattern
            alpha=0.3, # Transparency of hatches
        )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Adjust fraction and pad for size
    cbar.set_label(cbar_label_text)
    # Only add chance line to colorbar if plotting AUCs
    if observed_t_values_map_tgm is None: # i.e., if plotting AUCs
        cbar.ax.axhline(chance_level, color="black", linestyle="--", lw=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) # Adjust layout to accommodate title
    safe_fname_tgm = "".join(
        c if c.isalnum() else "_" for c in group_identifier_for_plot.replace(" ", "_")
    )
    save_path_tgm = os.path.join(
        output_directory_path, f"group_TGM_stats_{safe_fname_tgm}.png"
    )
    try:
        fig.savefig(save_path_tgm, dpi=200)
        logger_run_decoding.info(f"Group TGM plot saved: {save_path_tgm}")
    except Exception as e_tgm_save:
        logger_run_decoding.error(
            f"Failed to save group TGM plot '{save_path_tgm}': {e_tgm_save}", exc_info=True
        )
    plt.close(fig)


# --- Intra-subject decoding function ---
def execute_single_subject_decoding(
    subject_identifier,
    group_affiliation,
    decoding_protocol_identifier="Main_and_Specific_PP_AP_Comparisons",
    save_results_flag=True,
    enable_verbose_logging=True,
    generate_plots_flag=True,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_processing=-1, # Default to auto
    compute_intra_subject_stats_flag=True,
    n_perms_for_intra_subject_clusters=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type=CLASSIFIER_MODEL_TYPE, # This will be the source for the title
    compute_tgm_flag=True,
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
    cluster_threshold_config_intra_fold=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
):
    total_start_time = time.time()
    subject_results = {
        "subject_id": subject_identifier, "group": group_affiliation,
        "main_epochs_time_points": None, "main_original_labels_array": None,
        "main_predicted_probabilities_global": None, "main_predicted_labels_global": None,
        "main_cross_validation_global_scores": None,
        "main_decoding_scores_1d_all_folds": None,
        "main_decoding_scores_1d_mean": None,
        "main_temporal_1d_fdr_sig_data": None,
        "main_temporal_1d_cluster_sig_data": None,
        "main_decoding_tgm_all_folds": None,
        "main_decoding_scores_tgm_mean": None,
        "main_tgm_fdr_sig_data": None,
        "main_decoding_mean_auc_global": np.nan,
        "main_decoding_global_metrics": {},
        "specific_ap_decoding_results": [],
        "mean_of_specific_scores_1d": None,
        "sem_of_specific_scores_1d": None,
        "mean_specific_fdr_sig_data": None,
        "mean_specific_cluster_sig_data": None,
    }

    try:
        if not base_input_data_path or not base_output_results_path:
            current_user = getuser()
            cfg_input_path, cfg_output_path = configure_project_paths(current_user)
            base_input_data_path = base_input_data_path or cfg_input_path
            base_output_results_path = base_output_results_path or cfg_output_path

    
        actual_classifier_name_for_plot_titles = classifier_type
    

        logger_run_decoding.info(
            f"Starting decoding for subject: {subject_identifier} (Group: {group_affiliation}, "
            f"Classifier: {actual_classifier_name_for_plot_titles}, Main TGM: {compute_tgm_flag}, "
            f"IntraStats (on CV folds): {'Enabled (FDR & Cluster)' if compute_intra_subject_stats_flag else 'Disabled'})"
        )
        epochs_object, returned_data_dict = load_epochs_data_for_decoding(
            subject_identifier, group_affiliation, base_input_data_path,
            loading_conditions_config, enable_verbose_logging,
        )
        if epochs_object is None:
            logger_run_decoding.error(f"Epochs object failed to load for {subject_identifier}. Aborting subject.")
            return subject_results
        subject_results["main_epochs_time_points"] = epochs_object.times.copy()

        logger_run_decoding.info(f"--- 1. Main Decoding (e.g., PP/all vs AP/all) for {subject_identifier} ---")
        XPP_main = returned_data_dict.get("XPP_ALL")
        XAP_main = returned_data_dict.get("XAP_ALL")

        if XPP_main is not None and XAP_main is not None and XPP_main.size > 0 and XAP_main.size > 0:
            main_data = np.concatenate([XAP_main, XPP_main], axis=0)
            main_labels_orig = np.concatenate([np.zeros(XAP_main.shape[0]), np.ones(XPP_main.shape[0])])
            subject_results["main_original_labels_array"] = main_labels_orig
            main_labels_enc_temp = LabelEncoder().fit_transform(main_labels_orig)

            if len(np.unique(main_labels_enc_temp)) < 2:
                logger_run_decoding.error("Only one class present for main decoding. Skipping this analysis.")
            else:
                clf_main_pipeline = _build_standard_classifier_pipeline(
                    classifier_model_type=classifier_type, random_seed_state=42
                )
                min_samp_main = np.min(np.bincount(main_labels_enc_temp))
                n_cv_main = min(5, min_samp_main) if min_samp_main >= 2 else 0

                if n_cv_main < 2:
                    logger_run_decoding.error(f"Not enough samples for main CV ({min_samp_main} in smallest class). Skipping main decoding.")
                else:
                    cv_main = StratifiedKFold(n_splits=n_cv_main, shuffle=True, random_state=42)
                    main_out = run_temporal_decoding_analysis(
                        main_data, main_labels_orig, classifier_pipeline=clf_main_pipeline, 
                        cross_validation_splitter=cv_main,
                        n_jobs_external=n_jobs_for_processing,
                        compute_intra_fold_stats=compute_intra_subject_stats_flag,
                        n_permutations_for_intra_fold_clusters=n_perms_for_intra_subject_clusters,
                        compute_temporal_generalization_matrix=compute_tgm_flag,
                        chance_level=CHANCE_LEVEL_AUC_SCORE,
                        cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold,
                    )
                    subject_results.update({
                        "main_predicted_probabilities_global": main_out[0],
                        "main_predicted_labels_global": main_out[1],
                        "main_cross_validation_global_scores": main_out[2],
                        "main_decoding_scores_1d_mean": main_out[3],
                        "main_decoding_global_metrics": main_out[4],
                        "main_temporal_1d_fdr_sig_data": main_out[5],
                        "main_temporal_1d_cluster_sig_data": main_out[6],
                        "main_decoding_scores_1d_all_folds": main_out[7],
                        "main_decoding_scores_tgm_mean": main_out[8],
                        "main_tgm_fdr_sig_data": main_out[9],
                        "main_decoding_tgm_all_folds": main_out[11],
                        "main_decoding_mean_auc_global": np.nanmean(main_out[2]) if main_out[2] is not None and main_out[2].size > 0 else np.nan
                    })
                    logger_run_decoding.info(f"Main Decoding DONE for {subject_identifier}. Mean Global AUC: {subject_results['main_decoding_mean_auc_global']:.3f}")
        else:
            logger_run_decoding.warning(f"Missing XPP_ALL or XAP_ALL data for main decoding for {subject_identifier}. Skipping.")

        if generate_plots_flag or compute_intra_subject_stats_flag:
            logger_run_decoding.info(f"--- 2. Specific PP vs AP_family Tasks for {subject_identifier} ---")
            PP_spec_data = returned_data_dict.get("PP_FOR_SPECIFIC_COMPARISON")
            if PP_spec_data is None or PP_spec_data.size == 0:
                logger_run_decoding.warning("PP_FOR_SPECIFIC_COMPARISON data missing. Skipping all specific tasks.")
            else:
          
                clf_spec_pipeline = _build_standard_classifier_pipeline(
                    classifier_model_type=classifier_type, random_seed_state=42
                )
                for ap_fam_key, _ in AP_FAMILIES_FOR_SPECIFIC_COMPARISON.items():
                    AP_fam_data = returned_data_dict.get(ap_fam_key)
                    comp_name = f"PP_spec vs {ap_fam_key.replace('_', ' ').replace('AP FAMILY', 'AP Fam.')}"
                    task_res = {
                        "comparison_name": comp_name, "scores_1d_mean": None,
                        "all_folds_scores_1d": None, "times": subject_results["main_epochs_time_points"],
                        "fdr_significance_data": None, "cluster_significance_data": None,
                    }
                    if AP_fam_data is None or AP_fam_data.size == 0:
                        logger_run_decoding.warning(f"Data for '{ap_fam_key}' missing. Skipping specific comparison: {comp_name}.")
                    else:
                        task_d = np.concatenate([PP_spec_data, AP_fam_data], axis=0)
                        task_l_orig = np.concatenate([np.zeros(PP_spec_data.shape[0]), np.ones(AP_fam_data.shape[0])])
                        task_l_enc_temp = LabelEncoder().fit_transform(task_l_orig)
                        if len(np.unique(task_l_enc_temp)) < 2:
                            logger_run_decoding.warning(f"Only one class present for specific comparison {comp_name}. Skipping.")
                        else:
                            min_s_task = np.min(np.bincount(task_l_enc_temp))
                            n_cv_task = min(5, min_s_task) if min_s_task >= 2 else 0
                            if n_cv_task < 2:
                                logger_run_decoding.warning(f"Not enough samples for CV in {comp_name} ({min_s_task} in smallest class). Skipping.")
                            else:
                                cv_task = StratifiedKFold(n_splits=n_cv_task, shuffle=True, random_state=42)
                                task_out_spec = run_temporal_decoding_analysis(
                                    task_d, task_l_orig, classifier_pipeline=clone(clf_spec_pipeline), 
                                    cross_validation_splitter=cv_task,
                                    n_jobs_external=n_jobs_for_processing,
                                    compute_intra_fold_stats=compute_intra_subject_stats_flag,
                                    n_permutations_for_intra_fold_clusters=n_perms_for_intra_subject_clusters,
                                    compute_temporal_generalization_matrix=False,
                                    chance_level=CHANCE_LEVEL_AUC_SCORE,
                                    cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold,
                                )
                                task_res["scores_1d_mean"] = task_out_spec[3]
                                task_res["fdr_significance_data"] = task_out_spec[5]
                                task_res["cluster_significance_data"] = task_out_spec[6]
                                task_res["all_folds_scores_1d"] = task_out_spec[7]
                                peak_auc_spec = np.nanmax(task_out_spec[3]) if task_out_spec[3] is not None and task_out_spec[3].size > 0 else np.nan
                                logger_run_decoding.info(f"  Specific task: {comp_name}. Peak Mean AUC: {peak_auc_spec:.3f}")
                    subject_results["specific_ap_decoding_results"].append(task_res)
            logger_run_decoding.info(f"--- Specific PP vs AP_family Tasks DONE for {subject_identifier} ---")

        if compute_intra_subject_stats_flag and subject_results["specific_ap_decoding_results"]:
            valid_spec_mean_scores_for_stack = [
                r["scores_1d_mean"] for r in subject_results["specific_ap_decoding_results"]
                if r["scores_1d_mean"] is not None and subject_results["main_epochs_time_points"] is not None and
                   r["scores_1d_mean"].shape == subject_results["main_epochs_time_points"].shape
            ]
            if len(valid_spec_mean_scores_for_stack) >= 2:
                logger_run_decoding.info(f"--- 3. Stats on stack of {len(valid_spec_mean_scores_for_stack)} Specific Task Mean Curves for {subject_identifier} ---")
                stacked_specific_mean_curves = np.array(valid_spec_mean_scores_for_stack)
                subject_results["mean_of_specific_scores_1d"] = np.nanmean(stacked_specific_mean_curves, axis=0)
                if stacked_specific_mean_curves.shape[0] > 1:
                    subject_results["sem_of_specific_scores_1d"] = scipy_stats.sem(stacked_specific_mean_curves, axis=0, nan_policy='omit')
                _, fdr_mask_mean_spec, fdr_pval_mean_spec = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_specific_mean_curves, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                )
                subject_results["mean_specific_fdr_sig_data"] = {
                    "mask": fdr_mask_mean_spec, "p_values": fdr_pval_mean_spec,
                    "method": f"FDR on stack of {len(valid_spec_mean_scores_for_stack)} specific task mean curves"
                }
                _, clu_obj_mean_spec, p_clu_mean_spec, _ = bEEG_stats.perform_cluster_permutation_test(
                    stacked_specific_mean_curves, CHANCE_LEVEL_AUC_SCORE, n_perms_for_intra_subject_clusters,
                    alternative_hypothesis="greater", cluster_threshold_config=cluster_threshold_config_intra_fold
                )
                combined_cluster_mask_mean_spec = np.zeros_like(subject_results["mean_of_specific_scores_1d"], dtype=bool)
                sig_cluster_objects_mean_spec = []
                if clu_obj_mean_spec and p_clu_mean_spec is not None:
                    for i, cluster_mask_from_stack in enumerate(clu_obj_mean_spec):
                        if p_clu_mean_spec[i] < 0.05:
                            sig_cluster_objects_mean_spec.append(cluster_mask_from_stack)
                            squeezed_mask = cluster_mask_from_stack.squeeze()
                            if squeezed_mask.ndim == 1 and squeezed_mask.shape == combined_cluster_mask_mean_spec.shape:
                                combined_cluster_mask_mean_spec = np.logical_or(combined_cluster_mask_mean_spec, squeezed_mask)
                subject_results["mean_specific_cluster_sig_data"] = {
                    "mask": combined_cluster_mask_mean_spec, "cluster_objects": sig_cluster_objects_mean_spec,
                    "p_values_all_clusters": p_clu_mean_spec,
                    "method": f"ClusterPerm on stack of {len(valid_spec_mean_scores_for_stack)} specific task mean curves"
                }
                logger_run_decoding.info(f"--- Stats on stack of Specific Task Mean Curves for {subject_identifier} DONE ---")
            else:
                logger_run_decoding.warning(f"Not enough valid specific task mean curves ({len(valid_spec_mean_scores_for_stack)}) for stack averaging/stats for {subject_identifier}.")

        subject_results_dir = None
        if save_results_flag or generate_plots_flag:
            subject_results_dir = setup_analysis_results_directory(
                base_output_results_path, "intra_subject_results", group_affiliation,
                f"{subject_identifier}_{decoding_protocol_identifier}",
            )
        if save_results_flag and subject_results_dir:
            np.savez_compressed(
                os.path.join(subject_results_dir, f"decoding_results_full_{subject_identifier}_{group_affiliation}.npz"),
                **subject_results
            )
            csv_summary = {
                "subject_id": subject_identifier, "group": group_affiliation,
                "protocol": decoding_protocol_identifier,
                "mean_global_auc_main": subject_results["main_decoding_mean_auc_global"],
                "std_global_auc_main": np.nanstd(subject_results["main_cross_validation_global_scores"]) if subject_results["main_cross_validation_global_scores"] is not None and subject_results["main_cross_validation_global_scores"].size > 0 else np.nan
            }
            if subject_results["main_decoding_global_metrics"]:
                csv_summary.update({f"main_{k}": v for k,v in subject_results["main_decoding_global_metrics"].items()})
            pd.DataFrame([csv_summary]).to_csv(
                os.path.join(subject_results_dir, f"summary_metrics_{subject_identifier}_{group_affiliation}.csv"), index=False
            )

        if generate_plots_flag and epochs_object is not None and subject_results_dir:
            create_subject_decoding_dashboard_plots(
                main_epochs_time_points=subject_results["main_epochs_time_points"],
                main_original_labels_array=subject_results["main_original_labels_array"],
                main_predicted_probabilities_global=subject_results["main_predicted_probabilities_global"],
                main_predicted_labels_global=subject_results["main_predicted_labels_global"],
                main_cross_validation_global_scores=subject_results["main_cross_validation_global_scores"],
                main_temporal_scores_1d_all_folds=subject_results["main_decoding_scores_1d_all_folds"],
                main_mean_temporal_decoding_scores_1d=subject_results["main_decoding_scores_1d_mean"],
                main_temporal_1d_fdr_sig_data=subject_results["main_temporal_1d_fdr_sig_data"],
                main_temporal_1d_cluster_sig_data=subject_results["main_temporal_1d_cluster_sig_data"],
                main_mean_temporal_generalization_matrix_scores=subject_results["main_decoding_scores_tgm_mean"],
                main_tgm_fdr_sig_data=subject_results["main_tgm_fdr_sig_data"],
                classifier_name_for_title=actual_classifier_name_for_plot_titles, 
                subject_identifier=subject_identifier, group_identifier=group_affiliation,
                output_directory_path=subject_results_dir,
                specific_ap_decoding_results=subject_results["specific_ap_decoding_results"],
                mean_of_specific_scores_1d=subject_results["mean_of_specific_scores_1d"],
                sem_of_specific_scores_1d=subject_results["sem_of_specific_scores_1d"],
                mean_specific_fdr_sig_data=subject_results["mean_specific_fdr_sig_data"],
                mean_specific_cluster_sig_data=subject_results["mean_specific_cluster_sig_data"],
                chance_level_auc=CHANCE_LEVEL_AUC_SCORE
            )
        logger_run_decoding.info(f"Finished subject {subject_identifier}. Total time: {time.time() - total_start_time:.2f}s")
        return subject_results
    except FileNotFoundError as fnfe:
        logger_run_decoding.error(f"FileNotFoundError for subject {subject_identifier}: {fnfe}. Aborting subject processing.")
    except ValueError as ve:
        logger_run_decoding.error(f"ValueError for subject {subject_identifier}: {ve}. Aborting subject processing.", exc_info=True)
    except KeyError as ke:
        logger_run_decoding.error(f"KeyError for subject {subject_identifier}: {ke}. Aborting subject processing.", exc_info=True)
    except Exception as e:
        logger_run_decoding.error(f"Unexpected error processing subject {subject_identifier}: {e}", exc_info=True)
    return subject_results

# --- Group Intra-Subject Analysis ---
def execute_group_intra_subject_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Main_and_Specific_PP_AP_Comparisons",
    save_results_flag=True,
    enable_verbose_logging=True,
    generate_plots_flag=True,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=4, # n_jobs for each call to execute_single_subject_decoding
    compute_group_level_stats_flag=True, # For stats on aggregated group data
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    compute_intra_subject_stats_for_group_runs_flag=True, # For stats on each subject's CV folds
    n_perms_for_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    n_jobs_for_group_cluster_stats=6, # n_jobs for group-level MNE cluster stats
    # n_jobs_for_intra_fold_cluster_stats was mentioned but not directly used at this level
):
    """
    Executes intra-subject decoding for all subjects in a group and then performs group-level statistics.
    """
    total_group_analysis_start_time = time.time()
    logger_run_decoding.info(
        f"Starting intra-subject decoding analysis for GROUP: {group_identifier}. "
        f"n_jobs_for_each_subject (MNE decoding per subject): {n_jobs_for_each_subject}, "
        f"n_jobs_for_group_cluster_stats (MNE group cluster stats): {n_jobs_for_group_cluster_stats}."
    )

    group_results_collection = {
        "subject_global_auc_scores": {}, "subject_global_metrics_maps": {},
        "subject_temporal_scores_1d_mean_list": [], # List of mean 1D temporal scores (one per subject)
        "subject_epochs_time_points_list": [], # Should be consistent across subjects
        "subject_tgm_scores_mean_list": [], # List of mean TGM scores (one per subject)
        "subject_mean_of_specific_scores_list": [], # List of 'mean of specific scores' (one per subject)
        "processed_subject_ids": [], # Track successfully processed subjects
    }

    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_decoding.info(
            f"\n--- Group '{group_identifier}': Processing Subject {i}/{len(subject_ids_in_group)}: {subject_id_current} ---"
        )
        subject_output_dict = execute_single_subject_decoding(
            subject_identifier=subject_id_current,
            group_affiliation=group_identifier, # Use group_identifier as affiliation for this subject
            decoding_protocol_identifier=decoding_protocol_identifier,
            save_results_flag=save_results_flag,
            enable_verbose_logging=enable_verbose_logging, # Can be set to False for less output during group runs
            generate_plots_flag=generate_plots_flag, # Generate individual subject dashboards
            base_input_data_path=base_input_data_path,
            base_output_results_path=base_output_results_path,
            n_jobs_for_processing=n_jobs_for_each_subject,
            compute_intra_subject_stats_flag=compute_intra_subject_stats_for_group_runs_flag,
            n_perms_for_intra_subject_clusters=n_perms_intra_subject_folds_for_group_runs,
            classifier_type=classifier_type_for_group_runs,
            compute_tgm_flag=compute_tgm_for_group_subjects_flag,
            loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
            cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold_group,
        )
        s_auc = subject_output_dict.get("main_decoding_mean_auc_global", np.nan)
        s_metrics = subject_output_dict.get("main_decoding_global_metrics", {})
        s_scores_t_1d_mean = subject_output_dict.get("main_decoding_scores_1d_mean")
        s_times_t = subject_output_dict.get("main_epochs_time_points")
        s_scores_tgm_mean = subject_output_dict.get("main_decoding_scores_tgm_mean")
        s_mean_specific_scores = subject_output_dict.get("mean_of_specific_scores_1d")

        group_results_collection["subject_global_auc_scores"][subject_id_current] = s_auc
        group_results_collection["subject_global_metrics_maps"][subject_id_current] = s_metrics

        if pd.notna(s_auc) and s_scores_t_1d_mean is not None and s_times_t is not None and \
           s_scores_t_1d_mean.size > 0 and s_times_t.size > 0:
            group_results_collection["subject_temporal_scores_1d_mean_list"].append(s_scores_t_1d_mean)
            group_results_collection["subject_epochs_time_points_list"].append(s_times_t)
            group_results_collection["processed_subject_ids"].append(subject_id_current)

            if compute_tgm_for_group_subjects_flag and s_scores_tgm_mean is not None and not np.all(np.isnan(s_scores_tgm_mean)):
                group_results_collection["subject_tgm_scores_mean_list"].append(s_scores_tgm_mean)
            elif compute_tgm_for_group_subjects_flag: # TGM expected but not valid, add placeholder
                # Create a NaN array of expected TGM shape if possible, else just NaN
                nan_tgm_placeholder = np.full_like(s_scores_t_1d_mean[:, np.newaxis] * s_scores_t_1d_mean[np.newaxis, :], np.nan) \
                                      if s_scores_t_1d_mean is not None and s_scores_t_1d_mean.ndim == 1 and s_scores_t_1d_mean.size > 0 \
                                      else np.nan
                group_results_collection["subject_tgm_scores_mean_list"].append(nan_tgm_placeholder)


            if s_mean_specific_scores is not None and not np.all(np.isnan(s_mean_specific_scores)):
                group_results_collection["subject_mean_of_specific_scores_list"].append(s_mean_specific_scores)
            else: # Placeholder if mean specific scores not valid
                group_results_collection["subject_mean_of_specific_scores_list"].append(np.nan)
        else:
            logger_run_decoding.warning(
                f"Skipping subject {subject_id_current} from group aggregation due to errors or no valid main scores (Global AUC: {s_auc})."
            )

    group_summary_dir = None
    if save_results_flag or generate_plots_flag:
        group_summary_dir = setup_analysis_results_directory(
            base_output_results_path, "group_summary_intra_subject", group_identifier
        )

    valid_subject_global_scores = np.array(
        [s for s in group_results_collection["subject_global_auc_scores"].values() if pd.notna(s)]
    )

    # --- Group Stats: Global AUC (mean of subject's mean global AUCs for main decoding) ---
    if len(valid_subject_global_scores) > 0:
        mean_group_global_auc = np.mean(valid_subject_global_scores)
        std_group_global_auc = np.std(valid_subject_global_scores)
        logger_run_decoding.info(
            f"Group {group_identifier} - Overall Global Mean AUC (Main Dec.): {mean_group_global_auc:.3f} "
            f"± {std_group_global_auc:.3f} (N={len(valid_subject_global_scores)} subjects)"
        )
        if compute_group_level_stats_flag and len(valid_subject_global_scores) >= 2:
            stat_global, p_val_global = bEEG_stats.compare_global_scores_to_chance(
                valid_subject_global_scores, CHANCE_LEVEL_AUC_SCORE, "ttest", "greater"
            )
            logger_run_decoding.info(
                f"  Global AUC (Main Dec.) vs Chance: t={stat_global:.3f}, p={p_val_global:.4f}"
            )
            if save_results_flag and group_summary_dir:
                with open(os.path.join(group_summary_dir, f"stats_global_auc_{group_identifier}.txt"), "w") as f_stat:
                    f_stat.write(
                        f"Intra-Subject Global AUC (Main Dec.) vs Chance ({CHANCE_LEVEL_AUC_SCORE})\n"
                        f"Group: {group_identifier}, N subjects: {len(valid_subject_global_scores)}\n"
                        f"Mean AUC: {mean_group_global_auc:.4f}, Std Dev: {std_group_global_auc:.4f}\n"
                        f"T-statistic: {stat_global:.4f}, P-value: {p_val_global:.4f} (one-sided 'greater')\n"
                    )
        if generate_plots_flag and group_summary_dir:
            plot_group_mean_scores_barplot(
                group_results_collection["subject_global_auc_scores"],
                f"{group_identifier} - Subject Global AUCs (Main Dec.)",
                group_summary_dir,
                "Global ROC AUC",
                CHANCE_LEVEL_AUC_SCORE,
            )

    # --- Group Stats: 1D Temporal (Main Decoding: PP/all vs AP/all) ---
    # Stats on the stack of subject's mean 1D temporal curves.
    if len(group_results_collection["subject_temporal_scores_1d_mean_list"]) >= 2 and \
       compute_group_level_stats_flag:
        # Find a reference time array (assuming all subjects have same time points for main decoding)
        ref_times_idx_main = next(
            (j for j, t_arr in enumerate(group_results_collection["subject_epochs_time_points_list"])
             if t_arr is not None and t_arr.size > 0), -1
        )
        if ref_times_idx_main != -1:
            ref_times_1d_main = group_results_collection["subject_epochs_time_points_list"][ref_times_idx_main]
            valid_1d_scores_main_group = []
            # Align scores based on consistent time arrays and ensure data is valid
            for i_subj, _ in enumerate(group_results_collection["processed_subject_ids"]):
                 if i_subj < len(group_results_collection["subject_temporal_scores_1d_mean_list"]) and \
                    i_subj < len(group_results_collection["subject_epochs_time_points_list"]):
                    s_scores = group_results_collection["subject_temporal_scores_1d_mean_list"][i_subj]
                    s_times = group_results_collection["subject_epochs_time_points_list"][i_subj]

                    if s_times is not None and s_times.size == ref_times_1d_main.size and \
                       np.allclose(s_times, ref_times_1d_main) and \
                       s_scores is not None and not np.all(np.isnan(s_scores)):
                        valid_1d_scores_main_group.append(s_scores)

            if len(valid_1d_scores_main_group) >= 2:
                stacked_1d_scores_main_group = np.array(valid_1d_scores_main_group) # (n_subjects, n_timepoints)
                logger_run_decoding.info(
                    f"Group {group_identifier} (N={stacked_1d_scores_main_group.shape[0]} for Main 1D temporal stats): Running group stats..."
                )
                # FDR
                t_obs_fdr_main, fdr_mask_1d_main_grp, pvals_fdr_1d_main_grp = \
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        stacked_1d_scores_main_group, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                    )
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(
                        os.path.join(group_summary_dir, f"stats_group_temporal_1D_FDR_Main_{group_identifier}.npz"),
                        observed_t_values=t_obs_fdr_main, significance_mask=fdr_mask_1d_main_grp,
                        fdr_corrected_p_values=pvals_fdr_1d_main_grp, time_points=ref_times_1d_main
                    )
                # Cluster Permutation
                t_thresh_1d_clu_main_grp = group_cluster_test_t_thresh_value
                if group_cluster_test_threshold_method == "stat" and t_thresh_1d_clu_main_grp is None:
                    t_thresh_1d_clu_main_grp = scipy_stats.t.ppf(1.0 - 0.05 / 2, df=stacked_1d_scores_main_group.shape[0] - 1)

                clu_thresh_cfg_main_grp = t_thresh_1d_clu_main_grp if group_cluster_test_threshold_method == "stat" else \
                                          INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method == "tfce" else None

                t_obs_clu_main_grp, clu_1d_main_grp, p_clu_1d_main_grp, _ = \
                    bEEG_stats.perform_cluster_permutation_test(
                        stacked_1d_scores_main_group, CHANCE_LEVEL_AUC_SCORE, n_perms_for_group_cluster_test,
                        cluster_threshold_config=clu_thresh_cfg_main_grp,
                        alternative_hypothesis="greater", n_jobs=n_jobs_for_group_cluster_stats
                    )
                clu_map_1d_main_grp = bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d_main.shape, clu_1d_main_grp, p_clu_1d_main_grp
                ) if clu_1d_main_grp and p_clu_1d_main_grp is not None else None

                if save_results_flag and group_summary_dir:
                     np.savez_compressed(
                        os.path.join(group_summary_dir, f"stats_group_temporal_1D_CLUSTER_Main_{group_identifier}.npz"),
                        observed_t_values=t_obs_clu_main_grp, clusters=clu_1d_main_grp,
                        cluster_p_values=p_clu_1d_main_grp, cluster_p_map=clu_map_1d_main_grp,
                        time_points=ref_times_1d_main
                    )
                if generate_plots_flag and group_summary_dir:
                    plot_group_temporal_decoding_statistics(
                        ref_times_1d_main, np.mean(stacked_1d_scores_main_group, axis=0),
                        f"{group_identifier} (Main 1D Temporal)", group_summary_dir,
                        scipy_stats.sem(stacked_1d_scores_main_group, axis=0, nan_policy="omit") if stacked_1d_scores_main_group.shape[0] > 1 else None,
                        clu_map_1d_main_grp, fdr_mask_1d_main_grp, CHANCE_LEVEL_AUC_SCORE
                    )
            else:
                logger_run_decoding.warning(f"Not enough valid subject data ({len(valid_1d_scores_main_group)}) for Main 1D temporal group stats for {group_identifier}.")
        else:
            logger_run_decoding.warning(f"Could not establish reference time points for Main 1D group stats for {group_identifier}.")


    # --- Group Stats: TGM (Main decoding: PP/all vs AP/all) ---
    # Stats on the stack of subject's mean TGM scores. Only FDR for TGM.
    valid_tgms_for_group_stat = [
        tgm for tgm in group_results_collection["subject_tgm_scores_mean_list"]
        if tgm is not None and not (isinstance(tgm, float) and np.isnan(tgm)) and tgm.ndim == 2
    ]

    if len(valid_tgms_for_group_stat) >= 2 and compute_group_level_stats_flag and compute_tgm_for_group_subjects_flag:
        ref_times_tgm_idx_main = next(
            (j for j, t_arr in enumerate(group_results_collection["subject_epochs_time_points_list"])
             if t_arr is not None and t_arr.size > 0), -1
        )
        if ref_times_tgm_idx_main != -1:
            ref_times_tgm_main = group_results_collection["subject_epochs_time_points_list"][ref_times_tgm_idx_main]
            expected_tgm_shape = (ref_times_tgm_main.size, ref_times_tgm_main.size)
            valid_tgms_to_stack_main_grp = []

            for i_subj, _ in enumerate(group_results_collection["processed_subject_ids"]):
                if i_subj < len(group_results_collection["subject_tgm_scores_mean_list"]) and \
                   i_subj < len(group_results_collection["subject_epochs_time_points_list"]):
                    tgm_s = group_results_collection["subject_tgm_scores_mean_list"][i_subj]
                    times_s = group_results_collection["subject_epochs_time_points_list"][i_subj]
                    if tgm_s is not None and not (isinstance(tgm_s, float) and np.isnan(tgm_s)) and \
                       tgm_s.shape == expected_tgm_shape and not np.all(np.isnan(tgm_s)) and \
                       times_s is not None and times_s.size == ref_times_tgm_main.size and np.allclose(times_s, ref_times_tgm_main):
                        valid_tgms_to_stack_main_grp.append(tgm_s)

            if len(valid_tgms_to_stack_main_grp) >= 2:
                stacked_tgm_scores_main_grp = np.array(valid_tgms_to_stack_main_grp) # (n_subjects, n_times, n_times)
                logger_run_decoding.info(
                    f"Group {group_identifier} (N={stacked_tgm_scores_main_grp.shape[0]} for Main TGM stats): Running group TGM FDR stats..."
                )
                n_s_tgm, n_t_tgm, _ = stacked_tgm_scores_main_grp.shape
                tgm_flat_fdr_main_grp = stacked_tgm_scores_main_grp.reshape(n_s_tgm, n_t_tgm * n_t_tgm)

                t_obs_tgm_fdr_main_grp, fdr_mask_tgm_flat_main_grp, pvals_fdr_tgm_flat_main_grp = \
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        tgm_flat_fdr_main_grp, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                    )
                fdr_mask_tgm_main_grp = fdr_mask_tgm_flat_main_grp.reshape(n_t_tgm, n_t_tgm)
                t_obs_tgm_fdr_map_main_grp = t_obs_tgm_fdr_main_grp.reshape(n_t_tgm, n_t_tgm) if hasattr(t_obs_tgm_fdr_main_grp, 'reshape') else t_obs_tgm_fdr_main_grp

                if save_results_flag and group_summary_dir:
                    np.savez_compressed(
                        os.path.join(group_summary_dir, f"stats_group_TGM_FDR_Main_{group_identifier}.npz"),
                        observed_t_values_map=t_obs_tgm_fdr_map_main_grp, significance_mask=fdr_mask_tgm_main_grp,
                        fdr_corrected_p_values_map=pvals_fdr_tgm_flat_main_grp.reshape(n_t_tgm, n_t_tgm),
                        time_points_array=ref_times_tgm_main,
                        mean_group_tgm_scores=np.mean(stacked_tgm_scores_main_grp, axis=0)
                    )
                # NO Cluster Permutation for TGM at group level
                logger_run_decoding.info(f"Group {group_identifier}: Cluster permutation for TGM SKIPPED.")

                if generate_plots_flag and group_summary_dir:
                    plot_group_tgm_statistics(
                        np.mean(stacked_tgm_scores_main_grp, axis=0), ref_times_tgm_main,
                        significant_cluster_masks_tgm=None, # No cluster masks for TGM
                        cluster_p_values_tgm=None,          # No cluster p-values for TGM
                        group_identifier_for_plot=f"{group_identifier} (Main TGM - FDR only)",
                        output_directory_path=group_summary_dir,
                        observed_t_values_map_tgm=t_obs_tgm_fdr_map_main_grp, # Plot t-values from FDR
                        fdr_significance_mask_tgm=fdr_mask_tgm_main_grp,
                        chance_level=CHANCE_LEVEL_AUC_SCORE
                    )
            else:
                logger_run_decoding.warning(f"Not enough valid TGM data ({len(valid_tgms_to_stack_main_grp)}) for Main TGM group stats for {group_identifier}.")
        else:
            logger_run_decoding.warning(f"Could not establish reference time points for Main TGM group stats for {group_identifier}.")
    elif compute_group_level_stats_flag:
        logger_run_decoding.info(f"Skipping Main TGM group stats for {group_identifier} (TGM computation disabled or not enough valid data).")


    # --- Group Stats: 1D Temporal (Average of Specific curves) ---
    # Stats on the stack of subject's "mean_of_specific_scores_1d".
    valid_mean_specific_for_group_stat = [
        ms for ms in group_results_collection["subject_mean_of_specific_scores_list"]
        if ms is not None and not (isinstance(ms, float) and np.isnan(ms)) and ms.ndim == 1
    ]

    if len(valid_mean_specific_for_group_stat) >= 2 and compute_group_level_stats_flag:
        # Assuming ref_times_1d_main (from main decoding) is applicable here
        if ref_times_1d_main is not None:
            valid_mean_specific_scores_grp = []
            # Align and validate
            for i_subj, _ in enumerate(group_results_collection["processed_subject_ids"]):
                if i_subj < len(group_results_collection["subject_mean_of_specific_scores_list"]) and \
                   i_subj < len(group_results_collection["subject_epochs_time_points_list"]): # Ensure time points list is also indexed
                    s_scores_ms = group_results_collection["subject_mean_of_specific_scores_list"][i_subj]
                    # Use corresponding time array for this subject's specific scores average
                    s_times_ms = group_results_collection["subject_epochs_time_points_list"][i_subj]
                    if s_scores_ms is not None and not (isinstance(s_scores_ms, float) and np.isnan(s_scores_ms)) and \
                       s_times_ms is not None and s_times_ms.size == ref_times_1d_main.size and \
                       np.allclose(s_times_ms, ref_times_1d_main) and not np.all(np.isnan(s_scores_ms)):
                        valid_mean_specific_scores_grp.append(s_scores_ms)

            if len(valid_mean_specific_scores_grp) >= 2:
                stacked_mean_specific_grp = np.array(valid_mean_specific_scores_grp) # (n_subjects, n_timepoints)
                logger_run_decoding.info(
                    f"Group {group_identifier} (N={stacked_mean_specific_grp.shape[0]} for Avg. Specific 1D stats): Running group stats..."
                )
                # FDR
                t_obs_fdr_ms_grp, fdr_mask_1d_ms_grp, pvals_fdr_1d_ms_grp = \
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        stacked_mean_specific_grp, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                    )
                if save_results_flag and group_summary_dir:
                     np.savez_compressed(
                        os.path.join(group_summary_dir, f"stats_group_temporal_1D_FDR_MeanSpecific_{group_identifier}.npz"),
                        observed_t_values=t_obs_fdr_ms_grp, significance_mask=fdr_mask_1d_ms_grp,
                        fdr_corrected_p_values=pvals_fdr_1d_ms_grp, time_points=ref_times_1d_main
                    )
                # Cluster Permutation
                t_thresh_1d_clu_ms_grp = group_cluster_test_t_thresh_value
                if group_cluster_test_threshold_method == "stat" and t_thresh_1d_clu_ms_grp is None:
                    t_thresh_1d_clu_ms_grp = scipy_stats.t.ppf(1.0 - 0.05 / 2, df=stacked_mean_specific_grp.shape[0] - 1)

                clu_thresh_cfg_ms_grp = t_thresh_1d_clu_ms_grp if group_cluster_test_threshold_method == "stat" else \
                                        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method == "tfce" else None

                t_obs_clu_ms_grp, clu_1d_ms_grp, p_clu_1d_ms_grp, _ = \
                    bEEG_stats.perform_cluster_permutation_test(
                        stacked_mean_specific_grp, CHANCE_LEVEL_AUC_SCORE, n_perms_for_group_cluster_test,
                        cluster_threshold_config=clu_thresh_cfg_ms_grp,
                        alternative_hypothesis="greater", n_jobs=n_jobs_for_group_cluster_stats
                    )
                clu_map_1d_ms_grp = bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d_main.shape, clu_1d_ms_grp, p_clu_1d_ms_grp
                ) if clu_1d_ms_grp and p_clu_1d_ms_grp is not None else None

                if save_results_flag and group_summary_dir:
                    np.savez_compressed(
                        os.path.join(group_summary_dir, f"stats_group_temporal_1D_CLUSTER_MeanSpecific_{group_identifier}.npz"),
                        observed_t_values=t_obs_clu_ms_grp, clusters=clu_1d_ms_grp,
                        cluster_p_values=p_clu_1d_ms_grp, cluster_p_map=clu_map_1d_ms_grp,
                        time_points=ref_times_1d_main
                    )
                if generate_plots_flag and group_summary_dir:
                    plot_group_temporal_decoding_statistics(
                        ref_times_1d_main, np.mean(stacked_mean_specific_grp, axis=0),
                        f"{group_identifier} (Avg. of Specific Tasks - Group)", group_summary_dir,
                        scipy_stats.sem(stacked_mean_specific_grp, axis=0, nan_policy="omit") if stacked_mean_specific_grp.shape[0] > 1 else None,
                        clu_map_1d_ms_grp, fdr_mask_1d_ms_grp, CHANCE_LEVEL_AUC_SCORE
                    )
            else:
                logger_run_decoding.warning(f"Not enough valid subject data ({len(valid_mean_specific_scores_grp)}) for Avg. Specific 1D group stats for {group_identifier}.")
        else:
            logger_run_decoding.warning(f"Could not establish reference time points for Avg. Specific 1D group stats for {group_identifier} (ref_times_1d_main not available).")
    elif compute_group_level_stats_flag:
        logger_run_decoding.info(f"Skipping Avg. Specific 1D group stats for {group_identifier} (not enough valid data).")

    # --- Save overall subject metrics summary for the group ---
    if save_results_flag and group_summary_dir and len(group_results_collection["processed_subject_ids"]) > 0:
        all_subjects_metrics_list_grp = []
        for subj_id_csv in group_results_collection["processed_subject_ids"]:
            subj_metrics_csv = {
                "subject_id": subj_id_csv, "group": group_identifier,
                "global_auc_main": group_results_collection["subject_global_auc_scores"].get(subj_id_csv, np.nan)
            }
            main_metrics_subj_csv = group_results_collection["subject_global_metrics_maps"].get(subj_id_csv, {})
            if main_metrics_subj_csv:
                subj_metrics_csv.update({f"main_{k}": v for k,v in main_metrics_subj_csv.items()})
            all_subjects_metrics_list_grp.append(subj_metrics_csv)

        if all_subjects_metrics_list_grp: # Ensure list is not empty
            pd.DataFrame(all_subjects_metrics_list_grp).to_csv(
                os.path.join(group_summary_dir, f"all_subjects_summary_metrics_{group_identifier}.csv"),
                index=False
            )
            logger_run_decoding.info(
                f"Saved group summary CSV for {group_identifier} to {group_summary_dir}"
            )

    logger_run_decoding.info(
        f"Finished INTRA-SUBJECT DECODING ANALYSIS for GROUP {group_identifier}. "
        f"Total time: {(time.time() - total_group_analysis_start_time) / 60:.1f} min"
    )
    return group_results_collection["subject_global_auc_scores"] # Return for potential cross-group comparisons


# --- Cross-Subject Decoding ---
def execute_group_cross_subject_decoding_analysis(
    subject_ids_for_cs_set,
    cross_subject_set_identifier, # e.g., 'controls', 'all_participants'
    decoding_protocol_identifier="PP_AP_General_CS",
    save_results_flag=True,
    enable_verbose_logging=True,
    generate_plots_flag=True,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_fold_temporal_decoding=4, # n_jobs for temporal decoding within each LOSO fold
    compute_group_stats_for_cs_flag=True, # For stats on aggregated LOSO fold results
    classifier_type_for_cs_runs=CLASSIFIER_MODEL_TYPE,
    n_perms_for_cs_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    cs_group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    cs_group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
):
    """
    Executes cross-subject (Leave-One-Subject-Out) decoding analysis for a given set of subjects.
    """
    total_cs_analysis_start_time = time.time()
    if not base_input_data_path or not base_output_results_path:
        current_user = getuser()
        cfg_input, cfg_output = configure_project_paths(current_user)
        base_input_data_path = base_input_data_path or cfg_input
        base_output_results_path = base_output_results_path or cfg_output

    logger_run_decoding.info(
        f"Starting CROSS-SUBJECT (CS) decoding analysis for set: {cross_subject_set_identifier} "
        f"({len(subject_ids_for_cs_set)} subjects, Classifier: {classifier_type_for_cs_runs}). "
        f"Group stats (FDR & Cluster on LOSO folds): {'Enabled' if compute_group_stats_for_cs_flag else 'Disabled'}"
    )

    # --- Pre-load data for all subjects in the set ---
    subject_data_cache = {}
    valid_subject_ids_for_cs = []
    for subj_id_cs in subject_ids_for_cs_set:
        try:
            # Determine actual group for loading (could be different from cs_set_identifier if it's a mixed set)
            actual_group_for_loading_cs = cross_subject_set_identifier
            for grp_name_map, s_list_map in SUBJECT_GROUPS_MAPPING.items():
                if subj_id_cs in s_list_map:
                    actual_group_for_loading_cs = grp_name_map
                    break

            epochs_obj_cs, data_dict_cs = load_epochs_data_for_decoding(
                subj_id_cs, actual_group_for_loading_cs, base_input_data_path,
                conditions_to_load=CONFIG_LOAD_MAIN_DECODING, # Only load main conditions for CS
                verbose_logging=False # Less verbose during pre-loading
            )
            if epochs_obj_cs is None:
                logger_run_decoding.warning(f"Skipping {subj_id_cs} from CS set '{cross_subject_set_identifier}' due to data loading error.")
                continue

            xpp_cs = data_dict_cs.get("XPP_ALL")
            xap_cs = data_dict_cs.get("XAP_ALL")
            if xpp_cs is None or xap_cs is None or xpp_cs.size == 0 or xap_cs.size == 0:
                logger_run_decoding.warning(f"Skipping {subj_id_cs} from CS set '{cross_subject_set_identifier}' due to empty XPP_ALL/XAP_ALL data.")
                continue
            subject_data_cache[subj_id_cs] = {"epochs_obj": epochs_obj_cs, "XAP_data": xap_cs, "XPP_data": xpp_cs}
            valid_subject_ids_for_cs.append(subj_id_cs)
        except Exception as e_load_cs:
            logger_run_decoding.error(
                f"Failed to load data for {subj_id_cs} for CS set '{cross_subject_set_identifier}': {e_load_cs}. Skipping.",
                exc_info=True
            )

    if len(valid_subject_ids_for_cs) < 2: # Need at least 2 for LOSO (1 test, 1 train)
        logger_run_decoding.error(
            f"Not enough valid subjects ({len(valid_subject_ids_for_cs)}) for CS in '{cross_subject_set_identifier}'. Aborting CS analysis for this set."
        )
        return np.nan, np.nan, None # mean_auc, std_auc, results_dict (None here)

    cs_results_main_dir = None
    if save_results_flag or generate_plots_flag:
        cs_results_main_dir = setup_analysis_results_directory(
            base_output_results_path, "cross_subject_analysis_results",
              cross_subject_set_identifier
        )

    # --- Leave-One-Subject-Out Cross-Validation ---
    loso_cv = LeaveOneGroupOut()
    subject_indices_for_loso = np.arange(len(valid_subject_ids_for_cs)) # Groups for LOSO are subject indices
    valid_subject_ids_array = np.array(valid_subject_ids_for_cs)

    cs_fold_results = {
        "fold_global_auc_scores": {}, # {test_subject_id: auc_score}
        "fold_global_metrics_maps": {}, # {test_subject_id: metrics_dict}
        "fold_temporal_scores_1d_list": [], # List of 1D temporal scores (one per fold/test_subject)
        "fold_time_points_list": [], # List of time arrays (from test subjects)
        "fold_test_subject_ids_processed": [], # Track successfully processed test subjects
    }
    base_clf_cs = _build_standard_classifier_pipeline(
        classifier_model_type=classifier_type_for_cs_runs, random_seed_state=42
    )

    # Iterate over LOSO folds
    for fold_num, (train_subj_indices, test_subj_idx_tuple) in enumerate(
        loso_cv.split(X=np.zeros(len(valid_subject_ids_for_cs)), groups=subject_indices_for_loso), 1
    ):
        test_subj_id = valid_subject_ids_array[test_subj_idx_tuple[0]] # ID of current test subject
        train_subj_ids = valid_subject_ids_array[train_subj_indices]   # IDs of training subjects

        logger_run_decoding.info(
            f"\n--- CS Fold {fold_num}/{len(valid_subject_ids_for_cs)} (Set: '{cross_subject_set_identifier}') --- Test Subject: {test_subj_id} ---"
        )

        # Aggregate training data
        X_train_fold_list, y_train_fold_list_orig = [], []
        for tr_subj_id in train_subj_ids:
            if tr_subj_id in subject_data_cache: # Should always be true if pre-loading worked
                tr_data = subject_data_cache[tr_subj_id]
                if tr_data["XAP_data"].size > 0 and tr_data["XPP_data"].size > 0:
                    X_train_fold_list.append(np.concatenate([tr_data["XAP_data"], tr_data["XPP_data"]]))
                    y_train_fold_list_orig.append(np.concatenate([
                        np.zeros(tr_data["XAP_data"].shape[0]), np.ones(tr_data["XPP_data"].shape[0])
                    ]))
        if not X_train_fold_list: # No training data aggregated
            logger_run_decoding.warning(f"CS Fold {fold_num} (Test: {test_subj_id}): No valid training data. Skipping fold.")
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue

        X_train_fold = np.concatenate(X_train_fold_list)
        y_train_fold_orig = np.concatenate(y_train_fold_list_orig).astype(int)

        # Get testing data
        test_data = subject_data_cache[test_subj_id]
        if test_data["XAP_data"].size == 0 or test_data["XPP_data"].size == 0:
            logger_run_decoding.warning(f"CS Fold {fold_num}: Test subject {test_subj_id} has empty XAP/XPP data. Skipping fold.")
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue
        X_test_fold = np.concatenate([test_data["XAP_data"], test_data["XPP_data"]])
        y_test_fold_orig = np.concatenate([
            np.zeros(test_data["XAP_data"].shape[0]), np.ones(test_data["XPP_data"].shape[0])
        ]).astype(int)

        # Check for >1 class in both train and test
        if len(np.unique(y_train_fold_orig)) < 2 or len(np.unique(y_test_fold_orig)) < 2:
            logger_run_decoding.warning(f"CS Fold {fold_num} (Test: {test_subj_id}): Train or test set has < 2 classes. Skipping fold.")
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue
        try:
            # --- Core CS decoding call for this fold ---
            fold_auc, fold_metrics, _, fold_probas, fold_labels, fold_scores_1d = \
                run_cross_subject_decoding_for_fold(
                    X_train_fold, y_train_fold_orig, X_test_fold, y_test_fold_orig,
                    test_subj_id, cross_subject_set_identifier, decoding_protocol_identifier,
                    base_clf_cs, n_jobs_for_fold_temporal_decoding
                )
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = fold_auc
            cs_fold_results["fold_global_metrics_maps"][test_subj_id] = fold_metrics if fold_metrics else {}

            if pd.notna(fold_auc): # Only add to processed if decoding was successful
                cs_fold_results["fold_test_subject_ids_processed"].append(test_subj_id)
                if fold_scores_1d is not None and not np.all(np.isnan(fold_scores_1d)):
                    cs_fold_results["fold_temporal_scores_1d_list"].append(fold_scores_1d)
                    cs_fold_results["fold_time_points_list"].append(test_data["epochs_obj"].times.copy())

                # Generate dashboard for this fold's test subject (no TGM or intra-fold stats for CS folds)
                if generate_plots_flag and test_data["epochs_obj"] is not None and cs_results_main_dir:
                    fold_plot_dir_cs = os.path.join(cs_results_main_dir, "dashboards_cs_folds", f"test_subject_{test_subj_id}")
                    os.makedirs(fold_plot_dir_cs, exist_ok=True)
                    create_subject_decoding_dashboard_plots(
                        main_epochs_time_points=test_data["epochs_obj"].times,
                        main_original_labels_array=y_test_fold_orig,
                        main_predicted_probabilities_global=fold_probas,
                        main_predicted_labels_global=fold_labels,
                        main_cross_validation_global_scores=np.array([fold_auc]) if pd.notna(fold_auc) else np.array([np.nan]), # Treat fold AUC as a single CV score for plot
                        main_temporal_scores_1d_all_folds=fold_scores_1d[np.newaxis, :] if fold_scores_1d is not None else None, # Wrap in 2D for plot function
                        main_mean_temporal_decoding_scores_1d=fold_scores_1d,
                        main_temporal_1d_fdr_sig_data=None, # No intra-fold stats for CS fold results
                        main_temporal_1d_cluster_sig_data=None,
                        main_mean_temporal_generalization_matrix_scores=None, # No TGM for CS folds
                        main_tgm_fdr_sig_data=None,
                        subject_identifier=test_subj_id,
                        group_identifier=f"{cross_subject_set_identifier}_CS_TestFold",
                        output_directory_path=fold_plot_dir_cs,
                        chance_level_auc=CHANCE_LEVEL_AUC_SCORE,
                        specific_ap_decoding_results=None, # No specific comparisons in CS context
                        mean_of_specific_scores_1d=None,
                        sem_of_specific_scores_1d=None,
                        mean_specific_fdr_sig_data=None,
                        mean_specific_cluster_sig_data=None
                    )
        except Exception as e_cs_fold:
            logger_run_decoding.error(f"Error in CS Fold {fold_num} (Test: {test_subj_id}): {e_cs_fold}", exc_info=True)
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan

    # --- Aggregation and stats for Cross-Subject (on LOSO fold results) ---
    valid_cs_fold_global_aucs = np.array([
        s for s in cs_fold_results["fold_global_auc_scores"].values() if pd.notna(s)
    ])
    mean_cs_auc_overall, std_cs_auc_overall = np.nan, np.nan

    if len(valid_cs_fold_global_aucs) > 0:
        mean_cs_auc_overall = np.mean(valid_cs_fold_global_aucs)
        std_cs_auc_overall = np.std(valid_cs_fold_global_aucs)
        logger_run_decoding.info(
            f"CS Analysis for '{cross_subject_set_identifier}' (LOSO): Mean Global AUC = {mean_cs_auc_overall:.3f} "
            f"± {std_cs_auc_overall:.3f} (N={len(valid_cs_fold_global_aucs)} successful folds)"
        )
        if compute_group_stats_for_cs_flag and len(valid_cs_fold_global_aucs) >= 2:
            stat_cs_global, p_val_cs_global = bEEG_stats.compare_global_scores_to_chance(
                valid_cs_fold_global_aucs, CHANCE_LEVEL_AUC_SCORE, "ttest", "greater"
            )
            logger_run_decoding.info(f"  CS Global AUC vs Chance: t={stat_cs_global:.3f}, p={p_val_cs_global:.4f}")
        if generate_plots_flag and cs_results_main_dir:
            plot_group_mean_scores_barplot(
                cs_fold_results["fold_global_auc_scores"],
                f"CS Set '{cross_subject_set_identifier}' - Test Subject Global AUCs",
                cs_results_main_dir, "Global ROC AUC (CS Fold)", CHANCE_LEVEL_AUC_SCORE
            )

    # Stats on 1D temporal scores from CS folds (stacking the 1D curves from each fold)
    if len(cs_fold_results["fold_temporal_scores_1d_list"]) >= 2 and compute_group_stats_for_cs_flag:
        ref_times_cs_idx = next(
            (j for j, t_arr in enumerate(cs_fold_results["fold_time_points_list"])
             if t_arr is not None and t_arr.size > 0), -1
        )
        if ref_times_cs_idx != -1:
            ref_times_cs_grp = cs_fold_results["fold_time_points_list"][ref_times_cs_idx]
            valid_cs_temporal_scores_to_stack = []
            for i_fold_cs, _ in enumerate(cs_fold_results["fold_test_subject_ids_processed"]): # Iterate using processed IDs
                 if i_fold_cs < len(cs_fold_results["fold_temporal_scores_1d_list"]) and \
                    i_fold_cs < len(cs_fold_results["fold_time_points_list"]):
                    scores_fold_cs = cs_fold_results["fold_temporal_scores_1d_list"][i_fold_cs]
                    times_fold_cs = cs_fold_results["fold_time_points_list"][i_fold_cs]
                    if times_fold_cs is not None and times_fold_cs.size == ref_times_cs_grp.size and \
                       np.allclose(times_fold_cs, ref_times_cs_grp) and \
                       scores_fold_cs is not None and not np.all(np.isnan(scores_fold_cs)):
                        valid_cs_temporal_scores_to_stack.append(scores_fold_cs)

            if len(valid_cs_temporal_scores_to_stack) >= 2:
                stacked_cs_1d_scores_grp = np.array(valid_cs_temporal_scores_to_stack) # (n_folds, n_timepoints)
                logger_run_decoding.info(
                    f"CS Set '{cross_subject_set_identifier}' (N={stacked_cs_1d_scores_grp.shape[0]} folds for temporal stats): Running group stats..."
                )
                # FDR
                _, fdr_mask_cs_1d_grp, _ = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_cs_1d_scores_grp, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                )
                # Cluster Permutation
                t_thresh_cs_clu_grp = cs_group_cluster_test_t_thresh_value
                if cs_group_cluster_test_threshold_method == "stat" and t_thresh_cs_clu_grp is None:
                     t_thresh_cs_clu_grp = scipy_stats.t.ppf(1.0 - 0.05 / 2, df=stacked_cs_1d_scores_grp.shape[0] - 1)

                cs_clu_thresh_cfg_grp = t_thresh_cs_clu_grp if cs_group_cluster_test_threshold_method == "stat" else \
                                        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if cs_group_cluster_test_threshold_method == "tfce" else None

                _, clu_cs_1d_grp, p_clu_cs_1d_grp, _ = bEEG_stats.perform_cluster_permutation_test(
                    stacked_cs_1d_scores_grp, CHANCE_LEVEL_AUC_SCORE, n_perms_for_cs_group_cluster_test,
                    cluster_threshold_config=cs_clu_thresh_cfg_grp,
                    alternative_hypothesis="greater", n_jobs=n_jobs_for_fold_temporal_decoding # Re-check n_jobs usage
                )
                clu_map_cs_1d_grp = bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_cs_grp.shape, clu_cs_1d_grp, p_clu_cs_1d_grp
                ) if clu_cs_1d_grp and p_clu_cs_1d_grp is not None else None

                if generate_plots_flag and cs_results_main_dir:
                    plot_group_temporal_decoding_statistics(
                        ref_times_cs_grp, np.mean(stacked_cs_1d_scores_grp, axis=0),
                        f"CS Set '{cross_subject_set_identifier}' (Mean Temporal Across Folds)", cs_results_main_dir,
                        scipy_stats.sem(stacked_cs_1d_scores_grp, axis=0, nan_policy="omit") if stacked_cs_1d_scores_grp.shape[0] > 1 else None,
                        clu_map_cs_1d_grp, fdr_mask_cs_1d_grp, CHANCE_LEVEL_AUC_SCORE
                    )
            else:
                logger_run_decoding.warning(f"Not enough valid CS fold temporal scores ({len(valid_cs_temporal_scores_to_stack)}) for group stats in '{cross_subject_set_identifier}'.")
        else:
            logger_run_decoding.warning(f"Could not establish reference time points for CS group temporal stats in '{cross_subject_set_identifier}'.")
    elif compute_group_stats_for_cs_flag:
        logger_run_decoding.info(f"Skipping CS group temporal stats for '{cross_subject_set_identifier}' (not enough valid fold data).")

    # Save summary CSV for this CS set (metrics per fold)
    if save_results_flag and cs_results_main_dir and len(cs_fold_results["fold_test_subject_ids_processed"]) > 0:
        cs_summary_list_for_csv = []
        for test_subj_id_csv in cs_fold_results["fold_test_subject_ids_processed"]:
            fold_metrics_for_csv = {
                "test_subject_id": test_subj_id_csv,
                "cs_set_identifier": cross_subject_set_identifier,
                "fold_global_auc": cs_fold_results["fold_global_auc_scores"].get(test_subj_id_csv, np.nan)
            }
            # Add other metrics if present
            fold_metrics_for_csv.update(cs_fold_results["fold_global_metrics_maps"].get(test_subj_id_csv, {}))
            cs_summary_list_for_csv.append(fold_metrics_for_csv)

        if cs_summary_list_for_csv:
            pd.DataFrame(cs_summary_list_for_csv).to_csv(
                os.path.join(cs_results_main_dir, f"all_folds_summary_metrics_cs_{cross_subject_set_identifier}.csv"),
                index=False
            )
            logger_run_decoding.info(f"Saved CS summary CSV for '{cross_subject_set_identifier}' to {cs_results_main_dir}")

    logger_run_decoding.info(
        f"Finished CS decoding for '{cross_subject_set_identifier}'. "
        f"Time: {(time.time() - total_cs_analysis_start_time) / 60:.1f} min."
    )
    # The third element was 'None' for a results_dict, keep it that way if not used.
    return mean_cs_auc_overall, std_cs_auc_overall, None


# --- Main Execution Block ---
if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="EEG Decoding Analysis Orchestration Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    cli_parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "single_subject", "group_intra", "all_groups_intra",
            "cross_subject_set", "all_groups_cross",
        ],
        required=True,
        help="Execution mode.",
    )
    cli_parser.add_argument(
        "--clf_type_override",
        type=str,
        default=None,
        choices=["svc", "logreg", "rf"],
        help="Override default classifier type for this run.",
    )
    cli_parser.add_argument(
        "--n_jobs_override",
        type=str,
        default=None,
        help="Override n_jobs (e.g., '4' or 'auto') for primary MNE operations per analysis unit.",
    )
    command_line_args = cli_parser.parse_args()

    # Determine n_jobs to use
    n_jobs_str_arg = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use_main = -1 if n_jobs_str_arg.lower() == "auto" else int(n_jobs_str_arg)
    except ValueError:
        logger_run_decoding.warning(
        f"Invalid n_jobs_override ('{n_jobs_str_arg}'). Using global default: {N_JOBS_PROCESSING} -> -1 (auto).")
    n_jobs_to_use_main = -1 


    classifier_type_to_use_main = (
        command_line_args.clf_type_override
        if command_line_args.clf_type_override is not None
        else CLASSIFIER_MODEL_TYPE
    )

    user_login_main = getuser()
    main_input_path_script, main_output_path_script = configure_project_paths(user_login_main)

    logger_run_decoding.info(
        f"\n{'='*20} EEG DECODING SCRIPT STARTED ({datetime.now().strftime('%Y-%m-%d %H:%M')}) {'='*20}"
    )
    logger_run_decoding.info(f"User: {user_login_main}, Mode: {command_line_args.mode}")
    logger_run_decoding.info(
        f"  n_jobs (for primary MNE ops per unit): {n_jobs_str_arg} (parsed to: {n_jobs_to_use_main}), "
        f"Classifier Type: {classifier_type_to_use_main}"
    )
    logger_run_decoding.info(
        f"  Intra-Subject Stats (FDR & Cluster on CV folds): {'ON' if COMPUTE_INTRA_SUBJECT_STATISTICS else 'OFF'}"
    )
    logger_run_decoding.info(
        f"  Group-Level Stats (FDR & Cluster on aggregated data): {'ON' if COMPUTE_GROUP_LEVEL_STATISTICS else 'OFF'}"
    )
    logger_run_decoding.info(
        f"  Save Results: {SAVE_ANALYSIS_RESULTS}, Generate Plots: {GENERATE_PLOTS}, "
        f"TGM (main decoding): {COMPUTE_TEMPORAL_GENERALIZATION_MATRICES}"
    )
    logger_run_decoding.info(
        f"  Intra-subject permutations: {N_PERMUTATIONS_INTRA_SUBJECT}, "
        f"Group-level permutations: {N_PERMUTATIONS_GROUP_LEVEL}"
    )
    logger_run_decoding.info(f"  Intra-fold cluster threshold config: {INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG}")
    logger_run_decoding.info(
        f"  Group-level cluster threshold method: {GROUP_LEVEL_STAT_THRESHOLD_TYPE}, "
        f"t-thresh (if 'stat'): {T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING}"
    )

    # --- Parameter Dictionaries for different modes ---
    single_subject_params_main = {
        "base_input_data_path": main_input_path_script,
        "base_output_results_path": main_output_path_script,
        "n_jobs_for_processing": n_jobs_to_use_main,
        "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
        "classifier_type": classifier_type_to_use_main,
        "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "generate_plots_flag": GENERATE_PLOTS,
        "enable_verbose_logging": True, # Detailed logging for single subject
        "decoding_protocol_identifier": "Main_and_Specific_PP_AP",
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        "loading_conditions_config": CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
    }

    group_intra_params_main = {
        "base_input_data_path": main_input_path_script,
        "base_output_results_path": main_output_path_script,
        "decoding_protocol_identifier": "Main_and_Specific_PP_AP",
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "enable_verbose_logging": True, # Can be set to False for quieter group runs
        "generate_plots_flag": GENERATE_PLOTS,
        "n_jobs_for_each_subject": n_jobs_to_use_main, # n_jobs for individual subject processing within group
        "compute_intra_subject_stats_for_group_runs_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_intra_subject_folds_for_group_runs": N_PERMUTATIONS_INTRA_SUBJECT,
        "classifier_type_for_group_runs": classifier_type_to_use_main,
        "compute_tgm_for_group_subjects_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "cluster_threshold_config_intra_fold_group": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        "compute_group_level_stats_flag": COMPUTE_GROUP_LEVEL_STATISTICS,
        "n_perms_for_group_cluster_test": N_PERMUTATIONS_GROUP_LEVEL,
        "group_cluster_test_threshold_method": GROUP_LEVEL_STAT_THRESHOLD_TYPE,
        "group_cluster_test_t_thresh_value": T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
        "n_jobs_for_group_cluster_stats": n_jobs_to_use_main, # n_jobs for MNE group stat functions
    }

    cross_subject_params_main = {
        "base_input_data_path": main_input_path_script,
        "base_output_results_path": main_output_path_script,
        "decoding_protocol_identifier": "PP_AP_General_CS",
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "enable_verbose_logging": True,
        "generate_plots_flag": GENERATE_PLOTS,
        "n_jobs_for_fold_temporal_decoding": n_jobs_to_use_main, # n_jobs for temporal decoding in each CS fold
        "compute_group_stats_for_cs_flag": COMPUTE_GROUP_LEVEL_STATISTICS,
        "classifier_type_for_cs_runs": classifier_type_to_use_main,
        "n_perms_for_cs_group_cluster_test": N_PERMUTATIONS_GROUP_LEVEL,
        "cs_group_cluster_test_threshold_method": GROUP_LEVEL_STAT_THRESHOLD_TYPE,
        "cs_group_cluster_test_t_thresh_value": T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    }

    # --- Mode Execution ---
    if command_line_args.mode == "single_subject":
        input_subject_id_ss = input("Enter Subject ID to process: ").strip()
        input_group_affiliation_ss = input(
            f"Enter group affiliation for subject '{input_subject_id_ss}' "
            f"(e.g., 'controls', or leave blank to auto-detect from SUBJECT_GROUPS_MAPPING): "
        ).strip()

        resolved_group_affiliation_ss = input_group_affiliation_ss
        if not resolved_group_affiliation_ss: # Attempt auto-detection
            resolved_group_affiliation_ss = next(
                (g for g, s_list in SUBJECT_GROUPS_MAPPING.items() if input_subject_id_ss in s_list), None
            )
        if not resolved_group_affiliation_ss:
            logger_run_decoding.error(
                f"Could not determine group for '{input_subject_id_ss}'. "
                "Please provide a group affiliation or add the subject to SUBJECT_GROUPS_MAPPING."
            )
            sys.exit(1)

        logger_run_decoding.info(
            f"Processing subject '{input_subject_id_ss}' with resolved group '{resolved_group_affiliation_ss}'."
        )
        execute_single_subject_decoding(
            subject_identifier=input_subject_id_ss,
            group_affiliation=resolved_group_affiliation_ss,
            **single_subject_params_main,
        )

    elif command_line_args.mode == "group_intra":
        input_group_to_run_gi = input(
            "Enter group name to process (e.g., 'controls', 'del', 'nodel'): "
        ).strip()
        if input_group_to_run_gi in SUBJECT_GROUPS_MAPPING:
            execute_group_intra_subject_decoding_analysis(
                subject_ids_in_group=SUBJECT_GROUPS_MAPPING[input_group_to_run_gi],
                group_identifier=input_group_to_run_gi,
                **group_intra_params_main,
            )
        else:
            logger_run_decoding.error(
                f"Unknown group name '{input_group_to_run_gi}'. Available groups: "
                f"{list(SUBJECT_GROUPS_MAPPING.keys())}"
            )

    elif command_line_args.mode == "all_groups_intra":
        logger_run_decoding.info("\n--- Mode Selected: Group Intra-Subject Decoding (All Defined Groups) ---")
        for group_name_agi, subject_list_agi in SUBJECT_GROUPS_MAPPING.items():
            logger_run_decoding.info(
                f"\n{'='*10} Processing Intra-Subject Analysis for Group: {group_name_agi} {'='*10}"
            )
            execute_group_intra_subject_decoding_analysis(
                subject_list_agi, group_name_agi, **group_intra_params_main
            )

    elif command_line_args.mode == "cross_subject_set":
        input_cs_set_name_css = input(
            "Enter a unique name for this Cross-Subject analysis set (e.g., 'all_participants_mixed'): "
        ).strip()
        input_cs_subjects_str_css = input(
            f"Enter comma-separated subject IDs for the CS set '{input_cs_set_name_css}' (e.g., 'LAB1,TpAB19,TpAC23'): "
        ).strip()

        if not input_cs_set_name_css:
            logger_run_decoding.error("Cross-Subject set name cannot be empty.")
            sys.exit(1)
        input_cs_subject_list_css = [s.strip() for s in input_cs_subjects_str_css.split(",") if s.strip()]
        if len(input_cs_subject_list_css) < 2:
            logger_run_decoding.error("Cross-Subject analysis requires at least 2 subjects.")
            sys.exit(1)

        execute_group_cross_subject_decoding_analysis(
            input_cs_subject_list_css, input_cs_set_name_css, **cross_subject_params_main
        )

    elif command_line_args.mode == "all_groups_cross":
        logger_run_decoding.info("\n--- Mode Selected: Cross-Subject Decoding (All Defined Groups Individually) ---")
        for group_name_agc, subject_list_agc in SUBJECT_GROUPS_MAPPING.items():
            if len(subject_list_agc) < 2: # CS needs at least 2 subjects
                logger_run_decoding.warning(
                    f"Skipping CS for group '{group_name_agc}': needs >= 2 subjects, found {len(subject_list_agc)}."
                )
                continue
            logger_run_decoding.info(
                f"\n{'='*10} Processing Cross-Subject Analysis for Group: {group_name_agc} {'='*10}"
            )
            execute_group_cross_subject_decoding_analysis(
                subject_list_agc, group_name_agc, **cross_subject_params_main
            )

    logger_run_decoding.info(
        f"\n{'='*20} EEG DECODING SCRIPT FINISHED ({datetime.now().strftime('%Y-%m-%d %H:%M')}) {'='*20}"
    )