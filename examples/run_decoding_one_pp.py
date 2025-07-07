import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    CHANCE_LEVEL_AUC, INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_INTRA_SUBJECT_STATISTICS,
    CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS,
    N_JOBS_PROCESSING, AP_FAMILIES_FOR_SPECIFIC_COMPARISON,
    # Specific TGM configurations
    COMPUTE_TGM_FOR_MAIN_COMPARISON, COMPUTE_TGM_FOR_SPECIFIC_COMPARISONS,
    COMPUTE_TGM_FOR_INTER_FAMILY_COMPARISONS,
    # Protocol-specific functions
    get_protocol_config,get_protocol_ap_families, get_protocol_pp_comparison_events
)
from config.config import ALL_SUBJECT_GROUPS
from utils import stats_utils as bEEG_stats
from utils.loading_PP_utils import (
    load_epochs_data_for_decoding_delirium,
    load_epochs_data_for_decoding_battery,
    load_epochs_data_for_decoding_ppext3,
    load_epochs_data_auto_protocol
)
from utils.utils import (
    configure_project_paths, setup_analysis_results_directory
)
from utils.vizualization_utils_PP import create_subject_decoding_dashboard_plots
from Baking_EEG._4_decoding_core import run_temporal_decoding_analysis


import logging
import time
import argparse
from getpass import getuser
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import itertools
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# =============================================================================
# LABEL ENCODING STANDARDIZATION
# =============================================================================
# Throughout this script, we maintain consistent label encoding:
# - PP (Propre Prénom) = 1 (positive class)
# - AP () = 0 (negative class)
# This ensures consistent interpretation of:
# - AUC scores (>0.5 = PP better decoded than AP)
# - Probability outputs (predict_proba[:, 1] = probability of PP)
# - All classification metrics across different comparison types
# =============================================================================


# --- Logging Configuration ---
LOG_DIR_RUN_ONE = './logs_run_single_subject'  # Specific log directory for single subject analysis
os.makedirs(LOG_DIR_RUN_ONE, exist_ok=True)
LOG_FILENAME_RUN_ONE = os.path.join(
    LOG_DIR_RUN_ONE,
    datetime.now().strftime('log_run_single_subject_%Y-%m-%d_%H%M%S.log')
)

# Remove existing handlers to prevent log duplication
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_ONE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_one = logging.getLogger(__name__)
# Configure loggers for imported modules if needed
logging.getLogger("Baking_EEG.decoding_core").setLevel(logging.INFO)
logging.getLogger("Baking_EEG.utils.data_loading_utils").setLevel(logging.INFO)



def get_protocol_cv_folds(detected_protocol):
    """Determine number of CV folds based on protocol type.
    
    Args:
        detected_protocol (str): Protocol type ('delirium', 'battery', 'ppext3', etc.)
        
    Returns:
        int: Number of CV folds to use
    """
    if detected_protocol and detected_protocol.lower() == 'delirium':
        return 10  # 10 folds for Delirium protocol
    else:
        return 5   # 5 folds for PPext3, Battery and other protocols




def extract_frequency_info_from_path(base_input_data_path, group_affiliation):
    """Extract frequency information (01Hz or 1Hz) from data path.
    
    Args:
        base_input_data_path (str): Base path to data
        group_affiliation (str): Group name
        
    Returns:
        str: Frequency suffix ("01Hz", "1Hz", or "")
    """
    # Check for frequency-specific directories for certain groups
    freq_sensitive_groups = ["COMA", "MCS+", "MCS-", "VS"]
    
    if group_affiliation.upper() in freq_sensitive_groups:
        # Try to detect frequency from existing directories
        for freq_suffix in ["_01HZ", "_1HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_{group_affiliation.upper()}{freq_suffix}")
            if os.path.isdir(potential_path):
                return freq_suffix.replace("_", "").replace("HZ", "Hz")  # Return "01Hz" or "1Hz"
    
    return ""  # No frequency info





def execute_single_subject_decoding(
    subject_identifier,
    group_affiliation,
    decoding_protocol_identifier="Single_Protocol_Analysis",
    # Using imported constants for default values
    save_results_flag=None,
    enable_verbose_logging=False,
    generate_plots_flag=None,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_processing=None,  # Can be overridden
    classifier_type=None,
    use_grid_search_for_subject=None,
    use_csp_for_temporal_subject=None,
    use_anova_fs_for_temporal_subject=None,
    param_grid_config_for_subject=None,
    cv_folds_for_gs_subject=None,
    fixed_params_for_subject=None,
    compute_intra_subject_stats_flag=None,
    n_perms_for_intra_subject_clusters=None,
    compute_tgm_flag=None,
    loading_conditions_config=None,
    cluster_threshold_config_intra_fold=None
):
    """Executes decoding analysis for a single subject for the defined
    single protocol."""

    # Initialize default values
    if save_results_flag is None:
        save_results_flag = SAVE_ANALYSIS_RESULTS
    if generate_plots_flag is None:
        generate_plots_flag = GENERATE_PLOTS
    if n_jobs_for_processing is None:
        n_jobs_for_processing = N_JOBS_PROCESSING
    if classifier_type is None:
        classifier_type = CLASSIFIER_MODEL_TYPE
    if use_grid_search_for_subject is None:
        use_grid_search_for_subject = USE_GRID_SEARCH_OPTIMIZATION
    if use_csp_for_temporal_subject is None:
        use_csp_for_temporal_subject = USE_CSP_FOR_TEMPORAL_PIPELINES
    if use_anova_fs_for_temporal_subject is None:
        use_anova_fs_for_temporal_subject = USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
    if param_grid_config_for_subject is None:
        param_grid_config_for_subject = (
            PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION
            else None
        )
    if cv_folds_for_gs_subject is None:
        cv_folds_for_gs_subject = CV_FOLDS_FOR_GRIDSEARCH_INTERNAL
    if fixed_params_for_subject is None:
        fixed_params_for_subject = (
            FIXED_CLASSIFIER_PARAMS_CONFIG
            if not USE_GRID_SEARCH_OPTIMIZATION else None
        )
    if compute_intra_subject_stats_flag is None:
        compute_intra_subject_stats_flag = COMPUTE_INTRA_SUBJECT_STATISTICS
    if n_perms_for_intra_subject_clusters is None:
        n_perms_for_intra_subject_clusters = N_PERMUTATIONS_INTRA_SUBJECT
    if compute_tgm_flag is None:
        compute_tgm_flag = None
    if loading_conditions_config is None:
        loading_conditions_config = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    if cluster_threshold_config_intra_fold is None:
        cluster_threshold_config_intra_fold = (
            INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG
        )

    total_start_time = time.time()
    subject_results_dir = None
    subject_results = {
        "subject_id": subject_identifier, 
        "group": group_affiliation,
        "decoding_protocol_identifier": decoding_protocol_identifier,
        "classifier_type_used": classifier_type,
        "epochs_time_points": None,
        "pp_ap_main_original_labels": None,
        "pp_ap_main_pred_probas_global": None,
        "pp_ap_main_pred_labels_global": None, 
        "pp_ap_main_cv_global_scores": None,
        "pp_ap_main_scores_1d_all_folds": None, 
        "pp_ap_main_scores_1d_mean": None,
        "pp_ap_main_temporal_1d_fdr": None, 
        "pp_ap_main_temporal_1d_cluster": None,
        "pp_ap_main_tgm_all_folds": None, 
        "pp_ap_main_tgm_mean": None,
        "pp_ap_main_tgm_fdr": None, 
        "pp_ap_main_mean_auc_global": np.nan,
        "pp_ap_main_global_metrics": {}, 
        "pp_ap_specific_ap_results": [],
        "pp_ap_mean_of_specific_scores_1d": None, 
        "pp_ap_sem_of_specific_scores_1d": None,
        "pp_ap_mean_specific_fdr": None, 
        "pp_ap_mean_specific_cluster": None,
        "pp_ap_ap_vs_ap_results": [], 
        "pp_ap_ap_centric_avg_results": [],
    }

    # Convert n_jobs_for_processing if it's "auto"
    if isinstance(n_jobs_for_processing, str) and n_jobs_for_processing.lower() == "auto":
        actual_n_jobs = -1
    else:
        try:
            actual_n_jobs = int(n_jobs_for_processing)
        except ValueError:
            logger_run_one.warning(
                "Invalid n_jobs_for_processing '%s', using -1.",
                n_jobs_for_processing)
            actual_n_jobs = -1

    try:
        # Ensure paths are configured (fallback for direct function calls)
        if not base_input_data_path or not base_output_results_path:
            current_user = getuser()
            cfg_input, cfg_output = configure_project_paths(current_user)
            base_input_data_path = base_input_data_path or cfg_input
            base_output_results_path = base_output_results_path or cfg_output
            logger_run_one.info("Auto-configured paths: input=%s, output=%s", 
                               base_input_data_path, base_output_results_path)

        logger_run_one.info(
            "Starting decoding for subject: %s (Group: %s, Task Set ID: %s, Classifier: %s, "
            "GS: %s, CSP : %s, ANOVA FS : %s, n_jobs: %s)",
            subject_identifier, group_affiliation, decoding_protocol_identifier,
            classifier_type, use_grid_search_for_subject,
            use_csp_for_temporal_subject, use_anova_fs_for_temporal_subject, actual_n_jobs
        )

        # First, load epochs with basic config to detect protocol
        basic_loading_conditions = {"XPP_ALL": "PP/", "XAP_ALL": "AP/"}
        epochs_object, returned_data_dict_basic, detected_protocol = load_epochs_data_auto_protocol(
            subject_identifier, group_affiliation, base_input_data_path,
            basic_loading_conditions, enable_verbose_logging
        )

        # Log detected protocol information
        logger_run_one.info(
            "Detected protocol '%s' for subject %s",
            detected_protocol, subject_identifier
        )

        # Get protocol-specific configuration and reload data
        actual_loading_conditions = get_protocol_config(detected_protocol.lower() if detected_protocol else 'delirium')
        
        # Reload epochs with protocol-specific configuration
        epochs_object, returned_data_dict, detected_protocol = load_epochs_data_auto_protocol(
            subject_identifier, group_affiliation, base_input_data_path,
            actual_loading_conditions, enable_verbose_logging
        )
        
        # Get protocol-specific configurations after detection
        protocol_pp_events = get_protocol_pp_comparison_events(detected_protocol)
        
        logger_run_one.info(
            "Using protocol-specific loading configuration for %s: %s", 
            detected_protocol, list(actual_loading_conditions.keys())
        )
        
        logger_run_one.info("Using loading conditions: %s", list(
            actual_loading_conditions.keys()))

        if epochs_object is None:
            logger_run_one.error(
                "Epochs object loading failed for %s. Aborting subject.", subject_identifier)
            return subject_results

        # Check if we have any valid data before proceeding
        total_valid_epochs = sum(
            arr.shape[0] for arr in returned_data_dict.values()
            if hasattr(arr, 'shape') and arr.ndim == 3 and arr.shape[0] > 0
        )
        
        if total_valid_epochs == 0:
            logger_run_one.error(
                "No valid epoch data loaded for subject %s. "
                "Available conditions: %s. Aborting analysis.",
                subject_identifier, list(returned_data_dict.keys())
            )
            # Log what was actually loaded for debugging
            for condition, data in returned_data_dict.items():
                if hasattr(data, 'shape'):
                    logger_run_one.info("  %s: shape %s", condition, data.shape)
                else:
                    logger_run_one.info("  %s: %s", condition, type(data))
            return subject_results
        
        logger_run_one.info(
            "Successfully loaded %d total epochs across all conditions for %s",
            total_valid_epochs, subject_identifier
        )

        # Store protocol information and epoch times
        subject_results["detected_protocol"] = detected_protocol
        subject_results["epochs_time_points"] = epochs_object.times
        
        # Get protocol-specific configurations
        protocol_ap_families = get_protocol_ap_families(detected_protocol)
        protocol_cv_folds = get_protocol_cv_folds(detected_protocol)
        # protocol_pp_events already defined above after protocol detection
        
        logger_run_one.info(
            "Protocol-specific settings for %s: CV folds=%d, AP families=%d, PP events=%s",
            detected_protocol, protocol_cv_folds, len(protocol_ap_families), protocol_pp_events
        )

        current_fixed_params_for_clf_dict = None
        current_param_grid_for_clf_dict = None

        if use_grid_search_for_subject:
            if param_grid_config_for_subject and classifier_type in param_grid_config_for_subject:
                # Déjà le dictionnaire complet
                current_param_grid_for_clf_dict = param_grid_config_for_subject
            else:
                logger_run_one.warning(
                    "GS for %s, but no grid for '%s' in param_grid_config_for_subject. Core decoding defaults will be used.",
                    subject_identifier, classifier_type)
        else:
            if fixed_params_for_subject and classifier_type in fixed_params_for_subject:
                current_fixed_params_for_clf_dict = fixed_params_for_subject[classifier_type]
            else:
                logger_run_one.warning(
                    "No GS for %s, no fixed params for '%s' in fixed_params_for_subject. Core decoding defaults will be used.",
                    subject_identifier, classifier_type)

        # --- Start of the single protocol's decoding logic (formerly PP_AP logic) ---
        logger_run_one.info(
            "--- Starting Main Protocol Decoding for %s ---", subject_identifier)

        logger_run_one.info(
            "  --- 1. Main Decoding (e.g., PP/all vs AP/all) for %s ---", subject_identifier)
        xpp_main_data = returned_data_dict.get("XPP_ALL")
        xap_main_data = returned_data_dict.get("XAP_ALL")

        if xpp_main_data is not None and xap_main_data is not None and \
           xpp_main_data.size > 0 and xap_main_data.size > 0:
            # STANDARDIZED: PP=1 (positive class), AP=0 (negative class)
            main_protocol_data = np.concatenate(
                [xpp_main_data, xap_main_data], axis=0)
            main_protocol_labels_orig = np.concatenate(
                [np.ones(xpp_main_data.shape[0]),   # PP = 1 (positive class)
                 np.zeros(xap_main_data.shape[0])]  # AP = 0 (negative class)
            )
            subject_results["pp_ap_main_original_labels"] = main_protocol_labels_orig.copy(
            )

            main_labels_encoded = LabelEncoder().fit_transform(main_protocol_labels_orig)
            if len(np.unique(main_labels_encoded)) < 2:
                logger_run_one.error(
                    "Subj %s: Only one class for main decoding. Skipping.", subject_identifier)
            else:
                min_samples_main = np.min(np.bincount(main_labels_encoded))
                num_cv_splits_main = min(protocol_cv_folds, min_samples_main)
                
                main_decoding_output = run_temporal_decoding_analysis(
                        epochs_data=main_protocol_data, 
                        target_labels=main_labels_encoded,
                        classifier_model_type=classifier_type, 
                        use_grid_search=use_grid_search_for_subject,
                        use_csp_for_temporal_pipelines=use_csp_for_temporal_subject, 
                        use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                        param_grid_config=current_param_grid_for_clf_dict, 
                        cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                        fixed_classifier_params=current_fixed_params_for_clf_dict,
                        cross_validation_splitter=num_cv_splits_main,
                        trial_sample_weights="auto",
                        n_jobs_external=actual_n_jobs,
                        group_labels_for_cv=None,
                        compute_intra_fold_stats=compute_intra_subject_stats_flag,
                        chance_level=CHANCE_LEVEL_AUC,
                        n_permutations_for_intra_fold_clusters=n_perms_for_intra_subject_clusters,
                        compute_temporal_generalization_matrix=compute_tgm_flag if compute_tgm_flag is not None else COMPUTE_TGM_FOR_MAIN_COMPARISON,
                        cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold
                    )
                subject_results.update({
                        "pp_ap_main_pred_probas_global": main_decoding_output[0],
                        "pp_ap_main_pred_labels_global": main_decoding_output[1],
                        "pp_ap_main_cv_global_scores": main_decoding_output[2],
                        "pp_ap_main_scores_1d_mean": main_decoding_output[3],
                        "pp_ap_main_global_metrics": main_decoding_output[4],
                        "pp_ap_main_temporal_1d_fdr": main_decoding_output[5],
                        "pp_ap_main_temporal_1d_cluster": main_decoding_output[6],
                        "pp_ap_main_scores_1d_all_folds": main_decoding_output[7],
                        "pp_ap_main_tgm_mean": main_decoding_output[8],
                        "pp_ap_main_tgm_fdr": main_decoding_output[9],
                        "pp_ap_main_tgm_all_folds": main_decoding_output[11],
                    })
                mean_auc_val = np.nanmean(
                        main_decoding_output[2]) if main_decoding_output[2] is not None and main_decoding_output[2].size > 0 else np.nan
                subject_results["pp_ap_main_mean_auc_global"] = mean_auc_val
                logger_run_one.info("Main Decoding for %s DONE. Mean Global AUC: %.3f",
                                        subject_identifier, mean_auc_val if pd.notna(mean_auc_val) else -1)
        else:
            logger_run_one.warning(
                "Subj %s: Missing XPP_ALL or XAP_ALL data. Skipping main decoding.", subject_identifier)

        logger_run_one.info(
            "  --- 2. Specific Task Comparisons (e.g. PP_spec vs AP_family_X) for %s ---", subject_identifier)
        current_time_points_ref = subject_results.get("epochs_time_points")
        if current_time_points_ref is None:
            logger_run_one.error(
                "Time points not available for %s, cannot run specific tasks.", subject_identifier)
        else:
            pp_specific_data = returned_data_dict.get(
                "PP_FOR_SPECIFIC_COMPARISON")
            if pp_specific_data is not None and pp_specific_data.size > 0:
                subject_results["pp_ap_specific_ap_results"] = []
                for ap_family_key_enum, _ in protocol_ap_families.items():
                    ap_family_data_enum = returned_data_dict.get(
                        ap_family_key_enum)
                    comparison_name_specific = f"PP_spec vs {ap_family_key_enum.replace('_', ' ').replace('AP FAMILY', 'AP Fam.')}"

                    task_result_specific = {
                        "comparison_name": comparison_name_specific, "scores_1d_mean": None,
                        "all_folds_scores_1d": None, "times": current_time_points_ref.copy(),
                        "fdr_significance_data": None, "cluster_significance_data": None
                    }
                    if ap_family_data_enum is not None and ap_family_data_enum.size > 0:
                        # STANDARDIZED: PP=1 (positive class), AP=0 (negative class)
                        task_data_specific_current = np.concatenate(
                            [pp_specific_data, ap_family_data_enum], axis=0)
                        task_labels_specific_orig = np.concatenate(
                            [np.ones(pp_specific_data.shape[0]),    # PP = 1 (positive class)
                             np.zeros(ap_family_data_enum.shape[0])] # AP = 0 (negative class)
                        )
                        task_labels_specific_enc = LabelEncoder().fit_transform(task_labels_specific_orig)

                        if len(np.unique(task_labels_specific_enc)) < 2:
                            logger_run_one.warning(
                                "Subj %s, Task '%s': Only one class. Skipping.", subject_identifier, comparison_name_specific)
                        else:
                            min_samples_task_spec = np.min(
                                np.bincount(task_labels_specific_enc))
                            num_cv_task_spec = min(
                                protocol_cv_folds, min_samples_task_spec) if min_samples_task_spec >= 2 else 0
                            if num_cv_task_spec < 2:
                                logger_run_one.warning(
                                    "Subj %s, Task '%s': Not enough samples for CV (%d splits). Skipping.", subject_identifier, comparison_name_specific, num_cv_task_spec)
                            else:
                                cv_splitter_task_spec = StratifiedKFold(
                                    n_splits=num_cv_task_spec, shuffle=True, random_state=42)
                                specific_task_output = run_temporal_decoding_analysis(
                                    epochs_data=task_data_specific_current,
                                    target_labels=task_labels_specific_orig,
                                    classifier_model_type=classifier_type,
                                    use_grid_search=use_grid_search_for_subject,
                                    use_csp_for_temporal_pipelines=use_csp_for_temporal_subject,
                                    use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                                    param_grid_config=current_param_grid_for_clf_dict,
                                    cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                                    fixed_classifier_params=current_fixed_params_for_clf_dict,
                                    cross_validation_splitter=cv_splitter_task_spec,
                                    n_jobs_external=actual_n_jobs,
                                    compute_intra_fold_stats=compute_intra_subject_stats_flag,
                                    n_permutations_for_intra_fold_clusters=n_perms_for_intra_subject_clusters,
                                    compute_temporal_generalization_matrix=COMPUTE_TGM_FOR_SPECIFIC_COMPARISONS,
                                    chance_level=CHANCE_LEVEL_AUC,
                                    cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold
                                )
                                task_result_specific.update({
                                    "scores_1d_mean": specific_task_output[3],
                                    "fdr_significance_data": specific_task_output[5],
                                    "cluster_significance_data": specific_task_output[6],
                                    "all_folds_scores_1d": specific_task_output[7]
                                })
                                peak_auc_val = (np.nanmax(
                                    specific_task_output[3]) if specific_task_output[3] is not None and specific_task_output[3].size > 0 else np.nan)
                                logger_run_one.info("  Specific task '%s' for %s: Peak AUC = %.3f", comparison_name_specific,
                                                    subject_identifier, peak_auc_val if pd.notna(peak_auc_val) else -1)
                    else:
                        logger_run_one.info("Subject %s: Missing data for %s in specific task '%s'. This may be normal depending on subject's protocol.",
                                            subject_identifier, ap_family_key_enum, comparison_name_specific)
                    subject_results["pp_ap_specific_ap_results"].append(
                        task_result_specific)
            else:
                logger_run_one.info(
                    "Subject %s: PP_FOR_SPECIFIC_COMPARISON data missing. "
                    "This is normal if the subject doesn't have this type of specific data according to the protocol. "
                    "Moving to inter-family comparisons.", subject_identifier)
        logger_run_one.info(
            "  --- Specific Task Comparisons for %s DONE ---", subject_identifier)

        if compute_intra_subject_stats_flag and subject_results.get("pp_ap_specific_ap_results"):
            logger_run_one.info(
                "  --- 3. Stats on stack of Specific Task Mean Curves for %s ---", subject_identifier)
            valid_mean_scores_for_stack = [
                res["scores_1d_mean"] for res in subject_results["pp_ap_specific_ap_results"]
                if res.get("scores_1d_mean") is not None and current_time_points_ref is not None and
                res["scores_1d_mean"].shape == current_time_points_ref.shape
            ]
            if len(valid_mean_scores_for_stack) >= 2:
                stacked_specific_curves = np.array(valid_mean_scores_for_stack)
                subject_results["pp_ap_mean_of_specific_scores_1d"] = np.nanmean(
                    stacked_specific_curves, axis=0)
                if stacked_specific_curves.shape[0] > 1:
                    subject_results["pp_ap_sem_of_specific_scores_1d"] = scipy.stats.sem(
                        stacked_specific_curves, axis=0, nan_policy='omit')

                _, fdr_mask_stack, fdr_pval_stack, fdr_test_info_stack = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_specific_curves, CHANCE_LEVEL_AUC, alternative_hypothesis="greater",
                    statistical_test_type="wilcoxon"  # Force Wilcoxon test
                )
                subject_results["pp_ap_mean_specific_fdr"] = {
                    "mask": fdr_mask_stack, "p_values": fdr_pval_stack, 
                    "p_values_raw": fdr_test_info_stack.get("p_values_raw", fdr_pval_stack),
                    "method": f"FDR on stack of {len(valid_mean_scores_for_stack)} specific curves (Wilcoxon)"}

                _, clu_obj_stack, p_clu_stack, clu_info_stack = bEEG_stats.perform_cluster_permutation_test(
                    stacked_specific_curves, CHANCE_LEVEL_AUC, n_perms_for_intra_subject_clusters,
                    cluster_threshold_config_intra_fold, "greater", actual_n_jobs
                )
                combined_mask_clu_stack = np.zeros_like(
                    subject_results["pp_ap_mean_of_specific_scores_1d"], dtype=bool) if subject_results["pp_ap_mean_of_specific_scores_1d"] is not None else np.array([], dtype=bool)
                sig_clu_objects_stack = []
                if clu_obj_stack and p_clu_stack is not None and combined_mask_clu_stack.size > 0:
                    for i_c, c_mask_item_stack in enumerate(clu_obj_stack):
                        if p_clu_stack[i_c] < 0.05:
                            sig_clu_objects_stack.append(
                                c_mask_item_stack)  
                            combined_mask_clu_stack = np.logical_or(
                                combined_mask_clu_stack, c_mask_item_stack)  
                subject_results["pp_ap_mean_specific_cluster"] = {
                    "mask": combined_mask_clu_stack, "cluster_objects": sig_clu_objects_stack,
                    "p_values_all_clusters": p_clu_stack, 
                    "cluster_info": clu_info_stack,
                    "method": f"CluPerm on stack of {len(valid_mean_scores_for_stack)} specific curves (TTest)"}
                logger_run_one.info(
                    "  --- Stats on stack of Specific Task Mean Curves for %s DONE ---", subject_identifier)
            else:
                logger_run_one.warning(
                    "Subj %s: Not enough valid specific task curves (%d) for stack statistics.", subject_identifier, len(valid_mean_scores_for_stack))


        logger_run_one.info(
            "  --- 4. Inter-Family Decoding Tasks (e.g. AP_fam_X vs AP_fam_Y) for %s ---", subject_identifier)
        subject_results["pp_ap_ap_vs_ap_results"] = []
        ap_family_keys_list = list(protocol_ap_families.keys())
        if len(ap_family_keys_list) >= 2 and current_time_points_ref is not None:
            for ap_key_1, ap_key_2 in itertools.permutations(ap_family_keys_list, 2):
                data_ap1_current = returned_data_dict.get(ap_key_1)
                data_ap2_current = returned_data_dict.get(ap_key_2)
                comparison_name_ap_vs_ap = f"{ap_key_1.replace('_', ' ')} vs {ap_key_2.replace('_', ' ')}"
                task_result_ap_vs_ap = {
                    "comparison_name": comparison_name_ap_vs_ap, "scores_1d_mean": None, "all_folds_scores_1d": None,
                    "times": current_time_points_ref.copy(), "fdr_significance_data": None, "cluster_significance_data": None
                }
                if data_ap1_current is not None and data_ap1_current.size > 0 and data_ap2_current is not None and data_ap2_current.size > 0:
                    task_data_ap_vs_ap = np.concatenate(
                        [data_ap1_current, data_ap2_current], axis=0)
                    task_labels_ap_vs_ap_orig = np.concatenate(
                        [np.zeros(data_ap1_current.shape[0]), np.ones(data_ap2_current.shape[0])])
                    task_labels_ap_vs_ap_enc = LabelEncoder().fit_transform(task_labels_ap_vs_ap_orig)
                    if len(np.unique(task_labels_ap_vs_ap_enc)) < 2:
                        logger_run_one.warning(
                            "Subj %s, Task '%s': Only one class. Skipping.", subject_identifier, comparison_name_ap_vs_ap)
                    else:
                        min_samples_ap_vs_ap = np.min(
                            np.bincount(task_labels_ap_vs_ap_enc))
                        num_cv_ap_vs_ap = min(
                            protocol_cv_folds, min_samples_ap_vs_ap) if min_samples_ap_vs_ap >= 2 else 0
                        if num_cv_ap_vs_ap < 2:
                            logger_run_one.warning(
                                "Subj %s, Task '%s': Not enough samples for CV (%d splits). Skipping.", subject_identifier, comparison_name_ap_vs_ap, num_cv_ap_vs_ap)
                        else:
                            cv_splitter_ap_vs_ap = StratifiedKFold(
                                n_splits=num_cv_ap_vs_ap, shuffle=True, random_state=42)
                            ap_vs_ap_task_output = run_temporal_decoding_analysis(
                                epochs_data=task_data_ap_vs_ap, target_labels=task_labels_ap_vs_ap_orig,
                                classifier_model_type=classifier_type, use_grid_search=use_grid_search_for_subject,
                                use_csp_for_temporal_pipelines=use_csp_for_temporal_subject,
                                use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                                param_grid_config=current_param_grid_for_clf_dict,
                                cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                                fixed_classifier_params=current_fixed_params_for_clf_dict,
                                cross_validation_splitter=cv_splitter_ap_vs_ap,
                                n_jobs_external=actual_n_jobs,
                                compute_intra_fold_stats=compute_intra_subject_stats_flag,
                                n_permutations_for_intra_fold_clusters=n_perms_for_intra_subject_clusters,
                                compute_temporal_generalization_matrix=COMPUTE_TGM_FOR_INTER_FAMILY_COMPARISONS,
                                chance_level=CHANCE_LEVEL_AUC,
                                cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold
                            )
                            task_result_ap_vs_ap.update({
                                "scores_1d_mean": ap_vs_ap_task_output[3],
                                "fdr_significance_data": ap_vs_ap_task_output[5],
                                "cluster_significance_data": ap_vs_ap_task_output[6],
                                "all_folds_scores_1d": ap_vs_ap_task_output[7]
                            })
                            peak_auc_val_apap = (np.nanmax(
                                ap_vs_ap_task_output[3]) if ap_vs_ap_task_output[3] is not None and ap_vs_ap_task_output[3].size > 0 else np.nan)
                            logger_run_one.info("  Inter-Family task '%s' for %s: Peak AUC = %.3f", comparison_name_ap_vs_ap,
                                                subject_identifier, peak_auc_val_apap if pd.notna(peak_auc_val_apap) else -1)
                else:
                    logger_run_one.info("Subj %s: Données manquantes pour %s ou %s dans la tâche '%s'. Ceci peut être normal selon le protocole du sujet.",
                                        subject_identifier, ap_key_1, ap_key_2, comparison_name_ap_vs_ap)
                subject_results["pp_ap_ap_vs_ap_results"].append(
                    task_result_ap_vs_ap)
        else:
            logger_run_one.warning(
                "Subj %s: Not enough AP families (%d) or time_points missing. Skipping Inter-Family tasks.", subject_identifier, len(ap_family_keys_list))
        logger_run_one.info(
            "  --- Inter-Family Tasks for %s DONE ---", subject_identifier)

        if compute_intra_subject_stats_flag:
            logger_run_one.info(
                "  --- 5. Computing Anchor-Centric Averages for %s ---", subject_identifier)
            subject_results["pp_ap_ap_centric_avg_results"] = []
            if current_time_points_ref is not None:
                for anchor_ap_key_centric in ap_family_keys_list:
                    curves_to_average_this_anchor = []
                    constituent_names_debug_this_anchor = []
                    anchor_ap_display_name = anchor_ap_key_centric.replace(
                        '_', ' ').replace('AP FAMILY', 'AP Fam.')
                    anchor_ap_storage_name = anchor_ap_key_centric.replace(
                        '_', ' ')
                    lookup_pp_vs_anchor = f"PP_spec vs {anchor_ap_display_name}"
                    for res_spec in subject_results.get("pp_ap_specific_ap_results", []):
                        if res_spec.get("comparison_name") == lookup_pp_vs_anchor:
                            if res_spec.get("scores_1d_mean") is not None:
                                curves_to_average_this_anchor.append(
                                    res_spec["scores_1d_mean"])
                                constituent_names_debug_this_anchor.append(
                                    lookup_pp_vs_anchor)
                            break
                    for other_ap_key_comparison in ap_family_keys_list:
                        if other_ap_key_comparison == anchor_ap_key_centric:
                            continue
                        other_ap_storage_name = other_ap_key_comparison.replace(
                            '_', ' ')
                        lookup_anchor_vs_other = f"{anchor_ap_storage_name} vs {other_ap_storage_name}"
                        found_this_ap_vs_ap_curve = False
                        for res_ap_vs_ap in subject_results.get("pp_ap_ap_vs_ap_results", []):
                            if res_ap_vs_ap.get("comparison_name") == lookup_anchor_vs_other:
                                if res_ap_vs_ap.get("scores_1d_mean") is not None:
                                    curves_to_average_this_anchor.append(
                                        res_ap_vs_ap["scores_1d_mean"])
                                    constituent_names_debug_this_anchor.append(
                                        lookup_anchor_vs_other)
                                    found_this_ap_vs_ap_curve = True
                                break
                        if not found_this_ap_vs_ap_curve:
                            logger_run_one.debug(
                                "Anchor %s: Curve '%s' not found.", anchor_ap_display_name, lookup_anchor_vs_other)

                    ap_centric_avg_item = {"anchor_ap_family_key_name": anchor_ap_display_name, "average_scores_1d": None, "sem_scores_1d": None, "fdr_sig_data": None,
                                           "cluster_sig_data": None, "constituent_comparison_names_detail": constituent_names_debug_this_anchor, "num_constituent_curves": 0}
                    if len(curves_to_average_this_anchor) >= 2:
                        valid_curves_for_stacking = [
                            c for c in curves_to_average_this_anchor if c is not None and c.shape == current_time_points_ref.shape]
                        if len(valid_curves_for_stacking) >= 2:
                            stacked_curves_for_avg = np.array(
                                valid_curves_for_stacking)
                            ap_centric_avg_item["average_scores_1d"] = np.nanmean(
                                stacked_curves_for_avg, axis=0)
                            ap_centric_avg_item["num_constituent_curves"] = stacked_curves_for_avg.shape[0]
                            if stacked_curves_for_avg.shape[0] > 1:
                                ap_centric_avg_item["sem_scores_1d"] = scipy.stats.sem(
                                    stacked_curves_for_avg, axis=0, nan_policy='omit')

                            _, fdr_mask_centric, fdr_pval_centric, fdr_test_info_centric = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                                stacked_curves_for_avg, CHANCE_LEVEL_AUC, alternative_hypothesis="greater",
                                statistical_test_type="wilcoxon"  # Force Wilcoxon test
                            )
                            ap_centric_avg_item["fdr_sig_data"] = {
                                "mask": fdr_mask_centric, "p_values": fdr_pval_centric, 
                                "p_values_raw": fdr_test_info_centric.get("p_values_raw", fdr_pval_centric),
                                "method": f"FDR on stack for {anchor_ap_display_name} (Wilcoxon)"}

                            _, clu_obj_centric, p_clu_centric, clu_info_centric = bEEG_stats.perform_cluster_permutation_test(
                                stacked_curves_for_avg, CHANCE_LEVEL_AUC, n_perms_for_intra_subject_clusters,
                                cluster_threshold_config_intra_fold, "greater", actual_n_jobs)
                            combined_mask_clu_centric = np.zeros_like(
                                ap_centric_avg_item["average_scores_1d"], dtype=bool) if ap_centric_avg_item["average_scores_1d"] is not None else np.array([], dtype=bool)
                            sig_clu_objects_centric = []
                            if clu_obj_centric and p_clu_centric is not None and combined_mask_clu_centric.size > 0:
                                
                                for i_cc, c_mask_item_centric in enumerate(clu_obj_centric):
                                    if p_clu_centric[i_cc] < 0.05:
                                        sig_clu_objects_centric.append(
                                            c_mask_item_centric)  
                                        combined_mask_clu_centric = np.logical_or(
                                            combined_mask_clu_centric, c_mask_item_centric)  
                            ap_centric_avg_item["cluster_sig_data"] = {
                                "mask": combined_mask_clu_centric, "cluster_objects": sig_clu_objects_centric,
                                "p_values_all_clusters": p_clu_centric, 
                                "cluster_info": clu_info_centric,
                                "method": f"CluPerm on stack for {anchor_ap_display_name} (TTest)"}
                            logger_run_one.info("  Anchor-centric avg for %s from %d curves. Found: %s",
                                                anchor_ap_display_name, stacked_curves_for_avg.shape[0], constituent_names_debug_this_anchor)
                        else:
                            logger_run_one.warning("Subj %s, Anchor %s: Not enough valid curves after filtering (%d) for avg. Orig: %d. Debug: %s", subject_identifier, anchor_ap_display_name, len(
                                valid_curves_for_stacking), len(curves_to_average_this_anchor), constituent_names_debug_this_anchor)
                    else:
                        logger_run_one.info("Subj %s, Anchor %s: Pas assez de courbes constituantes (%d) pour moyenne. Ceci est normal si les données spécifiques manquent. Debug: %s", subject_identifier, anchor_ap_display_name, len(
                            curves_to_average_this_anchor), constituent_names_debug_this_anchor)
                    subject_results["pp_ap_ap_centric_avg_results"].append(
                        ap_centric_avg_item)
            else:
                logger_run_one.warning(
                    "Time points ref not available for %s, skipping anchor-centric avgs.", subject_identifier)
            logger_run_one.info(
                "  --- Anchor-Centric Averages for %s DONE ---", subject_identifier)
        # --- End of the single protocol's decoding logic ---

    except FileNotFoundError as fnfe:
        logger_run_one.error(
            "FileNotFoundError for subject %s: %s.", subject_identifier, fnfe, exc_info=True)
        return subject_results
    except ValueError as ve:
        logger_run_one.error(
            "ValueError during processing for subject %s: %s.", subject_identifier, ve, exc_info=True)
        return subject_results
    except KeyError as ke:
        logger_run_one.error(
            "KeyError during processing for subject %s: %s.", subject_identifier, ke, exc_info=True)
        return subject_results
    except Exception as e:
        logger_run_one.error(
            "Unexpected error during main processing logic for subject %s: %s", subject_identifier, e, exc_info=True)
        return subject_results

    # Setup results directory and save/plot if requested
    if save_results_flag or generate_plots_flag:
        try:
            # Get the detected protocol for folder organization
            detected_protocol = subject_results.get("detected_protocol", "unknown")
            
            # Extract frequency information from data path
            frequency_info = extract_frequency_info_from_path(base_input_data_path, group_affiliation)
            
            # Create hierarchical folder structure: Group / Protocol / Subject_details
            dec_prot_id_str = str(
                decoding_protocol_identifier if decoding_protocol_identifier else "UnknownProtocolID")
            subfolder_name_components = [subject_identifier, dec_prot_id_str.replace(
                " ", "_").replace("/", "-"), classifier_type]
            valid_subfolder_components = [
                comp for comp in subfolder_name_components if comp]
            subfolder_name_for_setup = "_".join(valid_subfolder_components)
            
            # Create group_protocol path with frequency info for better organization
            group_protocol_components = [group_affiliation]
            if frequency_info:
                group_protocol_components.append(frequency_info)
            group_protocol_components.append(detected_protocol)
            group_protocol_path = "_".join(group_protocol_components)
            
            logger_run_one.info(
                "Results will be saved in: %s/%s", 
                group_protocol_path, subfolder_name_for_setup
            )
            
            subject_results_dir = setup_analysis_results_directory(
                base_output_results_path, "intra_subject_results", group_protocol_path, subfolder_name_for_setup
            )
        except Exception as e_setup_dir:
            logger_run_one.error(
                "Failed to setup results directory for %s: %s. Plots/saving skipped.", subject_identifier, e_setup_dir, exc_info=True)
            subject_results_dir = None

    # Save results if requested
    if save_results_flag and subject_results_dir:
        try:
            results_file_path = os.path.join(
                subject_results_dir, "decoding_results_full.npz")
            np.savez_compressed(results_file_path, **subject_results)

            csv_summary_data = {"subject_id": subject_identifier, "group": group_affiliation,
                                "protocol_task_id": decoding_protocol_identifier, "classifier": classifier_type}
            csv_summary_data["main_global_auc_mean"] = subject_results.get(
                "pp_ap_main_mean_auc_global", np.nan)
            cv_scores_main = subject_results.get("pp_ap_main_cv_global_scores")
            csv_summary_data["main_global_auc_std"] = np.nanstd(
                cv_scores_main) if cv_scores_main is not None and cv_scores_main.size > 0 else np.nan
            main_metrics = subject_results.get("pp_ap_main_global_metrics", {})
            for k_metric, v_metric in main_metrics.items():
                csv_summary_data[f"main_metric_{k_metric}"] = v_metric

            summary_csv_path = os.path.join(
                subject_results_dir, "summary_metrics.csv")
            pd.DataFrame([csv_summary_data]).to_csv(
                summary_csv_path, index=False)
            logger_run_one.info(
                "Results saved for subject %s in %s", subject_identifier, subject_results_dir)
        except Exception as e_save:
            logger_run_one.error("Failed to save results for %s to %s: %s",
                                 subject_identifier, subject_results_dir, e_save, exc_info=True)

    # Generate plots if requested
    if generate_plots_flag and subject_results.get("epochs_time_points") is not None:
        if subject_results_dir:
            try:
                dashboard_plot_args = {
                    "main_epochs_time_points": subject_results.get("epochs_time_points"),
                    "classifier_name_for_title": classifier_type,
                    "subject_identifier": subject_identifier,
                    "group_identifier": group_affiliation,
                    "output_directory_path": subject_results_dir,
                    "CHANCE_LEVEL_AUC": CHANCE_LEVEL_AUC, 
                    "protocol_type": detected_protocol if detected_protocol else "PP_AP",  # Dynamic protocol type
                    "n_folds": protocol_cv_folds,  # Dynamic CV folds

                    "main_original_labels_array": subject_results.get("pp_ap_main_original_labels"),
                    "main_predicted_probabilities_global": subject_results.get("pp_ap_main_pred_probas_global"),
                    "main_predicted_labels_global": subject_results.get("pp_ap_main_pred_labels_global"),
                    "main_cross_validation_global_scores": subject_results.get("pp_ap_main_cv_global_scores"),
                    "main_temporal_scores_1d_all_folds": subject_results.get("pp_ap_main_scores_1d_all_folds"),
                    "main_mean_temporal_decoding_scores_1d": subject_results.get("pp_ap_main_scores_1d_mean"),
                    "main_temporal_1d_fdr_sig_data": subject_results.get("pp_ap_main_temporal_1d_fdr"),
                    "main_temporal_1d_cluster_sig_data": subject_results.get("pp_ap_main_temporal_1d_cluster"),
                    "main_mean_temporal_generalization_matrix_scores": subject_results.get("pp_ap_main_tgm_mean"),
                    "main_tgm_fdr_sig_data": subject_results.get("pp_ap_main_tgm_fdr"),
                    "main_decoding_global_metrics_for_plot": subject_results.get("pp_ap_main_global_metrics", {}),

                    "specific_ap_decoding_results": subject_results.get("pp_ap_specific_ap_results"),
                    "mean_of_specific_scores_1d": subject_results.get("pp_ap_mean_of_specific_scores_1d"),
                    "sem_of_specific_scores_1d": subject_results.get("pp_ap_sem_of_specific_scores_1d"),
                    "mean_specific_fdr_sig_data": subject_results.get("pp_ap_mean_specific_fdr"),
                    "mean_specific_cluster_sig_data": subject_results.get("pp_ap_mean_specific_cluster"),
                    "ap_vs_ap_decoding_results": subject_results.get("pp_ap_ap_vs_ap_results"),
                    "ap_centric_average_results_list": subject_results.get("pp_ap_ap_centric_avg_results"),
                }
                create_subject_decoding_dashboard_plots(**dashboard_plot_args)
                logger_run_one.info(
                    "Dashboard plots generated for subject %s in %s", subject_identifier, subject_results_dir)
            except Exception as e_plot:
                logger_run_one.error(
                    "Failed to generate dashboard plots for subject %s: %s", subject_identifier, e_plot, exc_info=True)
        else:
            if generate_plots_flag:
                logger_run_one.warning(
                    "Dashboard plot generation skipped for %s (missing results dir).", subject_identifier)
    elif generate_plots_flag:
        logger_run_one.warning(
            "Dashboard plot generation skipped for %s (missing 'epochs_time_points').", subject_identifier)


    logger_run_one.info("Finished processing subject %s (Task Set ID: %s). Total time: %.2fs",
                        subject_identifier, decoding_protocol_identifier, time.time() - total_start_time)
    return subject_results


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="EEG Single Subject Decoding Analysis Script",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli_parser.add_argument("--subject_id", type=str,
                            required=True, help="Subject ID for analysis.")
    cli_parser.add_argument("--group", type=str, required=False, default=None,
                            help="Group affiliation (optional, will try to resolve).")
    cli_parser.add_argument("--clf_type_override", type=str, default=None, choices=["svc", "logreg", "rf"],
                            help="Override default classifier type from config.")
    cli_parser.add_argument("--n_jobs_override", type=str, default=None,
                            help="Override n_jobs from config (e.g., '4' or 'auto').")


    command_line_args = cli_parser.parse_args()


    n_jobs_arg_str = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use = -1 if n_jobs_arg_str.lower() == "auto" else int(n_jobs_arg_str)
    except ValueError:
        logger_run_one.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str}'). Using default from config: {N_JOBS_PROCESSING} (becomes -1 if 'auto').")
        n_jobs_to_use = -1 if N_JOBS_PROCESSING.lower() == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login = getuser()
    main_input_path, main_output_path = configure_project_paths(user_login)

    logger_run_one.info("\n%s EEG SINGLE SUBJECT DECODING SCRIPT STARTED (%s) %s",
                        "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger_run_one.info("User: %s, Subject ID: %s",
                        user_login, command_line_args.subject_id)
    logger_run_one.info("  Classifier: %s, n_jobs (main ops): %s",
                        classifier_type_to_use, n_jobs_to_use)
    logger_run_one.info(
        "  GridSearch Optimization (from config): %s", USE_GRID_SEARCH_OPTIMIZATION)
    logger_run_one.info(
        "  CSP for Temporal Pipelines (from config): %s", USE_CSP_FOR_TEMPORAL_PIPELINES)
    logger_run_one.info("  ANOVA FS for Temporal Pipelines (from config): %s",
                        USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES)


    resolved_group_affiliation = command_line_args.group
    if not resolved_group_affiliation:
        resolved_group_affiliation = "unknown"  # Default
        for grp, s_list in ALL_SUBJECT_GROUPS.items():
            if command_line_args.subject_id in s_list:
                resolved_group_affiliation = grp
                logger_run_one.info(
                    f"Subject ID '{command_line_args.subject_id}' found in group '{grp}'.")
                break
        if resolved_group_affiliation == "unknown":
            logger_run_one.warning(
                f"Subject ID '{command_line_args.subject_id}' not found in any predefined group. Using affiliation 'unknown'.")

    execute_single_subject_decoding(
        subject_identifier=command_line_args.subject_id,
        group_affiliation=resolved_group_affiliation,
        base_input_data_path=main_input_path,
        base_output_results_path=main_output_path,
        n_jobs_for_processing=n_jobs_to_use, 
        classifier_type=classifier_type_to_use, 

    )

    logger_run_one.info("\n%s EEG SINGLE SUBJECT DECODING SCRIPT FINISHED (%s) %s",
                        "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
