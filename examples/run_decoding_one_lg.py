


import sys
import os
import logging
import time
import argparse
from getpass import getuser
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
import itertools
import scipy.stats


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from Baking_EEG._4_decoding_core import run_temporal_decoding_analysis
from utils.vizualization_utils_LG import create_subject_decoding_dashboard_plots_lg
from utils.utils import (
    configure_project_paths, setup_analysis_results_directory
)
from utils.loading_LG_utils import (
    load_epochs_data_for_lg_decoding,
   
)
from utils import stats_utils as bEEG_stats
from config.config import ALL_SUBJECT_GROUPS
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    CHANCE_LEVEL_AUC, INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_INTRA_SUBJECT_STATISTICS, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    COMPUTE_TGM_FOR_MAIN_COMPARISON, COMPUTE_TGM_FOR_SPECIFIC_COMPARISONS,
    COMPUTE_TGM_FOR_INTER_FAMILY_COMPARISONS,
    CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG, SAVE_ANALYSIS_RESULTS,
    GENERATE_PLOTS, N_JOBS_PROCESSING
)

LOG_DIR = './logs_run_single_subject_lg'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILENAME = os.path.join(
    LOG_DIR,
    datetime.now().strftime('log_run_single_subject_lg_%Y-%m-%d_%H%M%S.log')
)


for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


logging.getLogger("Baking_EEG.decoding_core").setLevel(logging.INFO)
logging.getLogger("Baking_EEG.utils.data_loading_utils").setLevel(logging.INFO)


def execute_single_subject_lg_decoding(
    subject_identifier,
    group_affiliation,
    decoding_protocol_identifier="Single_LG_Protocol_Analysis",
    save_results_flag=None,
    enable_verbose_logging=False,
    generate_plots_flag=None,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_processing=None,
    classifier_type=None,
    use_grid_search_for_subject=None,
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
$
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

    use_csp_for_temporal_subject = False  
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
        compute_tgm_flag = COMPUTE_TGM_FOR_MAIN_COMPARISON  
    if loading_conditions_config is None:
        loading_conditions_config = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG
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
        # Main LG decoding results (LS vs LD)
        "lg_ls_ld_original_labels": None,
        "lg_ls_ld_pred_probas_global": None,
        "lg_ls_ld_pred_labels_global": None,
        "lg_ls_ld_cv_global_scores": None,
        "lg_ls_ld_scores_1d_all_folds": None,
        "lg_ls_ld_scores_1d_mean": None,
        "lg_ls_ld_temporal_1d_fdr": None,
        "lg_ls_ld_temporal_1d_cluster": None,
        "lg_ls_ld_tgm_all_folds": None,
        "lg_ls_ld_tgm_mean": None,
        "lg_ls_ld_tgm_fdr": None,
        "lg_ls_ld_mean_auc_global": np.nan,
        "lg_ls_ld_global_metrics": {},
        # Global Standard vs Global Deviant decoding results (GS vs GD)
        "lg_gs_gd_original_labels": None,
        "lg_gs_gd_pred_probas_global": None,
        "lg_gs_gd_pred_labels_global": None,
        "lg_gs_gd_cv_global_scores": None,
        "lg_gs_gd_scores_1d_all_folds": None,
        "lg_gs_gd_scores_1d_mean": None,
        "lg_gs_gd_temporal_1d_fdr": None,
        "lg_gs_gd_temporal_1d_cluster": None,
        "lg_gs_gd_tgm_all_folds": None,
        "lg_gs_gd_tgm_mean": None,
        "lg_gs_gd_tgm_fdr": None,
        "lg_gs_gd_mean_auc_global": np.nan,
        "lg_gs_gd_global_metrics": {},
       
        "lg_specific_comparison_results": [],
        "lg_lsgs_vs_lsgd_scores_1d_mean": None,
        "lg_lsgs_vs_lsgd_temporal_1d_fdr": None,
        "lg_lsgs_vs_lsgd_temporal_1d_cluster": None,
        "lg_ldgs_vs_ldgd_scores_1d_mean": None,
        "lg_ldgs_vs_ldgd_temporal_1d_fdr": None,
        "lg_ldgs_vs_ldgd_temporal_1d_cluster": None,
        "lg_lsgs_vs_ldgs_scores_1d_mean": None,
        "lg_lsgs_vs_ldgs_temporal_1d_fdr": None,
        "lg_lsgs_vs_ldgs_temporal_1d_cluster": None,
        "lg_lsgd_vs_ldgd_scores_1d_mean": None,
        "lg_lsgd_vs_ldgd_temporal_1d_fdr": None,
        "lg_lsgd_vs_ldgd_temporal_1d_cluster": None,
      
        "lg_mean_of_specific_scores_1d": None,
        "lg_sem_of_specific_scores_1d": None,
        "lg_mean_specific_fdr": None,
        "lg_mean_specific_cluster": None,
      
        "lg_global_effect_results": [],

        "lg_local_effect_centric_avg_results": [],
    }

    # Convert n_jobs_for_processing if "auto"
    if (isinstance(n_jobs_for_processing, str) and
            n_jobs_for_processing.lower() == "auto"):
        actual_n_jobs = -1
    else:
        try:
            actual_n_jobs = int(n_jobs_for_processing)
        except ValueError:
            logger.warning(
                "Invalid n_jobs_for_processing %s, using -1.",
                n_jobs_for_processing)
            actual_n_jobs = -1

    try:
        if not base_input_data_path or not base_output_results_path:
            current_user = getuser()
            cfg_input, cfg_output = configure_project_paths(current_user)
            base_input_data_path = base_input_data_path or cfg_input
            base_output_results_path = base_output_results_path or cfg_output

        logger.info(
            "Starting LG decoding for subject: %s (Group: %s, "
            "Task Set ID: %s, Classifier: %s, GS: %s, "
            "ANOVA FS: %s, n_jobs: %s)",
            subject_identifier, group_affiliation,
            decoding_protocol_identifier, classifier_type,
            use_grid_search_for_subject,
            use_anova_fs_for_temporal_subject, actual_n_jobs
        )

        actual_loading_conditions = loading_conditions_config
        logger.info("Using LG loading conditions: %s",
                    list(actual_loading_conditions.keys()))

        epochs_object, returned_data_dict, detected_protocol =  load_epochs_data_for_lg_decoding(
            subject_identifier, group_affiliation, base_input_data_path,
            actual_loading_conditions, enable_verbose_logging
        )

    
        logger.info(
            "Detected protocol '%s' for subject %s",
            detected_protocol, subject_identifier
        )

        if epochs_object is None:
            logger.error(
                "Epochs object loading failed for %s. Aborting subject.",
                subject_identifier)
            return subject_results

        subject_results["epochs_time_points"] = epochs_object.times.copy()
        subject_results["detected_protocol"] = detected_protocol


        current_fixed_params_for_clf_dict = None
        current_param_grid_for_clf_dict = None

        if use_grid_search_for_subject:
            if (param_grid_config_for_subject and
                    classifier_type in param_grid_config_for_subject):
                current_param_grid_for_clf_dict = param_grid_config_for_subject
            else:
                logger.warning(
                    "GS for %s, but no grid for '%s' in "
                    "param_grid_config_for_subject. Core decoding "
                    "defaults will be used.",
                    subject_identifier, classifier_type)
        else:
            if (fixed_params_for_subject and
                    classifier_type in fixed_params_for_subject):
                current_fixed_params_for_clf_dict = (
                    fixed_params_for_subject[classifier_type])
            else:
                logger.warning(
                    "No GS for %s, no fixed params for '%s' in "
                    "fixed_params_for_subject. Core decoding "
                    "defaults will be used.",
                    subject_identifier, classifier_type)


        logger.info("--- Starting Main LG Protocol Decoding for %s ---",
                    subject_identifier)

        logger.info("  --- 1. Main LG Decoding (LS vs LD) for %s ---",
                    subject_identifier)

        ls_data = returned_data_dict.get("LS_ALL")
        ld_data = returned_data_dict.get("LD_ALL")

        if (ls_data is not None and ld_data is not None and
                ls_data.size > 0 and ld_data.size > 0):

            ls_ld_data = np.concatenate([ls_data, ld_data], axis=0)
            ls_ld_labels_orig = np.concatenate([
                np.zeros(ls_data.shape[0]),
                np.ones(ld_data.shape[0])
            ])
            subject_results["lg_ls_ld_original_labels"] = (
                ls_ld_labels_orig.copy())

            main_labels_encoded = LabelEncoder().fit_transform(
                ls_ld_labels_orig)

            if len(np.unique(main_labels_encoded)) < 2:
                logger.error(
                    "Subj %s: Only one class for main LG decoding. Skipping.",
                    subject_identifier)
            else:
                min_samples_main = np.min(np.bincount(main_labels_encoded))
                num_cv_splits_main = (
                    min(10, min_samples_main)
                    if min_samples_main >= 2 else 0)

                if num_cv_splits_main < 2:
                    logger.error(
                        "Subj %s: Not enough samples for CV in main LG "
                        "decoding (%d splits). Skipping.",
                        subject_identifier, num_cv_splits_main)
                    return subject_results
                else:
                    cv_splitter_main = RepeatedStratifiedKFold(
                        n_splits=num_cv_splits_main, n_repeats=3,
                        random_state=42)

                    ls_ld_decoding_output = run_temporal_decoding_analysis(
                        epochs_data=ls_ld_data,
                        target_labels=ls_ld_labels_orig,
                        classifier_model_type=classifier_type,
                        use_grid_search=use_grid_search_for_subject,
                        use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                        param_grid_config=current_param_grid_for_clf_dict,
                        cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                        fixed_classifier_params=(
                            current_fixed_params_for_clf_dict),
                        cross_validation_splitter=cv_splitter_main,
                        n_jobs_external=actual_n_jobs,
                        compute_intra_fold_stats=(
                            compute_intra_subject_stats_flag),
                        n_permutations_for_intra_fold_clusters=(
                            n_perms_for_intra_subject_clusters),
                        compute_temporal_generalization_matrix=(
                            compute_tgm_flag),
                        chance_level=CHANCE_LEVEL_AUC,
                        cluster_threshold_config_intra_fold=(
                            cluster_threshold_config_intra_fold)
                    )

                    subject_results.update({
                        "lg_ls_ld_pred_probas_global": ls_ld_decoding_output[0],
                        "lg_ls_ld_pred_labels_global": ls_ld_decoding_output[1],
                        "lg_ls_ld_cv_global_scores": ls_ld_decoding_output[2],
                        "lg_ls_ld_scores_1d_mean": ls_ld_decoding_output[3],
                        "lg_ls_ld_global_metrics": ls_ld_decoding_output[4],
                        "lg_ls_ld_temporal_1d_fdr": ls_ld_decoding_output[5],
                        "lg_ls_ld_temporal_1d_cluster": ls_ld_decoding_output[6],
                        "lg_ls_ld_scores_1d_all_folds": ls_ld_decoding_output[7],
                        "lg_ls_ld_tgm_mean": ls_ld_decoding_output[8],
                        "lg_ls_ld_tgm_fdr": ls_ld_decoding_output[9],
                        "lg_ls_ld_tgm_all_folds": ls_ld_decoding_output[11],
                    })

                    mean_auc_val = (
                        np.nanmean(ls_ld_decoding_output[2])
                        if (ls_ld_decoding_output[2] is not None and
                            ls_ld_decoding_output[2].size > 0)
                        else np.nan)

                    subject_results["lg_ls_ld_mean_auc_global"] = mean_auc_val

                    logger.info(
                        "Main LG Decoding for %s DONE. Mean Global AUC: %.3f",
                        subject_identifier,
                        mean_auc_val if pd.notna(mean_auc_val) else -1)
        else:
            logger.warning(
                "Subj %s: Missing LS_ALL or LD_ALL data. "
                "Skipping main LG decoding.", subject_identifier)

        # === 1.2. GS vs GD Decoding ===
        logger.info("  --- 1.2. GS vs GD Decoding for %s ---", subject_identifier)

        gs_data = returned_data_dict.get("GS_ALL")
        gd_data = returned_data_dict.get("GD_ALL")

        if (gs_data is not None and gd_data is not None and
                gs_data.size > 0 and gd_data.size > 0):

            gs_gd_data = np.concatenate([gs_data, gd_data], axis=0)
            gs_gd_labels_orig = np.concatenate([
                np.zeros(gs_data.shape[0]),
                np.ones(gd_data.shape[0])
            ])
            subject_results["lg_gs_gd_original_labels"] = (
                gs_gd_labels_orig.copy())

            gs_gd_labels_encoded = LabelEncoder().fit_transform(
                gs_gd_labels_orig)

            if len(np.unique(gs_gd_labels_encoded)) < 2:
                logger.error(
                    "Subj %s: Only one class for GS vs GD decoding. Skipping.",
                    subject_identifier)
            else:
                min_samples_gs_gd = np.min(np.bincount(gs_gd_labels_encoded))
                num_cv_splits_gs_gd = (
                    min(10, min_samples_gs_gd)
                    if min_samples_gs_gd >= 2 else 0)

                if num_cv_splits_gs_gd < 2:
                    logger.error(
                        "Subj %s: Not enough samples for CV in GS vs GD "
                        "decoding (%d splits). Skipping.",
                        subject_identifier, num_cv_splits_gs_gd)
                else:
                    cv_splitter_gs_gd = RepeatedStratifiedKFold(
                        n_splits=num_cv_splits_gs_gd, n_repeats=3,
                        random_state=42)

                    gs_gd_decoding_output = run_temporal_decoding_analysis(
                        epochs_data=gs_gd_data,
                        target_labels=gs_gd_labels_orig,
                        classifier_model_type=classifier_type,
                        use_grid_search=use_grid_search_for_subject,
                        use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                        param_grid_config=current_param_grid_for_clf_dict,
                        cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                        fixed_classifier_params=(
                            current_fixed_params_for_clf_dict),
                        cross_validation_splitter=cv_splitter_gs_gd,
                        n_jobs_external=actual_n_jobs,
                        compute_intra_fold_stats=(
                            compute_intra_subject_stats_flag),
                        n_permutations_for_intra_fold_clusters=(
                            n_perms_for_intra_subject_clusters),
                        compute_temporal_generalization_matrix=(
                            compute_tgm_flag),
                        chance_level=CHANCE_LEVEL_AUC,
                        cluster_threshold_config_intra_fold=(
                            cluster_threshold_config_intra_fold)
                    )

                    subject_results.update({
                        "lg_gs_gd_pred_probas_global": gs_gd_decoding_output[0],
                        "lg_gs_gd_pred_labels_global": gs_gd_decoding_output[1],
                        "lg_gs_gd_cv_global_scores": gs_gd_decoding_output[2],
                        "lg_gs_gd_scores_1d_mean": gs_gd_decoding_output[3],
                        "lg_gs_gd_global_metrics": gs_gd_decoding_output[4],
                        "lg_gs_gd_temporal_1d_fdr": gs_gd_decoding_output[5],
                        "lg_gs_gd_temporal_1d_cluster": gs_gd_decoding_output[6],
                        "lg_gs_gd_scores_1d_all_folds": gs_gd_decoding_output[7],
                        "lg_gs_gd_tgm_mean": gs_gd_decoding_output[8],
                        "lg_gs_gd_tgm_fdr": gs_gd_decoding_output[9],
                        "lg_gs_gd_tgm_all_folds": gs_gd_decoding_output[11],
                    })

                    mean_auc_val_gs_gd = (
                        np.nanmean(gs_gd_decoding_output[2])
                        if (gs_gd_decoding_output[2] is not None and
                            gs_gd_decoding_output[2].size > 0)
                        else np.nan)

                    subject_results["lg_gs_gd_mean_auc_global"] = mean_auc_val_gs_gd

                    logger.info(
                        "GS vs GD Decoding for %s DONE. Mean Global AUC: %.3f",
                        subject_identifier,
                        mean_auc_val_gs_gd if pd.notna(mean_auc_val_gs_gd) else -1)
        else:
            logger.warning(
                "Subj %s: Missing GS_ALL or GD_ALL data. "
                "Skipping GS vs GD decoding.", subject_identifier)

        logger.info(
            "  --- 2. Specific LG Comparisons (e.g. LS/GS vs LS/GD, "
            "LD/GS vs LD/GD) for %s ---", subject_identifier)

        current_time_points_ref = subject_results.get("epochs_time_points")
        if current_time_points_ref is None:
            logger.error(
                "Time points not available for %s, cannot run "
                "specific LG tasks.", subject_identifier)
        else:
            # Define specific LG comparisons
            lg_specific_comparisons = [
                ("LSGS", "LSGD",
                 "Local Standard: Global Standard vs Global Deviant",
                 "lsgs_vs_lsgd"),
                ("LDGS", "LDGD",
                 "Local Deviant: Global Standard vs Global Deviant",
                 "ldgs_vs_ldgd"),
                ("LSGS", "LDGS",
                 "Global Standard: Local Standard vs Local Deviant",
                 "lsgs_vs_ldgs"),
                ("LSGD", "LDGD",
                 "Global Deviant: Local Standard vs Local Deviant",
                 "lsgd_vs_ldgd"),
            ]

            subject_results["lg_specific_comparison_results"] = []

            for condition1_key, condition2_key, comparison_name, result_key in (
                    lg_specific_comparisons):

                condition1_data = returned_data_dict.get(condition1_key)
                condition2_data = returned_data_dict.get(condition2_key)

                task_result_specific = {
                    "comparison_name": comparison_name,
                    "scores_1d_mean": None,
                    "all_folds_scores_1d": None,
                    "times": current_time_points_ref.copy(),
                    "fdr_significance_data": None,
                    "cluster_significance_data": None
                }

                if (condition1_data is not None and
                        condition2_data is not None and
                        condition1_data.size > 0 and condition2_data.size > 0):

                    task_data_specific_current = np.concatenate(
                        [condition1_data, condition2_data], axis=0)
                    task_labels_specific_orig = np.concatenate([
                        np.zeros(condition1_data.shape[0]),
                        np.ones(condition2_data.shape[0])
                    ])
                    task_labels_specific_enc = (
                        LabelEncoder().fit_transform(task_labels_specific_orig))

                    if len(np.unique(task_labels_specific_enc)) < 2:
                        logger.warning(
                            "Subj %s, Task '%s': Only one class. Skipping.",
                            subject_identifier, comparison_name)
                    else:
                        min_samples_task_spec = np.min(
                            np.bincount(task_labels_specific_enc))
                        num_cv_task_spec = (
                            min(10, min_samples_task_spec)
                            if min_samples_task_spec >= 2 else 0)

                        if num_cv_task_spec < 2:
                            logger.warning(
                                "Subj %s, Task '%s': Not enough samples for "
                                "CV (%d splits). Skipping.",
                                subject_identifier, comparison_name,
                                num_cv_task_spec)
                        else:
                            cv_splitter_task_spec = RepeatedStratifiedKFold(
                                n_splits=num_cv_task_spec, n_repeats=3,
                                random_state=42)

                            specific_task_output = (
                                run_temporal_decoding_analysis(
                                    epochs_data=task_data_specific_current,
                                    target_labels=task_labels_specific_orig,
                                    classifier_model_type=classifier_type,
                                    use_grid_search=use_grid_search_for_subject,
                                    use_anova_fs_for_temporal_pipelines=use_anova_fs_for_temporal_subject,
                                    param_grid_config=current_param_grid_for_clf_dict,
                                    cv_folds_for_gridsearch=cv_folds_for_gs_subject,
                                    fixed_classifier_params=current_fixed_params_for_clf_dict,
                                    cross_validation_splitter=cv_splitter_task_spec,
                                    n_jobs_external=actual_n_jobs,
                                    compute_intra_fold_stats=(
                                        compute_intra_subject_stats_flag),
                                    n_permutations_for_intra_fold_clusters=(
                                        n_perms_for_intra_subject_clusters),
                                    compute_temporal_generalization_matrix=compute_tgm_flag,
                                    chance_level=CHANCE_LEVEL_AUC,
                                    cluster_threshold_config_intra_fold=(
                                        cluster_threshold_config_intra_fold)
                                ))

                            task_result_specific.update({
                                "scores_1d_mean": specific_task_output[3],
                                "fdr_significance_data": (
                                    specific_task_output[5]),
                                "cluster_significance_data": (
                                    specific_task_output[6]),
                                "all_folds_scores_1d": specific_task_output[7]
                            })

                            # Save results with individual keys for each comparison
                            subject_results[f"lg_{result_key}_scores_1d_mean"] = specific_task_output[3]
                            subject_results[f"lg_{result_key}_temporal_1d_fdr"] = specific_task_output[5]
                            subject_results[f"lg_{result_key}_temporal_1d_cluster"] = specific_task_output[6]

                            peak_auc_val = (
                                np.nanmax(specific_task_output[3])
                                if (specific_task_output[3] is not None and
                                    specific_task_output[3].size > 0)
                                else np.nan)

                            logger.info(
                                "  Specific LG task '%s' for %s: "
                                "Peak AUC = %.3f",
                                comparison_name, subject_identifier,
                                peak_auc_val if pd.notna(peak_auc_val) else -1)
                else:
                    logger.info(
                        "Subj %s: Missing data for %s or %s in task '%s'. "
                        "This may be normal depending on subject protocol.",
                        subject_identifier, condition1_key, condition2_key,
                        comparison_name)

                subject_results["lg_specific_comparison_results"].append(
                    task_result_specific)

        logger.info("  --- Specific LG Comparisons for %s DONE ---",
                    subject_identifier)

        # Calculate stats on stack of specific LG curves
        if (compute_intra_subject_stats_flag and
                subject_results.get("lg_specific_comparison_results")):

            logger.info(
                "  --- 3. Stats on stack of Specific LG Curves for %s ---",
                subject_identifier)

            valid_mean_scores_for_stack = [
                res["scores_1d_mean"]
                for res in subject_results["lg_specific_comparison_results"]
                if (res.get("scores_1d_mean") is not None and
                    current_time_points_ref is not None and
                    res["scores_1d_mean"].shape == current_time_points_ref.shape)
            ]

            if len(valid_mean_scores_for_stack) >= 2:
                stacked_specific_curves = np.array(valid_mean_scores_for_stack)
                subject_results["lg_mean_of_specific_scores_1d"] = (
                    np.nanmean(stacked_specific_curves, axis=0))

                if stacked_specific_curves.shape[0] > 1:
                    subject_results["lg_sem_of_specific_scores_1d"] = (
                        scipy.stats.sem(stacked_specific_curves, axis=0,
                                        nan_policy='omit'))

                _, fdr_mask_stack, fdr_pval_stack, fdr_test_info_stack = (
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        stacked_specific_curves, CHANCE_LEVEL_AUC,
                        alternative_hypothesis="greater"))

                subject_results["lg_mean_specific_fdr"] = {
                    "mask": fdr_mask_stack, "p_values": fdr_pval_stack,
                    "method": (f"FDR on stack of "
                               f"{len(valid_mean_scores_for_stack)} "
                               f"specific LG curves")}

                _, clu_obj_stack, p_clu_stack, _ = (
                    bEEG_stats.perform_cluster_permutation_test(
                        stacked_specific_curves, CHANCE_LEVEL_AUC,
                        n_perms_for_intra_subject_clusters,
                        cluster_threshold_config_intra_fold,
                        "greater", actual_n_jobs))

                combined_mask_clu_stack = (
                    np.zeros_like(
                        subject_results["lg_mean_of_specific_scores_1d"],
                        dtype=bool)
                    if subject_results["lg_mean_of_specific_scores_1d"] is not None
                    else np.array([], dtype=bool))

                sig_clu_objects_stack = []
                if (clu_obj_stack and p_clu_stack is not None and
                        combined_mask_clu_stack.size > 0):
                    for i_c, c_mask_item_stack in enumerate(clu_obj_stack):
                        if p_clu_stack[i_c] < 0.05:
                            sig_clu_objects_stack.append(c_mask_item_stack)
                            combined_mask_clu_stack = np.logical_or(
                                combined_mask_clu_stack, c_mask_item_stack)

                subject_results["lg_mean_specific_cluster"] = {
                    "mask": combined_mask_clu_stack,
                    "cluster_objects": sig_clu_objects_stack,
                    "p_values_all_clusters": p_clu_stack,
                    "method": (f"CluPerm on stack of "
                               f"{len(valid_mean_scores_for_stack)} "
                               f"specific LG curves")}

                logger.info(
                    "  --- Stats on stack of Specific LG Curves "
                    "for %s DONE ---", subject_identifier)
            else:
                logger.warning(
                    "Subj %s: Not enough valid specific LG curves (%d) "
                    "for stack statistics.",
                    subject_identifier, len(valid_mean_scores_for_stack))

        # Save results and generate plots
        if save_results_flag or generate_plots_flag:
            try:
                dec_prot_id_str = str(
                    decoding_protocol_identifier
                    if decoding_protocol_identifier else "UnknownProtocolID")
                subfolder_name_components = [
                    subject_identifier,
                    dec_prot_id_str.replace(" ", "_").replace("/", "-"),
                    classifier_type]
                valid_subfolder_components = [
                    comp for comp in subfolder_name_components if comp]
                subfolder_name_for_setup = "_".join(valid_subfolder_components)

                # Create group_protocol path for better organization
                detected_protocol = subject_results.get("detected_protocol", "unknown")
                group_protocol_path = f"{group_affiliation}_{detected_protocol}"

                subject_results_dir = setup_analysis_results_directory(
                    base_output_results_path, "intra_subject_lg_results",
                    group_protocol_path, subfolder_name_for_setup)
            except Exception as e_setup_dir:
                logger.error(
                    "Failed to setup results directory for %s: %s. "
                    "Plots/saving skipped.",
                    subject_identifier, e_setup_dir, exc_info=True)
                subject_results_dir = None

        if save_results_flag and subject_results_dir:
            try:
                results_file_path = os.path.join(
                    subject_results_dir, "lg_decoding_results_full.npz")
                np.savez_compressed(results_file_path, **subject_results)

                csv_summary_data = {
                    "subject_id": subject_identifier,
                    "group": group_affiliation,
                    "protocol_task_id": decoding_protocol_identifier,
                    "classifier": classifier_type}
                csv_summary_data["lg_ls_ld_global_auc_mean"] = (
                    subject_results.get("lg_ls_ld_mean_auc_global", np.nan))

                cv_scores_main = subject_results.get(
                    "lg_ls_ld_cv_global_scores")
                csv_summary_data["lg_ls_ld_global_auc_std"] = (
                    np.nanstd(cv_scores_main)
                    if (cv_scores_main is not None and
                        cv_scores_main.size > 0) else np.nan)

                # Add GS vs GD metrics
                csv_summary_data["lg_gs_gd_global_auc_mean"] = (
                    subject_results.get("lg_gs_gd_mean_auc_global", np.nan))

                cv_scores_gs_gd = subject_results.get(
                    "lg_gs_gd_cv_global_scores")
                csv_summary_data["lg_gs_gd_global_auc_std"] = (
                    np.nanstd(cv_scores_gs_gd)
                    if (cv_scores_gs_gd is not None and
                        cv_scores_gs_gd.size > 0) else np.nan)

                main_metrics = subject_results.get(
                    "lg_ls_ld_global_metrics", {})
                for k_metric, v_metric in main_metrics.items():
                    csv_summary_data[f"lg_ls_ld_metric_{k_metric}"] = v_metric

                # Add GS vs GD metrics
                gs_gd_metrics = subject_results.get(
                    "lg_gs_gd_global_metrics", {})
                for k_metric, v_metric in gs_gd_metrics.items():
                    csv_summary_data[f"lg_gs_gd_metric_{k_metric}"] = v_metric

                # Add peak AUC for each specific comparison
                specific_comparisons_keys = [
                    "lsgs_vs_lsgd", "ldgs_vs_ldgd", "lsgs_vs_ldgs", "lsgd_vs_ldgd"
                ]
                for comp_key in specific_comparisons_keys:
                    scores_key = f"lg_{comp_key}_scores_1d_mean"
                    if scores_key in subject_results:
                        scores = subject_results[scores_key]
                        if scores is not None and len(scores) > 0:
                            peak_auc = np.nanmax(scores)
                            csv_summary_data[f"lg_{comp_key}_peak_auc"] = peak_auc
                        else:
                            csv_summary_data[f"lg_{comp_key}_peak_auc"] = np.nan
                    else:
                        csv_summary_data[f"lg_{comp_key}_peak_auc"] = np.nan

                summary_csv_path = os.path.join(
                    subject_results_dir, "lg_summary_metrics.csv")
                pd.DataFrame([csv_summary_data]).to_csv(
                    summary_csv_path, index=False)

                logger.info("LG Results saved for subject %s in %s",
                            subject_identifier, subject_results_dir)
            except Exception as e_save:
                logger.error(
                    "Failed to save LG results for %s to %s: %s",
                    subject_identifier, subject_results_dir, e_save,
                    exc_info=True)

        if (generate_plots_flag and
                subject_results.get("epochs_time_points") is not None):
            if subject_results_dir:
                try:
                    dashboard_plot_args_lg = {
                        "main_epochs_time_points": (
                            subject_results.get("epochs_time_points")),
                        "classifier_name_for_title": classifier_type,
                        "subject_identifier": subject_identifier,
                        "group_identifier": group_affiliation,
                        "output_directory_path": subject_results_dir,
                        "CHANCE_LEVEL_AUC": CHANCE_LEVEL_AUC,
                        "protocol_type": "LG",

                        "lg_ls_ld_original_labels_array": (
                            subject_results.get("lg_ls_ld_original_labels")),
                        "lg_ls_ld_predicted_probabilities_global": (
                            subject_results.get("lg_ls_ld_pred_probas_global")),
                        "lg_ls_ld_predicted_labels_global": (
                            subject_results.get("lg_ls_ld_pred_labels_global")),
                        "lg_ls_ld_cross_validation_global_scores": (
                            subject_results.get("lg_ls_ld_cv_global_scores")),
                        "lg_ls_ld_temporal_scores_1d_all_folds": (
                            subject_results.get("lg_ls_ld_scores_1d_all_folds")),
                        "lg_ls_ld_mean_temporal_decoding_scores_1d": (
                            subject_results.get("lg_ls_ld_scores_1d_mean")),
                        "lg_ls_ld_temporal_1d_fdr_sig_data": (
                            subject_results.get("lg_ls_ld_temporal_1d_fdr")),
                        "lg_ls_ld_temporal_1d_cluster_sig_data": (
                            subject_results.get("lg_ls_ld_temporal_1d_cluster")),
                        "lg_ls_ld_mean_temporal_generalization_matrix_scores": (
                            subject_results.get("lg_ls_ld_tgm_mean")),
                        "lg_ls_ld_tgm_fdr_sig_data": (
                            subject_results.get("lg_ls_ld_tgm_fdr")),
                        "lg_ls_ld_decoding_global_metrics_for_plot": (
                            subject_results.get("lg_ls_ld_global_metrics", {})),

                        # GS vs GD decoding results
                        "lg_gs_gd_original_labels_array": (
                            subject_results.get("lg_gs_gd_original_labels")),
                        "lg_gs_gd_predicted_probabilities_global": (
                            subject_results.get("lg_gs_gd_pred_probas_global")),
                        "lg_gs_gd_predicted_labels_global": (
                            subject_results.get("lg_gs_gd_pred_labels_global")),
                        "lg_gs_gd_cross_validation_global_scores": (
                            subject_results.get("lg_gs_gd_cv_global_scores")),
                        "lg_gs_gd_temporal_scores_1d_all_folds": (
                            subject_results.get("lg_gs_gd_scores_1d_all_folds")),
                        "lg_gs_gd_mean_temporal_decoding_scores_1d": (
                            subject_results.get("lg_gs_gd_scores_1d_mean")),
                        "lg_gs_gd_temporal_1d_fdr_sig_data": (
                            subject_results.get("lg_gs_gd_temporal_1d_fdr")),
                        "lg_gs_gd_temporal_1d_cluster_sig_data": (
                            subject_results.get("lg_gs_gd_temporal_1d_cluster")),
                        "lg_gs_gd_mean_temporal_generalization_matrix_scores": (
                            subject_results.get("lg_gs_gd_tgm_mean")),
                        "lg_gs_gd_tgm_fdr_sig_data": (
                            subject_results.get("lg_gs_gd_tgm_fdr")),
                        "lg_gs_gd_decoding_global_metrics_for_plot": (
                            subject_results.get("lg_gs_gd_global_metrics", {})),

                        # Individual comparison results
                        "lg_lsgs_vs_lsgd_scores_1d_mean": (
                            subject_results.get("lg_lsgs_vs_lsgd_scores_1d_mean")),
                        "lg_lsgs_vs_lsgd_temporal_1d_fdr": (
                            subject_results.get("lg_lsgs_vs_lsgd_temporal_1d_fdr")),
                        "lg_lsgs_vs_lsgd_temporal_1d_cluster": (
                            subject_results.get("lg_lsgs_vs_lsgd_temporal_1d_cluster")),
                        "lg_ldgs_vs_ldgd_scores_1d_mean": (
                            subject_results.get("lg_ldgs_vs_ldgd_scores_1d_mean")),
                        "lg_ldgs_vs_ldgd_temporal_1d_fdr": (
                            subject_results.get("lg_ldgs_vs_ldgd_temporal_1d_fdr")),
                        "lg_ldgs_vs_ldgd_temporal_1d_cluster": (
                            subject_results.get("lg_ldgs_vs_ldgd_temporal_1d_cluster")),
                        "lg_lsgs_vs_ldgs_scores_1d_mean": (
                            subject_results.get("lg_lsgs_vs_ldgs_scores_1d_mean")),
                        "lg_lsgs_vs_ldgs_temporal_1d_fdr": (
                            subject_results.get("lg_lsgs_vs_ldgs_temporal_1d_fdr")),
                        "lg_lsgs_vs_ldgs_temporal_1d_cluster": (
                            subject_results.get("lg_lsgs_vs_ldgs_temporal_1d_cluster")),
                        "lg_lsgd_vs_ldgd_scores_1d_mean": (
                            subject_results.get("lg_lsgd_vs_ldgd_scores_1d_mean")),
                        "lg_lsgd_vs_ldgd_temporal_1d_fdr": (
                            subject_results.get("lg_lsgd_vs_ldgd_temporal_1d_fdr")),
                        "lg_lsgd_vs_ldgd_temporal_1d_cluster": (
                            subject_results.get("lg_lsgd_vs_ldgd_temporal_1d_cluster")),

                        "lg_specific_comparison_results": (
                            subject_results.get("lg_specific_comparison_results")),
                        "lg_mean_of_specific_scores_1d": (
                            subject_results.get("lg_mean_of_specific_scores_1d")),
                        "lg_sem_of_specific_scores_1d": (
                            subject_results.get("lg_sem_of_specific_scores_1d")),
                        "lg_mean_specific_fdr_sig_data": (
                            subject_results.get("lg_mean_specific_fdr")),
                        "lg_mean_specific_cluster_sig_data": (
                            subject_results.get("lg_mean_specific_cluster")),
                        "lg_global_effect_results": (
                            subject_results.get("lg_global_effect_results")),
                        "lg_local_effect_centric_average_results_list": (
                            subject_results.get(
                                "lg_local_effect_centric_avg_results")),
                    }

                    create_subject_decoding_dashboard_plots_lg(
                        **dashboard_plot_args_lg)

                    logger.info(
                        "LG Dashboard plots generated for subject %s in %s",
                        subject_identifier, subject_results_dir)
                except Exception as e_plot:
                    logger.error(
                        "Failed to generate LG dashboard plots for "
                        "subject %s: %s",
                        subject_identifier, e_plot, exc_info=True)
            else:
                if generate_plots_flag:
                    logger.warning(
                        "LG Dashboard plot generation skipped for %s "
                        "(missing results dir).", subject_identifier)
        elif generate_plots_flag:
            logger.warning(
                "LG Dashboard plot generation skipped for %s "
                "(missing 'epochs_time_points').", subject_identifier)

    except FileNotFoundError as fnfe:
        logger.error(
            "FileNotFoundError for subject %s: %s.",
            subject_identifier, fnfe, exc_info=True)
        return subject_results
    except ValueError as ve:
        logger.error(
            "ValueError during processing for subject %s: %s.",
            subject_identifier, ve, exc_info=True)
        return subject_results
    except KeyError as ke:
        logger.error(
            "KeyError during processing for subject %s: %s.",
            subject_identifier, ke, exc_info=True)
        return subject_results
    except Exception as e:
        logger.error(
            "Unexpected error during main processing logic for subject %s: %s",
            subject_identifier, e, exc_info=True)
        return subject_results

    logger.info(
        "Finished processing LG for subject %s (Task Set ID: %s). "
        "Total time: %.2fs",
        subject_identifier, decoding_protocol_identifier,
        time.time() - total_start_time)

    return subject_results


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(
        description="EEG Single Subject LG Decoding Analysis Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cli_parser.add_argument(
        "--subject_id", type=str, required=True,
        help="Subject ID for LG analysis.")
    cli_parser.add_argument(
        "--group", type=str, required=False, default=None,
        help="Group affiliation (optional, will try to resolve).")
    cli_parser.add_argument(
        "--clf_type_override", type=str, default=None,
        choices=["svc", "logreg", "rf"],
        help="Override default classifier type from config.")
    cli_parser.add_argument(
        "--n_jobs_override", type=str, default=None,
        help="Override n_jobs from config (e.g., '4' or 'auto').")
    cli_parser.add_argument(
        "--no-tgm", action="store_true",
        help="Disable TGM computation (recommended for cluster to avoid timeouts).")

    command_line_args = cli_parser.parse_args()

    # Determine n_jobs
    n_jobs_arg_str = (command_line_args.n_jobs_override
                      if command_line_args.n_jobs_override is not None
                      else N_JOBS_PROCESSING)
    try:
        n_jobs_to_use = (-1 if n_jobs_arg_str.lower() == "auto"
                         else int(n_jobs_arg_str))
    except ValueError:
        logger.warning(
            "Invalid n_jobs_override ('%s'). Using default from config: "
            "%s (becomes -1 if 'auto').",
            n_jobs_arg_str, N_JOBS_PROCESSING)
        n_jobs_to_use = (-1 if N_JOBS_PROCESSING.lower() == "auto"
                         else int(N_JOBS_PROCESSING))

    classifier_type_to_use = (command_line_args.clf_type_override
                              if command_line_args.clf_type_override is not None
                              else CLASSIFIER_MODEL_TYPE)

    user_login = getuser()
    main_input_path, main_output_path = configure_project_paths(user_login)

    logger.info(
        "\n%s EEG SINGLE SUBJECT LG DECODING SCRIPT STARTED (%s) %s",
        "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger.info("User: %s, Subject ID: %s",
                user_login, command_line_args.subject_id)
    logger.info("  Classifier: %s, n_jobs (main ops): %s",
                classifier_type_to_use, n_jobs_to_use)
    logger.info("  GridSearch Optimization (from config): %s",
                USE_GRID_SEARCH_OPTIMIZATION)
    # CSP functionality has been removed from the pipeline
    logger.info("  ANOVA FS for Temporal Pipelines (from config): %s",
                USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES)

    # Resolve group affiliation if not provided
    resolved_group_affiliation = command_line_args.group
    if not resolved_group_affiliation:
        resolved_group_affiliation = "unknown"  # Default
        for grp, s_list in ALL_SUBJECT_GROUPS.items():
            if command_line_args.subject_id in s_list:
                resolved_group_affiliation = grp
                logger.info(
                    "Subject ID '%s' found in group '%s'.",
                    command_line_args.subject_id, grp)
                break
        if resolved_group_affiliation == "unknown":
            logger.warning(
                "Subject ID '%s' not found in any predefined group. "
                "Using affiliation 'unknown'.",
                command_line_args.subject_id)

    # Determine TGM flag
    compute_tgm_flag = not command_line_args.no_tgm
    
    if command_line_args.no_tgm:
        logger.info("TGM computation disabled via --no-tgm flag")
    else:
        logger.info("TGM computation enabled (default or via config)")

    # Call the orchestration function for a single LG subject
    execute_single_subject_lg_decoding(
        subject_identifier=command_line_args.subject_id,
        group_affiliation=resolved_group_affiliation,
        base_input_data_path=main_input_path,
        base_output_results_path=main_output_path,
        n_jobs_for_processing=n_jobs_to_use,
        classifier_type=classifier_type_to_use,
        compute_tgm_flag=compute_tgm_flag,
    )

    logger.info(
        "\n%s EEG SINGLE SUBJECT LG DECODING SCRIPT FINISHED (%s) %s",
        "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
