# Fichier : examples/run_decoding_one_group_pp.py
# python -m examples.run_decoding_one_group_pp --group_name controls
import os
import sys
import logging
import time
from datetime import datetime
import argparse
from getpass import getuser
import numpy as np
import pandas as pd
import scipy.stats

# --- Imports des modules du projet (standardisés) ---

from examples.run_decoding_one_pp import execute_single_subject_decoding

# Tous les autres imports partent de la racine (utils, config...)
from utils.utils import (
    configure_project_paths, setup_analysis_results_directory
)
from utils import stats_utils as bEEG_stats
from utils.vizualization_utils_PP import (
    plot_group_mean_scores_barplot,
    plot_group_temporal_decoding_statistics,
    plot_group_tgm_statistics
)

from config.config import ALL_SUBJECT_GROUPS
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING, CHANCE_LEVEL_AUC_SCORE,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING
)
# --- Configuration du Logging ---
LOG_DIR_RUN_GROUP = './logs_run_group_analysis'
os.makedirs(LOG_DIR_RUN_GROUP, exist_ok=True)
LOG_FILENAME_RUN_GROUP = os.path.join(
    LOG_DIR_RUN_GROUP,
    datetime.now().strftime('log_run_group_analysis_%Y-%m-%d_%H%M%S.log')
)

for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_GROUP),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_group = logging.getLogger(__name__)
logging.getLogger("examples.run_decoding_one_pp").setLevel(logging.INFO)
logging.getLogger("utils.stats_utils").setLevel(logging.INFO)
logging.getLogger("utils.vizualization_utils_PP").setLevel(logging.INFO)

# La suite du fichier est identique à la version précédente et devrait être correcte
def execute_group_intra_subject_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Single_Protocol_Group_Analysis",
    save_results_flag=SAVE_ANALYSIS_RESULTS,
    enable_verbose_logging=False,
    generate_plots_flag=GENERATE_PLOTS,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=N_JOBS_PROCESSING,
    compute_group_level_stats_flag=True,
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    compute_intra_subject_stats_for_group_runs_flag=True,
    n_perms_for_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    n_jobs_for_group_cluster_stats=N_JOBS_PROCESSING,
    use_grid_search_for_group=USE_GRID_SEARCH_OPTIMIZATION,
    use_csp_for_temporal_group=USE_CSP_FOR_TEMPORAL_PIPELINES,
    use_anova_fs_for_temporal_group=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config_for_group=PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
    cv_folds_for_gs_group=CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    fixed_params_for_group=FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
):
    """Executes intra-subject decoding for all subjects in a group and aggregates results."""
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group.error("subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group.error("group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject, str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)
    actual_n_jobs_group_stats = -1 if isinstance(n_jobs_for_group_cluster_stats, str) and n_jobs_for_group_cluster_stats.lower() == "auto" else int(n_jobs_for_group_cluster_stats)

    logger_run_group.info(
        "Starting intra-subject decoding analysis for GROUP: %s. GS: %s, CSP: %s, ANOVA FS: %s. n_jobs_subj: %s, n_jobs_grp_stats: %s.",
        group_identifier, use_grid_search_for_group, use_csp_for_temporal_group,
        use_anova_fs_for_temporal_group, actual_n_jobs_subject, actual_n_jobs_group_stats
    )

    group_results_collection = {
        "subject_global_auc_scores": {}, "subject_global_metrics_maps": {},
        "subject_temporal_scores_1d_mean_list": [],
        "subject_epochs_time_points_list": [], "subject_tgm_scores_mean_list": [],
        "subject_mean_of_specific_scores_list": [], "processed_subject_ids": []
    }

    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_group.info("\n--- Group '%s': Processing Subject %d/%d: %s ---",
                              group_identifier, i, len(subject_ids_in_group), subject_id_current)

        # C'est ici que l'appel est fait. Grâce à l'import correct, Python sait
        # maintenant ce qu'est 'execute_single_subject_decoding'.
        subject_output_dict = execute_single_subject_decoding(
            subject_identifier=subject_id_current,
            group_affiliation=group_identifier,
            decoding_protocol_identifier=f"{decoding_protocol_identifier}_Subj_{subject_id_current}",
            save_results_flag=save_results_flag,
            enable_verbose_logging=enable_verbose_logging,
            generate_plots_flag=generate_plots_flag,
            base_input_data_path=base_input_data_path,
            base_output_results_path=base_output_results_path,
            n_jobs_for_processing=actual_n_jobs_subject,
            classifier_type=classifier_type_for_group_runs,
            use_grid_search_for_subject=use_grid_search_for_group,
            use_csp_for_temporal_subject=use_csp_for_temporal_group,
            use_anova_fs_for_temporal_subject=use_anova_fs_for_temporal_group,
            param_grid_config_for_subject=param_grid_config_for_group,
            cv_folds_for_gs_subject=cv_folds_for_gs_group,
            fixed_params_for_subject=fixed_params_for_group,
            compute_intra_subject_stats_flag=compute_intra_subject_stats_for_group_runs_flag,
            n_perms_for_intra_subject_clusters=n_perms_intra_subject_folds_for_group_runs,
            compute_tgm_flag=compute_tgm_for_group_subjects_flag,
            cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold_group,
            loading_conditions_config=loading_conditions_config
        )
        
        s_auc = subject_output_dict.get("pp_ap_main_mean_auc_global", np.nan)
        s_metrics = subject_output_dict.get("pp_ap_main_global_metrics", {})
        s_scores_t_1d_mean = subject_output_dict.get("pp_ap_main_scores_1d_mean")
        s_times_t = subject_output_dict.get("epochs_time_points")
        s_scores_tgm_mean = subject_output_dict.get("pp_ap_main_tgm_mean")
        s_mean_specific = subject_output_dict.get("pp_ap_mean_of_specific_scores_1d")

        group_results_collection["subject_global_auc_scores"][subject_id_current] = s_auc
        group_results_collection["subject_global_metrics_maps"][subject_id_current] = s_metrics

        if pd.notna(s_auc) and s_scores_t_1d_mean is not None and s_times_t is not None and \
           s_scores_t_1d_mean.size > 0 and s_times_t.size > 0:
            group_results_collection["subject_temporal_scores_1d_mean_list"].append(s_scores_t_1d_mean)
            group_results_collection["subject_epochs_time_points_list"].append(s_times_t)
            group_results_collection["processed_subject_ids"].append(subject_id_current)

            if compute_tgm_for_group_subjects_flag and s_scores_tgm_mean is not None and s_scores_tgm_mean.ndim == 2:
                group_results_collection["subject_tgm_scores_mean_list"].append(s_scores_tgm_mean)
            
            if s_mean_specific is not None and s_mean_specific.ndim == 1:
                group_results_collection["subject_mean_of_specific_scores_list"].append(s_mean_specific)
        else:
            logger_run_group.warning(
                "Skipping subject %s from group '%s' aggregation (errors or no valid main scores).", subject_id_current, group_identifier)

    # Le reste du fichier pour l'agrégation, les stats et les plots.
    # Cette partie est longue mais ne devrait pas poser de problème.
    # ... [Le reste du fichier, inchangé, est omis pour la brièveté] ...
    logger_run_group.info("Finished aggregation logic. Total group analysis time: %.1f min", (time.time() - total_group_analysis_start_time) / 60)
    return group_results_collection["subject_global_auc_scores"]


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="EEG Group Intra-Subject Decoding Analysis Script",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli_parser.add_argument("--group_name", type=str, required=True,
                            choices=ALL_SUBJECT_GROUPS.keys(), help="Name of the group to process.")
    cli_parser.add_argument("--clf_type_override", type=str, default=None, choices=["svc", "logreg", "rf"],
                            help="Override default classifier type from config.")
    cli_parser.add_argument("--n_jobs_override", type=str, default=None,
                            help="Override n_jobs for subject and group stats processing (e.g., '4' or 'auto').")

    command_line_args = cli_parser.parse_args()

    n_jobs_arg_str_main = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use_main = -1 if n_jobs_arg_str_main.lower() == "auto" else int(n_jobs_arg_str_main)
    except (ValueError, AttributeError):
        logger_run_group.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str_main}'). Using default from config: {N_JOBS_PROCESSING}.")
        n_jobs_to_use_main = -1 if str(N_JOBS_PROCESSING).lower() == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use_main = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login_main = getuser()
    main_input_path_main, main_output_path_main = configure_project_paths(user_login_main)

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT STARTED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger_run_group.info("User: %s, Group to process: %s",
                          user_login_main, command_line_args.group_name)
    logger_run_group.info("  Classifier: %s, n_jobs (main ops): %s",
                          classifier_type_to_use_main, n_jobs_to_use_main)

    execute_group_intra_subject_decoding_analysis(
        subject_ids_in_group=ALL_SUBJECT_GROUPS[command_line_args.group_name],
        group_identifier=command_line_args.group_name,
        base_input_data_path=main_input_path_main,
        base_output_results_path=main_output_path_main,
        n_jobs_for_each_subject=n_jobs_to_use_main,
        n_jobs_for_group_cluster_stats=n_jobs_to_use_main,
        classifier_type_for_group_runs=classifier_type_to_use_main,
    )

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT FINISHED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)