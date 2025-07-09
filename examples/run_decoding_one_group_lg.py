import sys
import os
import logging
import time
import argparse
from datetime import datetime
from getpass import getuser
import numpy as np
import pandas as pd

SCRIPT_DIR_EXAMPLE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EXAMPLE = os.path.abspath(os.path.join(SCRIPT_DIR_EXAMPLE, ".."))
if PROJECT_ROOT_EXAMPLE not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_EXAMPLE)

from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING,
    COMPUTE_TGM_FOR_MAIN_COMPARISON
)
from config.config import ALL_SUBJECT_GROUPS
from utils.utils import configure_project_paths

try:
    from examples.run_decoding_one_lg import (
        execute_single_subject_lg_decoding
    )
except ImportError as e_import:
    print(f"Erreur d'import execute_single_subject_lg_decoding: {e_import}")
    sys.exit(1)

LOG_DIR_RUN_GROUP_LG = './logs_run_group_lg_analysis'
os.makedirs(LOG_DIR_RUN_GROUP_LG, exist_ok=True)
LOG_FILENAME_RUN_GROUP_LG = os.path.join(
    LOG_DIR_RUN_GROUP_LG,
    datetime.now().strftime('log_run_group_lg_analysis_%Y-%m-%d_%H%M%S.log')
)

for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_GROUP_LG),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_group_lg = logging.getLogger(__name__)


def execute_group_intra_subject_lg_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Single_LG_Protocol_Group_Analysis",
    save_results_flag=SAVE_ANALYSIS_RESULTS,
    enable_verbose_logging=False,
    generate_plots_flag=GENERATE_PLOTS,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=N_JOBS_PROCESSING,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    use_grid_search_for_group=USE_GRID_SEARCH_OPTIMIZATION,
    use_csp_for_temporal_group=USE_CSP_FOR_TEMPORAL_PIPELINES,
    use_anova_fs_for_temporal_group=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config_for_group=PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
    cv_folds_for_gs_group=CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    fixed_params_for_group=FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    compute_intra_subject_stats_for_group_runs_flag=True,
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    compute_tgm_for_group_subjects_flag=COMPUTE_TGM_FOR_MAIN_COMPARISON
):
    """Executes intra-subject LG decoding for all subjects in a group."""
    
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group_lg.error("subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group_lg.error("group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    # Convert n_jobs
    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject, str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)

    logger_run_group_lg.info(
        "Starting intra-subject LG decoding analysis for GROUP: %s. GS: %s, CSP: %s, ANOVA FS: %s. n_jobs_subj: %s.",
        group_identifier, use_grid_search_for_group, use_csp_for_temporal_group,
        use_anova_fs_for_temporal_group, actual_n_jobs_subject
    )

    processed_subjects = []

    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_group_lg.info("\n--- LG Group '%s': Processing Subject %d/%d: %s ---",
                                 group_identifier, i, len(subject_ids_in_group), subject_id_current)

        try:
            subject_output_dict = execute_single_subject_lg_decoding(
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

            # Check if processing was successful
            lg_auc = subject_output_dict.get("lg_ls_ld_mean_auc_global", np.nan)
            if pd.notna(lg_auc):
                processed_subjects.append(subject_id_current)
                logger_run_group_lg.info(
                    "Subject %s processed successfully. LG AUC: %.3f",
                    subject_id_current, lg_auc)
            else:
                logger_run_group_lg.warning(
                    "Subject %s processing failed or returned invalid results.",
                    subject_id_current)

        except Exception as e:
            logger_run_group_lg.error(
                "Error processing subject %s: %s", subject_id_current, e, exc_info=True)

    logger_run_group_lg.info(
        "LG Group analysis for '%s' completed in %.2fs. Successfully processed: %d/%d subjects.",
        group_identifier, time.time() - total_group_analysis_start_time, 
        len(processed_subjects), len(subject_ids_in_group))
    
    if processed_subjects:
        logger_run_group_lg.info("Successfully processed subjects: %s", 
                                ", ".join(processed_subjects))
    else:
        logger_run_group_lg.warning("No subjects were successfully processed!")

    return {"processed_subjects": processed_subjects}


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="EEG Group LG Decoding Analysis Script",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli_parser.add_argument("--group", type=str, required=True, choices=list(ALL_SUBJECT_GROUPS.keys()),
                            help="Group identifier for LG analysis.")
    cli_parser.add_argument("--clf_type_override", type=str, default=None, choices=["svc", "logreg", "rf"],
                            help="Override default classifier type from config.")
    cli_parser.add_argument("--n_jobs_override", type=str, default=None,
                            help="Override n_jobs from config (e.g., '4' or 'auto').")
    cli_parser.add_argument("--no-tgm", action="store_true",
                            help="Disable TGM computation for all subjects (recommended for cluster to avoid timeouts).")

    command_line_args = cli_parser.parse_args()

    n_jobs_arg_str = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use = -1 if n_jobs_arg_str.lower() == "auto" else int(n_jobs_arg_str)
    except ValueError:
        logger_run_group_lg.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str}'). Using default from config: {N_JOBS_PROCESSING} (becomes -1 if 'auto').")
        n_jobs_to_use = -1 if N_JOBS_PROCESSING.lower() == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login = getuser()
    main_input_path, main_output_path = configure_project_paths(user_login)

    group_name = command_line_args.group
    subject_list = ALL_SUBJECT_GROUPS.get(group_name, [])
    if not subject_list:
        logger_run_group_lg.error("No subjects found for group '%s'.", group_name)
        sys.exit(1)

    logger_run_group_lg.info("\n%s EEG GROUP LG DECODING SCRIPT STARTED (%s) %s",
                             "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger_run_group_lg.info("User: %s, Group: %s, N subjects: %d",
                             user_login, group_name, len(subject_list))
    logger_run_group_lg.info("  Subjects: %s", ", ".join(subject_list))
    logger_run_group_lg.info("  Classifier: %s, n_jobs: %s",
                             classifier_type_to_use, n_jobs_to_use)
    logger_run_group_lg.info("  GridSearch Optimization: %s", USE_GRID_SEARCH_OPTIMIZATION)
    logger_run_group_lg.info("  CSP for Temporal Pipelines: %s", USE_CSP_FOR_TEMPORAL_PIPELINES)
    logger_run_group_lg.info("  ANOVA FS for Temporal Pipelines: %s", USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES)

    # Determine TGM flag
    compute_tgm_for_group = not command_line_args.no_tgm
    
    if command_line_args.no_tgm:
        logger_run_group_lg.info("TGM computation disabled for all subjects via --no-tgm flag")
    else:
        logger_run_group_lg.info("TGM computation enabled for all subjects (default or via config)")

    execute_group_intra_subject_lg_decoding_analysis(
        subject_ids_in_group=subject_list,
        group_identifier=group_name,
        base_input_data_path=main_input_path,
        base_output_results_path=main_output_path,
        n_jobs_for_each_subject=n_jobs_to_use,
        classifier_type_for_group_runs=classifier_type_to_use,
        generate_plots_flag=False,  # Disable plots
        compute_tgm_for_group_subjects_flag=compute_tgm_for_group,
    )

    logger_run_group_lg.info("\n%s EEG GROUP LG DECODING SCRIPT FINISHED (%s) %s",
                             "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
