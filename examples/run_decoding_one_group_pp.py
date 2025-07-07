

import os
import sys
import logging
import time
from datetime import datetime
import argparse
from getpass import getuser
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING,
    COMPUTE_TGM_FOR_MAIN_COMPARISON
)
from config.config import ALL_SUBJECT_GROUPS
from utils.utils import configure_project_paths

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger_import = logging.getLogger(__name__)


execute_single_subject_decoding = None

try:

    logger_import.info(
        "Tentative d'import de execute_single_subject_decoding...")
    from examples.run_decoding_one_pp import execute_single_subject_decoding
    logger_import.info("✅ Import de execute_single_subject_decoding réussi")
except ImportError as e_import:
    logger_import.error(
        f"ERREUR D'IMPORT CRITIQUE: Impossible d'importer 'execute_single_subject_decoding' depuis 'examples/run_decoding_one_pp.py'.")
    logger_import.error(
        f"Vérifiez que le fichier existe et que le nom de la fonction est correct.")
    logger_import.error(f"Erreur originale: {e_import}")
    sys.exit(1)
except Exception as e_other:
    logger_import.error(f"ERREUR INATTENDUE lors de l'import: {e_other}")
    sys.exit(1)



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
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TGM_FOR_MAIN_COMPARISON,
    compute_intra_subject_stats_for_group_runs_flag=True,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    use_grid_search_for_group=USE_GRID_SEARCH_OPTIMIZATION,
    use_csp_for_temporal_group=USE_CSP_FOR_TEMPORAL_PIPELINES,
    use_anova_fs_for_temporal_group=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config_for_group=PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
    cv_folds_for_gs_group=CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    fixed_params_for_group=FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
):
    """Executes intra-subject decoding for all subjects in a group - simplified loop only."""
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group.error("subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group.error("group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject,
                                             str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)

    logger_run_group.info(
        "Starting intra-subject decoding loop for GROUP: %s. GS: %s, CSP: %s, ANOVA FS: %s. n_jobs_subj: %s.",
        group_identifier, use_grid_search_for_group, use_csp_for_temporal_group,
        use_anova_fs_for_temporal_group, actual_n_jobs_subject
    )

    subject_global_auc_scores = {}

    # Simple loop over subjects without group aggregation
    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_group.info("\n--- Group '%s': Processing Subject %d/%d: %s ---",
                              group_identifier, i, len(subject_ids_in_group), subject_id_current)

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
        subject_global_auc_scores[subject_id_current] = s_auc
        
        if pd.notna(s_auc):
            logger_run_group.info("Subject %s completed successfully with AUC: %.3f", 
                                subject_id_current, s_auc)
        else:
            logger_run_group.warning("Subject %s completed with errors or no valid AUC", 
                                   subject_id_current)

    n_processed_subjects = len([auc for auc in subject_global_auc_scores.values() if pd.notna(auc)])
    logger_run_group.info("Successfully processed %d/%d subjects for group '%s'",
                          n_processed_subjects, len(subject_ids_in_group), group_identifier)

    logger_run_group.info("Finished subject loop. Total analysis time: %.1f min",
                          (time.time() - total_group_analysis_start_time) / 60)
    
    return subject_global_auc_scores


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
        n_jobs_to_use_main = - \
            1 if n_jobs_arg_str_main.lower() == "auto" else int(n_jobs_arg_str_main)
    except (ValueError, AttributeError):
        logger_run_group.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str_main}'). Using default from config: {N_JOBS_PROCESSING}.")
        n_jobs_to_use_main = - \
            1 if str(N_JOBS_PROCESSING).lower(
            ) == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use_main = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login_main = getuser()
    main_input_path_main, main_output_path_main = configure_project_paths(
        user_login_main)

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
        classifier_type_for_group_runs=classifier_type_to_use_main,
    )

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT FINISHED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
