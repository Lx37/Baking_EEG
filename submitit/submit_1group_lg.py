# =============================================================================
#  File name        : submit_1group_lg.py
#  Author           : Tom Balay (and a bit of Copilot)
#  Created          : 2025-07-28
#  Description      :
#  Submitit script to run group-level LG decoding analysis for all subjects in a single group.
#  distributed on a single node.
#  This script is designed to be run on a cluster with Slurm.
# 
#  BSD 3-Clause License 2025, CNRS, Tom Balay
# =============================================================================
import os
import sys
import logging
from datetime import datetime
import getpass
import submitit
from submitit.core.utils import FailedJobError
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

log_dir = os.path.join(SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(log_dir, exist_ok=True)
master_log = os.path.join(log_dir, f"master_submitit_LG_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(master_log), logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("--- Démarrage du script de soumission pour sujets LG individuels ---")

try:
    from utils.utils import configure_project_paths
    from examples.run_decoding_one_group_lg import execute_group_intra_subject_lg_decoding_analysis
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL, FIXED_CLASSIFIER_PARAMS_CONFIG,
        COMPUTE_TGM_FOR_MAIN_COMPARISON, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_INTRA_SUBJECT_STATISTICS,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING
    )
    from config.config import ALL_SUBJECTS_GROUPS
    logger.info("Imports projet LG réussis")
except ImportError as e:
    logger.critical(f"Import error: {e}", exc_info=True)
    sys.exit(1)

# --- 4. SLURM SETUP SCRIPT ---
VENV_ACTIVATE = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
SETUP = f"""
cd {PROJECT_ROOT}
export PYTHONPATH={PROJECT_ROOT}:$PYTHONPATH
source {VENV_ACTIVATE}
"""

# --- 5. WRAPPER WORKER ---
def execute_wrapper(**kwargs):
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from examples.run_decoding_one_group_lg import execute_group_intra_subject_lg_decoding_analysis
        return execute_group_intra_subject_lg_decoding_analysis(**kwargs)
    except Exception:
        traceback.print_exc()
        raise


def main():
    user = getpass.getuser()
    base_in, base_out = configure_project_paths(user)
    group = 'DELIRIUM+'
    subjects = ALL_SUBJECTS_GROUPS.get(group)
    if not subjects:
        logger.error(f"No subjects for group {group}")
        sys.exit(1)
    slurm_cpus = 40

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_folder = os.path.join(log_dir, f"slurm_job_LG_{group}_{timestamp}")

    executor = submitit.AutoExecutor(folder=job_folder)

    executor.update_parameters(
        name=f"decoding_LG_{group}",
        timeout_min=120000,
        slurm_partition="CPU",
        slurm_mem="79G",
        slurm_cpus_per_task=slurm_cpus,
        local_setup=SETUP,
        slurm_additional_parameters={"account":"tom.balay","export":"ALL"}
    )

    logger.info(f"Préparation de la soumission d'un job LG pour le groupe {group}...")

    kwargs = {
        "subject_ids_in_group": subjects,
        "group_identifier": group,
        "decoding_protocol_identifier": "Single_LG_Protocol_Group_Analysis",
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "enable_verbose_logging": True,
        "generate_plots_flag": GENERATE_PLOTS,
        "base_input_data_path": base_in,
        "base_output_results_path": base_out,
        "n_jobs_for_each_subject": slurm_cpus,
        "classifier_type_for_group_runs": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_group": USE_GRID_SEARCH_OPTIMIZATION,
        "use_anova_fs_for_temporal_group": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        "param_grid_config_for_group": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_group": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
        "fixed_params_for_group": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "loading_conditions_config": None,
        "cluster_threshold_config_intra_fold_group": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        "compute_intra_subject_stats_for_group_runs_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_intra_subject_folds_for_group_runs": N_PERMUTATIONS_INTRA_SUBJECT,
        "compute_tgm_for_group_subjects_flag": COMPUTE_TGM_FOR_MAIN_COMPARISON
    }

    job = executor.submit(execute_wrapper, **kwargs)
    logger.info(f"Job {job.job_id} soumis pour le groupe LG {group}.")

    logger.info(f"--- Job LG pour le groupe {group} soumis. En attente du résultat... ---")

    try:
        result = job.result()
        logger.info(f"Job {job.job_id} (Groupe LG: {group}) a terminé avec le statut {job.state}.")
    except FailedJobError as e:
        logger.error(f"Job {job.job_id} (Groupe LG: {group}) A ÉCHOUÉ: {e}")
        logger.error(f"STDOUT Log: \n{job.stdout()}")
        logger.error(f"STDERR Log: \n{job.stderr()}")

if __name__ == "__main__":
    main()
