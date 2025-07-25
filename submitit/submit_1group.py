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
master_log = os.path.join(log_dir, f"master_submitit_{datetime.now():%Y%m%d_%H%M%S}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.FileHandler(master_log), logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)
logger.info("--- Démarrage du script de soumission pour sujets individuels ---")

try:
    from utils.utils import configure_project_paths
    from examples.run_decoding_one_group_pp import execute_single_subject_decoding
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL, FIXED_CLASSIFIER_PARAMS_CONFIG,
        COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_INTRA_SUBJECT_STATISTICS,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS
    )
    from config.config import ALL_SUBJECTS_GROUPS
    logger.info("Imports projet réussis")
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
        from examples.run_decoding_one_group_pp import execute_single_subject_decoding
        return execute_single_subject_decoding(**kwargs)
    except Exception:
        traceback.print_exc()
        raise


def main():
    user = getpass.getuser()
    base_in, base_out = configure_project_paths(user)
    group = 'CONTROLS_DELIRIUM'
    subjects = ALL_SUBJECTS_GROUPS.get(group)
    if not subjects:
        logger.error(f"No subjects for group {group}")
        sys.exit(1)
    slurm_cpus = 40

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    job_folder = os.path.join(log_dir, f"slurm_job_{group}_{timestamp}")

    executor = submitit.AutoExecutor(folder=job_folder)

    executor.update_parameters(
        name=f"decoding_{group}",
        timeout_min=1200,
        slurm_partition="CPU",
        slurm_mem="79G",
        slurm_cpus_per_task=slurm_cpus,
        local_setup=SETUP,
        slurm_additional_parameters={"account":"tom.balay","export":"ALL"}
    )


    submitted_jobs = []
    logger.info(f"Préparation de la soumission de {len(subjects)} jobs pour le groupe {group}...")

    for subject_id in subjects:
        logger.info(f"Configuration du job pour le sujet : {subject_id}")
        
        kwargs = {
            "subject_identifier": subject_id,
            "group_affiliation": group,
            "decoding_protocol_identifier": f"Single_Sub_{subject_id}",
            "base_input_data_path": base_in,
            "base_output_results_path": base_out,
            "enable_verbose_logging": True,
            "save_results_flag": SAVE_ANALYSIS_RESULTS,
            "generate_plots_flag": GENERATE_PLOTS,
            "n_jobs_for_processing": slurm_cpus,
            "classifier_type": CLASSIFIER_MODEL_TYPE,
            "use_grid_search_for_subject": USE_GRID_SEARCH_OPTIMIZATION,
            "fixed_params_for_subject": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
            "use_anova_fs_for_temporal_subject": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
            "param_grid_config_for_subject": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
            "cv_folds_for_gs_subject": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
            "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
            "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
            "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
            "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
            "loading_conditions_config": None
        }

        job = executor.submit(execute_wrapper, **kwargs)

        submitted_jobs.append((job, subject_id))
        logger.info(f"Job {job.job_id} soumis pour le sujet {subject_id}.")

    logger.info(f"--- Tous les {len(submitted_jobs)} jobs ont été soumis. En attente des résultats... ---")

    for job, subject_id in submitted_jobs:
        try:
            result = job.result()

            logger.info(f"Job {job.job_id} (Sujet: {subject_id}) a terminé avec le statut {job.state}.")
        except FailedJobError as e:

            logger.error(f"Job {job.job_id} (Sujet: {subject_id}) A ÉCHOUÉ: {e}")
            logger.error(f"STDOUT Log: \n{job.stdout()}")
            logger.error(f"STDERR Log: \n{job.stderr()}")

if __name__ == "__main__":
    main()