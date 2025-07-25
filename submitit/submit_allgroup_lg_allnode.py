import os
import sys
import signal
import logging
from datetime import datetime
import getpass
import submitit
import time 

# Configuration du projet
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration des logs
LOG_DIR_SUBMITIT_GROUPS = './logs_submitit_groups_LG'
os.makedirs(LOG_DIR_SUBMITIT_GROUPS, exist_ok=True)

# Configuration de l'environnement cluster
PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT

SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm LG (PID: $$) ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"

# Configuration des signaux pour éviter les warnings
trap 'echo "Signal reçu, poursuite de l execution..." ; exit 0' SIGCONT SIGTERM SIGINT

module purge
echo "Activation de l'environnement virtuel: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
source {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}
if [ $? -ne 0 ]; then echo "ERREUR: Échec de l'activation de l'environnement virtuel."; exit 1; fi

export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}:${{PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Configuration pour éviter les warnings submitit
export SUBMITIT_BYPASS_SIGNALS=1

echo "PYTHONPATH: $PYTHONPATH"
echo "--- Environnement LG configuré ---"
"""


from config.config import ALL_SUBJECTS_GROUPS
from utils.utils import configure_project_paths
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, SAVE_ANALYSIS_RESULTS,
    GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG,
    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
)


def group_decoding_task_wrapper(**kwargs):
    """
    Wrapper function to execute group LG decoding on cluster.
    """
    import sys
    import os
    import signal
    
    # Gestionnaire de signaux pour éviter les warnings de bypassing
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, continuing execution...")
        return
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    project_root = "/home/tom.balay/Baking_EEG" 
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # Import the LG group analysis function
    from examples.run_decoding_one_group_lg import execute_group_intra_subject_lg_decoding_analysis
    
    return execute_group_intra_subject_lg_decoding_analysis(**kwargs)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to submit group LG decoding jobs to cluster.
    Each group is processed on a separate node.
    """
    logger.info("--- Démarrage de la soumission pour TOUS LES GROUPES (Protocole LG) ---")

    user = getpass.getuser()
    base_input_path, base_output_path = configure_project_paths(user)

    # Configuration Slurm pour les jobs de groupe
    SLURM_CPUS_PER_GROUP_JOB = 40  # CPUs par groupe
    SLURM_MEMORY_PER_JOB = "77G"   # Mémoire par groupe
    SLURM_TIMEOUT_MINUTES = 130000 * 60  
    SLURM_PARTITION = "CPU"
    SLURM_ACCOUNT = "tom.balay"

    logger.info(f"Configuration Slurm:")
    logger.info(f"  CPUs par groupe: {SLURM_CPUS_PER_GROUP_JOB}")
    logger.info(f"  Mémoire par groupe: {SLURM_MEMORY_PER_JOB}")
    logger.info(f"  Timeout: {SLURM_TIMEOUT_MINUTES} minutes")
    logger.info(f"  Partition: {SLURM_PARTITION}")

    # Groupes à traiter
    GROUPS_TO_PROCESS = list(ALL_SUBJECTS_GROUPS.keys())
    logger.info(f"Groupes à traiter: {GROUPS_TO_PROCESS}")
    logger.info(f"Nombre total de groupes: {len(GROUPS_TO_PROCESS)}")

    submitted_jobs = []
    failed_submissions = []

    for group_name in GROUPS_TO_PROCESS:
        if group_name not in ALL_SUBJECTS_GROUPS:
            logger.warning(f"Groupe '{group_name}' non trouvé dans la configuration. Ignoré.")
            continue

        subjects_for_this_group = ALL_SUBJECTS_GROUPS[group_name]
        if not subjects_for_this_group:
            logger.warning(f"Aucun sujet défini pour le groupe '{group_name}'. Ignoré.")
            continue

        logger.info(f"\n--- Préparation de la soumission pour le groupe: {group_name} ---")
        logger.info(f"  Nombre de sujets dans ce groupe: {len(subjects_for_this_group)}")
        logger.info(f"  Sujets: {', '.join(subjects_for_this_group)}")

        # Pour chaque sujet du groupe, soumettre un job sur un noeud séparé
        for subject_id in subjects_for_this_group:
            current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_folder_subject = f"logs_submitit_groups_LG/{current_time_str}_{group_name}_{subject_id}_LG_SUBJECT"
            executor = submitit.AutoExecutor(folder=log_folder_subject)
            executor.update_parameters(
                timeout_min=SLURM_TIMEOUT_MINUTES,
                setup=SETUP_COMMANDS_FOR_SLURM_JOB_CPU,
                slurm_partition=SLURM_PARTITION,
                slurm_mem=SLURM_MEMORY_PER_JOB,
                slurm_cpus_per_task=SLURM_CPUS_PER_GROUP_JOB,
                slurm_job_name=f"LG_SUBJECT_{group_name}_{subject_id}",
                slurm_additional_parameters={
                    "account": SLURM_ACCOUNT,
                    "nodes": 1,
                    "signal": "USR1@180",
                    "requeue": True,
                }
            )

            # Paramètres pour le job du sujet
            subject_job_kwargs = {
                "subject_ids_in_group": [subject_id],
                "group_identifier": group_name,
                "decoding_protocol_identifier": f'Group_LG_Intra_{group_name}_{subject_id}',
                "base_input_data_path": base_input_path,
                "base_output_results_path": base_output_path,
                "enable_verbose_logging": True,
                "n_jobs_for_each_subject": SLURM_CPUS_PER_GROUP_JOB,
                "save_results_flag": SAVE_ANALYSIS_RESULTS,
                "generate_plots_flag": False,
                "loading_conditions_config": CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
                "classifier_type_for_group_runs": CLASSIFIER_MODEL_TYPE,
                "use_grid_search_for_group": USE_GRID_SEARCH_OPTIMIZATION,
                "use_anova_fs_for_temporal_group": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
                "param_grid_config_for_group": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
                "cv_folds_for_gs_group": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
                "fixed_params_for_group": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
                "compute_intra_subject_stats_for_group_runs_flag": True,
                "n_perms_intra_subject_folds_for_group_runs": N_PERMUTATIONS_INTRA_SUBJECT,
                "compute_tgm_for_group_subjects_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
                "cluster_threshold_config_intra_fold_group": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
            }

            try:
                logger.info(f"  Soumission du job pour le sujet {subject_id} du groupe {group_name}...")
                job = executor.submit(group_decoding_task_wrapper, **subject_job_kwargs)
                logger.info(f"  Job pour le sujet {subject_id} du groupe {group_name} soumis avec l'ID: {job.job_id}")
                logger.info(f"  Logs dans: {os.path.abspath(log_folder_subject)}")
                submitted_jobs.append({
                    "job": job,
                    "group_name": group_name,
                    "subject_id": subject_id,
                    "log_folder": log_folder_subject
                })
            except Exception as e_submit:
                logger.error(f"  Échec de la soumission pour le sujet {subject_id} du groupe {group_name}: {e_submit}", exc_info=True)
                failed_submissions.append({"group_name": group_name, "subject_id": subject_id, "error": str(e_submit)})

    logger.info(f"\n--- Fin des soumissions ---")
    logger.info(f"Nombre total de jobs soumis: {len(submitted_jobs)}")
    logger.info(f"Nombre total de sujets à traiter: {len(submitted_jobs)}")
    if failed_submissions:
        logger.warning(f"Nombre de soumissions échouées: {len(failed_submissions)}")
        for failed in failed_submissions:
            logger.warning(f"  - Groupe: {failed['group_name']} | Sujet: {failed['subject_id']} | Erreur: {failed['error']}")
    if submitted_jobs:
        logger.info(f"\n--- Résumé des jobs soumis ---")
        for job_info in submitted_jobs:
            logger.info(f"  Groupe: {job_info['group_name']} | Sujet: {job_info['subject_id']} | Job ID: {job_info['job'].job_id} | Logs: {job_info['log_folder']}")
    logger.info("\n--- Script de soumission terminé ---")

if __name__ == "__main__":
    main()