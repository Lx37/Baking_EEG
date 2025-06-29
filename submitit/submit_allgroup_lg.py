

import os
import sys
import logging
from datetime import datetime
import getpass
import submitit
import time 
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd())

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from config.config import ALL_SUBJECT_GROUPS
from utils.utils import configure_project_paths
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, SAVE_ANALYSIS_RESULTS,
    GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, USE_CSP_FOR_TEMPORAL_PIPELINES,
    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
)


def decoding_task_wrapper(**kwargs):
 
    import sys
    import os

    project_root = "/home/tom.balay/Baking_EEG" 
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    from examples.run_decoding_one_lg import execute_single_subject_lg_decoding
    
    return execute_single_subject_lg_decoding(**kwargs)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    
    logger.info("--- Démarrage de la soumission pour TOUS LES SUJETS (Protocole LG) ---")

    user = getpass.getuser()
    base_input_path, base_output_path = configure_project_paths(user)

    
    common_job_kwargs = {
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path,
        "n_jobs_for_processing": 40, # Sera utilisé par le worker
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "generate_plots_flag": GENERATE_PLOTS,
        "loading_conditions_config": CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
        "classifier_type": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_subject": USE_GRID_SEARCH_OPTIMIZATION,
        "param_grid_config_for_subject": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_subject": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
        "fixed_params_for_subject": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
        "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        "use_csp_for_temporal_subject": USE_CSP_FOR_TEMPORAL_PIPELINES,
        "use_anova_fs_for_temporal_subject": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
    }


    submitted_jobs = []
    failed_submissions = []

    for group_name, subject_list in ALL_SUBJECT_GROUPS.items():
        logger.info(f"\n--- Préparation des soumissions pour le groupe: {group_name} ---")
        for subject_id in subject_list:
            logger.info(f"  Configuration de la soumission pour le sujet: {subject_id} du groupe {group_name}")

          
            log_folder_subject = f"logs_submitit_jobs_LG/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{subject_id}_LG"
            
          
            executor = submitit.AutoExecutor(folder=log_folder_subject)
            executor.update_parameters(
                timeout_min=12 * 60,  # Temps maximum pour un sujet
                slurm_partition="CPU",
                slurm_mem="60G", # Mémoire par sujet
                slurm_cpus_per_task=40, # CPUs par sujet (ou ajuster si N_JOBS_PROCESSING est différent)
                slurm_job_name=f"LG_{subject_id}", # Nom du job dans Slurm
                slurm_additional_parameters={"account": "tom.balay"} # Compte Slurm
            )


            subject_specific_kwargs = {
                "subject_identifier": subject_id,
                "group_affiliation": group_name,
                **common_job_kwargs # Fusionner avec les paramètres communs
            }

            try:
                logger.info(f"    Soumission du WRAPPER pour {subject_id} (Groupe: {group_name})...")
                job = executor.submit(decoding_task_wrapper, **subject_specific_kwargs)
                logger.info(f"    Job pour {subject_id} soumis avec l'ID: {job.job_id}. Logs dans: {os.path.abspath(log_folder_subject)}")
                submitted_jobs.append(job)
              
            
            except Exception as e_submit:
                logger.error(f"    Échec de la soumission pour {subject_id}: {e_submit}", exc_info=True)
                failed_submissions.append({"subject_id": subject_id, "error": str(e_submit)})

    logger.info(f"\n--- Fin des soumissions ---")
    logger.info(f"Nombre total de jobs soumis: {len(submitted_jobs)}")
    if failed_submissions:
        logger.warning(f"Nombre de soumissions échouées: {len(failed_submissions)}")
        for failed in failed_submissions:
            logger.warning(f"  - Sujet: {failed['subject_id']}, Erreur: {failed['error']}")

if __name__ == "__main__":
    main()