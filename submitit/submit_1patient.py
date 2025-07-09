
import os
import sys
import logging
import getpass
import shutil
import submitit
from submitit.core.utils import FailedJobError

PROJECT_ROOT = "/home/tom.balay/Baking_EEG"


TARGET_SUBJECT_ID_FOR_JOB = "TpSM49"


PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def execute_single_subject_decoding_wrapper(**kwargs):
   
    import sys
    import os
    
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    
 
    print(f"DEBUG Worker - project_root: {PROJECT_ROOT}")
    print(f"DEBUG Worker - sys.path: {sys.path[:3]}...")
    

    from config.config import ALL_SUBJECT_GROUPS
    
    subject_id = kwargs.get("subject_identifier")
    group_affiliation = kwargs.get("group_affiliation")
    
    print(f"DEBUG Worker - Processing subject {subject_id} from group {group_affiliation}")
    
    
    if group_affiliation in ["DELIRIUM+", "DELIRIUM-", "CONTROLS_DELIRIUM"]:
        # Protocole PP
        print("DEBUG Worker - Using PP protocol")
        from examples.run_decoding_one_pp import execute_single_subject_decoding
        return execute_single_subject_decoding(**kwargs)
    elif group_affiliation in ["COMA", "VS", "MCS+", "MCS-", "CONTROLS_COMA"]:
        
        print("DEBUG Worker - Using PP protocol (extended/battery)")
        from examples.run_decoding_one_pp import execute_single_subject_decoding
        return execute_single_subject_decoding(**kwargs)
    elif group_affiliation in ["DEL", "NODEL"]:
      
        print("DEBUG Worker - Using LG protocol")
        from examples.run_decoding_one_lg import execute_single_subject_lg_decoding
        return execute_single_subject_lg_decoding(**kwargs)
    else:
        # Fallback sur PP par défaut
        print("DEBUG Worker - Using PP protocol (fallback)")
        from examples.run_decoding_one_pp import execute_single_subject_decoding
        return execute_single_subject_decoding(**kwargs)


def main_submission_logic():

    
    logger.info(f"--- Démarrage de la soumission pour le sujet: {TARGET_SUBJECT_ID_FOR_JOB} | Protocole: Auto-détection ---")


    if not os.path.isdir(PROJECT_ROOT):
        logger.critical(f"ERREUR : Le chemin racine du projet '{PROJECT_ROOT}' est invalide. Arrêt.")
        sys.exit(1)
        
   
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        
    try:
        from config.config import ALL_SUBJECT_GROUPS
        from utils.utils import configure_project_paths
        from config.decoding_config import (
            CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
            USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES, PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
            FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
            INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS, SAVE_ANALYSIS_RESULTS,
            GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
        )
        logger.info("Configuration du projet importée avec succès.")
    except (ModuleNotFoundError, ImportError) as e:
        logger.critical(f"ERREUR CRITIQUE lors de l'import de la configuration : {e}", exc_info=True)
        logger.critical(f"Vérifiez que le chemin '{PROJECT_ROOT}' est correct et contient vos modules.")
        sys.exit(1)


    target_subject_group = next((group for group, subjects in ALL_SUBJECT_GROUPS.items() if TARGET_SUBJECT_ID_FOR_JOB in subjects), None)
    if not target_subject_group:
        logger.critical(f"ERREUR: Sujet '{TARGET_SUBJECT_ID_FOR_JOB}' non trouvé dans ALL_SUBJECT_GROUPS. Arrêt.")
        sys.exit(1)
    
    if target_subject_group in ["DELIRIUM+", "DELIRIUM-", "CONTROLS_DELIRIUM"]:
        protocol_type = "PP"
    elif target_subject_group in ["COMA", "VS", "MCS+", "MCS-", "CONTROLS_COMA"]:
        protocol_type = "PP"  # PP étendu ou Battery
    elif target_subject_group in ["DEL", "NODEL"]:
        protocol_type = "LG"
    else:
        protocol_type = "PP"  # Fallback
    
    logger.info(f"Protocole détecté pour le groupe '{target_subject_group}': {protocol_type}")
    
    
    user_for_paths = getpass.getuser()
    base_input_path, base_output_path = configure_project_paths(user_for_paths)
    logger.info(f"Chemin des données d'entrée : {base_input_path}")
    logger.info(f"Chemin des résultats de sortie : {base_output_path}")

   
    setup_commands = [
        f"source {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}",
        f"export PYTHONPATH={PROJECT_ROOT}:${{PYTHONPATH}}",
    ]


    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    submitit_log_folder = os.path.join(
        current_script_dir,
        "submitit_logs",
        f"{TARGET_SUBJECT_ID_FOR_JOB}_{protocol_type}"
    )
    

    if os.path.exists(submitit_log_folder):
        logger.warning(f"Nettoyage du dossier de log existant : {submitit_log_folder}")
        shutil.rmtree(submitit_log_folder)
    os.makedirs(submitit_log_folder, exist_ok=True)
    logger.info(f"Les logs pour ce job seront sauvegardés dans : {submitit_log_folder}")

    # Configuration du job Slurm
    executor = submitit.AutoExecutor(folder=submitit_log_folder)
    executor.update_parameters(
        timeout_min=12000 * 60,
        slurm_partition="CPU",
        slurm_mem="60G",
        slurm_cpus_per_task=40,
        slurm_additional_parameters={"account": "tom.balay"},
        local_setup=setup_commands,
    )


    if protocol_type == "LG":
       
        try:
            from config.decoding_config import CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG
            loading_conditions_config = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG
        except ImportError:
            logger.warning("CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG non trouvé, utilisation de la config par défaut")
            loading_conditions_config = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    else:
    
        loading_conditions_config = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    
    kwargs_for_function_call = {
        "subject_identifier": TARGET_SUBJECT_ID_FOR_JOB,
        "group_affiliation": target_subject_group,
        "decoding_protocol_identifier": f'Analysis_{protocol_type}_Individual',
        "save_results_flag": SAVE_ANALYSIS_RESULTS, 
        "enable_verbose_logging": True,
        "generate_plots_flag": GENERATE_PLOTS, 
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path, 
        "n_jobs_for_processing": 40,
        "classifier_type": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_subject": USE_GRID_SEARCH_OPTIMIZATION,
        "use_anova_fs_for_temporal_subject": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        "param_grid_config_for_subject": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_subject": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
        "fixed_params_for_subject": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
        "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "loading_conditions_config": loading_conditions_config,
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    }

   
    logger.info(f"Configuration des chemins pour le job:")
    logger.info(f"  base_input_data_path: {base_input_path}")
    logger.info(f"  base_output_results_path: {base_output_path}")
    logger.info(f"  target_subject_group: {target_subject_group}")
    logger.info(f"  protocol_type: {protocol_type}")
    logger.info(f"  loading_conditions_config: {list(loading_conditions_config.keys()) if loading_conditions_config else 'None'}")

    # Soumission et attente des résultats
    try:
        logger.info(f"Soumission du job à Slurm...")
        job = executor.submit(execute_single_subject_decoding_wrapper, **kwargs_for_function_call)
        logger.info(f"Job soumis avec succès ! ID du job : {job.job_id}")
        logger.info("En attente de la fin du job... (cela peut prendre du temps)")

        results = job.result()
        
        logger.info(f"🎉 Job {job.job_id} terminé avec succès !")
        logger.info(f"Résultats reçus : {str(results)[:500]}...") 

    except FailedJobError as e:
        logger.error(f"❌ Le job {e.job_id} a échoué.")
        logger.error("--- Début des logs d'erreur du job ---")

        stderr_log = e.stderr()
        if stderr_log:
            logger.error(stderr_log.strip().split('\n')[-20:])
        logger.error("--- Fin des logs d'erreur du job ---")
        logger.error(f"Consultez les fichiers complets dans : {submitit_log_folder}")
    except Exception as e:
        logger.critical(f"❌ Une erreur inattendue est survenue lors de la soumission ou de l'attente du job.", exc_info=True)


if __name__ == "__main__":
    main_submission_logic()
    logger.info("--- Script de soumission terminé. ---")