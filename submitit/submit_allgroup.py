
import os
import sys
import logging
from datetime import datetime
import getpass
import time
import numpy as np
import pandas as pd
import submitit
from submitit.core.utils import FailedJobError

try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_SCRIPT_DIR = os.getcwd()
    print(f"AVERTISSEMENT: __file__ non défini. CURRENT_SCRIPT_DIR initialisé à : {CURRENT_SCRIPT_DIR}", file=sys.stderr)


project_root_tentative = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, ".."))

if os.path.isdir(project_root_tentative):
    PROJECT_ROOT_FOR_PYTHONPATH = project_root_tentative
    print(f"INFO: PROJECT_ROOT_FOR_PYTHONPATH confirmé : {PROJECT_ROOT_FOR_PYTHONPATH}")
else:
    print(f"AVERTISSEMENT: Le chemin '{project_root_tentative}' n'existe pas. Recherche de solution de repli.")
    potential_paths = [
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..")),
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "../..")),
        "/home/tom.balay/Baking_EEG", 
    ]
    found_path = False
    for path_candidate in potential_paths:
        if os.path.isdir(os.path.join(path_candidate, "examples")):
            PROJECT_ROOT_FOR_PYTHONPATH = path_candidate
            print(f"INFO: PROJECT_ROOT_FOR_PYTHONPATH trouvé via fallback : {PROJECT_ROOT_FOR_PYTHONPATH}")
            found_path = True
            break
    if not found_path:
        print(f"ERREUR CRITIQUE: Impossible de localiser le dossier racine du projet.")
        print(f"                 CURRENT_SCRIPT_DIR: {CURRENT_SCRIPT_DIR}")
        print(f"                 Chemins testés : {[project_root_tentative] + potential_paths}")
        sys.exit(1)


if PROJECT_ROOT_FOR_PYTHONPATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_PYTHONPATH)
    print(f"INFO: '{PROJECT_ROOT_FOR_PYTHONPATH}' a été ajouté à sys.path.")


LOG_DIR_SUBMITIT_MASTER = os.path.join(CURRENT_SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(LOG_DIR_SUBMITIT_MASTER, exist_ok=True)

MASTER_LOG_FILE_NAME = datetime.now().strftime('master_submitit_ALL_GROUPS_INTRA_%Y-%m-%d_%H-%M-%S.log')
MASTER_LOG_FILE_PATH = os.path.join(LOG_DIR_SUBMITIT_MASTER, MASTER_LOG_FILE_NAME)

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(MASTER_LOG_FILE_PATH, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("--- Script de soumission Submitit pour analyses de groupe démarré ---")
logger.info(f"Log principal de ce script de soumission : {MASTER_LOG_FILE_PATH}")
logger.info(f"CURRENT_SCRIPT_DIR (où ce script est exécuté) : {CURRENT_SCRIPT_DIR}")
logger.info(f"PROJECT_ROOT_FOR_PYTHONPATH (ajouté à sys.path) : {PROJECT_ROOT_FOR_PYTHONPATH}")


try:
    from utils.utils import configure_project_paths
    from config.config import ALL_SUBJECT_GROUPS
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_CSP_FOR_TEMPORAL_PIPELINES,
        USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES, PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
        FIXED_CLASSIFIER_PARAMS_CONFIG, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        N_PERMUTATIONS_INTRA_SUBJECT, INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS,
    )
    logger.info("Importations depuis 'config' et 'utils' réussies.")
except (ModuleNotFoundError, ImportError) as e:
    logger.critical(f"ERREUR CRITIQUE lors de l'importation de modules projet : {e}", exc_info=True)
    logger.critical(f"Vérifiez PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}') et sys.path.")
    sys.exit(1)


PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT_FOR_PYTHONPATH

SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm (PID: $$) ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial: $(pwd)"
module purge
echo "Modules purgés."
echo "Activation de l'environnement virtuel: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
source {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}
if [ $? -ne 0 ]; then echo "ERREUR: Échec de l'activation de l'environnement virtuel."; exit 1; fi
echo "Configuration de PYTHONPATH pour le job..."
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}:${{PYTHONPATH}}"
echo "PYTHONPATH actuel du job: $PYTHONPATH"
echo "--- Environnement CPU configuré (venv) ---"
echo "Chemin Python utilisé: $(which python)"
echo "Version Python: $(python -V)"
echo "Vérification des dépendances clés..."
python -c "import sys; print(f'sys.path dans Python: {{sys.path}}')"
python -c "import mne; print(f'MNE version: {{mne.__version__}} (depuis {{mne.__file__}})')"
python -c "import sklearn; print(f'scikit-learn version: {{sklearn.__version__}} (depuis {{sklearn.__file__}})')"
python -c "import numpy; print(f'NumPy version: {{numpy.__version__}} (depuis {{numpy.__file__}})')"
echo "--- Fin de la configuration, lancement de la tâche Python ---"
"""


def execute_group_decoding_wrapper(**kwargs):
   
    import sys
    import os

   
    project_root = PROJECT_ROOT_ON_CLUSTER_FOR_JOB
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    
    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)


    print(f"DEBUG Worker - project_root: {project_root}")
    print(f"DEBUG Worker - parent_dir: {parent_dir}")
    print(f"DEBUG Worker - sys.path: {sys.path[:3]}...")

    from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
    return execute_group_intra_subject_decoding_analysis(**kwargs)


def main_submission_logic():
    logger.info("--- Début de la logique de soumission principale (main_submission_logic) ---")
    
    try:
        user_for_paths = getpass.getuser()
    except Exception:
        user_for_paths = os.environ.get('LOGNAME') or os.environ.get('USER', "unknown_user_env")
    logger.info(f"Utilisateur détecté pour la configuration des chemins : {user_for_paths}")

    base_input_path, base_output_path = configure_project_paths(user_for_paths)
    if not os.path.isdir(base_input_path):
        logger.critical(f"ERREUR: Le chemin des données d'entrée '{base_input_path}' n'existe pas. Arrêt.")
        sys.exit(1)
    logger.info(f"Chemin de base des données d'entrée : {base_input_path}")
    logger.info(f"Chemin de base des résultats de sortie : {base_output_path}")


    GROUPS_TO_PROCESS = list(ALL_SUBJECT_GROUPS.keys())
    logger.info(f"Groupes ciblés pour l'analyse : {GROUPS_TO_PROCESS}")
    logger.info(f"Nombre total de groupes à traiter : {len(GROUPS_TO_PROCESS)}")

    # Configuration des ressources Slurm - Optimisée pour 7 groupes
    SLURM_CPUS_PER_GROUP_JOB = 40  # Réduit légèrement pour permettre 7 jobs simultanés
    SLURM_MEMORY_PER_JOB = "75G"   # Réduit légèrement pour permettre 7 jobs simultanés
    SLURM_TIMEOUT_MINUTES = 120000 * 60  # Augmenté car certains groupes sont plus gros
    SLURM_PARTITION = "CPU"
    SLURM_ACCOUNT = "tom.balay"

    slurm_params = {
        "timeout_min": SLURM_TIMEOUT_MINUTES,
        "setup": SETUP_COMMANDS_FOR_SLURM_JOB_CPU,
        "slurm_partition": SLURM_PARTITION,
        "slurm_mem": SLURM_MEMORY_PER_JOB,
        "slurm_cpus_per_task": SLURM_CPUS_PER_GROUP_JOB,
        "slurm_additional_parameters": {
            "account": SLURM_ACCOUNT,
            "nodes": 1,  # Force 1 nœud par job
            "exclusive": True  # Utilisation exclusive du nœud
        }
    }
    logger.info(f"Paramètres Slurm pour chaque job de groupe : {slurm_params}")
    logger.info(f"Avec {len(GROUPS_TO_PROCESS)} groupes, cela nécessitera {len(GROUPS_TO_PROCESS)} nœuds au total")

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    submitit_job_log_folder = os.path.abspath(os.path.join(
        LOG_DIR_SUBMITIT_MASTER, f"slurm_jobs_ALL_{len(GROUPS_TO_PROCESS)}_GROUPS_INTRA_{current_time_str}"
    ))
    os.makedirs(submitit_job_log_folder, exist_ok=True)
    logger.info(f"Logs Submitit spécifiques aux jobs Slurm dans : {submitit_job_log_folder}")

    executor = submitit.AutoExecutor(folder=submitit_job_log_folder)
    executor.update_parameters(**slurm_params)

    group_jobs_info = []

    logger.info("Configuration de la soumission en batch des jobs de groupe...")
    try:
        with executor.batch():
            for group_name_to_analyze in GROUPS_TO_PROCESS:
                if group_name_to_analyze not in ALL_SUBJECT_GROUPS:
                    logger.warning(f"Groupe '{group_name_to_analyze}' non trouvé dans la configuration. Ignoré.")
                    continue

                subjects_for_this_group = ALL_SUBJECT_GROUPS[group_name_to_analyze]
                if not subjects_for_this_group:
                    logger.warning(f"Aucun sujet défini pour le groupe '{group_name_to_analyze}'. Ignoré.")
                    continue

                logger.info(f"Préparation du job pour le groupe '{group_name_to_analyze}' ({len(subjects_for_this_group)} sujets)")

               
                n_jobs_for_ops_in_job = SLURM_CPUS_PER_GROUP_JOB

                func_kwargs_group_analysis = {
                    "subject_ids_in_group": subjects_for_this_group,
                    "group_identifier": group_name_to_analyze,
                    "decoding_protocol_identifier": f'Group_Intra_{group_name_to_analyze}',
                    "base_input_data_path": base_input_path,
                    "base_output_results_path": base_output_path,
                    "enable_verbose_logging": True,
                    "n_jobs_for_each_subject": n_jobs_for_ops_in_job,

                   
                    "classifier_type_for_group_runs": CLASSIFIER_MODEL_TYPE,
                    "use_grid_search_for_group": USE_GRID_SEARCH_OPTIMIZATION,
                    "use_csp_for_temporal_group": USE_CSP_FOR_TEMPORAL_PIPELINES,
                    "use_anova_fs_for_temporal_group": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
                    "param_grid_config_for_group": PARAM_GRID_CONFIG_EXTENDED,
                    "cv_folds_for_gs_group": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
                    "fixed_params_for_group": FIXED_CLASSIFIER_PARAMS_CONFIG,
                    "compute_tgm_for_group_subjects_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,

              
                    "compute_intra_subject_stats_for_group_runs_flag": True,
                    "n_perms_intra_subject_folds_for_group_runs": N_PERMUTATIONS_INTRA_SUBJECT,
                    "cluster_threshold_config_intra_fold_group": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,

          
                    "save_results_flag": SAVE_ANALYSIS_RESULTS,
                    "generate_plots_flag": GENERATE_PLOTS,
                    "loading_conditions_config": CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
                }

           
                job = executor.submit(execute_group_decoding_wrapper, **func_kwargs_group_analysis)
                group_jobs_info.append({"job_object": job, "group_name": group_name_to_analyze})
                logger.info(f"    -> Job pour le groupe '{group_name_to_analyze}' soumis au batch.")

    except Exception as e:
        logger.critical(f"Erreur critique lors de la configuration ou de la soumission du batch : {e}", exc_info=True)
        return

    if not group_jobs_info:
        logger.warning("Aucun job n'a été soumis.")
        return

    logger.info(f"{len(group_jobs_info)} job(s) de groupe soumis avec succès. Attente des résultats...")

    group_analysis_results_summary = {}

    for item in group_jobs_info:
        job_obj = item["job_object"]
        group_name = item["group_name"]
        job_id_str = str(getattr(job_obj, 'job_id', 'ID_INCONNU'))
        logger.info(f"Attente du résultat pour le groupe : {group_name} (Job ID: {job_id_str})...")

        try:
   
            output_subject_auc_dict = job_obj.result()
            job_final_state = job_obj.state
            logger.info(f"Job {job_id_str} (Groupe: {group_name}) terminé. État final : {job_final_state}")

            mean_auc_for_group = np.nan
            num_subjects_processed = 0
            if isinstance(output_subject_auc_dict, dict) and output_subject_auc_dict:
                valid_aucs = [auc for auc in output_subject_auc_dict.values() if pd.notna(auc)]
                if valid_aucs:
                    mean_auc_for_group = np.mean(valid_aucs)
                num_subjects_processed = len(output_subject_auc_dict)
                logger.info(f"  -> Résultat pour {group_name} : Moyenne AUCs = {mean_auc_for_group:.3f} (N={num_subjects_processed} sujets).")
            else:
                logger.warning(f"  -> Résultat inattendu/vide pour {group_name}. Type reçu : {type(output_subject_auc_dict)}")

            group_analysis_results_summary[group_name] = {
                'mean_auc': mean_auc_for_group,
                'num_subjects': num_subjects_processed,
                'job_id': job_id_str,
                'state': job_final_state
            }
            
        except FailedJobError as e_failed:
            logger.error(f"Le job pour le groupe {group_name} (ID: {job_id_str}) A ÉCHOUÉ.", exc_info=False)
            logger.error(f"  Message de Submitit : {e_failed}")
            logger.error(f"  Veuillez consulter les logs du worker dans : {submitit_job_log_folder}")
            group_analysis_results_summary[group_name] = {
                'mean_auc': np.nan, 'num_subjects': 0, 'job_id': job_id_str, 'state': 'FAILED'}
                
        except Exception as e_result:
            logger.error(f"Erreur lors de la récupération du résultat pour le groupe {group_name} (ID: {job_id_str}) : {e_result}", exc_info=True)
            group_analysis_results_summary[group_name] = {
                'mean_auc': np.nan, 'num_subjects': 0, 'job_id': job_id_str, 
                'state': f'ERROR_RETRIEVAL (état submitit: {getattr(job_obj, "state", "N/A")})'}

    logger.info("\n" + "="*25 + " RÉSUMÉ FINAL DES ANALYSES DE GROUPE " + "="*25)
    all_jobs_ok = True
    for group, summary in group_analysis_results_summary.items():
        auc_str = f"{summary['mean_auc']:.3f}" if pd.notna(summary['mean_auc']) else 'N/A'
        logger.info(f"Groupe: {group:<10} | Job ID: {summary['job_id']:<15} | État: {summary['state']:<10} | Moyenne AUC: {auc_str:<5} | N Sujets: {summary['num_subjects']}")
        if summary['state'] != 'DONE':
            all_jobs_ok = False

    if all_jobs_ok and group_analysis_results_summary:
        logger.info("Toutes les analyses de groupe semblent s'être terminées avec succès.")
    else:
        logger.warning("Au moins un job de groupe a échoué ou a rencontré une erreur. Veuillez vérifier les logs.")
    logger.info("="*81)



if __name__ == "__main__":
    logger.info(f"--- Démarrage du script de soumission principal ({os.path.basename(__file__)}) ---")
    try:
        main_submission_logic()
    except Exception as e_main:
        logger.critical(f"Erreur non gérée dans main_submission_logic : {e_main}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"--- Fin du script de soumission principal. Log complet dans : {MASTER_LOG_FILE_PATH} ---")
