import os
import sys
import logging
from datetime import datetime
import getpass
import numpy as np
import pandas as pd
import time
import submitit

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
    print(f"AVERTISSEMENT: __file__ non défini, SCRIPT_DIR mis à {SCRIPT_DIR}")

PROJECT_ROOT_FOR_IMPORTS = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "Baking_EEG"))
if PROJECT_ROOT_FOR_IMPORTS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_IMPORTS)

logger_temp_init = logging.getLogger("SubmititMasterInit")
logger_temp_init.info(
    f"PROJECT_ROOT_FOR_IMPORTS mis à: {PROJECT_ROOT_FOR_IMPORTS}")

log_dir_submitit_master = './logs_submitit_master/'
os.makedirs(log_dir_submitit_master, exist_ok=True)
master_log_file = os.path.join(
    log_dir_submitit_master,
    
    datetime.now().strftime(f'master_submitit_CONTROLS_ONLY_INTRA_%Y-%m-%d_%H-%M-%S.log')
)

root_logger_submitit = logging.getLogger()
if root_logger_submitit.hasHandlers():
    for handler in root_logger_submitit.handlers[:]:
        root_logger_submitit.removeHandler(handler)
        handler.close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(master_log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    from examples.run_decoding_one_group_pp import (
        execute_group_intra_subject_decoding_analysis,
        configure_project_paths,

        CLASSIFIER_MODEL_TYPE,
        USE_GRID_SEARCH_OPTIMIZATION,
        USE_CSP_FOR_TEMPORAL_PIPELINES,
        USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        PARAM_GRID_CONFIG_EXTENDED,
        CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
        FIXED_CLASSIFIER_PARAMS_CONFIG,
        COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,

        N_PERMUTATIONS_INTRA_SUBJECT,
        COMPUTE_INTRA_SUBJECT_STATISTICS,
        N_PERMUTATIONS_GROUP_LEVEL,
        GROUP_LEVEL_STAT_THRESHOLD_TYPE,
        T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,

        CONFIG_LOAD_SINGLE_PROTOCOL,
        SAVE_ANALYSIS_RESULTS,
        GENERATE_PLOTS,

    )
    from config.config import ALL_SUBJECT_GROUPS
    logger.info(
        "Importations depuis 'examples.run_decoding_one_group_pp' réussies.")
except ModuleNotFoundError as e:
    logger.critical(
        f"ERREUR CRITIQUE: Module 'examples.run_decoding_one_group_pp' non trouvé.")
    logger.critical(
        f"  Erreur: {e}\n  SCRIPT_DIR: {SCRIPT_DIR}\n  PROJECT_ROOT_FOR_IMPORTS: {PROJECT_ROOT_FOR_IMPORTS}\n  sys.path: {sys.path}")
    sys.exit(1)
except ImportError as e_imp:
    logger.critical(
        f"ERREUR CRITIQUE: Problème d'importation depuis 'examples.run_decoding_one_group_pp': {e_imp}")
    sys.exit(1)

PATH_TO_VENV_ACTIVATE = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"  # À vérifier
logger.info(
    f"Utilisation de l'environnement virtuel Python pour les jobs Slurm CPU: {PATH_TO_VENV_ACTIVATE}")

PROJECT_ROOT_ON_CLUSTER = PROJECT_ROOT_FOR_IMPORTS
logger.info(
    f"Racine du projet pour PYTHONPATH dans les jobs Slurm: {PROJECT_ROOT_ON_CLUSTER}")

SETUP_COMMANDS_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm (PID: $$) ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial: $(pwd)"
if command -v deactivate &> /dev/null ; then echo "Tentative de désactivation venv existant..."; deactivate || true; fi
module purge
echo "Modules purgés."
echo "Activation de l'environnement virtuel: {PATH_TO_VENV_ACTIVATE}"
source {PATH_TO_VENV_ACTIVATE}
if [ $? -ne 0 ]; then echo "ERREUR: Échec de l'activation de l'environnement virtuel."; exit 1; fi
echo "Configuration de PYTHONPATH pour le job..."
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER}:${{PYTHONPATH}}"
echo "PYTHONPATH actuel du job: $PYTHONPATH"
echo "--- Environnement CPU configuré (venv) ---"
echo "Chemin Python utilisé: $(which python)"
echo "Version Python: $(python -V)"
python -c "import sys; print(f'Chemins sys.path dans Python: {{sys.path}}')"
python -c "import mne; print(f'MNE version: {{mne.__version__}}'); print(f'MNE path: {{mne.__file__}}')"
python -c "import sklearn; print(f'scikit-learn version: {{sklearn.__version__}}'); print(f'sklearn path: {{sklearn.__file__}}')"
python -c "import numpy; print(f'NumPy version: {{numpy.__version__}}'); print(f'NumPy path: {{numpy.__file__}}')"
echo "-------------------------------------------"
"""


def main():
    try:
        user = getpass.getuser()
        logger.info(f"Utilisateur détecté par getpass.getuser(): {user}")
    except Exception:
        user = os.environ.get('LOGNAME') or os.environ.get(
            'USER', "unknown_user_env")
        if user != "unknown_user_env":
            logger.info(
                f"Utilisateur détecté via variable d'environnement: {user}")
        else:
            logger.warning("Impossible de déterminer l'utilisateur.")

    base_input_path, base_output_path = configure_project_paths(user)
    if not os.path.isdir(base_input_path):
        logger.error(
            f"ERREUR CRITIQUE: Le chemin des données d'entrée '{base_input_path}' n'existe pas. Arrêt.")
        sys.exit(1)
    logger.info(f"Chemin de base des données d'entrée: {base_input_path}")
    logger.info(f"Chemin de base des résultats: {base_output_path}")


    GROUPS_TO_PROCESS = ['controls']
    logger.info(f"Groupes ciblés (modifié pour test): {GROUPS_TO_PROCESS}")



    SLURM_CPUS_PER_GROUP_JOB = 40  
    logger.info(
        f"SLURM_CPUS_PER_GROUP_JOB (cpus_per_task pour Slurm) sera: {SLURM_CPUS_PER_GROUP_JOB}")

    group_job_slurm_params = {
        "timeout_min": 60 * 60,
        "slurm_additional_parameters": {"account": "tom.balay"},
        "setup": SETUP_COMMANDS_CPU,
        "slurm_partition": "CPU",
        "slurm_mem": "80G",
        "slurm_cpus_per_task": SLURM_CPUS_PER_GROUP_JOB,
    }
    logger.info(
        f"Paramètres Slurm pour chaque job de groupe: {group_job_slurm_params}")

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_folder_submitit_jobs = os.path.abspath(

        os.path.join(log_dir_submitit_master,
                     f"slurm_jobs_CONTROLS_ONLY_INTRA_{current_time_str}")
    )

    executor = submitit.AutoExecutor(folder=log_folder_submitit_jobs)
    logger.info(
        f"Submitit AutoExecutor initialisé. Logs des jobs dans: {log_folder_submitit_jobs}")
    executor.update_parameters(**group_job_slurm_params)

    group_jobs_info = []

    try:
        with executor.batch():
            for group_name_to_analyze in GROUPS_TO_PROCESS:  
                if group_name_to_analyze not in ALL_SUBJECT_GROUPS:
                    logger.warning(
                        f"Groupe '{group_name_to_analyze}' non trouvé dans ALL_SUBJECT_GROUPS. Ignoré.")
                    continue
                subjects_for_this_group_job = ALL_SUBJECT_GROUPS[group_name_to_analyze]
                if not subjects_for_this_group_job:
                    logger.warning(
                        f"Aucun sujet pour le groupe '{group_name_to_analyze}'. Ignoré.")
                    continue

                logger.info(
                    f"Configuration du job pour groupe: {group_name_to_analyze} ({len(subjects_for_this_group_job)} sujets)")

        
                n_jobs_for_ops_in_job = group_job_slurm_params["slurm_cpus_per_task"]
                logger.info(
                    f"    Utilisation de n_jobs={n_jobs_for_ops_in_job} pour les opérations parallèles dans ce job de groupe.")

                func_kwargs_group_analysis = {
                    "subject_ids_in_group": subjects_for_this_group_job,
                    "group_identifier": group_name_to_analyze,
                    "decoding_protocol_identifier": f'Group_Intra_{group_name_to_analyze}',
                    "base_input_data_path": base_input_path,
                    "base_output_results_path": base_output_path,
                    "enable_verbose_logging": True,
                    "n_jobs_for_each_subject": n_jobs_for_ops_in_job,
                    "n_jobs_for_group_cluster_stats": n_jobs_for_ops_in_job,

                    "classifier_type_for_group_runs": CLASSIFIER_MODEL_TYPE,
                    "use_grid_search_for_group": USE_GRID_SEARCH_OPTIMIZATION,
                    "use_csp_for_temporal_group": USE_CSP_FOR_TEMPORAL_PIPELINES,
                    "use_anova_fs_for_temporal_group": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
                    "param_grid_config_for_group": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
                    "cv_folds_for_gs_group": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
                    "fixed_params_for_group": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
                    "compute_tgm_for_group_subjects_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,

                    "compute_intra_subject_stats_for_group_runs_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
                    "n_perms_intra_subject_folds_for_group_runs": N_PERMUTATIONS_INTRA_SUBJECT,
                    "cluster_threshold_config_intra_fold_group": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
                    "compute_group_level_stats_flag": True,
                    "n_perms_for_group_cluster_test": N_PERMUTATIONS_GROUP_LEVEL,
                    "group_cluster_test_threshold_method": GROUP_LEVEL_STAT_THRESHOLD_TYPE,
                    "group_cluster_test_t_thresh_value": T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,

                    "save_results_flag": SAVE_ANALYSIS_RESULTS,
                    "generate_plots_flag": GENERATE_PLOTS,
                    "loading_conditions_config": CONFIG_LOAD_SINGLE_PROTOCOL,
                }

                job = executor.submit(
                    execute_group_intra_subject_decoding_analysis, **func_kwargs_group_analysis)
                group_jobs_info.append(
                    {"job_object": job, "group_name": group_name_to_analyze, "submitted_job_id": None})
                logger.info(
                    f"    Job pour groupe {group_name_to_analyze} soumis au batch.")

    except Exception as e_batch_config:
        logger.critical(
            f"Erreur lors de la configuration/soumission du batch: {e_batch_config}", exc_info=True)
        return

    if not group_jobs_info:
        logger.warning("Aucun job soumis.")
        return

    logger.info(
        f"{len(group_jobs_info)} job(s) soumis. Attente des IDs Slurm...")

    if isinstance(executor, (submitit.SlurmExecutor, submitit.AutoExecutor)) and getattr(executor, 'cluster', None) == "slurm":
        time.sleep(5)
        for item in group_jobs_info:
            job_obj = item["job_object"]
            slurm_job_id_str = "Non disponible"
            try:
                if hasattr(job_obj, 'job_id') and job_obj.job_id:
                    slurm_job_id_str = str(job_obj.job_id)
                elif hasattr(job_obj, '_job_id') and job_obj._job_id:
                    slurm_job_id_str = str(job_obj._job_id)
                item["submitted_job_id"] = slurm_job_id_str
            except Exception:
                pass
            logger.info(
                f"  Groupe: {item['group_name']}, ID Slurm: {slurm_job_id_str}, État submitit: {getattr(job_obj, 'state', 'N/A')}")

    logger.info("\nAttente des résultats des jobs...")
    group_analysis_results_summary = {}

    for item in group_jobs_info:
        job_obj = item["job_object"]
        group_name_done = item["group_name"]
        job_id_for_log_final = item.get("submitted_job_id", "ID_INCONNU")
        logger.info(
            f"Attente résultat pour groupe: {group_name_done}, ID Slurm: {job_id_for_log_final}...")
        try:
            output_subject_auc_dict = job_obj.result()
            job_final_id_done_actual = str(
                job_obj.job_id or job_id_for_log_final)
            logger.info(
                f"Job {job_final_id_done_actual} (Groupe: {group_name_done}) terminé. État: {job_obj.state}")

            mean_auc_for_group = np.nan
            num_subjects_in_job_output = 0
            if isinstance(output_subject_auc_dict, dict) and output_subject_auc_dict:
                valid_aucs_from_job = [
                    auc for auc in output_subject_auc_dict.values() if pd.notna(auc)]
                if valid_aucs_from_job:
                    mean_auc_for_group = np.mean(valid_aucs_from_job)
                num_subjects_in_job_output = len(output_subject_auc_dict)
                logger.info(
                    f"  Résultat pour {group_name_done}: Moyenne AUCs sujets = {mean_auc_for_group:.3f} (N={num_subjects_in_job_output} sujets dans l'output du job).")
            else:
                logger.warning(
                    f"  Résultat pour {group_name_done}: Output inattendu/vide du job. Type: {type(output_subject_auc_dict)}")

            group_analysis_results_summary[group_name_done] = {
                'mean_of_subject_aucs': mean_auc_for_group,
                'num_subjects_in_output': num_subjects_in_job_output,
                'job_id': job_final_id_done_actual,
                'state': job_obj.state
            }
        except Exception as e_result_retrieval:
            logger.error(
                f"Erreur récup/échec job groupe {group_name_done} (ID Slurm: {job_id_for_log_final}):\n{e_result_retrieval}", exc_info=False)
            group_analysis_results_summary[group_name_done] = {
                'mean_of_subject_aucs': np.nan,
                'num_subjects_in_output': 0,
                'job_id': job_id_for_log_final,
                'state': f'ERREUR/FAILED (submitit state: {getattr(job_obj, "state", "N/A")})'
            }

    logger.info("\n--- Résumé final ---")
    all_jobs_successful = True
    for group_key_summary, res_data_summary in group_analysis_results_summary.items():
        auc_val_summary = res_data_summary.get('mean_of_subject_aucs')
        auc_str_summary = f"{auc_val_summary:.3f}" if pd.notna(
            auc_val_summary) else 'N/A'
        job_state_summary = res_data_summary.get('state', 'INCONNU')
        logger.info(f"Groupe: {group_key_summary:<10} | Job ID: {res_data_summary.get('job_id', 'N/A'):<15} | "
                    f"État: {job_state_summary:<45} | Moyenne AUCs: {auc_str_summary:<5} | "
                    f"N Sujets traités: {res_data_summary.get('num_subjects_in_output',0)}")
        if "failed" in job_state_summary.lower() or "erreur" in job_state_summary.lower():
            all_jobs_successful = False

    if all_jobs_successful and group_analysis_results_summary:
        logger.info(
            "Toutes les analyses de groupe semblent terminées avec succès.")
    elif group_analysis_results_summary:
        logger.warning(
            "Au moins un job de groupe a échoué ou a produit un output inattendu. Vérifiez les logs des jobs individuels.")
    else:
        logger.warning("Aucun résultat d'analyse de groupe à résumer.")

    logger.info(
        f"Script de soumission terminé. Log principal: {master_log_file}")
    logger.info(
        f"Logs des jobs Slurm (stdout/stderr) dans: {log_folder_submitit_jobs}")


if __name__ == "__main__":
    main()
