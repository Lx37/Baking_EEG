
import os
import sys
import logging
from datetime import datetime
import getpass
import submitit
import traceback
import platform
import socket




def setup_enhanced_logging():

   
    detailed_format = '%(asctime)s | %(levelname)8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'

   
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(detailed_format))

   
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.addHandler(console_handler)

    return logging.getLogger(__name__)



def log_info(message, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"[INFO] {message}")


def log_error(message, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.error(f"[ERROR] {message}")


def log_debug(message, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.debug(f"[DEBUG] {message}")


def log_warning(message, logger=None):

    if logger is None:
        logger = logging.getLogger(__name__)
    logger.warning(f"[WARNING] {message}")



logger = setup_enhanced_logging()

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_info(f"PROJECT_ROOT calculé: {PROJECT_ROOT}", logger)
except NameError as e:
    PROJECT_ROOT = os.path.abspath(os.getcwd())
    log_warning(
        f"Utilisation du fallback PROJECT_ROOT: {PROJECT_ROOT}, erreur: {e}", logger)


if not os.path.exists(PROJECT_ROOT):
    log_error(f"PROJECT_ROOT n'existe pas: {PROJECT_ROOT}", logger)
    raise FileNotFoundError(f"PROJECT_ROOT invalide: {PROJECT_ROOT}")


if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    log_info(f"PROJECT_ROOT ajouté au sys.path: {PROJECT_ROOT}", logger)
else:
    log_info(f"PROJECT_ROOT déjà dans sys.path", logger)

log_debug(f"sys.path (premiers éléments): {sys.path[:3]}", logger)




def validate_project_structure(project_root):

    log_info("Validation de la structure du projet...", logger)

    required_dirs = ['examples', 'config', 'utils', 'submitit']
    required_files = [
        'examples/run_decoding_one_lg.py',
        'config/config.py',
        'config/decoding_config.py',
        'utils/utils.py'
    ]


    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if os.path.exists(dir_path):
            log_info(f"✓ Dossier trouvé: {dir_name}", logger)
        else:
            log_error(f"✗ Dossier manquant: {dir_name} ({dir_path})", logger)
            return False

    
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            log_info(f"✓ Fichier trouvé: {file_path}", logger)
        else:
            log_error(f"✗ Fichier manquant: {file_path} ({full_path})", logger)
            return False

    log_info("Structure du projet validée avec succès", logger)
    return True



if not validate_project_structure(PROJECT_ROOT):
    log_error("Structure du projet invalide, arrêt du script", logger)
    sys.exit(1)




def import_configurations():
    """Importe toutes les configurations nécessaires avec gestion d'erreur."""
    log_info("Import des configurations...", logger)

    try:
        from config.config import ALL_SUBJECT_GROUPS
        log_info(
            f"✓ ALL_SUBJECT_GROUPS importé ({len(ALL_SUBJECT_GROUPS)} groupes)", logger)

        from utils.utils import configure_project_paths
        log_info("✓ configure_project_paths importé", logger)

        from config.decoding_config import (
            CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, SAVE_ANALYSIS_RESULTS,
            GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
            N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
            INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS,
            PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
            FIXED_CLASSIFIER_PARAMS_CONFIG,
            USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
        )
        log_info(
            f"✓ decoding_config importé (CLASSIFIER: {CLASSIFIER_MODEL_TYPE})", logger)

        return {
            'ALL_SUBJECT_GROUPS': ALL_SUBJECT_GROUPS,
            'configure_project_paths': configure_project_paths,
            'decoding_config': {
                'CLASSIFIER_MODEL_TYPE': CLASSIFIER_MODEL_TYPE,
                'USE_GRID_SEARCH_OPTIMIZATION': USE_GRID_SEARCH_OPTIMIZATION,
                'SAVE_ANALYSIS_RESULTS': SAVE_ANALYSIS_RESULTS,
                'GENERATE_PLOTS': GENERATE_PLOTS,
                'CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG': CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
                'N_PERMUTATIONS_INTRA_SUBJECT': N_PERMUTATIONS_INTRA_SUBJECT,
                'COMPUTE_TEMPORAL_GENERALIZATION_MATRICES': COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
                'INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG': INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
                'COMPUTE_INTRA_SUBJECT_STATISTICS': COMPUTE_INTRA_SUBJECT_STATISTICS,
                'PARAM_GRID_CONFIG_EXTENDED': PARAM_GRID_CONFIG_EXTENDED,
                'CV_FOLDS_FOR_GRIDSEARCH_INTERNAL': CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
                'FIXED_CLASSIFIER_PARAMS_CONFIG': FIXED_CLASSIFIER_PARAMS_CONFIG,
                # CSP functionality removed - no longer used
                'USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES': USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
            }
        }

    except ImportError as e:
        log_error(f"Erreur d'import: {e}", logger)
        log_error(f"Traceback: {traceback.format_exc()}", logger)
        raise
    except Exception as e:
        log_error(f"Erreur inattendue lors de l'import: {e}", logger)
        log_error(f"Traceback: {traceback.format_exc()}", logger)
        raise



configs = import_configurations()




def enhanced_decoding_task_wrapper(**kwargs):

    
    import sys
    import os
    import logging
    import traceback
    from datetime import datetime
    import platform
    import psutil


    worker_log_format = '%(asctime)s | %(levelname)8s | [WORKER:%(funcName)s:%(lineno)d] | %(message)s'
    logging.basicConfig(level=logging.DEBUG,
                        format=worker_log_format, force=True)
    worker_logger = logging.getLogger('slurm_worker')

    worker_logger.info("=" * 80)
    worker_logger.info("=== DÉBUT DU WORKER SUBMITIT - VERSION ENHANCED ===")
    worker_logger.info("=" * 80)


    worker_logger.info("=== DIAGNOSTICS SYSTÈME ===")
    worker_logger.info(f"Hostname: {socket.gethostname()}")
    worker_logger.info(f"Platform: {platform.platform()}")
    worker_logger.info(f"Python version: {sys.version}")
    worker_logger.info(f"Working directory: {os.getcwd()}")
    worker_logger.info(f"User: {os.getenv('USER', 'unknown')}")
    worker_logger.info(f"Home: {os.getenv('HOME', 'unknown')}")


    try:
        worker_logger.info(f"CPU count: {psutil.cpu_count()}")
        worker_logger.info(
            f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    except Exception as e:
        worker_logger.warning(f"Impossible d'obtenir les infos système: {e}")


    worker_logger.info("=== ARGUMENTS REÇUS ===")
    for key, value in kwargs.items():
        if isinstance(value, str) and len(value) > 100:
            worker_logger.info(f"{key}: {value[:100]}...")
        else:
            worker_logger.info(f"{key}: {value}")


    worker_logger.info("=== CONFIGURATION PATH ===")


    possible_project_paths = [
        "/home/tom.balay/Baking_EEG",
        "/home/tom.balay/Stage_CAP/BakingEEG/Baking_EEG",
        os.path.join(os.getenv('HOME', ''), 'Baking_EEG'),
        os.path.join(os.getcwd(), 'Baking_EEG')
    ]

    project_root = None
    for path in possible_project_paths:
        worker_logger.info(f"Test du chemin: {path}")
        if os.path.exists(path):
            examples_check = os.path.join(
                path, "examples", "run_decoding_one_lg.py")
            if os.path.exists(examples_check):
                project_root = path
                worker_logger.info(f"✓ Chemin valide trouvé: {project_root}")
                break
            else:
                worker_logger.warning(
                    f"Chemin existe mais structure invalide: {path}")
        else:
            worker_logger.warning(f"Chemin n'existe pas: {path}")

    if project_root is None:
        worker_logger.error(
            "ERREUR CRITIQUE: Aucun chemin projet valide trouvé")
        worker_logger.error("Listage du répertoire de travail:")
        try:
            for item in os.listdir('.'):
                worker_logger.error(f"  - {item}")
        except Exception as e:
            worker_logger.error(f"Impossible de lister le répertoire: {e}")
        raise FileNotFoundError("Aucun chemin projet valide trouvé")


    worker_logger.info(f"sys.path initial: {sys.path[:3]}...")

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        worker_logger.info(
            f"✓ Chemin projet ajouté au sys.path: {project_root}")
    else:
        worker_logger.info(f"✓ Chemin projet déjà dans sys.path")

    worker_logger.info(f"sys.path après modification: {sys.path[:5]}...")


    worker_logger.info("=== VALIDATION STRUCTURE ===")
    critical_paths = {
        'examples': os.path.join(project_root, "examples"),
        'run_decoding_script': os.path.join(project_root, "examples", "run_decoding_one_lg.py"),
        'config': os.path.join(project_root, "config"),
        'utils': os.path.join(project_root, "utils")
    }

    for name, path in critical_paths.items():
        if os.path.exists(path):
            worker_logger.info(f"✓ {name}: {path}")
        else:
            worker_logger.error(f"✗ {name} MANQUANT: {path}")
            raise FileNotFoundError(
                f"Élément critique manquant: {name} - {path}")


    worker_logger.info("=== TEST D'IMPORT ===")
    try:
        worker_logger.info("Import de run_decoding_one_lg...")
        from examples.run_decoding_one_lg import execute_single_subject_lg_decoding
        worker_logger.info(
            "✓ execute_single_subject_lg_decoding importé avec succès")


        subject_id = kwargs.get('subject_identifier', 'UNKNOWN')
        group = kwargs.get('group_affiliation', 'UNKNOWN')
        worker_logger.info(
            f"Paramètres principaux - Sujet: {subject_id}, Groupe: {group}")

        if subject_id == 'UNKNOWN' or group == 'UNKNOWN':
            worker_logger.error("Paramètres manquants ou invalides")
            raise ValueError(
                "subject_identifier ou group_affiliation manquant")

  
        worker_logger.info("=== DÉBUT EXÉCUTION ===")
        start_time = datetime.now()
        worker_logger.info(f"Heure de début: {start_time}")

        result = execute_single_subject_lg_decoding(**kwargs)

        end_time = datetime.now()
        duration = end_time - start_time
        worker_logger.info(f"Heure de fin: {end_time}")
        worker_logger.info(f"Durée d'exécution: {duration}")
        worker_logger.info(f"Type de résultat: {type(result)}")

        if hasattr(result, 'keys'):
            worker_logger.info(f"Clés du résultat: {list(result.keys())}")

        worker_logger.info("✓ EXÉCUTION TERMINÉE AVEC SUCCÈS")
        return result

    except ImportError as e:
        worker_logger.error(f"ERREUR D'IMPORT: {e}")
        worker_logger.error(f"Module recherché: examples.run_decoding_one_lg")
        worker_logger.error(f"sys.path: {sys.path}")
        worker_logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    except Exception as e:
        worker_logger.error(f"ERREUR LORS DE L'EXÉCUTION: {e}")
        worker_logger.error(f"Type d'erreur: {type(e).__name__}")
        worker_logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        worker_logger.info("=" * 80)
        worker_logger.info("=== FIN DU WORKER SUBMITIT ===")
        worker_logger.info("=" * 80)




def main():

    TARGET_SUBJECT_ID = "TJR7"

    log_info("=" * 80, logger)
    log_info(
        f"DÉMARRAGE SOUMISSION - SUJET {TARGET_SUBJECT_ID} (Protocole LG)", logger)
    log_info("=" * 80, logger)


    log_info("=== DIAGNOSTICS ENVIRONNEMENT ===", logger)
    log_info(f"Hostname: {socket.gethostname()}", logger)
    log_info(f"Python version: {sys.version}", logger)
    log_info(f"Répertoire de travail: {os.getcwd()}", logger)
    log_info(f"PROJECT_ROOT: {PROJECT_ROOT}", logger)
    log_info(f"Utilisateur: {getpass.getuser()}", logger)
    log_info(f"Platform: {platform.platform()}", logger)


    log_info("=== CONFIGURATION CHEMINS ===", logger)
    try:
        user = getpass.getuser()
        base_input_path, base_output_path = configs['configure_project_paths'](
            user)
        log_info(f"✓ Chemin d'entrée: {base_input_path}", logger)
        log_info(f"✓ Chemin de sortie: {base_output_path}", logger)


        for path_name, path in [("entrée", base_input_path), ("sortie", base_output_path)]:
            if os.path.exists(path):
                log_info(f"✓ Chemin {path_name} existe", logger)
            else:
                log_warning(f"Chemin {path_name} n'existe pas: {path}", logger)

    except Exception as e:
        log_error(f"Erreur configuration chemins: {e}", logger)
        log_error(f"Traceback: {traceback.format_exc()}", logger)
        raise


    log_info("=== RÉSOLUTION GROUPE SUJET ===", logger)
    all_groups = configs['ALL_SUBJECT_GROUPS']
    log_debug(f"Groupes disponibles: {list(all_groups.keys())}", logger)

    subject_group = None
    for group, subjects in all_groups.items():
        if TARGET_SUBJECT_ID in subjects:
            subject_group = group
            break

    if subject_group is None:
        log_error(f"ERREUR: Sujet {TARGET_SUBJECT_ID} non trouvé", logger)
        log_error("Groupes et sujets disponibles:", logger)
        for group, subjects in all_groups.items():
            display_subjects = subjects[:5] + \
                ["..."] if len(subjects) > 5 else subjects
            log_error(f"  {group}: {display_subjects}", logger)
        return False

    log_info(
        f"✓ Sujet {TARGET_SUBJECT_ID} trouvé dans le groupe: {subject_group}", logger)


    log_info("=== CONFIGURATION SUBMITIT ===", logger)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_folder = f"logs_submitit_enhanced/{timestamp}_{TARGET_SUBJECT_ID}_LG"

    log_info(f"Dossier de logs: {log_folder}", logger)
    os.makedirs(log_folder, exist_ok=True)

    executor = submitit.AutoExecutor(folder=log_folder)


    slurm_params = {
        "timeout_min": 24 * 60, 
        "slurm_partition": "CPU",
        "slurm_mem": "60G",
        "slurm_cpus_per_task": 40,
        "slurm_additional_parameters": {
            "account": "tom.balay",
            "job-name": f"LG_{TARGET_SUBJECT_ID}",
            "output": f"{log_folder}/slurm-%j.out",
            "error": f"{log_folder}/slurm-%j.err"
        }
    }

    log_info(f"Paramètres Slurm: {slurm_params}", logger)
    executor.update_parameters(**slurm_params)


    log_info("=== PRÉPARATION ARGUMENTS ===", logger)

    dc = configs['decoding_config']
    kwargs = {
        "subject_identifier": TARGET_SUBJECT_ID,
        "group_affiliation": subject_group,
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path,
        "n_jobs_for_processing": 40,
        "save_results_flag": dc['SAVE_ANALYSIS_RESULTS'],
        "generate_plots_flag": dc['GENERATE_PLOTS'],
        "loading_conditions_config": dc['CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG'],
        "classifier_type": dc['CLASSIFIER_MODEL_TYPE'],
        "use_grid_search_for_subject": dc['USE_GRID_SEARCH_OPTIMIZATION'],
        "param_grid_config_for_subject": dc['PARAM_GRID_CONFIG_EXTENDED'] if dc['USE_GRID_SEARCH_OPTIMIZATION'] else None,
        "cv_folds_for_gs_subject": dc['CV_FOLDS_FOR_GRIDSEARCH_INTERNAL'] if dc['USE_GRID_SEARCH_OPTIMIZATION'] else 0,
        "fixed_params_for_subject": dc['FIXED_CLASSIFIER_PARAMS_CONFIG'] if not dc['USE_GRID_SEARCH_OPTIMIZATION'] else None,
        "compute_intra_subject_stats_flag": dc['COMPUTE_INTRA_SUBJECT_STATISTICS'],
        "n_perms_for_intra_subject_clusters": dc['N_PERMUTATIONS_INTRA_SUBJECT'],
        "compute_tgm_flag": dc['COMPUTE_TEMPORAL_GENERALIZATION_MATRICES'],
        "cluster_threshold_config_intra_fold": dc['INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG'],
        # CSP functionality removed - no longer used
        "use_anova_fs_for_temporal_subject": dc['USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES']
    }


    log_info("Paramètres clés:", logger)
    critical_params = [
        'classifier_type', 'use_grid_search_for_subject',
        'use_anova_fs_for_temporal_subject', 'save_results_flag', 'generate_plots_flag',
        'compute_tgm_flag', 'compute_intra_subject_stats_flag'
    ]

    for param in critical_params:
        if param in kwargs:
            log_info(f"  {param}: {kwargs[param]}", logger)


    log_info("=== SOUMISSION JOB ===", logger)
    try:
        job = executor.submit(enhanced_decoding_task_wrapper, **kwargs)

        log_info("✓ Job soumis avec succès!", logger)
        log_info(f"Job ID: {job.job_id}", logger)
        log_info(
            f"Logs disponibles dans: {os.path.abspath(log_folder)}", logger)


        log_info("Attente de la completion du job...", logger)
        log_info("(Ceci peut prendre plusieurs heures)", logger)

        result = job.result()

        log_info("=" * 80, logger)
        log_info("✓ JOB TERMINÉ AVEC SUCCÈS!", logger)
        log_info(f"Type de résultat: {type(result)}", logger)
        if hasattr(result, 'keys'):
            log_info(f"Clés du résultat: {list(result.keys())}", logger)
        log_info("=" * 80, logger)

        return result

    except Exception as e:
        log_error("=" * 80, logger)
        log_error("✗ ERREUR LORS DE L'EXÉCUTION DU JOB!", logger)
        log_error(f"Erreur: {e}", logger)
        log_error(f"Type: {type(e).__name__}", logger)
        log_error(f"Traceback: {traceback.format_exc()}", logger)
        log_error(
            f"Logs détaillés dans: {os.path.abspath(log_folder)}", logger)

        # Liste des fichiers de logs
        if os.path.exists(log_folder):
            log_error("Fichiers de logs à vérifier:", logger)
            for file in os.listdir(log_folder):
                log_error(f"  - {os.path.join(log_folder, file)}", logger)

        log_error("=" * 80, logger)
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log_info("Interruption par l'utilisateur", logger)
        sys.exit(0)
    except Exception as e:
        log_error(f"Erreur fatale: {e}", logger)
        log_error(f"Traceback: {traceback.format_exc()}", logger)
        sys.exit(1)
