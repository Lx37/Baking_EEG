#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de soumission Submitit pour un sujet unique - Protocole Local-Global (LG).
Ce script configure et soumet un job Slurm pour exécuter un traitement LG sur un sujet spécifique.
"""

# --- Imports ---
import os
import sys
import logging
from datetime import datetime
import getpass
import time
import submitit
from submitit.core.utils import FailedJobError

# --- Configuration Initiale des Chemins et Variables Globales ---
try:
    # Chemin du script de soumission lui-même
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback si __file__ n'est pas défini (par exemple, dans un interpréteur interactif)
    CURRENT_SCRIPT_DIR = os.getcwd()
    print(
        f"AVERTISSEMENT: __file__ non défini. CURRENT_SCRIPT_DIR initialisé à : {CURRENT_SCRIPT_DIR}",
        file=sys.stderr
    )

# --- Détermination de PROJECT_ROOT_FOR_PYTHONPATH ---
project_root_tentative = os.path.dirname(CURRENT_SCRIPT_DIR)
print(f"INFO: Tentative de PROJECT_ROOT_FOR_PYTHONPATH (parent de CURRENT_SCRIPT_DIR): {project_root_tentative}")

# Vérifier si 'examples' est présent dans ce chemin_racine_projet_tentatif
if os.path.isdir(os.path.join(project_root_tentative, "examples")):
    PROJECT_ROOT_FOR_PYTHONPATH = project_root_tentative
    print(f"INFO: PROJECT_ROOT_FOR_PYTHONPATH confirmé (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
else:
    print(f"AVERTISSEMENT: Le dossier 'examples' n'a pas été trouvé dans '{project_root_tentative}'.")
    print(f"         Contenu de '{project_root_tentative}': {os.listdir(project_root_tentative) if os.path.exists(project_root_tentative) else 'N/A'}")

    potential_paths = [
        CURRENT_SCRIPT_DIR,
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..")),
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..", "..")),
    ]

    found_path = False
    for path_candidate in potential_paths:
        if os.path.isdir(os.path.join(path_candidate, "examples")):
            PROJECT_ROOT_FOR_PYTHONPATH = path_candidate
            print(f"INFO: PROJECT_ROOT_FOR_PYTHONPATH trouvé via fallback (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
            found_path = True
            break

    if not found_path:
        hardcoded_project_root = "/home/tom.balay/Baking_EEG" # Fallback to hardcoded path
        if os.path.isdir(os.path.join(hardcoded_project_root, "examples")):
            PROJECT_ROOT_FOR_PYTHONPATH = hardcoded_project_root
            print(f"AVERTISSEMENT: PROJECT_ROOT_FOR_PYTHONPATH défini sur une valeur codée en dur (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
        else:
            print(f"ERREUR CRITIQUE: Impossible de localiser le dossier racine du projet contenant 'examples'.")
            print(f"                 CURRENT_SCRIPT_DIR: {CURRENT_SCRIPT_DIR}")
            print(f"                 Chemins testés: {[project_root_tentative] + potential_paths}")
            sys.exit(1)

# Vérifier que le dossier 'examples' est un module Python valide
examples_dir = os.path.join(PROJECT_ROOT_FOR_PYTHONPATH, "examples")
if not os.path.isfile(os.path.join(examples_dir, "__init__.py")):
    print(f"AVERTISSEMENT: Le dossier 'examples' n'est pas un module Python valide (manque __init__.py).")
    print(f"                Vous pouvez créer un fichier vide __init__.py dans {examples_dir}.")
else:
    print(f"INFO: Le dossier 'examples' est un module Python valide (contient __init__.py).")

# Ajout de PROJECT_ROOT_FOR_PYTHONPATH à sys.path AVANT tout autre import de module du projet
if PROJECT_ROOT_FOR_PYTHONPATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_PYTHONPATH)
    print(f"INFO: '{PROJECT_ROOT_FOR_PYTHONPATH}' a été ajouté à sys.path.")
else:
    print(f"INFO: '{PROJECT_ROOT_FOR_PYTHONPATH}' est déjà dans sys.path.")

print(f"INFO: Contenu de PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}'): {os.listdir(PROJECT_ROOT_FOR_PYTHONPATH)}")
print(f"INFO: sys.path commence par: {sys.path[:3]}...")

# Testez l'importation du module 'examples' pour vérifier que tout est correct
try:
    import examples
    print("INFO: Le module 'examples' a été importé avec succès.")
except ModuleNotFoundError as e:
    print(f"ERREUR: Impossible d'importer le module 'examples': {e}")
    print(f"         Assurez-vous que le dossier 'examples' est un module Python valide et que le chemin '{PROJECT_ROOT_FOR_PYTHONPATH}' est correct.")
    sys.exit(1)

# --- Configuration du Logger Principal ---
LOG_DIR_SUBMITIT_MASTER = os.path.join(CURRENT_SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(LOG_DIR_SUBMITIT_MASTER, exist_ok=True)

TARGET_SUBJECT_ID_FOR_JOB = "TpSM49"
TARGET_PROTOCOL_TYPE_FOR_JOB = "LG"

MASTER_LOG_FILE_NAME = datetime.now().strftime(
    f'master_submitit_SINGLE_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_%Y-%m-%d_%H-%M-%S.log'
)
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

logger.info(f"--- Script de soumission Submitit pour sujet unique (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) démarré ---")
logger.info(f"Sujet Cible: {TARGET_SUBJECT_ID_FOR_JOB}")
logger.info(f"Log principal de ce script de soumission: {MASTER_LOG_FILE_PATH}")
logger.info(f"CURRENT_SCRIPT_DIR (où ce script est exécuté): {CURRENT_SCRIPT_DIR}")
logger.info(f"PROJECT_ROOT_FOR_PYTHONPATH (ajouté à sys.path): {PROJECT_ROOT_FOR_PYTHONPATH}")

# Import only configuration modules here, NOT the main execution function
try:
    from config.config import ALL_SUBJECT_GROUPS
    from utils.utils import configure_project_paths
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_CSP_FOR_TEMPORAL_PIPELINES,
        USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES, PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
        FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS, SAVE_ANALYSIS_RESULTS,
        GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG, CHANCE_LEVEL_AUC_SCORE
    )
    logger.info("Importations depuis 'config' et 'utils' réussies.")
except ModuleNotFoundError as e_mod:
    logger.critical(f"ERREUR CRITIQUE: Module introuvable: {e_mod.name}. Vérifiez PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}') et sys.path: {sys.path}", exc_info=True)
    sys.exit(1)
except ImportError as e_imp:
    logger.critical(f"ERREUR CRITIQUE: Nom introuvable lors de l'import: {e_imp.name}. Vérifiez les définitions dans les modules.", exc_info=True)
    sys.exit(1)

PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
logger.info(f"Chemin vers 'activate' de l'environnement virtuel sur le cluster: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}")

PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT_FOR_PYTHONPATH
logger.info(f"Racine du projet à utiliser pour PYTHONPATH dans le job Slurm: {PROJECT_ROOT_ON_CLUSTER_FOR_JOB}")

# Python code for detailed diagnostics on the worker node
python_diagnostic_script_content = f"""
import sys, os
print("--- Python Diagnostic Script on Worker (from temp file) STARTS ---")
print(f"DEBUG_WORKER_FILE: Python Executable: {{sys.executable}}")
print(f"DEBUG_WORKER_FILE: Python Version: {{sys.version.replace('\\n', ' ')}}")
print(f"DEBUG_WORKER_FILE: Current Working Dir: {{os.getcwd()}}")
print(f"DEBUG_WORKER_FILE: os.environ['PYTHONPATH'] as seen by this Python: '{{os.environ.get('PYTHONPATH', 'PYTHONPATH Not Set')}}'")
print(f"DEBUG_WORKER_FILE: sys.path at START of diagnostic: {{sys.path}}")

project_root_expected_in_pythonpath = "{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}" # Injected by outer f-string
print(f"DEBUG_WORKER_FILE: Project root expected in PYTHONPATH: {{project_root_expected_in_pythonpath}}")

if project_root_expected_in_pythonpath in sys.path:
    print(f"DEBUG_WORKER_FILE: SUCCESS - Project root '{{project_root_expected_in_pythonpath}}' IS in sys.path.")
else:
    print(f"DEBUG_WORKER_FILE: WARNING - Project root '{{project_root_expected_in_pythonpath}}' IS NOT in sys.path.")
    if os.path.isdir(project_root_expected_in_pythonpath):
        sys.path.insert(0, project_root_expected_in_pythonpath)
        print(f"DEBUG_WORKER_FILE: Manually prepended.")
        print(f"DEBUG_WORKER_FILE: sys.path after manual prepend: {{sys.path}}")
    else:
        print(f"DEBUG_WORKER_FILE: ERROR - Project root '{{project_root_expected_in_pythonpath}}' is not a valid directory.")

print("DEBUG_WORKER_FILE: Attempting to import 'examples' module...")
try:
    import examples
    print(f"DEBUG_WORKER_FILE: SUCCESS - 'import examples' worked.")
    if hasattr(examples, '__file__'): print(f"DEBUG_WORKER_FILE: examples.__file__ is '{{examples.__file__}}'")
    if hasattr(examples, 'run_decoding_one_lg'): print(f"DEBUG_WORKER_FILE: examples.run_decoding_one_lg exists.")
    else: print(f"DEBUG_WORKER_FILE: WARNING - examples.run_decoding_one_lg NOT FOUND.")
except ModuleNotFoundError as e:
    print(f"DEBUG_WORKER_FILE: FAILED - 'import examples' ModuleNotFoundError: {{e}}")
except Exception as e:
    print(f"DEBUG_WORKER_FILE: FAILED - 'import examples' Exception: {{e}}")
print("--- Python Diagnostic Script on Worker (from temp file) ENDS ---")
"""

# Prepare the diagnostic script content for embedding in a Bash heredoc
python_diagnostic_script_content_for_heredoc = python_diagnostic_script_content.replace("{", "{{").replace("}", "}}")
python_diagnostic_script_content_for_heredoc = python_diagnostic_script_content_for_heredoc.replace(f"{{{{{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}}}}}", f"{{{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}}}")

# Ajout de logs de diagnostic supplémentaires pour vérifier le chemin du projet sur le nœud de calcul
SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""#!/bin/bash
# ... (initial SBATCH directives and echos) ...
echo "Initial PYTHONPATH: [${{PYTHONPATH:-Not Set}}]"

set -e
set -u
set -o pipefail

# Vérifier le chemin du projet sur le nœud de calcul
echo "INFO: Checking project root directory on worker node..."
if [ -d "{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}" ]; then
    echo "INFO: Project root directory exists on worker node."
    echo "INFO: Contents of project root directory:"
    ls -la "{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}"
    if [ -d "{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}/examples" ]; then
        echo "INFO: 'examples' directory exists in project root."
    else
        echo "ERROR: 'examples' directory does not exist in project root."
    fi
else
    echo "ERROR: Project root directory does not exist on worker node."
fi

# Activer l'environnement virtuel
echo "INFO: Activating virtual environment..."
source {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}
echo "INFO: Virtual environment activated. Python path: $(which python)"
echo "INFO: Active Python version: $(python --version)"

echo "INFO: Configuring PYTHONPATH for the job..."
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}${{PYTHONPATH:+:$PYTHONPATH}}"
echo "INFO: PYTHONPATH for job set to: [$PYTHONPATH]"

echo "INFO: Preparing and running detailed Python diagnostic script from temp file..."
THE_PYTHON_INTERPRETER=$(which python)
if [ -z "$THE_PYTHON_INTERPRETER" ]; then
    echo "CRITICAL ERROR: 'which python' returned empty. Exiting."
    exit 1
fi
echo "INFO: Using Python interpreter for diagnostic: $THE_PYTHON_INTERPRETER"

# Create a temporary Python script
cat << EOF > diagnostic_script_on_worker.py
{python_diagnostic_script_content_for_heredoc}
EOF

echo "INFO: Content of diagnostic_script_on_worker.py:"
cat diagnostic_script_on_worker.py # Log the content for debugging
echo "INFO: Executing diagnostic_script_on_worker.py..."
"$THE_PYTHON_INTERPRETER" diagnostic_script_on_worker.py
rm diagnostic_script_on_worker.py # Clean up
echo "INFO: Detailed Python diagnostic script finished."

echo "--- Slurm Job Setup Script COMPLETED ---"
echo "INFO: Proceeding to execute the submitit task..."
"""

# Wrapper function that imports the execution function inside the worker
def execute_group_decoding_wrapper(**kwargs):
    """
    Wrapper qui importe la fonction d'analyse de groupe à l'intérieur du worker.
    Ceci évite les problèmes de pickling des modules et assure que les chemins sont corrects sur le worker.
    """
    import sys
    import os
    
    # Assurer que la racine du projet est bien dans le path du worker
    project_root = PROJECT_ROOT_ON_CLUSTER_FOR_JOB  # Ex: /home/tom.balay/Baking_EEG
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # CRUCIAL: Ajouter le répertoire parent pour les imports absolus "Baking_EEG.xxx"
    parent_dir = os.path.dirname(project_root)  # Ex: /home/tom.balay/
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Debug: afficher les chemins
    print(f"DEBUG Worker - project_root: {project_root}")
    print(f"DEBUG Worker - parent_dir: {parent_dir}")
    print(f"DEBUG Worker - sys.path premiers éléments: {sys.path[:3]}")
    
    # Test d'import pour vérifier que ça marche
    try:
        import Baking_EEG.utils.utils
        print("DEBUG Worker - Import Baking_EEG.utils.utils réussi")
    except ImportError as e:
        print(f"DEBUG Worker - Échec import Baking_EEG.utils.utils: {e}")
    
    # Importer et exécuter la fonction cible
    from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
    return execute_group_intra_subject_decoding_analysis(**kwargs)

def main_submission_logic():
    logger.info("--- Début de la logique de soumission principale (main_submission_logic) pour protocole LG ---")

    try:
        user_for_paths = getpass.getuser()
        logger.info(f"Utilisateur détecté (getpass): {user_for_paths}")
    except Exception as e_getpass:
        logger.warning(f"getpass.getuser() a échoué ({e_getpass}). Tentative avec LOGNAME/USER.")
        user_for_paths = os.environ.get('LOGNAME') or os.environ.get('USER') or "unknown_user_fallback"
        logger.info(f"Utilisateur (fallback): {user_for_paths}")

    base_input_path, base_output_path = configure_project_paths(user_for_paths)
    if not os.path.isdir(base_input_path):
        logger.critical(f"ERREUR: Chemin d'entrée '{base_input_path}' invalide. Arrêt.")
        sys.exit(1)
    logger.info(f"Chemin de base des données d'entrée: {base_input_path}")
    logger.info(f"Chemin de base des résultats de sortie: {base_output_path}")

    target_subject_group = None
    if ALL_SUBJECT_GROUPS:
        for group, subjects_in_group in ALL_SUBJECT_GROUPS.items():
            if TARGET_SUBJECT_ID_FOR_JOB in subjects_in_group:
                target_subject_group = group
                break
    if target_subject_group is None:
        logger.critical(f"ERREUR: Sujet '{TARGET_SUBJECT_ID_FOR_JOB}' non trouvé dans ALL_SUBJECT_GROUPS. Arrêt.")
        sys.exit(1)
    logger.info(f"Groupe du sujet '{TARGET_SUBJECT_ID_FOR_JOB}': {target_subject_group}")

    N_CPUS_FOR_JOB = 40
    MEMORY_FOR_JOB = "60G"
    TIMEOUT_MINUTES = 12 * 60
    SLURM_PARTITION = "CPU"
    SLURM_ACCOUNT = "tom.balay"

    logger.info(
        f"Ressources Slurm: CPUs={N_CPUS_FOR_JOB}, Mémoire={MEMORY_FOR_JOB}, Timeout={TIMEOUT_MINUTES}min, "
        f"Partition={SLURM_PARTITION}, Account={SLURM_ACCOUNT or 'N/A'}"
    )

    slurm_params = {
        "timeout_min": TIMEOUT_MINUTES,
        "local_setup": SETUP_COMMANDS_FOR_SLURM_JOB_CPU.splitlines(), # Critical: ensure this setup runs!
        "slurm_partition": SLURM_PARTITION,
        "slurm_mem": MEMORY_FOR_JOB,
        "slurm_cpus_per_task": N_CPUS_FOR_JOB,
    }
    if SLURM_ACCOUNT:
        slurm_params["slurm_additional_parameters"] = {"account": SLURM_ACCOUNT}
    logger.info(f"Paramètres Slurm finaux pour le job: {slurm_params}")

    current_timestamp_for_log = datetime.now().strftime('%Y%m%d_%H%M%S')
    submitit_job_log_folder = os.path.join(
        CURRENT_SCRIPT_DIR, "logs_submitit_jobs",
        f"job_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_{current_timestamp_for_log}"
    )
    os.makedirs(submitit_job_log_folder, exist_ok=True)
    logger.info(f"Logs Submitit spécifiques au job Slurm dans: {os.path.abspath(submitit_job_log_folder)}")

    executor = submitit.AutoExecutor(folder=submitit_job_log_folder)
    executor.update_parameters(**slurm_params)

    loading_config_for_job = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG
    logger.info(f"Config de chargement des données (LG): CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG")

    kwargs_for_function_call = {
        "subject_identifier": TARGET_SUBJECT_ID_FOR_JOB,
        "group_affiliation": target_subject_group,
        "decoding_protocol_identifier": f'Analysis_{TARGET_PROTOCOL_TYPE_FOR_JOB}_Individual',
        "save_results_flag": SAVE_ANALYSIS_RESULTS, 
        "enable_verbose_logging": True,
        "generate_plots_flag": GENERATE_PLOTS, 
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path, 
        "n_jobs_for_processing": N_CPUS_FOR_JOB,
        "classifier_type": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_subject": USE_GRID_SEARCH_OPTIMIZATION,
        "use_csp_for_temporal_subject": USE_CSP_FOR_TEMPORAL_PIPELINES,
        "use_anova_fs_for_temporal_subject": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        "param_grid_config_for_subject": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_subject": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
        "fixed_params_for_subject": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
        "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "loading_conditions_config": loading_config_for_job,
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    }
    logger.info(f"Arguments (clés) pour execute_single_subject_lg_decoding: {list(kwargs_for_function_call.keys())}")

    submitted_job = None
    try:
        logger.info(f"Soumission du job pour {TARGET_SUBJECT_ID_FOR_JOB}, Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}...")
        # Submit the wrapper function instead of the direct import
        submitted_job = executor.submit(execute_single_subject_decoding_lg_wrapper, **kwargs_for_function_call)
        logger.info(f"Job soumis: {submitted_job}")
        if hasattr(submitted_job, 'job_id') and submitted_job.job_id:
            slurm_job_id_str = str(submitted_job.job_id)
            logger.info(f"ID (Slurm/Local) du job: {slurm_job_id_str}")
            job_id_file = os.path.join(submitit_job_log_folder, f"slurm_job_id_{slurm_job_id_str}.info")
            with open(job_id_file, "w") as f_id:
                f_id.write(f"Slurm Job ID: {slurm_job_id_str}\nSujet: {TARGET_SUBJECT_ID_FOR_JOB}\nProtocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}\nSoumis le: {datetime.now().isoformat()}\nLogs: {os.path.abspath(submitit_job_log_folder)}\n")
            logger.info(f"Infos du job sauvegardées dans: {job_id_file}")
    except Exception as e_submit:
        logger.critical(f"Erreur lors de la soumission du job pour {TARGET_SUBJECT_ID_FOR_JOB}: {e_submit}", exc_info=True)
        return

    if not submitted_job:
        logger.error("Soumission échouée, aucun objet job retourné.")
        return

    logger.info(f"--- Attente du résultat du job (ID Submitit: {getattr(submitted_job, 'job_id', 'N/A')}) ---")
    try:
        subject_job_results = submitted_job.result()
        job_id_final = str(getattr(submitted_job, 'job_id', 'ID_Final_Inconnu'))
        logger.info(f"Job {job_id_final} ({TARGET_SUBJECT_ID_FOR_JOB}) terminé. État: {submitted_job.state}")

        # Result processing logic for LG protocol
        import numpy as np 
        import pandas as pd
        if isinstance(subject_job_results, dict) and \
           subject_job_results.get("subject_id") == TARGET_SUBJECT_ID_FOR_JOB and \
           subject_job_results.get("protocol_type_processed") == TARGET_PROTOCOL_TYPE_FOR_JOB:
            logger.info(f"Résultats LG valides reçus pour {TARGET_SUBJECT_ID_FOR_JOB}.")
            
            # Log specific LG results structure
            if "lg_main_decoding_results" in subject_job_results:
                logger.info(f"Résultats de décodage principal LG (LS vs LD) trouvés.")
            if "lg_specific_effects_results" in subject_job_results:
                logger.info(f"Résultats d'effets spécifiques LG trouvés.")
            if "lg_temporal_generalization_results" in subject_job_results:
                logger.info(f"Résultats de généralisation temporelle LG trouvés.")
                
        else:
            logger.warning(f"Structure de résultat LG inattendue pour {TARGET_SUBJECT_ID_FOR_JOB}. Reçu: {type(subject_job_results)}. Contenu (partiel): {str(subject_job_results)[:500]}")

    except FailedJobError as e_failed_job:
        job_id_err = str(getattr(submitted_job, 'job_id', 'ID_Erreur_Inconnu'))
        logger.error(f"Job Slurm {job_id_err} ({TARGET_SUBJECT_ID_FOR_JOB}) ÉCHOUÉ.", exc_info=False)
        logger.error(f"  Msg Submitit: {e_failed_job}")
        logger.error(f"  Consultez logs du worker: {os.path.abspath(submitit_job_log_folder)}")
    except Exception as e_result:
        job_id_err = str(getattr(submitted_job, 'job_id', 'ID_Erreur_Inconnu'))
        logger.error(f"Erreur récupération résultat job {job_id_err} ({TARGET_SUBJECT_ID_FOR_JOB}): {e_result}", exc_info=True)
        logger.warning(f"  État job: {getattr(submitted_job, 'state', 'N/A')}. Logs worker: {os.path.abspath(submitit_job_log_folder)}")

    logger.info(f"--- Logique de soumission pour {TARGET_SUBJECT_ID_FOR_JOB} ({TARGET_PROTOCOL_TYPE_FOR_JOB}) terminée. ---")

if __name__ == "__main__":
    logger.info(f"--- Démarrage du script de soumission principal LG ({os.path.basename(__file__)}) ---")
    try:
        main_submission_logic()
    except Exception as e_main:
        logger.critical(f"Erreur non gérée dans main_submission_logic: {e_main}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"--- Fin du script de soumission principal LG. Log: {MASTER_LOG_FILE_PATH} ---")
