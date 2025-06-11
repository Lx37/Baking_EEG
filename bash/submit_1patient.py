# --- Imports ---
import os
import sys
import logging
from datetime import datetime
import getpass
import time
import mne
# Tenter d'importer submitit plus tard pour permettre la configuration de sys.path
# import submitit # Déplacé

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
print(
    f"INFO: Tentative de PROJECT_ROOT_FOR_PYTHONPATH (parent de CURRENT_SCRIPT_DIR): {project_root_tentative}")

# Vérifier si 'examples' est présent dans ce chemin_racine_projet_tentatif
if os.path.isdir(os.path.join(project_root_tentative, "examples")):
    PROJECT_ROOT_FOR_PYTHONPATH = project_root_tentative
    print(
        f"INFO: PROJECT_ROOT_FOR_PYTHONPATH confirmé (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
else:
    print(
        f"AVERTISSEMENT: Le dossier 'examples' n'a pas été trouvé dans '{project_root_tentative}'.")
    print(
        f"         Contenu de '{project_root_tentative}': {os.listdir(project_root_tentative) if os.path.exists(project_root_tentative) else 'N/A'}")

    potential_paths = [
        CURRENT_SCRIPT_DIR,
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..")),
        os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, "..", "..")),
    ]

    found_path = False
    for path_candidate in potential_paths:
        if os.path.isdir(os.path.join(path_candidate, "examples")):
            PROJECT_ROOT_FOR_PYTHONPATH = path_candidate
            print(
                f"INFO: PROJECT_ROOT_FOR_PYTHONPATH trouvé via fallback (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
            found_path = True
            break

    if not found_path:
        hardcoded_project_root = "/home/tom.balay/Baking_EEG"
        if os.path.isdir(os.path.join(hardcoded_project_root, "examples")):
            PROJECT_ROOT_FOR_PYTHONPATH = hardcoded_project_root
            print(
                f"AVERTISSEMENT: PROJECT_ROOT_FOR_PYTHONPATH défini sur une valeur codée en dur (contient 'examples'): {PROJECT_ROOT_FOR_PYTHONPATH}")
        else:
            print(
                f"ERREUR CRITIQUE: Impossible de localiser le dossier racine du projet contenant 'examples'.")
            print(f"                 CURRENT_SCRIPT_DIR: {CURRENT_SCRIPT_DIR}")
            print(
                f"                 Chemins testés: {[project_root_tentative] + potential_paths}")
            sys.exit(1)

# Vérifier que le dossier 'examples' est un module Python valide
examples_dir = os.path.join(PROJECT_ROOT_FOR_PYTHONPATH, "examples")
if not os.path.isfile(os.path.join(examples_dir, "__init__.py")):
    print(f"AVERTISSEMENT: Le dossier 'examples' n'est pas un module Python valide (manque __init__.py).")
    print(
        f"                Vous pouvez créer un fichier vide __init__.py dans {examples_dir}.")
else:
    print(f"INFO: Le dossier 'examples' est un module Python valide (contient __init__.py).")

# Ajout de PROJECT_ROOT_FOR_PYTHONPATH à sys.path AVANT tout autre import de module du projet
if PROJECT_ROOT_FOR_PYTHONPATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_PYTHONPATH)
    print(f"INFO: '{PROJECT_ROOT_FOR_PYTHONPATH}' a été ajouté à sys.path.")
else:
    print(f"INFO: '{PROJECT_ROOT_FOR_PYTHONPATH}' est déjà dans sys.path.")

print(
    f"INFO: Contenu de PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}'): {os.listdir(PROJECT_ROOT_FOR_PYTHONPATH)}")
print(f"INFO: sys.path commence par: {sys.path[:3]}...")

# Testez l'importation du module 'examples' pour vérifier que tout est correct
try:
    import examples
    print("INFO: Le module 'examples' a été importé avec succès.")
except ModuleNotFoundError as e:
    print(f"ERREUR: Impossible d'importer le module 'examples': {e}")
    print(
        f"         Assurez-vous que le dossier 'examples' est un module Python valide et que le chemin '{PROJECT_ROOT_FOR_PYTHONPATH}' est correct.")
    sys.exit(1)

# --- Importation de Submitit (maintenant que sys.path est configuré) ---
try:
    import submitit
    from submitit.core.utils import FailedJobError
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer le module 'submitit' ou 'FailedJobError'. Assurez-vous qu'il est installé dans votre environnement Python.")
    print(f"               Erreur originale: {e}")
    sys.exit(1)

# --- Configuration du Logger Principal ---
LOG_DIR_SUBMITIT_MASTER = os.path.join(
    CURRENT_SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(LOG_DIR_SUBMITIT_MASTER, exist_ok=True)

# Définition des cibles pour ce job spécifique
TARGET_SUBJECT_ID_FOR_JOB = "TpSM49"
TARGET_PROTOCOL_TYPE_FOR_JOB = "PP_AP"

MASTER_LOG_FILE_NAME = datetime.now().strftime(
    f'master_submitit_SINGLE_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_%Y-%m-%d_%H-%M-%S.log'
)
MASTER_LOG_FILE_PATH = os.path.join(
    LOG_DIR_SUBMITIT_MASTER, MASTER_LOG_FILE_NAME)

# Nettoyage des handlers de logging existants pour éviter la duplication
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

logger.info(
    f"--- Script de soumission Submitit pour sujet unique (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) démarré ---")
logger.info(f"Sujet Cible: {TARGET_SUBJECT_ID_FOR_JOB}")
logger.info(
    f"Log principal de ce script de soumission: {MASTER_LOG_FILE_PATH}")
logger.info(
    f"CURRENT_SCRIPT_DIR (où ce script est exécuté): {CURRENT_SCRIPT_DIR}")
logger.info(
    f"PROJECT_ROOT_FOR_PYTHONPATH (ajouté à sys.path): {PROJECT_ROOT_FOR_PYTHONPATH}")

# --- Importations des modules du projet (maintenant que sys.path est correct) ---


def import_project_modules():
    """Import project modules with better error handling."""
    try:
        # Ensure the examples module can be imported
        import examples
        logger.info("Module 'examples' importé avec succès.")

        from examples.run_decoding_one_pp import execute_single_subject_decoding
        from config.config import ALL_SUBJECT_GROUPS
        from utils.utils import configure_project_paths
        from config.decoding_config import (
            CLASSIFIER_MODEL_TYPE,
            USE_GRID_SEARCH_OPTIMIZATION,
            USE_CSP_FOR_TEMPORAL_PIPELINES,
            USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
            PARAM_GRID_CONFIG_EXTENDED,
            CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
            FIXED_CLASSIFIER_PARAMS_CONFIG,
            N_PERMUTATIONS_INTRA_SUBJECT,
            COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
            INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
            COMPUTE_INTRA_SUBJECT_STATISTICS,
            SAVE_ANALYSIS_RESULTS,
            GENERATE_PLOTS,
            CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
            CHANCE_LEVEL_AUC_SCORE
        )
        logger.info(
            "Importations depuis 'examples.run_decoding_one_pp', 'config' et 'utils' réussies.")

        return {
            'execute_single_subject_decoding': execute_single_subject_decoding,
            'ALL_SUBJECT_GROUPS': ALL_SUBJECT_GROUPS,
            'configure_project_paths': configure_project_paths,
            'CLASSIFIER_MODEL_TYPE': CLASSIFIER_MODEL_TYPE,
            'USE_GRID_SEARCH_OPTIMIZATION': USE_GRID_SEARCH_OPTIMIZATION,
            'USE_CSP_FOR_TEMPORAL_PIPELINES': USE_CSP_FOR_TEMPORAL_PIPELINES,
            'USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES': USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
            'PARAM_GRID_CONFIG_EXTENDED': PARAM_GRID_CONFIG_EXTENDED,
            'CV_FOLDS_FOR_GRIDSEARCH_INTERNAL': CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
            'FIXED_CLASSIFIER_PARAMS_CONFIG': FIXED_CLASSIFIER_PARAMS_CONFIG,
            'N_PERMUTATIONS_INTRA_SUBJECT': N_PERMUTATIONS_INTRA_SUBJECT,
            'COMPUTE_TEMPORAL_GENERALIZATION_MATRICES': COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
            'INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG': INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
            'COMPUTE_INTRA_SUBJECT_STATISTICS': COMPUTE_INTRA_SUBJECT_STATISTICS,
            'SAVE_ANALYSIS_RESULTS': SAVE_ANALYSIS_RESULTS,
            'GENERATE_PLOTS': GENERATE_PLOTS,
            'CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT': CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
            'CHANCE_LEVEL_AUC_SCORE': CHANCE_LEVEL_AUC_SCORE
        }

    except ModuleNotFoundError as e_mod:
        logger.critical(
            f"ERREUR CRITIQUE: Un module requis n'a pas été trouvé. Cela peut indiquer un problème avec PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}') "
            f"ou que le module/fichier spécifié est manquant dans la structure du projet."
        )
        logger.critical(f"  Module problématique probable: {e_mod.name}")
        logger.critical(f"  Erreur originale: {e_mod}")
        logger.critical(f"  Chemins Python actuels (sys.path): {sys.path}")
        sys.exit(1)
    except ImportError as e_imp:
        logger.critical(
            f"ERREUR CRITIQUE: Problème d'importation d'un nom spécifique (variable, fonction, classe) depuis un module."
        )
        logger.critical(f"  Nom problématique probable: {e_imp.name}")
        logger.critical(f"  Erreur originale: {e_imp}")
        logger.critical(
            f"  Vérifiez que toutes les constantes/fonctions listées dans les imports sont bien définies et exportables "
            f"(au niveau global du module) dans leurs fichiers respectifs (ex: 'examples.run_decoding_one_pp.py', 'config.decoding_config.py')."
        )
        sys.exit(1)


# Import modules and store references
project_modules = import_project_modules()

# Extract modules and constants for easier access
execute_single_subject_decoding = project_modules['execute_single_subject_decoding']
ALL_SUBJECT_GROUPS = project_modules['ALL_SUBJECT_GROUPS']
configure_project_paths = project_modules['configure_project_paths']
CLASSIFIER_MODEL_TYPE = project_modules['CLASSIFIER_MODEL_TYPE']
USE_GRID_SEARCH_OPTIMIZATION = project_modules['USE_GRID_SEARCH_OPTIMIZATION']
USE_CSP_FOR_TEMPORAL_PIPELINES = project_modules['USE_CSP_FOR_TEMPORAL_PIPELINES']
USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES = project_modules['USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES']
PARAM_GRID_CONFIG_EXTENDED = project_modules['PARAM_GRID_CONFIG_EXTENDED']
CV_FOLDS_FOR_GRIDSEARCH_INTERNAL = project_modules['CV_FOLDS_FOR_GRIDSEARCH_INTERNAL']
FIXED_CLASSIFIER_PARAMS_CONFIG = project_modules['FIXED_CLASSIFIER_PARAMS_CONFIG']
N_PERMUTATIONS_INTRA_SUBJECT = project_modules['N_PERMUTATIONS_INTRA_SUBJECT']
COMPUTE_TEMPORAL_GENERALIZATION_MATRICES = project_modules[
    'COMPUTE_TEMPORAL_GENERALIZATION_MATRICES']
INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG = project_modules['INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG']
COMPUTE_INTRA_SUBJECT_STATISTICS = project_modules['COMPUTE_INTRA_SUBJECT_STATISTICS']
SAVE_ANALYSIS_RESULTS = project_modules['SAVE_ANALYSIS_RESULTS']
GENERATE_PLOTS = project_modules['GENERATE_PLOTS']
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT = project_modules[
    'CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT']
CHANCE_LEVEL_AUC_SCORE = project_modules['CHANCE_LEVEL_AUC_SCORE']


def execute_single_subject_decoding_wrapper(**kwargs):
    """
    Wrapper function that ensures proper imports in the Slurm environment.
    This function will be executed in the worker node.
    """
    import os
    import sys

    # Get the project root from the kwargs or determine it
    project_root = kwargs.pop(
        'project_root_path', '/home/tom.balay/Baking_EEG')

    # Ensure project root is in sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Change to project directory
    os.chdir(project_root)

    try:
        # Import the actual function in the worker environment
        from examples.run_decoding_one_pp import execute_single_subject_decoding

        # Execute the function with the provided arguments
        return execute_single_subject_decoding(**kwargs)

    except ImportError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            f"Failed to import execute_single_subject_decoding in worker: {e}")
        logger.error(f"sys.path: {sys.path}")
        logger.error(f"Current directory: {os.getcwd()}")
        logger.error(f"Project root: {project_root}")

        # List directory contents for debugging
        if os.path.exists(project_root):
            logger.error(
                f"Contents of {project_root}: {os.listdir(project_root)}")
            examples_dir = os.path.join(project_root, 'examples')
            if os.path.exists(examples_dir):
                logger.error(
                    f"Contents of {examples_dir}: {os.listdir(examples_dir)}")

        raise


# --- Configuration pour l'exécution sur le cluster Slurm ---
PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
logger.info(
    f"Chemin vers 'activate' de l'environnement virtuel sur le cluster: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}")

# La racine du projet sur le cluster sera la même que celle détectée localement
PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT_FOR_PYTHONPATH
logger.info(
    f"Racine du projet à ajouter au PYTHONPATH du job Slurm: {PROJECT_ROOT_ON_CLUSTER_FOR_JOB}")

# Commandes de setup pour l'environnement du job Slurm
SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""#!/bin/bash
#SBATCH --output=slurm_job_output_%j.txt # Capture stdout/stderr du setup
#SBATCH --error=slurm_job_error_%j.txt

echo "--- Configuration de l'environnement pour le job Slurm (PID: $$) ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial du job: $(pwd)"

# Pour plus de robustesse dans le script bash
set -e # Quitte immédiatement si une commande échoue
set -u # Traite les variables non définies comme une erreur
set -o pipefail # Le code de sortie d'un pipeline est celui de la dernière commande à avoir échoué

# Désactiver un venv potentiellement actif (si la session Slurm en hérite un)
if command -v deactivate &> /dev/null ; then
    echo "INFO: Tentative de désactivation d'un environnement virtuel existant..."
    deactivate || echo "AVERTISSEMENT: 'deactivate' a échoué ou aucun venv n'était actif."
fi

# Purger les modules système pour un environnement propre
echo "INFO: Purge des modules système (module purge)..."
module purge
echo "INFO: Modules système purgés."

# Charger les modules spécifiques au cluster si nécessaire AVANT d'activer le venv
# Exemple: module load python/3.11.5 # Si votre cluster le requiert
# Si votre venv est construit avec un Python système, ce n'est peut-être pas nécessaire.
# Si le venv est construit avec un Python chargé par module, chargez CE module.

echo "INFO: Activation de l'environnement virtuel Python: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
source "{PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
if [ $? -ne 0 ]; then
    echo "ERREUR CRITIQUE: Échec de l'activation de l'environnement virtuel Python. Arrêt du job."
    exit 1
fi
echo "INFO: Environnement virtuel activé. Chemin Python: $(which python)"
echo "INFO: Version Python active: $(python --version)"

echo "INFO: Configuration de PYTHONPATH pour le job..."
# S'assurer que la racine du projet est au début de PYTHONPATH pour prioriser ses modules
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}:$PYTHONPATH"
echo "INFO: PYTHONPATH actuel du job: $PYTHONPATH"

echo "INFO: Changement de répertoire vers la racine du projet..."
cd "{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}"
echo "INFO: Répertoire de travail actuel: $(pwd)"

echo "INFO: Vérification de la structure du projet..."
if [ -d "examples" ]; then
    echo "INFO: Dossier 'examples' trouvé"
    if [ -f "examples/__init__.py" ]; then
        echo "INFO: Fichier 'examples/__init__.py' trouvé"
    else
        echo "AVERTISSEMENT: Fichier 'examples/__init__.py' manquant"
        touch "examples/__init__.py"
        echo "INFO: Fichier 'examples/__init__.py' créé"
    fi
else
    echo "ERREUR: Dossier 'examples' non trouvé dans $(pwd)"
    ls -la
    exit 1
fi

echo "INFO: Test d'importation du module examples..."
python -c "import sys; sys.path.insert(0, '{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}'); import examples; print('Module examples importé avec succès')" || {{
    echo "ERREUR: Impossible d'importer le module examples"
    python -c "import sys; print('sys.path:', sys.path)"
    exit 1
}}

echo "--- Environnement CPU configuré avec succès ---"
echo "INFO: Vérification des dépendances clés (les chemins doivent pointer vers votre venv) :"

echo "---------------------------------------------------------------------"
"""


def main_submission_logic():
    """Contient la logique principale de configuration et de soumission du job."""
    logger.info(
        "--- Début de la logique de soumission principale (main_submission_logic) ---")

    # Détermination de l'utilisateur pour les chemins de données
    try:
        user_for_paths = getpass.getuser()
        logger.info(f"Utilisateur détecté (getpass): {user_for_paths}")
    except Exception as e_getpass:
        logger.warning(
            f"getpass.getuser() a échoué ({e_getpass}). Tentative avec variables d'environnement LOGNAME/USER.")
        user_for_paths = os.environ.get('LOGNAME') or os.environ.get('USER')
        if not user_for_paths:
            user_for_paths = "unknown_user_fallback"
            logger.error(
                f"Impossible de déterminer l'utilisateur. Utilisation de '{user_for_paths}'.")
        else:
            logger.info(
                f"Utilisateur déterminé par variable d'environnement: {user_for_paths}")

    # Configuration des chemins d'entrée/sortie
    try:
        base_input_path, base_output_path = configure_project_paths(
            user_for_paths)
    except Exception as e_paths:
        logger.critical(
            f"ERREUR CRITIQUE: Échec de la configuration des chemins de projet via configure_project_paths pour l'utilisateur '{user_for_paths}': {e_paths}", exc_info=True)
        sys.exit(1)

    if not os.path.isdir(base_input_path):
        logger.critical(
            f"ERREUR CRITIQUE: Le chemin des données d'entrée '{base_input_path}' (pour l'utilisateur '{user_for_paths}') n'existe pas ou n'est pas un répertoire. Arrêt."
        )
        sys.exit(1)
    logger.info(
        f"Chemin de base des données d'entrée utilisé: {base_input_path}")
    logger.info(
        f"Chemin de base des résultats de sortie utilisé: {base_output_path}")

    # Détermination du groupe du sujet
    target_subject_group = None
    if ALL_SUBJECT_GROUPS:  # S'assurer que ALL_SUBJECT_GROUPS n'est pas vide ou None
        for group_name, subject_list_in_group in ALL_SUBJECT_GROUPS.items():
            if TARGET_SUBJECT_ID_FOR_JOB in subject_list_in_group:
                target_subject_group = group_name
                break

    if target_subject_group is None:
        logger.critical(
            f"ERREUR CRITIQUE: Sujet '{TARGET_SUBJECT_ID_FOR_JOB}' non trouvé dans ALL_SUBJECT_GROUPS défini dans config.config. Arrêt."
        )
        sys.exit(1)
    logger.info(
        f"Groupe du sujet '{TARGET_SUBJECT_ID_FOR_JOB}': {target_subject_group}")

    # Paramètres des ressources Slurm
    N_CPUS_FOR_JOB = 40
    MEMORY_FOR_JOB = "60G"  # ex: "60G" ou "60000M"
    TIMEOUT_MINUTES = 12 * 60  # 12 heures
    SLURM_PARTITION = "CPU"  # Partition cible
    # Compte Slurm à utiliser (vérifier si nécessaire pour votre cluster)
    SLURM_ACCOUNT = "tom.balay"

    logger.info(
        f"Ressources Slurm demandées: CPUs={N_CPUS_FOR_JOB}, Mémoire={MEMORY_FOR_JOB}, Timeout={TIMEOUT_MINUTES}min, "
        f"Partition={SLURM_PARTITION}, Account={SLURM_ACCOUNT if SLURM_ACCOUNT else 'Non spécifié'}"
    )

    slurm_params = {
        "timeout_min": TIMEOUT_MINUTES,
        "local_setup": SETUP_COMMANDS_FOR_SLURM_JOB_CPU.splitlines(),
        "slurm_partition": SLURM_PARTITION,
        "slurm_mem": MEMORY_FOR_JOB,
        "slurm_cpus_per_task": N_CPUS_FOR_JOB,
    }
    if SLURM_ACCOUNT:  # Ajouter le compte seulement s'il est spécifié
        slurm_params["slurm_additional_parameters"] = {
            "account": SLURM_ACCOUNT}

    logger.info(f"Paramètres Slurm finaux pour le job: {slurm_params}")

    # Configuration du dossier de log pour les jobs Submitit individuels
    current_timestamp_for_log = datetime.now().strftime('%Y%m%d_%H%M%S')
    submitit_job_log_folder = os.path.join(
        CURRENT_SCRIPT_DIR,
        "logs_submitit_jobs",
        f"job_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_{current_timestamp_for_log}"
    )
    os.makedirs(submitit_job_log_folder, exist_ok=True)
    logger.info(
        f"Les logs spécifiques à ce job Slurm (gérés par submitit) seront dans: {os.path.abspath(submitit_job_log_folder)}"
    )

    # Initialisation de l'exécuteur Submitit
    executor = submitit.AutoExecutor(folder=submitit_job_log_folder)
    executor.update_parameters(**slurm_params)

    # Configuration de chargement des données (spécifique au protocole PP_AP ici)
    loading_config_for_job = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    logger.info(
        f"Configuration de chargement des données (CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT) pour le job PP_AP utilisée."
    )

    # Préparation des arguments pour la fonction cible
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
        "protocol_type": TARGET_PROTOCOL_TYPE_FOR_JOB,

        # Add the project root path for the wrapper function
        "project_root_path": PROJECT_ROOT_ON_CLUSTER_FOR_JOB,
    }
    logger.info(
        f"Arguments (clés) passés à execute_single_subject_decoding_wrapper: {list(kwargs_for_function_call.keys())}")

    # Soumission du job
    submitted_job = None
    try:
        logger.info(
            f"Soumission du job pour le sujet: {TARGET_SUBJECT_ID_FOR_JOB}, Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}..."
        )
        submitted_job = executor.submit(
            execute_single_subject_decoding_wrapper, **kwargs_for_function_call
        )
        logger.info(
            f"Job soumis avec succès. Objet Job Submitit: {submitted_job}")
        if hasattr(submitted_job, 'job_id') and submitted_job.job_id:
            slurm_job_id_str = str(submitted_job.job_id)
            logger.info(f"ID (Slurm/Local) du job soumis: {slurm_job_id_str}")
            # Sauvegarder l'ID du job pour un suivi facile
            job_id_file = os.path.join(
                submitit_job_log_folder, f"slurm_job_id_{slurm_job_id_str}.info")
            with open(job_id_file, "w") as f_id:
                f_id.write(f"Slurm Job ID: {slurm_job_id_str}\n")
                f_id.write(f"Sujet: {TARGET_SUBJECT_ID_FOR_JOB}\n")
                f_id.write(f"Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}\n")
                f_id.write(f"Soumis le: {datetime.now().isoformat()}\n")
                f_id.write(
                    f"Dossier de logs Submitit: {os.path.abspath(submitit_job_log_folder)}\n")
            logger.info(
                f"Informations du job sauvegardées dans : {job_id_file}")

    except Exception as e_submit:
        logger.critical(
            f"Erreur critique lors de la soumission du job pour {TARGET_SUBJECT_ID_FOR_JOB}: {e_submit}", exc_info=True
        )
        return

    if not submitted_job:
        logger.error(
            "La soumission du job a échoué, aucun objet job n'a été retourné. Vérifiez les logs précédents.")
        return

    # Attente et traitement du résultat du job
    logger.info(
        f"--- Attente du résultat du job (ID Submitit: {submitted_job.job_id if hasattr(submitted_job, 'job_id') else 'N/A'}) ---")
    try:
        subject_job_results = submitted_job.result()
        final_job_id = str(
            getattr(submitted_job, 'job_id', 'ID_Final_Inconnu'))
        logger.info(
            f"Job {final_job_id} (Sujet: {TARGET_SUBJECT_ID_FOR_JOB}) terminé. État final Submitit: {submitted_job.state}"
        )

        import numpy as np
        import pandas as pd

        if isinstance(subject_job_results, dict) and \
           subject_job_results.get("subject_id") == TARGET_SUBJECT_ID_FOR_JOB and \
           subject_job_results.get("protocol_type_processed") == TARGET_PROTOCOL_TYPE_FOR_JOB:

            logger.info(
                f"Résultats reçus pour {TARGET_SUBJECT_ID_FOR_JOB} (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}):")

            mean_global_auc = subject_job_results.get(
                'pp_ap_main_mean_auc_global', np.nan)
            auc_str = f"{mean_global_auc:.3f}" if pd.notna(
                mean_global_auc) else "N/A"
            logger.info(f"  Moyenne Global AUC (pp_ap_main): {auc_str}")

            global_metrics = subject_job_results.get(
                "pp_ap_main_global_metrics")
            if global_metrics:
                logger.info(
                    f"  Métriques globales (pp_ap_main): {global_metrics}")
            else:
                logger.info(
                    "  Aucune métrique globale (pp_ap_main) trouvée dans les résultats.")

            ap_centric_results = subject_job_results.get(
                "pp_ap_ap_centric_avg_results", [])
            if ap_centric_results:
                num_valid_ap_centric_avg = sum(
                    1 for r in ap_centric_results
                    if r is not None and r.get('average_scores_1d') is not None and r.get('num_constituent_curves', 0) >= 2
                )
                logger.info(
                    f"  Nombre de moyennes AP-centriques valides (>=2 courbes) générées: {num_valid_ap_centric_avg} sur {len(ap_centric_results)} APs traités."
                )
                expected_ap_families = 6
                if num_valid_ap_centric_avg < expected_ap_families:
                    logger.warning(
                        f"  Moins de {expected_ap_families} moyennes AP-centriques valides. Certaines visualisations pourraient être incomplètes."
                    )
            else:
                logger.warning(
                    "  Aucun résultat pour 'pp_ap_ap_centric_avg_results' trouvé dans les résultats du job.")
        else:
            logger.warning(
                f"Résultat inattendu ou structure de dictionnaire incorrecte pour {TARGET_SUBJECT_ID_FOR_JOB}. "
                f"Type reçu: {type(subject_job_results)}. "
                f"Contenu (partiel, max 500 caractères): {str(subject_job_results)[:500]}"
            )
            if isinstance(subject_job_results, dict):
                logger.warning(
                    f"  Clés reçues: {list(subject_job_results.keys())}")
                logger.warning(
                    f"  ID sujet attendu: {TARGET_SUBJECT_ID_FOR_JOB}, reçu: {subject_job_results.get('subject_id')}")
                logger.warning(
                    f"  Protocole attendu: {TARGET_PROTOCOL_TYPE_FOR_JOB}, reçu: {subject_job_results.get('protocol_type_processed')}")

    except FailedJobError as e_failed_job:
        final_job_id_err = str(
            getattr(submitted_job, 'job_id', 'ID_Erreur_Inconnu'))
        logger.error(
            f"Le job Slurm {final_job_id_err} (Sujet: {TARGET_SUBJECT_ID_FOR_JOB}) a ÉCHOUÉ.", exc_info=False)
        logger.error(f"  Message d'erreur de Submitit: {e_failed_job}")
        logger.error(
            f"  Consultez les logs du job dans: {os.path.abspath(submitit_job_log_folder)} pour le traceback complet du worker.")
    except Exception as e_result:
        final_job_id_err = str(
            getattr(submitted_job, 'job_id', 'ID_Erreur_Inconnu'))
        logger.error(
            f"Erreur lors de la récupération du résultat du job {final_job_id_err} (Sujet: {TARGET_SUBJECT_ID_FOR_JOB}): {e_result}",
            exc_info=True
        )
        logger.warning(
            f"État du job Submitit au moment de l'erreur: {getattr(submitted_job, 'state', 'N/A')}")
        logger.warning(
            f"Consultez les logs du job dans: {os.path.abspath(submitit_job_log_folder)} pour plus de détails sur l'exécution du worker.")

    logger.info(
        f"--- Logique de soumission pour le sujet {TARGET_SUBJECT_ID_FOR_JOB} (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) terminée. ---")


if __name__ == "__main__":
    logger.info(
        f"--- Démarrage du script de soumission principal ({os.path.basename(__file__)}) ---")
    try:
        main_submission_logic()
    except Exception as e_main:
        logger.critical(
            f"Une erreur non gérée s'est produite dans la fonction main_submission_logic: {e_main}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(
            f"--- Fin du script de soumission principal. Log principal dans: {MASTER_LOG_FILE_PATH} ---")
