import submitit
import os
import sys
import logging
from datetime import datetime
import getpass
import numpy as np
import pandas as pd
import time

# --- Configuration du chemin du projet pour les imports ---
try:
    # Chemin du script de soumission lui-même
    SUBMITIT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SUBMITIT_SCRIPT_DIR = os.getcwd()
    print(
        f"AVERTISSEMENT: __file__ non défini, SUBMITIT_SCRIPT_DIR mis à {SUBMITIT_SCRIPT_DIR}", file=sys.stderr)

# Configuration de PROJECT_ROOT_FOR_PYTHONPATH
# Si ce script de soumission est dans /home/tom.balay/ et que run_decoding.py est dans /home/tom.balay/examples/
# alors PROJECT_ROOT_FOR_PYTHONPATH doit être /home/tom.balay
# Supposant que ce script est à la racine du projet contenant 'examples'
PROJECT_ROOT_FOR_PYTHONPATH = SUBMITIT_SCRIPT_DIR
if not os.path.isdir(os.path.join(PROJECT_ROOT_FOR_PYTHONPATH, "Baking_EEG")):
    # Fallback si 'Baking_EEG' n'est pas un sous-dossier direct (par exemple, ce script est dans 'scripts/')
    PROJECT_ROOT_FOR_PYTHONPATH = os.path.abspath(
        os.path.join(SUBMITIT_SCRIPT_DIR, "..", "Baking_EEG"))
    if not os.path.isdir(os.path.join(PROJECT_ROOT_FOR_PYTHONPATH, "examples")):
        print(f"ERREUR CRITIQUE: Impossible de localiser le dossier 'examples' par rapport à {SUBMITIT_SCRIPT_DIR}. "
              f"PROJECT_ROOT_FOR_PYTHONPATH actuel : {PROJECT_ROOT_FOR_PYTHONPATH}. Veuillez vérifier ce chemin.", file=sys.stderr)
        sys.exit(1)  # Arrêt si le chemin crucial n'est pas bon.

if PROJECT_ROOT_FOR_PYTHONPATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_PYTHONPATH)
    print(
        f"Ajout de {PROJECT_ROOT_FOR_PYTHONPATH} à sys.path pour les imports.", file=sys.stderr)

# --- Configuration du Logger Principal pour ce script de soumission ---
# Relatif à l'endroit où ce script est exécuté
LOG_DIR_SUBMITIT_MASTER = './logs_submitit_master/'
os.makedirs(LOG_DIR_SUBMITIT_MASTER, exist_ok=True)

TARGET_SUBJECT_ID_FOR_JOB = "TpSM49"  # Sujet cible pour ce script
TARGET_PROTOCOL_TYPE_FOR_JOB = "PP_AP"  # Fixé au protocole PP_AP

MASTER_LOG_FILE = os.path.join(
    LOG_DIR_SUBMITIT_MASTER,
    datetime.now().strftime(
        f'master_submitit_SINGLE_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_%Y-%m-%d_%H-%M-%S.log')
)

# Nettoyage des handlers existants pour éviter les logs dupliqués
root_logger_submitit_master = logging.getLogger()
if root_logger_submitit_master.hasHandlers():
    for handler in root_logger_submitit_master.handlers[:]:
        root_logger_submitit_master.removeHandler(handler)
        handler.close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        # 'w' pour écraser les anciens logs de ce master
        logging.FileHandler(MASTER_LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)  # Logger spécifique pour ce script

logger.info(
    f"--- Script de soumission Submitit pour sujet unique (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) démarré ---")
logger.info(f"Sujet Cible: {TARGET_SUBJECT_ID_FOR_JOB}")
logger.info(f"Log principal de ce script: {MASTER_LOG_FILE}")
logger.info(
    f"SUBMITIT_SCRIPT_DIR (où ce script est exécuté): {SUBMITIT_SCRIPT_DIR}")
logger.info(
    f"PROJECT_ROOT_FOR_PYTHONPATH (ajouté à sys.path): {PROJECT_ROOT_FOR_PYTHONPATH}")


# --- Importation des fonctions et constantes depuis run_decoding ---
try:
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
    logger.info("Importations depuis 'examples.run_decoding_one_pp' réussies.")
except ModuleNotFoundError as e_mod:
    logger.critical(
        f"ERREUR CRITIQUE: Module 'examples.run_decoding_one_pp' non trouvé. Cela signifie probablement que PROJECT_ROOT_FOR_PYTHONPATH ('{PROJECT_ROOT_FOR_PYTHONPATH}') n'est pas correct ou que le fichier/dossier 'examples' n'y est pas.")
    logger.critical(f"  Erreur originale: {e_mod}")
    logger.critical(f"  Chemins Python actuels (sys.path): {sys.path}")
    sys.exit(1)
except ImportError as e_imp:
    logger.critical(
        f"ERREUR CRITIQUE: Problème d'importation d'un nom spécifique depuis 'examples.run_decoding_one_pp': {e_imp}")
    logger.critical(
        f"  Vérifiez que toutes les constantes listées ci-dessus sont bien définies et exportables (au niveau global du module) dans 'examples.run_decoding_one_pp.py'.")
    sys.exit(1)

# --- Configuration pour l'exécution sur le cluster Slurm ---
PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
logger.info(
    f"Chemin vers 'activate' de l'environnement virtuel sur le cluster: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}")

PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT_FOR_PYTHONPATH
logger.info(
    f"Racine du projet à ajouter au PYTHONPATH du job Slurm: {PROJECT_ROOT_ON_CLUSTER_FOR_JOB}")

SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm (PID: $$) ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial du job: $(pwd)"

# Désactiver un venv potentiellement actif
if command -v deactivate &> /dev/null ; then
    echo "Tentative de désactivation d'un environnement virtuel existant..."
    deactivate || echo "Avertissement: 'deactivate' a échoué ou aucun venv n'était actif."
fi

module purge
echo "Modules système purgés."
# Ajoutez ici le chargement de modules Python du cluster si nécessaire AVANT d'activer le venv
# Exemple: module load python/3.11.5 # Si votre cluster le requiert pour cette version de Python

echo "Activation de l'environnement virtuel Python: {PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
source "{PATH_TO_VENV_ACTIVATE_ON_CLUSTER}"
if [ $? -ne 0 ]; then
    echo "ERREUR CRITIQUE: Échec de l'activation de l'environnement virtuel Python. Arrêt du job."
    exit 1
fi
echo "Environnement virtuel activé. Chemin Python: $(which python)"

echo "Configuration de PYTHONPATH pour le job..."
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}:${{PYTHONPATH}}"
echo "PYTHONPATH actuel du job: $PYTHONPATH"
echo "--- Environnement CPU configuré ---"
echo "Chemin Python complet utilisé par le job: $(readlink -f $(which python))"
echo "Version Python utilisée par le job: $(python --version)"
echo "Vérification des dépendances clés (les chemins doivent pointer vers votre venv) :"
python -c "import sys; print(f'  Chemins sys.path dans Python du job: {{sys.path}}')"
python -c "import mne; print(f'  MNE version: {{mne.__version__}} (chemin: {{mne.__file__}})')"
python -c "import sklearn; print(f'  scikit-learn version: {{sklearn.__version__}} (chemin: {{sklearn.__file__}})')"
python -c "import numpy; print(f'  NumPy version: {{numpy.__version__}} (chemin: {{numpy.__file__}})')"
python -c "import pandas; print(f'  Pandas version: {{pandas.__version__}} (chemin: {{pandas.__file__}})')"
echo "-------------------------------------------"
"""


def main_submission_logic():
    """Contient la logique principale de configuration et de soumission du job."""
    try:
        user_for_paths = getpass.getuser()
        logger.info(f"Utilisateur détecté (getpass): {user_for_paths}")
    except Exception:
        user_for_paths = os.environ.get('LOGNAME') or os.environ.get(
            'USER', "unknown_user_submitit_fallback")
        logger.warning(
            f"getpass.getuser() a échoué. Tentative d'utilisation de LOGNAME/USER. Utilisateur final: {user_for_paths}")

    base_input_path, base_output_path = configure_project_paths(user_for_paths)
    if not os.path.isdir(base_input_path):
        logger.critical(
            f"ERREUR CRITIQUE: Le chemin des données d'entrée '{base_input_path}' pour l'utilisateur '{user_for_paths}' n'existe pas ou n'est pas un répertoire. Arrêt.")
        sys.exit(1)
    logger.info(
        f"Chemin de base des données d'entrée utilisé: {base_input_path}")
    logger.info(
        f"Chemin de base des résultats de sortie utilisé: {base_output_path}")

    target_subject_group = None
    for group_name, subject_list_in_group in ALL_SUBJECT_GROUPS.items():
        if TARGET_SUBJECT_ID_FOR_JOB in subject_list_in_group:
            target_subject_group = group_name
            break
    if target_subject_group is None:
        logger.critical(
            f"ERREUR CRITIQUE: Sujet '{TARGET_SUBJECT_ID_FOR_JOB}' non trouvé dans ALL_SUBJECT_GROUPS. Arrêt.")
        sys.exit(1)
    logger.info(
        f"Groupe du sujet '{TARGET_SUBJECT_ID_FOR_JOB}': {target_subject_group}")

    N_CPUS_FOR_JOB = 40
    MEMORY_FOR_JOB = "60G"
    TIMEOUT_MINUTES = 12 * 60
    SLURM_PARTITION = "CPU"  # Mis à jour
    SLURM_ACCOUNT = "tom.balay"  # S'assurer que c'est correct

    logger.info(
        f"Ressources Slurm demandées: CPUs={N_CPUS_FOR_JOB}, Mémoire={MEMORY_FOR_JOB}, Timeout={TIMEOUT_MINUTES}min, Partition={SLURM_PARTITION}, Account={SLURM_ACCOUNT}")

    slurm_parameters_for_job = {
        "timeout_min": TIMEOUT_MINUTES,
        "slurm_additional_parameters": {"account": SLURM_ACCOUNT},
        "setup": SETUP_COMMANDS_FOR_SLURM_JOB_CPU,
        "slurm_partition": SLURM_PARTITION,
        "slurm_mem": MEMORY_FOR_JOB,
        "slurm_cpus_per_task": N_CPUS_FOR_JOB,
    }
    logger.info(
        f"Paramètres Slurm finaux pour le job: {slurm_parameters_for_job}")

    current_timestamp_for_log = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Le dossier de log sera relatif à l'endroit où CE SCRIPT EST EXÉCUTÉ
    submitit_job_log_folder = os.path.join(
        SUBMITIT_SCRIPT_DIR,  # Utiliser le répertoire du script de soumission comme base
        "logs_submitit_jobs",  # Sous-dossier pour les logs des jobs individuels
        f"job_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{TARGET_SUBJECT_ID_FOR_JOB}_{current_timestamp_for_log}"
    )
    # S'assurer que le dossier existe
    os.makedirs(submitit_job_log_folder, exist_ok=True)
    logger.info(
        f"Les logs spécifiques à ce job Slurm (submitit) seront dans: {submitit_job_log_folder} (ABSOLU: {os.path.abspath(submitit_job_log_folder)})")

    executor = submitit.AutoExecutor(folder=submitit_job_log_folder)
    executor.update_parameters(**slurm_parameters_for_job)

    # La configuration de chargement est spécifique au protocole PP_AP ici
    loading_config_for_job = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    logger.info(
        f"Configuration de chargement des données (CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT) pour le job PP_AP utilisée.")

    kwargs_for_function_call = {
        "subject_identifier": TARGET_SUBJECT_ID_FOR_JOB,
        "group_affiliation": target_subject_group,
        "decoding_protocol_identifier": f'Analysis_{TARGET_PROTOCOL_TYPE_FOR_JOB}_Individual',
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "enable_verbose_logging": True,
        "generate_plots_flag": GENERATE_PLOTS,
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path,
        # Utiliser tous les CPUs alloués au job
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
        # Assigner la config de chargement
        "loading_conditions_config": loading_config_for_job,
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        # Transmettre explicitement le type de protocole
        "protocol_type": TARGET_PROTOCOL_TYPE_FOR_JOB,
    }
    logger.info(
        f"Arguments qui seront passés à execute_single_subject_decoding: {list(kwargs_for_function_call.keys())}")
    # Pour débogage, afficher les valeurs si nécessaire:
    # for key, value in kwargs_for_function_call.items():
    # logger.debug(f"  {key}: {str(value)[:200]}") # Affiche les 200 premiers caractères de la valeur

    submitted_job = None
    try:
        logger.info(
            f"Soumission du job pour le sujet: {TARGET_SUBJECT_ID_FOR_JOB}, Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}")
        # Le callable est execute_single_subject_decoding
        submitted_job = executor.submit(
            execute_single_subject_decoding, **kwargs_for_function_call)
        logger.info(f"Job soumis. Objet Job Submitit: {submitted_job}")
        if hasattr(submitted_job, 'job_id'):  # Pour les exécuteurs Slurm/Local
            slurm_job_id_str = str(submitted_job.job_id)
            logger.info(f"ID (Slurm/Local) du job soumis: {slurm_job_id_str}")
            # Créer un fichier avec l'ID du job Slurm pour suivi facile
            with open(os.path.join(submitit_job_log_folder, f"slurm_job_id_{slurm_job_id_str}.txt"), "w") as f_id:
                f_id.write(f"Job ID: {slurm_job_id_str}\n")
                f_id.write(f"Sujet: {TARGET_SUBJECT_ID_FOR_JOB}\n")
                f_id.write(f"Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}\n")
                f_id.write(f"Soumis le: {datetime.now().isoformat()}\n")

    except Exception as e_submit:
        logger.critical(
            f"Erreur critique lors de la soumission du job pour {TARGET_SUBJECT_ID_FOR_JOB}: {e_submit}", exc_info=True)
        return

    if not submitted_job:
        logger.error(
            "La soumission du job a échoué, aucun objet job retourné.")
        return

    logger.info(
        f"\n--- Attente du résultat du job (ID Submitit initial: {submitted_job}) ---")
    try:
        subject_job_results = submitted_job.result()  # Bloquant jusqu'à la fin du job
        final_job_id = str(
            getattr(submitted_job, 'job_id', 'ID_Final_Inconnu'))
        logger.info(
            f"Job {final_job_id} (Sujet: {TARGET_SUBJECT_ID_FOR_JOB}) terminé. État Submitit: {submitted_job.state}")

        if isinstance(subject_job_results, dict) and \
           subject_job_results.get("subject_id") == TARGET_SUBJECT_ID_FOR_JOB and \
           subject_job_results.get("protocol_type_processed") == TARGET_PROTOCOL_TYPE_FOR_JOB:

            # Clés spécifiques au protocole PP_AP
            mean_global_auc = subject_job_results.get(
                'pp_ap_main_mean_auc_global', np.nan)
            global_metrics = subject_job_results.get(
                "pp_ap_main_global_metrics", {})

            auc_str = f"{mean_global_auc:.3f}" if pd.notna(
                mean_global_auc) else "N/A"
            logger.info(f"  Résultat pour {TARGET_SUBJECT_ID_FOR_JOB} (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}): "
                        f"Moyenne Global AUC (pp_ap_main) = {auc_str}")
            if global_metrics:
                logger.info(
                    f"  Métriques globales (pp_ap_main): {global_metrics}")

            # Vérifier si les moyennes AP-centriques ont été produites (elles sont dans une liste)
            ap_centric_results = subject_job_results.get(
                "pp_ap_ap_centric_avg_results", [])
            if ap_centric_results:
                num_valid_ap_centric_avg = sum(
                    1 for r in ap_centric_results
                    if r is not None and r.get('average_scores_1d') is not None and r.get('num_constituent_curves', 0) >= 2
                )
                logger.info(
                    f"  Nombre de moyennes AP-centriques valides (>=2 courbes) générées: {num_valid_ap_centric_avg} sur {len(ap_centric_results)} potentielles.")
                # 6 AP families
                if num_valid_ap_centric_avg < len(ALL_SUBJECT_GROUPS.get(target_subject_group, [])) and num_valid_ap_centric_avg < 6:
                    logger.warning(
                        f"  Moins de 6 moyennes AP-centriques valides. La page 8 du dashboard pourrait être incomplète ou manquante.")

            else:
                logger.warning(
                    f"  Aucun résultat pour 'pp_ap_ap_centric_avg_results' trouvé dans les résultats du job.")

        else:
            logger.warning(f"  Résultat inattendu ou vide pour {TARGET_SUBJECT_ID_FOR_JOB}. "
                           f"Type reçu: {type(subject_job_results)}. "
                           f"ID sujet attendu: {TARGET_SUBJECT_ID_FOR_JOB}, reçu: {subject_job_results.get('subject_id')}. "
                           f"Protocole attendu: {TARGET_PROTOCOL_TYPE_FOR_JOB}, reçu: {subject_job_results.get('protocol_type_processed')}. "
                           f"Contenu (partiel): {str(subject_job_results)[:500]}")  # Augmenter la taille du log partiel
    except Exception as e_result:
        final_job_id_err = str(
            getattr(submitted_job, 'job_id', 'ID_Erreur_Inconnu'))
        logger.error(f"Erreur lors de la récupération du résultat ou échec du job {final_job_id_err} "
                     f"(Sujet: {TARGET_SUBJECT_ID_FOR_JOB}):\n{e_result}", exc_info=False)  # exc_info=False pour ne pas dupliquer le traceback du worker
        logger.warning(
            f"État du job submitit au moment de l'erreur: {getattr(submitted_job, 'state', 'N/A')}")
        logger.warning(
            f"Consultez les logs du job dans: {os.path.abspath(submitit_job_log_folder)} pour le traceback complet du worker.")

    logger.info(
        f"--- Script de soumission pour le sujet {TARGET_SUBJECT_ID_FOR_JOB} (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) terminé. ---")


if __name__ == "__main__":
    logger.info(
        f"Lancement de la logique de soumission principale (main_submission_logic)...")
    main_submission_logic()
    logger.info(
        f"Fin du script de soumission principal. Logs principaux dans : {MASTER_LOG_FILE}")
