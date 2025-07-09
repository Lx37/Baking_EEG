

# --- Imports ---
import os
import sys
import logging
from datetime import datetime
import getpass
import time
import submitit
from submitit.core.utils import FailedJobError

ALL_SUBJECT_GROUPS = {
    "CONTROLS_DELIRIUM": [
        "TPC2",
        "TPLV4",
        "TTDV5",
        "TYS6",
        "LAB1",
        "LAG6",
        "LAT3",
        "LBM4",
        "LCM2",
        "LP05",
        "TJL3",
        "TJR7",
        "TLP8",
    ],
    "CONTROLS_COMA": [
        "AO05",
        "BT13",
        "HM10",
        "JM14",
        "LS07",
        "LT12",
        "PB20",
        "SB09",
        "SP03",
        "TAK7",
        "TCG5",
        "TEN1",
        "TFB6",
        "TGD8",
        "TJL3",
        "TNC11",
        "TSS4",
        "TTV2",
        "TVM10",
        "TVR9",
        "TZ11",
        "VB01",
        "FG104",
        "FP102",
        "MB103",
    ],
    "COMA": [
        "AD94",
        "AE93",
        "AM88",
        "AP84",
        "AS_FRA",
        "BO_AXE",
        "BR_JEA",
        "BS81",
        "CA55",
        "CI_MIC",
        "CS38",
        "DE_HEN",
        "DR92",
        "DU_CHR",
        "FB83",
        "GA_MAR",
        "GV77",
        "JC39",
        "JM78",
        "JR79",
        "KS76",
        "LA_PIE",
        "MA_VAL",
        "MB73",
        "ME63",
        "ME64",
        "MCS",
        "MP68",
        "MV48",
        "NN65",
        "RE_JOS",
        "SB67",
        "TF53",
        "TpAB15J1",
        "TpAF11J1",
        "TpAT19J1",
        "TpBD10J1",
        "TpCF24J1",
        "TpCL14j1",
        "TpDC12J1",
        "TpDC12J8",
        "TpDC22J1",
        "TpDL8J1",
        "TpDP7J1",
        "TpDP7J8",
        "TpDP7_Surnom",
        "TpEM13J1",
        "TpEP16J1",
        "TpEP20J1",
        "TpFJ5J1",
        "TpFJ5J8",
        "TpFM25J1",
        "TpJM2J1",
        "TpLC21J1",
        "TpLJ6J1",
        "TpMD4J1",
        "TpMG17J1",
        "TpML3J1",
        "TpPC23J1",
        "TpTP1J1",
        "TpTP1J8",
        "TT45",
        "YG72",
    ],
    "VS": [
        "CB31",
        "DZ44",
        "FM60",
        "FR43",
        "GM37",
        "GU32",
        "HZ24",
        "KA70",
        "KG85",
        "MH74",
        "MM86",
        "OD69",
        "OS90",
        "PB28",
        "SR57",
        "TL36",
        "TpAB15J8",
        "TpAT19J8",
        "TpBD10J8",
        "TpDC22J8",
        "TpDL8J8",
        "TpEP16j8",
        "TpFM25J8",
        "TpLJ6J8",
        "TpML3J8",
        "VS91",
    ],
    "MCS-": [
        "BT25",
        "CB34",
        "CG29",
        "CR26",
        "MC40",
        "ML33",
    ],
    "MCS+": [
        "AG42",
        "CW41",
        "DA75",
        "GT50",
        "HM52",
        "JA71",
        "KN49",
        "LP54",
        "MC58",
        "MK80",
        "PL82",
        "SR59",
        "TB56",
        "TpCF24J8",
        "PE_SAM",
        "YG66",
        "IR27",
        "NF35",
    ],
    "DELIRIUM+": [
        "TpAB19",
        "TpAK24",
        "TpAK27",
        "TpBL47",
        "TpCB15",
        "TpCF1",
        "TpDRL3",
        "TpFF34",
        "TpFY57",
        "TpJA20",
        "TpJB25",
        "TpJB26",
        "TpJC5",
        "TpJCD29",
        "TpJLR17",
        "TpJPS55",
        "TpLA28",
        "TpMB45",
        "TpMM4",
        "TpMN42",
        "TpPC21",
        "TpPM14",
        "TpPM31",
        "TpRD38",
        "TpSM49",
    ],
    "DELIRIUM-": [
        "TpAC23",
        "TpAG51",
        "TpAM43",
        "TpBD16",
        "TpDD2",
        "TpFB18",
        "TpFL53",
        "TpGB8",
        "TpGT32",
        "TpJPG7",
        "TpJPL10",
        "TpKS6",
        "TpLP11",
        "TpMA9",
        "TpMD13",
        "TpMD52",
        "TpME22",
        "TpPA35",
        "TpPI46",
        "TpPL48",
        "TpRB50",
        "TpRK39",
        "TpSD30",
        "TpYB41",
    ],
    "DELIRIUM+": [
        "TpAB19", "TpAK24", "TpAK27", "TpBL47", "TpCB15", "TpCF1", "TpDRL3", "TpFF34",
        "TpFY57", "TpJA20", "TpJB25", "TpJB26", "TpJC5", "TpJCD29", "TpJLR17", "TpJPS55",
        "TpLA28", "TpMB45", "TpMM4", "TpMN42", "TpPC21", "TpPM14", "TpPM31", "TpRD38",
        "TpSM49",
    ],
    "DELIRIUM-": [
        "TpAC23", "TpAG51", "TpAM43", "TpBD16", "TpDD2", "TpFB18", "TpFL53", "TpGB8",
        "TpGT32", "TpJPG7", "TpJPL10", "TpKS6", "TpLP11", "TpMA9", "TpMD13", "TpMD52",
        "TpME22", "TpPA35", "TpPI46", "TpPL48", "TpRB50", "TpRK39", "TpSD30", "TpYB41",
    ],
}

# --- Configuration Initiale des Chemins et Variables Globales ---
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    CURRENT_SCRIPT_DIR = os.getcwd()
    print(f"AVERTISSEMENT: __file__ non défini. CURRENT_SCRIPT_DIR initialisé à : {CURRENT_SCRIPT_DIR}", file=sys.stderr)

# --- Détermination de PROJECT_ROOT_FOR_PYTHONPATH ---
project_root_tentative = os.path.dirname(CURRENT_SCRIPT_DIR)
# (Le reste de la logique de détection du chemin racine reste inchangé)
if os.path.isdir(os.path.join(project_root_tentative, "examples")):
    PROJECT_ROOT_FOR_PYTHONPATH = project_root_tentative
else:
    # (Logique de fallback inchangée)
    hardcoded_project_root = "/home/tom.balay/Baking_EEG" 
    if os.path.isdir(os.path.join(hardcoded_project_root, "examples")):
        PROJECT_ROOT_FOR_PYTHONPATH = hardcoded_project_root
    else:
        print("ERREUR CRITIQUE: Impossible de localiser le dossier racine du projet contenant 'examples'.")
        sys.exit(1)

if PROJECT_ROOT_FOR_PYTHONPATH not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_PYTHONPATH)
    
# --- Configuration du Logger Principal ---
LOG_DIR_SUBMITIT_MASTER = os.path.join(CURRENT_SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(LOG_DIR_SUBMITIT_MASTER, exist_ok=True)

TARGET_PROTOCOL_TYPE_FOR_JOB = "PP_AP" # Protocole fixe pour tous les jobs

MASTER_LOG_FILE_NAME = datetime.now().strftime(
    f'master_submitit_ALL_SUBJECTS_{TARGET_PROTOCOL_TYPE_FOR_JOB}_%Y-%m-%d_%H-%M-%S.log'
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

logger.info(f"--- Script de soumission Submitit pour TOUS les sujets (Protocole: {TARGET_PROTOCOL_TYPE_FOR_JOB}) démarré ---")
logger.info(f"Log principal de ce script de soumission: {MASTER_LOG_FILE_PATH}")
logger.info(f"PROJECT_ROOT_FOR_PYTHONPATH (ajouté à sys.path): {PROJECT_ROOT_FOR_PYTHONPATH}")

try:
    from utils.utils import configure_project_paths
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_CSP_FOR_TEMPORAL_PIPELINES,
        USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES, PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
        FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS, SAVE_ANALYSIS_RESULTS,
        GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT, CHANCE_LEVEL_AUC
    )
    logger.info("Importations depuis 'config' et 'utils' réussies.")
except (ModuleNotFoundError, ImportError) as e:
    logger.critical(f"ERREUR CRITIQUE lors des imports. Vérifiez PROJECT_ROOT_FOR_PYTHONPATH. Erreur: {e}", exc_info=True)
    sys.exit(1)

# --- Configuration du Job Slurm (commune à tous les jobs) ---
PATH_TO_VENV_ACTIVATE_ON_CLUSTER = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
PROJECT_ROOT_ON_CLUSTER_FOR_JOB = PROJECT_ROOT_FOR_PYTHONPATH

# La logique de diagnostic du worker reste la même.
python_diagnostic_script_content = f"""
# ... (contenu du script de diagnostic inchangé) ...
"""
python_diagnostic_script_content_for_heredoc = python_diagnostic_script_content.replace("{", "{{").replace("}", "}}")
python_diagnostic_script_content_for_heredoc = python_diagnostic_script_content_for_heredoc.replace(f"{{{{{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}}}}}", f"{{{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}}}")

SETUP_COMMANDS_FOR_SLURM_JOB_CPU = f"""#!/bin/bash
# ... (le reste des commandes de setup est inchangé) ...
echo "INFO: Configuring PYTHONPATH for the job..."
export PYTHONPATH="{PROJECT_ROOT_ON_CLUSTER_FOR_JOB}${{PYTHONPATH:+:$PYTHONPATH}}"
echo "INFO: PYTHONPATH for job set to: [$PYTHONPATH]"
# ... (le reste des commandes de setup est inchangé) ...
"""

# --- Wrapper pour l'exécution sur le worker (inchangé) ---
def execute_single_subject_decoding_wrapper(**kwargs):
    """
    Wrapper function that performs the import inside the worker node.
    """
    import sys
    project_root = "/home/tom.balay/Baking_EEG"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from examples.run_decoding_one_pp import execute_single_subject_decoding
    return execute_single_subject_decoding(**kwargs)

# --- LOGIQUE DE SOUMISSION PRINCIPALE (MODIFIÉE) ---
def main_submission_logic():
    logger.info("--- Début de la logique de soumission principale (main_submission_logic) ---")

    try:
        user_for_paths = getpass.getuser()
        logger.info(f"Utilisateur détecté (getpass): {user_for_paths}")
    except Exception:
        user_for_paths = os.environ.get('LOGNAME') or os.environ.get('USER') or "unknown_user"
        logger.info(f"Utilisateur (fallback): {user_for_paths}")

    base_input_path, base_output_path = configure_project_paths(user_for_paths)
    if not os.path.isdir(base_input_path):
        logger.critical(f"ERREUR: Chemin d'entrée '{base_input_path}' invalide. Arrêt.")
        sys.exit(1)
    logger.info(f"Chemin de base des données d'entrée: {base_input_path}")
    logger.info(f"Chemin de base des résultats de sortie: {base_output_path}")

    # --- Configuration Slurm et Submitit ---
    N_CPUS_FOR_JOB = 40
    MEMORY_FOR_JOB = "60G"
    TIMEOUT_MINUTES = 1200000 * 60
    SLURM_PARTITION = "CPU"
    SLURM_ACCOUNT = "tom.balay"

    logger.info(
        f"Ressources Slurm pour chaque job: CPUs={N_CPUS_FOR_JOB}, Mémoire={MEMORY_FOR_JOB}, "
        f"Timeout={TIMEOUT_MINUTES}min, Partition={SLURM_PARTITION}, Account={SLURM_ACCOUNT or 'N/A'}"
    )

    # Crée un dossier de log unique pour cette exécution groupée
    current_timestamp_for_log = datetime.now().strftime('%Y%m%d_%H%M%S')
    submitit_job_log_folder = os.path.join(
        CURRENT_SCRIPT_DIR, "logs_submitit_jobs",
        f"run_{TARGET_PROTOCOL_TYPE_FOR_JOB}_{current_timestamp_for_log}"
    )
    os.makedirs(submitit_job_log_folder, exist_ok=True)
    logger.info(f"Logs Submitit pour cette série de jobs dans: {os.path.abspath(submitit_job_log_folder)}")

    executor = submitit.AutoExecutor(folder=submitit_job_log_folder)
    executor.update_parameters(
        timeout_min=TIMEOUT_MINUTES,
        setup=SETUP_COMMANDS_FOR_SLURM_JOB_CPU.splitlines(),
        slurm_partition=SLURM_PARTITION,
        slurm_mem=MEMORY_FOR_JOB,
        slurm_cpus_per_task=N_CPUS_FOR_JOB,
        slurm_additional_parameters={"account": SLURM_ACCOUNT} if SLURM_ACCOUNT else {}
    )

    submitted_jobs = []
    total_subjects = sum(len(sub_list) for sub_list in ALL_SUBJECT_GROUPS.values())
    processed_count = 0

    # --- Boucle sur tous les groupes et tous les sujets ---
    for group_name, subjects_in_group in ALL_SUBJECT_GROUPS.items():
        if not subjects_in_group:
            logger.info(f"Groupe '{group_name}' est vide, ignoré.")
            continue
        
        for subject_id in subjects_in_group:
            processed_count += 1
            logger.info(f"--- Préparation du job {processed_count}/{total_subjects}: Sujet={subject_id}, Groupe={group_name} ---")

            loading_config_for_job = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT

            kwargs_for_function_call = {
                "subject_identifier": subject_id,
                "group_affiliation": group_name,
                "decoding_protocol_identifier": f'Analysis_{TARGET_PROTOCOL_TYPE_FOR_JOB}_Individual',
                "save_results_flag": SAVE_ANALYSIS_RESULTS, "enable_verbose_logging": True,
                "generate_plots_flag": GENERATE_PLOTS, "base_input_data_path": base_input_path,
                "base_output_results_path": base_output_path, "n_jobs_for_processing": N_CPUS_FOR_JOB,
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

            try:
                logger.info(f"Soumission du job pour {subject_id}...")
                job = executor.submit(execute_single_subject_decoding_wrapper, **kwargs_for_function_call)
                submitted_jobs.append(job)
                logger.info(f"Job pour {subject_id} soumis avec succès. ID: {job.job_id}")
            except Exception as e_submit:
                logger.error(f"Erreur lors de la soumission du job pour {subject_id}: {e_submit}", exc_info=True)

    # --- Attente et collecte des résultats ---
    logger.info(f"\n--- {len(submitted_jobs)}/{total_subjects} jobs ont été soumis. Attente des résultats... ---")

    successful_jobs = []
    failed_jobs = []

    for job in submitted_jobs:
        subject_info = job.kwargs['function_kwargs']['subject_identifier']
        logger.info(f"Attente du résultat pour le job {job.job_id} (Sujet: {subject_info})...")
        try:
            result = job.result()
            logger.info(f"Job {job.job_id} (Sujet: {subject_info}) terminé avec succès. État: {job.state}")
            successful_jobs.append({'job': job, 'result': result})
        except FailedJobError as e_failed_job:
            logger.error(f"Job Slurm {job.job_id} (Sujet: {subject_info}) A ÉCHOUÉ.", exc_info=False)
            logger.error(f"  Message Submitit: {e_failed_job}")
            logger.error(f"  Consultez les logs du worker dans : {os.path.abspath(submitit_job_log_folder)}")
            failed_jobs.append(job)
        except Exception as e_result:
            logger.error(f"Erreur lors de la récupération du résultat du job {job.job_id} (Sujet: {subject_info}): {e_result}", exc_info=True)
            failed_jobs.append(job)
            
    # --- Résumé Final ---
    logger.info("\n" + "="*80)
    logger.info("--- RÉSUMÉ DE L'EXÉCUTION ---")
    logger.info(f"Total de sujets à traiter: {total_subjects}")
    logger.info(f"Jobs soumis avec succès: {len(submitted_jobs)}")
    logger.info(f"Jobs terminés avec succès: {len(successful_jobs)}")
    logger.info(f"Jobs en échec: {len(failed_jobs)}")
    if failed_jobs:
        failed_subjects = [j.kwargs['function_kwargs']['subject_identifier'] for j in failed_jobs]
        logger.warning(f"Liste des sujets dont le job a échoué: {failed_subjects}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    logger.info(f"--- Démarrage du script de soumission principal ({os.path.basename(__file__)}) ---")
    try:
        main_submission_logic()
    except Exception as e_main:
        logger.critical(f"Erreur non gérée dans main_submission_logic: {e_main}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info(f"--- Fin du script de soumission principal. Log complet dans: {MASTER_LOG_FILE_PATH} ---")