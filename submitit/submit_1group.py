#!/usr/bin/env python3
import os
import sys
import logging
from datetime import datetime
import getpass
import numpy as np
import pandas as pd
import time
import submitit
import traceback

# Variables globales définies au début
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
log_dir_submitit_master = os.path.join(SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(log_dir_submitit_master, exist_ok=True)
master_log_file = None  # Sera défini dans configure_logging

def debug_project_structure(project_root):
    """Affiche la structure du projet pour le débogage."""
    logger = logging.getLogger(__name__)

    if not os.path.exists(project_root):
        logger.error(f"Le répertoire du projet n'existe pas: {project_root}")
        return False

    logger.info(f"Contenu du répertoire du projet ({project_root}):")
    try:
        for item in os.listdir(project_root):
            item_path = os.path.join(project_root, item)
            if os.path.isdir(item_path):
                logger.info(f"  [D] {item}/")
            else:
                logger.info(f"  [F] {item}")
    except Exception as e:
        logger.error(f"Impossible de lister le contenu du projet: {e}")
        return False

    examples_path = os.path.join(project_root, "examples")
    if not os.path.exists(examples_path):
        logger.error(f"Le répertoire 'examples' est introuvable dans: {project_root}")
        return False

    if not os.path.isdir(examples_path):
        logger.error(f"'examples' existe mais n'est pas un répertoire: {examples_path}")
        return False

    init_file = os.path.join(examples_path, "__init__.py")
    if not os.path.exists(init_file):
        logger.warning(f"Fichier __init__.py manquant dans examples: {init_file}")

    run_decoding_file = os.path.join(examples_path, "run_decoding_one_group_pp.py")
    if not os.path.exists(run_decoding_file):
        logger.error(f"Fichier run_decoding_one_group_pp.py manquant dans: {examples_path}")
        return False

    logger.info("Structure du projet validée avec succès.")
    return True

def configure_logging():
    """Configure le logging avec plus de détails pour le débogage."""
    global master_log_file

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    master_log_file = os.path.join(
        log_dir_submitit_master,
        f'master_submitit_CONTROLS_ONLY_INTRA_{current_time_str}.log'
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(master_log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    return logging.getLogger(__name__)

# Configuration initiale du logger pour les erreurs précoces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Chemin du script: {SCRIPT_DIR}")
logger.info(f"Racine du projet calculée: {PROJECT_ROOT}")

# Reconfiguration complète du logging après la configuration initiale
logger = configure_logging()

# Vérification de la structure du projet
if not debug_project_structure(PROJECT_ROOT):
    logger.error("La structure du projet n'est pas valide. Vérifiez les chemins et la structure.")
    sys.exit(1)

# Ajout de la racine du projet au sys.path
if PROJECT_ROOT not in sys.path:
    logger.info(f"Ajout de {PROJECT_ROOT} à sys.path")
    sys.path.insert(0, PROJECT_ROOT)
else:
    logger.info(f"{PROJECT_ROOT} est déjà dans sys.path")

# Affichage des informations de débogage
logger.info(f"sys.path actuel: {sys.path}")
logger.info(f"Contenu de PROJECT_ROOT: {os.listdir(PROJECT_ROOT)}")

# Vérification que Python peut trouver le module examples
try:
    import examples
    logger.info(f"Module 'examples' trouvé à: {os.path.dirname(examples.__file__)}")
except ImportError as e:
    logger.error(f"Impossible d'importer le module examples: {e}")
    logger.error("Ceci est probablement dû à un problème de chemin. Vérifiez:")
    logger.error(f"1. PROJECT_ROOT est défini comme: {PROJECT_ROOT}")
    logger.error(f"2. Le répertoire 'examples' existe à cette adresse: {os.path.join(PROJECT_ROOT, 'examples')}")
    logger.error("3. Que PROJECT_ROOT est dans sys.path")
    sys.exit(1)

# Maintenant que nous savons que le module est importable, nous pouvons faire les imports normaux
try:
    from examples.run_decoding_one_group_pp import (
        execute_group_intra_subject_decoding_analysis,
        configure_project_paths
    )
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL, FIXED_CLASSIFIER_PARAMS_CONFIG,
        COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_INTRA_SUBJECT_STATISTICS,
        N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE, T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS
    )
    from config.config import ALL_SUBJECTS_GROUPS
    logger.info("Toutes les importations depuis le projet ont réussi.")
except Exception as e:
    logger.error(f"ERREUR CRITIQUE D'IMPORTATION: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    logger.error(f"Vérifiez que la racine du projet '{PROJECT_ROOT}' est correcte et dans sys.path.")
    logger.error(f"sys.path actuel: {sys.path}")
    sys.exit(1)

# --- PARAMÈTRES SLURM ---
PATH_TO_VENV_ACTIVATE = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"
logger.info(f"Utilisation de l'environnement virtuel pour les jobs Slurm: {PATH_TO_VENV_ACTIVATE}")
logger.info(f"Racine du projet pour PYTHONPATH dans les jobs Slurm: {PROJECT_ROOT}")

# Configuration SLURM améliorée avec plus de logs de débogage
SETUP_COMMANDS_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm ---"
echo "Date et heure: $(date)"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial: $(pwd)"
echo "Utilisateur: $(whoami)"
echo "Contenu du répertoire courant: $(ls)"

# Purge des modules et activation du venv
module purge
echo "Activation de l'environnement virtuel: {PATH_TO_VENV_ACTIVATE}"
source {PATH_TO_VENV_ACTIVATE}
if [ $? -ne 0 ]; then
    echo "ERREUR: Échec de l'activation de l'environnement virtuel."
    exit 1
fi

# Configuration cruciale du PYTHONPATH
echo "Configuration de PYTHONPATH pour le job..."
export PYTHONPATH="{PROJECT_ROOT}:${{PYTHONPATH}}"
echo "PYTHONPATH actuel du job: $PYTHONPATH"

# Vérification de la structure du projet sur le nœud
if [ -d "{PROJECT_ROOT}" ]; then
    echo "Contenu du répertoire du projet sur le nœud:"
    ls -l "{PROJECT_ROOT}"
    echo "Contenu du répertoire examples:"
    ls -l "{os.path.join(PROJECT_ROOT, 'examples')}"
else
    echo "ERREUR: Le répertoire du projet n'est pas accessible: {PROJECT_ROOT}"
    exit 1
fi

echo "--- Environnement CPU configuré (venv) ---"
echo "Chemin Python utilisé: $(which python)"
echo "Version Python: $(python -V)"
echo "Contenu de sys.path dans Python:"
python -c "import sys; print('\n'.join(sys.path))"

# Vérification spécifique de l'importation
echo "Test d'importation du module examples:"
python -c "import sys; sys.path.insert(0, '{PROJECT_ROOT}'); from examples import run_decoding_one_group_pp; print('Importation réussie')"

echo "Version de MNE:"
python -c "import mne; print(mne.__version__)"
echo "-------------------------------------------"
"""

def main():
    try:
        user = getpass.getuser()
        logger.info(f"Utilisateur détecté: {user}")
    except Exception as e:
        user = "unknown_user"
        logger.warning(f"Impossible de déterminer l'utilisateur: {e}")

    try:
        base_input_path, base_output_path = configure_project_paths(user)
        logger.info(f"Chemin de base des données d'entrée: {base_input_path}")
        logger.info(f"Chemin de base des résultats: {base_output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la configuration des chemins: {e}")
        sys.exit(1)

    GROUP_NAME = 'CONTROLS_DELIRIUM'
    logger.info(f"Groupe ciblé pour l'analyse : {GROUP_NAME}")

    try:
        subjects_for_this_group = ALL_SUBJECTS_GROUPS.get(GROUP_NAME)
        if not subjects_for_this_group:
            logger.error(f"Groupe '{GROUP_NAME}' non trouvé ou vide dans la configuration. Arrêt.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des sujets pour le groupe: {e}")
        sys.exit(1)

    SLURM_CPUS_PER_GROUP_JOB = 40
    group_job_slurm_params = {
        "timeout_min": 60 * 60,
        "slurm_additional_parameters": {"account": "tom.balay"},
        "local_setup": SETUP_COMMANDS_CPU,
        "slurm_partition": "CPU",
        "slurm_mem": "80G",
        "slurm_cpus_per_task": SLURM_CPUS_PER_GROUP_JOB,
    }

    current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_folder_submitit_jobs = os.path.join(log_dir_submitit_master, f"slurm_job_{GROUP_NAME}_{current_time_str}")

    try:
        executor = submitit.AutoExecutor(folder=log_folder_submitit_jobs)
        logger.info(f"Submitit AutoExecutor initialisé. Logs du job dans: {log_folder_submitit_jobs}")
        executor.update_parameters(**group_job_slurm_params)
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'executor submitit: {e}")
        sys.exit(1)

    func_kwargs_group_analysis = {
        "subject_ids_in_group": subjects_for_this_group,
        "group_identifier": GROUP_NAME,
        "decoding_protocol_identifier": f'Group_Intra_{GROUP_NAME}',
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path,
        "enable_verbose_logging": True,
        "n_jobs_for_each_subject": SLURM_CPUS_PER_GROUP_JOB,
        "n_jobs_for_group_cluster_stats": SLURM_CPUS_PER_GROUP_JOB,
        "classifier_type_for_group_runs": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_group": USE_GRID_SEARCH_OPTIMIZATION,
        "fixed_params_for_group": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "use_csp_for_temporal_group": False,
        "use_anova_fs_for_temporal_group": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        "param_grid_config_for_group": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_group": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
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
    }

    job = None
    try:
        job = executor.submit(execute_group_intra_subject_decoding_analysis, **func_kwargs_group_analysis)
        logger.info(f"Job pour le groupe {GROUP_NAME} (ID: {job.job_id}) soumis au batch.")
    except Exception as e:
        logger.error(f"Erreur lors de la soumission du job : {e}", exc_info=True)
        return

    # Initialisation des variables pour éviter UnboundLocalError
    job_final_id = str(getattr(job, 'job_id', 'ID_INCONNU'))
    job_state = "SUBMISSION_FAILED"
    mean_auc = np.nan
    num_subjects = 0

    logger.info(f"Attente du résultat du job {job_final_id}...")
    try:
        output_subject_auc_dict = job.result()
        job_state = job.state
        logger.info(f"Job {job_final_id} (Groupe: {GROUP_NAME}) terminé. État: {job_state}")

        if isinstance(output_subject_auc_dict, dict) and output_subject_auc_dict:
            valid_aucs = [auc for auc in output_subject_auc_dict.values() if pd.notna(auc)]
            if valid_aucs:
                mean_auc = np.mean(valid_aucs)
            num_subjects = len(output_subject_auc_dict)
            logger.info(f"  Résultat: Moyenne AUCs = {mean_auc:.3f} (N={num_subjects} sujets)")
        else:
            logger.warning(f"  Résultat inattendu/vide pour {GROUP_NAME}. Type reçu: {type(output_subject_auc_dict)}")
    except Exception as e_result:
        job_state = job.state if job else "UNKNOWN"
        logger.error(f"Erreur lors de la récupération du résultat du job {job_final_id}. État final: {job_state}", exc_info=True)

    logger.info("--- Résumé final ---")
    logger.info(f"Groupe: {GROUP_NAME} | Job ID: {job_final_id} | État: {job_state} | Moyenne AUCs: {mean_auc if pd.notna(mean_auc) else 'N/A'} | N Sujets: {num_subjects}")
    logger.info(f"Script de soumission terminé. Log principal: {master_log_file}")
    logger.info(f"Logs du job Slurm (stdout/stderr) dans: {log_folder_submitit_jobs}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erreur inattendue dans le script principal: {e}", exc_info=True)
        sys.exit(1)
