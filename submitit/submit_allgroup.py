"""
==============================================================================
 File name       : submit_allgroup.py
 Author          : Tom Balay (and a bit Copilot)
 Created         : 2025-07-29
 Description     :
 This script submits EEG PP decoding jobs for all subjects in all groups, each group on a separate node.
 It prepares and submits a job for each subject, handling environment setup, job submission, and logging.
 The script is designed for the Baking_EEG project and uses predefined configurations
 for decoding parameters and data paths.
 It manages decoding results, intra-subject statistics, and temporal generalization matrices (TGM).
 Results and logs are saved for each subject.

 BSD 3-Clause License 2025, CNRS, Tom Balay
==============================================================================
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import getpass
import numpy as np
import pandas as pd
import submitit
from submitit.core.utils import FailedJobError
import traceback

# --- 1. GESTION ROBUSTE DES CHEMINS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if not os.path.isdir(os.path.join(PROJECT_ROOT, "examples")):
    logging.critical(f"ERREUR: Impossible de trouver le dossier 'examples' dans la racine de projet supposée: {PROJECT_ROOT}")
    sys.exit(1)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- 2. CONFIGURATION DU LOGGING ---
log_dir_submitit_master = os.path.join(SCRIPT_DIR, 'logs_submitit_master')
os.makedirs(log_dir_submitit_master, exist_ok=True)

current_time_str_log = datetime.now().strftime('%Y%m%d_%H%M%S')
master_log_file = os.path.join(
    log_dir_submitit_master,
    f'master_submitit_allgroup_{current_time_str_log}.log'
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
logger = logging.getLogger(__name__)

logger.info("--- Démarrage du script de soumission (un job Slurm par groupe) ---")
logger.info(f"Racine du projet détectée et ajoutée à sys.path : {PROJECT_ROOT}")
logger.info(f"Log principal de ce script : {master_log_file}")

# --- 3. IMPORTS DU PROJET ---
try:
    from utils.utils import configure_project_paths
    from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
    from config.decoding_config import (
        CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL, FIXED_CLASSIFIER_PARAMS_CONFIG,
        COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_INTRA_SUBJECT_STATISTICS,
        N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE, T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
        INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS
    )
    from config.config import ALL_SUBJECTS_GROUPS
    logger.info("Toutes les importations depuis le projet ont réussi.")
except ImportError as e:
    logger.critical(f"ERREUR CRITIQUE D'IMPORTATION: {e}", exc_info=True)
    sys.exit(1)

# --- 4. PARAMÈTRES SLURM ET SCRIPT DE SETUP ---
PATH_TO_VENV_ACTIVATE = "/home/tom.balay/.venvs/py3.11_cluster/bin/activate"

# Configuration complète de l'environnement pour le job Slurm
SETUP_COMMANDS_CPU = f"""
echo "--- Configuration de l'environnement pour le job Slurm ---"
echo "Hostname: $(hostname)"
echo "Job ID Slurm: $SLURM_JOB_ID"
echo "Répertoire de travail initial: $(pwd)"

# Changement du répertoire de travail vers la racine du projet
echo "Changement du répertoire de travail vers {PROJECT_ROOT}"
cd "{PROJECT_ROOT}"
if [ $? -ne 0 ]; then echo "ERREUR: Échec du changement de répertoire."; exit 1; fi
echo "Nouveau répertoire de travail: $(pwd)"

# Configuration du PYTHONPATH
export PYTHONPATH="{PROJECT_ROOT}:$PYTHONPATH"
echo "PYTHONPATH configuré: $PYTHONPATH"

# Purge des modules et activation du venv
module purge
echo "Activation de l'environnement virtuel: {PATH_TO_VENV_ACTIVATE}"
source "{PATH_TO_VENV_ACTIVATE}"
if [ $? -ne 0 ]; then echo "ERREUR: Échec de l'activation du venv."; exit 1; fi

# Vérification de l'environnement
echo "Chemin Python utilisé: $(which python)"
echo "Version Python: $(python --version)"
echo "Test d'import du module examples:"
python -c "import sys; sys.path.insert(0, '{PROJECT_ROOT}'); import examples; print('✅ Import examples réussi')" || echo "❌ Échec de l'import examples"
echo "Contenu de sys.path au démarrage de Python:"
python -c "import sys; import os; print(f'CWD: {{os.getcwd()}}'); print('\\n'.join(sys.path[:5]))"
echo "-------------------------------------------"
"""

# --- 5. FONCTION WRAPPER AMÉLIORÉE ---
def execute_decoding_wrapper(**kwargs):
    """Wrapper robuste pour s'assurer que le chemin du projet est correctement configuré sur le nœud worker."""
    import os
    import sys
    
    # Définir le chemin absolu du projet
    project_root = f"{PROJECT_ROOT}"
    
    # S'assurer qu'on est dans le bon répertoire de travail
    if os.getcwd() != project_root:
        print(f"Changement du répertoire de travail vers: {project_root}")
        os.chdir(project_root)
    
    # S'assurer que la racine du projet est dans sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Informations de débogage
    print(f"Worker node - CWD: {os.getcwd()}")
    print(f"Worker node - PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"Worker node - sys.path (premiers 3): {sys.path[:3]}")
    
    try:
        # Import de la fonction principale
        from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
        print("✅ Import successful on worker node")
        
        # Exécution de l'analyse
        return execute_group_intra_subject_decoding_analysis(**kwargs)
        
    except ImportError as e:
        print(f"❌ Import error on worker: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"sys.path: {sys.path}")
        print(f"Contents of current directory: {os.listdir('.')}")
        if os.path.exists('examples'):
            print(f"Contents of examples directory: {os.listdir('examples')}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error in worker: {e}")
        traceback.print_exc()
        raise

def _slurm_params(cpus, timeout_min, account, partition, mem):
    """Parametres Slurm communs a tous les jobs de groupe."""
    return {
        "timeout_min": timeout_min,
        "slurm_additional_parameters": {"account": account, "export": "ALL"},
        "setup": SETUP_COMMANDS_CPU.splitlines(),
        "slurm_partition": partition,
        "slurm_mem": mem,
        "slurm_cpus_per_task": cpus,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Soumet une analyse de decodage PP intra-sujet, un job "
                    "Slurm par groupe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--groups", nargs="+", default=None,
        help="Groupes a traiter. Par defaut : TOUS les groupes de "
             "ALL_SUBJECTS_GROUPS.")
    parser.add_argument("--cpus", type=int, default=40,
                        help="CPU par job de groupe.")
    parser.add_argument("--mem", type=str, default="80G",
                        help="Memoire par job de groupe.")
    parser.add_argument("--partition", type=str, default="CPU",
                        help="Partition Slurm.")
    parser.add_argument("--account", type=str, default="tom.balay",
                        help="Compte Slurm.")
    parser.add_argument(
        "--timeout-hours", type=float, default=24.0,
        help="Duree maximale par job, en heures. La valeur precedente "
             "(60*60 minutes, soit 2,5 jours) depassait le MaxTime de la "
             "plupart des partitions et faisait rejeter la soumission.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche ce qui serait soumis, sans soumettre.")
    args = parser.parse_args()

    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get('USER', 'unknown_user')
    logger.info(f"Utilisateur détecté : {user}")

    base_input_path, base_output_path = configure_project_paths(user)
    logger.info(f"Chemin de base des données d'entrée : {base_input_path}")
    logger.info(f"Chemin de base des résultats : {base_output_path}")

    # Le script s'appelait "allgroup" mais codait en dur GROUP_NAME =
    # 'CONTROLS_DELIRIUM', une cle absente de ALL_SUBJECTS_GROUPS : le .get()
    # renvoyait None et le script sortait sans rien soumettre. On itere
    # desormais reellement sur tous les groupes.
    groups_to_process = args.groups or list(ALL_SUBJECTS_GROUPS.keys())
    unknown = [g for g in groups_to_process if g not in ALL_SUBJECTS_GROUPS]
    if unknown:
        logger.error(
            "Groupe(s) inconnu(s) : %s. Groupes disponibles : %s. Arret.",
            unknown, list(ALL_SUBJECTS_GROUPS.keys()))
        sys.exit(1)

    total_subjects = sum(len(ALL_SUBJECTS_GROUPS[g]) for g in groups_to_process)
    logger.info("Groupes à traiter : %s", groups_to_process)
    logger.info("Nombre total de sujets : %d", total_subjects)

    timeout_min = max(1, int(round(args.timeout_hours * 60)))
    current_time_str_job = datetime.now().strftime('%Y%m%d_%H%M%S')

    jobs = []
    for group_name in groups_to_process:
        subjects_for_this_group = ALL_SUBJECTS_GROUPS.get(group_name)
        if not subjects_for_this_group:
            logger.warning("Groupe '%s' vide, ignoré.", group_name)
            continue

        logger.info("--- Groupe %s : %d sujets ---",
                    group_name, len(subjects_for_this_group))

        log_folder_submitit_jobs = os.path.join(
            log_dir_submitit_master,
            f"slurm_job_{group_name}_{current_time_str_job}")

        if args.dry_run:
            logger.info("[dry-run] Job non soumis pour le groupe %s "
                        "(%d sujets, %d CPU, %d min).",
                        group_name, len(subjects_for_this_group), args.cpus,
                        timeout_min)
            continue

        executor = submitit.AutoExecutor(folder=log_folder_submitit_jobs)
        executor.update_parameters(
            **_slurm_params(args.cpus, timeout_min, args.account,
                            args.partition, args.mem),
            env={**os.environ,
                 "PYTHONPATH": f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"}
        )

        func_kwargs_group_analysis = {
            "subject_ids_in_group": subjects_for_this_group,
            "group_identifier": group_name,
            "decoding_protocol_identifier": f'Group_Intra_{group_name}',
            "base_input_data_path": base_input_path,
            "base_output_results_path": base_output_path,
            "enable_verbose_logging": True,
            "n_jobs_for_each_subject": args.cpus,
            "n_jobs_for_group_cluster_stats": args.cpus,
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

        try:
            job = executor.submit(execute_decoding_wrapper,
                                  **func_kwargs_group_analysis)
            logger.info("Job groupe %s soumis (ID: %s). Logs: %s",
                        group_name, job.job_id, log_folder_submitit_jobs)
            jobs.append((group_name, job, log_folder_submitit_jobs))
        except Exception as e:
            logger.error("Erreur de soumission pour le groupe %s : %s",
                         group_name, e, exc_info=True)

    if args.dry_run:
        logger.info("[dry-run] Terminé, aucun job soumis.")
        return

    if not jobs:
        logger.error("Aucun job soumis. Arrêt.")
        sys.exit(1)

    logger.info("%d job(s) soumis. Attente des résultats...", len(jobs))
    summary = []
    for group_name, job, log_folder in jobs:
        try:
            output_subject_auc_dict = job.result()
            job_state = job.state
            if isinstance(output_subject_auc_dict, dict) and output_subject_auc_dict:
                valid_aucs = [auc for auc in output_subject_auc_dict.values()
                              if pd.notna(auc)]
                mean_auc = np.mean(valid_aucs) if valid_aucs else np.nan
                num_subjects = len(output_subject_auc_dict)
            else:
                logger.warning("Groupe %s : résultat inattendu/vide (%s).",
                               group_name, type(output_subject_auc_dict))
                mean_auc, num_subjects = np.nan, 0
        except FailedJobError as e_failed:
            logger.error("Le job du groupe %s A ÉCHOUÉ : %s", group_name,
                         e_failed)
            logger.error("Logs du worker : %s", log_folder)
            try:
                logger.error("--- STDERR ---\n%s", job.stderr())
            except Exception:
                logger.error("stderr indisponible.")
            job_state, mean_auc, num_subjects = "FAILED", np.nan, 0
        except Exception as e_result:
            logger.error("Groupe %s : erreur de récupération du résultat : %s",
                         group_name, e_result, exc_info=True)
            job_state, mean_auc, num_subjects = "UNKNOWN", np.nan, 0
        summary.append((group_name, job.job_id, job_state, mean_auc,
                        num_subjects))

    logger.info("--- Résumé final ---")
    for group_name, job_id, job_state, mean_auc, num_subjects in summary:
        logger.info("  %-18s job=%-12s état=%-10s AUC moyenne=%s  N=%d",
                    group_name, job_id, job_state,
                    f"{mean_auc:.3f}" if pd.notna(mean_auc) else "N/A",
                    num_subjects)
    logger.info("Script de soumission terminé. Log principal: %s",
                master_log_file)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Erreur inattendue dans le script principal: {e}", exc_info=True)
        sys.exit(1)