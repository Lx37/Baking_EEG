#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de soumission Submitit ROBUSTE avec WRAPPER pour un sujet unique - Protocole LG.
Cette version force la configuration du sys.path dans le worker pour une fiabilité maximale.
"""

import os
import sys
import logging
from datetime import datetime
import getpass
import submitit

# --- ÉTAPE 1: DÉFINIR LA RACINE DU PROJET ---
# Détermine le chemin absolu de la racine du projet (Baking_EEG)
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.getcwd()) # Fallback

# --- ÉTAPE 2: AJOUTER LA RACINE AU PATH POUR CE SCRIPT ---
# Indispensable pour que les imports de configuration ci-dessous fonctionnent
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- ÉTAPE 3: IMPORTER UNIQUEMENT LES CONFIGURATIONS ---
# NE PAS IMPORTER LA FONCTION D'EXÉCUTION ICI
from config.config import ALL_SUBJECT_GROUPS
from utils.utils import configure_project_paths
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION, SAVE_ANALYSIS_RESULTS,
    GENERATE_PLOTS, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    N_PERMUTATIONS_INTRA_SUBJECT, COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG, COMPUTE_INTRA_SUBJECT_STATISTICS,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, USE_CSP_FOR_TEMPORAL_PIPELINES,
    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
)

# --- ÉTAPE 4: DÉFINIR LA FONCTION WRAPPER (VERSION CORRIGÉE) ---
# C'est la partie la plus importante.

def decoding_task_wrapper(**kwargs):
    """
    Wrapper ROBUSTE qui s'exécute sur le nœud de calcul Slurm.
    Il force la configuration du sys.path AVANT toute importation.
    """
    import sys
    import os

    # Définir le chemin racine du projet en dur. C'est la garantie la plus forte.
    project_root = "/home/tom.balay/Baking_EEG"
    
    # Si le chemin n'est pas dans sys.path, on l'ajoute au début.
    # C'est la correction qui résout le "ModuleNotFoundError: No module named 'examples'".
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    # Maintenant que le sys.path est GARANTI d'être correct, on importe.
    from examples.run_decoding_one_lg import execute_single_subject_lg_decoding
    
    # Et on exécute la fonction.
    return execute_single_subject_lg_decoding(**kwargs)


# --- ÉTAPE 5: CONFIGURATION ET SOUMISSION ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fonction principale de soumission."""
    TARGET_SUBJECT_ID = "TpSM49"
    logger.info(f"--- Démarrage de la soumission pour le sujet {TARGET_SUBJECT_ID} (Protocole LG) ---")

    log_folder = f"logs_submitit_jobs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{TARGET_SUBJECT_ID}_LG"
    executor = submitit.AutoExecutor(folder=log_folder)

    # Note: On a retiré le paramètre "setup" car la logique est maintenant dans le wrapper.
    executor.update_parameters(
        timeout_min=12 * 60,
        slurm_partition="CPU",
        slurm_mem="60G",
        slurm_cpus_per_task=40,
        slurm_additional_parameters={"account": "tom.balay"}
    )

    user = getpass.getuser()
    base_input_path, base_output_path = configure_project_paths(user)
    subject_group = next((group for group, subjects in ALL_SUBJECT_GROUPS.items() if TARGET_SUBJECT_ID in subjects), "unknown")

    if subject_group == "unknown":
        logger.error(f"Sujet {TARGET_SUBJECT_ID} non trouvé. Arrêt.")
        return

    kwargs = {
        "subject_identifier": TARGET_SUBJECT_ID,
        "group_affiliation": subject_group,
        "base_input_data_path": base_input_path,
        "base_output_results_path": base_output_path,
        "n_jobs_for_processing": 40,
        "save_results_flag": SAVE_ANALYSIS_RESULTS,
        "generate_plots_flag": GENERATE_PLOTS,
        "loading_conditions_config": CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
        "classifier_type": CLASSIFIER_MODEL_TYPE,
        "use_grid_search_for_subject": USE_GRID_SEARCH_OPTIMIZATION,
        "param_grid_config_for_subject": PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
        "cv_folds_for_gs_subject": CV_FOLDS_FOR_GRIDSEARCH_INTERNAL if USE_GRID_SEARCH_OPTIMIZATION else 0,
        "fixed_params_for_subject": FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
        "compute_intra_subject_stats_flag": COMPUTE_INTRA_SUBJECT_STATISTICS,
        "n_perms_for_intra_subject_clusters": N_PERMUTATIONS_INTRA_SUBJECT,
        "compute_tgm_flag": COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
        "cluster_threshold_config_intra_fold": INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
        "use_csp_for_temporal_subject": USE_CSP_FOR_TEMPORAL_PIPELINES,
        "use_anova_fs_for_temporal_subject": USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES
    }

    logger.info("Soumission du WRAPPER (version robuste) à Slurm...")
    # On soumet la fonction WRAPPER
    job = executor.submit(decoding_task_wrapper, **kwargs)
    logger.info(f"Job soumis avec l'ID: {job.job_id}")

    try:
        result = job.result()
        logger.info(f"Job terminé avec succès. Résultat (type): {type(result)}")
    except Exception as e:
        logger.error(f"Le job a échoué: {e}", exc_info=False)
        logger.error(f"Traceback complet dans les logs du worker.")
        logger.error(f"Consultez les logs dans le dossier: {os.path.abspath(log_folder)}")

if __name__ == "__main__":
    main()