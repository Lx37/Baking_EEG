#!/usr/bin/env python3
"""
Script pour exécuter l'analyse de décodage LG pour tous les groupes automatiquement
"""

import os
import sys
import logging
import time
from datetime import datetime
import argparse
from getpass import getuser

# Add the project root to the Python path
SCRIPT_DIR_EXAMPLE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EXAMPLE = os.path.abspath(os.path.join(SCRIPT_DIR_EXAMPLE, ".."))
if PROJECT_ROOT_EXAMPLE not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_EXAMPLE)

from Baking_EEG.config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    N_JOBS_PROCESSING, SAVE_ANALYSIS_RESULTS
)
from Baking_EEG.config.config import ALL_SUBJECT_GROUPS
from Baking_EEG.utils.utils import configure_project_paths

# Import des fonctions d'analyse individuelle et de groupe
try:
    from Baking_EEG.examples.run_decoding_one_lg import execute_single_subject_lg_decoding
    from Baking_EEG.examples.run_decoding_one_group_lg import execute_group_intra_subject_lg_decoding_analysis
except ImportError as e_import:
    print(f"Erreur d'import des fonctions LG: {e_import}")
    sys.exit(1)

# Configuration du logging
LOG_DIR_RUN_ALL_LG = './logs_run_all_groups_lg'
os.makedirs(LOG_DIR_RUN_ALL_LG, exist_ok=True)
LOG_FILENAME_RUN_ALL_LG = os.path.join(
    LOG_DIR_RUN_ALL_LG,
    datetime.now().strftime('log_run_all_groups_lg_%Y-%m-%d_%H%M%S.log')
)

# Supprimer les handlers existants
for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_ALL_LG),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_all_lg = logging.getLogger(__name__)

def run_all_groups_lg_analysis(
    base_input_data_path=None,
    base_output_results_path=None,
    classifier_type=None,
    n_jobs_processing=None,
    save_results_flag=True,
    generate_plots_flag=False,  # Désactivé par défaut pour accélération
    groups_to_process=None,
    run_individual_subjects=True,
    run_group_analysis=True,
    enable_verbose_logging=False
):
    """
    Exécute l'analyse de décodage LG pour tous les groupes ou une sélection de groupes.
    
    Args:
        base_input_data_path (str): Chemin des données d'entrée
        base_output_results_path (str): Chemin de sortie des résultats
        classifier_type (str): Type de classificateur ('svc', 'logreg', 'rf')
        n_jobs_processing: Nombre de jobs pour le traitement parallèle
        save_results_flag (bool): Sauvegarder les résultats
        generate_plots_flag (bool): Générer les visualisations (désactivé par défaut)
        groups_to_process (list): Liste des groupes à traiter (None = tous)
        run_individual_subjects (bool): Exécuter l'analyse individuelle des sujets
        run_group_analysis (bool): Exécuter l'analyse de groupe
        enable_verbose_logging (bool): Logging verbeux
    """
    
    # Configuration par défaut
    if classifier_type is None:
        classifier_type = CLASSIFIER_MODEL_TYPE
    if n_jobs_processing is None:
        n_jobs_processing = N_JOBS_PROCESSING
    if base_input_data_path is None or base_output_results_path is None:
        user_login = getuser()
        cfg_input, cfg_output = configure_project_paths(user_login)
        base_input_data_path = base_input_data_path or cfg_input
        base_output_results_path = base_output_results_path or cfg_output
    
    # Déterminer les groupes à traiter
    if groups_to_process is None:
        groups_to_process = list(ALL_SUBJECT_GROUPS.keys())
    
    # Filtrer les groupes vides ou de test
    valid_groups = {}
    for group in groups_to_process:
        subjects = ALL_SUBJECT_GROUPS.get(group, [])
        if subjects and len(subjects) > 0 and group not in ['xx', 'xxx']:
            valid_groups[group] = subjects
    
    logger_run_all_lg.info("=== DÉBUT ANALYSE LG POUR TOUS LES GROUPES ===")
    logger_run_all_lg.info(f"Groupes à traiter: {list(valid_groups.keys())}")
    logger_run_all_lg.info(f"Total sujets: {sum(len(subjects) for subjects in valid_groups.values())}")
    logger_run_all_lg.info(f"Classificateur: {classifier_type}")
    logger_run_all_lg.info(f"n_jobs: {n_jobs_processing}")
    logger_run_all_lg.info(f"Sauvegarder résultats: {save_results_flag}")
    logger_run_all_lg.info(f"Générer plots: {generate_plots_flag}")
    logger_run_all_lg.info(f"Analyse individuelle: {run_individual_subjects}")
    logger_run_all_lg.info(f"Analyse de groupe: {run_group_analysis}")
    
    total_start_time = time.time()
    group_results_summary = {}
    
    for group_name, subject_list in valid_groups.items():
        logger_run_all_lg.info(f"\n{'='*60}")
        logger_run_all_lg.info(f"TRAITEMENT DU GROUPE: {group_name}")
        logger_run_all_lg.info(f"Nombre de sujets: {len(subject_list)}")
        logger_run_all_lg.info(f"Sujets: {', '.join(subject_list)}")
        logger_run_all_lg.info(f"{'='*60}")
        
        group_start_time = time.time()
        
        try:
            if run_individual_subjects:
                logger_run_all_lg.info(f"--- ANALYSE INDIVIDUELLE POUR LE GROUPE {group_name} ---")
                
                # Exécuter l'analyse individuelle pour chaque sujet
                individual_results = {}
                for i, subject_id in enumerate(subject_list, 1):
                    logger_run_all_lg.info(f"Traitement sujet {i}/{len(subject_list)}: {subject_id}")
                    
                    try:
                        subject_result = execute_single_subject_lg_decoding(
                            subject_identifier=subject_id,
                            group_affiliation=group_name,
                            decoding_protocol_identifier=f"LG_Analysis_{group_name}_{subject_id}",
                            save_results_flag=save_results_flag,
                            enable_verbose_logging=enable_verbose_logging,
                            generate_plots_flag=generate_plots_flag,
                            base_input_data_path=base_input_data_path,
                            base_output_results_path=base_output_results_path,
                            n_jobs_for_processing=n_jobs_processing,
                            classifier_type=classifier_type
                        )
                        
                        # Extraire les métriques principales
                        main_auc = subject_result.get("lg_main_mean_auc_global", float('nan'))
                        individual_results[subject_id] = {
                            'main_auc': main_auc,
                            'status': 'success'
                        }
                        
                        logger_run_all_lg.info(f"  {subject_id}: AUC principal = {main_auc:.3f}")
                        
                    except Exception as e:
                        logger_run_all_lg.error(f"Erreur pour sujet {subject_id}: {e}")
                        individual_results[subject_id] = {
                            'main_auc': float('nan'),
                            'status': 'error',
                            'error': str(e)
                        }
            
            if run_group_analysis:
                logger_run_all_lg.info(f"--- ANALYSE DE GROUPE POUR {group_name} ---")
                
                try:
                    group_result = execute_group_intra_subject_lg_decoding_analysis(
                        subject_ids_in_group=subject_list,
                        group_identifier=group_name,
                        decoding_protocol_identifier=f"LG_Group_Analysis_{group_name}",
                        save_results_flag=save_results_flag,
                        enable_verbose_logging=enable_verbose_logging,
                        generate_plots_flag=generate_plots_flag,
                        base_input_data_path=base_input_data_path,
                        base_output_results_path=base_output_results_path,
                        n_jobs_for_each_subject=n_jobs_processing,
                        n_jobs_for_group_cluster_stats=n_jobs_processing,
                        classifier_type_for_group_runs=classifier_type
                    )
                    
                    # Extraire les statistiques de groupe
                    group_auc_scores = group_result.get("subject_lg_global_auc_scores", {})
                    valid_scores = [score for score in group_auc_scores.values() if not pd.isna(score)]
                    
                    group_summary = {
                        'n_subjects_processed': len(group_result.get("processed_subject_ids", [])),
                        'n_subjects_valid': len(valid_scores),
                        'mean_auc': np.mean(valid_scores) if valid_scores else float('nan'),
                        'std_auc': np.std(valid_scores) if valid_scores else float('nan'),
                        'status': 'success'
                    }
                    
                    logger_run_all_lg.info(f"  Groupe {group_name}: {group_summary['n_subjects_valid']}/{len(subject_list)} sujets valides")
                    logger_run_all_lg.info(f"  AUC moyen du groupe: {group_summary['mean_auc']:.3f} ± {group_summary['std_auc']:.3f}")
                    
                except Exception as e:
                    logger_run_all_lg.error(f"Erreur analyse de groupe {group_name}: {e}")
                    group_summary = {
                        'n_subjects_processed': 0,
                        'n_subjects_valid': 0,
                        'mean_auc': float('nan'),
                        'std_auc': float('nan'),
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Enregistrer le résumé du groupe
            group_results_summary[group_name] = {
                'individual_results': individual_results if run_individual_subjects else {},
                'group_summary': group_summary if run_group_analysis else {},
                'processing_time': time.time() - group_start_time
            }
            
            logger_run_all_lg.info(f"Groupe {group_name} terminé en {time.time() - group_start_time:.2f}s")
            
        except Exception as e:
            logger_run_all_lg.error(f"Erreur critique pour le groupe {group_name}: {e}")
            group_results_summary[group_name] = {
                'individual_results': {},
                'group_summary': {'status': 'critical_error', 'error': str(e)},
                'processing_time': time.time() - group_start_time
            }
    
    # Résumé final
    total_time = time.time() - total_start_time
    logger_run_all_lg.info(f"\n{'='*80}")
    logger_run_all_lg.info("RÉSUMÉ FINAL DE L'ANALYSE LG POUR TOUS LES GROUPES")
    logger_run_all_lg.info(f"{'='*80}")
    logger_run_all_lg.info(f"Temps total: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    for group_name, results in group_results_summary.items():
        logger_run_all_lg.info(f"\nGroupe {group_name}:")
        if run_individual_subjects:
            individual = results['individual_results']
            success_count = sum(1 for r in individual.values() if r['status'] == 'success')
            logger_run_all_lg.info(f"  Sujets individuels: {success_count}/{len(individual)} réussis")
        
        if run_group_analysis:
            group_sum = results['group_summary']
            if group_sum['status'] == 'success':
                logger_run_all_lg.info(f"  Analyse de groupe: {group_sum['n_subjects_valid']} sujets valides")
                logger_run_all_lg.info(f"  AUC moyen: {group_sum['mean_auc']:.3f} ± {group_sum['std_auc']:.3f}")
            else:
                logger_run_all_lg.info(f"  Analyse de groupe: {group_sum['status']}")
        
        logger_run_all_lg.info(f"  Temps: {results['processing_time']:.2f}s")
    
    logger_run_all_lg.info(f"\n=== ANALYSE LG TERMINÉE POUR TOUS LES GROUPES ===")
    
    return group_results_summary

if __name__ == "__main__":
    # Import numpy et pandas ici pour éviter les erreurs
    import numpy as np
    import pandas as pd
    
    cli_parser = argparse.ArgumentParser(
        description="Script d'analyse LG pour tous les groupes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    cli_parser.add_argument("--groups", nargs='+', default=None,
                            help="Groupes spécifiques à traiter (défaut: tous)")
    cli_parser.add_argument("--clf_type", type=str, default=None, 
                            choices=["svc", "logreg", "rf"],
                            help="Type de classificateur")
    cli_parser.add_argument("--n_jobs", type=str, default=None,
                            help="Nombre de jobs (e.g., '4' ou 'auto')")
    cli_parser.add_argument("--no_save", action="store_true",
                            help="Ne pas sauvegarder les résultats")
    cli_parser.add_argument("--enable_plots", action="store_true",
                            help="Activer la génération de plots (ralentit le processus)")
    cli_parser.add_argument("--skip_individual", action="store_true",
                            help="Ignorer l'analyse individuelle des sujets")
    cli_parser.add_argument("--skip_group", action="store_true",
                            help="Ignorer l'analyse de groupe")
    cli_parser.add_argument("--verbose", action="store_true",
                            help="Logging verbeux")
    
    args = cli_parser.parse_args()
    
    # Vérifier que au moins une analyse est activée
    if args.skip_individual and args.skip_group:
        logger_run_all_lg.error("Au moins une analyse (individuelle ou groupe) doit être activée")
        sys.exit(1)
    
    # Traiter n_jobs
    n_jobs_to_use = args.n_jobs if args.n_jobs else N_JOBS_PROCESSING
    if isinstance(n_jobs_to_use, str) and n_jobs_to_use.lower() == "auto":
        n_jobs_to_use = -1
    else:
        try:
            n_jobs_to_use = int(n_jobs_to_use)
        except ValueError:
            logger_run_all_lg.warning(f"n_jobs invalide '{n_jobs_to_use}', utilisation de -1")
            n_jobs_to_use = -1
    
    logger_run_all_lg.info("DÉMARRAGE DU SCRIPT D'ANALYSE LG POUR TOUS LES GROUPES")
    logger_run_all_lg.info(f"Paramètres: groupes={args.groups}, clf={args.clf_type}, n_jobs={n_jobs_to_use}")
    logger_run_all_lg.info(f"Options: save={not args.no_save}, plots={args.enable_plots}, individual={not args.skip_individual}, group={not args.skip_group}")
    
    run_all_groups_lg_analysis(
        classifier_type=args.clf_type,
        n_jobs_processing=n_jobs_to_use,
        save_results_flag=not args.no_save,
        generate_plots_flag=args.enable_plots,
        groups_to_process=args.groups,
        run_individual_subjects=not args.skip_individual,
        run_group_analysis=not args.skip_group,
        enable_verbose_logging=args.verbose
    )
