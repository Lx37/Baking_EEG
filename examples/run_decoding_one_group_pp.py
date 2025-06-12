# Fichier : examples/run_decoding_one_group_pp.py
# python -m examples.run_decoding_one_group_pp --group_name controls

# --- Configuration robuste des imports avec gestion d'erreurs ---
from config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING, CHANCE_LEVEL_AUC_SCORE,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING
)
from config.config import ALL_SUBJECT_GROUPS
from utils.vizualization_utils_PP import (
    plot_group_mean_scores_barplot,
    plot_group_temporal_decoding_statistics,
    plot_group_tgm_statistics
)
from utils import stats_utils as bEEG_stats
from utils.utils import (
    configure_project_paths, setup_analysis_results_directory
)
import os
import sys
import logging
import time
from datetime import datetime
import argparse
from getpass import getuser
import numpy as np
import pandas as pd
import scipy.stats

# Ajouter le chemin parent au Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration précoce du logging pour capturer les erreurs d'import
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger_import = logging.getLogger(__name__)

# --- GESTION ROBUSTE DES IMPORTS CRITIQUES ---
execute_single_subject_decoding = None

try:
    # Import avec gestion d'erreur explicite
    logger_import.info(
        "Tentative d'import de execute_single_subject_decoding...")
    from examples.run_decoding_one_pp import execute_single_subject_decoding
    logger_import.info("✅ Import de execute_single_subject_decoding réussi")
except ImportError as e_import:
    logger_import.error(
        f"ERREUR D'IMPORT CRITIQUE: Impossible d'importer 'execute_single_subject_decoding' depuis 'examples/run_decoding_one_pp.py'.")
    logger_import.error(
        f"Vérifiez que le fichier existe et que le nom de la fonction est correct.")
    logger_import.error(f"Erreur originale: {e_import}")
    sys.exit(1)
except Exception as e_other:
    logger_import.error(f"ERREUR INATTENDUE lors de l'import: {e_other}")
    sys.exit(1)

# --- IMPORTS DES MODULES DU PROJET (standardisés) ---

# Tous les autres imports partent de la racine (utils, config...)

# --- Configuration du Logging ---
LOG_DIR_RUN_GROUP = './logs_run_group_analysis'
os.makedirs(LOG_DIR_RUN_GROUP, exist_ok=True)
LOG_FILENAME_RUN_GROUP = os.path.join(
    LOG_DIR_RUN_GROUP,
    datetime.now().strftime('log_run_group_analysis_%Y-%m-%d_%H%M%S.log')
)

for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_GROUP),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_group = logging.getLogger(__name__)
logging.getLogger("examples.run_decoding_one_pp").setLevel(logging.INFO)
logging.getLogger("utils.stats_utils").setLevel(logging.INFO)
logging.getLogger("utils.vizualization_utils_PP").setLevel(logging.INFO)

# La suite du fichier est identique à la version précédente et devrait être correcte


def execute_group_intra_subject_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Single_Protocol_Group_Analysis",
    save_results_flag=SAVE_ANALYSIS_RESULTS,
    enable_verbose_logging=False,
    generate_plots_flag=GENERATE_PLOTS,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=N_JOBS_PROCESSING,
    compute_group_level_stats_flag=True,
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    compute_intra_subject_stats_for_group_runs_flag=True,
    n_perms_for_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    n_jobs_for_group_cluster_stats=N_JOBS_PROCESSING,
    use_grid_search_for_group=USE_GRID_SEARCH_OPTIMIZATION,
    use_csp_for_temporal_group=USE_CSP_FOR_TEMPORAL_PIPELINES,
    use_anova_fs_for_temporal_group=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config_for_group=PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
    cv_folds_for_gs_group=CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    fixed_params_for_group=FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
):
    """Executes intra-subject decoding for all subjects in a group and aggregates results."""
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group.error(
            "subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group.error("group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject,
                                             str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)
    actual_n_jobs_group_stats = -1 if isinstance(n_jobs_for_group_cluster_stats,
                                                 str) and n_jobs_for_group_cluster_stats.lower() == "auto" else int(n_jobs_for_group_cluster_stats)

    logger_run_group.info(
        "Starting intra-subject decoding analysis for GROUP: %s. GS: %s, CSP: %s, ANOVA FS: %s. n_jobs_subj: %s, n_jobs_grp_stats: %s.",
        group_identifier, use_grid_search_for_group, use_csp_for_temporal_group,
        use_anova_fs_for_temporal_group, actual_n_jobs_subject, actual_n_jobs_group_stats
    )

    group_results_collection = {
        "subject_global_auc_scores": {}, "subject_global_metrics_maps": {},
        "subject_temporal_scores_1d_mean_list": [],
        "subject_epochs_time_points_list": [], "subject_tgm_scores_mean_list": [],
        "subject_mean_of_specific_scores_list": [], "processed_subject_ids": []
    }

    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_group.info("\n--- Group '%s': Processing Subject %d/%d: %s ---",
                              group_identifier, i, len(subject_ids_in_group), subject_id_current)

        # C'est ici que l'appel est fait. Grâce à l'import correct, Python sait
        # maintenant ce qu'est 'execute_single_subject_decoding'.
        subject_output_dict = execute_single_subject_decoding(
            subject_identifier=subject_id_current,
            group_affiliation=group_identifier,
            decoding_protocol_identifier=f"{decoding_protocol_identifier}_Subj_{subject_id_current}",
            save_results_flag=save_results_flag,
            enable_verbose_logging=enable_verbose_logging,
            generate_plots_flag=generate_plots_flag,
            base_input_data_path=base_input_data_path,
            base_output_results_path=base_output_results_path,
            n_jobs_for_processing=actual_n_jobs_subject,
            classifier_type=classifier_type_for_group_runs,
            use_grid_search_for_subject=use_grid_search_for_group,
            use_csp_for_temporal_subject=use_csp_for_temporal_group,
            use_anova_fs_for_temporal_subject=use_anova_fs_for_temporal_group,
            param_grid_config_for_subject=param_grid_config_for_group,
            cv_folds_for_gs_subject=cv_folds_for_gs_group,
            fixed_params_for_subject=fixed_params_for_group,
            compute_intra_subject_stats_flag=compute_intra_subject_stats_for_group_runs_flag,
            n_perms_for_intra_subject_clusters=n_perms_intra_subject_folds_for_group_runs,
            compute_tgm_flag=compute_tgm_for_group_subjects_flag,
            cluster_threshold_config_intra_fold=cluster_threshold_config_intra_fold_group,
            loading_conditions_config=loading_conditions_config
        )

        s_auc = subject_output_dict.get("pp_ap_main_mean_auc_global", np.nan)
        s_metrics = subject_output_dict.get("pp_ap_main_global_metrics", {})
        s_scores_t_1d_mean = subject_output_dict.get(
            "pp_ap_main_scores_1d_mean")
        s_times_t = subject_output_dict.get("epochs_time_points")
        s_scores_tgm_mean = subject_output_dict.get("pp_ap_main_tgm_mean")
        s_mean_specific = subject_output_dict.get(
            "pp_ap_mean_of_specific_scores_1d")

        group_results_collection["subject_global_auc_scores"][subject_id_current] = s_auc
        group_results_collection["subject_global_metrics_maps"][subject_id_current] = s_metrics

        if pd.notna(s_auc) and s_scores_t_1d_mean is not None and s_times_t is not None and \
           s_scores_t_1d_mean.size > 0 and s_times_t.size > 0:
            group_results_collection["subject_temporal_scores_1d_mean_list"].append(
                s_scores_t_1d_mean)
            group_results_collection["subject_epochs_time_points_list"].append(
                s_times_t)
            group_results_collection["processed_subject_ids"].append(
                subject_id_current)

            if compute_tgm_for_group_subjects_flag and s_scores_tgm_mean is not None and s_scores_tgm_mean.ndim == 2:
                group_results_collection["subject_tgm_scores_mean_list"].append(
                    s_scores_tgm_mean)

            if s_mean_specific is not None and s_mean_specific.ndim == 1:
                group_results_collection["subject_mean_of_specific_scores_list"].append(
                    s_mean_specific)
        else:
            logger_run_group.warning(
                "Skipping subject %s from group '%s' aggregation (errors or no valid main scores).", subject_id_current, group_identifier)

    # === AGRÉGATION ET ANALYSES STATISTIQUES DE GROUPE ===

    n_processed_subjects = len(
        group_results_collection["processed_subject_ids"])
    logger_run_group.info("Successfully processed %d/%d subjects for group '%s'",
                          n_processed_subjects, len(subject_ids_in_group), group_identifier)

    if n_processed_subjects == 0:
        logger_run_group.error(
            "No subjects were successfully processed for group '%s'. Aborting group analysis.", group_identifier)
        return {}

    # --- 1. ANALYSES DES SCORES GLOBAUX AUC ---
    valid_global_auc_scores = [score for score in group_results_collection["subject_global_auc_scores"].values()
                               if pd.notna(score)]

    if len(valid_global_auc_scores) > 0:
        group_mean_auc = np.mean(valid_global_auc_scores)
        group_std_auc = np.std(valid_global_auc_scores)
        group_sem_auc = scipy.stats.sem(valid_global_auc_scores)

        logger_run_group.info(
            "Group '%s' Global AUC Statistics:", group_identifier)
        logger_run_group.info("  Mean AUC: %.3f ± %.3f (SEM: %.3f)",
                              group_mean_auc, group_std_auc, group_sem_auc)
        logger_run_group.info(
            "  Range: [%.3f - %.3f]", np.min(valid_global_auc_scores), np.max(valid_global_auc_scores))

        # Test statistique contre le niveau de chance
        if len(valid_global_auc_scores) >= 3:
            t_stat, p_val = scipy.stats.ttest_1samp(
                valid_global_auc_scores, CHANCE_LEVEL_AUC_SCORE)
            logger_run_group.info("  One-sample t-test vs chance (%.2f): t=%.3f, p=%.6f",
                                  CHANCE_LEVEL_AUC_SCORE, t_stat, p_val)
    else:
        logger_run_group.warning(
            "No valid global AUC scores for group '%s'", group_identifier)
        group_mean_auc = group_std_auc = group_sem_auc = np.nan

    # --- 2. ANALYSES TEMPORELLES DE GROUPE ---
    group_temporal_results = {}

    if len(group_results_collection["subject_temporal_scores_1d_mean_list"]) >= 2:
        logger_run_group.info("Computing group-level temporal decoding statistics for %d subjects...",
                              len(group_results_collection["subject_temporal_scores_1d_mean_list"]))

        # Vérifier la cohérence des dimensions temporelles
        time_points_ref = group_results_collection["subject_epochs_time_points_list"][0]
        temporal_scores_stack = []

        for i, (scores, times) in enumerate(zip(group_results_collection["subject_temporal_scores_1d_mean_list"],
                                                group_results_collection["subject_epochs_time_points_list"])):
            if scores.shape == time_points_ref.shape and np.allclose(times, time_points_ref, atol=1e-6):
                temporal_scores_stack.append(scores)
            else:
                logger_run_group.warning("Subject %s has inconsistent temporal dimensions. Excluding from group temporal analysis.",
                                         group_results_collection["processed_subject_ids"][i])

        if len(temporal_scores_stack) >= 2:
            # Shape: (n_subjects, n_times)
            temporal_scores_array = np.array(temporal_scores_stack)

            # Calculs des statistiques de groupe
            group_temporal_mean = np.nanmean(temporal_scores_array, axis=0)
            group_temporal_std = np.nanstd(temporal_scores_array, axis=0)
            group_temporal_sem = scipy.stats.sem(
                temporal_scores_array, axis=0, nan_policy='omit')

            group_temporal_results.update({
                "group_temporal_mean_scores": group_temporal_mean,
                "group_temporal_std_scores": group_temporal_std,
                "group_temporal_sem_scores": group_temporal_sem,
                "time_points": time_points_ref.copy(),
                "n_subjects_temporal": len(temporal_scores_stack)
            })

            # Analyses statistiques temporelles si demandées
            if compute_group_level_stats_flag and len(temporal_scores_stack) >= 3:
                logger_run_group.info(
                    "Performing group-level statistical tests on temporal scores...")

                # Test pointwise FDR
                try:
                    _, fdr_mask_group, fdr_pvalues_group = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        temporal_scores_array, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"
                    )
                    group_temporal_results["group_temporal_fdr_mask"] = fdr_mask_group
                    group_temporal_results["group_temporal_fdr_pvalues"] = fdr_pvalues_group
                    logger_run_group.info("FDR correction completed. Significant time points: %d/%d",
                                          np.sum(fdr_mask_group), len(fdr_mask_group))
                except Exception as e_fdr:
                    logger_run_group.error(
                        "Failed to compute group FDR correction: %s", e_fdr)

                # Test de permutation par clusters
                try:
                    cluster_threshold_config_group = {
                        "threshold_method": group_cluster_test_threshold_method,
                        "threshold_value": group_cluster_test_t_thresh_value
                    }

                    _, cluster_objects_group, cluster_pvalues_group, _ = bEEG_stats.perform_cluster_permutation_test(
                        temporal_scores_array, CHANCE_LEVEL_AUC_SCORE, n_perms_for_group_cluster_test,
                        cluster_threshold_config_group, "greater", actual_n_jobs_group_stats
                    )

                    if cluster_objects_group and cluster_pvalues_group is not None:
                        significant_clusters = []
                        combined_cluster_mask = np.zeros_like(
                            group_temporal_mean, dtype=bool)

                        for i_cluster, (cluster_mask, p_val) in enumerate(zip(cluster_objects_group, cluster_pvalues_group)):
                            if p_val < 0.05:
                                significant_clusters.append({
                                    "cluster_id": i_cluster,
                                    "cluster_mask": cluster_mask,
                                    "p_value": p_val,
                                    "cluster_size": np.sum(cluster_mask)
                                })
                                combined_cluster_mask = np.logical_or(
                                    combined_cluster_mask, cluster_mask)

                        group_temporal_results.update({
                            "group_temporal_cluster_mask": combined_cluster_mask,
                            "group_temporal_significant_clusters": significant_clusters,
                            "group_temporal_all_cluster_pvalues": cluster_pvalues_group
                        })

                        logger_run_group.info("Cluster permutation test completed. Significant clusters: %d",
                                              len(significant_clusters))

                except Exception as e_cluster:
                    logger_run_group.error(
                        "Failed to compute group cluster permutation test: %s", e_cluster)

            logger_run_group.info("Group temporal analysis completed. Peak group AUC: %.3f at time %.3fs",
                                  np.nanmax(group_temporal_mean),
                                  time_points_ref[np.nanargmax(group_temporal_mean)])

    # --- 3. ANALYSES TGM (TEMPORAL GENERALIZATION MATRIX) DE GROUPE ---
    group_tgm_results = {}

    if (compute_tgm_for_group_subjects_flag and
            len(group_results_collection["subject_tgm_scores_mean_list"]) >= 2):

        logger_run_group.info("Computing group-level TGM statistics for %d subjects...",
                              len(group_results_collection["subject_tgm_scores_mean_list"]))

        # Vérifier la cohérence des dimensions TGM
        tgm_shape_ref = group_results_collection["subject_tgm_scores_mean_list"][0].shape
        tgm_scores_stack = []

        for i, tgm_scores in enumerate(group_results_collection["subject_tgm_scores_mean_list"]):
            if tgm_scores.shape == tgm_shape_ref:
                tgm_scores_stack.append(tgm_scores)
            else:
                logger_run_group.warning("Subject %s has inconsistent TGM dimensions. Excluding from group TGM analysis.",
                                         group_results_collection["processed_subject_ids"][i])

        if len(tgm_scores_stack) >= 2:
            # Shape: (n_subjects, n_times, n_times)
            tgm_scores_array = np.array(tgm_scores_stack)

            group_tgm_mean = np.nanmean(tgm_scores_array, axis=0)
            group_tgm_std = np.nanstd(tgm_scores_array, axis=0)

            group_tgm_results.update({
                "group_tgm_mean_scores": group_tgm_mean,
                "group_tgm_std_scores": group_tgm_std,
                "n_subjects_tgm": len(tgm_scores_stack)
            })

            logger_run_group.info("Group TGM analysis completed. Peak group TGM AUC: %.3f",
                                  np.nanmax(group_tgm_mean))

    # --- 4. ANALYSES DES SCORES SPÉCIFIQUES DE GROUPE ---
    group_specific_results = {}

    if len(group_results_collection["subject_mean_of_specific_scores_list"]) >= 2:
        logger_run_group.info("Computing group-level specific task statistics for %d subjects...",
                              len(group_results_collection["subject_mean_of_specific_scores_list"]))

        # Vérifier la cohérence des dimensions
        specific_shape_ref = group_results_collection["subject_mean_of_specific_scores_list"][0].shape
        specific_scores_stack = []

        for i, specific_scores in enumerate(group_results_collection["subject_mean_of_specific_scores_list"]):
            if specific_scores.shape == specific_shape_ref:
                specific_scores_stack.append(specific_scores)
            else:
                logger_run_group.warning("Subject %s has inconsistent specific scores dimensions.",
                                         group_results_collection["processed_subject_ids"][i])

        if len(specific_scores_stack) >= 2:
            specific_scores_array = np.array(specific_scores_stack)

            group_specific_mean = np.nanmean(specific_scores_array, axis=0)
            group_specific_std = np.nanstd(specific_scores_array, axis=0)
            group_specific_sem = scipy.stats.sem(
                specific_scores_array, axis=0, nan_policy='omit')

            group_specific_results.update({
                "group_specific_mean_scores": group_specific_mean,
                "group_specific_std_scores": group_specific_std,
                "group_specific_sem_scores": group_specific_sem,
                "n_subjects_specific": len(specific_scores_stack)
            })

            logger_run_group.info("Group specific tasks analysis completed. Peak group specific AUC: %.3f",
                                  np.nanmax(group_specific_mean))

    # --- 5. SAUVEGARDE DES RÉSULTATS DE GROUPE ---
    if save_results_flag:
        try:
            group_results_dir = setup_analysis_results_directory(
                base_output_results_path, "group_analysis_results", group_identifier,
                f"group_{group_identifier}_{classifier_type_for_group_runs}"
            )

            # Sauvegarder les résultats complets
            group_results_complete = {
                "group_identifier": group_identifier,
                "classifier_type": classifier_type_for_group_runs,
                "n_subjects_total": len(subject_ids_in_group),
                "n_subjects_processed": n_processed_subjects,
                "processed_subject_ids": group_results_collection["processed_subject_ids"],
                "subject_global_auc_scores": group_results_collection["subject_global_auc_scores"],
                "subject_global_metrics_maps": group_results_collection["subject_global_metrics_maps"],
                "group_global_auc_mean": group_mean_auc,
                "group_global_auc_std": group_std_auc,
                "group_global_auc_sem": group_sem_auc,
                **group_temporal_results,
                **group_tgm_results,
                **group_specific_results
            }

            # Sauvegarder en format NPZ
            results_file_path = os.path.join(
                group_results_dir, f"group_{group_identifier}_results_complete.npz")
            np.savez_compressed(results_file_path, **group_results_complete)

            # Sauvegarder un résumé CSV
            summary_data = {
                "group": group_identifier,
                "classifier": classifier_type_for_group_runs,
                "n_subjects_total": len(subject_ids_in_group),
                "n_subjects_processed": n_processed_subjects,
                "group_mean_auc": group_mean_auc,
                "group_std_auc": group_std_auc,
                "group_sem_auc": group_sem_auc
            }

            if len(valid_global_auc_scores) >= 3:
                t_stat, p_val = scipy.stats.ttest_1samp(
                    valid_global_auc_scores, CHANCE_LEVEL_AUC_SCORE)
                summary_data.update({
                    "t_stat_vs_chance": t_stat,
                    "p_val_vs_chance": p_val
                })

            summary_df = pd.DataFrame([summary_data])
            summary_csv_path = os.path.join(
                group_results_dir, f"group_{group_identifier}_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)

            logger_run_group.info(
                "Group results saved to: %s", group_results_dir)

        except Exception as e_save:
            logger_run_group.error(
                "Failed to save group results: %s", e_save, exc_info=True)

    # --- 6. GÉNÉRATION DES VISUALISATIONS DE GROUPE ---
    if generate_plots_flag and n_processed_subjects >= 2:
        try:
            if save_results_flag and 'group_results_dir' in locals():
                plots_dir = group_results_dir
            else:
                plots_dir = os.path.join(
                    os.getcwd(), f"group_{group_identifier}_plots")
                os.makedirs(plots_dir, exist_ok=True)

            logger_run_group.info("Generating group-level visualizations...")

            # 1. Graphique en barres des scores moyens
            if len(valid_global_auc_scores) > 0:
                try:
                    plot_group_mean_scores_barplot(
                        group_results_collection["subject_global_auc_scores"],
                        group_identifier,
                        CHANCE_LEVEL_AUC_SCORE,
                        plots_dir
                    )
                    logger_run_group.info(
                        "Generated group mean scores barplot")
                except Exception as e_bar:
                    logger_run_group.error(
                        "Failed to generate barplot: %s", e_bar)

            # 2. Statistiques de décodage temporel
            if group_temporal_results:
                try:
                    plot_group_temporal_decoding_statistics(
                        group_temporal_results,
                        group_identifier,
                        CHANCE_LEVEL_AUC_SCORE,
                        plots_dir
                    )
                    logger_run_group.info(
                        "Generated group temporal decoding statistics plot")
                except Exception as e_temporal:
                    logger_run_group.error(
                        "Failed to generate temporal plot: %s", e_temporal)

            # 3. Statistiques TGM
            if group_tgm_results:
                try:
                    plot_group_tgm_statistics(
                        group_tgm_results,
                        group_identifier,
                        CHANCE_LEVEL_AUC_SCORE,
                        plots_dir
                    )
                    logger_run_group.info(
                        "Generated group TGM statistics plot")
                except Exception as e_tgm:
                    logger_run_group.error(
                        "Failed to generate TGM plot: %s", e_tgm)

            logger_run_group.info(
                "Group visualizations saved to: %s", plots_dir)

        except Exception as e_plots:
            logger_run_group.error(
                "Failed to generate group plots: %s", e_plots, exc_info=True)

    logger_run_group.info("Finished aggregation logic. Total group analysis time: %.1f min",
                          (time.time() - total_group_analysis_start_time) / 60)
    return group_results_collection["subject_global_auc_scores"]


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="EEG Group Intra-Subject Decoding Analysis Script",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli_parser.add_argument("--group_name", type=str, required=True,
                            choices=ALL_SUBJECT_GROUPS.keys(), help="Name of the group to process.")
    cli_parser.add_argument("--clf_type_override", type=str, default=None, choices=["svc", "logreg", "rf"],
                            help="Override default classifier type from config.")
    cli_parser.add_argument("--n_jobs_override", type=str, default=None,
                            help="Override n_jobs for subject and group stats processing (e.g., '4' or 'auto').")

    command_line_args = cli_parser.parse_args()

    n_jobs_arg_str_main = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use_main = - \
            1 if n_jobs_arg_str_main.lower() == "auto" else int(n_jobs_arg_str_main)
    except (ValueError, AttributeError):
        logger_run_group.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str_main}'). Using default from config: {N_JOBS_PROCESSING}.")
        n_jobs_to_use_main = - \
            1 if str(N_JOBS_PROCESSING).lower(
            ) == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use_main = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login_main = getuser()
    main_input_path_main, main_output_path_main = configure_project_paths(
        user_login_main)

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT STARTED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger_run_group.info("User: %s, Group to process: %s",
                          user_login_main, command_line_args.group_name)
    logger_run_group.info("  Classifier: %s, n_jobs (main ops): %s",
                          classifier_type_to_use_main, n_jobs_to_use_main)

    execute_group_intra_subject_decoding_analysis(
        subject_ids_in_group=ALL_SUBJECT_GROUPS[command_line_args.group_name],
        group_identifier=command_line_args.group_name,
        base_input_data_path=main_input_path_main,
        base_output_results_path=main_output_path_main,
        n_jobs_for_each_subject=n_jobs_to_use_main,
        n_jobs_for_group_cluster_stats=n_jobs_to_use_main,
        classifier_type_for_group_runs=classifier_type_to_use_main,
    )

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT FINISHED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
