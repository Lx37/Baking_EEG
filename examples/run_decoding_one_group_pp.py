# examples/run_group_analysis.py (ou gardez run_decoding_one_group.py si vous préférez)

import os
import sys
import logging
import time
from datetime import datetime  # Ajouté pour le logging du __main__
import argparse  # Ajouté pour rendre le script exécutable avec des arguments
from getpass import getuser
import numpy as np
import pandas as pd
import scipy.stats  # Pour scipy.stats.t.ppf et scipy.stats.sem

# --- Configuration du chemin pour les imports ---
SCRIPT_DIR_EXAMPLE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EXAMPLE = os.path.abspath(os.path.join(SCRIPT_DIR_EXAMPLE, ".."))
if PROJECT_ROOT_EXAMPLE not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_EXAMPLE)
# --- Fin Configuration du chemin ---

# Imports depuis votre module Baking_EEG
from Baking_EEG.config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    # Correction du nom de la constante
    T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING, CHANCE_LEVEL_AUC_SCORE,
    # COMPUTE_INTRA_SUBJECT_STATISTICS, # Est un argument de fonction
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_TEMPORAL_GENERALIZATION_MATRICES, CONFIG_LOAD_SINGLE_PROTOCOL,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING,
    # AP_FAMILIES_FOR_SPECIFIC_COMPARISON # Utilisé dans execute_single_subject_decoding
)
from Baking_EEG.config.config import ALL_SUBJECT_GROUPS
from Baking_EEG.utils.vizualization_utils import (  # create_subject_decoding_dashboard_plots, # Appelée dans execute_single_subject_decoding
    plot_group_mean_scores_barplot,
    plot_group_temporal_decoding_statistics,
    plot_group_tgm_statistics)
from Baking_EEG.utils import stats_utils as bEEG_stats
from Baking_EEG.utils.utils import configure_project_paths

# La fonction execute_single_subject_decoding est maintenant attendue dans son propre fichier
# sous examples (par exemple, run_single_subject_analysis.py)
# Assurez-vous que le nom du module est correct.
# Si run_decoding_one.py devient run_single_subject_analysis.py:
try:
    from Baking_EEG.examples.run_decoding_one import execute_single_subject_decoding
except ImportError:
    from Baking_EEG.examples.run_decoding_one import execute_single_subject_decoding


# from Baking_EEG.utils.loading_PP_utils import load_epochs_data_for_decoding # Cette fonction est appelée DANS execute_single_subject_decoding

# Imports depuis les fichiers de configuration

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
logging.getLogger("Baking_EEG.examples.run_single_subject_analysis").setLevel(
    logging.INFO)  # Ou le nom du module où est execute_single_subject_decoding
logging.getLogger("Baking_EEG.utils.stats_utils").setLevel(logging.INFO)
logging.getLogger("Baking_EEG.viz.visualization_utils").setLevel(logging.INFO)


def execute_group_intra_subject_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Single_Protocol_Group_Analysis",
    # Utilisation des constantes pour les valeurs par défaut
    save_results_flag=SAVE_ANALYSIS_RESULTS,
    # False pour moins de logs par défaut dans les analyses de groupe
    enable_verbose_logging=False,
    generate_plots_flag=GENERATE_PLOTS,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=N_JOBS_PROCESSING,  # Peut être surchargé
    # Par défaut à True pour une analyse de groupe
    compute_group_level_stats_flag=True,
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    # Par défaut à True pour les analyses sujet par sujet
    compute_intra_subject_stats_for_group_runs_flag=True,
    n_perms_for_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    cluster_threshold_config_intra_fold_group=INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    n_jobs_for_group_cluster_stats=N_JOBS_PROCESSING,  # Peut être surchargé
    use_grid_search_for_group=USE_GRID_SEARCH_OPTIMIZATION,
    use_csp_for_temporal_group=USE_CSP_FOR_TEMPORAL_PIPELINES,
    use_anova_fs_for_temporal_group=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config_for_group=PARAM_GRID_CONFIG_EXTENDED if USE_GRID_SEARCH_OPTIMIZATION else None,
    cv_folds_for_gs_group=CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    fixed_params_for_group=FIXED_CLASSIFIER_PARAMS_CONFIG if not USE_GRID_SEARCH_OPTIMIZATION else None,
    loading_conditions_config=CONFIG_LOAD_SINGLE_PROTOCOL
):
    """Executes intra-subject decoding for all subjects in a group and aggregates results."""
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group.error(  # MODIFIÉ logger
            "subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group.error(  # MODIFIÉ logger
            "group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    # Convertir n_jobs_for_each_subject et n_jobs_for_group_cluster_stats
    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject,
                                             str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)
    actual_n_jobs_group_stats = -1 if isinstance(n_jobs_for_group_cluster_stats,
                                                 str) and n_jobs_for_group_cluster_stats.lower() == "auto" else int(n_jobs_for_group_cluster_stats)

    logger_run_group.info(  # MODIFIÉ logger
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
        logger_run_group.info("\n--- Group '%s': Processing Subject %d/%d: %s ---",  # MODIFIÉ logger
                              group_identifier, i, len(subject_ids_in_group), subject_id_current)

        subject_output_dict = execute_single_subject_decoding(
            subject_identifier=subject_id_current,
            group_affiliation=group_identifier,
            decoding_protocol_identifier=f"{decoding_protocol_identifier}_Subj_{subject_id_current}",
            # Passer les flags de la fonction de groupe
            save_results_flag=save_results_flag,
            # Passer le flag de la fonction de groupe
            enable_verbose_logging=enable_verbose_logging,
            # Passer le flag de la fonction de groupe
            generate_plots_flag=generate_plots_flag,
            base_input_data_path=base_input_data_path,
            base_output_results_path=base_output_results_path,
            n_jobs_for_processing=actual_n_jobs_subject,  # Utiliser la valeur convertie
            classifier_type=classifier_type_for_group_runs,
            # Utiliser les params de groupe pour le sujet
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

            if compute_tgm_for_group_subjects_flag:
                if s_scores_tgm_mean is not None and not (isinstance(s_scores_tgm_mean, float) and np.isnan(s_scores_tgm_mean)) and s_scores_tgm_mean.ndim == 2:
                    group_results_collection["subject_tgm_scores_mean_list"].append(
                        s_scores_tgm_mean)
                else:
                    nan_tgm = np.full_like(s_scores_t_1d_mean[:, np.newaxis] * s_scores_t_1d_mean[np.newaxis, :],
                                           # Modifié pour éviter erreur si s_scores_t_1d_mean est vide
                                           np.nan) if s_scores_t_1d_mean is not None and s_scores_t_1d_mean.ndim == 1 and s_scores_t_1d_mean.size > 0 else np.array([[]])
                    group_results_collection["subject_tgm_scores_mean_list"].append(
                        nan_tgm)

            if s_mean_specific is not None and not (isinstance(s_mean_specific, float) and np.isnan(s_mean_specific)) and s_mean_specific.ndim == 1:
                group_results_collection["subject_mean_of_specific_scores_list"].append(
                    s_mean_specific)
            else:
                nan_specific = np.full_like(
                    # Modifié
                    s_scores_t_1d_mean, np.nan) if s_scores_t_1d_mean is not None and s_scores_t_1d_mean.size > 0 else np.array([])
                group_results_collection["subject_mean_of_specific_scores_list"].append(
                    nan_specific)
        else:
            logger_run_group.warning(  # MODIFIÉ logger
                "Skipping subject %s from group '%s' aggregation (errors or no valid main scores).", subject_id_current, group_identifier)

    group_summary_dir = None
    if save_results_flag or generate_plots_flag:
        dir_suffix = (
            f"{classifier_type_for_group_runs}_GS{use_grid_search_for_group}_CSP{use_csp_for_temporal_group}_ANOVA{use_anova_fs_for_temporal_group}")
        group_summary_dir = setup_analysis_results_directory(
            base_output_results_path, "group_summary_intra_subject", group_identifier, dir_suffix
        )

    valid_global_scores = np.array(
        [s for s in group_results_collection["subject_global_auc_scores"].values() if pd.notna(s)])
    if len(valid_global_scores) > 0:
        mean_auc = np.mean(valid_global_scores)
        std_auc = np.std(valid_global_scores)
        logger_run_group.info("Group %s - Overall Global Mean AUC (Main Dec.): %.3f +/- %.3f (N=%d subjects)",  # MODIFIÉ logger
                              group_identifier, mean_auc, std_auc, len(valid_global_scores))
        if compute_group_level_stats_flag and len(valid_global_scores) >= 2:
            stat_g, p_val_g = bEEG_stats.compare_global_scores_to_chance(
                valid_global_scores, CHANCE_LEVEL_AUC_SCORE, "ttest", "greater")  # MODIFIÉ constante
            logger_run_group.info(  # MODIFIÉ logger
                "  Global AUC (Main Dec.) vs Chance: t=%.3f, p=%.4f", stat_g, p_val_g)
            if save_results_flag and group_summary_dir:
                with open(os.path.join(group_summary_dir, "stats_global_auc.txt"), "w") as f_stat:
                    f_stat.write(
                        # MODIFIÉ constante
                        f"Intra-Subject Global AUC (Main Dec.) vs Chance ({CHANCE_LEVEL_AUC_SCORE})\nGroup: {group_identifier}, N: {len(valid_global_scores)}\nMean AUC: {mean_auc:.4f}, Std: {std_auc:.4f}\nT-stat: {stat_g:.4f}, P-val: {p_val_g:.4f}\n")
        if generate_plots_flag and group_summary_dir:
            plot_group_mean_scores_barplot(group_results_collection["subject_global_auc_scores"],
                                           # MODIFIÉ constante
                                           f"{group_identifier} - Subject Global AUCs (Main)", group_summary_dir, "Global ROC AUC", CHANCE_LEVEL_AUC_SCORE)
    else:
        logger_run_group.warning(  # MODIFIÉ logger
            "No valid global scores for group %s.", group_identifier)

    ref_times_1d = None
    if len(group_results_collection["subject_temporal_scores_1d_mean_list"]) >= 2 and compute_group_level_stats_flag:
        ref_times_idx = next((j for j, t_arr in enumerate(
            group_results_collection["subject_epochs_time_points_list"]) if t_arr is not None and t_arr.size > 0), -1)
        if ref_times_idx != -1:
            ref_times_1d = group_results_collection["subject_epochs_time_points_list"][ref_times_idx]
            valid_1d_scores_main = [s for i, s in enumerate(group_results_collection["subject_temporal_scores_1d_mean_list"]) if i < len(group_results_collection["subject_epochs_time_points_list"]) and (
                t_cur := group_results_collection["subject_epochs_time_points_list"][i]) is not None and t_cur.size == ref_times_1d.size and np.allclose(t_cur, ref_times_1d) and s is not None and not np.all(np.isnan(s))]
            if len(valid_1d_scores_main) >= 2:
                stacked_1d = np.array(valid_1d_scores_main)
                logger_run_group.info(  # MODIFIÉ logger
                    "Group %s (N=%d for Main 1D stats): Running group stats...", group_identifier, stacked_1d.shape[0])
                t_obs_fdr, fdr_mask, p_fdr = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_1d, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater")  # MODIFIÉ constante
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_temp_1D_FDR_Main.npz"),
                                        t_obs=t_obs_fdr, sig_mask=fdr_mask, p_vals=p_fdr, times=ref_times_1d)

                t_thresh_clu = (group_cluster_test_t_thresh_value if group_cluster_test_threshold_method ==
                                "stat" else INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method == "tfce" else None)
                # Ajout condition df > 0
                if group_cluster_test_threshold_method == "stat" and t_thresh_clu is None and stacked_1d.shape[0] > 1:
                    t_thresh_clu = scipy.stats.t.ppf(
                        1.0 - 0.05 / 2, df=stacked_1d.shape[0] - 1)
                elif group_cluster_test_threshold_method == "stat" and t_thresh_clu is None and stacked_1d.shape[0] <= 1:
                    logger_run_group.warning(
                        f"Cannot calculate t_thresh_clu for stat method with N={stacked_1d.shape[0]}. Skipping cluster test or using TFCE if applicable.")
                    # Fallback to TFCE if t_thresh_clu is for stat and N is too small
                    t_thresh_clu = INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method != "tfce" else t_thresh_clu

                t_obs_clu, clu, p_clu, _ = bEEG_stats.perform_cluster_permutation_test(
                    # MODIFIÉ constante, n_jobs
                    stacked_1d, CHANCE_LEVEL_AUC_SCORE, n_perms_for_group_cluster_test, t_thresh_clu, "greater", actual_n_jobs_group_stats)
                clu_map = (bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d.shape, clu, p_clu) if clu and p_clu is not None else None)
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_temp_1D_CLUSTER_Main.npz"),
                                        t_obs=t_obs_clu, clusters=clu, p_vals=p_clu, p_map=clu_map, times=ref_times_1d)
                if generate_plots_flag and group_summary_dir:
                    sem_1d = (scipy.stats.sem(
                        stacked_1d, axis=0, nan_policy="omit") if stacked_1d.shape[0] > 1 else None)
                    plot_group_temporal_decoding_statistics(ref_times_1d, np.mean(
                        # MODIFIÉ constante
                        stacked_1d, axis=0), f"{group_identifier} (Main 1D Temporal)", group_summary_dir, sem_1d, clu_map, fdr_mask, CHANCE_LEVEL_AUC_SCORE)
            else:
                logger_run_group.warning("Not enough valid data (%d) for Main 1D group stats for %s.", len(  # MODIFIÉ logger
                    valid_1d_scores_main), group_identifier)
        else:
            logger_run_group.warning(  # MODIFIÉ logger
                "No reference times found for Main 1D group stats for %s.", group_identifier)

    valid_tgms = [tgm for tgm in group_results_collection["subject_tgm_scores_mean_list"]
                  if tgm is not None and not (isinstance(tgm, float) and np.isnan(tgm)) and
                  tgm.ndim == 2 and tgm.size > 0]  # Ajout de tgm.size > 0
    if len(valid_tgms) >= 2 and compute_group_level_stats_flag and \
       compute_tgm_for_group_subjects_flag:
        ref_times_idx_tgm = next(
            (j for j, t_arr in enumerate(group_results_collection["subject_epochs_time_points_list"])
             if t_arr is not None and t_arr.size > 0), -1
        )
        if ref_times_idx_tgm != -1:
            ref_times_tgm = group_results_collection["subject_epochs_time_points_list"][ref_times_idx_tgm]
            if ref_times_tgm.size > 0:  # S'assurer que ref_times_tgm n'est pas vide
                expected_shape = (ref_times_tgm.size, ref_times_tgm.size)
                valid_tgms_stack_list = []
                for i_subj_with_tgm in range(len(group_results_collection["subject_tgm_scores_mean_list"])):
                    tgm_item = group_results_collection["subject_tgm_scores_mean_list"][i_subj_with_tgm]
                    # Utiliser l'index directement car subject_epochs_time_points_list est aligné
                    # avec subject_tgm_scores_mean_list (même s'il y a des NaNs ou des tableaux vides)
                    if i_subj_with_tgm < len(group_results_collection["subject_epochs_time_points_list"]):
                        t_cur = group_results_collection["subject_epochs_time_points_list"][i_subj_with_tgm]
                        if tgm_item is not None and not (isinstance(tgm_item, float) and np.isnan(tgm_item)) and \
                           tgm_item.ndim == 2 and tgm_item.shape == expected_shape and \
                           t_cur is not None and t_cur.size == ref_times_tgm.size and np.allclose(t_cur, ref_times_tgm) and \
                           not np.all(np.isnan(tgm_item)):
                            valid_tgms_stack_list.append(tgm_item)

                if len(valid_tgms_stack_list) >= 2:
                    stacked_tgms = np.array(valid_tgms_stack_list)
                    logger_run_group.info(  # MODIFIÉ logger
                        "Group %s (N=%d for Main TGM stats): Running TGM FDR...",
                        group_identifier, stacked_tgms.shape[0]
                    )
                    n_s, n_t_train, n_t_test = stacked_tgms.shape  # Peut ne pas être carré
                    tgm_flat = stacked_tgms.reshape(n_s, n_t_train * n_t_test)
                    t_obs_tgm, fdr_mask_flat, p_fdr_flat = \
                        bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                            tgm_flat, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"  # MODIFIÉ constante
                        )
                    fdr_mask_2d = (fdr_mask_flat.reshape(n_t_train, n_t_test)
                                   if hasattr(fdr_mask_flat, 'reshape') and fdr_mask_flat is not None else None)
                    t_obs_map = (t_obs_tgm.reshape(n_t_train, n_t_test)
                                 if hasattr(t_obs_tgm, 'reshape') and t_obs_tgm is not None else t_obs_tgm)
                    p_fdr_map = (p_fdr_flat.reshape(n_t_train, n_t_test)
                                 if hasattr(p_fdr_flat, 'reshape') and p_fdr_flat is not None else None)

                    if save_results_flag and group_summary_dir:
                        np.savez_compressed(
                            os.path.join(group_summary_dir,
                                         "stats_TGM_FDR_Main.npz"),
                            t_obs_map=t_obs_map, sig_mask=fdr_mask_2d, p_vals_map=p_fdr_map,
                            times=ref_times_tgm, mean_tgm=np.mean(
                                stacked_tgms, axis=0)
                        )
                    logger_run_group.info(  # MODIFIÉ logger
                        "Group %s: Cluster permutation for TGM SKIPPED.", group_identifier)
                    if generate_plots_flag and group_summary_dir:
                        plot_group_tgm_statistics(
                            np.mean(stacked_tgms,
                                    axis=0), ref_times_tgm, None, None,
                            f"{group_identifier} (Main TGM - FDR)", group_summary_dir,
                            t_obs_map, fdr_mask_2d, CHANCE_LEVEL_AUC_SCORE  # MODIFIÉ constante
                        )
                else:
                    logger_run_group.warning(  # MODIFIÉ logger
                        "Not enough valid TGM data (%d after filtering) for Main TGM group stats for %s.",
                        len(valid_tgms_stack_list), group_identifier
                    )
            else:
                logger_run_group.warning(  # MODIFIÉ logger
                    "Reference TGM times array is empty for group %s.", group_identifier
                )

        else:
            logger_run_group.warning(  # MODIFIÉ logger
                "No reference times found for Main TGM group stats for %s.", group_identifier
            )
    elif compute_group_level_stats_flag:
        logger_run_group.info(  # MODIFIÉ logger
            "Skipping Main TGM group stats for %s (TGM disabled or not enough data).",
            group_identifier
        )

    valid_mean_specific_list = [ms for ms in group_results_collection["subject_mean_of_specific_scores_list"]
                                if ms is not None and not (isinstance(ms, float) and np.isnan(ms)) and
                                ms.ndim == 1 and ms.size > 0]

    if len(valid_mean_specific_list) >= 2 and compute_group_level_stats_flag and ref_times_1d is not None and ref_times_1d.size > 0:
        valid_ms_scores_list = []
        for i_subj_with_ms in range(len(group_results_collection["subject_mean_of_specific_scores_list"])):
            s_ms = group_results_collection["subject_mean_of_specific_scores_list"][i_subj_with_ms]
            if i_subj_with_ms < len(group_results_collection["subject_epochs_time_points_list"]):
                t_cur_ms = group_results_collection["subject_epochs_time_points_list"][i_subj_with_ms]
                if s_ms is not None and not (isinstance(s_ms, float) and np.isnan(s_ms)) and \
                        s_ms.ndim == 1 and s_ms.size == ref_times_1d.size and \
                        t_cur_ms is not None and t_cur_ms.size == ref_times_1d.size and np.allclose(t_cur_ms, ref_times_1d) and \
                        not np.all(np.isnan(s_ms)):
                    valid_ms_scores_list.append(s_ms)

        if len(valid_ms_scores_list) >= 2:
            stacked_ms = np.array(valid_ms_scores_list)
            logger_run_group.info(  # MODIFIÉ logger
                "Group %s (N=%d for Avg. Specific 1D stats): Running group stats...",
                group_identifier, stacked_ms.shape[0]
            )
            t_obs_fdr_ms, fdr_mask_ms, p_fdr_ms = \
                bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_ms, CHANCE_LEVEL_AUC_SCORE, alternative_hypothesis="greater"  # MODIFIÉ constante
                )
            if save_results_flag and group_summary_dir:
                np.savez_compressed(
                    os.path.join(group_summary_dir,
                                 "stats_temp_1D_FDR_MeanSpecific.npz"),
                    t_obs=t_obs_fdr_ms, sig_mask=fdr_mask_ms, p_vals=p_fdr_ms, times=ref_times_1d
                )

            t_thresh_clu_ms = (group_cluster_test_t_thresh_value
                               if group_cluster_test_threshold_method == "stat" else
                               INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG
                               if group_cluster_test_threshold_method == "tfce" else None)
            # Ajout condition df > 0
            if group_cluster_test_threshold_method == "stat" and t_thresh_clu_ms is None and stacked_ms.shape[0] > 1:
                t_thresh_clu_ms = scipy.stats.t.ppf(
                    1.0 - 0.05 / 2, df=stacked_ms.shape[0] - 1)
            elif group_cluster_test_threshold_method == "stat" and t_thresh_clu_ms is None and stacked_ms.shape[0] <= 1:
                logger_run_group.warning(
                    f"Cannot calculate t_thresh_clu_ms for stat method with N={stacked_ms.shape[0]}. Skipping cluster test or using TFCE if applicable.")
                t_thresh_clu_ms = INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method != "tfce" else t_thresh_clu_ms

            t_obs_clu_ms, clu_ms, p_clu_ms, _ = bEEG_stats.perform_cluster_permutation_test(
                stacked_ms, CHANCE_LEVEL_AUC_SCORE, n_perms_for_group_cluster_test,  # MODIFIÉ constante
                t_thresh_clu_ms, "greater", actual_n_jobs_group_stats  # MODIFIÉ n_jobs
            )
            clu_map_ms = (bEEG_stats.create_p_value_map_from_cluster_results(ref_times_1d.shape, clu_ms, p_clu_ms)
                          if clu_ms and p_clu_ms is not None else None)
            if save_results_flag and group_summary_dir:
                np.savez_compressed(
                    os.path.join(group_summary_dir,
                                 "stats_temp_1D_CLUSTER_MeanSpecific.npz"),
                    t_obs=t_obs_clu_ms, clusters=clu_ms, p_vals=p_clu_ms, p_map=clu_map_ms, times=ref_times_1d
                )
            if generate_plots_flag and group_summary_dir:
                sem_ms = (scipy.stats.sem(stacked_ms, axis=0, nan_policy="omit")
                          if stacked_ms.shape[0] > 1 else None)
                plot_group_temporal_decoding_statistics(
                    ref_times_1d, np.mean(stacked_ms, axis=0),
                    f"{group_identifier} (Avg. Specific Tasks - Group)", group_summary_dir,
                    sem_ms, clu_map_ms, fdr_mask_ms, CHANCE_LEVEL_AUC_SCORE  # MODIFIÉ constante
                )
        else:
            logger_run_group.warning(  # MODIFIÉ logger
                "Not enough valid data (%d after filtering) for Avg. Specific 1D group stats for %s.",
                len(valid_ms_scores_list), group_identifier
            )
    elif compute_group_level_stats_flag:
        logger_run_group.info(  # MODIFIÉ logger
            "Skipping Avg. Specific 1D group stats for %s (not enough data or no reference times).",
            group_identifier
        )

    if save_results_flag and group_summary_dir and \
       len(group_results_collection["processed_subject_ids"]) > 0:
        all_subj_metrics = []
        for subj_id_csv in group_results_collection["processed_subject_ids"]:
            subj_met = {
                "subject_id": subj_id_csv, "group": group_identifier,
                "global_auc_main": group_results_collection["subject_global_auc_scores"].get(subj_id_csv, np.nan)
            }
            main_met_subj = group_results_collection["subject_global_metrics_maps"].get(
                subj_id_csv, {})
            if main_met_subj:
                subj_met.update(
                    {f"main_{k}": v for k, v in main_met_subj.items()})
            all_subj_metrics.append(subj_met)
        if all_subj_metrics:
            pd.DataFrame(all_subj_metrics).to_csv(
                os.path.join(group_summary_dir, "all_subjects_summary_metrics.csv"), index=False
            )
            logger_run_group.info(  # MODIFIÉ logger
                "Saved group summary CSV to %s", group_summary_dir)

    logger_run_group.info(  # MODIFIÉ logger
        "Finished INTRA-SUBJECT DECODING for GROUP %s. Total time: %.1f min",
        group_identifier, (time.time() - total_group_analysis_start_time) / 60
    )
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

    # Déterminer n_jobs_to_use
    n_jobs_arg_str_main = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use_main = - \
            1 if n_jobs_arg_str_main.lower() == "auto" else int(n_jobs_arg_str_main)
    except ValueError:
        logger_run_group.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str_main}'). Using default from config: {N_JOBS_PROCESSING} (becomes -1 if 'auto').")
        n_jobs_to_use_main = -1 if N_JOBS_PROCESSING.lower() == "auto" else int(N_JOBS_PROCESSING)

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
        # Les autres paramètres utiliseront les valeurs par défaut de la fonction,
        # qui sont basées sur les constantes importées.
    )

    logger_run_group.info("\n%s EEG GROUP INTRA-SUBJECT DECODING SCRIPT FINISHED (%s) %s",
                          "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
