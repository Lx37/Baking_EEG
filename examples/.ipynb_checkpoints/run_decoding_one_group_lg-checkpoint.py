
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

# --- Configuration du chemin pour les imports ---
SCRIPT_DIR_EXAMPLE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_EXAMPLE = os.path.abspath(os.path.join(SCRIPT_DIR_EXAMPLE, ".."))
if PROJECT_ROOT_EXAMPLE not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_EXAMPLE)
# --- Fin Configuration du chemin --



from Baking_EEG.config.decoding_config import (
    CLASSIFIER_MODEL_TYPE, USE_GRID_SEARCH_OPTIMIZATION,
    USE_CSP_FOR_TEMPORAL_PIPELINES, USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    PARAM_GRID_CONFIG_EXTENDED, CV_FOLDS_FOR_GRIDSEARCH_INTERNAL,
    FIXED_CLASSIFIER_PARAMS_CONFIG, N_PERMUTATIONS_INTRA_SUBJECT,
    N_PERMUTATIONS_GROUP_LEVEL, GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING, CHANCE_LEVEL_AUC,
    INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG,
    COMPUTE_TGM_FOR_MAIN_COMPARISON, CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG,
    SAVE_ANALYSIS_RESULTS, GENERATE_PLOTS, N_JOBS_PROCESSING
)
from Baking_EEG.config.config import ALL_SUBJECT_GROUPS
from Baking_EEG.utils.vizualization_utils_LG import (
    plot_group_mean_scores_barplot_lg,
    plot_group_temporal_decoding_statistics_lg,
    plot_group_tgm_statistics_lg
)
from Baking_EEG.utils import stats_utils as bEEG_stats
from Baking_EEG.utils.utils import (
    configure_project_paths, setup_analysis_results_directory
)
# La fonction execute_single_subject_lg_decoding est dans run_decoding_one_lg.py
try:
    from Baking_EEG.examples.run_decoding_one_lg import (
        execute_single_subject_lg_decoding
    )
except ImportError as e_import:
    print(f"Erreur d'import execute_single_subject_lg_decoding: {e_import}")
    sys.exit(1)

# --- Configuration du Logging ---
LOG_DIR_RUN_GROUP_LG = './logs_run_group_lg_analysis'
os.makedirs(LOG_DIR_RUN_GROUP_LG, exist_ok=True)
LOG_FILENAME_RUN_GROUP_LG = os.path.join(
    LOG_DIR_RUN_GROUP_LG,
    datetime.now().strftime('log_run_group_lg_analysis_%Y-%m-%d_%H%M%S.log')
)

for handler in logging.getLogger().handlers[:]:
    logging.getLogger().removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format=('%(asctime)s - %(levelname)s - %(name)s - '
            '[%(funcName)s:%(lineno)d] - %(message)s'),
    handlers=[
        logging.FileHandler(LOG_FILENAME_RUN_GROUP_LG),
        logging.StreamHandler(sys.stdout)
    ]
)
logger_run_group_lg = logging.getLogger(__name__)
logging.getLogger(
    "Baking_EEG.examples.run_single_subject_lg_analysis").setLevel(logging.INFO)
logging.getLogger("Baking_EEG.utils.stats_utils").setLevel(logging.INFO)
logging.getLogger(
    "Baking_EEG.viz.visualization_utils_LG").setLevel(logging.INFO)


def execute_group_intra_subject_lg_decoding_analysis(
    subject_ids_in_group,
    group_identifier,
    decoding_protocol_identifier="Single_LG_Protocol_Group_Analysis",
    save_results_flag=SAVE_ANALYSIS_RESULTS,
    enable_verbose_logging=False,
    generate_plots_flag=GENERATE_PLOTS,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_each_subject=N_JOBS_PROCESSING,
    compute_group_level_stats_flag=True,
    n_perms_intra_subject_folds_for_group_runs=N_PERMUTATIONS_INTRA_SUBJECT,
    classifier_type_for_group_runs=CLASSIFIER_MODEL_TYPE,
    compute_tgm_for_group_subjects_flag=COMPUTE_TGM_FOR_MAIN_COMPARISON,
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
    loading_conditions_config=CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG
):
    """Executes intra-subject LG decoding for all subjects in a group and aggregates results."""
    if not isinstance(subject_ids_in_group, list) or not subject_ids_in_group:
        logger_run_group_lg.error(
            "subject_ids_in_group must be a non-empty list.")
        return {}
    if not isinstance(group_identifier, str) or not group_identifier:
        logger_run_group_lg.error(
            "group_identifier must be a non-empty string.")
        return {}

    total_group_analysis_start_time = time.time()

    # Convertir n_jobs
    actual_n_jobs_subject = -1 if isinstance(n_jobs_for_each_subject,
                                             str) and n_jobs_for_each_subject.lower() == "auto" else int(n_jobs_for_each_subject)
    actual_n_jobs_group_stats = -1 if isinstance(n_jobs_for_group_cluster_stats,
                                                 str) and n_jobs_for_group_cluster_stats.lower() == "auto" else int(n_jobs_for_group_cluster_stats)

    logger_run_group_lg.info(
        "Starting intra-subject LG decoding analysis for GROUP: %s. GS: %s, CSP: %s, ANOVA FS: %s. n_jobs_subj: %s, n_jobs_grp_stats: %s.",
        group_identifier, use_grid_search_for_group, use_csp_for_temporal_group,
        use_anova_fs_for_temporal_group, actual_n_jobs_subject, actual_n_jobs_group_stats
    )

    group_results_collection = {
        "subject_lg_global_auc_scores": {}, "subject_lg_global_metrics_maps": {},
        "subject_lg_temporal_scores_1d_mean_list": [],
        "subject_epochs_time_points_list": [], "subject_lg_tgm_scores_mean_list": [],
        "subject_lg_mean_of_specific_scores_list": [],
        "subject_lg_local_effect_scores_list": [],
        "processed_subject_ids": []
    }

    for i, subject_id_current in enumerate(subject_ids_in_group, 1):
        logger_run_group_lg.info("\n--- LG Group '%s': Processing Subject %d/%d: %s ---",
                                 group_identifier, i, len(subject_ids_in_group), subject_id_current)

        subject_output_dict = execute_single_subject_lg_decoding(
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

        s_lg_auc = subject_output_dict.get("lg_main_mean_auc_global", np.nan)
        s_lg_metrics = subject_output_dict.get("lg_main_global_metrics", {})
        s_lg_scores_t_1d_mean = subject_output_dict.get(
            "lg_main_scores_1d_mean")
        s_times_t = subject_output_dict.get("epochs_time_points")
        s_lg_scores_tgm_mean = subject_output_dict.get("lg_main_tgm_mean")
        s_lg_mean_specific = subject_output_dict.get(
            "lg_mean_of_specific_scores_1d")

        # Extract local effect centric averages for group analysis
        s_lg_local_effects = subject_output_dict.get(
            "lg_local_effect_centric_avg_results", [])
        local_effect_score = None
        if s_lg_local_effects:
            # Take the first local effect average or combine them
            for effect in s_lg_local_effects:
                if effect.get("average_scores_1d") is not None:
                    local_effect_score = effect["average_scores_1d"]
                    break

        group_results_collection["subject_lg_global_auc_scores"][subject_id_current] = s_lg_auc
        group_results_collection["subject_lg_global_metrics_maps"][subject_id_current] = s_lg_metrics

        if pd.notna(s_lg_auc) and s_lg_scores_t_1d_mean is not None and s_times_t is not None and \
           s_lg_scores_t_1d_mean.size > 0 and s_times_t.size > 0:
            group_results_collection["subject_lg_temporal_scores_1d_mean_list"].append(
                s_lg_scores_t_1d_mean)
            group_results_collection["subject_epochs_time_points_list"].append(
                s_times_t)
            group_results_collection["processed_subject_ids"].append(
                subject_id_current)

            if compute_tgm_for_group_subjects_flag:
                if s_lg_scores_tgm_mean is not None and not (isinstance(s_lg_scores_tgm_mean, float) and np.isnan(s_lg_scores_tgm_mean)) and s_lg_scores_tgm_mean.ndim == 2:
                    group_results_collection["subject_lg_tgm_scores_mean_list"].append(
                        s_lg_scores_tgm_mean)
                else:
                    nan_tgm = np.full_like(s_lg_scores_t_1d_mean[:, np.newaxis] * s_lg_scores_t_1d_mean[np.newaxis, :],
                                           np.nan) if s_lg_scores_t_1d_mean is not None and s_lg_scores_t_1d_mean.ndim == 1 and s_lg_scores_t_1d_mean.size > 0 else np.array([[]])
                    group_results_collection["subject_lg_tgm_scores_mean_list"].append(
                        nan_tgm)

            if s_lg_mean_specific is not None and not (isinstance(s_lg_mean_specific, float) and np.isnan(s_lg_mean_specific)) and s_lg_mean_specific.ndim == 1:
                group_results_collection["subject_lg_mean_of_specific_scores_list"].append(
                    s_lg_mean_specific)
            else:
                nan_specific = np.full_like(
                    s_lg_scores_t_1d_mean, np.nan) if s_lg_scores_t_1d_mean is not None and s_lg_scores_t_1d_mean.size > 0 else np.array([])
                group_results_collection["subject_lg_mean_of_specific_scores_list"].append(
                    nan_specific)

            if local_effect_score is not None and not (isinstance(local_effect_score, float) and np.isnan(local_effect_score)) and local_effect_score.ndim == 1:
                group_results_collection["subject_lg_local_effect_scores_list"].append(
                    local_effect_score)
            else:
                nan_local_effect = np.full_like(
                    s_lg_scores_t_1d_mean, np.nan) if s_lg_scores_t_1d_mean is not None and s_lg_scores_t_1d_mean.size > 0 else np.array([])
                group_results_collection["subject_lg_local_effect_scores_list"].append(
                    nan_local_effect)
        else:
            logger_run_group_lg.warning(
                "Skipping subject %s from LG group '%s' aggregation (errors or no valid main scores).", subject_id_current, group_identifier)

    group_summary_dir = None
    if save_results_flag or generate_plots_flag:
        dir_suffix = (
            f"LG_{classifier_type_for_group_runs}_GS{use_grid_search_for_group}_CSP{use_csp_for_temporal_group}_ANOVA{use_anova_fs_for_temporal_group}")
        
        # Create group_protocol path for better organization (LG protocol)
        group_protocol_path = f"{group_identifier}_LG"
        
        group_summary_dir = setup_analysis_results_directory(
            base_output_results_path, "group_summary_intra_subject_lg", group_protocol_path, dir_suffix
        )

    valid_lg_global_scores = np.array(
        [s for s in group_results_collection["subject_lg_global_auc_scores"].values() if pd.notna(s)])
    if len(valid_lg_global_scores) > 0:
        mean_lg_auc = np.mean(valid_lg_global_scores)
        std_lg_auc = np.std(valid_lg_global_scores)
        logger_run_group_lg.info("LG Group %s - Overall Global Mean AUC (Main LG Dec.): %.3f +/- %.3f (N=%d subjects)",
                                 group_identifier, mean_lg_auc, std_lg_auc, len(valid_lg_global_scores))
        if compute_group_level_stats_flag and len(valid_lg_global_scores) >= 2:
            stat_lg_g, p_val_lg_g = bEEG_stats.compare_global_scores_to_chance(
                valid_lg_global_scores, CHANCE_LEVEL_AUC, "ttest", "greater")
            logger_run_group_lg.info(
                "  LG Global AUC (Main Dec.) vs Chance: t=%.3f, p=%.4f", stat_lg_g, p_val_lg_g)
            if save_results_flag and group_summary_dir:
                with open(os.path.join(group_summary_dir, "stats_lg_global_auc.txt"), "w") as f_stat:
                    f_stat.write(
                        f"Intra-Subject LG Global AUC (Main Dec.) vs Chance ({CHANCE_LEVEL_AUC})\nGroup: {group_identifier}, N: {len(valid_lg_global_scores)}\nMean AUC: {mean_lg_auc:.4f}, Std: {std_lg_auc:.4f}\nT-stat: {stat_lg_g:.4f}, P-val: {p_val_lg_g:.4f}\n")
        if generate_plots_flag and group_summary_dir:
            plot_group_mean_scores_barplot_lg(group_results_collection["subject_lg_global_auc_scores"],
                                              f"{group_identifier} - Subject LG Global AUCs (Main)", group_summary_dir, "Global ROC AUC", CHANCE_LEVEL_AUC)
    else:
        logger_run_group_lg.warning(
            "No valid LG global scores for group %s.", group_identifier)

    ref_times_1d = None
    if len(group_results_collection["subject_lg_temporal_scores_1d_mean_list"]) >= 2 and compute_group_level_stats_flag:
        ref_times_idx = next((j for j, t_arr in enumerate(
            group_results_collection["subject_epochs_time_points_list"]) if t_arr is not None and t_arr.size > 0), -1)
        if ref_times_idx != -1:
            ref_times_1d = group_results_collection["subject_epochs_time_points_list"][ref_times_idx]
            valid_1d_scores_lg_main = [s for i, s in enumerate(group_results_collection["subject_lg_temporal_scores_1d_mean_list"]) if i < len(group_results_collection["subject_epochs_time_points_list"]) and (
                t_cur := group_results_collection["subject_epochs_time_points_list"][i]) is not None and t_cur.size == ref_times_1d.size and np.allclose(t_cur, ref_times_1d) and s is not None and not np.all(np.isnan(s))]
            if len(valid_1d_scores_lg_main) >= 2:
                stacked_1d_lg = np.array(valid_1d_scores_lg_main)
                logger_run_group_lg.info(
                    "LG Group %s (N=%d for Main 1D stats): Running group stats...", group_identifier, stacked_1d_lg.shape[0])
                t_obs_fdr_lg, fdr_mask_lg, p_fdr_lg, fdr_test_info_lg = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_1d_lg, CHANCE_LEVEL_AUC, alternative_hypothesis="greater")
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_FDR_Main.npz"),
                                        t_obs=t_obs_fdr_lg, sig_mask=fdr_mask_lg, p_vals=p_fdr_lg, times=ref_times_1d)

                t_thresh_clu_lg = (group_cluster_test_t_thresh_value if group_cluster_test_threshold_method ==
                                   "stat" else INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method == "tfce" else None)
                if group_cluster_test_threshold_method == "stat" and t_thresh_clu_lg is None and stacked_1d_lg.shape[0] > 1:
                    t_thresh_clu_lg = scipy.stats.t.ppf(
                        1.0 - 0.05 / 2, df=stacked_1d_lg.shape[0] - 1)
                elif group_cluster_test_threshold_method == "stat" and t_thresh_clu_lg is None and stacked_1d_lg.shape[0] <= 1:
                    logger_run_group_lg.warning(
                        f"Cannot calculate t_thresh_clu_lg for stat method with N={stacked_1d_lg.shape[0]}. Skipping cluster test or using TFCE if applicable.")
                    t_thresh_clu_lg = INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG if group_cluster_test_threshold_method != "tfce" else t_thresh_clu_lg

                t_obs_clu_lg, clu_lg, p_clu_lg, _ = bEEG_stats.perform_cluster_permutation_test(
                    stacked_1d_lg, CHANCE_LEVEL_AUC, n_perms_for_group_cluster_test, t_thresh_clu_lg, "greater", actual_n_jobs_group_stats)
                clu_map_lg = (bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d.shape, clu_lg, p_clu_lg) if clu_lg and p_clu_lg is not None else None)
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_CLUSTER_Main.npz"),
                                        t_obs=t_obs_clu_lg, clusters=clu_lg, p_vals=p_clu_lg, p_map=clu_map_lg, times=ref_times_1d)
                # Plots de groupe désactivés -
                # if generate_plots_flag and group_summary_dir:
                #     sem_1d_lg = (scipy.stats.sem(
                #         stacked_1d_lg, axis=0, nan_policy="omit") if stacked_1d_lg.shape[0] > 1 else None)
                #     plot_group_temporal_decoding_statistics_lg(ref_times_1d, np.mean(
                #         stacked_1d_lg, axis=0), f"{group_identifier} (Main LG 1D Temporal)", group_summary_dir, sem_1d_lg, clu_map_lg, fdr_mask_lg, CHANCE_LEVEL_AUC)

            # Analyze LG specific comparisons at group level
            valid_1d_scores_lg_specific = [s for i, s in enumerate(group_results_collection["subject_lg_mean_of_specific_scores_list"]) if i < len(group_results_collection["subject_epochs_time_points_list"]) and (
                t_cur := group_results_collection["subject_epochs_time_points_list"][i]) is not None and t_cur.size == ref_times_1d.size and np.allclose(t_cur, ref_times_1d) and s is not None and not np.all(np.isnan(s))]
            if len(valid_1d_scores_lg_specific) >= 2:
                stacked_1d_lg_specific = np.array(valid_1d_scores_lg_specific)
                logger_run_group_lg.info(
                    "LG Group %s (N=%d for Specific 1D stats): Running specific comparison group stats...", group_identifier, stacked_1d_lg_specific.shape[0])
                t_obs_fdr_lg_spec, fdr_mask_lg_spec, p_fdr_lg_spec, fdr_test_info_lg_spec = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_1d_lg_specific, CHANCE_LEVEL_AUC, alternative_hypothesis="greater")
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_FDR_Specific.npz"),
                                        t_obs=t_obs_fdr_lg_spec, sig_mask=fdr_mask_lg_spec, p_vals=p_fdr_lg_spec, times=ref_times_1d)

                t_obs_clu_lg_spec, clu_lg_spec, p_clu_lg_spec, _ = bEEG_stats.perform_cluster_permutation_test(
                    stacked_1d_lg_specific, CHANCE_LEVEL_AUC, n_perms_for_group_cluster_test, t_thresh_clu_lg, "greater", actual_n_jobs_group_stats)
                clu_map_lg_spec = (bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d.shape, clu_lg_spec, p_clu_lg_spec) if clu_lg_spec and p_clu_lg_spec is not None else None)
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_CLUSTER_Specific.npz"),
                                        t_obs=t_obs_clu_lg_spec, clusters=clu_lg_spec, p_vals=p_clu_lg_spec, p_map=clu_map_lg_spec, times=ref_times_1d)
                if generate_plots_flag and group_summary_dir:
                    sem_1d_lg_spec = (scipy.stats.sem(
                        stacked_1d_lg_specific, axis=0, nan_policy="omit") if stacked_1d_lg_specific.shape[0] > 1 else None)
                    plot_group_temporal_decoding_statistics_lg(ref_times_1d, np.mean(
                        stacked_1d_lg_specific, axis=0), f"{group_identifier} (LG Specific 1D Temporal)", group_summary_dir, sem_1d_lg_spec, clu_map_lg_spec, fdr_mask_lg_spec, CHANCE_LEVEL_AUC, plot_suffix="_specific")

            # Analyze LG local effects at group level
            valid_1d_scores_lg_local = [s for i, s in enumerate(group_results_collection["subject_lg_local_effect_scores_list"]) if i < len(group_results_collection["subject_epochs_time_points_list"]) and (
                t_cur := group_results_collection["subject_epochs_time_points_list"][i]) is not None and t_cur.size == ref_times_1d.size and np.allclose(t_cur, ref_times_1d) and s is not None and not np.all(np.isnan(s))]
            if len(valid_1d_scores_lg_local) >= 2:
                stacked_1d_lg_local = np.array(valid_1d_scores_lg_local)
                logger_run_group_lg.info(
                    "LG Group %s (N=%d for Local Effect 1D stats): Running local effect group stats...", group_identifier, stacked_1d_lg_local.shape[0])
                t_obs_fdr_lg_local, fdr_mask_lg_local, p_fdr_lg_local, fdr_test_info_lg_local = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    stacked_1d_lg_local, CHANCE_LEVEL_AUC, alternative_hypothesis="greater")
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_FDR_LocalEffect.npz"),
                                        t_obs=t_obs_fdr_lg_local, sig_mask=fdr_mask_lg_local, p_vals=p_fdr_lg_local, times=ref_times_1d)

                t_obs_clu_lg_local, clu_lg_local, p_clu_lg_local, _ = bEEG_stats.perform_cluster_permutation_test(
                    stacked_1d_lg_local, CHANCE_LEVEL_AUC, n_perms_for_group_cluster_test, t_thresh_clu_lg, "greater", actual_n_jobs_group_stats)
                clu_map_lg_local = (bEEG_stats.create_p_value_map_from_cluster_results(
                    ref_times_1d.shape, clu_lg_local, p_clu_lg_local) if clu_lg_local and p_clu_lg_local is not None else None)
                if save_results_flag and group_summary_dir:
                    np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_temp_1D_CLUSTER_LocalEffect.npz"),
                                        t_obs=t_obs_clu_lg_local, clusters=clu_lg_local, p_vals=p_clu_lg_local, p_map=clu_map_lg_local, times=ref_times_1d)
                if generate_plots_flag and group_summary_dir:
                    sem_1d_lg_local = (scipy.stats.sem(
                        stacked_1d_lg_local, axis=0, nan_policy="omit") if stacked_1d_lg_local.shape[0] > 1 else None)
                    plot_group_temporal_decoding_statistics_lg(ref_times_1d, np.mean(
                        stacked_1d_lg_local, axis=0), f"{group_identifier} (LG Local Effect 1D Temporal)", group_summary_dir, sem_1d_lg_local, clu_map_lg_local, fdr_mask_lg_local, CHANCE_LEVEL_AUC, plot_suffix="_local_effect")

            logger_run_group_lg.info(
                "LG Group %s temporal 1D statistics completed.", group_identifier)
        else:
            logger_run_group_lg.warning(
                "No valid reference times found for LG group %s temporal statistics.", group_identifier)

    # Process TGM results if available
    if compute_tgm_for_group_subjects_flag and len(group_results_collection["subject_lg_tgm_scores_mean_list"]) >= 2:
        valid_tgm_lg = [tgm for tgm in group_results_collection["subject_lg_tgm_scores_mean_list"] if tgm is not None and not (
            isinstance(tgm, float) and np.isnan(tgm)) and tgm.ndim == 2 and tgm.size > 1]
        if len(valid_tgm_lg) >= 2 and ref_times_1d is not None:
            try:
                ref_shape = valid_tgm_lg[0].shape
                valid_tgm_lg_same_shape = [
                    tgm for tgm in valid_tgm_lg if tgm.shape == ref_shape]
                if len(valid_tgm_lg_same_shape) >= 2:
                    stacked_tgm_lg = np.array(valid_tgm_lg_same_shape)
                    logger_run_group_lg.info(
                        "LG Group %s (N=%d for TGM stats): Running TGM group stats...", group_identifier, stacked_tgm_lg.shape[0])
                    t_obs_tgm_lg, fdr_mask_tgm_lg, p_fdr_tgm_lg = bEEG_stats.perform_2d_fdr_correction_on_scores(
                        stacked_tgm_lg, CHANCE_LEVEL_AUC, alternative_hypothesis="greater")
                    if save_results_flag and group_summary_dir:
                        np.savez_compressed(os.path.join(group_summary_dir, "stats_lg_TGM_FDR_Main.npz"),
                                            t_obs=t_obs_tgm_lg, sig_mask=fdr_mask_tgm_lg, p_vals=p_fdr_tgm_lg, times=ref_times_1d)
                    if generate_plots_flag and group_summary_dir:
                        plot_group_tgm_statistics_lg(ref_times_1d, np.mean(stacked_tgm_lg, axis=0),
                                                     f"{group_identifier} (LG TGM)", group_summary_dir, fdr_mask_tgm_lg, CHANCE_LEVEL_AUC)
                    logger_run_group_lg.info(
                        "LG Group %s TGM statistics completed.", group_identifier)
                else:
                    logger_run_group_lg.warning(
                        "Not enough valid same-shaped TGMs for LG group %s (%d valid).", group_identifier, len(valid_tgm_lg_same_shape))
            except Exception as e_tgm:
                logger_run_group_lg.error(
                    "Error processing TGM for LG group %s: %s", group_identifier, e_tgm, exc_info=True)
        else:
            logger_run_group_lg.warning(
                "Not enough valid TGMs for LG group %s (%d valid) or no reference times.", group_identifier, len(valid_tgm_lg))

    logger_run_group_lg.info("LG Group analysis for '%s' completed in %.2fs. Total subjects processed: %d.",
                             group_identifier, time.time() - total_group_analysis_start_time, len(group_results_collection["processed_subject_ids"]))
    return group_results_collection


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="EEG Group LG Decoding Analysis Script",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli_parser.add_argument("--group", type=str, required=True, choices=list(ALL_SUBJECT_GROUPS.keys()),
                            help="Group identifier for LG analysis.")
    cli_parser.add_argument("--clf_type_override", type=str, default=None, choices=["svc", "logreg", "rf"],
                            help="Override default classifier type from config.")
    cli_parser.add_argument("--n_jobs_override", type=str, default=None,
                            help="Override n_jobs from config (e.g., '4' or 'auto').")
    cli_parser.add_argument("--skip_group_stats", action="store_true",
                            help="Skip group-level statistical analysis.")
    cli_parser.add_argument("--skip_tgm", action="store_true",
                            help="Skip temporal generalization matrix computation.")

    command_line_args = cli_parser.parse_args()

    n_jobs_arg_str = command_line_args.n_jobs_override if command_line_args.n_jobs_override is not None else N_JOBS_PROCESSING
    try:
        n_jobs_to_use = -1 if n_jobs_arg_str.lower() == "auto" else int(n_jobs_arg_str)
    except ValueError:
        logger_run_group_lg.warning(
            f"Invalid n_jobs_override ('{n_jobs_arg_str}'). Using default from config: {N_JOBS_PROCESSING} (becomes -1 if 'auto').")
        n_jobs_to_use = -1 if N_JOBS_PROCESSING.lower() == "auto" else int(N_JOBS_PROCESSING)

    classifier_type_to_use = command_line_args.clf_type_override if command_line_args.clf_type_override is not None else CLASSIFIER_MODEL_TYPE

    user_login = getuser()
    main_input_path, main_output_path = configure_project_paths(user_login)

    group_name = command_line_args.group
    subject_list = ALL_SUBJECT_GROUPS.get(group_name, [])
    if not subject_list:
        logger_run_group_lg.error(
            "No subjects found for group '%s'.", group_name)
        sys.exit(1)

    logger_run_group_lg.info("\n%s EEG GROUP LG DECODING SCRIPT STARTED (%s) %s",
                             "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
    logger_run_group_lg.info("User: %s, Group: %s, N subjects: %d",
                             user_login, group_name, len(subject_list))
    logger_run_group_lg.info("  Subjects: %s", ", ".join(subject_list))
    logger_run_group_lg.info("  Classifier: %s, n_jobs: %s",
                             classifier_type_to_use, n_jobs_to_use)
    logger_run_group_lg.info(
        "  GridSearch Optimization: %s", USE_GRID_SEARCH_OPTIMIZATION)
    logger_run_group_lg.info(
        "  CSP for Temporal Pipelines: %s", USE_CSP_FOR_TEMPORAL_PIPELINES)
    logger_run_group_lg.info(
        "  ANOVA FS for Temporal Pipelines: %s", USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES)
    logger_run_group_lg.info("  Compute Group Stats: %s",
                             not command_line_args.skip_group_stats)
    logger_run_group_lg.info(
        "  Compute TGM: %s", not command_line_args.skip_tgm)

    execute_group_intra_subject_lg_decoding_analysis(
        subject_ids_in_group=subject_list,
        group_identifier=group_name,
        base_input_data_path=main_input_path,
        base_output_results_path=main_output_path,
        n_jobs_for_each_subject=n_jobs_to_use,
        n_jobs_for_group_cluster_stats=n_jobs_to_use,
        classifier_type_for_group_runs=classifier_type_to_use,
        compute_group_level_stats_flag=not command_line_args.skip_group_stats,
        compute_tgm_for_group_subjects_flag=not command_line_args.skip_tgm,
        generate_plots_flag=False,  # Disable all group-level plots
    )

    logger_run_group_lg.info("\n%s EEG GROUP LG DECODING SCRIPT FINISHED (%s) %s",
                             "="*10, datetime.now().strftime('%Y-%m-%d %H:%M'), "="*10)
