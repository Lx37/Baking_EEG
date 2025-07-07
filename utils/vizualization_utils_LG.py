import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats

logger_viz_lg = logging.getLogger(__name__)

# Styling parameters
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 11
DPI_VALUE = 150

# Import stats utilities
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from utils import stats_utils as bEEG_stats
except ImportError:
    logger_viz_lg.warning(
        "Could not import stats_utils. Statistical tests will be disabled.")
    bEEG_stats = None


def create_subject_decoding_dashboard_plots_lg(
    main_epochs_time_points,
    classifier_name_for_title,
    subject_identifier,
    group_identifier,
    output_directory_path,
    CHANCE_LEVEL_AUC=0.5,
    protocol_type="LG",
    lg_main_original_labels_array=None,
    lg_main_predicted_probabilities_global=None,
    lg_main_predicted_labels_global=None,
    lg_main_cross_validation_global_scores=None,
    lg_main_temporal_scores_1d_all_folds=None,
    lg_main_mean_temporal_decoding_scores_1d=None,
    lg_main_temporal_1d_fdr_sig_data=None,
    lg_main_temporal_1d_cluster_sig_data=None,
    lg_main_mean_temporal_generalization_matrix_scores=None,
    lg_main_tgm_fdr_sig_data=None,
    lg_main_decoding_global_metrics_for_plot=None,
    lg_specific_comparison_results=None,
    lg_mean_of_specific_scores_1d=None,
    lg_sem_of_specific_scores_1d=None,
    lg_mean_specific_fdr_sig_data=None,
    lg_mean_specific_cluster_sig_data=None,
    lg_global_effect_results=None,
    lg_local_effect_centric_average_results_list=None
):
    """
    Create LG decoding plots - 6 specific comparisons with 2 plots per page.

    Les 6 comparaisons demandées:
    1. LD_ALL vs LS_ALL (AUC + TGM)
    2. GD_ALL vs GS_ALL (AUC + TGM)  
    3. LSGS vs LSGD (AUC)
    4. LDGS vs LDGD (AUC)
    5. LSGS vs LDGS (AUC)
    6. LSGD vs LDGD (AUC)
    """
    try:
        logger_viz_lg.info(
            "Creating LG dashboard plots for subject %s", subject_identifier)

        # Set up figure style
        plt.style.use('default')
        sns.set_palette("husl")

        # Apply statistical tests to comparison results if available
        if lg_specific_comparison_results:
            for i, result in enumerate(lg_specific_comparison_results):
                lg_specific_comparison_results[i] = _apply_statistical_tests_to_comparison_data(
                    result, main_epochs_time_points
                )

        # =================== PAGE 1: LD_ALL vs LS_ALL ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "LD_ALL_vs_LS_ALL",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=True
        )

        # =================== PAGE 2: GD_ALL vs GS_ALL ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "GD_ALL_vs_GS_ALL",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=True
        )

        # =================== PAGE 3: LSGS vs LSGD ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "LSGS_vs_LSGD",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=False
        )

        # =================== PAGE 4: LDGS vs LDGD ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "LDGS_vs_LDGD",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=False
        )

        # =================== PAGE 5: LSGS vs LDGS ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "LSGS_vs_LDGS",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=False
        )

        # =================== PAGE 6: LSGD vs LDGD ===================
        _create_lg_comparison_page(
            main_epochs_time_points,
            lg_specific_comparison_results,
            "LSGD_vs_LDGD",
            subject_identifier,
            group_identifier,
            output_directory_path,
            classifier_name_for_title,
            CHANCE_LEVEL_AUC,
            include_tgm=False
        )

        logger_viz_lg.info(
            "LG dashboard plots creation completed for subject %s", subject_identifier)

    except Exception as e:
        logger_viz_lg.error("Error creating LG dashboard plots for subject %s: %s",
                            subject_identifier, e, exc_info=True)


def _create_lg_comparison_page(times, lg_results, comparison_name, subject_id, group_id,
                               output_dir, classifier_name, chance_level, include_tgm=True):
    """Create a single page with 2 plots for a specific LG comparison."""

    try:
        # Find the specific comparison result
        comparison_data = None
        if lg_results:
            # Map from display names to actual comparison names in results
            name_mapping = {
                "LSGS_vs_LSGD": "Local Standard: Global Standard vs Global Deviant",
                "LDGS_vs_LDGD": "Local Deviant: Global Standard vs Global Deviant",
                "LSGS_vs_LDGS": "Global Standard: Local Standard vs Local Deviant",
                "LSGD_vs_LDGD": "Global Deviant: Local Standard vs Local Deviant",
                # Simplified aliases that map to the 4 actual comparisons we have
                # Using first comparison as fallback
                "LD_ALL_vs_LS_ALL": "Local Standard: Global Standard vs Global Deviant",
                # Using second comparison as fallback
                "GD_ALL_vs_GS_ALL": "Local Deviant: Global Standard vs Global Deviant"
            }

            target_name = name_mapping.get(comparison_name, comparison_name)

            for result in lg_results:
                result_name = result.get('comparison_name', '')
                if result_name == target_name:
                    comparison_data = result
                    break

        if comparison_data is None:
            logger_viz_lg.warning(
                f"No data found for comparison {comparison_name}. Available comparisons: {[r.get('comparison_name', 'UNKNOWN') for r in lg_results] if lg_results else 'None'}")
            return

        # Create figure with 2 subplots (1 row, 2 columns) or 1 subplot for AUC only
        if include_tgm:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
            ax2 = None

        # =================== Plot 1: AUC temporal curve ===================
        mean_scores = comparison_data.get('scores_1d_mean')
        all_folds_scores = comparison_data.get('all_folds_scores_1d')
        fdr_data = comparison_data.get('fdr_significance_data')
        cluster_data = comparison_data.get('cluster_significance_data')

        if mean_scores is not None and times is not None:
            original_labels = comparison_data.get('original_labels')
            _plot_lg_temporal_auc(ax1, times, mean_scores, all_folds_scores,
                                  fdr_data, cluster_data, comparison_name,
                                  chance_level, subject_id, original_labels)

        # =================== Plot 2: TGM (if included) ===================
        if include_tgm and ax2 is not None:
            tgm_scores = comparison_data.get('tgm_mean')
            tgm_fdr_data = comparison_data.get('tgm_fdr_data')

            if tgm_scores is not None:
                _plot_lg_tgm(ax2, times, tgm_scores, tgm_fdr_data,
                             comparison_name, chance_level)

        # Adjust layout and save
        plt.tight_layout()

        # Save the plot
        safe_comparison_name = comparison_name.replace(
            '_vs_', '_vs_').replace('_ALL', '_ALL')
        filename = f"lg_{safe_comparison_name}_{subject_id}_{group_id}_{classifier_name}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=DPI_VALUE, bbox_inches='tight')
        logger_viz_lg.info(f"LG comparison plot saved to: {filepath}")

        plt.close()

    except Exception as e:
        logger_viz_lg.error(f"Error creating LG comparison page {comparison_name}: {e}",
                            exc_info=True)


def _plot_lg_temporal_auc(ax, times, mean_scores, all_folds_scores, fdr_data,
                          cluster_data, comparison_name, chance_level, subject_id,
                          original_labels_array=None, label_encoder=None):
    """Plot temporal AUC curve with statistical significance and individual folds."""

    # Plot individual folds (with transparency in gray)
    if all_folds_scores is not None and all_folds_scores.ndim == 2:
        n_folds = all_folds_scores.shape[0]
        
        # Calculate detailed epoch information per fold
        fold_epoch_info = ""
        if original_labels_array is not None:
            avg_epochs_per_fold = len(original_labels_array) / n_folds
            
            # Calculate class distribution
            unique_labels, counts = np.unique(original_labels_array, return_counts=True)
            if len(unique_labels) == 2:
                # Binary classification - show class breakdown
                avg_class0_per_fold = counts[0] / n_folds
                avg_class1_per_fold = counts[1] / n_folds
                fold_epoch_info = f", ~{avg_epochs_per_fold:.0f} epochs/fold (~{avg_class0_per_fold:.0f}+{avg_class1_per_fold:.0f})"
            else:
                fold_epoch_info = f", ~{avg_epochs_per_fold:.0f} epochs/fold"
        
        for i, fold_scores in enumerate(all_folds_scores):
            if not np.all(np.isnan(fold_scores)):
                fold_label = f'Individual folds (n={n_folds}{fold_epoch_info})' if i == 0 else ""
                ax.plot(times, fold_scores, color='gray', alpha=0.4,
                        linewidth=1, label=fold_label)

    # Plot mean curve
    if mean_scores is not None:
        # Create label with fold and detailed epoch information
        fold_info = ""
        epoch_info = ""
        
        if all_folds_scores is not None and all_folds_scores.ndim == 2:
            n_folds = all_folds_scores.shape[0]
            fold_info = f" ({n_folds} folds)"
        
        # Add detailed epoch information if available
        if original_labels_array is not None:
            unique_labels, counts = np.unique(original_labels_array, return_counts=True)
            total_epochs = len(original_labels_array)
            
            if len(unique_labels) == 2:
                # Binary classification - show detailed breakdown
                epoch_info = f" - {total_epochs} epochs ({counts[0]}+{counts[1]})"
            else:
                epoch_info = f" - {total_epochs} epochs"
        
        ax.plot(times, mean_scores, 'b-', linewidth=3, 
               label=f'Mean AUC{fold_info}{epoch_info}')

        # Calculate and plot confidence interval/SEM
        if all_folds_scores is not None and all_folds_scores.ndim == 2:
            sem_scores = scipy.stats.sem(
                all_folds_scores, axis=0, nan_policy='omit')
            ci_lower = mean_scores - 1.96 * sem_scores
            ci_upper = mean_scores + 1.96 * sem_scores
            ax.fill_between(times, ci_lower, ci_upper, alpha=0.3,
                            color='blue', label='95% CI')

    # Add chance level
    ax.axhline(y=chance_level, color='red', linestyle='--',
               alpha=0.7, label=f'Chance ({chance_level})')

    # Add stimulus onset marker
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.8,
               label='Stimulus Onset')

    # Add FDR significance markers
    if fdr_data and fdr_data.get('mask') is not None:
        fdr_mask = fdr_data['mask']
        
        # Créer le label FDR avec l'information détaillée du test
        fdr_label = "FDR p<0.05"
        test_info = fdr_data.get('test_info', {})
        if test_info:
            test_type = test_info.get("test_type", "unknown")
            if test_type == "adaptive":
                n_ttest = test_info.get("ttest_features", 0)
                n_wilcoxon = test_info.get("wilcoxon_features", 0)
                fdr_label = f"FDR p<0.05 (adap: {n_ttest}t, {n_wilcoxon}W)"
            elif test_type == "ttest":
                fdr_label = "FDR p<0.05 (t-test)"
            elif test_type == "wilcoxon":
                fdr_label = "FDR p<0.05 (Wilcoxon)"
        
        if np.any(fdr_mask):
            # Create significance bar at bottom
            y_min, y_max = ax.get_ylim()
            y_sig_fdr = y_min + 0.02 * (y_max - y_min)
            ax.fill_between(times, y_sig_fdr - 0.01, y_sig_fdr,
                            where=fdr_mask, color='green', alpha=0.7,
                            step='mid', label=fdr_label)
        else:
            # Afficher le label FDR même si pas significatif
            fdr_label_no_sig = "FDR (no sig.)"
            if test_info:
                test_type = test_info.get("test_type", "unknown")
                if test_type == "adaptive":
                    n_ttest = test_info.get("ttest_features", 0)
                    n_wilcoxon = test_info.get("wilcoxon_features", 0)
                    fdr_label_no_sig = f"FDR (no sig., adap: {n_ttest}t, {n_wilcoxon}W)"
                elif test_type == "ttest":
                    fdr_label_no_sig = "FDR (no sig., t-test)"
                elif test_type == "wilcoxon":
                    fdr_label_no_sig = "FDR (no sig., Wilcoxon)"
            ax.plot([], [], color='green', alpha=0.7, label=fdr_label_no_sig)
    else:
        # Pas de données FDR disponibles
        ax.plot([], [], color='green', alpha=0.7, label="FDR (N/A)")

    # Add cluster significance markers
    if cluster_data and cluster_data.get('mask') is not None:
        cluster_mask = cluster_data['mask']
        
        if np.any(cluster_mask):
            # Create significance bar slightly below FDR
            y_min, y_max = ax.get_ylim()
            y_sig_cluster = y_min + 0.04 * (y_max - y_min)
            ax.fill_between(times, y_sig_cluster - 0.01, y_sig_cluster,
                            where=cluster_mask, color='orange', alpha=0.7,
                            step='mid', label="Cluster p<0.05")
        else:
            # Afficher le label cluster même si pas significatif
            ax.plot([], [], color='orange', alpha=0.7, label="Cluster (no sig.)")
    else:
        # Pas de données cluster disponibles
        ax.plot([], [], color='orange', alpha=0.7, label="Cluster (N/A)")

    # Formatting
    ax.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
    # Créer le titre avec les informations d'epochs si disponibles
    base_title = f'{comparison_name.replace("_", " ")} - Subject {subject_id}'
    if original_labels_array is not None:
        title_with_epochs = _add_epochs_info_to_title_lg(base_title, original_labels_array, label_encoder)
        ax.set_title(title_with_epochs, fontsize=FONT_SIZE_TITLE)
    else:
        ax.set_title(base_title, fontsize=FONT_SIZE_TITLE)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, alpha=0.3)

    # Add peak AUC info
    if mean_scores is not None:
        peak_auc = np.max(mean_scores)
        peak_time = times[np.argmax(mean_scores)]
        ax.text(0.02, 0.98, f'Peak AUC: {peak_auc:.3f}\nat {peak_time:.3f}s',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=FONT_SIZE_LEGEND)


def _plot_lg_tgm(ax, times, tgm_scores, tgm_fdr_data, comparison_name, chance_level):
    """Plot Temporal Generalization Matrix."""

    if tgm_scores is None or tgm_scores.ndim != 2:
        logger_viz_lg.warning(f"Invalid TGM data for {comparison_name}")
        return

    # Plot TGM heatmap
    vmin = chance_level - 0.1
    vmax = chance_level + 0.1

    im = ax.imshow(tgm_scores, cmap='RdBu_r', aspect='auto', origin='lower',
                   extent=[times[0], times[-1], times[0], times[-1]],
                   vmin=vmin, vmax=vmax)

    # Add stimulus onset lines
    ax.axvline(x=0, color='black', linestyle=':', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.8)

    # Add FDR significance contours if available
    if tgm_fdr_data and tgm_fdr_data.get('mask') is not None:
        fdr_mask = tgm_fdr_data['mask']
        if np.any(fdr_mask) and fdr_mask.shape == tgm_scores.shape:
            X, Y = np.meshgrid(times, times)
            ax.contourf(X, Y, fdr_mask.astype(float),
                        levels=[0.5, 1.5], colors="none",
                        hatches=["///"], alpha=0.4)

    # Formatting
    ax.set_xlabel('Testing Time (s)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Training Time (s)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'{comparison_name.replace("_", " ")} - TGM',
                 fontsize=FONT_SIZE_TITLE)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AUC Score', fontsize=FONT_SIZE_LABEL)
    cbar.ax.axhline(chance_level, color="black", linestyle="--", lw=1)


def _apply_statistical_tests_to_comparison_data(comparison_data, times, n_permutations=1024):
    """Apply FDR and cluster permutation tests to comparison data."""

    if not bEEG_stats or not comparison_data:
        return comparison_data

    all_folds_scores = comparison_data.get('all_folds_scores_1d')

    if all_folds_scores is None or all_folds_scores.ndim != 2:
        return comparison_data

    try:
        # FDR test
        _, fdr_mask, fdr_p, fdr_test_info = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
            all_folds_scores,
            chance_level=0.5,
            alternative_hypothesis="greater"
        )
        comparison_data['fdr_significance_data'] = {
            'mask': fdr_mask,
            'p_values': fdr_p,
            'method': 'FDR',
            'test_info': fdr_test_info
        }

        # Cluster permutation test
        t_obs, clusters, cluster_p, _ = bEEG_stats.perform_cluster_permutation_test(
            all_folds_scores,
            chance_level=0.5,
            n_permutations=n_permutations,
            cluster_threshold_config={"start": 0.1, "step": 0.1},
            alternative_hypothesis="greater"
        )

        # Create combined cluster mask
        cluster_mask = np.zeros(len(times), dtype=bool)
        if clusters and cluster_p is not None:
            for cluster, p_val in zip(clusters, cluster_p):
                if p_val < 0.05:
                    cluster_mask = np.logical_or(cluster_mask, cluster)

        comparison_data['cluster_significance_data'] = {
            'mask': cluster_mask,
            'p_values': cluster_p,
            'method': 'Cluster_Permutation'
        }

    except Exception as e:
        logger_viz_lg.error(f"Error applying statistical tests: {e}")

    return comparison_data


# =================== FONCTIONS DE COMPATIBILITÉ POUR L'ANCIEN CODE ===================

def _create_detailed_lg_comparison_plots(lg_specific_results, times, subject_id, output_dir, chance_level):
    """Create detailed plots for LG specific comparisons - backward compatibility."""
    if lg_specific_results is None or len(lg_specific_results) == 0:
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # Limit to 4 comparisons
        for i, result in enumerate(lg_specific_results[:4]):
            if i >= len(axes):
                break

            ax = axes[i]
            if result.get("scores_1d_mean") is not None and result.get("times") is not None:
                scores = result["scores_1d_mean"]
                times_res = result["times"]
                comparison_name = result.get(
                    "comparison_name", f"Comparison {i+1}")

                ax.plot(times_res, scores, linewidth=2, label=comparison_name)
                ax.axhline(y=chance_level, color='gray',
                           linestyle='--', alpha=0.7, label='Chance')

                # Add significance markers
                if result.get("fdr_significance_data") is not None:
                    fdr_data = result["fdr_significance_data"]
                    if fdr_data.get("mask") is not None:
                        sig_mask = fdr_data["mask"]
                        if len(sig_mask) == len(times_res):
                            sig_times = times_res[sig_mask]
                            if len(sig_times) > 0:
                                ax.scatter(sig_times, [max(scores) * 1.02] * len(sig_times),
                                           marker='*', color='red', s=30, label='FDR p<0.05')

                ax.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
                ax.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
                ax.set_title(comparison_name, fontsize=FONT_SIZE_TITLE)
                ax.legend(fontsize=FONT_SIZE_LEGEND-1)
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(lg_specific_results), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        detailed_filename = f"lg_detailed_comparisons_{subject_id}.png"
        detailed_path = os.path.join(output_dir, detailed_filename)
        plt.savefig(detailed_path, dpi=DPI_VALUE, bbox_inches='tight')
        plt.close()

        logger_viz_lg.info(
            "LG detailed comparison plots saved to: %s", detailed_path)

    except Exception as e:
        logger_viz_lg.error(
            "Error creating detailed LG comparison plots: %s", e, exc_info=True)


def plot_group_mean_scores_barplot_lg(subject_scores_dict, title, output_dir, score_label="AUC Score", chance_level=0.5, epoch_info_str=None):
    """Create bar plot of group mean scores for LG protocol."""
    try:
        valid_scores = {k: v for k,
                        v in subject_scores_dict.items() if pd.notna(v)}
        if len(valid_scores) == 0:
            logger_viz_lg.warning("No valid scores for group bar plot")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        subjects = list(valid_scores.keys())
        scores = list(valid_scores.values())

        bars = ax.bar(range(len(subjects)), scores, alpha=0.7,
                      color='lightblue', edgecolor='black')
        ax.axhline(y=chance_level, color='red', linestyle='--',
                   alpha=0.7, label=f'Chance ({chance_level})')

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45,
                           ha='right', fontsize=FONT_SIZE_TICK)
        ax.set_ylabel(score_label, fontsize=FONT_SIZE_LABEL)
        # Ajouter les informations d'epochs au titre si disponibles
        plot_title = title
        if epoch_info_str:
            # Parse epoch info to extract class breakdown if available
            if "Class_0:" in epoch_info_str and "Class_1:" in epoch_info_str:
                plot_title = f"{title}\n{epoch_info_str}"
            else:
                plot_title = f"{title}\n{epoch_info_str}"
        ax.set_title(plot_title, fontsize=FONT_SIZE_TITLE)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE_TICK)

        # Add statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ax.text(0.02, 0.98, f'Mean: {mean_score:.3f}\nStd: {std_score:.3f}\nN: {len(scores)}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=FONT_SIZE_LEGEND)

        plt.tight_layout()

        filename = f"lg_group_scores_barplot_{title.replace(' ', '_').replace('-', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DPI_VALUE, bbox_inches='tight')
        plt.close()

        logger_viz_lg.info("LG group bar plot saved to: %s", filepath)

    except Exception as e:
        logger_viz_lg.error(
            "Error creating LG group bar plot: %s", e, exc_info=True)


def plot_group_temporal_decoding_statistics_lg(times, mean_scores, title, output_dir,
                                               sem_scores=None, cluster_pval_map=None,
                                               fdr_mask=None, chance_level=0.5,
                                               plot_suffix="", epoch_info_str=None):
    """Plot group-level temporal decoding statistics for LG protocol."""
    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot mean curve
        ax.plot(times, mean_scores, 'b-', linewidth=3, label='Group Mean')

        # Add error bars if available
        if sem_scores is not None:
            ax.fill_between(times, mean_scores - sem_scores, mean_scores + sem_scores,
                            alpha=0.3, color='blue', label='SEM')

        # Add chance level
        ax.axhline(y=chance_level, color='gray', linestyle='--',
                   alpha=0.7, label=f'Chance ({chance_level})')

        # Add stimulus onset
        ax.axvline(x=0, color='black', linestyle=':', alpha=0.8,
                   label='Stimulus Onset')

        # Add FDR significance markers
        if fdr_mask is not None and len(fdr_mask) == len(times):
            sig_times = times[fdr_mask]
            if len(sig_times) > 0:
                ax.scatter(sig_times, [max(mean_scores) * 1.05] * len(sig_times),
                           marker='*', color='red', s=40, label='FDR p<0.05', zorder=5)

        # Add cluster significance shading
        if cluster_pval_map is not None and len(cluster_pval_map) == len(times):
            sig_cluster_mask = cluster_pval_map < 0.05
            for i, is_sig in enumerate(sig_cluster_mask):
                if is_sig:
                    ax.axvspan(times[i] - 0.002, times[i] + 0.002,
                               alpha=0.4, color='orange', zorder=1)

        ax.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
        
        # Create title with epoch information if available
        plot_title = title
        if epoch_info_str:
            plot_title = f"{title}\n{epoch_info_str}"
        ax.set_title(plot_title, fontsize=FONT_SIZE_TITLE)
        
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, alpha=0.3)

        # Add statistics text with epoch breakdown
        stats_text = f'Peak AUC: {np.max(mean_scores):.3f}\n'
        stats_text += f'Mean AUC: {np.mean(mean_scores):.3f}'
        if epoch_info_str:
            stats_text += f'\n{epoch_info_str}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=FONT_SIZE_LEGEND)

        plt.tight_layout()

        filename = f"lg_group_temporal_stats_{title.replace(' ', '_').replace('(', '').replace(')', '')}{plot_suffix}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DPI_VALUE, bbox_inches='tight')
        plt.close()

        logger_viz_lg.info(
            "LG group temporal statistics plot saved to: %s", filepath)

    except Exception as e:
        logger_viz_lg.error(
            "Error creating LG group temporal statistics plot: %s", e, exc_info=True)


def plot_group_tgm_statistics_lg(times, mean_tgm, title, output_dir, fdr_mask=None, chance_level=0.5):
    """Plot group-level TGM statistics for LG protocol."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # TGM heatmap
        ax1 = axes[0]
        im = ax1.imshow(mean_tgm, cmap='RdBu_r', aspect='auto', origin='lower',
                        extent=[times[0], times[-1], times[0], times[-1]],
                        vmin=chance_level - 0.1, vmax=chance_level + 0.1)
        ax1.set_xlabel('Testing Time (s)', fontsize=FONT_SIZE_LABEL)
        ax1.set_ylabel('Training Time (s)', fontsize=FONT_SIZE_LABEL)
        ax1.set_title(f'{title} - Mean TGM', fontsize=FONT_SIZE_TITLE)

        # Add stimulus onset lines
        ax1.axvline(x=0, color='black', linestyle=':', alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle=':', alpha=0.8)

        plt.colorbar(im, ax=ax1, label='AUC Score')

        # Add significance contours
        if fdr_mask is not None and fdr_mask.shape == mean_tgm.shape:
            ax1.contour(fdr_mask, levels=[
                        0.5], colors='black', linewidths=2, alpha=0.8)

        # Diagonal analysis
        ax2 = axes[1]
        diagonal_scores = np.diag(mean_tgm)
        ax2.plot(times, diagonal_scores, 'b-',
                 linewidth=3, label='Diagonal TGM')
        ax2.axhline(y=chance_level, color='gray', linestyle='--',
                    alpha=0.7, label=f'Chance ({chance_level})')
        ax2.axvline(x=0, color='black', linestyle=':', alpha=0.8,
                    label='Stimulus Onset')

        if fdr_mask is not None:
            diagonal_sig = np.diag(fdr_mask)
            sig_times = times[diagonal_sig]
            if len(sig_times) > 0:
                ax2.scatter(sig_times, diagonal_scores[diagonal_sig],
                            marker='*', color='red', s=40, label='Significant', zorder=5)

        ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
        ax2.set_title(f'{title} - Diagonal', fontsize=FONT_SIZE_TITLE)
        ax2.legend(fontsize=FONT_SIZE_LEGEND)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"lg_group_tgm_stats_{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=DPI_VALUE, bbox_inches='tight')
        plt.close()

        logger_viz_lg.info(
            "LG group TGM statistics plot saved to: %s", filepath)

    except Exception as e:
        logger_viz_lg.error(
            "Error creating LG group TGM statistics plot: %s", e, exc_info=True)


# === UTILITY FUNCTIONS FOR EPOCH INFORMATION ===

def _get_epochs_info_for_labels_lg(original_labels_array, label_encoder=None):
    """Get epoch count information for each class in LG protocol.
    
    Args:
        original_labels_array: Array of original labels
        label_encoder: Optional LabelEncoder object to get class names
        
    Returns:
        str: Formatted string with epoch counts per class
    """
    if original_labels_array is None or len(original_labels_array) == 0:
        return "N epochs: N/A"
    
    try:
        # Count unique labels
        unique_labels, counts = np.unique(original_labels_array, return_counts=True)
        
        # Format the information with class breakdown
        epoch_info_parts = []
        total_epochs = len(original_labels_array)
        
        for label, count in zip(unique_labels, counts):
            # Map label to meaningful name for LG protocol
            if label == 0:
                label_name = "Class_0"
            elif label == 1:
                label_name = "Class_1"
            else:
                label_name = f"Class_{label}"
            
            percentage = (count / total_epochs) * 100
            epoch_info_parts.append(f"{label_name}: {count} ({percentage:.1f}%)")
        
        epoch_info = f"N epochs: {total_epochs} total [{', '.join(epoch_info_parts)}]"
        return epoch_info
        
    except Exception as e:
        logger_viz_lg.warning(f"Error calculating epoch info: {e}")
        return f"N epochs: {len(original_labels_array)} total"


def _add_epochs_info_to_title_lg(base_title, original_labels_array, label_encoder=None):
    """Add epoch information to a plot title in LG protocol.
    
    Args:
        base_title: Base title string
        original_labels_array: Array of original labels
        label_encoder: Optional LabelEncoder object
        
    Returns:
        str: Title with epoch information appended
    """
    epoch_info = _get_epochs_info_for_labels_lg(original_labels_array, label_encoder)
    return f"{base_title}\n{epoch_info}"


def _add_epochs_info_to_legend_label_lg(base_label, original_labels_array, label_encoder=None):
    """Add epoch information to a legend label in LG protocol.
    
    Args:
        base_label: Base label string  
        original_labels_array: Array of original labels
        label_encoder: Optional LabelEncoder object
        
    Returns:
        str: Label with epoch information appended
    """
    if original_labels_array is None or len(original_labels_array) == 0:
        return f"{base_label} (N epochs: N/A)"
        
    unique_labels, counts = np.unique(original_labels_array, return_counts=True)
    total_epochs = len(original_labels_array)
    
    # Detailed format for legend showing class breakdown
    if len(unique_labels) == 2:
        # Binary classification - show class breakdown with meaningful names
        class0_count = counts[0] if unique_labels[0] == 0 else counts[1]
        class1_count = counts[1] if unique_labels[0] == 0 else counts[0]
        return f"{base_label} (N={total_epochs}: C0={class0_count}, C1={class1_count})"
    else:
        # Multi-class or single class
        class_info = ", ".join([f"C{label}={count}" for label, count in zip(unique_labels, counts)])
        return f"{base_label} (N={total_epochs}: {class_info})"

# === EXISTING FUNCTIONS ===
