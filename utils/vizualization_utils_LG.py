# Visualization utilities for LG (Local-Global) protocol

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
    Create a comprehensive dashboard of LG decoding results plots for a single subject.

    Parameters
    ----------
    main_epochs_time_points : array-like
        Time points of the epochs.
    classifier_name_for_title : str
        Name of the classifier for plot titles.
    subject_identifier : str
        Subject ID.
    group_identifier : str
        Group ID.
    output_directory_path : str
        Directory to save plots.
    CHANCE_LEVEL_AUC : float, optional
        Chance level for AUC, by default 0.5.
    protocol_type : str, optional
        Protocol type identifier, by default "LG".
    lg_main_* : various
        LG main decoding results.
    lg_specific_* : various
        LG specific comparison results.
    lg_global_effect_results : list, optional
        Global effect analysis results.
    lg_local_effect_centric_average_results_list : list, optional
        Local effect centric average results.
    """
    try:
        logger_viz_lg.info(
            "Creating LG dashboard plots for subject %s", subject_identifier)

        # Set up figure style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 24))

        # 1. Main LG decoding temporal curve
        ax1 = plt.subplot(5, 3, 1)
        if lg_main_mean_temporal_decoding_scores_1d is not None and main_epochs_time_points is not None:
            ax1.plot(main_epochs_time_points, lg_main_mean_temporal_decoding_scores_1d,
                     'b-', linewidth=2, label='LS vs LD Main')
            ax1.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')

            # Add significance markers
            if lg_main_temporal_1d_fdr_sig_data is not None and lg_main_temporal_1d_fdr_sig_data.get("mask") is not None:
                sig_mask = lg_main_temporal_1d_fdr_sig_data["mask"]
                if len(sig_mask) == len(main_epochs_time_points):
                    sig_times = main_epochs_time_points[sig_mask]
                    if len(sig_times) > 0:
                        ax1.scatter(sig_times, [max(lg_main_mean_temporal_decoding_scores_1d) * 1.02] * len(sig_times),
                                    marker='*', color='red', s=30, label='FDR p<0.05')

            if lg_main_temporal_1d_cluster_sig_data is not None and lg_main_temporal_1d_cluster_sig_data.get("mask") is not None:
                cluster_mask = lg_main_temporal_1d_cluster_sig_data["mask"]
                if len(cluster_mask) == len(main_epochs_time_points):
                    for i, is_sig in enumerate(cluster_mask):
                        if is_sig:
                            ax1.axvspan(main_epochs_time_points[i] - 0.002, main_epochs_time_points[i] + 0.002,
                                        alpha=0.3, color='orange')

            ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax1.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax1.set_title(
                f'LG Main Decoding (LS vs LD)\n{subject_identifier}', fontsize=FONT_SIZE_TITLE)
            ax1.legend(fontsize=FONT_SIZE_LEGEND)
            ax1.grid(True, alpha=0.3)

        # 2. Cross-validation scores distribution
        ax2 = plt.subplot(5, 3, 2)
        if lg_main_cross_validation_global_scores is not None and len(lg_main_cross_validation_global_scores) > 0:
            ax2.hist(lg_main_cross_validation_global_scores, bins=10,
                     alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=CHANCE_LEVEL_AUC, color='red',
                        linestyle='--', alpha=0.7, label='Chance')
            ax2.axvline(x=np.mean(lg_main_cross_validation_global_scores), color='blue', linestyle='-',
                        linewidth=2, label=f'Mean: {np.mean(lg_main_cross_validation_global_scores):.3f}')
            ax2.set_xlabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax2.set_ylabel('Frequency', fontsize=FONT_SIZE_LABEL)
            ax2.set_title(
                f'LG CV Scores Distribution\n{classifier_name_for_title}', fontsize=FONT_SIZE_TITLE)
            ax2.legend(fontsize=FONT_SIZE_LEGEND)
            ax2.grid(True, alpha=0.3)

        # 3. Temporal Generalization Matrix
        ax3 = plt.subplot(5, 3, 3)
        if lg_main_mean_temporal_generalization_matrix_scores is not None and lg_main_mean_temporal_generalization_matrix_scores.ndim == 2:
            tgm = lg_main_mean_temporal_generalization_matrix_scores
            im = ax3.imshow(tgm, cmap='RdBu_r', aspect='auto', origin='lower',
                            extent=[main_epochs_time_points[0], main_epochs_time_points[-1],
                                    main_epochs_time_points[0], main_epochs_time_points[-1]],
                            vmin=CHANCE_LEVEL_AUC - 0.1, vmax=CHANCE_LEVEL_AUC + 0.1)
            ax3.set_xlabel('Testing Time (s)', fontsize=FONT_SIZE_LABEL)
            ax3.set_ylabel('Training Time (s)', fontsize=FONT_SIZE_LABEL)
            ax3.set_title('LG Temporal Generalization Matrix',
                          fontsize=FONT_SIZE_TITLE)
            plt.colorbar(im, ax=ax3, label='AUC Score')

            # Add significance contours if available
            if lg_main_tgm_fdr_sig_data is not None and lg_main_tgm_fdr_sig_data.get("mask") is not None:
                sig_mask = lg_main_tgm_fdr_sig_data["mask"]
                if sig_mask.shape == tgm.shape:
                    ax3.contour(sig_mask, levels=[
                                0.5], colors='black', linewidths=1, alpha=0.7)

        # 4. LG Specific Comparisons
        ax4 = plt.subplot(5, 3, 4)
        if lg_specific_comparison_results is not None and len(lg_specific_comparison_results) > 0:
            colors = plt.cm.Set1(np.linspace(
                0, 1, len(lg_specific_comparison_results)))
            for i, result in enumerate(lg_specific_comparison_results):
                if result.get("scores_1d_mean") is not None and result.get("times") is not None:
                    times = result["times"]
                    scores = result["scores_1d_mean"]
                    label = result.get("comparison_name", f"Comparison {i+1}")
                    ax4.plot(times, scores,
                             color=colors[i], linewidth=2, label=label)

            ax4.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')
            ax4.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax4.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax4.set_title('LG Specific Comparisons', fontsize=FONT_SIZE_TITLE)
            ax4.legend(fontsize=FONT_SIZE_LEGEND - 2,
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)

        # 5. Mean of specific scores with error bars
        ax5 = plt.subplot(5, 3, 5)
        if lg_mean_of_specific_scores_1d is not None and main_epochs_time_points is not None:
            ax5.plot(main_epochs_time_points, lg_mean_of_specific_scores_1d, 'g-', linewidth=2,
                     label='Mean LG Specific')

            if lg_sem_of_specific_scores_1d is not None:
                ax5.fill_between(main_epochs_time_points,
                                 lg_mean_of_specific_scores_1d - lg_sem_of_specific_scores_1d,
                                 lg_mean_of_specific_scores_1d + lg_sem_of_specific_scores_1d,
                                 alpha=0.3, color='green')

            ax5.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')

            # Add significance markers for mean specific
            if lg_mean_specific_fdr_sig_data is not None and lg_mean_specific_fdr_sig_data.get("mask") is not None:
                sig_mask = lg_mean_specific_fdr_sig_data["mask"]
                if len(sig_mask) == len(main_epochs_time_points):
                    sig_times = main_epochs_time_points[sig_mask]
                    if len(sig_times) > 0:
                        ax5.scatter(sig_times, [max(lg_mean_of_specific_scores_1d) * 1.02] * len(sig_times),
                                    marker='*', color='red', s=30, label='FDR p<0.05')

            ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax5.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax5.set_title('Mean LG Specific Scores', fontsize=FONT_SIZE_TITLE)
            ax5.legend(fontsize=FONT_SIZE_LEGEND)
            ax5.grid(True, alpha=0.3)

        # 6. Global Effect Results
        ax6 = plt.subplot(5, 3, 6)
        if lg_global_effect_results is not None and len(lg_global_effect_results) > 0:
            for i, result in enumerate(lg_global_effect_results):
                if result.get("scores_1d_mean") is not None and result.get("times") is not None:
                    times = result["times"]
                    scores = result["scores_1d_mean"]
                    label = result.get("comparison_name",
                                       f"Global Effect {i+1}")
                    ax6.plot(times, scores, linewidth=2, label=label)

                    # Add significance markers
                    if result.get("fdr_significance_data") is not None and result["fdr_significance_data"].get("mask") is not None:
                        sig_mask = result["fdr_significance_data"]["mask"]
                        if len(sig_mask) == len(times):
                            sig_times = times[sig_mask]
                            if len(sig_times) > 0:
                                ax6.scatter(sig_times, [max(scores) * 1.02] * len(sig_times),
                                            marker='*', color='red', s=20)

            ax6.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')
            ax6.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax6.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax6.set_title('LG Global Effect (GS vs GD)',
                          fontsize=FONT_SIZE_TITLE)
            ax6.legend(fontsize=FONT_SIZE_LEGEND)
            ax6.grid(True, alpha=0.3)

        # 7. Local Effect Centric Averages
        ax7 = plt.subplot(5, 3, 7)
        if lg_local_effect_centric_average_results_list is not None and len(lg_local_effect_centric_average_results_list) > 0:
            colors = ['purple', 'orange', 'brown', 'pink']
            for i, effect_result in enumerate(lg_local_effect_centric_average_results_list):
                if effect_result.get("average_scores_1d") is not None and main_epochs_time_points is not None:
                    scores = effect_result["average_scores_1d"]
                    effect_type = effect_result.get(
                        "effect_type", f"Local Effect {i+1}")
                    color = colors[i % len(colors)]

                    ax7.plot(main_epochs_time_points, scores,
                             color=color, linewidth=2, label=effect_type)

                    if effect_result.get("sem_scores_1d") is not None:
                        sem = effect_result["sem_scores_1d"]
                        ax7.fill_between(main_epochs_time_points, scores - sem, scores + sem,
                                         alpha=0.3, color=color)

                    # Add significance markers
                    if effect_result.get("fdr_sig_data") is not None and effect_result["fdr_sig_data"].get("mask") is not None:
                        sig_mask = effect_result["fdr_sig_data"]["mask"]
                        if len(sig_mask) == len(main_epochs_time_points):
                            sig_times = main_epochs_time_points[sig_mask]
                            if len(sig_times) > 0:
                                ax7.scatter(sig_times, [max(scores) * 1.02] * len(sig_times),
                                            marker='*', color='red', s=20)

            ax7.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')
            ax7.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax7.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax7.set_title('LG Local Effect Averages', fontsize=FONT_SIZE_TITLE)
            ax7.legend(fontsize=FONT_SIZE_LEGEND)
            ax7.grid(True, alpha=0.3)

        # 8. Performance metrics summary
        ax8 = plt.subplot(5, 3, 8)
        if lg_main_decoding_global_metrics_for_plot is not None and len(lg_main_decoding_global_metrics_for_plot) > 0:
            metrics_names = list(
                lg_main_decoding_global_metrics_for_plot.keys())
            metrics_values = list(
                lg_main_decoding_global_metrics_for_plot.values())

            bars = ax8.bar(range(len(metrics_names)),
                           metrics_values, alpha=0.7, color='lightcoral')
            ax8.set_xticks(range(len(metrics_names)))
            ax8.set_xticklabels(metrics_names, rotation=45,
                                ha='right', fontsize=FONT_SIZE_TICK)
            ax8.set_ylabel('Score', fontsize=FONT_SIZE_LABEL)
            ax8.set_title('LG Global Metrics', fontsize=FONT_SIZE_TITLE)
            ax8.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                if pd.notna(value):
                    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE_TICK)

        # 9. All folds temporal scores (if available)
        ax9 = plt.subplot(5, 3, 9)
        if lg_main_temporal_scores_1d_all_folds is not None and main_epochs_time_points is not None:
            # (n_folds, n_timepoints)
            if lg_main_temporal_scores_1d_all_folds.ndim == 2:
                for fold_idx in range(lg_main_temporal_scores_1d_all_folds.shape[0]):
                    ax9.plot(main_epochs_time_points, lg_main_temporal_scores_1d_all_folds[fold_idx, :],
                             alpha=0.5, linewidth=1, label=f'Fold {fold_idx+1}')

                # Plot mean on top
                if lg_main_mean_temporal_decoding_scores_1d is not None:
                    ax9.plot(main_epochs_time_points, lg_main_mean_temporal_decoding_scores_1d,
                             'black', linewidth=3, label='Mean')

            ax9.axhline(y=CHANCE_LEVEL_AUC, color='gray',
                        linestyle='--', alpha=0.7, label='Chance')
            ax9.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
            ax9.set_ylabel('AUC Score', fontsize=FONT_SIZE_LABEL)
            ax9.set_title('LG All CV Folds', fontsize=FONT_SIZE_TITLE)
            ax9.legend(fontsize=FONT_SIZE_LEGEND-2,
                       bbox_to_anchor=(1.05, 1), loc='upper left')
            ax9.grid(True, alpha=0.3)

        # Adjust layout and save
        plt.tight_layout()

        # Save the main dashboard
        dashboard_filename = f"lg_decoding_dashboard_{subject_identifier}_{group_identifier}_{classifier_name_for_title}.png"
        dashboard_path = os.path.join(
            output_directory_path, dashboard_filename)
        plt.savefig(dashboard_path, dpi=DPI_VALUE, bbox_inches='tight')
        logger_viz_lg.info("LG Dashboard saved to: %s", dashboard_path)

        plt.close()

        # Create additional detailed plots
        _create_detailed_lg_comparison_plots(
            lg_specific_comparison_results, main_epochs_time_points,
            subject_identifier, output_directory_path, CHANCE_LEVEL_AUC
        )

        logger_viz_lg.info(
            "LG dashboard plots creation completed for subject %s", subject_identifier)

    except Exception as e:
        logger_viz_lg.error("Error creating LG dashboard plots for subject %s: %s",
                            subject_identifier, e, exc_info=True)


def _create_detailed_lg_comparison_plots(lg_specific_results, times, subject_id, output_dir, chance_level):
    """Create detailed plots for LG specific comparisons."""
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

                if result.get("cluster_significance_data") is not None:
                    cluster_data = result["cluster_significance_data"]
                    if cluster_data.get("mask") is not None:
                        cluster_mask = cluster_data["mask"]
                        if len(cluster_mask) == len(times_res):
                            for j, is_sig in enumerate(cluster_mask):
                                if is_sig:
                                    ax.axvspan(times_res[j] - 0.002, times_res[j] + 0.002,
                                               alpha=0.3, color='orange')

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


def plot_group_mean_scores_barplot_lg(subject_scores_dict, title, output_dir, score_label="AUC Score", chance_level=0.5):
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
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
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
                                               plot_suffix=""):
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
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'N subjects: {mean_scores.shape[0] if hasattr(mean_scores, "shape") else "N/A"}\n'
        stats_text += f'Peak AUC: {np.max(mean_scores):.3f}\n'
        stats_text += f'Mean AUC: {np.mean(mean_scores):.3f}'

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
