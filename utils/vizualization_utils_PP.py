import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import scipy.stats


from config.decoding_config import CHANCE_LEVEL_AUC

logger_viz_utils = logging.getLogger(__name__)

# --- GROUP BAR PLOT ---


def plot_group_mean_scores_barplot(
    subject_to_score_mapping,
    group_identifier_for_plot_title,
    output_directory_path=None,
    n_folds=None,
    score_metric_name="ROC AUC",
    chance_level_value=CHANCE_LEVEL_AUC,
):
    """Plot a bar chart of scores for each subject in a group."""
    if not isinstance(subject_to_score_mapping, dict):
        logger_viz_utils.error(
            "subject_to_score_mapping must be a dictionary.")
        return
    valid_scores = {k: v for k, v in subject_to_score_mapping.items()
                    if v is not None and not np.isnan(v)}
    if not valid_scores:
        logger_viz_utils.warning(
            "No valid scores for barplot: %s", group_identifier_for_plot_title)
        return

    subject_ids = list(valid_scores.keys())
    scores = list(valid_scores.values())
    n_subjects = len(subject_ids)

    logger_viz_utils.info("Plotting group barplot for %s (%d valid subjects)",
                          group_identifier_for_plot_title, n_subjects)
    plt.switch_backend("Agg")
    fig_width = max(8, n_subjects * 0.5 + 2)  # Dynamic width
    fig_bar, ax_bar = plt.subplots(figsize=(fig_width, 7))

    bars = ax_bar.bar(range(n_subjects), scores, color="teal", alpha=0.7,
                      label=f"Subject {score_metric_name}")
    mean_score = np.mean(scores) if scores else np.nan
    ax_bar.axhline(mean_score, color="darkred", ls="--",
                   label=f"Group Mean: {mean_score:.3f}")
    ax_bar.axhline(chance_level_value, color="black", ls=":",
                   label=f"Chance ({chance_level_value})")

    ax_bar.set_ylabel(f"Score ({score_metric_name})")
    ax_bar.set_xlabel("Subject ID")
    # Title with optional CV folds info
    title_str = f"Individual subject performance: {group_identifier_for_plot_title}"
    if n_folds:
        title_str += f" ({n_folds}-fold CV)"
    ax_bar.set_title(title_str, fontweight="bold")
    ax_bar.set_xticks(range(n_subjects))
    ax_bar.set_xticklabels(subject_ids, rotation=60, ha="right", fontsize=10)

    min_s_val = np.min(scores) if scores else chance_level_value - 0.1
    max_s_val = np.max(scores) if scores else chance_level_value + 0.1
    ax_bar.set_ylim(min(chance_level_value - 0.15, min_s_val - 0.05),
                    max(1.0, max_s_val + 0.1))

    for bar_item in bars:  # Add text labels on bars
        height = bar_item.get_height()
        ax_bar.text(bar_item.get_x() + bar_item.get_width() / 2.0, height + 0.01,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    ax_bar.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax_bar.grid(True, axis="y", ls=":", alpha=0.6)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if output_directory_path:
        os.makedirs(output_directory_path, exist_ok=True)
        safe_fname = "".join(c if c.isalnum() else "_" for c in
                             group_identifier_for_plot_title.replace(" ", "_"))
        save_path = os.path.join(output_directory_path,
                                 f"group_summary_scores_barplot_{safe_fname}.png")
        try:
            fig_bar.savefig(save_path, dpi=150)
            logger_viz_utils.info("Barplot saved: %s", save_path)
        except Exception as e_save:
            logger_viz_utils.error(
                "Failed to save barplot '%s': %s", save_path, e_save)
    plt.close(fig_bar)

# --- Plotting functions for group stats ---


def plot_group_temporal_decoding_statistics(
    time_points_array,
    mean_group_temporal_scores,
    group_identifier_for_plot,
    output_directory_path,
    std_error_group_temporal_scores=None,
    cluster_p_value_map_1d=None,
    fdr_significance_mask_1d=None,
    fdr_test_info=None,  # New parameter for FDR test information
    chance_level=CHANCE_LEVEL_AUC,
):
    """Plots group-level temporal decoding stats with significance overlays."""
    # Parameter Validation
    if not isinstance(time_points_array, np.ndarray) or time_points_array.ndim != 1:
        logger_viz_utils.error(
            "time_points_array must be a 1D NumPy array.")
        return
    if not isinstance(mean_group_temporal_scores, np.ndarray) or \
       mean_group_temporal_scores.shape != time_points_array.shape:
        logger_viz_utils.error(
            "mean_group_temporal_scores mismatch with time_points_array.")
        return
    if not isinstance(output_directory_path, str):
        logger_viz_utils.error("output_directory_path must be a string.")
        return

    logger_viz_utils.info(
        "Plotting group temporal decoding statistics for: %s", group_identifier_for_plot
    )
    plt.switch_backend("Agg")  # Ensure non-interactive backend
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(time_points_array, mean_group_temporal_scores,
            label="Mean Group AUC", color="black", linewidth=2.5)

    if std_error_group_temporal_scores is not None:
        if std_error_group_temporal_scores.shape == time_points_array.shape:
            ax.fill_between(
                time_points_array,
                mean_group_temporal_scores - std_error_group_temporal_scores,
                mean_group_temporal_scores + std_error_group_temporal_scores,
                color="gray", alpha=0.25, label="SEM",
            )
        else:
            logger_viz_utils.warning(
                "SEM shape mismatch, not plotting SEM.")

    ax.axhline(chance_level, color="dimgray", linestyle="--",
               linewidth=1.2, label=f"Chance ({chance_level})")
    if time_points_array.size > 0:
        ax.axvline(0, color="firebrick", linestyle=":",
                   linewidth=1.2, label="Stimulus Onset")

    # Significance bar plotting
    finite_scores = mean_group_temporal_scores[np.isfinite(
        mean_group_temporal_scores)]
    min_score_sig = np.min(
        finite_scores) if finite_scores.size > 0 else chance_level
    sig_y_base = min(min_score_sig, chance_level)
    sig_bar_offset = 0.03
    sig_bar_height = 0.01
    current_y_pos = sig_y_base - sig_bar_offset

    if fdr_significance_mask_1d is not None and \
       fdr_significance_mask_1d.shape == time_points_array.shape and \
       np.any(fdr_significance_mask_1d):
        # Create FDR label with test information
        fdr_label = "FDR p<0.05"
        if fdr_test_info is not None:
            test_type = fdr_test_info.get("test_type", "unknown")
            if test_type == "adaptive":
                n_ttest = fdr_test_info.get("ttest_features", 0)
                n_wilcoxon = fdr_test_info.get("wilcoxon_features", 0)
                fdr_label = f"FDR p<0.05 (adaptive: {n_ttest}t-test, {n_wilcoxon}Wilcoxon)"
            elif test_type == "ttest":
                fdr_label = "FDR p<0.05 (t-test)"
            elif test_type == "wilcoxon":
                fdr_label = "FDR p<0.05 (Wilcoxon)"
        
        ax.fill_between(
            time_points_array, current_y_pos, current_y_pos + sig_bar_height,
            where=fdr_significance_mask_1d, color="deepskyblue", alpha=0.7,
            step="mid", label=fdr_label,
        )
        current_y_pos -= (sig_bar_height + 0.005)

    if cluster_p_value_map_1d is not None and \
       cluster_p_value_map_1d.shape == time_points_array.shape and \
       np.any(cluster_p_value_map_1d < 0.05):
        ax.fill_between(
            time_points_array, current_y_pos, current_y_pos + sig_bar_height,
            where=(cluster_p_value_map_1d < 0.05), color="orangered", alpha=0.7,
            step="mid", label="Cluster Perm. p<0.05",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Decoding Score (ROC AUC)")
    ax.set_title(
        f"Group Temporal Decoding: {group_identifier_for_plot}",
        fontsize=15, fontweight="bold",
    )
    ymin_plot = np.min(
        finite_scores) if finite_scores.size > 0 else chance_level - 0.1
    ymax_plot = np.max(
        finite_scores) if finite_scores.size > 0 else chance_level + 0.1
    ax.set_ylim(
        min(current_y_pos - sig_bar_height * 0.5, ymin_plot - 0.05),
        max(1.01, ymax_plot + 0.05),
    )
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    safe_fname = "".join(
        c if c.isalnum() else "_" for c in group_identifier_for_plot.replace(" ", "_")
    )
    save_path = os.path.join(
        output_directory_path, f"group_temporal_decoding_stats_{safe_fname}.png"
    )
    try:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger_viz_utils.info("Group temporal plot saved: %s", save_path)
    except Exception as e:
        logger_viz_utils.error(
            "Failed to save group temporal plot '%s': %s", save_path, e, exc_info=True
        )
    plt.close(fig)


def plot_group_tgm_statistics(
    mean_group_tgm_scores,
    time_points_tgm_array,
    significant_cluster_masks_tgm,  # Expected None for TGM as per original comments
    cluster_p_values_tgm,          # Expected None
    group_identifier_for_plot,
    output_directory_path,
    observed_t_values_map_tgm=None,
    fdr_significance_mask_tgm=None,
    fdr_test_info_tgm=None,  # New parameter for TGM FDR test information
    chance_level=0.5,
    plot_vmin=None,
    plot_vmax=None,
):
    """Plots group-level TGM statistics with FDR significance overlay."""
    # Parameter Validation
    if not isinstance(mean_group_tgm_scores, np.ndarray) or mean_group_tgm_scores.ndim != 2:
        logger_viz_utils.error(
            "mean_group_tgm_scores must be a 2D NumPy array.")
        return
    if not isinstance(time_points_tgm_array, np.ndarray) or time_points_tgm_array.ndim != 1:
        logger_viz_utils.error(
            "time_points_tgm_array must be a 1D NumPy array.")
        return
    if mean_group_tgm_scores.shape[0] != time_points_tgm_array.size or \
       mean_group_tgm_scores.shape[1] != time_points_tgm_array.size:
        logger_viz_utils.error(
            "TGM scores dimensions mismatch with time_points_tgm_array.")
        return
    if not isinstance(output_directory_path, str):
        logger_viz_utils.error("output_directory_path must be a string.")
        return

    logger_viz_utils.info(
        "Plotting group TGM statistics for: %s", group_identifier_for_plot
    )
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    data_to_plot = (observed_t_values_map_tgm if observed_t_values_map_tgm is not None
                    else mean_group_tgm_scores)
    vmin_curr, vmax_curr = plot_vmin, plot_vmax
    cmap_use = "RdBu_r"
    cbar_label = "t-value vs chance" if observed_t_values_map_tgm is not None else "Mean AUC"

    if observed_t_values_map_tgm is not None:
        if vmin_curr is None or vmax_curr is None:
            finite_t = observed_t_values_map_tgm[np.isfinite(
                observed_t_values_map_tgm)]
            abs_max_t = np.max(np.abs(finite_t)) if finite_t.size > 0 else 3.0
            vmin_curr, vmax_curr = -abs_max_t, abs_max_t
    else:  # AUC scores
        if vmin_curr is None:
            vmin_curr = 0.35
        if vmax_curr is None:
            vmax_curr = 0.90

    if time_points_tgm_array.size == 0:
        logger_viz_utils.error("Time points array is empty for TGM plot of %s.",
                               group_identifier_for_plot)
        plt.close(fig)
        return

    im = ax.imshow(
        data_to_plot, interpolation="lanczos", origin="lower", cmap=cmap_use,
        extent=time_points_tgm_array[[0, -1, 0, -1]],
        vmin=vmin_curr, vmax=vmax_curr, aspect="auto",
    )
    ax.set_xlabel("Testing Time (s)")
    ax.set_ylabel("Training Time (s)")
    plot_title = f"Group TGM: {group_identifier_for_plot}"
    sig_info_parts = []
    if fdr_significance_mask_tgm is not None and np.any(fdr_significance_mask_tgm):
        # Add FDR test information to the title
        fdr_info = "FDR sig. hatched"
        if fdr_test_info_tgm is not None:
            test_type = fdr_test_info_tgm.get("test_type", "unknown")
            if test_type == "adaptive":
                n_ttest = fdr_test_info_tgm.get("ttest_features", 0)
                n_wilcoxon = fdr_test_info_tgm.get("wilcoxon_features", 0)
                fdr_info = f"FDR sig. hatched (adaptive: {n_ttest}t-test, {n_wilcoxon}Wilcoxon)"
            elif test_type == "ttest":
                fdr_info = "FDR sig. hatched (t-test)"
            elif test_type == "wilcoxon":
                fdr_info = "FDR sig. hatched (Wilcoxon)"
        sig_info_parts.append(fdr_info)
    if sig_info_parts:
        plot_title += f"\n({', '.join(sig_info_parts)}, p<0.05)"
    ax.set_title(plot_title)

    ax.axvline(0, color="k", linestyle=":", lw=0.8)
    ax.axhline(0, color="k", linestyle=":", lw=0.8)

    if fdr_significance_mask_tgm is not None and \
       fdr_significance_mask_tgm.shape == data_to_plot.shape and \
       np.any(fdr_significance_mask_tgm):
        X_coords, Y_coords = np.meshgrid(
            time_points_tgm_array, time_points_tgm_array)
        ax.contourf(
            X_coords, Y_coords, fdr_significance_mask_tgm,
            levels=[0.5, 1.5], colors="none", hatches=["///"], alpha=0.3,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    if observed_t_values_map_tgm is None:  # Plotting AUCs
        cbar.ax.axhline(chance_level, color="black", linestyle="--", lw=1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    safe_fname = "".join(
        c if c.isalnum() else "_" for c in group_identifier_for_plot.replace(" ", "_")
    )
    save_path = os.path.join(
        output_directory_path, f"group_TGM_stats_{safe_fname}.png"
    )
    try:
        fig.savefig(save_path, dpi=200)
        logger_viz_utils.info("Group TGM plot saved: %s", save_path)
    except Exception as e_save:
        logger_viz_utils.error(
            "Failed to save group TGM plot '%s': %s", save_path, e_save, exc_info=True
        )
    plt.close(fig)

# === PLOTTING HELPER FUNCTIONS ===


def _set_ticks(times_array):
    """
    Generate optimized tick positions and labels for time axes.

    Aims for a reasonable number of ticks, prioritizing key time points like 0,
    and ensuring start/end times are shown if space allows.

    Args:
        times_array (np.ndarray): Array of time points.

    Returns:
        tuple: (tick_positions, tick_labels)
            - tick_positions (np.ndarray): Positions for ticks.
            - tick_labels (list): Labels for the ticks (often in ms).
    """
    if not isinstance(times_array, np.ndarray) or times_array.ndim != 1 or times_array.size == 0:
        logger_viz_utils.warning(
            "Invalid 'times_array' for _set_ticks. Returning empty.")
        return np.array([]), []

    min_time, max_time = np.min(times_array), np.max(times_array)
    # Initial ticks, attempting to cover the range
    ticks = np.arange(np.ceil(min_time * 10) / 10.0,
                      np.floor(max_time * 10) / 10.0 + 0.1, 0.1)

    if not ticks.size:  # Handle very short time windows
        ticks = np.array([min_time, max_time]
                         ) if min_time != max_time else np.array([min_time])
    ticks = np.round(ticks * 10.0) / 10.0  # Round to nearest 0.1s

    # Ensure min_time and max_time are included if not close to existing ticks
    if not np.isclose(ticks[0], min_time) and min_time < ticks[0]:
        ticks = np.insert(ticks, 0, min_time)
    if not np.isclose(ticks[-1], max_time) and max_time > ticks[-1]:
        ticks = np.append(ticks, max_time)

    ticks = np.unique(np.round(ticks * 1000) / 1000)  # Round to ms, unique

    # If too many ticks, reduce to a fixed number (e.g., 7)
    if len(ticks) > 10:
        ticks = np.linspace(min_time, max_time, 7)
        ticks = np.round(ticks * 100) / 100  # Round to 0.01s
    ticks = np.unique(ticks)  # Ensure uniqueness after linspace

    tick_labels = []
    if len(ticks) > 0:
        tick_labels.append(f"{int(np.round(ticks[0] * 1000))}")
        for _ in ticks[1:-1]:  # Empty labels for intermediate ticks
            tick_labels.append("")
        if len(ticks) > 1:
            tick_labels.append(f"{int(np.round(ticks[-1] * 1000))}")
    return ticks, tick_labels


def pretty_plot(ax_obj=None):
    """
    Apply a 'pretty' style to a Matplotlib Axes object.

    Modifies spine visibility, tick colors, and label colors for a cleaner look.

    Args:
        ax_obj (matplotlib.axes.Axes, optional): The Axes object to style.
            If None, `plt.gca()` is used.

    Returns:
        matplotlib.axes.Axes: The styled Axes object.
    """
    if ax_obj is None:
        ax_obj = plt.gca()
    if not isinstance(ax_obj, plt.Axes):
        logger_viz_utils.warning(
            "Invalid 'ax_obj' for pretty_plot. Must be Axes.")
        return ax_obj  # Return as is

    ax_obj.tick_params(colors="dimgray")
    ax_obj.xaxis.label.set_color("dimgray")
    ax_obj.yaxis.label.set_color("dimgray")
    try:
        ax_obj.zaxis.label.set_color("dimgray")  # For 3D plots
    except AttributeError:
        pass  # Not a 3D plot

    try:
        ax_obj.xaxis.set_ticks_position("bottom")
        ax_obj.yaxis.set_ticks_position("left")
    except ValueError:  # Some plot types might not support this (e.g., polar)
        logger_viz_utils.debug(
            "Could not set tick positions for this plot type.")

    ax_obj.spines["left"].set_color("dimgray")
    ax_obj.spines["bottom"].set_color("dimgray")
    ax_obj.spines["right"].set_visible(False)
    ax_obj.spines["top"].set_visible(False)
    return ax_obj


def pretty_colorbar(image_mappable=None, ax_obj=None, colorbar_ticks=None,
                    colorbar_ticklabels=None, num_colorbar_ticks=3, **kwargs):
    """
    Create and style a 'pretty' colorbar for an image.

    Args:
        image_mappable (matplotlib.image.AxesImage, optional): The image mappable.
            If None, tries to find one in `ax_obj`.
        ax_obj (matplotlib.axes.Axes, optional): Axes to attach colorbar to.
            If None, `plt.gca()` is used.
        colorbar_ticks (list, optional): Positions for colorbar ticks. Auto-calculated if None.
        colorbar_ticklabels (list, optional): Labels for colorbar ticks. Auto-formatted if None.
        num_colorbar_ticks (int): Number of ticks if auto-calculated.
        **kwargs: Additional arguments for `plt.colorbar()`.

    Returns:
        matplotlib.colorbar.Colorbar or None: The styled Colorbar, or None on error.
    """
    if ax_obj is None:
        ax_obj = plt.gca()
    if not isinstance(ax_obj, plt.Axes):
        logger_viz_utils.error(
            "Invalid 'ax_obj' for pretty_colorbar. Must be Axes.")
        return None

    if image_mappable is None:
        images_in_axes = [obj for obj in ax_obj.get_children()
                          if isinstance(obj, plt.matplotlib.image.AxesImage)]
        if images_in_axes:
            image_mappable = images_in_axes[0]
        else:
            logger_viz_utils.error(
                "No image found in axes to create a colorbar for.")
            return None
    if not hasattr(image_mappable, 'get_clim'):
        logger_viz_utils.error("Invalid 'image_mappable' for pretty_colorbar.")
        return None

    current_clim = image_mappable.get_clim()
    if colorbar_ticks is None:
        if None in current_clim or np.isclose(current_clim[0], current_clim[1]):
            colorbar_ticks = np.array([0.0, 0.5, 1.0])  # Default if clim flat
        else:
            colorbar_ticks = np.linspace(
                current_clim[0], current_clim[1], num_colorbar_ticks)

    cb = plt.colorbar(image_mappable, ax=ax_obj,
                      ticks=colorbar_ticks, **kwargs)
    if colorbar_ticklabels is None:
        colorbar_ticklabels = [f"{val:.2f}" for val in colorbar_ticks]

    if len(colorbar_ticklabels) == len(colorbar_ticks):
        cb.ax.set_yticklabels(colorbar_ticklabels, color="dimgray")
    else:
        cb.ax.set_yticklabels(
            [f"{val:.2f}" for val in colorbar_ticks], color="dimgray")

    cb.ax.yaxis.label.set_color("dimgray")
    cb.outline.set_edgecolor("dimgray")
    return cb


def pretty_gat(
    scores_matrix,
    times_array=None,
    chance_level=0.5,
    ax_obj=None,
    cluster_sig_masks_list=None,
    fdr_sig_mask_matrix=None,
    colormap="RdBu_r",
    color_limits=None,
    show_colorbar=True,
    x_axis_label="Testing Time (s)",
    y_axis_label="Training Time (s)",
    sampling_frequency=None,
    diagonal_line_color=None,
    test_times_array=None
):
    """
    Plot a Temporal Generalization Matrix (TGM) with enhancements.

    Args:
        scores_matrix (np.ndarray): 2D TGM scores (train_time x test_time).
        times_array (np.ndarray, optional): Time points for training axis.
        chance_level (float): Chance level for centering colormap.
        ax_obj (matplotlib.axes.Axes, optional): Axes to plot on.
        cluster_sig_masks_list (list of np.ndarray, optional): Masks for significant clusters.
        fdr_sig_mask_matrix (np.ndarray, optional): Mask for FDR significance.
        colormap (str): Colormap name.
        color_limits (tuple/list, optional): Color limits (vmin, vmax) or single spread value.
        show_colorbar (bool): Whether to draw a colorbar.
        x_axis_label (str): Label for the x-axis.
        y_axis_label (str): Label for the y-axis.
        sampling_frequency (float, optional): Sampling frequency if `times_array` not provided.
        diagonal_line_color (str, optional): Color for diagonal line.
        test_times_array (np.ndarray, optional): Time points for testing axis if different.

    Returns:
        matplotlib.axes.Axes: The Axes object with the TGM plot.
    """
    scores_matrix = np.array(scores_matrix)
    if scores_matrix.ndim != 2:
        logger_viz_utils.error("pretty_gat: scores_matrix must be a 2D array.")
        if ax_obj:
            ax_obj.text(0.5, 0.5, "Invalid scores_matrix shape",
                        ha="center", va="center")
            return ax_obj
        return plt.gca()  # Return current axes to avoid crash

    if times_array is None:
        if sampling_frequency is None:
            logger_viz_utils.warning(
                "pretty_gat: times_array or sampling_frequency needed.")
            times_array = np.arange(scores_matrix.shape[0])
        else:
            times_array = np.arange(
                scores_matrix.shape[0]) / float(sampling_frequency)

    if test_times_array is None:
        if scores_matrix.shape[1] == scores_matrix.shape[0]:  # Square GAT
            test_times_array = times_array
        else:  # Non-square GAT, attempt to infer test times
            if times_array.size > 0:
                test_times_array = np.linspace(np.min(times_array), np.max(times_array),
                                               scores_matrix.shape[1])
            elif sampling_frequency:
                test_times_array = np.arange(
                    scores_matrix.shape[1]) / float(sampling_frequency)
            else:
                test_times_array = np.arange(scores_matrix.shape[1])

    # Determine color limits
    vmin, vmax = 0, 1  # Default
    if color_limits is None:
        if np.any(~np.isnan(scores_matrix)):
            spread = np.percentile(
                np.abs(scores_matrix - chance_level)[~np.isnan(scores_matrix)], 98)
        else:
            spread = 0.15 if chance_level == 0.5 else 1.0
        vmin, vmax = chance_level - spread, chance_level + spread
    elif len(color_limits) == 1:
        vmin, vmax = chance_level - \
            color_limits[0], chance_level + color_limits[0]
    else:
        vmin, vmax = color_limits
    if np.isclose(vmin, vmax):
        vmin, vmax = chance_level - 0.1, chance_level + 0.1

    extent = [
        np.min(test_times_array) if test_times_array.size > 0 else 0,
        np.max(
            test_times_array) if test_times_array.size > 0 else scores_matrix.shape[1],
        np.min(times_array) if times_array.size > 0 else 0,
        np.max(
            times_array) if times_array.size > 0 else scores_matrix.shape[0],
    ]

    if ax_obj is None:
        ax_obj = plt.gca()
    im = ax_obj.matshow(scores_matrix, extent=extent, cmap=colormap, origin="lower",
                        vmin=vmin, vmax=vmax, aspect="auto")

    # Significance overlays
    if cluster_sig_masks_list:
        X_clu, Y_clu = np.meshgrid(test_times_array, times_array)
        for mask in cluster_sig_masks_list:
            if mask is not None and mask.shape == scores_matrix.shape and np.any(mask):
                ax_obj.contour(X_clu, Y_clu, mask, colors="black", levels=[0.5],
                               linestyles="dotted", linewidths=1.5, corner_mask=False)

    if fdr_sig_mask_matrix is not None and fdr_sig_mask_matrix.shape == scores_matrix.shape \
       and np.any(fdr_sig_mask_matrix):
        X_fdr, Y_fdr = np.meshgrid(test_times_array, times_array)
        ax_obj.contourf(X_fdr, Y_fdr, fdr_sig_mask_matrix, levels=[0.5, 1.5],
                        colors="none", hatches=["///"], alpha=0.3)

    ax_obj.axhline(0, color="k", lw=0.5, linestyle=":")
    ax_obj.axvline(0, color="k", lw=0.5, linestyle=":")

    if show_colorbar:
        cb_ticks_vals = ([vmin, chance_level, vmax]
                         if not (np.isclose(vmin, chance_level) or
                                 np.isclose(vmax, chance_level) or
                                 np.isclose(vmin, vmax))
                         else np.linspace(vmin, vmax, 3))
        pretty_colorbar(im, ax_obj=ax_obj, colorbar_ticks=cb_ticks_vals,
                        colorbar_ticklabels=[f"{t:.2f}" for t in cb_ticks_vals])

    if diagonal_line_color:
        diag_min = np.max([extent[2], extent[0]])  # Max of y_min, x_min
        diag_max = np.min([extent[3], extent[1]])  # Min of y_max, x_max
        if diag_min < diag_max:
            ax_obj.plot([diag_min, diag_max], [diag_min, diag_max],
                        color=str(diagonal_line_color), linestyle="-")

    if test_times_array.size > 0:
        xt_vals, xt_labs = _set_ticks(test_times_array)
        ax_obj.set_xticks(xt_vals)
        ax_obj.set_xticklabels(xt_labs)
    if times_array.size > 0:
        yt_vals, yt_labs = _set_ticks(times_array)
        ax_obj.set_yticks(yt_vals)
        ax_obj.set_yticklabels(yt_labs)

    if x_axis_label:
        ax_obj.set_xlabel(x_axis_label)
    if y_axis_label:
        ax_obj.set_ylabel(y_axis_label)
    ax_obj.set_xlim(extent[0], extent[1])
    ax_obj.set_ylim(extent[2], extent[3])
    pretty_plot(ax_obj)
    return ax_obj

# === DASHBOARD PLOTTING ===


def create_subject_decoding_dashboard_plots(
    main_epochs_time_points,
    main_original_labels_array,
    main_predicted_probabilities_global,
    main_predicted_labels_global,
    main_cross_validation_global_scores,
    main_temporal_scores_1d_all_folds,
    main_mean_temporal_decoding_scores_1d,
    main_temporal_1d_fdr_sig_data,
    main_temporal_1d_cluster_sig_data,
    main_mean_temporal_generalization_matrix_scores,
    main_tgm_fdr_sig_data,
    main_decoding_global_metrics_for_plot,

    classifier_name_for_title,
    subject_identifier,
    group_identifier,
    output_directory_path=None,
    protocol_type="PP_AP",  # Pour la logique conditionnelle et les titres
    n_folds=None,  # Number of CV folds, computed dynamically if None

    # Arguments spécifiques au protocole PP_AP (seront None si protocol_type != "PP_AP")
    # Utilisé pour Page 4 (PP_AP: PP_spec vs AP_fam)
    specific_ap_decoding_results=None,
    # Utilisé pour Page 5 (PP_AP: Moyenne des PP_spec vs AP_fam)
    mean_of_specific_scores_1d=None,
    sem_of_specific_scores_1d=None,       # Utilisé pour Page 5
    mean_specific_fdr_sig_data=None,      # Utilisé pour Page 5
    mean_specific_cluster_sig_data=None,  # Utilisé pour Page 5
    # Utilisé pour Page 6 & 7 (PP_AP: AP_fam vs AP_fam)
    ap_vs_ap_decoding_results=None,
    # Utilisé pour Page 8 (PP_AP: Moyennes centrées sur AP_fam)
    ap_centric_average_results_list=None,

    CHANCE_LEVEL_AUC=CHANCE_LEVEL_AUC,  # Utilise la constante du module
):
    """Generate a multi-page PDF/PNGs dashboard for single subject results."""
    logger_viz_utils.info(
        "Generating dashboard for Subject: %s (Group: %s, Classifier: %s, Protocol: %s)",
        subject_identifier, group_identifier, classifier_name_for_title, protocol_type
    )

    # Dynamic calculation of n_folds if not provided
    if n_folds is None:
        if main_temporal_scores_1d_all_folds is not None and main_temporal_scores_1d_all_folds.ndim == 2:
            n_folds = main_temporal_scores_1d_all_folds.shape[0]
            logger_viz_utils.debug(f"Computed n_folds dynamically: {n_folds}")
        elif main_cross_validation_global_scores is not None and hasattr(main_cross_validation_global_scores, '__len__'):
            n_folds = len(main_cross_validation_global_scores)
            logger_viz_utils.debug(f"Computed n_folds from CV scores: {n_folds}")
        else:
            n_folds = 10  # Default fallback value
            logger_viz_utils.warning(f"Could not determine n_folds from data, using default: {n_folds}")

    if not isinstance(subject_identifier, str) or not subject_identifier:
        logger_viz_utils.error("Subject ID is required for dashboard.")
        return None
    if not output_directory_path:
        logger_viz_utils.error(
            "Output directory path is required for dashboard.")
        return None
    if main_epochs_time_points is None or main_epochs_time_points.size == 0:
        logger_viz_utils.error(
            "main_epochs_time_points is required and cannot be empty for dashboard.")
        return None

    # Label Encoding pour les labels originaux du protocole principal actuel
    main_label_enc_obj = LabelEncoder()
    labels_enc_main_protocol = np.array([])  # Initialisation
    unique_main_labels_for_cm = ['Class 0', 'Class 1']  # Default

    if main_original_labels_array is not None and main_original_labels_array.size > 0:
        try:
            labels_enc_main_protocol = main_label_enc_obj.fit_transform(
                main_original_labels_array)
            if hasattr(main_label_enc_obj, 'classes_') and main_label_enc_obj.classes_ is not None and \
               main_label_enc_obj.classes_.size > 0:
                unique_main_labels_for_cm = [
                    str(c) for c in main_label_enc_obj.classes_]
        except Exception as e_le:
            logger_viz_utils.error(
                f"Error encoding labels for dashboard (protocol: {protocol_type}): {e_le}")
            # Poursuivre avec labels_enc_main_protocol vide, certaines visualisations pourraient échouer

    os.makedirs(output_directory_path, exist_ok=True)
    plt.switch_backend("Agg")  # Ensure non-interactive backend

    # --- Calcul dynamique de TOTAL_PAGES_TO_GENERATE ---
    current_page_num_tracker = 0
    page_generation_active = {
        "page1_main_overview": True,  # Toujours générée si les données de base sont là
        "page2_main_global_perf": True,  # Toujours générée si les données de base sont là
        "page3_main_tgm": (main_mean_temporal_generalization_matrix_scores is not None and
                           main_mean_temporal_generalization_matrix_scores.size > 0),
        # Les pages suivantes dépendent du protocole et des données spécifiques
        "page4_specific_tasks": False,
        "page5_avg_specific_pp_ap": False,
        "page6_ap_vs_ap_part1": False,
        "page7_ap_vs_ap_part2": False,
        "page8_ap_centric_avg": False,
    }

    # Logique d'activation des pages spécifiques au protocole
    # Allow all PP protocols (PP_AP, battery, ppext3, delirium) to use the 8-page dashboard
    if protocol_type in ["PP_AP", "battery", "ppext3", "delirium", None] or protocol_type.startswith("PP"):
        # Page 4: Tâches spécifiques (PP_spec vs AP_fam)
        if specific_ap_decoding_results and isinstance(specific_ap_decoding_results, list) and \
           any(r.get('scores_1d_mean') is not None for r in specific_ap_decoding_results):
            page_generation_active["page4_specific_tasks"] = True

        # Page 5: Moyenne des tâches spécifiques (PP_spec vs AP_fam)
        if mean_of_specific_scores_1d is not None and \
           main_epochs_time_points.size == mean_of_specific_scores_1d.size and \
           not np.all(np.isnan(mean_of_specific_scores_1d)):
            page_generation_active["page5_avg_specific_pp_ap"] = True

        # Pages 6 & 7: AP_fam vs AP_fam
        plots_per_page_ap_vs_ap_const = 15  # Nombre de graphiques par page pour AP vs AP
        num_valid_ap_vs_ap_plots = 0
        if ap_vs_ap_decoding_results and isinstance(ap_vs_ap_decoding_results, list):
            num_valid_ap_vs_ap_plots = sum(1 for r_ap in ap_vs_ap_decoding_results
                                           if r_ap.get('scores_1d_mean') is not None)
        if num_valid_ap_vs_ap_plots > 0:
            page_generation_active["page6_ap_vs_ap_part1"] = True
        if num_valid_ap_vs_ap_plots > plots_per_page_ap_vs_ap_const:
            page_generation_active["page7_ap_vs_ap_part2"] = True

        # Page 8: Moyennes AP-centriques
        if ap_centric_average_results_list and isinstance(ap_centric_average_results_list, list) and \
           any(r_ac.get('average_scores_1d') is not None and r_ac.get('num_constituent_curves', 0) >= 2
               for r_ac in ap_centric_average_results_list if r_ac is not None):  # check r_ac is not None
            page_generation_active["page8_ap_centric_avg"] = True

    TOTAL_PAGES_TO_GENERATE_ACTUALLY = sum(page_generation_active.values())
    if TOTAL_PAGES_TO_GENERATE_ACTUALLY == 0:
        logger_viz_utils.warning(
            "No pages to generate for dashboard of subject %s (Protocol: %s). Check input data.",
            subject_identifier, protocol_type
        )
        return None

    logger_viz_utils.info(
        "Dashboard for subject %s (Protocol '%s') will generate %d pages.",
        subject_identifier, protocol_type, TOTAL_PAGES_TO_GENERATE_ACTUALLY
    )

    # --- Page 1: Main Overview (Courbe temporelle principale, Scores CV globaux, ROC globale) ---
    if page_generation_active["page1_main_overview"]:
        current_page_num_tracker += 1
        fig1 = None
        try:
            fig1 = plt.figure(figsize=(15, 12))  # Taille de la figure
            # Titre principal de la figure, incluant le protocole
            title_main_task_p1 = f"Class balanced PP vs all AP"
            fig1.suptitle(
                f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: Overview of - {title_main_task_p1}",
                fontsize=16, fontweight="bold",
            )
            gs1 = GridSpec(2, 2, figure=fig1, height_ratios=[
                           2.5, 1], hspace=0.4, wspace=0.3)  # Ajuster hspace/wspace

            # Plot 1.1: Décodage temporel principal
            # Prend toute la première ligne
            ax1_temp = fig1.add_subplot(gs1[0, :])
            if main_mean_temporal_decoding_scores_1d is not None and \
               main_epochs_time_points.size == main_mean_temporal_decoding_scores_1d.size and \
               not np.all(np.isnan(main_mean_temporal_decoding_scores_1d)):

                # Plot des courbes de chaque fold si disponibles
                if main_temporal_scores_1d_all_folds is not None and \
                   main_temporal_scores_1d_all_folds.ndim == 2 and \
                   main_temporal_scores_1d_all_folds.shape[1] == main_epochs_time_points.size:
                    for i_fold, fold_scores in enumerate(main_temporal_scores_1d_all_folds):
                        if not np.all(np.isnan(fold_scores)):
                            ax1_temp.plot(main_epochs_time_points, fold_scores, color='gray',
                                          alpha=0.3, lw=0.7, label=f'{n_folds}-CV Folds' if i_fold == 0 else None)

                # Plot de la moyenne
                ax1_temp.plot(main_epochs_time_points, main_mean_temporal_decoding_scores_1d,
                              color='blue', lw=2.0, label='Mean AUC')

                # Plot du SEM si disponible (calculé à partir de _all_folds)
                if main_temporal_scores_1d_all_folds is not None and main_temporal_scores_1d_all_folds.shape[0] > 1:
                    sem_main_temporal = scipy.stats.sem(
                        main_temporal_scores_1d_all_folds, axis=0, nan_policy='omit')
                    if sem_main_temporal is not None and not np.all(np.isnan(sem_main_temporal)):
                        ax1_temp.fill_between(main_epochs_time_points,
                                              main_mean_temporal_decoding_scores_1d - sem_main_temporal,
                                              main_mean_temporal_decoding_scores_1d + sem_main_temporal,
                                              color='blue', alpha=0.2, label='SEM (across folds)')

                # Barres de significativité
                scores_valid_p1 = main_mean_temporal_decoding_scores_1d[~np.isnan(
                    main_mean_temporal_decoding_scores_1d)]
                y_base_sig_p1 = min(np.min(scores_valid_p1) if scores_valid_p1.size >
                                    0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
                sig_bar_height_p1, current_y_sig_p1 = 0.01, y_base_sig_p1

                if main_temporal_1d_fdr_sig_data and main_temporal_1d_fdr_sig_data.get('mask') is not None and \
                   np.any(main_temporal_1d_fdr_sig_data['mask']):
                    # Créer le label FDR avec l'information du test
                    fdr_label = "FDR p<0.05"
                    test_info = main_temporal_1d_fdr_sig_data.get('test_info', {})
                    if test_info:
                        test_type = test_info.get("test_type", "unknown")
                        if test_type == "adaptive":
                            n_ttest = test_info.get("ttest_features", 0)
                            n_wilcoxon = test_info.get("wilcoxon_features", 0)
                            fdr_label = f"FDR p<0.05 (adaptive: {n_ttest}t-test, {n_wilcoxon}Wilcoxon)"
                        elif test_type == "ttest":
                            fdr_label = "FDR p<0.05 (t-test)"
                        elif test_type == "wilcoxon":
                            fdr_label = "FDR p<0.05 (Wilcoxon)"
                    
                    ax1_temp.fill_between(main_epochs_time_points, current_y_sig_p1 - sig_bar_height_p1, current_y_sig_p1,
                                          where=main_temporal_1d_fdr_sig_data['mask'], color='deepskyblue',
                                          alpha=0.7, step='mid', label=fdr_label)
                    # Décalage pour la prochaine barre
                    current_y_sig_p1 -= (sig_bar_height_p1 + 0.005)
                else:
                    # Afficher le label FDR même si pas significatif, avec info du test si disponible
                    fdr_label = "FDR (no sig.)"
                    test_info = main_temporal_1d_fdr_sig_data.get('test_info', {}) if main_temporal_1d_fdr_sig_data else {}
                    if test_info:
                        test_type = test_info.get("test_type", "unknown")
                        if test_type == "adaptive":
                            n_ttest = test_info.get("ttest_features", 0)
                            n_wilcoxon = test_info.get("wilcoxon_features", 0)
                            fdr_label = f"FDR (no sig., adaptive: {n_ttest}t-test, {n_wilcoxon}Wilcoxon)"
                        elif test_type == "ttest":
                            fdr_label = "FDR (no sig., t-test)"
                        elif test_type == "wilcoxon":
                            fdr_label = "FDR (no sig., Wilcoxon)"
                    # Ajouter une ligne invisible pour la légende
                    ax1_temp.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label)

                if main_temporal_1d_cluster_sig_data and main_temporal_1d_cluster_sig_data.get('mask') is not None and \
                   np.any(main_temporal_1d_cluster_sig_data['mask']):
                    ax1_temp.fill_between(main_epochs_time_points, current_y_sig_p1 - sig_bar_height_p1, current_y_sig_p1,
                                          where=main_temporal_1d_cluster_sig_data['mask'], color='orangered',
                                          alpha=0.7, step='mid', label="Cluster p<0.05")
                else:
                    # Afficher le label cluster même si pas significatif
                    ax1_temp.plot([], [], color='orangered', alpha=0.7, label="Cluster (no sig.)")

                ax1_temp.axhline(CHANCE_LEVEL_AUC, color='k',
                                 ls='--', label=f'Chance ({CHANCE_LEVEL_AUC})')
                if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                    ax1_temp.axvline(0, color='r', ls=':',
                                     label='Stimulus Onset')

                # Ajustement dynamique de ylim
                min_plot_y_p1 = min(current_y_sig_p1 - sig_bar_height_p1 - 0.01,
                                    (np.nanmin(scores_valid_p1) - 0.05 if scores_valid_p1.size > 0 else CHANCE_LEVEL_AUC - 0.15))
                max_plot_y_p1 = max(1.01,
                                    (np.nanmax(scores_valid_p1) + 0.05 if scores_valid_p1.size > 0 else CHANCE_LEVEL_AUC + 0.15))
                ax1_temp.set_ylim(min_plot_y_p1, max_plot_y_p1)
            else:
                ax1_temp.text(0.5, 0.5, f'Main Temporal Scores N/A',
                              ha='center', va='center', transform=ax1_temp.transAxes)

            ax1_temp.set_xlabel('Time (s)')
            ax1_temp.set_ylabel('ROC AUC')
            ax1_temp.set_title(f'Temporal decoding - {title_main_task_p1}')
            ax1_temp.legend(loc='best')
            ax1_temp.grid(True, alpha=0.6)

            # Plot 1.2: Scores CV Globaux
            ax1_cv = fig1.add_subplot(gs1[1, 0])  # Ligne 2, Colonne 1
            if main_cross_validation_global_scores is not None and \
               main_cross_validation_global_scores.size > 0 and \
               not np.all(np.isnan(main_cross_validation_global_scores)):
                num_folds_p1 = len(main_cross_validation_global_scores)
                mean_cv_score_p1 = np.nanmean(
                    main_cross_validation_global_scores)
                ax1_cv.bar(range(1, num_folds_p1 + 1), main_cross_validation_global_scores,
                           color='skyblue', label='ROC AUC/fold')
                ax1_cv.axhline(mean_cv_score_p1, color='r', ls='--',
                               label=f'Mean: {mean_cv_score_p1:.3f}')
                ax1_cv.set_xlabel(f'{n_folds}-CV Folds')
                ax1_cv.set_ylabel('ROC AUC')
                ax1_cv.set_xticks(range(1, num_folds_p1 + 1))
                ax1_cv.set_ylim(0.0, 1.05)
                ax1_cv.set_title(f'Global CV Scores ({num_folds_p1} folds)')
                ax1_cv.legend(loc='best')
            else:
                ax1_cv.text(0.5, 0.5, f'Global CV Scores N/A',
                            ha='center', va='center', transform=ax1_cv.transAxes)
                ax1_cv.set_title(f'Global CV Scores')
            ax1_cv.grid(True, axis='y', ls=':', alpha=0.5)

            # Plot 1.3: Courbe ROC Globale (agrégée)
            ax1_roc = fig1.add_subplot(gs1[1, 1])  # Ligne 2, Colonne 2
            if main_predicted_probabilities_global is not None and \
               main_predicted_probabilities_global.ndim == 2 and \
               main_predicted_probabilities_global.shape[1] >= 2 and \
               labels_enc_main_protocol.size > 0 and len(np.unique(labels_enc_main_protocol)) > 1:

                fpr, tpr, _ = roc_curve(
                    labels_enc_main_protocol, main_predicted_probabilities_global[:, 1])
                auc_aggregated = roc_auc_score(
                    labels_enc_main_protocol, main_predicted_probabilities_global[:, 1])
                ax1_roc.plot(fpr, tpr, color='darkorange', lw=2,
                             label=f'Agg. ROC (AUC={auc_aggregated:.3f})')
                ax1_roc.plot([0, 1], [0, 1], color='navy', lw=1,
                             ls=':', label=f'Chance ({CHANCE_LEVEL_AUC})')
                ax1_roc.set_xlim([-0.02, 1.0])
                ax1_roc.set_ylim([0.0, 1.05])
                ax1_roc.set_xlabel('False Positive Rate')
                ax1_roc.set_ylabel('True Positive Rate')
                ax1_roc.set_title(f'Global ROC (aggregated folds)')
                ax1_roc.legend(loc="lower right")
            else:
                ax1_roc.text(0.5, 0.5, f'Agg. Global ROC N/A',
                             ha='center', va='center', transform=ax1_roc.transAxes)
                ax1_roc.set_title(f'Global ROC')
            ax1_roc.grid(True, alpha=0.6)

            # Ajuster pour le suptitle
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig1.savefig(os.path.join(output_directory_path,
                         f"dashboard_{subject_identifier}_{group_identifier}_{protocol_type}_page1_main_overview.png"), dpi=150)
        except Exception as e_p1:
            logger_viz_utils.error("Error generating Page 1 for dashboard (Subject: %s, Protocol: %s): %s",
                                   subject_identifier, protocol_type, e_p1, exc_info=True)
        finally:
            if fig1:
                plt.close(fig1)

    # --- Page 2: Main Global Performance Details (Matrice de confusion, Densités de probas, Métriques) ---
    if page_generation_active["page2_main_global_perf"]:
        current_page_num_tracker += 1
        fig2 = None
        try:
            fig2 = plt.figure(figsize=(15, 10))
            title_main_task_p2 = f"Main Task ({protocol_type})"
            fig2.suptitle(
                f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: Global performance details - {title_main_task_p2}",
                fontsize=16, fontweight="bold"
            )
            gs2 = GridSpec(2, 2, figure=fig2, hspace=0.4, wspace=0.3)

            # Plot 2.1: Matrice de Confusion Normalisée
            ax2_cm = fig2.add_subplot(gs2[0, 0])
            if main_predicted_labels_global is not None and labels_enc_main_protocol.size > 0:
                cm = confusion_matrix(
                    labels_enc_main_protocol, main_predicted_labels_global, normalize='true')
                sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax2_cm,
                            xticklabels=unique_main_labels_for_cm, yticklabels=unique_main_labels_for_cm, cbar=False)
                ax2_cm.set_xlabel('Predicted Label')
                ax2_cm.set_ylabel('True Label')
                ax2_cm.set_title(f'Normalized Confusion Matrix')
            else:
                ax2_cm.text(0.5, 0.5, 'Confusion Matrix N/A',
                            ha='center', va='center', transform=ax2_cm.transAxes)
                ax2_cm.set_title(f'Normalized confusion matrix')

            # Plot 2.2: Distributions des probabilités prédites
            ax2_proba = fig2.add_subplot(gs2[0, 1])
            if main_predicted_probabilities_global is not None and \
               main_predicted_probabilities_global.ndim == 2 and \
               main_predicted_probabilities_global.shape[1] >= 2 and \
               labels_enc_main_protocol.size > 0 and hasattr(main_label_enc_obj, 'classes_') and \
               main_label_enc_obj.classes_ is not None and main_label_enc_obj.classes_.size >= 2:

                for cls_idx_encoded, cls_name_orig in enumerate(main_label_enc_obj.classes_):
                    indices_this_class = (
                        labels_enc_main_protocol == cls_idx_encoded)
                    if np.sum(indices_this_class) > 1:  # KDE necesita >1 point
                        sns.kdeplot(main_predicted_probabilities_global[indices_this_class, 1], ax=ax2_proba,
                                    label=f'True: {cls_name_orig}', fill=True, alpha=0.5)

                ax2_proba.axvline(0.5, color='r', ls='--',
                                  label='Threshold (0.5)')
                # Label de l'axe X en fonction de la classe positive (généralement la classe 1)
                positive_class_name = str(main_label_enc_obj.classes_[1]) if len(
                    main_label_enc_obj.classes_) > 1 else "Positive Class"
                ax2_proba.set_xlabel(
                    f'Predicted probability (for class "{positive_class_name}")')
                ax2_proba.set_ylabel('Density')
                ax2_proba.legend(loc='best')
                ax2_proba.grid(True, alpha=0.6)
                ax2_proba.set_title(f'Predicted probability distributions')
            else:
                ax2_proba.text(0.5, 0.5, 'Probability Distributions N/A',
                               ha='center', va='center', transform=ax2_proba.transAxes)
                ax2_proba.set_title(f'Predicted probability distributions')

            # Plot 2.3: Barplot des Métriques Globales
            # Prend toute la deuxième ligne
            ax2_metrics = fig2.add_subplot(gs2[1, :])
            if main_decoding_global_metrics_for_plot and isinstance(main_decoding_global_metrics_for_plot, dict):
                metric_names_p2 = list(
                    main_decoding_global_metrics_for_plot.keys())
                # Convertir les valeurs en float, gérant les strings comme '0.75' ou potentiels np.nan
                metric_values_p2_float = []
                for val_str in main_decoding_global_metrics_for_plot.values():
                    try:
                        metric_values_p2_float.append(float(val_str))
                    except (ValueError, TypeError):
                        metric_values_p2_float.append(np.nan)

                valid_metrics_to_plot_p2 = {
                    name: val for name, val in zip(metric_names_p2, metric_values_p2_float) if pd.notna(val)
                }
                if valid_metrics_to_plot_p2:
                    bars_p2 = ax2_metrics.bar(list(valid_metrics_to_plot_p2.keys()),
                                              list(valid_metrics_to_plot_p2.values()), color='lightcoral')
                    ax2_metrics.set_ylim(0, 1.05)
                    ax2_metrics.set_ylabel('Score')
                    for bar_item in bars_p2:  # Ajouter les valeurs sur les barres
                        ax2_metrics.text(bar_item.get_x() + bar_item.get_width() / 2.,
                                         bar_item.get_height() + 0.01,
                                         f'{bar_item.get_height():.3f}', ha='center', va='bottom')
                    plt.setp(ax2_metrics.get_xticklabels(),
                             rotation=15, ha="right")
                    ax2_metrics.set_title(f'Global performance metrics')
                else:
                    ax2_metrics.text(0.5, 0.5, 'Global Metrics N/A or invalid',
                                     ha='center', va='center', transform=ax2_metrics.transAxes)
                    ax2_metrics.set_title(f'global performance metrics')
            else:
                ax2_metrics.text(0.5, 0.5, 'Global Metrics Data N/A',
                                 ha='center', va='center', transform=ax2_metrics.transAxes)
                ax2_metrics.set_title(f'Global performance metrics')
            ax2_metrics.grid(True, axis='y', ls=':', alpha=0.5)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig2.savefig(os.path.join(output_directory_path,
                         f"dashboard_{subject_identifier}_{group_identifier}_{protocol_type}_page2_main_global_perf.png"), dpi=150)
        except Exception as e_p2:
            logger_viz_utils.error("Error generating Page 2 for dashboard (Subject: %s, Protocol: %s): %s",
                                   subject_identifier, protocol_type, e_p2, exc_info=True)
        finally:
            if fig2:
                plt.close(fig2)

    # --- Page 3: Main TGM (Matrice de Généralisation Temporelle principale) ---
    # Conditionnée par la dispo de la TGM
    if page_generation_active["page3_main_tgm"]:
        current_page_num_tracker += 1
        fig3 = None
        try:
            fig3 = plt.figure(figsize=(10, 8))
            title_main_task_p3 = f"Main Task ({protocol_type})"
            fig3.suptitle(
                f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: Temporal Generalization Matrix - {title_main_task_p3}",
                fontsize=16, fontweight="bold"
            )
            ax3_tgm = fig3.add_subplot(111)
            plot_title_tgm_p3 = f"TGM (Mean AUC over {n_folds}-CV folds)"

            # Vérification que toutes les données pour pretty_gat sont valides
            if main_mean_temporal_generalization_matrix_scores is not None and \
               main_epochs_time_points is not None and main_epochs_time_points.size > 0 and \
               main_mean_temporal_generalization_matrix_scores.ndim == 2 and \
               main_mean_temporal_generalization_matrix_scores.shape == \
               (main_epochs_time_points.size, main_epochs_time_points.size) and \
               not np.all(np.isnan(main_mean_temporal_generalization_matrix_scores)):

                fdr_mask_for_tgm_p3 = main_tgm_fdr_sig_data.get(
                    'mask') if main_tgm_fdr_sig_data else None
                sig_info_list_p3 = []
                if fdr_mask_for_tgm_p3 is not None and np.any(fdr_mask_for_tgm_p3):
                    sig_info_list_p3.append("FDR sig. hatched")

                if sig_info_list_p3:
                    plot_title_tgm_p3 += f"\n({', '.join(sig_info_list_p3)}, p<0.05 from CV folds)"
                else:
                    plot_title_tgm_p3 += "\n(FDR: no significant points from CV folds)"

                pretty_gat(
                    main_mean_temporal_generalization_matrix_scores,
                    times_array=main_epochs_time_points,
                    test_times_array=main_epochs_time_points,  # Assume square GAT
                    chance_level=CHANCE_LEVEL_AUC,
                    ax_obj=ax3_tgm,
                    fdr_sig_mask_matrix=fdr_mask_for_tgm_p3,
                    # cluster_sig_masks_list=None, # Si vous avez des clusters pour TGM
                    show_colorbar=True,
                    diagonal_line_color="gray"  # Optionnel: ligne diagonale
                )
                # Ajuster taille police si besoin
                ax3_tgm.set_title(plot_title_tgm_p3, fontsize=12)
            else:
                ax3_tgm.text(0.5, 0.5, f'Main TGM Scores N/A',
                             ha='center', va='center', transform=ax3_tgm.transAxes)
                ax3_tgm.set_title(plot_title_tgm_p3 + " (N/A)")

            plt.tight_layout(rect=[0, 0.03, 1, 0.92])  # Ajuster pour suptitle
            fig3.savefig(os.path.join(output_directory_path,
                         f"dashboard_{subject_identifier}_{group_identifier}_{protocol_type}_page3_main_tgm.png"), dpi=150)
        except Exception as e_p3:
            logger_viz_utils.error("Error generating Page 3 for dashboard (Subject: %s, Protocol: %s): %s",
                                   subject_identifier, protocol_type, e_p3, exc_info=True)
        finally:
            if fig3:
                plt.close(fig3)

    # --- Page 4: Specific Tasks (Dépend du protocole, ex: PP_spec vs AP_fam pour PP_AP) ---
    if page_generation_active["page4_specific_tasks"]:
        current_page_num_tracker += 1
        fig4 = None
        try:
            # `specific_ap_decoding_results` est utilisé ici, mais le nom est générique.
            # Il contiendra les résultats spécifiques au protocole (PP_AP ou LG).
            plot_results_p4_list = [
                # Gérer le cas où c'est None
                r for r in (specific_ap_decoding_results or [])
                if r is not None and r.get('scores_1d_mean') is not None
            ]

            if plot_results_p4_list:
                num_plots_p4 = len(plot_results_p4_list)
                # Ajuster le nombre de colonnes dynamiquement, ex: max 3 par ligne
                n_cols_p4 = min(3, num_plots_p4) if num_plots_p4 > 0 else 1
                n_rows_p4 = (num_plots_p4 + n_cols_p4 -
                             1) // n_cols_p4 if num_plots_p4 > 0 else 1

                # Ajuster la taille
                fig4 = plt.figure(figsize=(7 * n_cols_p4, 5 * n_rows_p4 + 1.5))
                page_title_p4 = f"Specific Tasks ({protocol_type})"
                if protocol_type == "PP_AP":
                    page_title_p4 = "PP_spec vs AP_family_X)"
                elif protocol_type == "delirium":
                    page_title_p4 = "PP_spec vs AP_family_X (Delirium)"
                elif protocol_type == "battery":
                    page_title_p4 = "PP_spec vs AP_family_X (Battery)"
                elif protocol_type == "ppext3":
                    page_title_p4 = "PP_spec vs AP_family_X (PPext3)"
                    page_title_p4 = "PP_spec vs AP_family_X)"

                fig4.suptitle(
                    f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                    f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: {page_title_p4}",
                    fontsize=16, fontweight="bold"
                )
                gs4 = GridSpec(n_rows_p4, n_cols_p4, figure=fig4,
                               hspace=0.7, wspace=0.35)

                for i_plot_p4, result_item_p4 in enumerate(plot_results_p4_list):
                    ax_p4 = fig4.add_subplot(
                        gs4[i_plot_p4 // n_cols_p4, i_plot_p4 % n_cols_p4])

                    mean_scores_p4 = result_item_p4.get('scores_1d_mean')
                    all_folds_scores_p4 = result_item_p4.get(
                        'all_folds_scores_1d')
                    fdr_data_p4 = result_item_p4.get('fdr_significance_data')
                    cluster_data_p4 = result_item_p4.get(
                        'cluster_significance_data')
                    comparison_name_p4 = result_item_p4.get(
                        'comparison_name', f'Task {i_plot_p4 + 1}')

                    if mean_scores_p4 is not None and main_epochs_time_points.size == mean_scores_p4.size and \
                       not np.all(np.isnan(mean_scores_p4)):

                        if all_folds_scores_p4 is not None and all_folds_scores_p4.ndim == 2 and \
                           all_folds_scores_p4.shape[1] == main_epochs_time_points.size:
                            for i_f, f_s in enumerate(all_folds_scores_p4):
                                if not np.all(np.isnan(f_s)):
                                    ax_p4.plot(main_epochs_time_points, f_s, color='gray', alpha=0.25, lw=0.5,
                                               label=f'{n_folds}-CV Folds' if i_f == 0 else None)

                        ax_p4.plot(main_epochs_time_points, mean_scores_p4,
                                   color='blue', lw=1.5, label='Mean AUC')

                        if all_folds_scores_p4 is not None and all_folds_scores_p4.shape[0] > 1:
                            sem_p4 = scipy.stats.sem(
                                all_folds_scores_p4, axis=0, nan_policy='omit')
                            if sem_p4 is not None and not np.all(np.isnan(sem_p4)):
                                ax_p4.fill_between(main_epochs_time_points, mean_scores_p4 - sem_p4,
                                                   mean_scores_p4 + sem_p4, color='blue', alpha=0.2, label='SEM')

                        scores_valid_sp_p4 = mean_scores_p4[~np.isnan(
                            mean_scores_p4)]
                        y_base_sig_sp_p4 = min(np.min(
                            scores_valid_sp_p4) if scores_valid_sp_p4.size > 0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
                        sig_bar_h_sp_p4, cur_y_sig_sp_p4 = 0.01, y_base_sig_sp_p4

                        if fdr_data_p4 and fdr_data_p4.get('mask') is not None and np.any(fdr_data_p4['mask']):
                            # Créer le label FDR avec l'information du test
                            fdr_label_p4 = "FDR p<0.05"
                            test_info_p4 = fdr_data_p4.get('test_info', {})
                            if test_info_p4:
                                test_type = test_info_p4.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p4.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p4.get("wilcoxon_features", 0)
                                    fdr_label_p4 = f"FDR p<0.05 (adaptive: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p4 = "FDR p<0.05 (t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p4 = "FDR p<0.05 (Wilcoxon)"
                            
                            ax_p4.fill_between(main_epochs_time_points, cur_y_sig_sp_p4 - sig_bar_h_sp_p4, cur_y_sig_sp_p4,
                                               where=fdr_data_p4['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_p4)
                            cur_y_sig_sp_p4 -= (sig_bar_h_sp_p4 + 0.005)
                        else:
                            # Afficher le label FDR même si pas significatif
                            fdr_label_p4 = "FDR (no sig.)"
                            test_info_p4 = fdr_data_p4.get('test_info', {}) if fdr_data_p4 else {}
                            if test_info_p4:
                                test_type = test_info_p4.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p4.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p4.get("wilcoxon_features", 0)
                                    fdr_label_p4 = f"FDR (no sig., adaptive: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p4 = "FDR (no sig., t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p4 = "FDR (no sig., Wilcoxon)"
                            ax_p4.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label_p4)

                        if cluster_data_p4 and cluster_data_p4.get('mask') is not None and np.any(cluster_data_p4['mask']):
                            ax_p4.fill_between(main_epochs_time_points, cur_y_sig_sp_p4 - sig_bar_h_sp_p4, cur_y_sig_sp_p4,
                                               where=cluster_data_p4['mask'], color='orangered', alpha=0.7, step='mid', label="Cluster p<0.05")
                        else:
                            # Afficher le label cluster même si pas significatif
                            ax_p4.plot([], [], color='orangered', alpha=0.7, label="Cluster (no sig.)")

                        ax_p4.axhline(CHANCE_LEVEL_AUC, color='k',
                                      ls='--', label=f'Chance ({CHANCE_LEVEL_AUC})')
                        if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                            ax_p4.axvline(0, color='r', ls=':',
                                          label='Stimulus Onset')

                        min_plot_y_sp_p4 = min(cur_y_sig_sp_p4 - sig_bar_h_sp_p4 - 0.01,
                                               (np.nanmin(scores_valid_sp_p4) - 0.05 if scores_valid_sp_p4.size > 0 else CHANCE_LEVEL_AUC - 0.15))
                        max_plot_y_sp_p4 = max(1.01,
                                               (np.nanmax(scores_valid_sp_p4) + 0.05 if scores_valid_sp_p4.size > 0 else CHANCE_LEVEL_AUC + 0.15))
                        ax_p4.set_ylim(min_plot_y_sp_p4, max_plot_y_sp_p4)
                    else:
                        ax_p4.text(0.5, 0.5, 'Scores N/A', ha='center',
                                   va='center', transform=ax_p4.transAxes)

                    ax_p4.set_title(comparison_name_p4, fontsize=10)
                    ax_p4.set_xlabel('Time (s)', fontsize=9)
                    ax_p4.set_ylabel('ROC AUC', fontsize=9)
                    ax_p4.legend(loc='best', fontsize=8)
                    ax_p4.grid(True, ls=':', alpha=0.5)

                # Ajuster pour suptitle
                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                fig4.savefig(os.path.join(output_directory_path,
                             f"dashboard_{subject_identifier}_{group_identifier}_{protocol_type}_page4_specific_tasks.png"), dpi=150)
            else:  # plot_results_p4_list is empty
                logger_viz_utils.info("No valid specific task results to plot for Page 4 (Subject: %s, Protocol: %s).",
                                      subject_identifier, protocol_type)
                # On ne génère pas la page si elle est vide, TOTAL_PAGES_TO_GENERATE_ACTUALLY en tient compte

        except Exception as e_p4:
            logger_viz_utils.error("Error generating Page 4 for dashboard (Subject: %s, Protocol: %s): %s",
                                   subject_identifier, protocol_type, e_p4, exc_info=True)
        finally:
            if fig4:
                plt.close(fig4)

    # --- Page 5: Average of Specific PP_AP Task Curves (PP_AP ONLY) ---
    # Conditionné à PP_AP et données valides
    if page_generation_active["page5_avg_specific_pp_ap"]:
        current_page_num_tracker += 1
        fig5 = None
        try:
            fig5 = plt.figure(figsize=(12, 8))
            fig5.suptitle(
                f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: Average of specific PP vs AP_family ",
                fontsize=16, fontweight="bold"
            )
            ax5_mean_spec = fig5.add_subplot(111)

            num_curves_averaged_p5 = (len(specific_ap_decoding_results)
                                      if specific_ap_decoding_results and isinstance(specific_ap_decoding_results, list)
                                      else "N/A")
            num_valid_curves_for_avg_p5 = sum(1 for r in (
                specific_ap_decoding_results or []) if r.get('scores_1d_mean') is not None)

            ax5_mean_spec.plot(main_epochs_time_points, mean_of_specific_scores_1d, color='black', lw=2,
                               label=f'Average of {num_valid_curves_for_avg_p5} specific PP vs AP family')

            if sem_of_specific_scores_1d is not None and not np.all(np.isnan(sem_of_specific_scores_1d)):
                ax5_mean_spec.fill_between(main_epochs_time_points,
                                           mean_of_specific_scores_1d - sem_of_specific_scores_1d,
                                           mean_of_specific_scores_1d + sem_of_specific_scores_1d,
                                           color='black', alpha=0.2, label='SEM (across tasks)')

            scores_valid_mean_sp_p5 = mean_of_specific_scores_1d[~np.isnan(
                mean_of_specific_scores_1d)]
            y_base_sig_msp_p5 = min(np.min(
                scores_valid_mean_sp_p5) if scores_valid_mean_sp_p5.size > 0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
            sig_bar_h_msp_p5, cur_y_sig_msp_p5 = 0.01, y_base_sig_msp_p5

            # FDR sur la moyenne des courbes spécifiques
            if mean_specific_fdr_sig_data and mean_specific_fdr_sig_data.get('mask') is not None:
                if np.any(mean_specific_fdr_sig_data['mask']):
                    # Créer le label FDR avec l'information du test
                    fdr_label_p5 = "FDR p<0.05"
                    test_info_p5 = mean_specific_fdr_sig_data.get('test_info', {})
                    if test_info_p5:
                        test_type = test_info_p5.get("test_type", "unknown")
                        if test_type == "adaptive":
                            n_ttest = test_info_p5.get("ttest_features", 0)
                            n_wilcoxon = test_info_p5.get("wilcoxon_features", 0)
                            fdr_label_p5 = f"FDR p<0.05 (adaptive: {n_ttest}t, {n_wilcoxon}W)"
                        elif test_type == "ttest":
                            fdr_label_p5 = "FDR p<0.05 (t-test)"
                        elif test_type == "wilcoxon":
                            fdr_label_p5 = "FDR p<0.05 (Wilcoxon)"
                    
                    ax5_mean_spec.fill_between(main_epochs_time_points, cur_y_sig_msp_p5 - sig_bar_h_msp_p5, cur_y_sig_msp_p5,
                                               where=mean_specific_fdr_sig_data['mask'], color='deepskyblue', alpha=0.7,
                                               step='mid', label=fdr_label_p5)
                else:  # Pas de significativité mais la donnée existe
                    fdr_label_p5 = "FDR (no sig.)"
                    test_info_p5 = mean_specific_fdr_sig_data.get('test_info', {})
                    if test_info_p5:
                        test_type = test_info_p5.get("test_type", "unknown")
                        if test_type == "adaptive":
                            n_ttest = test_info_p5.get("ttest_features", 0)
                            n_wilcoxon = test_info_p5.get("wilcoxon_features", 0)
                            fdr_label_p5 = f"FDR (no sig., adaptive: {n_ttest}t, {n_wilcoxon}W)"
                        elif test_type == "ttest":
                            fdr_label_p5 = "FDR (no sig., t-test)"
                        elif test_type == "wilcoxon":
                            fdr_label_p5 = "FDR (no sig., Wilcoxon)"
                    ax5_mean_spec.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label_p5)
                cur_y_sig_msp_p5 -= (sig_bar_h_msp_p5 + 0.005)
            else:  # Donnée FDR non disponible
                ax5_mean_spec.plot([], [], color='deepskyblue',
                                   alpha=0.7, label="FDR (N/A)")

            # Cluster sur la moyenne des courbes spécifiques
            if mean_specific_cluster_sig_data and mean_specific_cluster_sig_data.get('mask') is not None:
                if np.any(mean_specific_cluster_sig_data['mask']):
                    ax5_mean_spec.fill_between(main_epochs_time_points, cur_y_sig_msp_p5 - sig_bar_h_msp_p5, cur_y_sig_msp_p5,
                                               where=mean_specific_cluster_sig_data['mask'], color='orangered', alpha=0.7,
                                               step='mid', label="Cluster p<0.05")
                else:  # Pas de significativité mais la donnée existe
                    ax5_mean_spec.plot(
                        [], [], color='orangered', alpha=0.7, label="Cluster (no sig)")
            else:  # Donnée Cluster non disponible
                ax5_mean_spec.plot([], [], color='orangered',
                                   alpha=0.7, label="Cluster (N/A)")

            ax5_mean_spec.axhline(
                CHANCE_LEVEL_AUC, color='k', ls='--', label=f'Chance ({CHANCE_LEVEL_AUC})')
            if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                ax5_mean_spec.axvline(
                    0, color='r', ls=':', label='Stimulus Onset')

            min_plot_y_msp_p5 = min(cur_y_sig_msp_p5 - sig_bar_h_msp_p5 - 0.01,
                                    (np.nanmin(scores_valid_mean_sp_p5) - 0.05 if scores_valid_mean_sp_p5.size > 0 else CHANCE_LEVEL_AUC - 0.15))
            max_plot_y_msp_p5 = max(1.01,
                                    (np.nanmax(scores_valid_mean_sp_p5) + 0.05 if scores_valid_mean_sp_p5.size > 0 else CHANCE_LEVEL_AUC + 0.15))
            ax5_mean_spec.set_ylim(min_plot_y_msp_p5, max_plot_y_msp_p5)

            ax5_mean_spec.set_title(
                f"Average temporal decoding across {num_valid_curves_for_avg_p5} specific PP vs AP_fam")
            ax5_mean_spec.set_xlabel('Time (s)')
            ax5_mean_spec.set_ylabel('Average ROC AUC')
            ax5_mean_spec.legend(loc='best')
            ax5_mean_spec.grid(True, ls=':', alpha=0.5)

            plt.tight_layout(rect=[0, 0.03, 1, 0.93])
            fig5.savefig(os.path.join(output_directory_path,
                         f"dashboard_{subject_identifier}_{group_identifier}_PP_AP_page5_mean_specific.png"), dpi=150)
        except Exception as e_p5:
            logger_viz_utils.error("Error generating Page 5 for dashboard (Subject: %s, Protocol: PP_AP): %s",
                                   subject_identifier, e_p5, exc_info=True)
        finally:
            if fig5:
                plt.close(fig5)

    # --- Page 6: AP Family vs AP Family Comparisons (Part 1 - PP_AP ONLY) ---
    # Conditionné à PP_AP et données valides
    if page_generation_active["page6_ap_vs_ap_part1"]:
        current_page_num_tracker += 1
        fig6 = None
        try:
            plot_results_ap_vs_ap_all_p6 = [
                r for r in (ap_vs_ap_decoding_results or [])
                if r is not None and r.get('scores_1d_mean') is not None
            ]
            plot_results_p6_page_content = plot_results_ap_vs_ap_all_p6[
                :plots_per_page_ap_vs_ap_const]
            num_plots_p6 = len(plot_results_p6_page_content)

            if num_plots_p6 > 0:
                n_cols_p6 = min(3, num_plots_p6)  # Max 3 colonnes
                n_rows_p6 = (num_plots_p6 + n_cols_p6 - 1) // n_cols_p6
                fig6 = plt.figure(figsize=(7 * n_cols_p6, 5 * n_rows_p6 + 1.5))
                fig6.suptitle(
                    f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                    f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: AP Family vs AP Family (Part 1)",
                    fontsize=16, fontweight="bold"
                )
                gs6 = GridSpec(n_rows_p6, n_cols_p6, figure=fig6,
                               hspace=0.7, wspace=0.35)

                for i_plot_p6, result_item_p6 in enumerate(plot_results_p6_page_content):
                    ax_p6 = fig6.add_subplot(
                        gs6[i_plot_p6 // n_cols_p6, i_plot_p6 % n_cols_p6])
                    mean_scores_p6 = result_item_p6.get('scores_1d_mean')
                    all_folds_p6 = result_item_p6.get('all_folds_scores_1d')
                    fdr_data_p6 = result_item_p6.get('fdr_significance_data')
                    cluster_data_p6 = result_item_p6.get(
                        'cluster_significance_data')
                    comparison_name_p6 = result_item_p6.get(
                        'comparison_name', f'AP vs AP {i_plot_p6 + 1}')

                    # ... (logique de plotting similaire à la Page 4 pour chaque subplot)
                    if mean_scores_p6 is not None and main_epochs_time_points.size == mean_scores_p6.size and \
                       not np.all(np.isnan(mean_scores_p6)):
                        if all_folds_p6 is not None:
                            # Plot all folds transparently
                            ax_p6.plot(main_epochs_time_points, all_folds_p6.T,
                                       color='lightcoral', alpha=0.2, lw=0.5)
                        ax_p6.plot(main_epochs_time_points, mean_scores_p6,
                                   color='darkcyan', lw=1.5, label='Mean AUC')
                        if all_folds_p6 is not None and all_folds_p6.shape[0] > 1:
                            sem_p6 = scipy.stats.sem(
                                all_folds_p6, axis=0, nan_policy='omit')

                            if sem_p6 is not None and not np.all(np.isnan(sem_p6)):
                                ax_p6.fill_between(main_epochs_time_points, mean_scores_p6 - sem_p6,
                                                   mean_scores_p6 + sem_p6, color='c', alpha=0.2, label='SEM')

                        scores_v_p6 = mean_scores_p6[~np.isnan(mean_scores_p6)]
                        y_b_ap6 = min(np.min(scores_v_p6) if scores_v_p6.size >
                                      0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
                        s_h_ap6, cur_y_ap6 = 0.01, y_b_ap6
                        if fdr_data_p6 and fdr_data_p6.get('mask') is not None and np.any(fdr_data_p6['mask']):
                            # Créer le label FDR avec l'information du test
                            fdr_label_p6 = "FDR p<0.05"
                            test_info_p6 = fdr_data_p6.get('test_info', {})
                            if test_info_p6:
                                test_type = test_info_p6.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p6.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p6.get("wilcoxon_features", 0)
                                    fdr_label_p6 = f"FDR p<0.05 (adap: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p6 = "FDR p<0.05 (t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p6 = "FDR p<0.05 (Wilcoxon)"
                            
                            ax_p6.fill_between(main_epochs_time_points, cur_y_ap6 - s_h_ap6, cur_y_ap6,
                                               where=fdr_data_p6['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_p6)
                            cur_y_ap6 -= (s_h_ap6 + 0.005)
                        else:
                            # Afficher le label FDR même si pas significatif
                            fdr_label_p6 = "FDR (no sig.)"
                            test_info_p6 = fdr_data_p6.get('test_info', {}) if fdr_data_p6 else {}
                            if test_info_p6:
                                test_type = test_info_p6.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p6.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p6.get("wilcoxon_features", 0)
                                    fdr_label_p6 = f"FDR (no sig., adap: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p6 = "FDR (no sig., t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p6 = "FDR (no sig., Wilcoxon)"
                            ax_p6.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label_p6)
                            
                        if cluster_data_p6 and cluster_data_p6.get('mask') is not None and np.any(cluster_data_p6['mask']):
                            ax_p6.fill_between(main_epochs_time_points, cur_y_ap6 - s_h_ap6, cur_y_ap6,
                                               where=cluster_data_p6['mask'], color='orangered', alpha=0.7, step='mid', label="Cluster p<0.05")
                        else:
                            # Afficher le label cluster même si pas significatif
                            ax_p6.plot([], [], color='orangered', alpha=0.7, label="Cluster (no sig.)")

                        ax_p6.axhline(CHANCE_LEVEL_AUC, color='k', ls='--',
                                      lw=1, label=f'Chance ({CHANCE_LEVEL_AUC:.1f})')
                        if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                            ax_p6.axvline(0, color='r', ls=':',
                                          lw=1, label='Stimulus Onset')

                        min_plot_y_p6 = min(cur_y_ap6 - s_h_ap6 - 0.01, (np.nanmin(
                            scores_v_p6) - 0.05 if scores_v_p6.size > 0 else CHANCE_LEVEL_AUC - 0.15))
                        max_plot_y_p6 = max(1.01, (np.nanmax(
                            scores_v_p6) + 0.05 if scores_v_p6.size > 0 else CHANCE_LEVEL_AUC + 0.15))
                        ax_p6.set_ylim(min_plot_y_p6, max_plot_y_p6)
                    else:
                        ax_p6.text(0.5, 0.5, 'Scores N/A', ha='center',
                                   va='center', transform=ax_p6.transAxes)

                    ax_p6.set_title(comparison_name_p6, fontsize=10)
                    ax_p6.set_xlabel('Time (s)', fontsize=9)
                    ax_p6.set_ylabel('ROC AUC', fontsize=9)
                    ax_p6.legend(loc='best', fontsize=7)
                    ax_p6.grid(True, ls=':', alpha=0.5)

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                fig6.savefig(os.path.join(output_directory_path,
                             f"dashboard_{subject_identifier}_{group_identifier}_PP_AP_page6_ap_vs_ap_part1.png"), dpi=150)
        except Exception as e_p6:
            logger_viz_utils.error("Error generating Page 6 for dashboard (Subject: %s, Protocol: PP_AP): %s",
                                   subject_identifier, e_p6, exc_info=True)
        finally:
            if fig6:
                plt.close(fig6)

    # --- Page 7: AP Family vs AP Family Comparisons (Part 2 - PP_AP ONLY) ---
    # Conditionné à PP_AP et assez de données pour une 2e page
    if page_generation_active["page7_ap_vs_ap_part2"]:
        current_page_num_tracker += 1
        fig7 = None
        try:
            plot_results_ap_vs_ap_all_p7 = [
                r for r in (ap_vs_ap_decoding_results or [])
                if r is not None and r.get('scores_1d_mean') is not None
            ]

            plot_results_p7_page_content = plot_results_ap_vs_ap_all_p7[
                plots_per_page_ap_vs_ap_const:]
            num_plots_p7 = len(plot_results_p7_page_content)

            if num_plots_p7 > 0:
                n_cols_p7 = min(3, num_plots_p7)
                n_rows_p7 = (num_plots_p7 + n_cols_p7 - 1) // n_cols_p7
                fig7 = plt.figure(figsize=(7 * n_cols_p7, 5 * n_rows_p7 + 1.5))
                fig7.suptitle(
                    f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                    f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: AP Family vs AP Family (Part 2)",
                    fontsize=16, fontweight="bold"
                )
                gs7 = GridSpec(n_rows_p7, n_cols_p7, figure=fig7,
                               hspace=0.7, wspace=0.35)

                for i_plot_p7_local, result_item_p7 in enumerate(plot_results_p7_page_content):
                    ax_p7 = fig7.add_subplot(
                        gs7[i_plot_p7_local // n_cols_p7, i_plot_p7_local % n_cols_p7])
                    # ... (logique de plotting identique à la Page 6 pour chaque subplot)
                    mean_scores_p7 = result_item_p7.get('scores_1d_mean')
                    all_folds_p7 = result_item_p7.get('all_folds_scores_1d')
                    fdr_d_p7 = result_item_p7.get('fdr_significance_data')
                    clu_d_p7 = result_item_p7.get('cluster_significance_data')
                    global_plot_index_p7 = plots_per_page_ap_vs_ap_const + i_plot_p7_local
                    comparison_name_p7 = result_item_p7.get(
                        'comparison_name', f'AP vs AP {global_plot_index_p7 + 1}')

                    if mean_scores_p7 is not None and main_epochs_time_points.size == mean_scores_p7.size and not np.all(np.isnan(mean_scores_p7)):
                        if all_folds_p7 is not None:
                            ax_p7.plot(main_epochs_time_points, all_folds_p7.T,
                                       color='lightcoral', alpha=0.2, lw=0.5)
                        ax_p7.plot(main_epochs_time_points, mean_scores_p7,
                                   color='darkcyan', lw=1.5, label='Mean AUC')
                        if all_folds_p7 is not None and all_folds_p7.shape[0] > 1:
                            sem_ap7 = scipy.stats.sem(
                                all_folds_p7, axis=0, nan_policy='omit')
                            if sem_ap7 is not None and not np.all(np.isnan(sem_ap7)):
                                ax_p7.fill_between(main_epochs_time_points, mean_scores_p7 - sem_ap7,
                                                   mean_scores_p7 + sem_ap7, color='c', alpha=0.2, label='SEM')

                        scores_v_p7 = mean_scores_p7[~np.isnan(mean_scores_p7)]
                        y_b_ap7 = min(np.min(scores_v_p7) if scores_v_p7.size >
                                      0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
                        s_h_ap7, cur_y_ap7 = 0.01, y_b_ap7
                        if fdr_d_p7 and fdr_d_p7.get('mask') is not None and np.any(fdr_d_p7['mask']):
                            # Créer le label FDR avec l'information du test
                            fdr_label_p7 = "FDR p<0.05"
                            test_info_p7 = fdr_d_p7.get('test_info', {})
                            if test_info_p7:
                                test_type = test_info_p7.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p7.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p7.get("wilcoxon_features", 0)
                                    fdr_label_p7 = f"FDR p<0.05 (adap: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p7 = "FDR p<0.05 (t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p7 = "FDR p<0.05 (Wilcoxon)"
                            
                            ax_p7.fill_between(main_epochs_time_points, cur_y_ap7 - s_h_ap7, cur_y_ap7,
                                               where=fdr_d_p7['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_p7)
                            cur_y_ap7 -= (s_h_ap7 + 0.005)
                        else:
                            # Afficher le label FDR même si pas significatif
                            fdr_label_p7 = "FDR (no sig.)"
                            test_info_p7 = fdr_d_p7.get('test_info', {}) if fdr_d_p7 else {}
                            if test_info_p7:
                                test_type = test_info_p7.get("test_type", "unknown")
                                if test_type == "adaptive":
                                    n_ttest = test_info_p7.get("ttest_features", 0)
                                    n_wilcoxon = test_info_p7.get("wilcoxon_features", 0)
                                    fdr_label_p7 = f"FDR (no sig., adap: {n_ttest}t, {n_wilcoxon}W)"
                                elif test_type == "ttest":
                                    fdr_label_p7 = "FDR (no sig., t-test)"
                                elif test_type == "wilcoxon":
                                    fdr_label_p7 = "FDR (no sig., Wilcoxon)"
                            ax_p7.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label_p7)
                            
                        if clu_d_p7 and clu_d_p7.get('mask') is not None and np.any(clu_d_p7['mask']):
                            ax_p7.fill_between(main_epochs_time_points, cur_y_ap7 - s_h_ap7, cur_y_ap7,
                                               where=clu_d_p7['mask'], color='orangered', alpha=0.7, step='mid', label="Cluster p<0.05")
                        else:
                            # Afficher le label cluster même si pas significatif
                            ax_p7.plot([], [], color='orangered', alpha=0.7, label="Cluster (no sig.)")

                        ax_p7.axhline(CHANCE_LEVEL_AUC, color='k', ls='--',
                                      lw=1, label=f'Chance ({CHANCE_LEVEL_AUC:.1f})')
                        if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                            ax_p7.axvline(0, color='r', ls=':',
                                          lw=1, label='Stimulus Onset')

                        min_plot_y_p7 = min(cur_y_ap7 - s_h_ap7 - 0.01, (np.nanmin(
                            scores_v_p7) - 0.05 if scores_v_p7.size > 0 else CHANCE_LEVEL_AUC - 0.15))
                        max_plot_y_p7 = max(1.01, (np.nanmax(
                            scores_v_p7) + 0.05 if scores_v_p7.size > 0 else CHANCE_LEVEL_AUC + 0.15))
                        ax_p7.set_ylim(min_plot_y_p7, max_plot_y_p7)
                    else:
                        ax_p7.text(0.5, 0.5, 'Scores N/A', ha='center',
                                   va='center', transform=ax_p7.transAxes)

                    ax_p7.set_title(comparison_name_p7, fontsize=10)
                    ax_p7.set_xlabel('Time (s)', fontsize=9)
                    ax_p7.set_ylabel('ROC AUC', fontsize=9)
                    ax_p7.legend(loc='best', fontsize=7)
                    ax_p7.grid(True, ls=':', alpha=0.5)

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                fig7.savefig(os.path.join(output_directory_path,
                             f"dashboard_{subject_identifier}_{group_identifier}_PP_AP_page7_ap_vs_ap_part2.png"), dpi=150)
        except Exception as e_p7:
            logger_viz_utils.error("Error generating Page 7 for dashboard (Subject: %s, Protocol: PP_AP): %s",
                                   subject_identifier, e_p7, exc_info=True)
        finally:
            if fig7:
                plt.close(fig7)

 # Mean of: APx vs (All Other APs & PP)
    if page_generation_active["page8_ap_centric_avg"]:
        current_page_num_tracker += 1
        fig8 = None
        try:
            # Filtrer pour les résultats valides avec au moins 2 courbes constitutives
            plot_results_p8_valid_list = [
                r for r in (ap_centric_average_results_list or [])
                if r is not None and r.get('average_scores_1d') is not None and
                r.get('num_constituent_curves', 0) >= 2  # Condition clé !
            ]

            if plot_results_p8_valid_list:
                num_plots_p8 = len(plot_results_p8_valid_list)
                n_cols_p8 = min(3, num_plots_p8)  # Max 3 colonnes
                n_rows_p8 = (num_plots_p8 + n_cols_p8 - 1) // n_cols_p8

                # Ajuster la taille
                fig8 = plt.figure(figsize=(7 * n_cols_p8, 5 * n_rows_p8 + 1.5))

                page8_title_base = "Mean of: APx vs (All Other APs & PP)"
                if num_plots_p8 > 0:
                    page8_title_base += f" [{num_plots_p8} plots]"
                else:
                    page8_title_base += " [No valid plots]"

                fig8.suptitle(
                    f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                    # Titre modifié
                    f"Page {current_page_num_tracker}/{TOTAL_PAGES_TO_GENERATE_ACTUALLY}: {page8_title_base}",
                    fontsize=16, fontweight="bold"
                )
                gs8 = GridSpec(n_rows_p8, n_cols_p8, figure=fig8,
                               hspace=0.7, wspace=0.35)

                for i_plot_p8, result_item_p8 in enumerate(plot_results_p8_valid_list):
                    ax_p8 = fig8.add_subplot(
                        gs8[i_plot_p8 // n_cols_p8, i_plot_p8 % n_cols_p8])

                    avg_scores_p8 = result_item_p8.get('average_scores_1d')
                    sem_scores_p8 = result_item_p8.get('sem_scores_1d')
                    fdr_data_p8 = result_item_p8.get('fdr_sig_data')
                    cluster_data_p8 = result_item_p8.get('cluster_sig_data')
                    anchor_name_p8 = result_item_p8.get(
                        'anchor_ap_family_key_name', f'Anchor {i_plot_p8 + 1}')
                    num_curves_p8_val = result_item_p8.get(
                        'num_constituent_curves', 'N/A')
                    # Les détails des courbes constitutives sont dans result_item_p8.get('constituent_comparison_names_detail')
                    # Vous pouvez les afficher si ce n'est pas trop chargé, ou simplement vous assurer qu'ils sont dans le .npz
                    # Pour le titre du subplot, nous indiquons juste l'ancre. Le titre de la page explique la nature de la moyenne.

                    ax_p8.plot(main_epochs_time_points, avg_scores_p8, color='purple', lw=2,
                               label=f'Average ({num_curves_p8_val} curves)')
                    if sem_scores_p8 is not None and not np.all(np.isnan(sem_scores_p8)):
                        ax_p8.fill_between(main_epochs_time_points, avg_scores_p8 - sem_scores_p8,
                                           avg_scores_p8 + sem_scores_p8, color='purple', alpha=0.2, label='SEM')

                    scores_valid_ac_p8 = avg_scores_p8[~np.isnan(
                        avg_scores_p8)]
                    y_base_sig_ac_p8 = min(np.min(
                        scores_valid_ac_p8) if scores_valid_ac_p8.size > 0 else CHANCE_LEVEL_AUC, CHANCE_LEVEL_AUC) - 0.02
                    sig_bar_h_ac_p8, cur_y_sig_ac_p8 = 0.01, y_base_sig_ac_p8

                    # FDR sur la moyenne AP-centrique
                    if fdr_data_p8 and fdr_data_p8.get('mask') is not None and np.any(fdr_data_p8['mask']):
                        # Créer le label FDR avec l'information du test
                        fdr_label_p8 = "FDR p<0.05"
                        test_info_p8 = fdr_data_p8.get('test_info', {})
                        if test_info_p8:
                            test_type = test_info_p8.get("test_type", "unknown")
                            if test_type == "adaptive":
                                n_ttest = test_info_p8.get("ttest_features", 0)
                                n_wilcoxon = test_info_p8.get("wilcoxon_features", 0)
                                fdr_label_p8 = f"FDR p<0.05 (adap: {n_ttest}t, {n_wilcoxon}W)"
                            elif test_type == "ttest":
                                fdr_label_p8 = "FDR p<0.05 (t-test)"
                            elif test_type == "wilcoxon":
                                fdr_label_p8 = "FDR p<0.05 (Wilcoxon)"
                        
                        ax_p8.fill_between(main_epochs_time_points, cur_y_sig_ac_p8 - sig_bar_h_ac_p8, cur_y_sig_ac_p8,
                                           where=fdr_data_p8['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_p8)
                        cur_y_sig_ac_p8 -= (sig_bar_h_ac_p8 + 0.005)
                    else:
                        # Afficher le label FDR même si pas significatif
                        fdr_label_p8 = "FDR (no sig.)"
                        test_info_p8 = fdr_data_p8.get('test_info', {}) if fdr_data_p8 else {}
                        if test_info_p8:
                            test_type = test_info_p8.get("test_type", "unknown")
                            if test_type == "adaptive":
                                n_ttest = test_info_p8.get("ttest_features", 0)
                                n_wilcoxon = test_info_p8.get("wilcoxon_features", 0)
                                fdr_label_p8 = f"FDR (no sig., adap: {n_ttest}t, {n_wilcoxon}W)"
                            elif test_type == "ttest":
                                fdr_label_p8 = "FDR (no sig., t-test)"
                            elif test_type == "wilcoxon":
                                fdr_label_p8 = "FDR (no sig., Wilcoxon)"
                        ax_p8.plot([], [], color='deepskyblue', alpha=0.7, label=fdr_label_p8)

                    # Cluster sur la moyenne AP-centrique
                    if cluster_data_p8 and cluster_data_p8.get('mask') is not None and np.any(cluster_data_p8['mask']):
                        ax_p8.fill_between(main_epochs_time_points, cur_y_sig_ac_p8 - sig_bar_h_ac_p8, cur_y_sig_ac_p8,
                                           where=cluster_data_p8['mask'], color='orangered', alpha=0.7, step='mid', label="Cluster p<0.05")
                    else:
                        # Afficher le label cluster même si pas significatif
                        ax_p8.plot([], [], color='orangered', alpha=0.7, label="Cluster (no sig.)")

                    ax_p8.axhline(CHANCE_LEVEL_AUC, color='k',
                                  ls='--', label=f'Chance ({CHANCE_LEVEL_AUC})')
                    if main_epochs_time_points.size > 0 and 0 >= main_epochs_time_points.min() and 0 <= main_epochs_time_points.max():
                        ax_p8.axvline(0, color='r', ls=':',
                                      label='Stimulus Onset')

                    min_plot_y_ac_p8 = min(cur_y_sig_ac_p8 - sig_bar_h_ac_p8 - 0.01,
                                           (np.nanmin(scores_valid_ac_p8) - 0.05 if scores_valid_ac_p8.size > 0 else CHANCE_LEVEL_AUC - 0.15))
                    max_plot_y_ac_p8 = max(1.01,
                                           (np.nanmax(scores_valid_ac_p8) + 0.05 if scores_valid_ac_p8.size > 0 else CHANCE_LEVEL_AUC + 0.15))
                    ax_p8.set_ylim(min_plot_y_ac_p8, max_plot_y_ac_p8)

                    # Titre du subplot simplifié pour indiquer l'ancre
                    ax_p8.set_title(
                        f" AP of: {anchor_name_p8}, VS other AP and PPspec", fontsize=11)
                    ax_p8.set_xlabel('Time (s)')
                    ax_p8.set_ylabel('Average ROC AUC')
                    ax_p8.legend(loc='best', fontsize=8)
                    ax_p8.grid(True, ls=':', alpha=0.5)

                plt.tight_layout(rect=[0, 0.03, 1, 0.93])
                fig8.savefig(os.path.join(output_directory_path,
                             f"dashboard_{subject_identifier}_{group_identifier}_PP_AP_page8_APx_mean_vs_PP_OtherAPs.png"), dpi=150)  # Nom de fichier modifié
            else:  # plot_results_p8_valid_list is empty
                logger_viz_utils.info("No valid AP-centric average results (with >=2 constituent curves) to plot for Page 8 (Subject: %s).",
                                      subject_identifier)
        except Exception as e_p8:
            logger_viz_utils.error("Error generating Page 8 for dashboard (Subject: %s, Protocol: PP_AP): %s",
                                   subject_identifier, e_p8, exc_info=True)
        finally:
            if fig8:
                plt.close(fig8)

    logger_viz_utils.info(
        "Finished dashboard plots for Subject: %s (Protocol: %s). Generated %d pages.",
        subject_identifier, protocol_type, current_page_num_tracker
    )
    return output_directory_path
