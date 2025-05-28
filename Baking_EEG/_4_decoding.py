import os
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats as scipy_stats 

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore, GeneralizingEstimator
from mne.parallel import parallel_func


from .stats import decoding_stats as bEEG_stats

logger_4_decoding = logging.getLogger(__name__)
DEFAULT_CHANCE_LEVEL_AUC = 0.5
DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL = "svc"
INTERNAL_N_JOBS_FOR_MNE_DECODING = -1


def _set_ticks(times):
    """
    Generates optimized tick positions and labels for time axes.

    Aims to create a reasonable number of ticks, prioritizing key time points
    like 0, and ensuring start/end times are shown if space allows.

    Args:
        times (np.ndarray): Array of time points.

    Returns:
        tuple: (tick_positions, tick_labels)
            - tick_positions (np.ndarray): Positions for ticks.
            - tick_labels (list): Labels for the ticks (often in ms).
    """
    if not hasattr(times, "__len__") or len(times) == 0:
        return np.array([]), []
    min_time, max_time = np.min(times), np.max(times)

    # Initial ticks at 0.1s intervals
    ticks = np.arange(
        np.ceil(min_time * 10) / 10.0, np.floor(max_time * 10) / 10.0 + 0.1, 0.1
    )

    if not ticks.size: # Handle very short time windows
        ticks = (
            np.array([min_time, max_time])
            if min_time != max_time
            else np.array([min_time])
        )
    ticks = np.round(ticks * 10.0) / 10.0 # Round to nearest 0.1s

    # Ensure min_time and max_time are included if they are not close to existing ticks
    if not np.isclose(ticks[0], min_time) and min_time < ticks[0]:
        ticks = np.insert(ticks, 0, min_time)
    if not np.isclose(ticks[-1], max_time) and max_time > ticks[-1]:
        ticks = np.append(ticks, max_time)

    ticks = np.unique(np.round(ticks * 1000) / 1000) # Round to ms, unique

    # If too many ticks, reduce to a fixed number (e.g., 7)
    if len(ticks) > 10:
        ticks = np.linspace(min_time, max_time, 7)
        ticks = np.round(ticks * 100) / 100 # Round to 0.01s

    ticklabels = []
    if len(ticks) > 0:
        ticklabels.append(f"{int(ticks[0] * 1e3)}") # First tick label in ms
        for _ in ticks[1:-1]: # Empty labels for intermediate ticks
            ticklabels.append("")
        if len(ticks) > 1:
            ticklabels.append(f"{int(ticks[-1] * 1e3)}") # Last tick label in ms
    return ticks, ticklabels


def pretty_plot(ax=None):
    """
    Applies a 'pretty' style to a Matplotlib Axes object.

    Modifies spine visibility, tick colors, and label colors for a cleaner look.

    Args:
        ax (matplotlib.axes.Axes, optional): The Axes object to style.
            If None, `plt.gca()` is used.

    Returns:
        matplotlib.axes.Axes: The styled Axes object.
    """
    if ax is None:
        ax = plt.gca()

    ax.tick_params(colors="dimgray")
    ax.xaxis.label.set_color("dimgray")
    ax.yaxis.label.set_color("dimgray")
    try:
        ax.zaxis.label.set_color("dimgray") # For 3D plots
    except AttributeError:
        pass # Not a 3D plot

    try:
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
    except ValueError:
        pass # Some plot types might not support this

    ax.spines["left"].set_color("dimgray")
    ax.spines["bottom"].set_color("dimgray")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def pretty_colorbar(im=None, ax=None, ticks=None, ticklabels=None, nticks=3, **kwargs):
    """
    Creates and styles a 'pretty' colorbar for an image.

    Args:
        im (matplotlib.image.AxesImage, optional): The image mappable. If None,
            tries to find one in `ax`.
        ax (matplotlib.axes.Axes, optional): Axes to attach the colorbar to.
            If None, `plt.gca()` is used.
        ticks (list, optional): Positions for colorbar ticks. Auto-calculated if None.
        ticklabels (list, optional): Labels for colorbar ticks. Auto-formatted if None.
        nticks (int): Number of ticks if auto-calculated.
        **kwargs: Additional arguments for `plt.colorbar()`.

    Returns:
        matplotlib.colorbar.Colorbar: The created and styled Colorbar object.
    """
    if ax is None:
        ax = plt.gca()

    if im is None:
        # Try to find an image in the axes
        images_in_axes = [
            obj
            for obj in ax.get_children()
            if isinstance(obj, plt.matplotlib.image.AxesImage)
        ]
        if images_in_axes:
            im = images_in_axes[0]
        else:
            raise RuntimeError("No image found to create a colorbar for.")

    current_clim = im.get_clim()
    if ticks is None:
        if None in current_clim or np.isclose(current_clim[0], current_clim[1]):
            # Default ticks if clim is not set or is flat
            ticks = np.array([0.0, 0.5, 1.0])
        else:
            ticks = np.linspace(current_clim[0], current_clim[1], nticks)

    cb = plt.colorbar(im, ax=ax, ticks=ticks, **kwargs)

    if ticklabels is None:
        ticklabels = [f"{val:.2f}" for val in ticks]

    if len(ticklabels) == len(ticks):
        cb.ax.set_yticklabels(ticklabels, color="dimgray")
    else: # Fallback if lengths don't match
        cb.ax.set_yticklabels([f"{val:.2f}" for val in ticks], color="dimgray")

    cb.ax.yaxis.label.set_color("dimgray")
    cb.outline.set_edgecolor("dimgray")
    return cb


def pretty_gat(
    scores,
    times=None,
    chance=0.5,
    ax=None,
    cluster_sig_masks=None, # List of boolean masks for cluster significance
    fdr_sig_mask=None, # Single boolean mask for FDR significance
    cmap="RdBu_r",
    clim=None,
    colorbar=True,
    xlabel="Testing Time (s)",
    ylabel="Training Time (s)",
    sfreq=None, # Sampling frequency, used if times is None
    diagonal_line_color=None, # Color for diagonal line, e.g., 'gray' or None
    test_times=None # For non-square GATs
):
    """
    Plots a Temporal Generalization Matrix (TGM) with enhancements.

    Includes options for significance overlays (cluster and FDR),
    custom colormap, and styling.

    Args:
        scores (np.ndarray): The 2D TGM scores (train_time x test_time).
        times (np.ndarray, optional): Time points for training axis. Auto-generated if None.
        chance (float): Chance level for centering the colormap.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, `plt.gca()` is used.
        cluster_sig_masks (list of np.ndarray, optional): List of boolean masks (same shape as scores)
            indicating significant clusters. Each mask is outlined.
        fdr_sig_mask (np.ndarray, optional): Boolean mask (same shape as scores) for FDR significance,
            plotted with hatches.
        cmap (str): Colormap name.
        clim (tuple or list, optional): Color limits (vmin, vmax). If a single value,
            it's used as spread around chance (chance - val, chance + val).
        colorbar (bool): Whether to draw a colorbar.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        sfreq (float, optional): Sampling frequency if `times` is not provided.
        diagonal_line_color (str, optional): If provided, draws a diagonal line with this color.
        test_times (np.ndarray, optional): Time points for testing axis if different from `times`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the TGM plot.
    """
    scores = np.array(scores)
    if scores.ndim != 2:
        if ax:
            ax.text(0.5, 0.5, "Invalid scores shape for GAT", ha="center", va="center")
            return ax
        logger_4_decoding.error("pretty_gat: scores must be a 2D array.")
        return plt.gca() # Return current axes to avoid crashing

    if times is None:
        times = np.arange(scores.shape[0]) / float(sfreq if sfreq else 1.0)

    if test_times is None:
        # Assume square GAT if test_times not given
        test_times = times if scores.shape[1] == scores.shape[0] else \
                     (np.linspace(np.min(times), np.max(times), scores.shape[1])
                      if times.size > 0 else
                      np.arange(scores.shape[1]) / float(sfreq if sfreq else 1.0))

    # Determine color limits
    if clim is None:
        # Automatic clim based on percentile of scores deviation from chance
        spread_val = (
            np.percentile(np.abs(scores - chance)[~np.isnan(scores)], 98)
            if np.any(~np.isnan(scores))
            else (0.15 if chance == 0.5 else 1.0) # Default spread
        )
        vmin, vmax = chance - spread_val, chance + spread_val
        if np.isclose(vmin, vmax): # Ensure clim has some range
            vmin, vmax = chance - 0.1, chance + 0.1
    elif len(clim) == 1: # Single value for symmetrical spread
        vmin, vmax = chance - clim[0], chance + clim[0]
    else: # Explicit (vmin, vmax)
        vmin, vmax = clim

    if np.isclose(vmin, vmax): # Further ensure clim range
        vmin -= 0.01
        vmax += 0.01

    # Define extent for matshow [left, right, bottom, top]
    extent = [
        np.min(test_times) if test_times.size > 0 else 0,
        np.max(test_times) if test_times.size > 0 else scores.shape[1],
        np.min(times) if times.size > 0 else 0,
        np.max(times) if times.size > 0 else scores.shape[0],
    ]

    if ax is None:
        ax = plt.gca()

    im = ax.matshow(
        scores, extent=extent, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, aspect="auto"
    )

    # Plot cluster significance contours
    if cluster_sig_masks: # Check if it's a non-empty list
        X_clu, Y_clu = np.meshgrid(test_times, times)
        for mask in cluster_sig_masks:
            if mask is not None and mask.shape == scores.shape and np.any(mask):
                ax.contour(
                    X_clu, Y_clu, mask, colors="black", levels=[0.5],
                    linestyles="dotted", linewidths=1.5, corner_mask=False
                )

    # Plot FDR significance hatches
    if fdr_sig_mask is not None and fdr_sig_mask.shape == scores.shape and np.any(fdr_sig_mask):
        X_fdr, Y_fdr = np.meshgrid(test_times, times)
        ax.contourf(
            X_fdr, Y_fdr, fdr_sig_mask, levels=[0.5, 1.5],
            colors="none", hatches=["///"], alpha=0.3 # Hatches for FDR
        )

    ax.axhline(0, color="k", lw=0.5, linestyle=":") # Zero line for training time
    ax.axvline(0, color="k", lw=0.5, linestyle=":") # Zero line for testing time

    if colorbar:
        cb_ticks = [vmin, chance, vmax] if not (
            np.isclose(vmin, chance) or np.isclose(vmax, chance) or np.isclose(vmin, vmax)
        ) else np.linspace(vmin, vmax, 3)
        pretty_colorbar(
            im, ax=ax, ticks=cb_ticks, ticklabels=[f"{t:.2f}" for t in cb_ticks]
        )

    if diagonal_line_color: # Plot diagonal line
        diag_start = np.max([extent[2], extent[0]]) # Max of y_min, x_min
        diag_end = np.min([extent[3], extent[1]])   # Min of y_max, x_max
        if diag_start < diag_end: # Ensure diagonal is within plot bounds
            ax.plot([diag_start, diag_end], [diag_start, diag_end],
                    color=str(diagonal_line_color), linestyle="-")

    if test_times.size > 0:
        xt_vals, xt_labs = _set_ticks(test_times)
        ax.set_xticks(xt_vals)
        ax.set_xticklabels(xt_labs)
    if times.size > 0:
        yt_vals, yt_labs = _set_ticks(times)
        ax.set_yticks(yt_vals)
        ax.set_yticklabels(yt_labs)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    pretty_plot(ax) # Apply general pretty styling
    return ax


def _build_standard_classifier_pipeline(
    classifier_model_type=DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL,
    random_seed_state=42,
    feature_percentile_for_linear_models=15,
):
    """
    Builds a standard scikit-learn classification pipeline.

    Includes scaling and optional feature selection (ANOVA F-test)
    followed by a classifier.

    Args:
        classifier_model_type (str): Type of classifier.
            Supported: 'svc', 'logreg', 'rf'.
        random_seed_state (int): Random seed for reproducibility.
        feature_percentile_for_linear_models (int): Percentile of features
            to select for linear models (SVC, Logistic Regression).

    Returns:
        sklearn.pipeline.Pipeline: The constructed classifier pipeline.

    Raises:
        ValueError: If `classifier_model_type` is not supported.
    """
    logger_4_decoding.info(
        f"Building classifier pipeline: type='{classifier_model_type}', "
        f"Feature selection percentile (if applicable)={feature_percentile_for_linear_models}"
    )
    steps = [("scaler", StandardScaler())] # Always start with scaling
    apply_feature_selection = False

    if classifier_model_type.lower() == "svc":
        classifier = SVC(
            C=1,
            kernel="linear",
            probability=True, # Needed for roc_auc_score
            class_weight="balanced", # Handles class imbalance
            random_state=random_seed_state,
        )
        pipeline_classifier_name = "svc_classifier"
        apply_feature_selection = True # Feature selection beneficial for linear SVM
    elif classifier_model_type.lower() == "logreg":
        classifier = LogisticRegression(
            C=1,
            solver="liblinear", # Good for smaller datasets
            penalty="l2",
            class_weight="balanced",
            random_state=random_seed_state,
            max_iter=1000, # Increased for convergence
        )
        pipeline_classifier_name = "logreg_classifier"
        apply_feature_selection = True # Feature selection beneficial for LogReg
    elif classifier_model_type.lower() == "rf":
        classifier = RandomForestClassifier(
            n_estimators=100, 
            class_weight="balanced",
            random_state=random_seed_state,
            n_jobs=-1, # Use all available cores
        )
        pipeline_classifier_name = "rf_classifier"
    
    else:
        raise ValueError(
            f"Unknown classifier_model_type: '{classifier_model_type}'. "
            f"Supported: 'svc', 'logreg', 'rf'."
        )

    if apply_feature_selection:
        steps.append(
            (
                "anova_feature_selection",
                SelectPercentile(
                    f_classif, percentile=feature_percentile_for_linear_models
                ),
            )
        )
    steps.append((pipeline_classifier_name, classifier))
    return Pipeline(steps)


def _execute_global_decoding_for_one_fold(
    classifier_pipeline_fold,
    epochs_data_flat_fold,
    target_labels_encoded_fold,
    train_indices,
    test_indices,
    trial_sample_weights_fold,
):
    """
    Executes global decoding (using all time points flattened) for a single CV fold.
    Designed for parallel execution.

    Args:
        classifier_pipeline_fold (sklearn.pipeline.Pipeline): Classifier pipeline for this fold.
        epochs_data_flat_fold (np.ndarray): Flattened epoch data (n_trials, n_features_total).
        target_labels_encoded_fold (np.ndarray): Encoded labels for all trials.
        train_indices (np.ndarray): Indices for training trials in this fold.
        test_indices (np.ndarray): Indices for testing trials in this fold.
        trial_sample_weights_fold (np.ndarray, optional): Sample weights for trials.

    Returns:
        tuple:
            - predicted_probabilities_test_fold (np.ndarray): Predicted probabilities for test set.
            - predicted_labels_test_fold (np.ndarray): Predicted labels for test set.
            - roc_auc_score_fold (float): ROC AUC score for this fold.
    """
    X_train_fold, X_test_fold = (
        epochs_data_flat_fold[train_indices],
        epochs_data_flat_fold[test_indices],
    )
    y_train_fold, y_test_fold = (
        target_labels_encoded_fold[train_indices],
        target_labels_encoded_fold[test_indices],
    )

    fit_parameters = {}
    final_estimator_step_name = classifier_pipeline_fold.steps[-1][0]
    estimator_instance = classifier_pipeline_fold.named_steps[final_estimator_step_name]

    # Check if the estimator supports 'sample_weight' in its fit method
    if trial_sample_weights_fold is not None and hasattr(
        estimator_instance, "fit"
    ) and "sample_weight" in estimator_instance.fit.__code__.co_varnames:
        fit_parameters[
            f"{final_estimator_step_name}__sample_weight"
        ] = trial_sample_weights_fold[train_indices]

    if fit_parameters:
        classifier_pipeline_fold.fit(X_train_fold, y_train_fold, **fit_parameters)
    else:
        classifier_pipeline_fold.fit(X_train_fold, y_train_fold)

    predicted_probabilities_test_fold = classifier_pipeline_fold.predict_proba(
        X_test_fold
    )
    # Probabilities for the positive class (typically class 1)
    probabilities_for_roc_auc = predicted_probabilities_test_fold[:, 1]
    predicted_labels_test_fold = classifier_pipeline_fold.predict(X_test_fold)

    auc_sample_weights = (
        trial_sample_weights_fold[test_indices]
        if trial_sample_weights_fold is not None
        else None
    )
    roc_auc_score_fold = np.nan

    # ROC AUC can only be computed if there are at least two classes in y_test
    # and probabilities for at least two classes are returned.
    if len(np.unique(y_test_fold)) > 1 and predicted_probabilities_test_fold.shape[1] >=2:
        try:
            roc_auc_score_fold = roc_auc_score(
                y_true=y_test_fold,
                y_score=probabilities_for_roc_auc,
                sample_weight=auc_sample_weights,
                average="weighted", # Use 'weighted' for multi-class or imbalanced binary
            )
        except ValueError as e:
            logger_4_decoding.debug(f"ROC AUC calculation error in fold: {e}. y_test_fold classes: {np.unique(y_test_fold)}")
            roc_auc_score_fold = np.nan # Ensure it's NaN on error
    elif len(np.unique(y_test_fold)) <= 1:
        logger_4_decoding.debug(
            "Skipping ROC AUC for fold due to single class in y_test_fold."
        )
    return predicted_probabilities_test_fold, predicted_labels_test_fold, roc_auc_score_fold


def run_temporal_decoding_analysis(
    epochs_data,
    target_labels, # Original, unencoded labels
    classifier_pipeline=None, # Must be a pre-built pipeline
    cross_validation_splitter=5, # int for KFold, or a CV splitter object
    trial_sample_weights="auto", # 'auto', None, or ndarray
    n_jobs_external=8, # Number of jobs requested by the caller
    group_labels_for_cv=None, # For GroupKFold
    compute_intra_fold_stats=True, # FDR/Cluster on CV fold scores
    chance_level=DEFAULT_CHANCE_LEVEL_AUC,
    n_permutations_for_intra_fold_clusters=256,
    compute_temporal_generalization_matrix=True,
    cluster_threshold_config_intra_fold=None,
):
    """
    Performs temporal decoding analysis on EEG data with cross-validation.
    """


    if classifier_pipeline is None:
        logger_4_decoding.warning(
            "classifier_pipeline is None in run_temporal_decoding_analysis. "
            f"Building with default type: {DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL}"
        )
        classifier_pipeline = _build_standard_classifier_pipeline(
            classifier_model_type=DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL
        )


    label_encoder_cv_setup = LabelEncoder()
    target_labels_encoded_cv_setup = label_encoder_cv_setup.fit_transform(target_labels)

    if isinstance(cross_validation_splitter, int):
        num_cv_splits_requested = cross_validation_splitter
        cv_random_state = 42
        if group_labels_for_cv is not None:
            unique_groups_cv = np.unique(group_labels_for_cv)
            actual_num_splits_cv_group = min(num_cv_splits_requested, len(unique_groups_cv))
            if actual_num_splits_cv_group < 2 and len(unique_groups_cv) >= 2 :
                actual_num_splits_cv_group = 2 if len(unique_groups_cv) >=2 else len(unique_groups_cv)
            if actual_num_splits_cv_group < 2:
                 logger_4_decoding.error(
                    f"Cannot create valid GroupKFold: only {len(unique_groups_cv)} unique group(s). "
                    f"Need at least 2 for cross-validation."
                )
                 cross_validation_splitter = None
            else:
                cross_validation_splitter = GroupKFold(n_splits=actual_num_splits_cv_group)
        else:
            min_samples_per_class_cv = np.min(np.bincount(target_labels_encoded_cv_setup))
            actual_num_splits_cv_strat = min(
                num_cv_splits_requested,
                min_samples_per_class_cv if min_samples_per_class_cv >= 2 else 2
            )
            if actual_num_splits_cv_strat < 2:
                logger_4_decoding.error(
                    f"Cannot create valid StratifiedKFold: only {min_samples_per_class_cv} "
                    f"samples in the smallest class. Need at least 2 for cross-validation."
                )
                cross_validation_splitter = None
            else:
                cross_validation_splitter = StratifiedKFold(
                    n_splits=actual_num_splits_cv_strat, shuffle=True, random_state=cv_random_state,
                )
        if cross_validation_splitter:
            logger_4_decoding.info(
                f"CV strategy: {type(cross_validation_splitter).__name__} "
                f"with {cross_validation_splitter.get_n_splits()} splits."
            )
    elif not (
        hasattr(cross_validation_splitter, "split")
        and hasattr(cross_validation_splitter, "get_n_splits")
    ):
        raise ValueError("Invalid cross_validation_splitter.")

    num_actual_cv_splits = (
        cross_validation_splitter.get_n_splits() if cross_validation_splitter else 0
    )

    if isinstance(trial_sample_weights, str) and trial_sample_weights.lower() == "auto":
        num_samples_total_sw = len(target_labels)
        unique_classes_sw, class_counts_sw = np.unique(target_labels, return_counts=True)
        if len(unique_classes_sw) < 2:
            trial_sample_weights = None
            logger_4_decoding.info("Sample weights: Disabled (single class or auto failed).")
        else:
            trial_sample_weights = np.zeros(num_samples_total_sw, dtype=float)
            for i_sw, class_val_sw in enumerate(unique_classes_sw):
                class_mask_sw = target_labels == class_val_sw
                trial_sample_weights[class_mask_sw] = num_samples_total_sw / (
                    len(unique_classes_sw) * class_counts_sw[i_sw]
                )
            logger_4_decoding.info("Sample weights: Calculated automatically for class balance.")
    elif not (trial_sample_weights is None or isinstance(trial_sample_weights, np.ndarray)):
        logger_4_decoding.warning(f"Invalid trial_sample_weights type: {type(trial_sample_weights)}. Disabling weights.")
        trial_sample_weights = None
    elif trial_sample_weights is not None:
        logger_4_decoding.info("Sample weights: Provided by user.")
    else:
        logger_4_decoding.info("Sample weights: Disabled by user.")

    n_trials, n_channels, n_time_points = epochs_data.shape
    label_encoder_main = LabelEncoder()
    target_labels_encoded_main = label_encoder_main.fit_transform(target_labels)

    # --- Logging class balance ---
    unique_labels_log, counts_log = np.unique(target_labels, return_counts=True)
    class_balance_str_log = ", ".join([f"'{label}': {count}" for label, count in zip(unique_labels_log, counts_log)])
    logger_4_decoding.info(f"Class balance for current analysis - Original Labels: {{{class_balance_str_log}}}")
    # --- End logging class balance ---

    logger_4_decoding.info(
        f"Input Data: {n_trials} trials, {n_channels} channels, {n_time_points} time points. "
        f"Encoded Classes for main decoding: {dict(zip(label_encoder_main.classes_, np.bincount(target_labels_encoded_main)))}"
    )

    cv_groups_for_mne_call = (
        group_labels_for_cv if isinstance(cross_validation_splitter, GroupKFold) else None
    )

    temporal_scores_1d_all_folds = np.full(
        (num_actual_cv_splits if num_actual_cv_splits > 0 else 1, n_time_points), np.nan
    )
    mean_temporal_decoding_scores_1d = np.full(n_time_points, np.nan)
    tgm_scores_all_folds = np.full(
        (num_actual_cv_splits if num_actual_cv_splits > 0 else 1, n_time_points, n_time_points), np.nan,
    )
    mean_tgm_scores = np.full((n_time_points, n_time_points), np.nan)
    temporal_1d_fdr_sig_data, temporal_1d_cluster_sig_data = None, None
    tgm_fdr_sig_data, tgm_cluster_significance_data = None, None
    n_classes_main = len(label_encoder_main.classes_)
    predicted_probas_global_agg = np.zeros((n_trials, n_classes_main if n_classes_main >=2 else 1))
    predicted_labels_global_agg = np.zeros(n_trials, dtype=int)
    cv_global_scores_array = np.array([])
    global_metrics_dict = {}

    if num_actual_cv_splits >= 2:
        logger_4_decoding.info("Starting 1D temporal decoding (SlidingEstimator)...")
        temporal_decoding_start_time = time.time()
        sliding_time_decoder = SlidingEstimator(
            base_estimator=clone(classifier_pipeline),
            n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
            scoring="roc_auc", verbose=False, # Changed verbose to True for MNE output
        )
        try:
            fit_params_sliding = {}
            final_estimator_name_sliding = sliding_time_decoder.base_estimator.steps[-1][0]
            if trial_sample_weights is not None:
                fit_params_sliding[f"{final_estimator_name_sliding}__sample_weight"] = trial_sample_weights
            temporal_scores_cv_result = cross_val_multiscore(
                sliding_time_decoder, epochs_data, target_labels_encoded_main,
                cv=cross_validation_splitter, groups=cv_groups_for_mne_call,
                fit_params=fit_params_sliding if fit_params_sliding else None,
                n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
            )
            if temporal_scores_cv_result is not None and len(temporal_scores_cv_result) > 0:
                temporal_scores_1d_all_folds = np.array(temporal_scores_cv_result)
                if temporal_scores_1d_all_folds.ndim == 2 and \
                   temporal_scores_1d_all_folds.shape[0] == num_actual_cv_splits and \
                   temporal_scores_1d_all_folds.shape[1] == n_time_points:
                    mean_temporal_decoding_scores_1d = np.nanmean(temporal_scores_1d_all_folds, axis=0)
                else:
                    logger_4_decoding.error(f"Shape mismatch for temporal_scores_1d_all_folds. Expected: ({num_actual_cv_splits}, {n_time_points}), Got: {temporal_scores_1d_all_folds.shape}")
            peak_auc_log_str = "N/A"
            if mean_temporal_decoding_scores_1d is not None and mean_temporal_decoding_scores_1d.size > 0 and not np.all(np.isnan(mean_temporal_decoding_scores_1d)):
                peak_auc_log_str = f"{np.nanmax(mean_temporal_decoding_scores_1d):.3f}"
            logger_4_decoding.info(
                f"1D temporal decoding completed in {time.time() - temporal_decoding_start_time:.2f}s. Peak mean AUC: {peak_auc_log_str}"
            )
        except Exception as e_temporal:
            logger_4_decoding.error(f"Error during 1D temporal decoding: {e_temporal}", exc_info=True)
    else:
        logger_4_decoding.warning(f"Skipping 1D temporal decoding due to insufficient CV splits ({num_actual_cv_splits} splits available, need >= 2).")

    if compute_temporal_generalization_matrix and num_actual_cv_splits >= 2:
        logger_4_decoding.info("Starting TGM computation (GeneralizingEstimator)...")
        tgm_computation_start_time = time.time()
        tgm_decoder = GeneralizingEstimator(
            clone(classifier_pipeline), n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
            scoring="roc_auc", verbose=False, # Changed verbose to True for MNE output
        )
        fit_params_tgm = {}
        final_estimator_name_tgm = tgm_decoder.base_estimator.steps[-1][0]
        if trial_sample_weights is not None:
            fit_params_tgm[f"{final_estimator_name_tgm}__sample_weight"] = trial_sample_weights
        try:
            tgm_scores_all_folds_raw = cross_val_multiscore(
                tgm_decoder, epochs_data, target_labels_encoded_main,
                cv=cross_validation_splitter, groups=cv_groups_for_mne_call,
                n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
                fit_params=fit_params_tgm if fit_params_tgm else None,
            )
            if tgm_scores_all_folds_raw is not None and tgm_scores_all_folds_raw.size > 0:
                tgm_scores_all_folds = np.array(tgm_scores_all_folds_raw)
                if tgm_scores_all_folds.ndim == 3 and \
                   tgm_scores_all_folds.shape[0] == num_actual_cv_splits and \
                   tgm_scores_all_folds.shape[1:] == (n_time_points, n_time_points):
                    mean_tgm_scores = np.nanmean(tgm_scores_all_folds, axis=0)
                else:
                    logger_4_decoding.error(f"Shape mismatch for tgm_scores_all_folds. Expected: ({num_actual_cv_splits}, {n_time_points}, {n_time_points}), Got: {tgm_scores_all_folds.shape}")
            peak_tgm_auc_log_str = "N/A"
            if mean_tgm_scores is not None and mean_tgm_scores.size > 0 and not np.all(np.isnan(mean_tgm_scores)):
                peak_tgm_auc_log_str = f"{np.nanmax(mean_tgm_scores):.3f}"
            logger_4_decoding.info(
                f"TGM computation completed in {time.time() - tgm_computation_start_time:.2f}s. Peak mean TGM AUC: {peak_tgm_auc_log_str}"
            )
        except Exception as e_tgm:
            logger_4_decoding.error(f"Error during TGM computation: {e_tgm}", exc_info=True)
    elif compute_temporal_generalization_matrix:
        logger_4_decoding.warning(f"TGM computation requested but skipped due to insufficient CV splits ({num_actual_cv_splits} splits available, need >= 2).")
    else:
        logger_4_decoding.info("TGM computation disabled by configuration.")

    if compute_intra_fold_stats and num_actual_cv_splits > 1:
        if temporal_scores_1d_all_folds is not None and temporal_scores_1d_all_folds.ndim == 2 and temporal_scores_1d_all_folds.shape[0] > 1 and not np.all(np.isnan(temporal_scores_1d_all_folds)):
            logger_4_decoding.info("Computing intra-fold FDR for 1D temporal scores...")
            _, fdr_sig_mask_1d_intra, fdr_pvals_1d_intra = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                input_data_array=temporal_scores_1d_all_folds, chance_level=chance_level,
                alpha_significance_level=0.05, fdr_correction_method="indep", alternative_hypothesis="greater",
            )
            temporal_1d_fdr_sig_data = {"mask": fdr_sig_mask_1d_intra, "p_values": fdr_pvals_1d_intra, "method": "FDR_on_CV_Folds_1D"}
            logger_4_decoding.info(f"Computing intra-fold Cluster Permutation for 1D temporal scores (n_jobs={INTERNAL_N_JOBS_FOR_MNE_DECODING})...")
            _, clusters_1d_intra, p_vals_1d_intra, _ = bEEG_stats.perform_cluster_permutation_test(
                input_data_array=temporal_scores_1d_all_folds, chance_level=chance_level,
                n_permutations=n_permutations_for_intra_fold_clusters, alternative_hypothesis="greater",
                cluster_threshold_config=cluster_threshold_config_intra_fold, n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
            )
            combined_mask_1d_clu = np.zeros(n_time_points, dtype=bool)
            sig_cluster_objects_1d = []
            if clusters_1d_intra and p_vals_1d_intra is not None:
                for i_clu, clu_mask_1d in enumerate(clusters_1d_intra):
                    if p_vals_1d_intra[i_clu] < 0.05:
                        sig_cluster_objects_1d.append(clu_mask_1d)
                        squeezed_mask_1d = clu_mask_1d.squeeze()
                        if squeezed_mask_1d.ndim == 1 and squeezed_mask_1d.shape == combined_mask_1d_clu.shape:
                            combined_mask_1d_clu = np.logical_or(combined_mask_1d_clu, squeezed_mask_1d)
            temporal_1d_cluster_sig_data = {
                "mask": combined_mask_1d_clu, "cluster_objects": sig_cluster_objects_1d,
                "p_values_all_clusters": p_vals_1d_intra, "method": "ClusterPerm_on_CV_Folds_1D",
            }
        else:
            logger_4_decoding.warning("Skipping 1D temporal intra-fold stats due to NaN, insufficient scores, or too few folds.")

        if compute_temporal_generalization_matrix and tgm_scores_all_folds is not None and tgm_scores_all_folds.ndim == 3 and tgm_scores_all_folds.shape[0] > 1 and not np.all(np.isnan(tgm_scores_all_folds)):
            logger_4_decoding.info("Computing intra-fold FDR for TGM scores...")
            n_f_tgm, n_tr_tgm, n_te_tgm = tgm_scores_all_folds.shape
            flat_tgm_for_fdr = tgm_scores_all_folds.reshape(n_f_tgm, n_tr_tgm * n_te_tgm)
            _, fdr_mask_tgm_flat_intra, pvals_tgm_flat_intra = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                input_data_array=flat_tgm_for_fdr, chance_level=chance_level,
                alpha_significance_level=0.05, fdr_correction_method="indep", alternative_hypothesis="greater",
            )
            tgm_fdr_sig_data = {
                "mask": fdr_mask_tgm_flat_intra.reshape(n_tr_tgm, n_te_tgm),
                "p_values": pvals_tgm_flat_intra.reshape(n_tr_tgm, n_te_tgm), "method": "FDR_on_CV_Folds_TGM",
            }
            tgm_cluster_significance_data = None
            logger_4_decoding.info("Intra-fold Cluster Permutation for TGM is SKIPPED as per request.")
        elif compute_temporal_generalization_matrix:
            logger_4_decoding.warning("Skipping TGM intra-fold stats due to NaN, insufficient scores, or too few folds.")
    else:
        logger_4_decoding.info("Skipping all intra-fold statistical tests (stats disabled or insufficient CV splits).")

    if num_actual_cv_splits >= 2:
        logger_4_decoding.info(f"Starting global decoding (using all time points, n_jobs_for_global_cv={n_jobs_external})...")
        global_decoding_start_time = time.time()
        epochs_data_flat_global = epochs_data.reshape(n_trials, -1)
        parallel_global, pfunc_global, _ = parallel_func(
            _execute_global_decoding_for_one_fold, n_jobs=n_jobs_external, verbose=0
        )
        cv_splits_global = list(
            cross_validation_splitter.split(epochs_data_flat_global, target_labels_encoded_main, groups=cv_groups_for_mne_call)
        )
        global_fold_results = parallel_global(
            pfunc_global(
                clone(classifier_pipeline), epochs_data_flat_global,
                target_labels_encoded_main, train_idx_g, test_idx_g, trial_sample_weights,
            ) for train_idx_g, test_idx_g in cv_splits_global
        )
        predicted_probas_global_agg = np.zeros((n_trials, len(label_encoder_main.classes_)))
        predicted_labels_global_agg = np.zeros(n_trials, dtype=int)
        cv_scores_list_global = []
        for i_fold_global, (_, test_idx_g) in enumerate(cv_splits_global):
            probas_fold_g, preds_fold_g, score_fold_g = global_fold_results[i_fold_global]
            predicted_probas_global_agg[test_idx_g] = probas_fold_g
            predicted_labels_global_agg[test_idx_g] = preds_fold_g
            cv_scores_list_global.append(score_fold_g)
        cv_global_scores_array = np.array(cv_scores_list_global)
        mean_auc_global_str = "N/A"
        std_auc_global_str = "N/A"
        if cv_global_scores_array is not None and cv_global_scores_array.size > 0 and not np.all(np.isnan(cv_global_scores_array)):
            mean_auc_global_str = f"{np.nanmean(cv_global_scores_array):.3f}"
            std_auc_global_str = f"{np.nanstd(cv_global_scores_array):.3f}"
        logger_4_decoding.info(
            f"Global decoding completed in {time.time() - global_decoding_start_time:.2f}s. Mean CV ROC AUC: {mean_auc_global_str} ± {std_auc_global_str}"
        )
        try:
            if len(target_labels_encoded_main) > 0 and predicted_probas_global_agg.ndim == 2 and predicted_probas_global_agg.shape[0] == len(target_labels_encoded_main) and predicted_probas_global_agg.shape[1] >= 2 and len(np.unique(target_labels_encoded_main)) > 1:
                try:
                    global_metrics_dict["roc_auc"] = roc_auc_score(
                        target_labels_encoded_main, predicted_probas_global_agg[:, 1],
                        average="weighted", sample_weight=trial_sample_weights,
                    )
                except ValueError as e_roc_agg:
                    logger_4_decoding.debug(f"Could not compute aggregated ROC AUC: {e_roc_agg}. y_true unique: {np.unique(target_labels_encoded_main)}")
                    global_metrics_dict["roc_auc"] = np.nan
            else:
                global_metrics_dict["roc_auc"] = np.nan
            if len(target_labels_encoded_main) > 0 and predicted_labels_global_agg.size == len(target_labels_encoded_main):
                avg_metric_mode = ("binary" if len(label_encoder_main.classes_) == 2 else "weighted")
                pos_label_metric = 1 if avg_metric_mode == "binary" and (1 in label_encoder_main.transform(label_encoder_main.classes_)) else None
                metric_args = {
                    "y_true": target_labels_encoded_main, "y_pred": predicted_labels_global_agg,
                    "average": avg_metric_mode, "sample_weight": trial_sample_weights, "zero_division": 0,
                }
                if avg_metric_mode == "binary" and pos_label_metric is not None:
                     metric_args['pos_label'] = pos_label_metric
                global_metrics_dict.update({
                    "accuracy": accuracy_score(target_labels_encoded_main, predicted_labels_global_agg, sample_weight=trial_sample_weights),
                    "precision": precision_score(**metric_args),
                    "recall": recall_score(**metric_args),
                    "f1_score": f1_score(**metric_args),
                    "balanced_accuracy": balanced_accuracy_score(target_labels_encoded_main, predicted_labels_global_agg),
                })
            if global_metrics_dict:
                logger_4_decoding.info(
                    f"  Overall global performance metrics (from aggregated predictions): {', '.join([f'{k}={v:.3f}' for k, v in global_metrics_dict.items() if pd.notna(v)])}"
                )
        except Exception as e_metrics:
            logger_4_decoding.error(f"Error calculating aggregated global metrics: {e_metrics}", exc_info=True)
    else:
        logger_4_decoding.warning("Skipping global decoding due to insufficient CV splits.")

    return (
        predicted_probas_global_agg, predicted_labels_global_agg, cv_global_scores_array,
        mean_temporal_decoding_scores_1d, global_metrics_dict,
        temporal_1d_fdr_sig_data, temporal_1d_cluster_sig_data,
        temporal_scores_1d_all_folds, mean_tgm_scores,
        tgm_fdr_sig_data, tgm_cluster_significance_data, tgm_scores_all_folds
    )


def _compute_temporal_score_at_single_timepoint_cs(
    timepoint_index,
    X_train_at_timepoint, # Data for this timepoint: (n_train_trials, n_channels)
    y_train_encoded,
    X_test_at_timepoint, # Data for this timepoint: (n_test_trials, n_channels)
    y_test_encoded,
    classifier_prototype_cs, # A fresh clone of the base pipeline
):
    """
    Computes decoding score (ROC AUC) at a single timepoint for cross-subject decoding.
    Helper for parallelizing temporal decoding in `run_cross_subject_decoding_for_fold`.

    Args:
        timepoint_index (int): Index of the current timepoint (for logging).
        X_train_at_timepoint (np.ndarray): Training data for this timepoint.
        y_train_encoded (np.ndarray): Encoded training labels.
        X_test_at_timepoint (np.ndarray): Testing data for this timepoint.
        y_test_encoded (np.ndarray): Encoded testing labels.
        classifier_prototype_cs (sklearn.pipeline.Pipeline): A clone of the classifier pipeline.

    Returns:
        float: ROC AUC score for this timepoint. Returns chance (0.5) on error or if
               test set has only one class.
    """
    try:
        clf = clone(classifier_prototype_cs) # Use a fresh clone
        clf.fit(X_train_at_timepoint, y_train_encoded)

        # If test set has only one class, AUC is undefined or typically 0.5
        if len(np.unique(y_test_encoded)) < 2:
            return 0.5 # Return chance level

        probas = clf.predict_proba(X_test_at_timepoint)
        # Ensure probabilities for at least two classes are returned
        if probas.ndim < 2 or probas.shape[1] < 2:
            return 0.5 # Return chance if probabilities are not as expected

        return roc_auc_score(y_test_encoded, probas[:, 1]) # Score using prob of positive class
    except Exception as e:
        logger_4_decoding.debug(
            f"Error during CS decoding at timepoint {timepoint_index}: {e}"
        )
        return 0.5 # Return chance on any error


def run_cross_subject_decoding_for_fold(
    training_epochs_data, # (n_train_trials, n_channels, n_times)
    training_original_labels,
    testing_epochs_data, # (n_test_trials, n_channels, n_times)
    testing_original_labels,
    testing_subject_identifier, # ID of the subject being tested
    group_identifier, # Identifier for the group/set this CS fold belongs to
    decoding_protocol_identifier, # Name of the decoding protocol
    base_classifier_pipeline_prototype=None,
    n_jobs_for_temporal_decoding=-1, # For parallelizing timepoints within this fold
):
    """
    Runs cross-subject decoding for one fold (one subject as test, others as train).

    Performs global decoding (all timepoints flattened) and 1D temporal decoding.

    Args:
        training_epochs_data (np.ndarray): Aggregated data from N-1 training subjects.
        training_original_labels (np.ndarray): Labels for training data.
        testing_epochs_data (np.ndarray): Data for the single test subject.
        testing_original_labels (np.ndarray): Labels for test subject data.
        testing_subject_identifier (str): ID of the test subject.
        group_identifier (str): Identifier of the cross-subject set.
        decoding_protocol_identifier (str): Name of the decoding protocol.
        base_classifier_pipeline_prototype (sklearn.pipeline.Pipeline, optional):
            Base classifier. A standard one is built if None.
        n_jobs_for_temporal_decoding (int): Number of jobs for parallelizing
            the 1D temporal decoding across timepoints.

    Returns:
        tuple:
            - auc_global_cs (float): Global ROC AUC for this fold.
            - metrics_global_cs (dict): Other global metrics for this fold.
            - clf_global_cs (sklearn.pipeline.Pipeline): Trained global classifier.
            - probas_global_cs (np.ndarray): Predicted probabilities (global).
            - labels_global_cs (np.ndarray): Predicted labels (global).
            - scores_1d_cs_arr (np.ndarray): 1D temporal decoding scores.
    """

    if n_jobs_for_temporal_decoding == -1:
        effective_n_jobs_cs_temporal = os.cpu_count() if os.cpu_count() is not None else 1
    elif n_jobs_for_temporal_decoding == 0:
        effective_n_jobs_cs_temporal = 1
    else:
        effective_n_jobs_cs_temporal = n_jobs_for_temporal_decoding

    logger_4_decoding.info(
        f"Cross-Subject Fold (Test Subject: {testing_subject_identifier}): "
        f"Temporal decoding n_jobs={effective_n_jobs_cs_temporal}."
    )

    cs_fold_start_time = time.time()
    label_encoder_cs = LabelEncoder()
    # Fit encoder on all unique labels from both train and test to ensure consistency
    all_labels_cs = np.concatenate((training_original_labels, testing_original_labels))
    label_encoder_cs.fit(all_labels_cs)
    train_labels_enc = label_encoder_cs.transform(training_original_labels)
    test_labels_enc = label_encoder_cs.transform(testing_original_labels)

    # Check if both train and test sets have at least two classes
    if len(np.unique(train_labels_enc)) < 2 or len(np.unique(test_labels_enc)) < 2:
        logger_4_decoding.warning(
            f"CS Fold (Test: {testing_subject_identifier}): Train or test set has < 2 classes. "
            f"Train classes: {np.unique(train_labels_enc)}, Test classes: {np.unique(test_labels_enc)}. "
            "Skipping fold."
        )
        n_times_fail_cs = testing_epochs_data.shape[2] if testing_epochs_data.ndim == 3 else 0
        return (np.nan, {}, None, np.array([]), np.array([]), np.full(n_times_fail_cs, np.nan))

    if base_classifier_pipeline_prototype is None:
        base_classifier_pipeline_prototype = _build_standard_classifier_pipeline()

    # --- Global Decoding for this CS Fold ---
    clf_global_cs = clone(base_classifier_pipeline_prototype)
    train_flat_cs = training_epochs_data.reshape(training_epochs_data.shape[0], -1)
    clf_global_cs.fit(train_flat_cs, train_labels_enc)

    test_flat_cs = testing_epochs_data.reshape(testing_epochs_data.shape[0], -1)
    probas_global_cs = clf_global_cs.predict_proba(test_flat_cs)
    labels_global_cs = clf_global_cs.predict(test_flat_cs)
    auc_global_cs = np.nan

    if probas_global_cs.ndim == 2 and probas_global_cs.shape[1] >= 2:
        try:
            auc_global_cs = roc_auc_score(test_labels_enc, probas_global_cs[:, 1])
        except ValueError: # If only one class in test_labels_enc despite earlier check (rare)
            auc_global_cs = np.nan

    metrics_global_cs = {"roc_auc": auc_global_cs}
    avg_m_cs = "binary" if len(label_encoder_cs.classes_) == 2 else "weighted"
    # Assuming positive class is 1 if binary
    pos_l_cs = 1 if avg_m_cs == "binary" and (1 in test_labels_enc) else None


    metric_args_cs = {
        "y_true": test_labels_enc,
        "y_pred": labels_global_cs,
        "average": avg_m_cs,
        "zero_division": 0,
    }
    if avg_m_cs == "binary" and pos_l_cs is not None:
        metric_args_cs['pos_label'] = pos_l_cs


    metrics_global_cs.update(
        {
            "accuracy": accuracy_score(test_labels_enc, labels_global_cs),
            "precision": precision_score(**metric_args_cs),
            "recall": recall_score(**metric_args_cs),
            "f1_score": f1_score(**metric_args_cs),
            "balanced_accuracy": balanced_accuracy_score(test_labels_enc, labels_global_cs),
        }
    )

    # --- 1D Temporal decoding for this CS fold ---
    n_times_cs = testing_epochs_data.shape[2]
    parallel_cs, pfunc_cs, _ = parallel_func(
        _compute_temporal_score_at_single_timepoint_cs,
        n_jobs=effective_n_jobs_cs_temporal, # Use the determined number of jobs
        verbose=0,
    )
    scores_1d_cs_list = parallel_cs(
        pfunc_cs(
            tp_idx, # Timepoint index
            training_epochs_data[:, :, tp_idx], # Train data at this timepoint
            train_labels_enc,
            testing_epochs_data[:, :, tp_idx], # Test data at this timepoint
            test_labels_enc,
            clone(base_classifier_pipeline_prototype), # Fresh classifier for each timepoint
        )
        for tp_idx in range(n_times_cs)
    )
    scores_1d_cs_arr = np.array(scores_1d_cs_list)

    logger_4_decoding.info(
        f"CS Fold for {testing_subject_identifier} processed in {time.time() - cs_fold_start_time:.2f}s. "
        f"Global AUC: {auc_global_cs:.3f if pd.notna(auc_global_cs) else 'N/A'}."
    )
    return (
        auc_global_cs,
        metrics_global_cs,
        clf_global_cs, # Trained global classifier
        probas_global_cs,
        labels_global_cs,
        scores_1d_cs_arr,
    )

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
    classifier_name_for_title, 
    subject_identifier,
    group_identifier,
    output_directory_path=None,
    specific_ap_decoding_results=None,
    mean_of_specific_scores_1d=None,
    sem_of_specific_scores_1d=None,
    mean_specific_fdr_sig_data=None,
    mean_specific_cluster_sig_data=None,
    chance_level_auc=DEFAULT_CHANCE_LEVEL_AUC,
    epochs_data_array_unused=None,
):
    """
    Generates a multi-page PDF dashboard summarizing decoding results for a single subject.
    """
    logger_4_decoding.info(
        f"Generating dashboard for Subject: {subject_identifier} (Group: {group_identifier}, Classifier: {classifier_name_for_title})"
    )
    main_label_encoder = LabelEncoder()
    if main_original_labels_array is not None and main_original_labels_array.size > 0:
        main_labels_enc = main_label_encoder.fit_transform(main_original_labels_array)
    else:
        main_labels_enc = np.array([])

    if output_directory_path:
        os.makedirs(output_directory_path, exist_ok=True)
    plt.switch_backend("Agg")

    # --- Page 1: Main Overview ---
    fig1 = None
    try:
        fig1 = plt.figure(figsize=(15, 12))
        fig1.suptitle(
            f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
            f"Page 1/5: Main overview ",
            fontsize=16, fontweight="bold",
        )
        gs1 = GridSpec(2, 2, figure=fig1, height_ratios=[2.5, 1], hspace=0.35, wspace=0.25)
        ax1_temp = fig1.add_subplot(gs1[0, :])

        if main_mean_temporal_decoding_scores_1d is not None and \
           main_epochs_time_points is not None and \
           main_epochs_time_points.size == main_mean_temporal_decoding_scores_1d.size and \
           not np.all(np.isnan(main_mean_temporal_decoding_scores_1d)):

            if main_temporal_scores_1d_all_folds is not None and \
               main_temporal_scores_1d_all_folds.ndim == 2 and \
               main_temporal_scores_1d_all_folds.shape[1] == main_epochs_time_points.size:
                for i_fold_p1, fold_scores_p1 in enumerate(main_temporal_scores_1d_all_folds):
                    if not np.all(np.isnan(fold_scores_p1)):
                        ax1_temp.plot(main_epochs_time_points, fold_scores_p1,
                                      color='gray', alpha=0.3, lw=0.7,
                                      label='CV Folds' if i_fold_p1 == 0 else None)

            ax1_temp.plot(main_epochs_time_points, main_mean_temporal_decoding_scores_1d,
                          color='blue', lw=2.0, label='Mean AUC (main task)')

            if main_temporal_scores_1d_all_folds is not None and main_temporal_scores_1d_all_folds.shape[0] > 1:
                sem_main_1d = scipy_stats.sem(main_temporal_scores_1d_all_folds, axis=0, nan_policy='omit')
                if sem_main_1d is not None and not np.all(np.isnan(sem_main_1d)):
                    ax1_temp.fill_between(main_epochs_time_points,
                                          main_mean_temporal_decoding_scores_1d - sem_main_1d,
                                          main_mean_temporal_decoding_scores_1d + sem_main_1d,
                                          color='blue', alpha=0.2, label='SEM (across folds)')

            scores_valid_p1 = main_mean_temporal_decoding_scores_1d[~np.isnan(main_mean_temporal_decoding_scores_1d)]
            y_base_p1 = min(np.min(scores_valid_p1) if scores_valid_p1.size > 0 else chance_level_auc, chance_level_auc) - 0.02
            sig_h_p1 = 0.01
            cur_y_p1 = y_base_p1

            fdr_label_p1 = "Mean FDR p<0.05 (Main Task)"
            has_fdr_sig_p1 = (main_temporal_1d_fdr_sig_data and main_temporal_1d_fdr_sig_data.get('mask') is not None and np.any(main_temporal_1d_fdr_sig_data['mask']))
            if has_fdr_sig_p1:
                ax1_temp.fill_between(main_epochs_time_points, cur_y_p1 - sig_h_p1, cur_y_p1,
                                      where=main_temporal_1d_fdr_sig_data['mask'],
                                      color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_p1)
                cur_y_p1 -= (sig_h_p1 + 0.005)
            elif main_temporal_1d_fdr_sig_data and main_temporal_1d_fdr_sig_data.get('mask') is not None:
                ax1_temp.plot([], [], color='deepskyblue', alpha=0.7, label="Mean FDR (no significance - Main Task)")
            else:
                ax1_temp.plot([], [], color='deepskyblue', alpha=0.7, label="Mean FDR (N/A - Main Task)")

            cluster_label_p1 = "Mean Cluster p<0.05 (Main Task)"
            has_cluster_sig_p1 = (main_temporal_1d_cluster_sig_data and main_temporal_1d_cluster_sig_data.get('mask') is not None and np.any(main_temporal_1d_cluster_sig_data['mask']))
            if has_cluster_sig_p1:
                ax1_temp.fill_between(main_epochs_time_points, cur_y_p1 - sig_h_p1, cur_y_p1,
                                      where=main_temporal_1d_cluster_sig_data['mask'],
                                      color='orangered', alpha=0.7, step='mid', label=cluster_label_p1)
            elif main_temporal_1d_cluster_sig_data and main_temporal_1d_cluster_sig_data.get('mask') is not None:
                ax1_temp.plot([], [], color='orangered', alpha=0.7, label="Mean Cluster (no significance - Main Task)")
            else:
                ax1_temp.plot([], [], color='orangered', alpha=0.7, label="Mean Cluster (N/A - Main Task)")
        
            ax1_temp.axhline(chance_level_auc, color='k', linestyle='--', label=f'Chance ({chance_level_auc})')
            if main_epochs_time_points.size > 0:
                ax1_temp.axvline(0, color='r', linestyle=':', label='Stimulus Onset')
            if scores_valid_p1.size > 0:
                max_auc_p1 = np.nanmax(scores_valid_p1)
                max_idx_p1 = np.nanargmax(main_mean_temporal_decoding_scores_1d)
                if max_idx_p1 < main_epochs_time_points.size:
                    ax1_temp.plot(main_epochs_time_points[max_idx_p1], max_auc_p1, 'ro', label=f'Max Mean AUC: {max_auc_p1:.3f}')
            ymin_plot_p1 = min(cur_y_p1 - sig_h_p1 - 0.01, (np.min(scores_valid_p1) - 0.05 if scores_valid_p1.size > 0 else chance_level_auc - 0.15))
            ymax_plot_p1 = max(1.01, (np.max(scores_valid_p1) + 0.05 if scores_valid_p1.size > 0 else chance_level_auc + 0.15))
            ax1_temp.set_ylim(ymin_plot_p1, ymax_plot_p1)
        else:
            ax1_temp.text(0.5, 0.5, 'Main Temporal Scores N/A', ha='center', va='center', transform=ax1_temp.transAxes)

        ax1_temp.set_xlabel('Time (s)')
        ax1_temp.set_ylabel('ROC AUC')
        ax1_temp.set_title('Temporal decoding performance PP/All vs AP/all (classes balanced)')
        ax1_temp.legend(loc='best')
        ax1_temp.grid(True, alpha=0.6)

        ax1_cv = fig1.add_subplot(gs1[1, 0])
        if main_cross_validation_global_scores is not None and hasattr(main_cross_validation_global_scores, 'size') and main_cross_validation_global_scores.size > 0 and not np.all(np.isnan(main_cross_validation_global_scores)):
            n_folds_p1 = len(main_cross_validation_global_scores)
            mean_cv_p1 = np.nanmean(main_cross_validation_global_scores)
            ax1_cv.bar(range(1, n_folds_p1 + 1), main_cross_validation_global_scores, color='skyblue', label='ROC AUC/fold')
            ax1_cv.axhline(mean_cv_p1, color='r', linestyle='--', label=f'Mean: {mean_cv_p1:.3f}')
            ax1_cv.set_xlabel('CV Fold')
            ax1_cv.set_ylabel('ROC AUC')
            ax1_cv.set_xticks(range(1, n_folds_p1 + 1))
            ax1_cv.set_ylim(0.0, 1.05)
            ax1_cv.set_title(f'Global CV scores ({n_folds_p1} folds) (All epochs / all channels / all points)')
            ax1_cv.legend()
        else:
            ax1_cv.text(0.5, 0.5, 'Global CV scores N/A', ha='center', va='center', transform=ax1_cv.transAxes)
            ax1_cv.set_title('Global CV scores')

        ax1_roc = fig1.add_subplot(gs1[1, 1])
        if main_predicted_probabilities_global is not None and main_predicted_probabilities_global.ndim == 2 and main_predicted_probabilities_global.shape[1] >= 2 and main_labels_enc.size > 0 and len(np.unique(main_labels_enc)) > 1:
            fpr_p1, tpr_p1, _ = roc_curve(main_labels_enc, main_predicted_probabilities_global[:, 1])
            auc_agg_p1 = roc_auc_score(main_labels_enc, main_predicted_probabilities_global[:, 1])
            ax1_roc.plot(fpr_p1, tpr_p1, color='darkorange', lw=2, label=f'Agg. ROC (AUC={auc_agg_p1:.3f})')
            ax1_roc.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':', label=f'Chance ({chance_level_auc})')
            ax1_roc.set_xlim([-0.02, 1.0])
            ax1_roc.set_ylim([0.0, 1.05])
            ax1_roc.set_xlabel('False Positive Rate')
            ax1_roc.set_ylabel('True Positive Rate')
            ax1_roc.set_title('Global ROC (Aggregated Folds)')
            ax1_roc.legend(loc="lower right")
            ax1_roc.grid(True, alpha=0.6)
        else:
            ax1_roc.text(0.5, 0.5, 'Agg. Global ROC N/A', ha='center', va='center', transform=ax1_roc.transAxes)
            ax1_roc.set_title('Global ROC')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_directory_path:
            fig1.savefig(os.path.join(output_directory_path, f"dashboard_{subject_identifier}_{group_identifier}_page1_main_overview.png"), dpi=150)
    except Exception as e:
        logger_4_decoding.error(f"Error generating Page 1 for dashboard: {e}", exc_info=True)
    finally:
        if fig1: plt.close(fig1)

    # --- Page 2: Main Global performance details ---
    fig2 = None
    try:
        fig2 = plt.figure(figsize=(15, 10))
        fig2.suptitle(
            f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
            f"Page 2/5: Main global performance",
            fontsize=16, fontweight="bold"
        )
        gs2 = GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.25)
        ax2_cm = fig2.add_subplot(gs2[0, 0])
        if main_predicted_labels_global is not None and main_labels_enc.size > 0:
            cm_p2 = confusion_matrix(main_labels_enc, main_predicted_labels_global, normalize='true')
            class_names_p2 = [str(c) for c in main_label_encoder.classes_] if hasattr(main_label_encoder, 'classes_') and main_label_encoder.classes_ is not None and main_label_encoder.classes_.size > 0 else ['Class 0', 'Class 1']
            sns.heatmap(cm_p2, annot=True, fmt='.2%', cmap='Blues', ax=ax2_cm, xticklabels=class_names_p2, yticklabels=class_names_p2, cbar=False)
            ax2_cm.set_xlabel('Predicted Label')
            ax2_cm.set_ylabel('True Label')
            ax2_cm.set_title('Normalized confusion matrix')
        else:
            ax2_cm.text(0.5, 0.5, 'Confusion matrix N/A', ha='center', va='center', transform=ax2_cm.transAxes)
            ax2_cm.set_title('Confusion matrix')
        ax2_proba = fig2.add_subplot(gs2[0, 1])
        if main_predicted_probabilities_global is not None and main_predicted_probabilities_global.ndim == 2 and main_predicted_probabilities_global.shape[1] >= 2 and main_labels_enc.size > 0 and hasattr(main_label_encoder, 'classes_') and main_label_encoder.classes_ is not None and main_label_encoder.classes_.size >= 2:
            for cls_val_p2 in np.unique(main_labels_enc):
                cls_name_p2 = main_label_encoder.inverse_transform([cls_val_p2])[0]
                indices_p2 = (main_labels_enc == cls_val_p2)
                if np.sum(indices_p2) > 1:
                    sns.kdeplot(main_predicted_probabilities_global[indices_p2, 1], ax=ax2_proba, label=f'True: {cls_name_p2}', fill=True, alpha=0.5)
            ax2_proba.axvline(0.5, color='r', linestyle='--', label='Threshold (0.5)')
            ax2_proba.set_xlabel(f'Predicted probability (for class "{main_label_encoder.classes_[1]}")')
            ax2_proba.set_ylabel('Density')
            ax2_proba.set_title('Predicted probability distributions')
            ax2_proba.legend()
            ax2_proba.grid(True, alpha=0.6)
        else:
            ax2_proba.text(0.5, 0.5, 'Probability distributions N/A', ha='center', va='center', transform=ax2_proba.transAxes)
            ax2_proba.set_title('Predicted probability distributions')
        ax2_metrics = fig2.add_subplot(gs2[1, :])
        metrics_disp_p2 = {}
        if main_predicted_labels_global is not None and main_predicted_probabilities_global is not None and main_labels_enc.size > 0:
            try:
                metrics_disp_p2["Accuracy"] = accuracy_score(main_labels_enc, main_predicted_labels_global)
                metrics_disp_p2["Balanced Acc."] = balanced_accuracy_score(main_labels_enc, main_predicted_labels_global)
                if len(np.unique(main_labels_enc)) > 1 and hasattr(main_label_encoder, 'classes_') and main_label_encoder.classes_ is not None and len(main_label_encoder.classes_) > 0:
                    avg_m_p2 = 'binary' if len(main_label_encoder.classes_) == 2 else 'weighted'
                    pos_l_p2 = 1 if avg_m_p2 == 'binary' and (1 in main_label_encoder.transform(main_label_encoder.classes_)) else None
                    m_args_p2 = {'y_true': main_labels_enc, 'y_pred': main_predicted_labels_global, 'average': avg_m_p2, 'zero_division': 0}
                    if avg_m_p2 == 'binary' and pos_l_p2 is not None:
                         m_args_p2['pos_label'] = pos_l_p2
                    metrics_disp_p2["Precision"] = precision_score(**m_args_p2)
                    metrics_disp_p2["Recall"] = recall_score(**m_args_p2)
                    metrics_disp_p2["F1-Score"] = f1_score(**m_args_p2)
                    if main_predicted_probabilities_global.ndim == 2 and main_predicted_probabilities_global.shape[1] >= 2:
                        metrics_disp_p2["AUC"] = roc_auc_score(main_labels_enc, main_predicted_probabilities_global[:, 1])
            except Exception as e_metrics_p2:
                logger_4_decoding.warning(f"Error calculating metrics for Page 2: {e_metrics_p2}")
        if metrics_disp_p2:
            names_p2 = list(metrics_disp_p2.keys())
            values_p2 = [metrics_disp_p2[n] for n in names_p2]
            bars_p2 = ax2_metrics.bar(names_p2, values_p2, color='lightcoral')
            ax2_metrics.set_ylim(0, 1.05)
            ax2_metrics.set_ylabel('Score')
            ax2_metrics.set_title('Global performance metrics')
            for bar_p2 in bars_p2:
                ax2_metrics.text(bar_p2.get_x() + bar_p2.get_width() / 2., bar_p2.get_height() + 0.01, f'{bar_p2.get_height():.3f}', ha='center', va='bottom')
            plt.setp(ax2_metrics.get_xticklabels(), rotation=15, ha="right")
        else:
            ax2_metrics.text(0.5, 0.5, 'Global Metrics N/A', ha='center', va='center', transform=ax2_metrics.transAxes)
            ax2_metrics.set_title('Global performance metrics')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if output_directory_path:
            fig2.savefig(os.path.join(output_directory_path, f"dashboard_{subject_identifier}_{group_identifier}_page2_main_global_perf.png"), dpi=150)
    except Exception as e:
        logger_4_decoding.error(f"Error generating Page 2 for dashboard: {e}", exc_info=True)
    finally:
        if fig2: plt.close(fig2)

    # --- Page 3: Main TGM ---
    fig3 = None
    try:
        fig3 = plt.figure(figsize=(10, 8))
        fig3.suptitle(
            f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
            f"Page 3/5:TGM (Temporal Generalisation Matrix)",
            fontsize=16, fontweight="bold"
        )
        ax3_tgm = fig3.add_subplot(111)
        title_p3 = "TGM (mean AUC over folds)"
        if main_mean_temporal_generalization_matrix_scores is not None and main_epochs_time_points is not None and main_epochs_time_points.size > 0 and main_mean_temporal_generalization_matrix_scores.ndim == 2 and main_mean_temporal_generalization_matrix_scores.shape[0] == main_epochs_time_points.size and main_mean_temporal_generalization_matrix_scores.shape[1] == main_epochs_time_points.size and not np.all(np.isnan(main_mean_temporal_generalization_matrix_scores)):
            fdr_tgm_p3 = main_tgm_fdr_sig_data.get('mask') if main_tgm_fdr_sig_data else None
            sig_info_p3 = []
            has_tgm_fdr_sig = fdr_tgm_p3 is not None and np.any(fdr_tgm_p3)
            if has_tgm_fdr_sig:
                sig_info_p3.append("FDR")
            # else: # Removed to avoid text on image, legend will handle it via pretty_gat
            #     ax3_tgm.text(0.02, 0.98, 'FDR: no significance', transform=ax3_tgm.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            if sig_info_p3:
                title_p3 = f"{title_p3}\n(Significance from CV folds: {', '.join(sig_info_p3)} p<0.05)"
            else: # Add note if no significance found
                title_p3 = f"{title_p3}\n(FDR: no significant points)"


            pretty_gat(main_mean_temporal_generalization_matrix_scores, times=main_epochs_time_points, test_times=main_epochs_time_points,
                       chance=chance_level_auc, ax=ax3_tgm, cluster_sig_masks=None, fdr_sig_mask=fdr_tgm_p3,
                       cmap='RdBu_r', colorbar=True, xlabel='Testing Time (s)', ylabel='Training Time (s)')
            ax3_tgm.set_title(title_p3, fontsize=14)
        else:
            ax3_tgm.text(0.5, 0.5, 'Main TGM Scores N/A', ha='center', va='center', transform=ax3_tgm.transAxes)
            ax3_tgm.set_title(title_p3 + " (N/A)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        if output_directory_path:
            fig3.savefig(os.path.join(output_directory_path, f"dashboard_{subject_identifier}_{group_identifier}_page3_main_tgm.png"), dpi=150)
    except Exception as e:
        logger_4_decoding.error(f"Error generating Page 3 for dashboard: {e}", exc_info=True)
    finally:
        if fig3: plt.close(fig3)

    # --- Page 4: Specific Comparisons ---
    fig4 = None
    try:
        if specific_ap_decoding_results and isinstance(specific_ap_decoding_results, list) and \
           any(r.get('scores_1d_mean') is not None or r.get('all_folds_scores_1d') is not None for r in specific_ap_decoding_results) and \
           main_epochs_time_points is not None and main_epochs_time_points.size > 0:
            plot_res_p4 = [r for r in specific_ap_decoding_results if r.get('scores_1d_mean') is not None or r.get('all_folds_scores_1d') is not None]
            if plot_res_p4:
                n_plots_p4 = len(plot_res_p4)
                n_cols_p4 = 2
                n_rows_p4 = (n_plots_p4 + n_cols_p4 - 1) // n_cols_p4
                fig4 = plt.figure(figsize=(7 * n_cols_p4, 5 * n_rows_p4 + 1.5)) # Adjusted height for suptitle
                fig4.suptitle(
                    f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                    f"Page 4/5: Each AP family vs same PP",
                    fontsize=16, fontweight="bold"
                )
                gs4 = GridSpec(n_rows_p4, n_cols_p4, figure=fig4, hspace=0.6, wspace=0.3) # Increased hspace

                for i_p4, res_p4 in enumerate(plot_res_p4):
                    ax_p4 = fig4.add_subplot(gs4[i_p4 // n_cols_p4, i_p4 % n_cols_p4])
                    mean_scores_p4 = res_p4.get('scores_1d_mean')
                    all_folds_scores_p4 = res_p4.get('all_folds_scores_1d')

                    if mean_scores_p4 is not None and main_epochs_time_points.size == mean_scores_p4.size and not np.all(np.isnan(mean_scores_p4)):
                        if all_folds_scores_p4 is not None and all_folds_scores_p4.ndim == 2 and all_folds_scores_p4.shape[1] == main_epochs_time_points.size:
                            for i_fold_spec, fold_scores_spec in enumerate(all_folds_scores_p4):
                                if not np.all(np.isnan(fold_scores_spec)):
                                    ax_p4.plot(main_epochs_time_points, fold_scores_spec, color='gray', alpha=0.3, lw=0.7, label='CV Folds' if i_fold_spec == 0 else None)
                        ax_p4.plot(main_epochs_time_points, mean_scores_p4, color='blue', lw=1.5, label='Mean AUC')
                        if all_folds_scores_p4 is not None and all_folds_scores_p4.shape[0] > 1:
                            sem_spec_1d = scipy_stats.sem(all_folds_scores_p4, axis=0, nan_policy='omit')
                            if sem_spec_1d is not None and not np.all(np.isnan(sem_spec_1d)):
                                ax_p4.fill_between(main_epochs_time_points, mean_scores_p4 - sem_spec_1d, mean_scores_p4 + sem_spec_1d, color='blue', alpha=0.2, label='SEM')
                        
                        s_valid_p4 = mean_scores_p4[~np.isnan(mean_scores_p4)]
                        y_base_s_p4 = min(np.min(s_valid_p4) if s_valid_p4.size > 0 else chance_level_auc, chance_level_auc) - 0.02
                        sig_h_s_p4 = 0.01
                        cur_y_s_p4 = y_base_s_p4
                        fdr_d_s_p4 = res_p4.get('fdr_significance_data')
                        clu_d_s_p4 = res_p4.get('cluster_significance_data')

                        fdr_label_s_p4 = "Mean FDR p<0.05"
                        has_fdr_sig_s_p4 = (fdr_d_s_p4 and fdr_d_s_p4.get('mask') is not None and np.any(fdr_d_s_p4['mask']))
                        if has_fdr_sig_s_p4:
                            ax_p4.fill_between(main_epochs_time_points, cur_y_s_p4 - sig_h_s_p4, cur_y_s_p4, where=fdr_d_s_p4['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_s_p4)
                            cur_y_s_p4 -= (sig_h_s_p4 + 0.005)
                        elif fdr_d_s_p4 and fdr_d_s_p4.get('mask') is not None:
                            ax_p4.plot([], [], color='deepskyblue', alpha=0.7, label="Mean FDR (no significance)")
                        else:
                            ax_p4.plot([], [], color='deepskyblue', alpha=0.7, label="Mean FDR (N/A)")

                        cluster_label_s_p4 = "Mean Cluster p<0.05"
                        has_cluster_sig_s_p4 = (clu_d_s_p4 and clu_d_s_p4.get('mask') is not None and np.any(clu_d_s_p4['mask']))
                        if has_cluster_sig_s_p4:
                            ax_p4.fill_between(main_epochs_time_points, cur_y_s_p4 - sig_h_s_p4, cur_y_s_p4, where=clu_d_s_p4['mask'], color='orangered', alpha=0.7, step='mid', label=cluster_label_s_p4)
                        elif clu_d_s_p4 and clu_d_s_p4.get('mask') is not None:
                            ax_p4.plot([], [], color='orangered', alpha=0.7, label="Mean Cluster (no significance)")
                        else:
                            ax_p4.plot([], [], color='orangered', alpha=0.7, label="Mean Cluster (N/A)")

                        ax_p4.axhline(chance_level_auc, color='k', ls='--', label=f'Chance ({chance_level_auc})')
                        ax_p4.axvline(0, color='r', ls=':', label='Stimulus Onset')
                        ymin_s_p4 = min(cur_y_s_p4 - sig_h_s_p4 - 0.01, (np.min(s_valid_p4) - 0.05 if s_valid_p4.size > 0 else chance_level_auc - 0.15))
                        ymax_s_p4 = max(1.01, (np.max(s_valid_p4) + 0.05 if s_valid_p4.size > 0 else chance_level_auc + 0.15))
                        ax_p4.set_ylim(ymin_s_p4, ymax_s_p4)
                    else:
                        ax_p4.text(0.5, 0.5, 'Scores N/A', ha='center', va='center', transform=ax_p4.transAxes)
                    ax_p4.set_title(res_p4.get('comparison_name', f'Comparison {i_p4 + 1}'), fontsize=11)
                    ax_p4.set_xlabel('Time (s)')
                    ax_p4.set_ylabel('ROC AUC')
                    ax_p4.legend(loc='best', fontsize=8)
                    ax_p4.grid(True, ls=':', alpha=0.5)
                plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Adjusted rect for suptitle
                if output_directory_path:
                    fig4.savefig(os.path.join(output_directory_path, f"dashboard_{subject_identifier}_{group_identifier}_page4_specific_comparisons.png"), dpi=150)
    except Exception as e:
        logger_4_decoding.error(f"Error generating Page 4 for dashboard: {e}", exc_info=True)
    finally:
        if fig4: plt.close(fig4)

    # --- Page 5: Average of Specific Comparisons ---
    fig5 = None
    try:
        if mean_of_specific_scores_1d is not None and main_epochs_time_points is not None and main_epochs_time_points.size == mean_of_specific_scores_1d.size and not np.all(np.isnan(mean_of_specific_scores_1d)):
            fig5 = plt.figure(figsize=(12, 8))
            fig5.suptitle(
                f"Dashboard - Subject: {subject_identifier} ({group_identifier}) - Classifier: {classifier_name_for_title.upper()}\n"
                f"Page 5/5: Average of 6 curves",
                fontsize=16, fontweight="bold"
            )
            ax5_mean = fig5.add_subplot(111)
            n_curves_avg_p5 = len(specific_ap_decoding_results) if specific_ap_decoding_results else "N/A"
            ax5_mean.plot(main_epochs_time_points, mean_of_specific_scores_1d, color='black', lw=2, label=f'Average of {n_curves_avg_p5} Specific Tasks')
            if sem_of_specific_scores_1d is not None and not np.all(np.isnan(sem_of_specific_scores_1d)):
                ax5_mean.fill_between(main_epochs_time_points, mean_of_specific_scores_1d - sem_of_specific_scores_1d, mean_of_specific_scores_1d + sem_of_specific_scores_1d, color='black', alpha=0.2, label='SEM (across tasks)')
            
            s_valid_m_p5 = mean_of_specific_scores_1d[~np.isnan(mean_of_specific_scores_1d)]
            y_base_m_p5 = min(np.min(s_valid_m_p5) if s_valid_m_p5.size > 0 else chance_level_auc, chance_level_auc) - 0.02
            sig_h_m_p5 = 0.01
            cur_y_m_p5 = y_base_m_p5

            fdr_label_m_p5 = "FDR p<0.05 (on stack)"
            has_fdr_sig_m_p5 = (mean_specific_fdr_sig_data and mean_specific_fdr_sig_data.get('mask') is not None and np.any(mean_specific_fdr_sig_data['mask']))
            if has_fdr_sig_m_p5:
                ax5_mean.fill_between(main_epochs_time_points, cur_y_m_p5 - sig_h_m_p5, cur_y_m_p5, where=mean_specific_fdr_sig_data['mask'], color='deepskyblue', alpha=0.7, step='mid', label=fdr_label_m_p5)
                cur_y_m_p5 -= (sig_h_m_p5 + 0.005)
            elif mean_specific_fdr_sig_data and mean_specific_fdr_sig_data.get('mask') is not None:
                ax5_mean.plot([], [], color='deepskyblue', alpha=0.7, label="FDR (no significance on stack)")
            else:
                ax5_mean.plot([], [], color='deepskyblue', alpha=0.7, label="FDR (N/A on stack)")

            cluster_label_m_p5 = "Cluster p<0.05 (on stack)"
            has_cluster_sig_m_p5 = (mean_specific_cluster_sig_data and mean_specific_cluster_sig_data.get('mask') is not None and np.any(mean_specific_cluster_sig_data['mask']))
            if has_cluster_sig_m_p5:
                ax5_mean.fill_between(main_epochs_time_points, cur_y_m_p5 - sig_h_m_p5, cur_y_m_p5, where=mean_specific_cluster_sig_data['mask'], color='orangered', alpha=0.7, step='mid', label=cluster_label_m_p5)
            elif mean_specific_cluster_sig_data and mean_specific_cluster_sig_data.get('mask') is not None:
                ax5_mean.plot([], [], color='orangered', alpha=0.7, label="Cluster (no significance on stack)")
            else:
                ax5_mean.plot([], [], color='orangered', alpha=0.7, label="Cluster (N/A on stack)")

            ax5_mean.axhline(chance_level_auc, color='k', ls='--', label=f'Chance ({chance_level_auc})')
            ax5_mean.axvline(0, color='r', ls=':', label='Stimulus Onset')
            ymin_m_p5 = min(cur_y_m_p5 - sig_h_m_p5 - 0.01, (np.min(s_valid_m_p5) - 0.05 if s_valid_m_p5.size > 0 else chance_level_auc - 0.15))
            ymax_m_p5 = max(1.01, (np.max(s_valid_m_p5) + 0.05 if s_valid_m_p5.size > 0 else chance_level_auc + 0.15))
            ax5_mean.set_ylim(ymin_m_p5, ymax_m_p5)
            ax5_mean.set_title(f"Average temporal decoding across {n_curves_avg_p5} specific tasks")
            ax5_mean.set_xlabel('Time (s)')
            ax5_mean.set_ylabel('Average ROC AUC')
            ax5_mean.legend(loc='best')
            ax5_mean.grid(True, ls=':', alpha=0.5)
            plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Adjusted rect for suptitle
            if output_directory_path:
                fig5.savefig(os.path.join(output_directory_path, f"dashboard_{subject_identifier}_{group_identifier}_page5_mean_specific_comparisons.png"), dpi=150)
        else:
             logger_4_decoding.info(f"Skipping Page 5 for subject {subject_identifier}: mean_of_specific_scores_1d is not available or invalid.")
    except Exception as e:
        logger_4_decoding.error(f"Error generating Page 5 for dashboard: {e}", exc_info=True)
    finally:
        if fig5: plt.close(fig5)

    logger_4_decoding.info(f"Finished dashboard plots for Subject: {subject_identifier}")
    return output_directory_path

def plot_group_mean_scores_barplot(
    subject_to_score_mapping, # Dict: {subject_id: score}
    group_identifier_for_plot_title,
    output_directory_path=None,
    score_metric_name="ROC AUC",
    chance_level_value=DEFAULT_CHANCE_LEVEL_AUC, # Renamed to avoid conflict
):
    """
    Plots a bar chart of scores for each subject in a group.

    Args:
        subject_to_score_mapping (dict): Dictionary mapping subject IDs to scores.
        group_identifier_for_plot_title (str): Title for the plot, identifying the group.
        output_directory_path (str, optional): Directory to save the plot.
        score_metric_name (str): Name of the score metric being plotted (e.g., "ROC AUC").
        chance_level_value (float): Value of the chance level to draw as a line.
    """
    # Filter out None or NaN scores
    valid_scores_dict = {
        k: v for k, v in subject_to_score_mapping.items() if v is not None and not np.isnan(v)
    }
    if not valid_scores_dict:
        logger_4_decoding.warning(
            f"No valid scores to plot for barplot: {group_identifier_for_plot_title}"
        )
        return

    subject_ids_bp = list(valid_scores_dict.keys())
    scores_bp = list(valid_scores_dict.values())
    n_subjects_bp = len(subject_ids_bp)

    logger_4_decoding.info(
        f"Plotting group barplot for {group_identifier_for_plot_title} ({n_subjects_bp} valid subjects)"
    )
    plt.switch_backend("Agg")
    # Adjust figure width based on number of subjects for readability
    fig_width_bp = max(8, n_subjects_bp * 0.5 + 2)
    fig_bp, ax_bp = plt.subplots(figsize=(fig_width_bp, 7))

    bars_bp = ax_bp.bar(range(n_subjects_bp), scores_bp, color="teal", alpha=0.7,
                      label=f"Subject {score_metric_name}")
    mean_score_bp = np.mean(scores_bp)
    ax_bp.axhline(mean_score_bp, color="darkred", ls="--",
                  label=f"Group Mean: {mean_score_bp:.3f}")
    ax_bp.axhline(chance_level_value, color="black", ls=":",
                  label=f"Chance ({chance_level_value})")

    ax_bp.set_ylabel(f"Score ({score_metric_name})")
    ax_bp.set_xlabel("Subject ID")
    ax_bp.set_title(
        f"Individual subject performance: {group_identifier_for_plot_title}", fontweight="bold"
    )
    ax_bp.set_xticks(range(n_subjects_bp))
    ax_bp.set_xticklabels(subject_ids_bp, rotation=60, ha="right", fontsize=10)

    # Adjust y-limits for better visualization
    min_score_val_bp = np.min(scores_bp) if scores_bp else chance_level_value - 0.1
    max_score_val_bp = np.max(scores_bp) if scores_bp else chance_level_value + 0.1
    ax_bp.set_ylim(min(chance_level_value - 0.15, min_score_val_bp - 0.05),
                   max(1.0, max_score_val_bp + 0.1))

    # Add text labels on bars
    for bar_item_bp in bars_bp:
        height_bp = bar_item_bp.get_height()
        ax_bp.text(bar_item_bp.get_x() + bar_item_bp.get_width() / 2.0, height_bp + 0.01,
                   f"{height_bp:.3f}", ha="center", va="bottom", fontsize=8)

    ax_bp.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax_bp.grid(True, axis="y", ls=":", alpha=0.6)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for title and labels

    if output_directory_path:
        os.makedirs(output_directory_path, exist_ok=True)
        # Sanitize filename
        safe_fname_bp = "".join(
            c if c.isalnum() else "_" for c in group_identifier_for_plot_title.replace(" ", "_")
        )
        save_path_bp = os.path.join(output_directory_path,
                                  f"group_summary_scores_barplot_{safe_fname_bp}.png")
        try:
            fig_bp.savefig(save_path_bp, dpi=150)
            logger_4_decoding.info(f"Barplot saved: {save_path_bp}")
        except Exception as e_save_bp:
            logger_4_decoding.error(f"Failed to save barplot {save_path_bp}: {e_save_bp}")
    plt.close(fig_bp)


__all__ = [
    "run_temporal_decoding_analysis",
    "run_cross_subject_decoding_for_fold",
    "create_subject_decoding_dashboard_plots",
    "plot_group_mean_scores_barplot",
    "_build_standard_classifier_pipeline",
    "pretty_gat",
    "pretty_plot",
    "pretty_colorbar",
    "_set_ticks",
]