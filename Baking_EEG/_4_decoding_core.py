#tom

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, GroupKFold, GridSearchCV
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from mne.parallel import parallel_func
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score)
import scipy.stats
from base.base_decoding import (_build_standard_classifier_pipeline,
                                 _execute_global_decoding_for_one_fold)
from utils import stats_utils as bEEG_stats

from config.decoding_config import (DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL,
                                     CHANCE_LEVEL_AUC,
                                     INTERNAL_N_JOBS_FOR_MNE_DECODING,
                                     USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
                                     USE_GRID_SEARCH,
                                     COMPUTE_TEMPORAL_GENERALIZATION_MATRICES)

logger_decoding_core = logging.getLogger(__name__) 

def run_temporal_decoding_analysis(
    epochs_data,  # (n_trials, n_channels, n_times)
    target_labels,  # (n_trials,)
    classifier_model_type=DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL,
    use_grid_search=USE_GRID_SEARCH,
    use_anova_fs_for_temporal_pipelines=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config=None,  
    cv_folds_for_gridsearch=3,
    fixed_classifier_params=None,  
    cross_validation_splitter=5,  
    trial_sample_weights="auto",  
    n_jobs_external=1,  
    group_labels_for_cv=None,  
    compute_intra_fold_stats=True,
    chance_level=CHANCE_LEVEL_AUC,
    n_permutations_for_intra_fold_clusters=1000,
    compute_temporal_generalization_matrix=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    cluster_threshold_config_intra_fold=None,
    random_state=42):
    """
    Core function / backend of decoding

    
    """
    
   
    if isinstance(use_anova_fs_for_temporal_pipelines, tuple):
        logger_decoding_core.warning(
            " conversion en booléen",
            use_anova_fs_for_temporal_pipelines
        )
        use_anova_fs_for_temporal_pipelines = bool(use_anova_fs_for_temporal_pipelines[0]) if use_anova_fs_for_temporal_pipelines else False
    
    
    logger_decoding_core.info("--- Temporal Ddcoding Analysis ---")
    logger_decoding_core.info(
        "Clf: %s, GS: %s, ANOVA FS (temporal): %s",
        classifier_model_type, use_grid_search,
        use_anova_fs_for_temporal_pipelines
    )

  
    if not isinstance(epochs_data, np.ndarray) or epochs_data.ndim != 3:
        logger_decoding_core.error("epochs_data must be a 3D NumPy array.")
        return (None,) * 12  # Return tuple of Nones matching expected output
    if not isinstance(target_labels, np.ndarray) or target_labels.ndim != 1 or \
       target_labels.shape[0] != epochs_data.shape[0]:
        logger_decoding_core.error("target_labels mismatch with epochs_data.")
        return (None,) * 12

    n_trials, n_channels, n_time_points = epochs_data.shape
    label_enc = LabelEncoder()
    target_labels_enc = label_enc.fit_transform(target_labels)
    n_classes = len(label_enc.classes_)

    # Ensure probas has at least 2 columns even for binary case if n_classes=1 (error case)
    empty_probas = np.zeros(
        (n_trials, max(2, n_classes if n_classes >= 2 else 2)))
    empty_labels = np.zeros(n_trials, dtype=int)
    empty_1d_scores = np.full(
        n_time_points, np.nan) if n_time_points > 0 else np.array([])
    empty_tgm_scores = (np.full((n_time_points, n_time_points), np.nan)
                        if n_time_points > 0 else np.array([[]]))

    empty_results_tuple = (
        empty_probas, empty_labels, np.array(
            []), empty_1d_scores, {}, None, None,
        (np.full((1, n_time_points), np.nan) if n_time_points > 0
         else np.array([[]])),  # scores_1d_all_folds
        empty_tgm_scores, None, None,
        (np.full((1, n_time_points, n_time_points), np.nan) if n_time_points > 0
         else np.array([[[]]])))  # tgm_all_folds

    if n_classes < 2:
        logger_decoding_core.error(
            "Need at least 2 classes for decoding. Found %d.", n_classes)
        return empty_results_tuple
    if n_time_points == 0:
        logger_decoding_core.error(
            "epochs_data has zero time points. Cannot proceed.")
        return empty_results_tuple

  
    if isinstance(use_anova_fs_for_temporal_pipelines, tuple):
        use_anova_fs_for_temporal_pipelines = use_anova_fs_for_temporal_pipelines[0] if len(use_anova_fs_for_temporal_pipelines) > 0 else False
        
    # --- Prepare classifier/pipeline for MNE (Sliding/Generalizing) ---
    
    logger_decoding_core.info("Building temporal pipeline with FeatureSelection support")
    pipeline_mne, clf_name_mne, fs_name_mne, _ = \
        _build_standard_classifier_pipeline(
            classifier_model_type=classifier_model_type,
            use_grid_search=use_grid_search,
            add_anova_fs_step=use_anova_fs_for_temporal_pipelines,  # FS is compatible with 2D data
            **(fixed_classifier_params if fixed_classifier_params and not use_grid_search else {})
        )

    final_estimator_mne = None
    if use_grid_search:
        current_grid_mne = {}
        if param_grid_config and classifier_model_type in param_grid_config:
            # Filter grid for relevant steps in pipeline_mne
            full_grid_for_clf = param_grid_config[classifier_model_type]
            current_grid_mne = {
                k: v for k, v in full_grid_for_clf.items()
                if k.startswith(f"{clf_name_mne}__") or
                (fs_name_mne and k.startswith(f"{fs_name_mne}__"))
            }
            logger_decoding_core.info(" param_grid for MNE GS: %s",
                                   current_grid_mne)
        else:
            logger_decoding_core.warning(
                "No param_grid_config for '%s' for MNE GS. Using basic defaults.",
                classifier_model_type
            )

            current_grid_mne = {f'{clf_name_mne}__C': [0.1, 1, 10]}  
            if fs_name_mne:
                current_grid_mne[f'{fs_name_mne}__percentile'] = [15, 30]

        if not current_grid_mne:
            logger_decoding_core.error(
                "No grid for MNE GS (%s). Aborting.", classifier_model_type)
            return empty_results_tuple
        # Ensure n_jobs for inner GS CV is 1 if outer MNE is parallel
        gs_cv_n_jobs = 1 if INTERNAL_N_JOBS_FOR_MNE_DECODING != 1 else n_jobs_external
        final_estimator_mne = GridSearchCV(
            estimator=pipeline_mne, param_grid=current_grid_mne, scoring='roc_auc',
            cv=cv_folds_for_gridsearch, n_jobs=gs_cv_n_jobs, verbose=0, refit=True,
            error_score='raise' if logger_decoding_core.getEffectiveLevel() <= logging.DEBUG else np.nan
        )
    else:
        final_estimator_mne = pipeline_mne

    # --- Prepare classifier/pipeline for global decoding ---
    logger_decoding_core.info("Building global pipeline with FeatureSelection support")
    pipeline_global, clf_name_global, fs_name_global, _ = \
        _build_standard_classifier_pipeline(
            classifier_model_type=classifier_model_type,
            use_grid_search=use_grid_search,
            add_anova_fs_step=use_anova_fs_for_temporal_pipelines,  # Can use FS for global decoding
            **(fixed_classifier_params if fixed_classifier_params and not use_grid_search else {})
        )
    final_estimator_global = None
    if use_grid_search:
        current_grid_global = {}
        if param_grid_config and classifier_model_type in param_grid_config:
            full_grid_for_clf = param_grid_config[classifier_model_type]
            current_grid_global = {
                k: v for k, v in full_grid_for_clf.items()
                # Include classifier and FS params for global decoding
                if k.startswith(f"{clf_name_global}__") or
                (fs_name_global and k.startswith(f"{fs_name_global}__"))
            }
            logger_decoding_core.info("Using (classifier+FS) param_grid for Global GS: %s",
                                   current_grid_global)
        else:
            logger_decoding_core.warning(
                "No param_grid_config for '%s' for Global GS. Using basic defaults.",
                classifier_model_type
            )
            current_grid_global = {
                f'{clf_name_global}__C': [0.1, 1, 10]}
            
            if fs_name_global:
                current_grid_global[f'{fs_name_global}__percentile'] = [15, 30]
        if not current_grid_global:
            logger_decoding_core.error(
                "No grid for Global GS (%s). Aborting.", classifier_model_type)
            return empty_results_tuple
        gs_cv_n_jobs_global = 1 if n_jobs_external != 1 else -1
        final_estimator_global = GridSearchCV(
            estimator=pipeline_global, param_grid=current_grid_global, scoring='roc_auc',
            cv=cv_folds_for_gridsearch, n_jobs=gs_cv_n_jobs_global, verbose=0, refit=True,
            error_score='raise' if logger_decoding_core.getEffectiveLevel() <= logging.DEBUG else np.nan
        )
    else:
        final_estimator_global = pipeline_global

    # --- CV splitter setup ---
    actual_cv_splitter = None
    if isinstance(cross_validation_splitter, int):
        n_splits_req = cross_validation_splitter
        if group_labels_for_cv is not None and len(np.unique(group_labels_for_cv)) >= 2:
            n_groups = len(np.unique(group_labels_for_cv))
            if n_splits >= 2:
                actual_cv_splitter = GroupKFold(n_splits=n_splits)
            else: 
                logger_decoding_core.warning(
                    f"Cannot perform GroupKFold with {n_splits_req} splits for {n_groups} groups. Will attempt RepeatedStratifiedKFold if possible.")
            
        # Fallback to RepeatedStratifiedKFold if GroupKFold not applicable or failed
        if actual_cv_splitter is None:
            min_class_count = np.min(np.bincount(target_labels_enc))
            # n_splits for RepeatedStratifiedKFold cannot exceed min_class_count
            n_splits = min(
                n_splits_req, min_class_count) if min_class_count >= 2 else 0
            if n_splits >= 2:
                actual_cv_splitter = RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=5, random_state=42)
    elif (hasattr(cross_validation_splitter, "split") and 
          hasattr(cross_validation_splitter, "get_n_splits")):
        actual_cv_splitter = cross_validation_splitter  # Use provided splitter

    if not actual_cv_splitter or actual_cv_splitter.get_n_splits(X=epochs_data, y=target_labels_enc, groups=group_labels_for_cv) < 2:
        logger_decoding_core.error(
            "Invalid CV setup: less than 2 splits possible with chosen strategy and data. Aborting."
        )
        return empty_results_tuple

    logger_decoding_core.info("CV strategy: %s with %d splits.",
                           type(actual_cv_splitter).__name__, actual_cv_splitter.get_n_splits(X=epochs_data, y=target_labels_enc, groups=group_labels_for_cv))
    num_cv_splits = actual_cv_splitter.get_n_splits(
        X=epochs_data, y=target_labels_enc, groups=group_labels_for_cv)

    # --- Sample weights strategy ---
    # Instead of calculating global weights, we'll determine the strategy and calculate per-fold
    use_fold_specific_weights = False
    effective_sample_weights = None
    
    if isinstance(trial_sample_weights, str) and trial_sample_weights.lower() == "auto":
        use_fold_specific_weights = True
        logger_decoding_core.info("Sample weights: Auto-calculated per fold (recommended for RepeatedStratifiedKFold).")
    elif isinstance(trial_sample_weights, np.ndarray):
        if trial_sample_weights.shape == (n_trials,):
            effective_sample_weights = trial_sample_weights
            logger_decoding_core.info("Sample weights: Provided by user (global weights).")
        else:
            logger_decoding_core.warning(
                "Provided sample_weights shape mismatch. Disabling sample weights.")
    else:
        logger_decoding_core.info("Sample weights: Disabled.")

    # Initialize result holders
    scores_1d_all_folds = np.full((num_cv_splits, n_time_points), np.nan)
    mean_scores_1d = np.full(n_time_points, np.nan)
    tgm_all_folds = np.full(
        (num_cv_splits, n_time_points, n_time_points), np.nan)
    mean_tgm = np.full((n_time_points, n_time_points), np.nan)
    fdr_1d_data, cluster_1d_data, fdr_tgm_data = None, None, None
    # Global results
    # Ensure probas has at least 2 columns for binary case
    probas_global_agg = np.zeros((n_trials, max(2, n_classes)))
    labels_global_agg = np.zeros(n_trials, dtype=int)
    cv_global_scores = np.array([])
    global_metrics = {}

    # --- MNE-based Temporal and TGM Decoding ---
    cv_groups_mne = group_labels_for_cv if isinstance(
        actual_cv_splitter, GroupKFold) else None
    fit_params_mne = {}
    
    # Sample weights for MNE estimators:
    # For fold-specific weights, we need to use class_weight='balanced' in classifier
    # or implement custom CV handling
    if use_fold_specific_weights:
        # For fold-specific weights, we recommend using 'class_weight=balanced' in classifier
        # MNE's cross_val_multiscore will handle this automatically per fold
        logger_decoding_core.info(
            "MNE Estimators: Using fold-specific balancing via class_weight='balanced' (if supported by classifier).")
       
    elif effective_sample_weights is not None:
        if not use_grid_search:
            if clf_name_mne:  # Ensure classifier step name is known
                # For global weights, pass them to MNE
                fit_params_mne[f"{clf_name_mne}__sample_weight"] = effective_sample_weights
                logger_decoding_core.info(
                    "MNE Estimators (Fixed mode): Passing global sample_weight to %s.", clf_name_mne)
        else:  # Using GridSearchCV
            # GridSearchCV with global weights - pass to fit_params
            logger_decoding_core.info(
                "MNE Estimators (GridSearch mode): Using global sample weights in fit_params.")

    # 1D Temporal Decoding (SlidingEstimator)
    logger_decoding_core.info(
        "Starting 1D temporal decoding (SlidingEstimator)...")
    sliding_decoder = SlidingEstimator(
        base_estimator=clone(final_estimator_mne),
        n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING, scoring="roc_auc", verbose=False
    )
    try:
        # Pass fit_params_mne only if it's not empty
        scores_1d_cv_raw = cross_val_multiscore(
            sliding_decoder, epochs_data, target_labels_enc, cv=actual_cv_splitter,
            groups=cv_groups_mne,
            fit_params=fit_params_mne if fit_params_mne else None,
            n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING  # MNE handles internal parallelism
        )
        if scores_1d_cv_raw is not None and len(scores_1d_cv_raw) > 0:
            scores_1d_all_folds_temp = np.array(
                scores_1d_cv_raw)  # Store temporarily
            if scores_1d_all_folds_temp.ndim == 2 and \
               scores_1d_all_folds_temp.shape == (num_cv_splits, n_time_points):
                scores_1d_all_folds = scores_1d_all_folds_temp
                mean_scores_1d = np.nanmean(scores_1d_all_folds, axis=0)
            else:
                logger_decoding_core.error(
                    "Shape mismatch for 1D temporal scores from CV. Expected (%d, %d), got %s.",
                    num_cv_splits, n_time_points, scores_1d_all_folds_temp.shape
                )
        peak_auc = np.nanmax(mean_scores_1d) if mean_scores_1d.size > 0 and not np.all(
            np.isnan(mean_scores_1d)) else "N/A"
        logger_decoding_core.info("1D temporal decoding done. Peak mean AUC: %s",
                               f"{peak_auc:.3f}" if isinstance(peak_auc, float) else peak_auc)
    except Exception as e_temporal:
        logger_decoding_core.error(
            "Error in 1D temporal decoding: %s", e_temporal, exc_info=True)

    # TGM Decoding (GeneralizingEstimator)
    if compute_temporal_generalization_matrix:
        logger_decoding_core.info(
            "Starting TGM computation (GeneralizingEstimator)...")
        tgm_decoder = GeneralizingEstimator(
            clone(final_estimator_mne), n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING,
            scoring="roc_auc", verbose=False
        )
        try:
            tgm_cv_raw = cross_val_multiscore(
                tgm_decoder, epochs_data, target_labels_enc, cv=actual_cv_splitter,
                groups=cv_groups_mne,
                fit_params=fit_params_mne if fit_params_mne else None,
                n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING
            )
            # Check if tgm_cv_raw is not empty
            if tgm_cv_raw is not None and len(tgm_cv_raw) > 0:
                tgm_all_folds_temp = np.array(tgm_cv_raw)  # Store temporarily
                if tgm_all_folds_temp.ndim == 3 and \
                   tgm_all_folds_temp.shape == (num_cv_splits, n_time_points, n_time_points):
                    tgm_all_folds = tgm_all_folds_temp
                    mean_tgm = np.nanmean(tgm_all_folds, axis=0)
                else:
                    logger_decoding_core.error(
                        "Shape mismatch for TGM scores from CV. Expected (%d, %d, %d), got %s.",
                        num_cv_splits, n_time_points, n_time_points, tgm_all_folds_temp.shape
                    )
            peak_tgm_auc = np.nanmax(mean_tgm) if mean_tgm.size > 0 and not np.all(
                np.isnan(mean_tgm)) else "N/A"
            logger_decoding_core.info("TGM computation done. Peak mean TGM AUC: %s",
                                   f"{peak_tgm_auc:.3f}" if isinstance(peak_tgm_auc, float) else peak_tgm_auc)
        except Exception as e_tgm:
            logger_decoding_core.error(
                "Error in TGM computation: %s", e_tgm, exc_info=True)
    else:
        logger_decoding_core.info("TGM computation disabled.")

    # --- Global Decoding (Flattened Features) ---
    logger_decoding_core.info(
        "Starting global decoding (n_jobs_folds=%s)...", n_jobs_external)
    epochs_flat_global = epochs_data.reshape(
        n_trials, -1)  # Flatten channels and times
    parallel_global, pfunc_global, _ = parallel_func(
        _execute_global_decoding_for_one_fold, n_jobs=n_jobs_external, verbose=0
    )   
    cv_splits_global = list(actual_cv_splitter.split(
        epochs_flat_global, target_labels_enc, groups=cv_groups_mne
    ))

    global_fold_outputs = parallel_global(
        pfunc_global(
            clone(final_estimator_global),  # Fresh clone for each fold
            epochs_flat_global,
            target_labels_enc,
            train_idx,
            test_idx,
            effective_sample_weights if not use_fold_specific_weights else "auto",  # Pass "auto" for fold-specific
            clf_name_global if use_grid_search else None  # Name of clf step if GS, for fit_params
        ) for train_idx, test_idx in cv_splits_global
    )

    cv_scores_list_g = []
    for i_fold, (_, test_idx_g) in enumerate(cv_splits_global):
        if i_fold < len(global_fold_outputs):
            probas_f, preds_f, score_f = global_fold_outputs[i_fold]
            # Check if probas_f is valid before assignment
            if isinstance(probas_f, np.ndarray) and probas_f.ndim == 2 and probas_f.shape[0] == len(test_idx_g):
                if probas_f.shape[1] == probas_global_agg.shape[1]:
                    probas_global_agg[test_idx_g] = probas_f
                elif probas_f.shape[1] == 1 and probas_global_agg.shape[1] == 2: 
                    probas_global_agg[test_idx_g, 1] = probas_f.ravel()
                    probas_global_agg[test_idx_g, 0] = 1 - probas_f.ravel()
                else:
                    logger_decoding_core.warning(
                        f"Fold {i_fold}: Proba shape mismatch. Agg shape: {probas_global_agg.shape[1]}, fold shape: {probas_f.shape[1]}. Skipping proba aggregation for this fold.")
            else:
                logger_decoding_core.warning(
                    f"Fold {i_fold}: Invalid probas_f. Skipping proba aggregation.")

            if isinstance(preds_f, np.ndarray) and preds_f.shape[0] == len(test_idx_g):
                labels_global_agg[test_idx_g] = preds_f
            else:
                logger_decoding_core.warning(
                    f"Fold {i_fold}: Invalid preds_f. Skipping label aggregation.")
            cv_scores_list_g.append(score_f)
        else:  # Should not happen if parallel_func returns all results
            logger_decoding_core.error(
                f"Missing output for global fold {i_fold}.")
            cv_scores_list_g.append(np.nan)

    cv_global_scores = np.array(cv_scores_list_g)
    mean_auc_g = np.nanmean(cv_global_scores) if cv_global_scores.size > 0 and not np.all(
        np.isnan(cv_global_scores)) else "N/A"
    std_auc_g = np.nanstd(cv_global_scores) if cv_global_scores.size > 0 and not np.all(
        np.isnan(cv_global_scores)) else "N/A"
    logger_decoding_core.info(
        "Global decoding done. Mean CV AUC: %s +/- %s",
        f"{mean_auc_g:.3f}" if isinstance(mean_auc_g, float) else mean_auc_g,
        f"{std_auc_g:.3f}" if isinstance(std_auc_g, float) else std_auc_g
    )

    try:  # Calculate overall metrics from aggregated predictions
        # Ensure probas_global_agg is valid for roc_auc_score
        if probas_global_agg.shape[0] == n_trials and \
           probas_global_agg.ndim == 2 and probas_global_agg.shape[1] >= 2 and \
           len(np.unique(target_labels_enc)) > 1 and \
           not np.all(np.isnan(probas_global_agg)):  # Check for NaNs
            global_metrics["roc_auc"] = roc_auc_score(
                target_labels_enc, probas_global_agg[:, 1], average="weighted",
                sample_weight=effective_sample_weights
            )
        else:
            global_metrics["roc_auc"] = np.nan
            logger_decoding_core.debug(
                "Skipping overall global ROC AUC calculation due to invalid aggregated probabilities or labels."
            )

        avg_mode = "binary" if n_classes == 2 else "weighted"
        # For binary, pos_label should be the encoded value of the positive class.
        # Assuming class 1 is the positive class after LabelEncoding [0, 1]
        pos_label_val = 1 if avg_mode == "binary" else None

        m_args_common = {"y_true": target_labels_enc, "y_pred": labels_global_agg,
                         "sample_weight": effective_sample_weights, "zero_division": 0}

        global_metrics.update({
            "accuracy": accuracy_score(target_labels_enc, labels_global_agg,
                                       sample_weight=effective_sample_weights),
            "precision": precision_score(**m_args_common, average=avg_mode, pos_label=pos_label_val) if avg_mode == "binary" else precision_score(**m_args_common, average=avg_mode),
            "recall": recall_score(**m_args_common, average=avg_mode, pos_label=pos_label_val) if avg_mode == "binary" else recall_score(**m_args_common, average=avg_mode),
            "f1_score": f1_score(**m_args_common, average=avg_mode, pos_label=pos_label_val) if avg_mode == "binary" else f1_score(**m_args_common, average=avg_mode),
            "balanced_accuracy": balanced_accuracy_score(target_labels_enc, labels_global_agg,
                                                         sample_weight=effective_sample_weights)
        })
        logger_decoding_core.info("  Overall global metrics (aggregated): %s",
                               {k: f"{v:.3f}" for k, v in global_metrics.items() if pd.notna(v)})
    except Exception as e_metrics:
        logger_decoding_core.error(
            "Error calculating global metrics: %s", e_metrics, exc_info=True)
        # Fill with NaNs if calculation fails
        for metric_key in ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]:
            if metric_key not in global_metrics:
                global_metrics[metric_key] = np.nan

    # --- Intra-fold Statistics ---
    if compute_intra_fold_stats and scores_1d_all_folds.shape[0] > 1 and n_time_points > 0:
        if not np.all(np.isnan(scores_1d_all_folds)):
            _, fdr_mask_1d, fdr_p_1d, fdr_test_info = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                scores_1d_all_folds, chance_level, alternative_hypothesis="greater",
                statistical_test_type="wilcoxon"  # Force Wilcoxon test
            )
            fdr_1d_data = {"mask": fdr_mask_1d,
                "p_values": fdr_p_1d, "p_values_raw": fdr_test_info.get("p_values_raw", fdr_p_1d),
                "method": "FDR_CV_Folds_1D_Wilcoxon", "test_info": fdr_test_info}

            t_obs_clu, clu_1d, p_clu_1d, clu_info = bEEG_stats.perform_cluster_permutation_test(
            scores_1d_all_folds, 
            chance_level, 
            n_permutations_for_intra_fold_clusters,
            cluster_threshold_config_intra_fold, 
            "greater", 
            INTERNAL_N_JOBS_FOR_MNE_DECODING,
            random_seed=random_state,
        )
            combined_mask_clu1d = np.zeros(n_time_points, dtype=bool)
            sig_clu_obj1d = []
            if clu_1d and p_clu_1d is not None: # clu_1d is a list of boolean masks
                for i, cluster_mask in enumerate(clu_1d): # Each item is already a boolean mask
                    if p_clu_1d[i] < 0.05:
                        sig_clu_obj1d.append(cluster_mask) # Store the boolean mask directly
                        combined_mask_clu1d = np.logical_or(combined_mask_clu1d, cluster_mask)

            cluster_1d_data = {"mask": combined_mask_clu1d, "cluster_objects": sig_clu_obj1d,
                               "p_values_all_clusters": p_clu_1d, "cluster_info": clu_info,
                               "method": "CluPerm_CV_Folds_1D_Wilcoxon"}

        if compute_temporal_generalization_matrix and not np.all(np.isnan(tgm_all_folds)) and \
           tgm_all_folds.shape[0] > 1:
            n_f, n_tr, n_te = tgm_all_folds.shape
            if n_tr > 0 and n_te > 0:  # Ensure dimensions are positive
                flat_tgm_scores = tgm_all_folds.reshape(n_f, n_tr * n_te)
                _, fdr_mask_tgm_flat, pvals_tgm_flat, fdr_test_info_tgm = \
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        flat_tgm_scores, chance_level, alternative_hypothesis="greater",
                        statistical_test_type="wilcoxon"  # Force Wilcoxon test
                    )
                fdr_tgm_data = {
                    "mask": (fdr_mask_tgm_flat.reshape(n_tr, n_te)
                             if hasattr(fdr_mask_tgm_flat, 'reshape') and fdr_mask_tgm_flat is not None else None),
                    "p_values": (pvals_tgm_flat.reshape(n_tr, n_te)
                                 if hasattr(pvals_tgm_flat, 'reshape') and pvals_tgm_flat is not None else None),
                    "p_values_raw": (fdr_test_info_tgm.get("p_values_raw", pvals_tgm_flat).reshape(n_tr, n_te)
                                   if hasattr(pvals_tgm_flat, 'reshape') and pvals_tgm_flat is not None else None),
                    "method": "FDR_CV_Folds_TGM_Wilcoxon",
                    "test_info": fdr_test_info_tgm
                }
            else:
                logger_decoding_core.warning(
                    "TGM dimensions are zero, skipping FDR on TGM.")
            # Cluster perm on TGM is usually skipped for intra-fold due to computational cost
            logger_decoding_core.info(
                "Intra-fold Cluster Permutation for TGM is SKIPPED.")
    else:
        logger_decoding_core.info(
            "Skipping intra-fold stats (flag disabled or not enough folds/timepoints).")

    return (
        probas_global_agg, labels_global_agg, cv_global_scores,
        mean_scores_1d, global_metrics,
        fdr_1d_data, cluster_1d_data,
        scores_1d_all_folds, mean_tgm,
        fdr_tgm_data, None,  # Placeholder for TGM cluster data
        tgm_all_folds
    )

def calculate_fold_sample_weights(train_labels_enc):
    """Calculate sample weights for a specific training fold.
    
    This ensures that class balancing is computed based on the actual
    class distribution in each training fold, which is more accurate
    than using global weights when using RepeatedStratifiedKFold.
    
    Args:
        train_labels_enc (np.ndarray): Encoded labels for training fold
        
    Returns:
        np.ndarray: Sample weights for the training fold
    """
    n_trials_fold = len(train_labels_enc)
    unique_classes = np.unique(train_labels_enc)
    n_classes_fold = len(unique_classes)
    
    if n_classes_fold < 2:
        # Single class, return uniform weights
        return np.ones(n_trials_fold)
    
    # Count occurrences of each class in this fold
    class_counts_fold = np.bincount(train_labels_enc, minlength=n_classes_fold)
    
    # Calculate balanced weights: inverse frequency weighting
    weights_map_fold = {}
    for cls_idx in unique_classes:
        count = class_counts_fold[cls_idx]
        if count > 0:
            weights_map_fold[cls_idx] = n_trials_fold / (n_classes_fold * count)
        else:
            weights_map_fold[cls_idx] = 0.0
    
    # Apply weights to each sample
    sample_weights_fold = np.array([
        weights_map_fold[label] for label in train_labels_enc
    ])
    
    return sample_weights_fold
