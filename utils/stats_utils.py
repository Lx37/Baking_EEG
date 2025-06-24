import logging
import os
import numpy as np
import scipy.stats
from mne.stats import (
    permutation_cluster_1samp_test,
    ttest_1samp_no_p,
    fdr_correction,
)

logger_stats_decoding = logging.getLogger(__name__)


def _custom_stat_function_ttest_1samp(
    data_array,
    sigma_val=0.0,
    method_mne="relative"
):
    """
    Custom t-test statistics function compatible with MNE permutation tests.

    This function is designed to be used as the `stat_fun` argument in MNE's
    permutation tests. It tests whether the mean of observations (which should
    already be centered around the null hypothesis value, e.g., scores - chance)
    differs from `sigma_val`.

    Args:
        data_array (np.ndarray): Input data array of shape 
                                 (n_observations, n_features...)
                                 MNE expects the first dimension to be 
                                 observations.
        sigma_val (float): Value to test against (population mean under H0).
                           Typically, if `data_array` is `scores - chance_level`,
                           `sigma_val` should be 0.
        method_mne (str): Method for MNE's t-test calculation (default: 
                          'relative'). 'relative' uses a pooled variance 
                          estimate if data_array is 2D+ and represents 
                          different conditions. For 1-sample, it's standard.

    Returns:
        np.ndarray: Calculated t-values, with NaNs replaced by 0 for robust 
                    clustering.
    """

    # Calculate t-values using MNE's ttest_1samp_no_p function
    # This function is optimized for use within permutation frameworks.
    t_values_array = ttest_1samp_no_p(
        data_array, sigma=sigma_val, method=method_mne
    )

    # Replace NaNs with 0 for robust clustering.
    # MNE's clustering can be sensitive to NaN values in the t-statistic map.
    t_values_array[np.isnan(t_values_array)] = 0.0
    return t_values_array


def _custom_stat_function_wilcoxon_1samp(
    data_array,
    sigma_val=0.0,
    method_mne="relative"
):
    """
    Custom Wilcoxon signed-rank test statistics function compatible with MNE permutation tests.

    This function applies Wilcoxon signed-rank test to each feature (time point, voxel, etc.)
    for use in cluster-based permutation testing.

    Args:
        data_array (np.ndarray): Input data array of shape 
                                 (n_observations, n_features...)
                                 MNE expects the first dimension to be observations.
        sigma_val (float): Value to test against (typically 0 for centered data).
        method_mne (str): Unused for Wilcoxon, kept for compatibility.

    Returns:
        np.ndarray: Calculated W-statistics (converted to pseudo t-values), 
                    with NaNs replaced by 0 for robust clustering.
    """
    
    if data_array.ndim == 1:
        # Single feature case
        try:
            if np.all(np.isclose(data_array - sigma_val, 0)):
                return np.array([0.0])
            else:
                w_stat, _ = scipy.stats.wilcoxon(
                    data_array - sigma_val, 
                    zero_method="wilcox",
                    alternative="two-sided"
                )
                # Convert W statistic to pseudo t-value for clustering
                # Use sign of median to determine direction
                median_diff = np.median(data_array - sigma_val)
                pseudo_t = np.sign(median_diff) * np.sqrt(w_stat)
                return np.array([pseudo_t])
        except (ValueError, AttributeError):
            return np.array([0.0])
    
    else:
        # Multiple features case
        n_features = data_array.shape[1] if data_array.ndim == 2 else np.prod(data_array.shape[1:])
        data_reshaped = data_array.reshape(data_array.shape[0], -1)
        
        w_values = np.zeros(n_features)
        
        for i in range(n_features):
            feature_data = data_reshaped[:, i] - sigma_val
            
            try:
                if np.all(np.isclose(feature_data, 0)) or np.any(np.isnan(feature_data)):
                    w_values[i] = 0.0
                else:
                    w_stat, _ = scipy.stats.wilcoxon(
                        feature_data, 
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                    # Convert W statistic to pseudo t-value for clustering
                    median_diff = np.median(feature_data)
                    w_values[i] = np.sign(median_diff) * np.sqrt(w_stat)
            except (ValueError, AttributeError):
                w_values[i] = 0.0
        
        # Reshape back to original feature dimensions if needed
        if data_array.ndim > 2:
            w_values = w_values.reshape(data_array.shape[1:])
    
    # Replace NaNs with 0 for robust clustering
    w_values[np.isnan(w_values)] = 0.0
    return w_values


def perform_cluster_permutation_test(
    input_data_array,
    chance_level=0.5,
    n_permutations=1024,
    cluster_threshold_config=None,  # e.g., TFCE dict or t-stat float
    alternative_hypothesis="greater",  # 'greater', 'less', or 'two-sided'
    n_jobs=1,  # Number of jobs for MNE parallelization
    stat_function_to_use="wilcoxon",  # 'ttest', 'wilcoxon', or 'auto' for MNE default
    random_seed=None,  # Seed for reproducibility
    connectivity_matrix=None,  # Connectivity for spatial clustering
    tfr_like_input=False  # Hint if input is (n_obs, n_freqs, n_times)
):
    """
    Performs a cluster-based permutation test on observed scores vs. a chance level.

    This function wraps `mne.stats.permutation_cluster_1samp_test` to test
    whether decoding scores (e.g., AUC) are significantly different from a
    specified chance level, with multiple comparisons correction via clustering.

    Args:
        input_data_array (np.ndarray): Input data with observations (e.g., subjects, CV folds)
                                      as the first dimension. Shape: (n_obs, n_features...).
        chance_level (float): Value to test against (null hypothesis mean).
        n_permutations (int): Number of permutations to build the null distribution.
        cluster_threshold_config: Threshold configuration for clustering.
                                 - If TFCE: dict like {'start': 0.1, 'step': 0.1}.
                                 - If stat-based: float representing the t-statistic threshold.
        alternative_hypothesis (str): Specifies the alternative hypothesis.
                                      Options: 'greater', 'less', 'two-sided'.
        n_jobs (int): Number of parallel jobs for MNE's computation.
        stat_function_to_use: Statistical function to use. 'auto' uses MNE's default.
                              Can be a callable or specific string like '_custom_stat_function_ttest_1samp'.
        random_seed (int): Random seed for ensuring reproducible results.
        connectivity_matrix: Connectivity matrix for spatial clustering (e.g., sensor adjacency).
        tfr_like_input (bool): If True, indicates input is Time-Frequency Representation-like,
                               affecting default connectivity assumptions if not provided.

    Returns:
        tuple: (observed_t_values, cluster_definitions_masks, cluster_p_values, H0_distribution)
            - observed_t_values (np.ndarray): T-values for the original data.
            - cluster_definitions_masks (list): List of boolean masks defining each cluster.
            - cluster_p_values (np.ndarray): Array of p-values, one for each cluster.
            - H0_distribution (np.ndarray): Array of max cluster statistics from permutations.
    """

    # === INPUT DATA VALIDATION ===
    if not isinstance(input_data_array, np.ndarray) or input_data_array.ndim < 1:
        logger_stats_decoding.error(
            f"Input data for cluster permutation test must be a NumPy array "
            f"with at least 1 dimension. Received type {type(input_data_array)} "
            f"with shape {getattr(input_data_array, 'shape', 'N/A')}."
        )
        return np.array([]), [], np.array([]), np.array([])

    # Handle 1D input: reshape to 2D for MNE compatibility
    if input_data_array.ndim == 1:
        logger_stats_decoding.debug(
            "Reshaping 1D input data (n_obs,) to 2D (n_obs, 1) for MNE cluster test."
        )
        input_data_array_for_test = input_data_array[:, np.newaxis]
    else:
        input_data_array_for_test = input_data_array

    # === VALIDITY CHECKS ===
    # Need at least 2 observations (e.g., subjects or folds)
    if input_data_array_for_test.shape[0] < 2:
        logger_stats_decoding.error(
            f"Not enough observations ({input_data_array_for_test.shape[0]}) "
            f"for cluster permutation test. Requires at least 2."
        )
        return np.array([]), [], np.array([]), np.array([])

    # Check for features to test
    if input_data_array_for_test.ndim > 1 and input_data_array_for_test.shape[1] == 0:
        logger_stats_decoding.error(
            f"No features to test (second dimension is 0). "
            f"Data shape: {input_data_array_for_test.shape}"
        )
        return np.array([]), [], np.array([]), np.array([])

    # === DATA PREPARATION ===
    # Center data around the null hypothesis: test (scores - chance) against 0
    data_array_vs_chance = input_data_array_for_test - chance_level

    # === MNE PARAMETER CONFIGURATION ===
    # Convert alternative hypothesis to MNE's tail parameter
    tail_param_mne = 0  # Default for 'two-sided'
    if alternative_hypothesis.lower() == "greater":
        tail_param_mne = 1
    elif alternative_hypothesis.lower() == "less":
        tail_param_mne = -1

    
    # not to over parralelise 
    if n_jobs == 1 or n_jobs == 0 or (os.cpu_count() is not None and n_jobs > os.cpu_count()):
        effective_n_jobs = min(
            4, max(1, os.cpu_count() // 2 if os.cpu_count() is not None else 2)
        )
        if n_jobs != effective_n_jobs:  
            logger_stats_decoding.info(
                f"Adjusting MNE parallelism from requested n_jobs={n_jobs} "
                f"to effective_n_jobs={effective_n_jobs} for resource management."
            )
        n_jobs = effective_n_jobs

    # === PARAMETER LOGGING ===
    logger_stats_decoding.info(
        f"Executing cluster-based permutation test. "
        f"Data shape (obs, features...): {data_array_vs_chance.shape}"
    )
    logger_stats_decoding.info(
        f"  Parameters: n_permutations={n_permutations}, "
        f"threshold_config={cluster_threshold_config}, "
        f"alternative='{alternative_hypothesis}' (MNE tail={tail_param_mne}), "
        f"stat_func='{str(stat_function_to_use)}', n_jobs={n_jobs}"
    )

    # === STATISTICAL FUNCTION CONFIGURATION ===
    effective_stat_function = None
    if stat_function_to_use == "ttest" or stat_function_to_use == "_custom_stat_function_ttest_1samp":
        effective_stat_function = _custom_stat_function_ttest_1samp
        logger_stats_decoding.info("  Using t-test based statistical function.")
    elif stat_function_to_use == "wilcoxon" or stat_function_to_use == "_custom_stat_function_wilcoxon_1samp":
        effective_stat_function = _custom_stat_function_wilcoxon_1samp
        logger_stats_decoding.info("  Using Wilcoxon based statistical function.")
    elif callable(stat_function_to_use) and stat_function_to_use != "auto":
        effective_stat_function = stat_function_to_use
        logger_stats_decoding.info(f"  Using provided custom stat_fun: {stat_function_to_use.__name__}")
    # If 'auto' or None, MNE's default will be used.

    # === PREPARE ARGUMENTS FOR MNE FUNCTION ===
    test_args_mne = {
        "n_permutations": n_permutations,
        "threshold": cluster_threshold_config,  # TFCE dict or t-stat float
        "tail": tail_param_mne,
        "n_jobs": n_jobs,
        "seed": random_seed,  # For reproducibility
        "out_type": "mask",  # Return boolean masks for clusters
    }

    if effective_stat_function is not None:
        test_args_mne["stat_fun"] = effective_stat_function
        logger_stats_decoding.info(
            f"  Using custom stat_fun: {effective_stat_function.__name__}")
    else:  # MNE default ttest_1samp_no_p will be used if 'stat_fun' not in test_args_mne
        logger_stats_decoding.info(
            "  Using MNE's default stat_fun (ttest_1samp_no_p).")

    # === CONNECTIVITY HANDLING ===
    if connectivity_matrix is not None:
        test_args_mne["connectivity"] = connectivity_matrix
        logger_stats_decoding.info(
            f"  Using provided connectivity matrix with shape "
            f"{connectivity_matrix.shape if hasattr(connectivity_matrix, 'shape') else 'N/A'}."
        )
    elif data_array_vs_chance.ndim > 2 and not tfr_like_input:
        # For >2D data (e.g., TGM: obs, times_train, times_test), MNE infers grid connectivity
        logger_stats_decoding.info(
            f"  Input data is >2D. MNE will attempt to infer connectivity (e.g., grid for TGM)."
        )
    elif data_array_vs_chance.ndim == 2 and data_array_vs_chance.shape[1] > 1:
        # For 2D data (obs, features, e.g., time series), MNE infers temporal connectivity
        logger_stats_decoding.info(
            f"  Input data is 2D (features). MNE will use default temporal "
            f"connectivity inference if applicable."
        )

    # === EXECUTE PERMUTATION TEST ===
    try:
        observed_t_values, cluster_definitions_masks, cluster_p_values, H0_distribution = \
            permutation_cluster_1samp_test(
                data_array_vs_chance, **test_args_mne)

#observed_t_values =  t test for my reel data on each temporal point (it means if I have 10 CV fold),
#it measures that for 10 AUC score at one temporal point ==> (mean_scores - chance_level) / SD 

#cluster_definitions_masks = list of boolean masks for each cluster found

#cluster_p_values = p-values for each cluster, indicating significance if <0.05 (sign). 
#p value is calculated  (number of permutations with cluster ≥ cluster_observed) / number_total_permutations

#H0_distribution = distribution of max cluster statistics under the null hypothesis
    except Exception as e_perm_test:
        logger_stats_decoding.error(
            f"Error during mne.stats.permutation_cluster_1samp_test: {e_perm_test}",
            exc_info=True,
        )
        return np.array([]), [], np.array([]), np.array([])

  # === POST-PROCESSING RESULTS ===
    # Squeeze t-values if original input was 1D bcs MNE transform in (n_obs, 1) 
    if input_data_array.ndim == 1 and observed_t_values.ndim >= 1:
        observed_t_values = observed_t_values.squeeze()

    # Convert slice objects in cluster_definitions_masks to boolean arrays
    # This is important because MNE can return slices for 1D contiguous clusters
    processed_cluster_masks = []
    if cluster_definitions_masks:
        # Determine the shape of the features for creating boolean masks
        # If input_data_array_for_test is (n_obs, n_features), feature_shape is (n_features,)
        # If input_data_array_for_test is (n_obs, n_dim1, n_dim2), feature_shape is (n_dim1, n_dim2)
        feature_shape = data_array_vs_chance.shape[1:]

        for clu_item in cluster_definitions_masks:
            if isinstance(clu_item, slice):
                # This typically happens for 1D data where feature_shape will be (n_features,)
                if len(feature_shape) == 1:  # Ensure it's indeed for 1D features
                    mask = np.zeros(feature_shape, dtype=bool)
                    mask[clu_item] = True
                    processed_cluster_masks.append(mask)
                else:
                    logger_stats_decoding.warning(
                        f"Received a slice object for a cluster in multi-dimensional data (shape {feature_shape}). "
                        "This is unexpected. Skipping this cluster mask processing."
                    )
                    processed_cluster_masks.append(
                        None)  # Or handle differently
            elif isinstance(clu_item, np.ndarray) and clu_item.dtype == bool:
                processed_cluster_masks.append(clu_item)
            elif isinstance(clu_item, tuple) and all(isinstance(s, slice) for s in clu_item):
                # For multi-dimensional slices (e.g., TGM)
                mask = np.zeros(feature_shape, dtype=bool)
                mask[clu_item] = True
                processed_cluster_masks.append(mask)
            else:
                logger_stats_decoding.warning(
                    f"Unexpected cluster definition type or content: {type(clu_item)}. Skipping."
                )
                processed_cluster_masks.append(None)  # Or handle differently

    # Replace original cluster_definitions_masks with the processed ones
    # This ensures subsequent code (like logging or create_p_value_map) gets boolean arrays
    cluster_definitions_masks_for_output = [
        m for m in processed_cluster_masks if m is not None]

    # === RESULT LOGGING ===
    logger_stats_decoding.info(
        f"Cluster permutation test completed. "
 
        f"{len(cluster_definitions_masks_for_output)} initial clusters found (after processing slices)."
    )

    if cluster_definitions_masks_for_output and cluster_p_values is not None:  
        num_significant_clusters_logged = 0
        # Make sure to iterate up to the minimum length of processed masks and p_values
        num_clusters_to_log = min(
            len(cluster_definitions_masks_for_output), len(cluster_p_values))

        for i_cluster_log in range(num_clusters_to_log):  
            p_val_cluster_log = cluster_p_values[i_cluster_log]
            if p_val_cluster_log < 0.05:  # Standard alpha for significance
                num_significant_clusters_logged += 1
       
                current_mask_details_log = cluster_definitions_masks_for_output[i_cluster_log]

                # This check is now safer as current_mask_details_log should be an ndarray
                log_mask_info = (
                    f"Mask shape: {current_mask_details_log.shape}, "
                    f"Num True points: {np.sum(current_mask_details_log)}"
                )
                logger_stats_decoding.info(
                    f"  Significant Cluster #{i_cluster_log + 1}: "
                    f"p-value = {p_val_cluster_log:.4f}. {log_mask_info}"
                )

        if num_significant_clusters_logged == 0 and num_clusters_to_log > 0:  
            logger_stats_decoding.info(
                "No clusters found to be statistically significant (p < 0.05)."
            )
    elif not cluster_definitions_masks_for_output: 
        logger_stats_decoding.info(
            "No clusters identified (e.g., no data points passed the initial threshold or masks were invalid)."
        )

    # Return the processed masks (boolean arrays)
    return observed_t_values, cluster_definitions_masks_for_output, cluster_p_values, H0_distribution


def perform_pointwise_fdr_correction_on_scores(
    input_data_array,
    chance_level=0.5,
    alpha_significance_level=0.05,
    fdr_correction_method="indep",  # Benjamini-Hochberg
    alternative_hypothesis="greater",
    statistical_test_type="wilcoxon",  # 'ttest' or 'wilcoxon'
):
    """
    Performs a pointwise statistical test for each feature against a chance 
    level, followed by FDR correction on the resulting p-values.

    This function is useful for identifying time points or spatial locations 
    where decoding scores are significantly different from chance, while 
    controlling the false discovery rate.

    Args:
        input_data_array (np.ndarray): Input data with observations as the 
                                      1st dimension. Shape: (n_observations, 
                                      n_features) or (n_obs,) for a single 
                                      feature.
        chance_level (float): Value to test against (chance level).
        alpha_significance_level (float): Alpha level for FDR (default: 0.05).
        fdr_correction_method (str): FDR method - 'indep' (Benjamini-Hochberg) 
                                    or 'negcorr' (Benjamini-Yekutieli for 
                                    negative correlations).
        alternative_hypothesis (str): 'two-sided', 'greater', or 'less' for 
                                     statistical tests.
        statistical_test_type (str): 'ttest' (parametric) or 'wilcoxon' (non-parametric).

    Returns:
        tuple: (observed_test_stats, fdr_significant_mask, fdr_corrected_p_values)
            - observed_test_stats (np.ndarray): Test statistics, shaped like the 
                                               original features.
            - fdr_significant_mask (np.ndarray): Boolean mask of significance 
                                                after FDR.
            - fdr_corrected_p_values (np.ndarray): FDR-corrected p-values.
    """

    # === INPUT DATA VALIDATION ===
    if not isinstance(input_data_array, np.ndarray) or input_data_array.ndim == 0:
        logger_stats_decoding.error(
            f"Input data for FDR correction must be a NumPy array "
            f"with at least 1 dimension. Received type {type(input_data_array)} "
            f"with shape {getattr(input_data_array, 'shape', 'N/A')}."
        )
        return np.array([]), np.array([], dtype=bool), np.array([])

    original_ndim_fdr = input_data_array.ndim
    data_for_ttest_fdr = input_data_array

    # === PREPARATION FOR T-TEST ===
    # scipy.stats.ttest_1samp expects (n_obs,) for a single feature test,
    # or (n_obs, n_features) for multiple independent tests along axis 0.
    if original_ndim_fdr > 2:
        n_obs_fdr = input_data_array.shape[0]
        # Product of all dimensions except the first (observations)
        n_total_features_fdr = np.prod(input_data_array.shape[1:])
        data_for_ttest_fdr = input_data_array.reshape(
            n_obs_fdr, n_total_features_fdr)
        logger_stats_decoding.debug(
            f"Reshaping {original_ndim_fdr}D data to 2D ({data_for_ttest_fdr.shape}) "
            f"for FDR t-tests."
        )

    # Check for minimum number of observations
    if data_for_ttest_fdr.shape[0] < 2:
        logger_stats_decoding.error(
            f"Not enough observations ({data_for_ttest_fdr.shape[0]}) "
            f"for FDR t-test. Requires at least 2."
        )
        # Return outputs matching the original feature shape
        original_feature_shape = input_data_array.shape[1:] if original_ndim_fdr > 1 else (
            1,)
        return (
            np.full(original_feature_shape, np.nan),
            np.full(original_feature_shape, False, dtype=bool),
            np.full(original_feature_shape, np.nan),
        )

    # === PARAMETER LOGGING ===
    logger_stats_decoding.info(
        f"Performing pointwise {statistical_test_type} with FDR correction. "
        f"Data shape for test (obs, features if any): {data_for_ttest_fdr.shape}"
    )
    logger_stats_decoding.info(
        f"  Parameters: chance={chance_level}, alpha={alpha_significance_level}, "
        f"test_type='{statistical_test_type}', FDR_method='{fdr_correction_method}', "
        f"alternative='{alternative_hypothesis}'"
    )    # === EXECUTE STATISTICAL TESTS ===
    if statistical_test_type.lower() == "ttest":
        # T-test = (x̄ - μ₀) / (s / √n)  
        # x̄ = moyenne de l'échantillon (moyenne des scores AUC observés)
        # μ₀ = moyenne de population sous H₀ (chance_level = 0.5)
        # s = écart-type de l'échantillon
        # n = taille de l'échantillon (nombre de folds CV)
        
        observed_test_stats_flat, uncorrected_p_values_flat = scipy.stats.ttest_1samp(
            data_for_ttest_fdr,
            popmean=chance_level,
            axis=0,  # Perform test along the observation axis
            nan_policy="propagate",  # If a feature column has NaNs, its result will be NaN
            alternative=alternative_hypothesis,
        )
        
    elif statistical_test_type.lower() == "wilcoxon":
        # Wilcoxon signed-rank test
        observed_test_stats_flat = []
        uncorrected_p_values_flat = []
        
        # Apply Wilcoxon to each feature (column)
        if data_for_ttest_fdr.ndim == 1:
            # Single feature
            differences = data_for_ttest_fdr - chance_level
            try:
                if np.all(np.isclose(differences, 0)):
                    stat, p_val = 0.0, 1.0
                else:
                    stat, p_val = scipy.stats.wilcoxon(
                        differences, 
                        zero_method="wilcox",
                        alternative=alternative_hypothesis
                    )
                observed_test_stats_flat.append(stat)
                uncorrected_p_values_flat.append(p_val)
            except ValueError:
                observed_test_stats_flat.append(np.nan)
                uncorrected_p_values_flat.append(np.nan)
        else:
            # Multiple features
            for feature_idx in range(data_for_ttest_fdr.shape[1]):
                feature_data = data_for_ttest_fdr[:, feature_idx]
                differences = feature_data - chance_level
                
                # Check for NaNs
                if np.any(np.isnan(differences)):
                    observed_test_stats_flat.append(np.nan)
                    uncorrected_p_values_flat.append(np.nan)
                    continue
                
                try:
                    if np.all(np.isclose(differences, 0)):
                        stat, p_val = 0.0, 1.0
                    else:
                        stat, p_val = scipy.stats.wilcoxon(
                            differences, 
                            zero_method="wilcox",
                            alternative=alternative_hypothesis
                        )
                    observed_test_stats_flat.append(stat)
                    uncorrected_p_values_flat.append(p_val)
                except ValueError:
                    # Peut arriver si tous les échantillons sont identiques
                    observed_test_stats_flat.append(np.nan)
                    uncorrected_p_values_flat.append(np.nan)
        
        observed_test_stats_flat = np.array(observed_test_stats_flat)
        uncorrected_p_values_flat = np.array(uncorrected_p_values_flat)
        
    else:
        raise ValueError(f"Unsupported test type: {statistical_test_type}. Use 'ttest' or 'wilcoxon'.")

    # Ensure p-values and test-statistics are arrays, even if only one feature was tested
    if np.isscalar(uncorrected_p_values_flat):
        uncorrected_p_values_flat = np.array([uncorrected_p_values_flat])
    if np.isscalar(observed_test_stats_flat):
        observed_test_stats_flat = np.array([observed_test_stats_flat])

    # === FDR CORRECTION ===
    # Perform FDR correction only on valid (non-NaN) p-values
    valid_p_value_indices_fdr = ~np.isnan(uncorrected_p_values_flat)
    p_values_for_fdr_input_fdr = uncorrected_p_values_flat[valid_p_value_indices_fdr]

    # Initialize with defaults for cases with no valid p-values
    fdr_significant_mask_subset_fdr = np.array([False], dtype=bool)
    fdr_corrected_p_values_subset_fdr = np.array([1.0])

    # Apply FDR if there are valid p-values
    if p_values_for_fdr_input_fdr.size > 0:
        fdr_significant_mask_subset_fdr, fdr_corrected_p_values_subset_fdr = fdr_correction(
            p_values_for_fdr_input_fdr,
            alpha=alpha_significance_level,
            method=fdr_correction_method,
        )
    else:
        logger_stats_decoding.warning(
            "No valid (non-NaN) p-values found to apply FDR correction."
        )

    # === RECONSTRUCT FULL ARRAYS ===
    # Initialize full output arrays (flat, corresponding to t-test output)
    fdr_significant_mask_flat = np.full_like(
        uncorrected_p_values_flat, False, dtype=bool)
    fdr_corrected_p_values_flat = np.full_like(
        uncorrected_p_values_flat, np.nan, dtype=float)

    # Populate results into flat arrays at valid indices
    if p_values_for_fdr_input_fdr.size > 0:
        fdr_significant_mask_flat[valid_p_value_indices_fdr] = fdr_significant_mask_subset_fdr
        fdr_corrected_p_values_flat[valid_p_value_indices_fdr] = fdr_corrected_p_values_subset_fdr

    # === RESHAPE TO ORIGINAL DIMENSIONS ===
    # Reshape outputs to original feature dimensions if necessary
    original_feature_shape_out = input_data_array.shape[1:] if original_ndim_fdr > 1 else (
        1,)

    if original_ndim_fdr == 1 and observed_test_stats_flat.size == 1:
        # Single feature input
        observed_test_stats_out = observed_test_stats_flat
        fdr_significant_mask_out = fdr_significant_mask_flat
        fdr_corrected_p_values_out = fdr_corrected_p_values_flat
    else:
        # Multi-feature input, reshape outputs
        observed_test_stats_out = observed_test_stats_flat.reshape(
            original_feature_shape_out)
        fdr_significant_mask_out = fdr_significant_mask_flat.reshape(
            original_feature_shape_out)
        fdr_corrected_p_values_out = fdr_corrected_p_values_flat.reshape(
            original_feature_shape_out)

    # === RESULT LOGGING ===
    num_sig_points = np.sum(fdr_significant_mask_out)
    logger_stats_decoding.info(
        f"FDR correction completed. {num_sig_points} point(s) found significant "
        f"(p_fdr < {alpha_significance_level})."
    )

    return observed_test_stats_out, fdr_significant_mask_out, fdr_corrected_p_values_out


def compare_global_scores_to_chance(
    global_scores_array,
    chance_level=0.5,
    statistical_test_type="wilcoxon",  # 'ttest' or 'wilcoxon'
    alternative_hypothesis="greater",
):
    """
    Tests if a collection of global scores is significantly different from a specified chance level.

    This function is used to test the significance of mean scores (e.g., mean AUC
    per subject or CV fold) against the chance level.

    Args:
        global_scores_array (np.ndarray): 1D array of scores (e.g., mean AUC per subject/fold).
        chance_level (float): Chance level to test against.
        statistical_test_type (str): Type of test - 'ttest' (parametric) or
                                   'wilcoxon' (non-parametric).
        alternative_hypothesis (str): 'two-sided', 'greater', or 'less'.

    Returns:
        tuple: (test_statistic, p_value) - Test statistic and p-value.
    """

    # === DATA CLEANING AND VALIDATION ===
    scores_cleaned_global = np.array(
        global_scores_array).flatten()  # Ensure 1D array
    scores_valid_for_test_global = scores_cleaned_global[~np.isnan(
        scores_cleaned_global)]  # Remove NaNs

    if len(scores_valid_for_test_global) == 0:
        logger_stats_decoding.warning(
            "No valid (non-NaN) scores provided for compare_global_scores_to_chance. "
            "Returning (NaN, NaN)."
        )
        return np.nan, np.nan

    # === PARAMETER LOGGING AND DESCRIPTIVE STATISTICS ===
    logger_stats_decoding.info(
        f"Testing global scores (N={len(scores_valid_for_test_global)} valid scores) "
        f"using {statistical_test_type} against chance={chance_level}, "
        f"alternative='{alternative_hypothesis}'."
    )

    mean_score_glob = np.mean(scores_valid_for_test_global)
    std_score_glob = np.std(scores_valid_for_test_global)
    logger_stats_decoding.info(
        f"  Summary of valid scores: Mean = {mean_score_glob:.4f}, "
        f"Std = {std_score_glob:.4f}, "
        f"Min = {np.min(scores_valid_for_test_global):.4f}, "
        f"Max = {np.max(scores_valid_for_test_global):.4f}"
    )

    # Initialize results
    test_statistic_glob, p_value_glob = np.nan, np.nan

    # === PARAMETRIC T-TEST ===
    if statistical_test_type.lower() == "ttest":
        if len(scores_valid_for_test_global) >= 2:  # t-test requires at least 2 samples
            test_statistic_glob, p_value_glob = scipy.stats.ttest_1samp(
                scores_valid_for_test_global,
                popmean=chance_level,
                nan_policy="omit",  # Should be handled by prior cleaning
                alternative=alternative_hypothesis,
            )
        else:
            logger_stats_decoding.warning(
                "Not enough valid scores (requires >= 2) for a t-test. Returning (NaN, NaN)."
            )

    # === NON-PARAMETRIC WILCOXON SIGNED-RANK TEST ===
    elif statistical_test_type.lower() == "wilcoxon":
        # Test on differences from chance
        differences_from_chance_glob = scores_valid_for_test_global - chance_level

        if np.all(np.isclose(differences_from_chance_glob, 0)):
            # If all scores are exactly at chance level
            logger_stats_decoding.info(
                "All scores are at chance level. Wilcoxon test yields p=1.0."
            )
            test_statistic_glob, p_value_glob = 0.0, 1.0
        elif len(differences_from_chance_glob) > 0:
            # If there are differences
            if len(differences_from_chance_glob) < 5:  # Warning for small N
                logger_stats_decoding.warning(
                    f"Very few data points ({len(differences_from_chance_glob)}) "
                    f"for Wilcoxon test. Results may be less reliable."
                )
            try:
                # zero_method options: 'wilcox' drops zero-differences,
                # 'pratt' includes them, 'zsplit' splits them
                test_statistic_glob, p_value_glob = scipy.stats.wilcoxon(
                    differences_from_chance_glob,
                    zero_method="wilcox",
                    alternative=alternative_hypothesis,
                )
            except ValueError as e_wilcox_glob:
                # e.g., if all differences are zero after zero_method
                logger_stats_decoding.warning(
                    f"ValueError during Wilcoxon test (e.g., all differences "
                    f"might be zero): {e_wilcox_glob}. "
                    f"Assigning p-value=1.0 if all differences are effectively zero, "
                    f"otherwise NaN."
                )
                # Re-check after potential effect of zero_method
                if np.all(np.isclose(differences_from_chance_glob, 0)):
                    test_statistic_glob, p_value_glob = 0.0, 1.0
                else:
                    test_statistic_glob, p_value_glob = np.nan, np.nan  # Undetermined
        else:
            logger_stats_decoding.warning(
                "Not enough valid scores for Wilcoxon test. Returning (NaN, NaN)."
            )
    else:
        logger_stats_decoding.error(
            f"Unknown statistical test type: '{statistical_test_type}'. "
            f"Choose 'ttest' or 'wilcoxon'."
        )
        raise ValueError(
            f"Unsupported statistical test type: {statistical_test_type}")

    # === RESULT LOGGING ===
    logger_stats_decoding.info(
        f"Global scores test result: Statistic = {test_statistic_glob:.3f}, "
        f"P-value = {p_value_glob:.4f}"
    )

    return test_statistic_glob, p_value_glob


def create_p_value_map_from_cluster_results(
    data_shape_to_map_onto,  # e.g., (n_times,) or (n_freqs, n_times)
    cluster_definitions_list_masks,  # List of boolean masks from MNE
    p_values_per_cluster,  # Corresponding p-values for each cluster mask
    default_p_value_for_map=1.0,  # Value for points not in any cluster
):
    """
    Creates a map (array of the same shape as original data features) where
    each point is assigned the p-value of the *most significant* cluster
    (smallest p-value) it belongs to.

    This function is useful for visualizing clustering results by assigning
    p-values to spatial/temporal points for plotting.

    Args:
        data_shape_to_map_onto (tuple or int): Shape of the data features
                                              (e.g., (n_times,) for 1D,
                                              (n_freqs, n_times) for 2D).
        cluster_definitions_list_masks (list): List of boolean masks from MNE's
                                               `permutation_cluster_1samp_test`.
        p_values_per_cluster (np.ndarray): Array of p-values corresponding to each cluster mask.
        default_p_value_for_map (float): Value for points not belonging to any cluster.

    Returns:
        np.ndarray: P-value map of the same shape as `data_shape_to_map_onto`.
    """

    # === SHAPE NORMALIZATION ===
    # Ensure data_shape_to_map_onto is a tuple
    if not isinstance(data_shape_to_map_onto, tuple):
        data_shape_to_map_onto = (data_shape_to_map_onto,) if np.isscalar(data_shape_to_map_onto) \
            else tuple(data_shape_to_map_onto)

    # === MAP INITIALIZATION ===
    # Initialize the p-value map with the default value
    p_value_map_output = np.full(
        data_shape_to_map_onto, default_p_value_for_map, dtype=float
    )

    # === INPUT VALIDATION ===
    if not cluster_definitions_list_masks or p_values_per_cluster is None or \
       len(cluster_definitions_list_masks) != len(p_values_per_cluster):
        logger_stats_decoding.warning(
            "Mismatch in cluster definitions and p-values, or empty inputs "
            "for p-value map creation. Returning default map "
            f"(all points = {default_p_value_for_map})."
        )
        return p_value_map_output

    # === PROCESS CLUSTERS BY SIGNIFICANCE ORDER ===
    # Sort cluster indices by p-value in ascending order (most significant first)
    sorted_indices_pmap = np.argsort(p_values_per_cluster)

    for i_sorted_pmap in sorted_indices_pmap:
        current_cluster_boolean_mask_pmap = cluster_definitions_list_masks[i_sorted_pmap]
        current_cluster_p_value_pmap = p_values_per_cluster[i_sorted_pmap]

        # === MASK VALIDATION ===
        # Validate mask shape and type
        if not (isinstance(current_cluster_boolean_mask_pmap, np.ndarray) and
                current_cluster_boolean_mask_pmap.dtype == bool and
                current_cluster_boolean_mask_pmap.shape == data_shape_to_map_onto):
            logger_stats_decoding.warning(
                f"Cluster definition (mask) at index {i_sorted_pmap} is not "
                f"a valid boolean mask matching data_shape {data_shape_to_map_onto}. "
                f"Mask shape: {getattr(current_cluster_boolean_mask_pmap, 'shape', 'N/A')}. "
                f"Skipping this cluster for the p-value map."
            )
            continue

        # === P-VALUE ASSIGNMENT ===
        # Assign the cluster's p-value to the corresponding points.
        # More significant clusters (processed first) will overwrite less significant ones
        # if a point belongs to multiple clusters (though this shouldn't happen with MNE's output).
        p_value_map_output[current_cluster_boolean_mask_pmap] = current_cluster_p_value_pmap

    # === RESULT LOGGING ===
    # Count points with p < 0.05
    num_sig_points_in_map = np.sum(p_value_map_output < 0.05)
    logger_stats_decoding.info(
        f"P-value map created. {num_sig_points_in_map} points have p < 0.05 "
        f"originating from clusters."
    )

    return p_value_map_output
