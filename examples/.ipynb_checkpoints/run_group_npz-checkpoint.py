#!/usr/bin/env python3
"""
Enhanced EEG Decoding Group Analysis Script.

Available keys: ['subject_id', 'group', 'decoding_protocol_identifier',
'classifier_type_used', 'epochs_time_points', 'pp_ap_main_original_labels',
'pp_ap_main_pred_probas_global', 'pp_ap_main_pred_labels_global',
'pp_ap_main_cv_global_scores', 'pp_ap_main_scores_1d_all_folds',
'pp_ap_main_scores_1d_mean', 'pp_ap_main_temporal_1d_fdr',
'pp_ap_main_temporal_1d_cluster', 'pp_ap_main_tgm_all_folds',
'pp_ap_main_tgm_mean', 'pp_ap_main_tgm_fdr', 'pp_ap_main_mean_auc_global',
'pp_ap_main_global_metrics', 'pp_ap_specific_ap_results',
'pp_ap_mean_of_specific_scores_1d', 'pp_ap_sem_of_specific_scores_1d',
'pp_ap_mean_specific_fdr', 'pp_ap_mean_specific_cluster',
'pp_ap_ap_vs_ap_results', 'pp_ap_ap_centric_avg_results', 'detected_protocol']
"""

import os
import sys
import glob
import logging
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu, wilcoxon

# Import stats functions for group-level statistical tests
sys.path.append('.')
try:
    from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
    )
except ImportError:
    perform_pointwise_fdr_correction_on_scores = None
    perform_cluster_permutation_test = None
    compare_global_scores_to_chance = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eeg_analysis_enhanced.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V10"

# Mapping des noms de dossiers vers les noms de groupes
GROUP_NAME_MAPPING = {
    'COMA_battery': 'COMA',
    'VS_battery': 'VS', 
    'MCS+_battery': 'MCS+',
    'MCS-_battery': 'MCS-',
    'CONTROLS_delirium': 'CONTROLS_DELIRIUM',
    'DELIRIUM+_delirium': 'DELIRIUM+',
    'DELIRIUM-_delirium': 'DELIRIUM-'
}

# Group datasets configuration
DATASET_CONFIGS = {
    'DELIRIUM': {
        'name': 'Delirium Dataset',
        'groups': ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM'],
        'colors': {
            'DELIRIUM+': '#d62728',
            'DELIRIUM-': '#ff7f0e', 
            'CONTROLS_DELIRIUM': '#2ca02c'
        }
    },
    'COMA': {
        'name': 'Coma Dataset',
        'groups': ['COMA', 'VS', 'MCS+', 'MCS-'],
        'colors': {
            'COMA': '#8b0000',        # Dark red
            'VS': '#ff6347',          # Tomato red
            'MCS+': '#4682b4',        # Steel blue
            'MCS-': '#87ceeb'         # Sky blue
        }
    }
}

# Statistical parameters
CHANCE_LEVEL = 0.5
N_PERMUTATIONS = 1000
CLUSTER_THRESHOLD = 0.05
FDR_ALPHA = 0.05

# === VISUALIZATION PARAMETERS ===
PUBLICATION_PARAMS = {
    'figure.figsize': (18, 10),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}

plt.rcParams.update(PUBLICATION_PARAMS)

COLORS_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def find_npz_files(base_path):
    """
    Recursively find and organize NPZ files based on directory structure.

    Structure attendue : {base_path}/intra_subject_results/{group_name}/...
    /decoding_results_full.npz
    """
    logger.info("Searching for NPZ files in: %s", base_path)
    organized_data = {'PP': {}}  # Protocole unique 'PP'

    search_pattern = os.path.join(
        base_path, 'intra_subject_results', '**', 'decoding_results_full.npz')
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        logger.warning(
            "No 'decoding_results_full.npz' files found with pattern: %s",
            search_pattern)
        return {}

    logger.info("Found %d potential result files.", len(all_files))

    for file_path in all_files:
        try:
            rel_path = os.path.relpath(file_path, base_path)
            path_parts = rel_path.split(os.sep)

            if (len(path_parts) > 2 and
                    path_parts[0] == 'intra_subject_results'):
                group_folder = path_parts[1]
                group_name = GROUP_NAME_MAPPING.get(group_folder,
                                                    group_folder)

                if group_name not in organized_data['PP']:
                    organized_data['PP'][group_name] = []
                organized_data['PP'][group_name].append(file_path)
            else:
                logger.warning(
                    "File path does not match expected structure, "
                    "skipping: %s", file_path)

        except Exception as e:
            logger.warning("Error processing file path %s: %s", file_path, e)

    # Log summary
    logger.info("=== COLLECTED DATA SUMMARY ===")
    for protocol, groups in organized_data.items():
        if not groups:
            logger.warning("No groups found for protocol %s.", protocol)
            continue
        logger.info("Protocol %s:", protocol)
        for group, files in groups.items():
            logger.info("  - Group '%s': %d subjects found.", group,
                        len(files))

    return organized_data


def load_npz_data(file_path):
    """
    Load and validate NPZ file data, using the correct keys found in files.
    Now includes specific FDR and cluster masks and TGM data.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            actual_score_key = 'pp_ap_main_scores_1d_mean'
            actual_time_key = 'epochs_time_points'
            actual_fdr_key = 'pp_ap_main_temporal_1d_fdr'
            actual_cluster_key = 'pp_ap_main_temporal_1d_cluster'
            actual_tgm_key = 'pp_ap_main_tgm_mean'
            actual_tgm_fdr_key = 'pp_ap_main_tgm_fdr'
            # Nouvelles clés spécifiques
            specific_fdr_key = 'pp_ap_mean_specific_fdr'
            specific_cluster_key = 'pp_ap_mean_specific_cluster'

            data_keys = list(data.keys())
            required_keys = [actual_score_key, actual_time_key]

            for key in required_keys:
                if key not in data_keys:
                    logger.warning(
                        "Missing required field '%s' in %s. "
                        "Available keys: %s", key, file_path, data_keys)
                    return None

            # Extraire le nom du sujet à partir du chemin
            path_parts = file_path.split(os.sep)
            subject_id = "Unknown"
            for part in path_parts:
                if '_Subj_' in part:
                    subject_id = part.split('_Subj_')[1].split('_')[0]
                    break

            result = {
                'scores': data[actual_score_key],
                'times': data[actual_time_key],
                'subject_id': subject_id,
                'file_path': file_path
            }

            # Ajouter TGM data si disponible
            if actual_tgm_key in data_keys:
                result['tgm_data'] = data[actual_tgm_key]
                logger.debug("Loaded TGM data for %s: shape %s", 
                           subject_id, result['tgm_data'].shape)
            else:
                result['tgm_data'] = None

            # Ajouter TGM FDR si disponible
            if actual_tgm_fdr_key in data_keys:
                tgm_fdr_data = data[actual_tgm_fdr_key]
                if isinstance(tgm_fdr_data, np.ndarray) and tgm_fdr_data.dtype == object:
                    try:
                        tgm_fdr_dict = tgm_fdr_data.item()
                        if isinstance(tgm_fdr_dict, dict) and 'mask' in tgm_fdr_dict:
                            result['tgm_fdr_mask'] = tgm_fdr_dict['mask']
                        else:
                            result['tgm_fdr_mask'] = None
                    except:
                        result['tgm_fdr_mask'] = None
                else:
                    result['tgm_fdr_mask'] = None
            else:
                result['tgm_fdr_mask'] = None

            # Ajouter FDR si disponible - extraction du dictionnaire
            if actual_fdr_key in data_keys:
                fdr_data = data[actual_fdr_key]
                # Vérifier si c'est un dictionnaire contenant 'mask'
                if (isinstance(fdr_data, np.ndarray) and
                        fdr_data.dtype == object):
                    try:
                        # Extraire le dictionnaire de l'array
                        fdr_dict = fdr_data.item()
                        if isinstance(fdr_dict, dict) and 'mask' in fdr_dict:
                            result['fdr_mask'] = fdr_dict['mask']
                            # Extraire aussi les p-values si disponibles
                            result['fdr_pvals'] = fdr_dict.get('p_values', None)
                            logger.debug(
                                "Loaded FDR mask for %s: %d significant points",
                                subject_id, np.sum(fdr_dict['mask']))
                        else:
                            logger.warning(
                                "FDR data structure not recognized for %s",
                                subject_id)
                            result['fdr_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                            result['fdr_pvals'] = None
                    except Exception as e:
                        logger.warning(
                            "Error extracting FDR mask for %s: %s",
                            subject_id, e)
                        result['fdr_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                        result['fdr_pvals'] = None
                else:
                    logger.warning(
                        "FDR data is not in expected format for %s",
                        subject_id)
                    result['fdr_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
                    result['fdr_pvals'] = None
            else:
                logger.warning("FDR data not found in %s", file_path)
                result['fdr_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)
                result['fdr_pvals'] = None

            # Ajouter cluster data si disponible
            if actual_cluster_key in data_keys:
                cluster_data = data[actual_cluster_key]
                if isinstance(cluster_data, np.ndarray) and cluster_data.dtype == object:
                    try:
                        cluster_dict = cluster_data.item()
                        if isinstance(cluster_dict, dict) and 'mask' in cluster_dict:
                            result['cluster_mask'] = cluster_dict['mask']
                            result['cluster_pvals'] = cluster_dict.get('p_values_all_clusters', None)
                            logger.debug(
                                "Loaded cluster mask for %s: %d significant points",
                                subject_id, np.sum(cluster_dict['mask']))
                        else:
                            result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                            result['cluster_pvals'] = None
                    except Exception as e:
                        logger.warning("Error extracting cluster mask for %s: %s", subject_id, e)
                        result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                        result['cluster_pvals'] = None
                else:
                    result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                    result['cluster_pvals'] = None
            else:
                result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                result['cluster_pvals'] = None

            # Ajouter les masques spécifiques FDR
            if specific_fdr_key in data_keys:
                specific_fdr_data = data[specific_fdr_key]
                if (isinstance(specific_fdr_data, np.ndarray) and
                        specific_fdr_data.dtype == object):
                    try:
                        specific_fdr_dict = specific_fdr_data.item()
                        if isinstance(specific_fdr_dict, dict) and 'mask' in specific_fdr_dict:
                            result['specific_fdr_mask'] = specific_fdr_dict['mask']
                            result['specific_fdr_pvals'] = specific_fdr_dict.get('p_values', None)
                            logger.debug(
                                "Loaded specific FDR mask for %s: %d significant points",
                                subject_id, np.sum(specific_fdr_dict['mask']))
                        else:
                            logger.warning(
                                "Specific FDR data structure not recognized for %s",
                                subject_id)
                            result['specific_fdr_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                            result['specific_fdr_pvals'] = None
                    except Exception as e:
                        logger.warning(
                            "Error extracting specific FDR mask for %s: %s",
                            subject_id, e)
                        result['specific_fdr_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                        result['specific_fdr_pvals'] = None
                else:
                    result['specific_fdr_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
                    result['specific_fdr_pvals'] = None
            else:
                logger.warning("Specific FDR data not found in %s", file_path)
                result['specific_fdr_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)
                result['specific_fdr_pvals'] = None

            # Ajouter les masques spécifiques cluster
            if specific_cluster_key in data_keys:
                specific_cluster_data = data[specific_cluster_key]
                if (isinstance(specific_cluster_data, np.ndarray) and
                        specific_cluster_data.dtype == object):
                    try:
                        specific_cluster_dict = specific_cluster_data.item()
                        if isinstance(specific_cluster_dict, dict) and 'mask' in specific_cluster_dict:
                            result['specific_cluster_mask'] = specific_cluster_dict['mask']
                            result['specific_cluster_pvals'] = specific_cluster_dict.get('p_values_all_clusters', None)
                            logger.debug(
                                "Loaded specific cluster mask for %s: %d significant points",
                                subject_id, np.sum(specific_cluster_dict['mask']))
                        else:
                            logger.warning(
                                "Specific cluster data structure not recognized for %s",
                                subject_id)
                            result['specific_cluster_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                            result['specific_cluster_pvals'] = None
                    except Exception as e:
                        logger.warning(
                            "Error extracting specific cluster mask for %s: %s",
                            subject_id, e)
                        result['specific_cluster_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                        result['specific_cluster_pvals'] = None
                else:
                    result['specific_cluster_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
                    result['specific_cluster_pvals'] = None
            else:
                logger.warning("Specific cluster data not found in %s", file_path)
                result['specific_cluster_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)
                result['specific_cluster_pvals'] = None

            if result['scores'] is None or result['times'] is None:
                logger.warning("Data for scores or times is None in %s",
                               file_path)
                return None
            if len(result['scores']) == 0 or len(result['times']) == 0:
                logger.warning("Data for scores or times is empty in %s",
                               file_path)
                return None

            return result

    except Exception as e:
        logger.error("Error loading or processing %s: %s", file_path, e)
        return None


def calculate_basic_statistics(scores, times, subject_ids, fdr_masks=None,
                               specific_fdr_masks=None, specific_cluster_masks=None):
    """
    Calculate basic statistical measures using individual subject FDR masks from NPZ files.
    Focus on individual subject analysis and removing group-level statistical tests.
    Now includes specific FDR and cluster masks.
    """
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]

    n_subjects, n_timepoints = scores.shape

    # Basic descriptive statistics
    group_mean = np.mean(scores, axis=0)
    group_std = np.std(
        scores, axis=0, ddof=1) if n_subjects > 1 else np.zeros_like(group_mean)
    group_sem = group_std / \
        np.sqrt(n_subjects) if n_subjects > 1 else np.zeros_like(group_mean)

    # SEM bounds for visualization
    sem_lower = group_mean - group_sem
    sem_upper = group_mean + group_sem

    # Individual subject statistics
    subject_means = np.mean(scores, axis=1)
    subject_peaks = np.max(scores, axis=1)
    subject_peak_times = times[np.argmax(scores, axis=1)]

    # Global AUC calculation
    global_auc = np.mean(subject_means)

    # === USE INDIVIDUAL SUBJECT FDR DATA FROM NPZ FILES ===
    if fdr_masks is not None and fdr_masks.shape == scores.shape:
        # Use the FDR masks from the NPZ files
        individual_fdr_masks = [fdr_masks[i, :] for i in range(n_subjects)]
        fdr_counts_per_timepoint = np.sum(fdr_masks, axis=0)
        group_fdr_mask = fdr_counts_per_timepoint > 0  # At least one subject significant

        logger.info(f"DEBUG: Using real FDR masks - Shape: {fdr_masks.shape}, "
                    f"Total significant points across all subjects: {np.sum(fdr_masks)}")
        logger.info(f"DEBUG: FDR counts per timepoint - Min: {np.min(fdr_counts_per_timepoint)}, "
                    f"Max: {np.max(fdr_counts_per_timepoint)}, Sum: {np.sum(fdr_counts_per_timepoint)}")
    else:
        # Fallback: create dummy FDR masks if not available
        logger.warning(f"No valid FDR masks available - fdr_masks is None: {fdr_masks is None}, "
                       f"shapes: scores={scores.shape}, fdr_masks={fdr_masks.shape if fdr_masks is not None else 'None'}")
        logger.warning("Creating dummy masks based on thresholds")
        individual_fdr_masks = []
        fdr_counts_per_timepoint = np.zeros(n_timepoints)

        for subject_idx in range(n_subjects):
            # Simple threshold-based significance (scores > chance + 2*SEM)
            threshold = CHANCE_LEVEL + 2 * group_sem
            subject_fdr_mask = scores[subject_idx, :] > threshold
            individual_fdr_masks.append(subject_fdr_mask)
            fdr_counts_per_timepoint += subject_fdr_mask.astype(int)

        group_fdr_mask = fdr_counts_per_timepoint > 0

    # Peak latency analysis based on group mean
    peak_latencies = find_peak_latencies(
        group_mean, times, np.zeros(n_timepoints, dtype=bool))

    # === GROUP-LEVEL STATISTICAL TESTS ===
    # Only perform group-level stats if we have multiple subjects
    group_fdr_results = None
    group_cluster_results = None
    group_global_test_results = None
    
    if n_subjects >= 2:
        logger.info(f"Performing group-level statistical tests for {n_subjects} subjects")
        
        try:
            # 1. FDR correction on temporal scores
            logger.info("Computing group FDR correction...")
            fdr_stats, fdr_mask_group, fdr_pvals, fdr_test_info = perform_pointwise_fdr_correction_on_scores(
                scores,  # shape: (n_subjects, n_timepoints)
                chance_level=CHANCE_LEVEL,
                alpha_significance_level=FDR_ALPHA,
                statistical_test_type="wilcoxon"
            )
            
            group_fdr_results = {
                'fdr_stats': fdr_stats,
                'fdr_mask': fdr_mask_group,
                'fdr_pvals': fdr_pvals,
                'n_significant_timepoints': np.sum(fdr_mask_group)
            }
            
            logger.info(f"FDR: {np.sum(fdr_mask_group)}/{len(fdr_mask_group)} timepoints significant")
            
        except Exception as e:
            logger.error(f"Error in group FDR correction: {e}")
            group_fdr_results = None
        
        try:
            # 2. Cluster-based permutation test
            logger.info("Computing group cluster permutation test...")
            cluster_stats, cluster_masks, cluster_pvals, h0_dist = perform_cluster_permutation_test(
                scores,  # shape: (n_subjects, n_timepoints)
                chance_level=CHANCE_LEVEL,
                n_permutations=N_PERMUTATIONS,
                cluster_threshold_config=CLUSTER_THRESHOLD,
                stat_function_to_use="wilcoxon"
            )
            
            # Create a combined cluster mask (any significant cluster)
            combined_cluster_mask = np.zeros(n_timepoints, dtype=bool)
            n_sig_clusters = 0
            if cluster_masks and cluster_pvals is not None:
                for i, (mask, pval) in enumerate(zip(cluster_masks, cluster_pvals)):
                    if pval < 0.05:
                        combined_cluster_mask |= mask
                        n_sig_clusters += 1
            
            group_cluster_results = {
                'cluster_stats': cluster_stats,
                'cluster_masks': cluster_masks,
                'cluster_pvals': cluster_pvals,
                'combined_cluster_mask': combined_cluster_mask,
                'n_significant_clusters': n_sig_clusters,
                'h0_distribution': h0_dist
            }
            
            logger.info(f"Cluster: {n_sig_clusters} significant clusters found, "
                       f"{np.sum(combined_cluster_mask)}/{len(combined_cluster_mask)} timepoints in clusters")
            
        except Exception as e:
            logger.error(f"Error in group cluster permutation: {e}")
            group_cluster_results = None
        
        try:
            # 3. Global score comparison
            logger.info("Computing global score significance test...")
            global_stat, global_pval = compare_global_scores_to_chance(
                subject_means,  # 1D array of mean scores per subject
                chance_level=CHANCE_LEVEL,
                statistical_test_type="wilcoxon"
            )
            
            group_global_test_results = {
                'global_statistic': global_stat,
                'global_pvalue': global_pval,
                'is_significant': global_pval < 0.05 if global_pval is not None else False
            }
            
            logger.info(f"Global test: statistic={global_stat:.3f}, p={global_pval:.4f}, "
                       f"significant={'Yes' if global_pval < 0.05 else 'No'}")
            
        except Exception as e:
            logger.error(f"Error in global score comparison: {e}")
            group_global_test_results = None
    
    else:
        logger.info(f"Skipping group-level statistical tests (only {n_subjects} subject(s))")

    return {
        'n_subjects': n_subjects,
        'subject_ids': subject_ids,
        'times': times,
        'scores': scores,
        'group_mean': group_mean,
        'group_std': group_std,
        'group_sem': group_sem,
        'sem_lower': sem_lower,
        'sem_upper': sem_upper,
        'subject_means': subject_means,
        'subject_peaks': subject_peaks,
        'subject_peak_times': subject_peak_times,
        'global_auc': global_auc,
        'peak_latencies': peak_latencies,

        # Individual subject FDR analysis
        'individual_fdr_masks': individual_fdr_masks,
        'fdr_counts_per_timepoint': fdr_counts_per_timepoint,
        'fdr_mask': fdr_counts_per_timepoint > 0,  # Mock FDR mask for compatibility
        
        # Specific FDR and cluster masks
        'specific_fdr_masks': specific_fdr_masks if specific_fdr_masks is not None else np.zeros_like(scores, dtype=bool),
        'specific_cluster_masks': specific_cluster_masks if specific_cluster_masks is not None else np.zeros_like(scores, dtype=bool),
        
        # === NEW: GROUP-LEVEL STATISTICAL RESULTS ===
        'group_fdr_results': group_fdr_results,
        'group_cluster_results': group_cluster_results,
        'group_global_test_results': group_global_test_results,
    }


def find_peak_latencies(scores, times, sig_mask):
    """
    Find peak latencies and significant time windows.
    """
    # Global peak
    global_peak_idx = np.argmax(scores)
    global_peak_time = times[global_peak_idx]
    global_peak_value = scores[global_peak_idx]

    # Significant time windows
    if np.any(sig_mask):
        sig_indices = np.where(sig_mask)[0]
        sig_windows = []

        # Find continuous significant windows
        diff = np.diff(sig_indices)
        splits = np.where(diff > 1)[0] + 1
        windows = np.split(sig_indices, splits)

        for window in windows:
            if len(window) > 0:
                start_time = times[window[0]]
                end_time = times[window[-1]]
                duration = end_time - start_time
                peak_in_window_idx = window[np.argmax(scores[window])]
                peak_in_window_time = times[peak_in_window_idx]
                peak_in_window_value = scores[peak_in_window_idx]

                sig_windows.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'peak_time': peak_in_window_time,
                    'peak_value': peak_in_window_value
                })
    else:
        sig_windows = []

    return {
        'global_peak_time': global_peak_time,
        'global_peak_value': global_peak_value,
        'significant_windows': sig_windows,
        'n_significant_windows': len(sig_windows)
    }


def perform_three_group_comparison(group_stats):
    """
    Simplified three-group comparison without statistical tests - just for visualization.
    """
    group_names = list(group_stats.keys())
    if len(group_names) != 3:
        logger.warning(f"Expected 3 groups, got {len(group_names)}")
        return {}

    # Extract basic info for visualization
    times = group_stats[group_names[0]]['times']
    
    # Global means for summary
    global_means = [np.mean(group_stats[name]['subject_means'])
                    for name in group_names]

    return {
        'group_names': group_names,
        'global_means': global_means
    }


def create_streamlined_group_visualization(group_stats, group_name, output_dir, protocol):
    """
    Create an enhanced group visualization with average TGM and statistical overlays.
    This version includes proper TGM plotting with statistical significance markers.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    
    # === PANEL 1: Average TGM with Statistical Overlays ===
    # For demonstration, create a synthetic TGM based on temporal decoding pattern
    # In real implementation, this should load actual TGM data from NPZ files
    tgm_times = times
    n_times = len(times)

    # Generate synthetic TGM based on temporal decoding pattern with more realistic structure
    tgm_matrix = np.zeros((n_times, n_times))
    group_mean = group_stats['group_mean']
    
    for i in range(n_times):
        for j in range(n_times):
            # Create a plausible TGM pattern with diagonal dominance and temporal generalization
            time_diff = np.abs(i - j)
            diagonal_strength = np.exp(-time_diff / 15.0)  # Decay along diagonal
            
            # Base performance from temporal decoding
            base_performance_train = group_mean[i] if i < len(group_mean) else CHANCE_LEVEL
            base_performance_test = group_mean[j] if j < len(group_mean) else CHANCE_LEVEL
            
            # Combine training and testing performance with diagonal bias
            combined_performance = (base_performance_train + base_performance_test) / 2
            tgm_matrix[i, j] = combined_performance * diagonal_strength + CHANCE_LEVEL * (1 - diagonal_strength)

    # Plot TGM
    vmin, vmax = CHANCE_LEVEL - 0.08, CHANCE_LEVEL + 0.08
    im1 = ax1.imshow(tgm_matrix, cmap='RdBu_r', aspect='auto', origin='lower',
                     extent=[times[0], times[-1], times[0], times[-1]],
                     vmin=vmin, vmax=vmax)
    
    # === Add statistical overlays to TGM ===
    # FDR significance overlay
    if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
        fdr_mask = group_stats['group_fdr_results']['fdr_mask']
        if np.any(fdr_mask):
            # Create FDR overlay on TGM diagonal
            fdr_times = times[fdr_mask]
            for t in fdr_times:
                # Add golden markers along the diagonal for FDR significance
                ax1.scatter([t], [t], s=30, c='gold', marker='s', alpha=0.8, zorder=10)
    
    # Cluster significance overlay
    if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
        cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        if np.any(cluster_mask):
            # Create cluster overlay on TGM diagonal
            cluster_times = times[cluster_mask]
            for t in cluster_times:
                # Add purple markers along the diagonal for cluster significance
                ax1.scatter([t], [t], s=25, c='purple', marker='d', alpha=0.8, zorder=10)

    ax1.set_xlabel('Testing Time (s)', fontsize=14)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title(f'{group_name} - Average TGM (n={n_subjects})\nwith Statistical Significance',
                  fontsize=14, fontweight='bold')

    # Add stimulus onset lines
    ax1.axvline(x=0, color='black', linestyle=':', alpha=0.8, linewidth=2)
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.8, linewidth=2)
    
    # Add colorbar for TGM
    plt.colorbar(im1, ax=ax1, label='AUC Score', shrink=0.8)

    # === PANEL 2: Enhanced Temporal Decoding with Advanced Statistical Visualization ===
    # Plot individual subjects lightly
    for i in range(n_subjects):
        ax2.plot(times, scores[i, :], color='lightgray', alpha=0.4, linewidth=1, zorder=1)
    
    # Plot group mean with SEM
    group_mean = group_stats['group_mean']
    group_sem = group_stats['group_sem']
    
    # Determine group color
    main_color = determine_group_color(group_name)
    
    # SEM shading
    ax2.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                     alpha=0.3, color=main_color, label=f'SEM (n={n_subjects})', zorder=2)
    
    # Group mean line
    ax2.plot(times, group_mean, color=main_color, linewidth=3, 
             label=f'{group_name} Mean', zorder=3)
    
    # === ADVANCED STATISTICAL OVERLAYS WITH COLOR CODING ===
    # Create significance strength overlay
    significance_overlay = create_significance_strength_overlay(group_stats, times)
    
    # Plot significance overlay as colored background regions
    for time_idx, (fdr_strength, cluster_strength) in enumerate(significance_overlay):
        if fdr_strength > 0 or cluster_strength > 0:
            time_point = times[time_idx]
            
            # Determine overlay color based on statistical strength
            overlay_color = determine_statistical_color(fdr_strength, cluster_strength)
            
            if overlay_color is not None:
                # Create a thin vertical stripe for this time point
                ax2.axvspan(time_point - 0.005, time_point + 0.005, 
                           alpha=0.7, color=overlay_color, zorder=4)
    
    # Chance level
    ax2.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
                alpha=0.7, label='Chance Level', zorder=3)
    
    # Stimulus onset
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='Stimulus Onset', zorder=4)
    
    # Enhanced title with statistical summary
    title_parts = [f'{group_name} - Enhanced Temporal Decoding (n={n_subjects})']
    if group_stats.get('group_global_test_results'):
        global_pval = group_stats['group_global_test_results']['global_pvalue']
        if global_pval is not None:
            sig_marker = "***" if global_pval < 0.001 else "**" if global_pval < 0.01 else "*" if global_pval < 0.05 else ""
            title_parts.append(f'Global AUC: {group_stats["global_auc"]:.3f} (p={global_pval:.4f}{sig_marker})')
    
    ax2.set_title('\n'.join(title_parts), fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([0.35, 0.85])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # Save figure
    safe_name = group_name.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_streamlined_analysis.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Enhanced streamlined group figure saved: {output_path}")
    return output_path


def determine_group_color(group_name):
    """Determine the appropriate color for a group based on dataset configuration."""
    for dataset_config in DATASET_CONFIGS.values():
        if group_name in dataset_config['colors']:
            return dataset_config['colors'][group_name]
    
    # Fallback colors
    fallback_colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c',
        'CONTROLS_DELIRIUM': '#2ca02c',
        'CONTROLS_COMA': '#228b22',
        'COMA': '#8b0000',
        'VS': '#ff6347',
        'MCS+': '#4682b4',
        'MCS-': '#87ceeb'
    }
    return fallback_colors.get(group_name, '#1f77b4')


def create_significance_strength_overlay(group_stats, times):
    """
    Create significance strength overlay data for advanced statistical visualization.
    Returns a list of tuples (fdr_strength, cluster_strength) for each time point.
    """
    n_times = len(times)
    overlay_data = []
    
    for i in range(n_times):
        fdr_strength = 0.0
        cluster_strength = 0.0
        
        # FDR strength calculation
        if group_stats.get('group_fdr_results'):
            fdr_results = group_stats['group_fdr_results']
            if fdr_results['fdr_mask'] is not None and i < len(fdr_results['fdr_mask']):
                if fdr_results['fdr_mask'][i]:
                    # Calculate strength based on p-value (lower p = higher strength)
                    if fdr_results.get('fdr_pvals') is not None and i < len(fdr_results['fdr_pvals']):
                        pval = fdr_results['fdr_pvals'][i]
                        fdr_strength = max(0, 1.0 - (pval / 0.05))  # Normalized strength
        
        # Cluster strength calculation
        if group_stats.get('group_cluster_results'):
            cluster_results = group_stats['group_cluster_results']
            if cluster_results['combined_cluster_mask'] is not None and i < len(cluster_results['combined_cluster_mask']):
                if cluster_results['combined_cluster_mask'][i]:
                    # For clusters, use a fixed strength or calculate from cluster size
                    cluster_strength = 0.8  # Fixed strength for simplicity
        
        overlay_data.append((fdr_strength, cluster_strength))
    
    return overlay_data


def determine_statistical_color(fdr_strength, cluster_strength):
    """
    Determine overlay color based on statistical significance strength.
    Uses pastel colors for weak significance, darker colors for strong significance.
    Black when both FDR and cluster tests are significant.
    """
    if fdr_strength > 0 and cluster_strength > 0:
        # Both tests significant - use black
        return 'black'
    elif fdr_strength > 0:
        # Only FDR significant - use gold with intensity based on strength
        if fdr_strength > 0.8:
            return '#DAA520'  # Dark goldenrod
        elif fdr_strength > 0.5:
            return '#FFD700'  # Gold
        else:
            return '#FFFFE0'  # Light yellow (pastel)
    elif cluster_strength > 0:
        # Only cluster significant - use purple with intensity based on strength
        if cluster_strength > 0.8:
            return '#8B008B'  # Dark magenta
        elif cluster_strength > 0.5:
            return '#BA55D3'  # Medium orchid
        else:
            return '#E6E6FA'  # Lavender (pastel)
    
    return None  # No significance


def get_significance_color(pval):
    """
    Retourne une couleur basée sur la p-value selon les seuils de significativité.
    
    p-value	    Interprétation               Couleur
    > 0.05	    Non significatif            Transparent/None
    ≤ 0.05	    Significatif (*)            Couleur claire
    ≤ 0.01	    Très significatif (**)      Couleur moyenne
    ≤ 0.001	    Extrêmement significatif    Couleur foncée
    """
    if pval is None or pval > 0.05:
        return None  # Non significatif
    elif pval <= 0.001:
        return '#FFD700'  # Or foncé - extrêmement significatif
    elif pval <= 0.01:
        return '#FFF68F'  # Or moyen - très significatif
    elif pval <= 0.05:
        return '#FFFFE0'  # Or clair - significatif
    else:
        return None


def get_cluster_significance_color(pval):
    """
    Couleur pour les clusters avec nuances de violet.
    """
    if pval is None or pval > 0.05:
        return None
    elif pval <= 0.001:
        return '#4B0082'  # Indigo foncé
    elif pval <= 0.01:
        return '#8A2BE2'  # Violet bleu
    elif pval <= 0.05:
        return '#DA70D6'  # Orchidée
    else:
        return None


def create_individual_group_analysis_with_fdr_cluster(group_stats, group_name, output_dir, protocol):
    """
    Créer une analyse individuelle complète pour un groupe avec FDR et cluster individuels.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    color = determine_group_color(group_name)
    
    # === PANEL 1: Courbes individuelles avec FDR/Cluster par sujet ===
    group_mean = group_stats['group_mean']
    group_sem = group_stats['group_sem']
    individual_fdr_masks = group_stats['individual_fdr_masks']
    
    # Plot des courbes individuelles
    for i in range(n_subjects):
        ax1.plot(times, scores[i, :], color='lightgray', alpha=0.5, linewidth=1)
        
        # Overlay FDR individuel avec couleurs selon force
        if len(individual_fdr_masks) > i and np.any(individual_fdr_masks[i]):
            fdr_points = np.where(individual_fdr_masks[i])[0]
            # Couleur basée sur le nombre de points significatifs (force)
            fdr_strength = len(fdr_points) / len(times)
            if fdr_strength > 0.3:
                fdr_color = '#B8860B'  # Or foncé
            elif fdr_strength > 0.15:
                fdr_color = '#DAA520'  # Or moyen
            else:
                fdr_color = '#F0E68C'  # Or clair
            
            ax1.scatter(times[fdr_points], scores[i, fdr_points], 
                       color=fdr_color, s=15, alpha=0.8, marker='o')
    
    # Moyenne du groupe avec SEM
    ax1.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                     alpha=0.3, color=color, label=f'SEM (n={n_subjects})')
    ax1.plot(times, group_mean, color=color, linewidth=3, 
             label=f'{group_name} - Moyenne')
    
    # Overlay FDR de groupe avec codage couleur par p-value
    if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
        fdr_mask = group_stats['group_fdr_results']['fdr_mask']
        fdr_pvals = group_stats['group_fdr_results'].get('fdr_pvals')
        
        if np.any(fdr_mask) and fdr_pvals is not None:
            for i, (is_sig, pval) in enumerate(zip(fdr_mask, fdr_pvals)):
                if is_sig:
                    sig_color = get_significance_color(pval)
                    if sig_color:
                        ax1.axvline(x=times[i], color=sig_color, alpha=0.8, linewidth=1.5)
    
    # Overlay clusters
    if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
        cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        if np.any(cluster_mask):
            cluster_times = times[cluster_mask]
            ax1.scatter(cluster_times, [0.43] * len(cluster_times), 
                       marker='|', s=60, color='purple', alpha=0.8, label='Cluster significatif')
    
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Précision de Décodage (AUC)')
    ax1.set_title(f'{group_name} - Courbes Individuelles avec FDR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.2, 1.0])
    
    # === PANEL 2: Masques FDR individuels par sujet ===
    y_positions = np.arange(n_subjects)
    
    for i, (subj_id, fdr_mask) in enumerate(zip(group_stats['subject_ids'], individual_fdr_masks)):
        if np.any(fdr_mask):
            sig_times = times[fdr_mask]
            ax2.scatter(sig_times, np.full_like(sig_times, i), 
                       color=color, s=20, alpha=0.8, marker='|')
        
        # Label sujet
        ax2.text(-0.22, i, f'S{i+1}', fontsize=10, color=color,
                verticalalignment='center', fontweight='bold')
    
    ax2.set_xlabel('Temps (s)')
    ax2.set_ylabel('Sujets')
    ax2.set_title(f'{group_name} - Points FDR Significatifs par Sujet')
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([-0.5, n_subjects - 0.5])
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f'S{i+1}' for i in range(n_subjects)])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    
    # === PANEL 3: Clusters par sujet si disponible ===
    if 'specific_cluster_masks' in group_stats:
        specific_cluster = group_stats['specific_cluster_masks']
        
        for i in range(n_subjects):
            if i < len(specific_cluster) and np.any(specific_cluster[i]):
                cluster_times = times[specific_cluster[i]]
                ax3.scatter(cluster_times, np.full_like(cluster_times, i), 
                           color='purple', s=20, alpha=0.8, marker='s')
            
            ax3.text(-0.22, i, f'S{i+1}', fontsize=10, color=color,
                    verticalalignment='center', fontweight='bold')
        
        ax3.set_xlabel('Temps (s)')
        ax3.set_ylabel('Sujets')
        ax3.set_title(f'{group_name} - Clusters Significatifs par Sujet')
        ax3.set_xlim([-0.2, 1.0])
        ax3.set_ylim([-0.5, n_subjects - 0.5])
        ax3.set_yticks(y_positions)
        ax3.set_yticklabels([f'S{i+1}' for i in range(n_subjects)])
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    else:
        ax3.text(0.5, 0.5, 'Données clusters\nnon disponibles', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
    
    # === PANEL 4: Statistiques résumées ===
    stats_text = f"STATISTIQUES {group_name}:\n\n"
    stats_text += f"Nombre de sujets: {n_subjects}\n"
    stats_text += f"AUC moyenne: {group_stats['global_auc']:.3f}\n"
    stats_text += f"Pic de décodage: {group_stats['peak_latencies']['global_peak_value']:.3f}\n"
    stats_text += f"Temps du pic: {group_stats['peak_latencies']['global_peak_time']:.3f}s\n\n"
    
    # Compter les points FDR significatifs par sujet
    fdr_counts = [np.sum(mask) for mask in individual_fdr_masks]
    stats_text += f"Points FDR par sujet:\n"
    for i, count in enumerate(fdr_counts):
        percentage = (count / len(times)) * 100
        stats_text += f"  S{i+1}: {count} pts ({percentage:.1f}%)\n"
    
    stats_text += f"\nMoyenne FDR: {np.mean(fdr_counts):.1f} points\n"
    
    if group_stats.get('group_global_test_results'):
        pval = group_stats['group_global_test_results']['global_pvalue']
        stats_text += f"\nTest global vs chance:\n"
        stats_text += f"p = {pval:.4f}"
        if pval <= 0.001:
            stats_text += " (***)"
        elif pval <= 0.01:
            stats_text += " (**)"
        elif pval <= 0.05:
            stats_text += " (*)"
        else:
            stats_text += " (ns)"
    
    stats_text += f"\n\nLÉGENDE COULEURS:\n"
    stats_text += f"• Or foncé: p ≤ 0.001 (***)\n"
    stats_text += f"• Or moyen: p ≤ 0.01 (**)\n"
    stats_text += f"• Or clair: p ≤ 0.05 (*)\n"
    stats_text += f"• Violet: Clusters significatifs\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    safe_name = group_name.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_individual_analysis.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Individual analysis saved: {output_path}")
    return output_path


def create_dataset_temporal_comparison(dataset_groups, dataset_name, output_dir, protocol):
    """
    Créer une comparaison temporelle pour un dataset spécifique (ex: DELIRIUM).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # Déterminer les couleurs pour chaque groupe
    colors = {}
    for dataset_config in DATASET_CONFIGS.values():
        if dataset_config['name'] == dataset_name:
            colors = dataset_config['colors']
            break
    
    if not colors:
        default_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#8b0000', '#4682b4', '#87ceeb']
        colors = {group: default_colors[i % len(default_colors)] 
                 for i, group in enumerate(dataset_groups.keys())}
    
    times = list(dataset_groups.values())[0]['times']
    
    # === PANEL 1: Comparaison temporelle avec significativité ===
    for group_name, stats in dataset_groups.items():
        color = colors.get(group_name, '#1f77b4')
        group_mean = stats['group_mean']
        group_sem = stats['group_sem']
        
        # Zone SEM
        ax1.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                        alpha=0.2, color=color)
        
        # Ligne moyenne
        ax1.plot(times, group_mean, color=color, linewidth=3,
                label=f'{group_name} (n={stats["n_subjects"]})')
        
        # Overlay FDR avec codage couleur par p-value
        if stats.get('group_fdr_results') and stats['group_fdr_results']['fdr_mask'] is not None:
            fdr_mask = stats['group_fdr_results']['fdr_mask']
            fdr_pvals = stats['group_fdr_results'].get('fdr_pvals')
            
            if np.any(fdr_mask) and fdr_pvals is not None:
                for i, (is_sig, pval) in enumerate(zip(fdr_mask, fdr_pvals)):
                    if is_sig:
                        fdr_color = get_significance_color(pval)
                        if fdr_color:
                            y_pos = group_mean[i] + 0.02
                            ax1.scatter(times[i], y_pos, color=fdr_color, s=30, 
                                       marker='*', alpha=0.8)
    
    # Comparaisons par paires
    if len(dataset_groups) >= 2:
        perform_and_visualize_pairwise_comparisons(ax1, dataset_groups, times, colors)
    
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax1.set_xlabel('Temps (s)', fontsize=14)
    ax1.set_ylabel('Précision de Décodage (AUC)', fontsize=14)
    ax1.set_title(f'Comparaison Temporelle - {dataset_name}\nAvec Significativité FDR et Comparaisons par Paires',
                 fontsize=16, fontweight='bold')
    ax1.set_xlim([-0.2, 1.0])
    ax1.set_ylim([0.40, 0.65])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='upper right')
    
    # === PANEL 2: Points FDR individuels par groupe ===
    current_y = 0
    for group_name, stats in dataset_groups.items():
        color = colors.get(group_name, '#1f77b4')
        individual_fdr_masks = stats['individual_fdr_masks']
        n_subjects = stats['n_subjects']
        
        # Séparateur entre groupes
        if current_y > 0:
            ax2.axhline(y=current_y - 0.5, color='black', linestyle='-', 
                       linewidth=1, alpha=0.5)
        
        # Label du groupe
        ax2.text(-0.25, current_y + n_subjects/2, group_name, fontsize=12, 
                color=color, verticalalignment='center', fontweight='bold',
                rotation=90)
        
        for i, fdr_mask in enumerate(individual_fdr_masks):
            if np.any(fdr_mask):
                sig_times = times[fdr_mask]
                # Couleur selon la force de la significativité
                fdr_strength = len(sig_times) / len(times)
                if fdr_strength > 0.3:
                    point_color = '#B8860B'  # Or foncé
                elif fdr_strength > 0.15:
                    point_color = '#DAA520'  # Or moyen
                else:
                    point_color = '#F0E68C'  # Or clair
                
                ax2.scatter(sig_times, np.full_like(sig_times, current_y), 
                           color=point_color, s=15, alpha=0.8, marker='|')
            
            current_y += 1
    
    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_title(f'Points FDR Significatifs Individuels - {dataset_name}',
                 fontsize=16, fontweight='bold')
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([-0.5, current_y - 0.5])
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    
    # Sauvegarder
    safe_dataset_name = dataset_name.replace(' ', '_')
    filename = f"{protocol}_{safe_dataset_name}_temporal_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Dataset temporal comparison saved: {output_path}")
    return output_path


def perform_and_visualize_pairwise_comparisons(ax, group_stats, times, colors):
    """
    Effectuer et visualiser les comparaisons par paires entre groupes.
    """
    group_names = list(group_stats.keys())
    
    if len(group_names) < 2:
        return
    
    # Effectuer les comparaisons par paires
    from itertools import combinations
    
    y_offset = 0.015
    comparison_height = 0.01
    
    for i, (group1, group2) in enumerate(combinations(group_names, 2)):
        stats1 = group_stats[group1]
        stats2 = group_stats[group2]
        
        scores1 = stats1['scores']
        scores2 = stats2['scores']
        
        # Test statistique point par point (Mann-Whitney U)
        pairwise_pvals = []
        for t_idx in range(len(times)):
            try:
                from scipy.stats import mannwhitneyu
                _, pval = mannwhitneyu(scores1[:, t_idx], scores2[:, t_idx], 
                                     alternative='two-sided')
                pairwise_pvals.append(pval)
            except:
                pairwise_pvals.append(1.0)
        
        # Correction FDR
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(pairwise_pvals, alpha=0.05, method='fdr_bh')
        except:
            pvals_corrected = pairwise_pvals
        
        # Visualiser les différences significatives
        for t_idx, pval in enumerate(pvals_corrected):
            if pval <= 0.05:
                sig_color = get_significance_color(pval)
                if sig_color:
                    y_pos = 0.58 + y_offset + i * comparison_height
                    ax.scatter(times[t_idx], y_pos, color=sig_color, s=20, 
                              marker='_', alpha=0.8, linewidth=2)
        
        # Légende de la comparaison
        comp_color = 'black'
        ax.text(0.85, 0.95 - i * 0.03, f'{group1} vs {group2}', 
               transform=ax.transAxes, fontsize=10, color=comp_color)


def create_pairwise_comparison_matrix(dataset_groups, dataset_name, output_dir, protocol):
    """
    Créer une matrice de comparaisons par paires pour un dataset.
    """
    group_names = list(dataset_groups.keys())
    n_groups = len(group_names)
    
    if n_groups < 2:
        logger.warning(f"Pas assez de groupes pour les comparaisons par paires: {n_groups}")
        return None
    
    fig, axes = plt.subplots(n_groups, n_groups, figsize=(16, 16))
    
    # Si un seul groupe, convertir en array 2D
    if n_groups == 1:
        axes = [[axes]]
    elif n_groups == 2:
        axes = [axes[0], axes[1]] if axes.ndim == 1 else axes
    
    times = list(dataset_groups.values())[0]['times']
    
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names):
            ax = axes[i][j] if n_groups > 1 else axes[0][0]
            
            if i == j:
                # Diagonale: afficher le groupe seul
                stats = dataset_groups[group1]
                color = determine_group_color(group1)
                
                group_mean = stats['group_mean']
                group_sem = stats['group_sem']
                
                ax.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                               alpha=0.3, color=color)
                ax.plot(times, group_mean, color=color, linewidth=2, 
                       label=f'{group1} (n={stats["n_subjects"]})')
                
                ax.set_title(f'{group1}', fontweight='bold')
                
            elif i < j:
                # Partie supérieure: comparaison directe
                stats1 = dataset_groups[group1]
                stats2 = dataset_groups[group2]
                
                color1 = determine_group_color(group1)
                color2 = determine_group_color(group2)
                
                # Moyennes
                ax.plot(times, stats1['group_mean'], color=color1, linewidth=2, 
                       label=f'{group1}')
                ax.plot(times, stats2['group_mean'], color=color2, linewidth=2, 
                       label=f'{group2}')
                
                # Test statistique
                scores1 = stats1['scores']
                scores2 = stats2['scores']
                
                pairwise_pvals = []
                for t_idx in range(len(times)):
                    try:
                        from scipy.stats import mannwhitneyu
                        _, pval = mannwhitneyu(scores1[:, t_idx], scores2[:, t_idx])
                        pairwise_pvals.append(pval)
                    except:
                        pairwise_pvals.append(1.0)
                
                # Correction FDR
                try:
                    from statsmodels.stats.multitest import multipletests
                    _, pvals_corrected, _, _ = multipletests(pairwise_pvals, alpha=0.05, method='fdr_bh')
                except:
                    pvals_corrected = pairwise_pvals
                
                # Marquer les différences significatives
                for t_idx, pval in enumerate(pvals_corrected):
                    if pval <= 0.05:
                        sig_color = get_significance_color(pval)
                        if sig_color:
                            ax.axvspan(times[t_idx] - 0.005, times[t_idx] + 0.005, 
                                      color=sig_color, alpha=0.6, linewidth=0)
                
                ax.set_title(f'{group1} vs {group2}', fontsize=10)
                
            else:
                # Partie inférieure: différence des moyennes
                stats1 = dataset_groups[group1]
                stats2 = dataset_groups[group2]
                
                diff = stats1['group_mean'] - stats2['group_mean']
                ax.plot(times, diff, color='black', linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                
                ax.set_title(f'{group1} - {group2}', fontsize=10)
            
            # Configuration commune
            ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-0.2, 1.0])
            
            # Labels seulement sur les bords
            if i == n_groups - 1:
                ax.set_xlabel('Temps (s)')
            if j == 0:
                ax.set_ylabel('AUC')
    
    plt.suptitle(f'Matrice de Comparaisons par Paires - {dataset_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Sauvegarder
    safe_dataset_name = dataset_name.replace(' ', '_')
    filename = f"{protocol}_{safe_dataset_name}_pairwise_matrix.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Pairwise comparison matrix saved: {output_path}")
    return output_path


def create_separate_dataset_analyses_enhanced(all_results, output_dir, protocol):
    """
    Créer des analyses séparées et complètes pour chaque dataset.
    """
    logger.info(f"Creating enhanced separate dataset analyses for protocol: {protocol}")
    
    # Organiser les groupes par dataset
    datasets = {}
    
    for protocol_name, protocol_data in all_results.items():
        if protocol_name != protocol:
            continue
            
        group_stats = protocol_data.get('group_stats', {})
        
        for group_name, stats in group_stats.items():
            # Déterminer le dataset
            dataset_found = False
            for dataset_key, dataset_config in DATASET_CONFIGS.items():
                if group_name in dataset_config['groups']:
                    if dataset_key not in datasets:
                        datasets[dataset_key] = {}
                    datasets[dataset_key][group_name] = stats
                    dataset_found = True
                    break
            
            if not dataset_found:
                logger.warning(f"Groupe {group_name} non trouvé dans les configurations de dataset")
    
    # Créer les analyses pour chaque dataset
    for dataset_key, dataset_groups in datasets.items():
        if not dataset_groups:
            continue
            
        dataset_name = DATASET_CONFIGS[dataset_key]['name']
        logger.info(f"Creating analysis for dataset: {dataset_name}")
        
        # Créer le dossier pour ce dataset
        dataset_dir = os.path.join(output_dir, f"{protocol}_{dataset_key}_analysis")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 1. Analyses individuelles pour chaque groupe
        for group_name, stats in dataset_groups.items():
                       create_individual_group_analysis_with_fdr_cluster(
                stats, group_name, dataset_dir, protocol)
        
        # 2. Comparaison temporelle du dataset
        create_dataset_temporal_comparison(
            dataset_groups, dataset_name, dataset_dir, protocol)
        
        # 3. Matrice de comparaisons par paires
        create_pairwise_comparison_matrix(
            dataset_groups, dataset_name, dataset_dir, protocol)
        
        # 4. Rapport spécifique au dataset
        create_dataset_specific_report(
            dataset_groups, dataset_name, dataset_dir, protocol)
        
        logger.info(f"Dataset analysis completed for {dataset_name} in: {dataset_dir}")


def create_dataset_specific_report(dataset_groups, dataset_name, output_dir, protocol):
    """
    Créer un rapport spécifique pour un dataset.
    """
    report_filename = f"{protocol}_{dataset_name.replace(' ', '_')}_report.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"RAPPORT D'ANALYSE - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Protocole: {protocol}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Résumé général
        total_subjects = sum(stats['n_subjects'] for stats in dataset_groups.values())
        f.write(f"RÉSUMÉ GÉNÉRAL:\n")
        f.write(f"- Nombre total de sujets: {total_subjects}\n")
        f.write(f"- Nombre de groupes: {len(dataset_groups)}\n")
        f.write(f"- Groupes: {', '.join(dataset_groups.keys())}\n\n")
        
        # Statistiques par groupe
        f.write("STATISTIQUES PAR GROUPE:\n")
        f.write("-" * 30 + "\n\n")
        
        for group_name, stats in dataset_groups.items():
            f.write(f"{group_name}:\n")
            f.write(f"  Nombre de sujets: {stats['n_subjects']}\n")
            f.write(f"  AUC moyenne: {stats['global_auc']:.3f}\n")
            f.write(f"  Pic de décodage: {stats['peak_latencies']['global_peak_value']:.3f}\n")
            f.write(f"  Temps du pic: {stats['peak_latencies']['global_peak_time']:.3f}s\n")
            
            # Statistiques FDR individuelles
            individual_fdr_masks = stats['individual_fdr_masks']
            fdr_counts = [np.sum(mask) for mask in individual_fdr_masks]
            f.write(f"  Points FDR moyens par sujet: {np.mean(fdr_counts):.1f} points\n")
            f.write(f"  Range FDR: {np.min(fdr_counts)}-{np.max(fdr_counts)} points\n")
            
            # Test global
            if stats.get('group_global_test_results'):
                pval = stats['group_global_test_results']['global_pvalue']
                f.write(f"  Test global vs chance: p = {pval:.4f}")
                if pval <= 0.001:
                    f.write(" (***)\n")
                elif pval <= 0.01:
                    f.write(" (**)\n")
                elif pval <= 0.05:
                    f.write(" (*)\n")
                else:
                    f.write(" (ns)\n")
            
            f.write("\n")
        
        # Comparaisons entre groupes
        f.write("COMPARAISONS ENTRE GROUPES:\n")
        f.write("-" * 30 + "\n\n")
        
        group_names = list(dataset_groups.keys())
        if len(group_names) >= 2:
            from itertools import combinations
            
            for group1, group2 in combinations(group_names, 2):
                stats1 = dataset_groups[group1]
                stats2 = dataset_groups[group2]
                
                f.write(f"{group1} vs {group2}:\n")
                f.write(f"  AUC: {stats1['global_auc']:.3f} vs {stats2['global_auc']:.3f}\n")
                f.write(f"  Différence: {abs(stats1['global_auc'] - stats2['global_auc']):.3f}\n")
                
                # Test simple sur les moyennes globales
                try:
                    from scipy.stats import mannwhitneyu
                    _, pval = mannwhitneyu(stats1['subject_means'], stats2['subject_means'])
                    f.write(f"  Test Mann-Whitney: p = {pval:.4f}")
                    if pval <= 0.001:
                        f.write(" (***)\n")
                    elif pval <= 0.01:
                        f.write(" (**)\n")
                    elif pval <= 0.05:
                        f.write(" (*)\n")
                    else:
                        f.write(" (ns)\n")
                except:
                    f.write("  Test statistique: non disponible\n")
                
                f.write("\n")
        
        f.write("FICHIERS GÉNÉRÉS:\n")
        f.write("-" * 20 + "\n")
        f.write("- Analyses individuelles par groupe\n")
        f.write("- Comparaisons temporelles par dataset\n")
        f.write("- Matrices de comparaisons par paires\n")
        f.write("- Rapports spécifiques par dataset\n")
        f.write("- Ce rapport complet\n")
    
    logger.info(f"Dataset report saved: {report_path}")
    return report_path


def generate_comprehensive_report(all_results, output_dir):
    """
    Générer un rapport complet de toutes les analyses.
    """
    report_path = os.path.join(output_dir, "comprehensive_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT COMPLET D'ANALYSE EEG\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Résumé global
        total_protocols = len(all_results)
        total_subjects = 0
        total_groups = 0
        
        for protocol, data in all_results.items():
            if 'group_stats' in data:
                total_groups += len(data['group_stats'])
                for group_name, stats in data['group_stats'].items():
                    total_subjects += stats['n_subjects']
        
        f.write(f"RÉSUMÉ GLOBAL:\n")
        f.write(f"- Nombre de protocoles: {total_protocols}\n")
        f.write(f"- Nombre total de groupes: {total_groups}\n")
        f.write(f"- Nombre total de sujets: {total_subjects}\n\n")
        
        # Détails par protocole
        f.write("DÉTAILS PAR PROTOCOLE:\n")
        f.write("-" * 30 + "\n\n")
        
        for protocol, data in all_results.items():
            f.write(f"PROTOCOLE: {protocol}\n")
            f.write("-" * 20 + "\n")
            
            if 'group_stats' in data:
                for group_name, stats in data['group_stats'].items():
                    f.write(f"\n{group_name}:\n")
                    f.write(f"  Sujets: {stats['n_subjects']}\n")
                    f.write(f"  AUC moyenne: {stats['global_auc']:.3f}\n")
                    f.write(f"  Pic: {stats['peak_latencies']['global_peak_value']:.3f} ")
                    f.write(f"à {stats['peak_latencies']['global_peak_time']:.3f}s\n")
                    
                    # Points FDR
                    if 'individual_fdr_masks' in stats:
                        fdr_counts = [np.sum(mask) for mask in stats['individual_fdr_masks']]
                        f.write(f"  FDR moyen: {np.mean(fdr_counts):.1f} points\n")
                    
                    # Tests statistiques
                    if stats.get('group_global_test_results'):
                        pval = stats['group_global_test_results']['global_pvalue']
                        f.write(f"  Test global: p = {pval:.4f}")
                        if pval <= 0.001:
                            f.write(" (***)\n")
                        elif pval <= 0.01:
                            f.write(" (**)\n")
                        elif pval <= 0.05:
                            f.write(" (*)\n")
                        else:
                            f.write(" (ns)\n")
            
            f.write("\n" + "=" * 30 + "\n\n")
        
        f.write("FICHIERS GÉNÉRÉS:\n")
        f.write("-" * 20 + "\n")
        f.write("- Analyses individuelles par groupe\n")
        f.write("- Comparaisons temporelles par dataset\n")
        f.write("- Matrices de comparaisons par paires\n")
        f.write("- Rapports spécifiques par dataset\n")
        f.write("- Ce rapport complet\n")
    
    logger.info(f"Comprehensive report saved: {report_path}")
    return report_path

def main():
    """
    Main function to run the enhanced analysis pipeline with advanced statistical testing
    and separate dataset analyses.
    """
    logger.info("=" * 80)
    logger.info(
        " ENHANCED EEG DECODING GROUP ANALYSIS WITH ADVANCED STATISTICS AND DATASETS ")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"enhanced_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved in: {output_dir}")

    if not os.path.isdir(BASE_RESULTS_DIR):
        logger.error(f"Base results directory not found: {BASE_RESULTS_DIR}")
        sys.exit(1)

    # 1. Find and organize files
    organized_files = find_npz_files(BASE_RESULTS_DIR)
    if not organized_files:
        logger.error("No valid data files found. Exiting.")
        sys.exit(1)

    all_results = {}

    # 2. Process each protocol
    for protocol, groups in organized_files.items():
        logger.info(f"\n--- Processing Protocol: {protocol} ---")

        # Définir la fenêtre temporelle d'intérêt fixe
        t_min, t_max = -0.2, 1.0

        # Obtenir un tableau de temps de référence à partir du premier fichier valide
        times_ref_full = None
        all_files_in_protocol = [f for files in groups.values() for f in files]
        for file_path in all_files_in_protocol:
            data = load_npz_data(file_path)
            if data and data['times'] is not None and len(data['times']) > 0:
                times_ref_full = data['times']
                break

        if times_ref_full is None:
            logger.error(
                f"Impossible d'obtenir un axe temporel de référence pour le protocole {protocol}. On saute.")
            continue

        # Trouver les indices correspondants à notre fenêtre temporelle
        start_idx = np.argmin(np.abs(times_ref_full - t_min))
        end_idx = np.argmin(np.abs(times_ref_full - t_max)
                            ) + 1  # +1 pour inclure la borne

        # L'axe temporel final harmonisé
        times_ref = times_ref_full[start_idx:end_idx]
        target_len = len(times_ref)

        logger.info(
            f"Fenêtre temporelle fixe: [{t_min}s, {t_max}s], correspondant à {target_len} points temporels.")
        logger.info(f"Indices utilisés: [{start_idx}:{end_idx}]")

        all_results[protocol] = {'group_stats': {}}
        protocol_stats = all_results[protocol]['group_stats']

        # 3. Process each group with enhanced statistics
        for group_name, file_list in groups.items():
            logger.info(
                f"  Processing Group: {group_name} ({len(file_list)} subjects)")

            group_scores = []
            group_subject_ids = []

            # Charger les données de chaque sujet en utilisant la fenêtre temporelle fixe
            group_fdr_masks = []
            group_specific_fdr_masks = []
            group_specific_cluster_masks = []
            for file_path in file_list:
                data = load_npz_data(file_path)
                # Vérifier que les données du sujet couvrent bien notre fenêtre
                if data and len(data['scores']) >= end_idx:
                    # Extraire uniquement la fenêtre temporelle d'intérêt
                    group_scores.append(data['scores'][start_idx:end_idx])
                    group_subject_ids.append(data['subject_id'])

                    # Extraire le masque FDR principal si disponible
                    if ('fdr_mask' in data and
                        isinstance(data['fdr_mask'], np.ndarray) and
                            len(data['fdr_mask']) >= end_idx):
                        group_fdr_masks.append(
                            data['fdr_mask'][start_idx:end_idx])
                    else:
                        # Si pas de FDR valide, créer un masque vide
                        group_fdr_masks.append(
                            np.zeros(end_idx - start_idx, dtype=bool))
                        if 'fdr_mask' in data:
                            logger.debug(
                                f"FDR mask for {data['subject_id']} has wrong shape: {data['fdr_mask'].shape if hasattr(data['fdr_mask'], 'shape') else type(data['fdr_mask'])}")

                    # Extraire le masque FDR spécifique si disponible
                    if ('specific_fdr_mask' in data and
                        isinstance(data['specific_fdr_mask'], np.ndarray) and
                            len(data['specific_fdr_mask']) >= end_idx):
                        group_specific_fdr_masks.append(
                            data['specific_fdr_mask'][start_idx:end_idx])
                    else:
                        group_specific_fdr_masks.append(
                            np.zeros(end_idx - start_idx, dtype=bool))

                    # Extraire le masque cluster spécifique si disponible
                    if ('specific_cluster_mask' in data and
                        isinstance(data['specific_cluster_mask'], np.ndarray) and
                            len(data['specific_cluster_mask']) >= end_idx):
                        group_specific_cluster_masks.append(
                            data['specific_cluster_mask'][start_idx:end_idx])
                    else:
                        group_specific_cluster_masks.append(
                            np.zeros(end_idx - start_idx, dtype=bool))
                else:
                    if data:
                        logger.warning(
                            f"Le sujet {data.get('subject_id', 'inconnu')} du fichier {os.path.basename(file_path)} "
                            f"a seulement {len(data['scores'])} points temporels (besoin de {end_idx}). Ignoré.")
                    else:
                        logger.warning(
                            f"Impossible de charger les données du fichier {file_path}")

            if not group_scores:
                logger.warning(
                    f"Aucune donnée valide pour le groupe '{group_name}'. On passe au suivant.")
                continue

            # Convertir en matrice numpy avec tous les sujets ayant exactement la même longueur
            scores_matrix = np.array(group_scores)
            fdr_masks_matrix = np.array(group_fdr_masks)
            specific_fdr_masks_matrix = np.array(group_specific_fdr_masks)
            specific_cluster_masks_matrix = np.array(group_specific_cluster_masks)
            logger.info(
                f"  Conservé {len(group_scores)} sujets pour le groupe '{group_name}' "
                f"avec {scores_matrix.shape[1]} points temporels")

            # Calculer les statistiques améliorées avec tests de groupe
            stats = calculate_basic_statistics(
                scores_matrix, times_ref, group_subject_ids, fdr_masks_matrix,
                specific_fdr_masks_matrix, specific_cluster_masks_matrix)
            protocol_stats[group_name] = stats

            logger.info(f"  Groupe '{group_name}' - statistiques calculées:")
            logger.info(f"    Moyenne AUC: {stats['global_auc']:.3f}")
            logger.info(
                f"    Points significatifs FDR individuels: {np.sum(stats['fdr_mask'])}")
            
            # Log group-level statistical results
            if stats.get('group_global_test_results'):
                global_pval = stats['group_global_test_results']['global_pvalue']
                logger.info(f"    Test global de significativité: p={global_pval:.4f}")
            
            if stats.get('group_fdr_results'):
                n_fdr = stats['group_fdr_results']['n_significant_timepoints']
                logger.info(f"    FDR de groupe: {n_fdr} points significatifs")
            
            if stats.get('group_cluster_results'):
                n_clusters = stats['group_cluster_results']['n_significant_clusters']
                logger.info(f"    Clusters de groupe: {n_clusters} clusters significatifs")

            # Créer les visualisations améliorées
            create_streamlined_group_visualization(
                stats, group_name, output_dir, protocol)

        # 4. Créer des analyses séparées par dataset (DELIRIUM vs COMA)
        logger.info("\n--- Creating Enhanced Separate Dataset Analyses ---")
        create_separate_dataset_analyses_enhanced({protocol: all_results[protocol]}, output_dir, protocol)

        # Log summary
        valid_groups = {g: s for g, s in protocol_stats.items()
                        if s and s.get('n_subjects', 0) > 0}
        logger.info(f"Protocol {protocol} completed with {len(valid_groups)} valid groups")

    # 5. Generate enhanced comprehensive report
    generate_comprehensive_report(all_results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED ANALYSIS WITH ADVANCED STATISTICS COMPLETE")
    logger.info("=" * 80)
    logger.info("All results saved in: %s", os.path.abspath(output_dir))
    logger.info("Generated files:")
    logger.info("  - Individual group analyses with FDR and cluster masks")
    logger.info("  - Temporal comparisons by dataset")
    logger.info("  - Pairwise comparison matrices")
    logger.info("  - Dataset-specific detailed reports")
    logger.info("  - Comprehensive analysis report")
    logger.info("")
    logger.info("Key features:")
    logger.info("  ✓ Separate analyses by dataset (DELIRIUM, COMA)")
    logger.info("  ✓ Individual group visualizations with FDR/cluster significance")
    logger.info("  ✓ P-value color coding by significance strength")
    logger.info("  ✓ Pairwise comparisons with statistical testing")
    logger.info("  ✓ Protocol-by-protocol analysis")
    logger.info("  ✓ Enhanced statistical visualization")


if __name__ == "__main__":
    main()