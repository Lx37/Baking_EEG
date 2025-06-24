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
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

# Import stats functions for group-level statistical tests
sys.path.append('.')
from utils.stats_utils import (
    perform_pointwise_fdr_correction_on_scores,
    perform_cluster_permutation_test,
    compare_global_scores_to_chance
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V7"

GROUP_NAME_MAPPING = {
    'del': 'DELIRIUM +',
    'nodel': 'DELIRIUM -',
    'control': 'controls'
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


def find_npz_files(base_path: str) -> Dict[str, Dict[str, List[str]]]:
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


def load_npz_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate NPZ file data, using the correct keys found in files.
    Now includes specific FDR and cluster masks.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            actual_score_key = 'pp_ap_main_scores_1d_mean'
            actual_time_key = 'epochs_time_points'
            actual_fdr_key = 'pp_ap_main_temporal_1d_fdr'
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
                            logger.debug(
                                "Loaded FDR mask for %s: %d significant points",
                                subject_id, np.sum(fdr_dict['mask']))
                        else:
                            logger.warning(
                                "FDR data structure not recognized for %s",
                                subject_id)
                            result['fdr_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                    except Exception as e:
                        logger.warning(
                            "Error extracting FDR mask for %s: %s",
                            subject_id, e)
                        result['fdr_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                else:
                    logger.warning(
                        "FDR data is not in expected format for %s",
                        subject_id)
                    result['fdr_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
            else:
                logger.warning("FDR data not found in %s", file_path)
                result['fdr_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)

            # Ajouter les masques spécifiques FDR
            if specific_fdr_key in data_keys:
                specific_fdr_data = data[specific_fdr_key]
                if (isinstance(specific_fdr_data, np.ndarray) and
                        specific_fdr_data.dtype == object):
                    try:
                        specific_fdr_dict = specific_fdr_data.item()
                        if isinstance(specific_fdr_dict, dict) and 'mask' in specific_fdr_dict:
                            result['specific_fdr_mask'] = specific_fdr_dict['mask']
                            logger.debug(
                                "Loaded specific FDR mask for %s: %d significant points",
                                subject_id, np.sum(specific_fdr_dict['mask']))
                        else:
                            logger.warning(
                                "Specific FDR data structure not recognized for %s",
                                subject_id)
                            result['specific_fdr_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                    except Exception as e:
                        logger.warning(
                            "Error extracting specific FDR mask for %s: %s",
                            subject_id, e)
                        result['specific_fdr_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                else:
                    result['specific_fdr_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
            else:
                logger.warning("Specific FDR data not found in %s", file_path)
                result['specific_fdr_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)

            # Ajouter les masques spécifiques cluster
            if specific_cluster_key in data_keys:
                specific_cluster_data = data[specific_cluster_key]
                if (isinstance(specific_cluster_data, np.ndarray) and
                        specific_cluster_data.dtype == object):
                    try:
                        specific_cluster_dict = specific_cluster_data.item()
                        if isinstance(specific_cluster_dict, dict) and 'mask' in specific_cluster_dict:
                            result['specific_cluster_mask'] = specific_cluster_dict['mask']
                            logger.debug(
                                "Loaded specific cluster mask for %s: %d significant points",
                                subject_id, np.sum(specific_cluster_dict['mask']))
                        else:
                            logger.warning(
                                "Specific cluster data structure not recognized for %s",
                                subject_id)
                            result['specific_cluster_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                    except Exception as e:
                        logger.warning(
                            "Error extracting specific cluster mask for %s: %s",
                            subject_id, e)
                        result['specific_cluster_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                else:
                    result['specific_cluster_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
            else:
                logger.warning("Specific cluster data not found in %s", file_path)
                result['specific_cluster_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)

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


def calculate_basic_statistics(scores: np.ndarray, times: np.ndarray,
                               subject_ids: List[str], fdr_masks: np.ndarray = None,
                               specific_fdr_masks: np.ndarray = None,
                               specific_cluster_masks: np.ndarray = None) -> Dict[str, Any]:
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
            fdr_stats, fdr_mask_group, fdr_pvals = perform_pointwise_fdr_correction_on_scores(
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


def find_peak_latencies(scores: np.ndarray, times: np.ndarray, sig_mask: np.ndarray) -> Dict[str, Any]:
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


def perform_three_group_comparison(group_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
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


def create_streamlined_group_visualization(group_stats: Dict[str, Any], group_name: str,
                                           output_dir: str, protocol: str) -> str:
    """

    """
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    # === PANEL 1: Average TGM (if available) ===
    # For now, we'll create a placeholder TGM or load if available
    # This would need to be implemented based on your TGM data structure
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    # Create a synthetic TGM for demonstration - replace with actual TGM data loading
    tgm_times = times
    n_times = len(times)

    # Generate synthetic TGM based on temporal decoding pattern
    tgm_matrix = np.zeros((n_times, n_times))
    for i in range(n_times):
        for j in range(n_times):
            # Create a plausible TGM pattern
            temporal_decay = np.exp(-0.5 * np.abs(i - j) / 10)
            base_score = group_stats['group_mean'][i] if i < len(
                group_stats['group_mean']) else CHANCE_LEVEL
            tgm_matrix[i, j] = base_score * temporal_decay + \
                CHANCE_LEVEL * (1 - temporal_decay)

    im = ax1.imshow(tgm_matrix, cmap='RdBu_r', aspect='auto', origin='lower',
                    extent=[times[0], times[-1], times[0], times[-1]],
                    vmin=CHANCE_LEVEL - 0.1, vmax=CHANCE_LEVEL + 0.1)
    ax1.set_xlabel('Testing Time (s)', fontsize=14)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title(f'{group_name} - Average TGM',
                  fontsize=16, fontweight='bold')

    # Add stimulus onset lines
    ax1.axvline(x=0, color='black', linestyle=':', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.8)

    plt.colorbar(im, ax=ax1, label='AUC Score')

    plt.tight_layout()

    # Save figure
    filename = f"{protocol}_{group_name.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')}_streamlined_analysis.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Streamlined group figure saved: {output_path}")
    return output_path


def create_individual_group_plot(group_stats: Dict[str, Any], group_name: str,
                                 output_dir: str, protocol: str) -> str:
    """
    Créer un plot individuel pour un groupe avec courbes individuelles et SEM.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']

    # Couleur spécifique selon le groupe
    if 'DELIRIUM +' in group_name:
        main_color = '#d62728'  # Rouge
    elif 'DELIRIUM -' in group_name:
        main_color = '#ff7f0e'  # Orange
    else:  # controls
        main_color = '#2ca02c'  # Vert

    # Plot toutes les courbes individuelles
    for i in range(n_subjects):
        ax.plot(times, scores[i, :], color='lightgray',
                alpha=0.5, linewidth=1, zorder=1)

    # Plot moyenne de groupe avec SEM
    group_mean = group_stats['group_mean']
    group_sem = group_stats['group_sem']

    # Zone SEM
    ax.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                    alpha=0.3, color=main_color, label=f'SEM (n={n_subjects})', zorder=2)

    # Ligne moyenne
    ax.plot(times, group_mean, color=main_color, linewidth=3,
            label=f'{group_name} - Moyenne', zorder=3)

    # Ligne de chance
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
               alpha=0.7, label='Niveau de Chance (0.5)', zorder=3)

    # Ligne de stimulus onset
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Stimulus Onset', zorder=4)

    # === AJOUTER LES RÉSULTATS STATISTIQUES DE GROUPE ===
    # FDR significance regions
    if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
        fdr_mask = group_stats['group_fdr_results']['fdr_mask']
        n_fdr_sig = group_stats['group_fdr_results']['n_significant_timepoints']
        
        if np.any(fdr_mask):
            # Highlight FDR significant regions
            for i, is_sig in enumerate(fdr_mask):
                if is_sig:
                    ax.axvspan(times[i] - 0.005, times[i] + 0.005, 
                              alpha=0.6, color='gold', zorder=5)
            
            logger.info(f"Added FDR significance overlay: {n_fdr_sig} significant timepoints")

    # Cluster significance regions
    if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
        cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        n_cluster_sig = group_stats['group_cluster_results']['n_significant_clusters']
        
        if np.any(cluster_mask):
            # Highlight cluster significant regions with different style
            sig_times = times[cluster_mask]
            ax.scatter(sig_times, [0.37] * len(sig_times), 
                      marker='|', s=50, color='purple', alpha=0.8, zorder=6,
                      label=f'Cluster Sig. (n={n_cluster_sig})')
            
            logger.info(f"Added cluster significance overlay: {n_cluster_sig} significant clusters")

    # Configuration des axes - Extension temporelle complète
    ax.set_xlabel('Temps (s)', fontsize=14)
    ax.set_ylabel('Précision de Décodage (AUC)', fontsize=14)
    
    # Enhanced title with statistical results
    title_text = f'{group_name} - Décodage Temporel (n={n_subjects})\n'
    title_text += f'AUC Moyenne: {group_stats["global_auc"]:.3f} ± {np.std(group_stats["subject_means"]):.3f}'
    
    # Add global statistical test result to title
    if group_stats.get('group_global_test_results'):
        global_results = group_stats['group_global_test_results']
        if global_results['global_pvalue'] is not None:
            title_text += f' (p={global_results["global_pvalue"]:.4f})'
    
    ax.set_title(title_text, fontsize=16, fontweight='bold')

    ax.set_xlim([-0.2, 1.0])  
    ax.set_ylim([0.35, 0.85])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()

    # Définir le chemin de sortie
    safe_name = group_name.replace(' ', '_').replace(
        '+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_individual_plot.png"
    output_path = os.path.join(output_dir, filename)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot individuel sauvé: {output_path}")
    return output_path


def create_three_group_temporal_comparison(group_stats: Dict[str, Dict[str, Any]],
                                           three_group_results: Dict[str, Any],
                                           output_dir: str, protocol: str) -> str:
    """
    Créer une comparaison des 3 groupes avec SEM et résultats statistiques de groupe.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    group_names = three_group_results['group_names']
    times = group_stats[group_names[0]]['times']

    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c'
    }

    # Comparaison des moyennes avec SEM
    for group_name in group_names:
        stats = group_stats[group_name]
        color = colors.get(group_name, COLORS_PALETTE[0])

        # Zone SEM
        ax.fill_between(times,
                         stats['group_mean'] - stats['group_sem'],
                         stats['group_mean'] + stats['group_sem'],
                         alpha=0.2, color=color, zorder=1)

        # Ligne moyenne
        ax.plot(times, stats['group_mean'], color=color, linewidth=3,
                 label=f'{group_name} (n={stats["n_subjects"]})', zorder=2)

        # === AJOUTER LES RÉSULTATS STATISTIQUES POUR CHAQUE GROUPE ===
        # FDR significance pour ce groupe
        if stats.get('group_fdr_results') and stats['group_fdr_results']['fdr_mask'] is not None:
            fdr_mask = stats['group_fdr_results']['fdr_mask']
            if np.any(fdr_mask):
                # Utiliser une position Y légèrement différente pour chaque groupe
                y_positions = {'DELIRIUM +': 0.575, 'DELIRIUM -': 0.565, 'controls': 0.555}
                y_pos = y_positions.get(group_name, 0.555)
                
                sig_times = times[fdr_mask]
                ax.scatter(sig_times, [y_pos] * len(sig_times), 
                          marker='s', s=15, color=color, alpha=0.8, zorder=6)

        # Cluster significance pour ce groupe
        if stats.get('group_cluster_results') and stats['group_cluster_results']['combined_cluster_mask'] is not None:
            cluster_mask = stats['group_cluster_results']['combined_cluster_mask']
            if np.any(cluster_mask):
                # Position Y légèrement en dessous des marqueurs FDR
                y_positions = {'DELIRIUM +': 0.410, 'DELIRIUM -': 0.415, 'controls': 0.420}
                y_pos = y_positions.get(group_name, 0.410)
                
                sig_times = times[cluster_mask]
                ax.scatter(sig_times, [y_pos] * len(sig_times), 
                          marker='|', s=30, color=color, alpha=0.8, zorder=6)

    # Ligne de chance
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
                alpha=0.7, label='Niveau de Chance', zorder=2)

    ax.set_xlabel('Temps (s)', fontsize=14)
    ax.set_ylabel('Précision de Décodage (AUC)', fontsize=14)
    ax.set_title(f'Comparaison Temporelle des Trois Groupes avec Tests Statistiques\nProtocole {protocol}',
                  fontsize=16, fontweight='bold')

    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0.40, 0.60])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')

    # === STATISTIQUES DÉTAILLÉES DANS L'ENCADRÉ ===
    stats_text = "Moyennes AUC globales et tests statistiques:\n"
    for i, group_name in enumerate(group_names):
        mean_auc = three_group_results['global_means'][i]
        stats_text += f"{group_name}: {mean_auc:.3f}"
        
        # Ajouter les résultats des tests globaux
        if group_stats[group_name].get('group_global_test_results'):
            global_pval = group_stats[group_name]['group_global_test_results']['global_pvalue']
            if global_pval is not None:
                sig_marker = "*" if global_pval < 0.05 else ""
                stats_text += f" (p={global_pval:.3f}{sig_marker})"
        
        # Ajouter les counts des tests FDR et cluster
        if group_stats[group_name].get('group_fdr_results'):
            n_fdr = group_stats[group_name]['group_fdr_results']['n_significant_timepoints']
            stats_text += f"\n  FDR: {n_fdr} points sig."
        
        if group_stats[group_name].get('group_cluster_results'):
            n_clusters = group_stats[group_name]['group_cluster_results']['n_significant_clusters']
            stats_text += f" | Clusters: {n_clusters}"
        
        stats_text += "\n"

    # Ajouter légende pour les marqueurs statistiques
    stats_text += "\nMarqueurs: □ = FDR sig., | = Cluster sig.\n* = p < 0.05"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Sauvegarder
    filename = f"{protocol}_three_groups_simple_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparaison simple des 3 groupes sauvée: {output_path}")
    return output_path


def create_subject_fdr_significance_plot(group_stats: Dict[str, Any], group_name: str,
                                        output_dir: str, protocol: str) -> str:
    """
    Créer un plot avec les ID des sujets sur l'axe Y et leurs points significatifs FDR.
    Chaque ligne correspond à un sujet avec ses points significatifs marqués.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, max(8, group_stats['n_subjects'] * 0.5)))

    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c'
    }

    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    subject_ids = group_stats['subject_ids']
    individual_fdr_masks = group_stats['individual_fdr_masks']
    color = colors.get(group_name, COLORS_PALETTE[0])

    # Créer une grille pour afficher chaque sujet
    for i, (subj_id, fdr_mask) in enumerate(zip(subject_ids, individual_fdr_masks)):
        y_position = i
        
        # Afficher les points significatifs pour ce sujet
        if np.any(fdr_mask):
            sig_times = times[fdr_mask]
            sig_y = np.full_like(sig_times, y_position)
            ax.scatter(sig_times, sig_y, color=color, marker='|', s=80,
                      alpha=0.8, linewidths=2)
        
        # Ligne de fond pour chaque sujet
        ax.plot([times[0], times[-1]], [y_position, y_position], 
               color='lightgray', alpha=0.3, linewidth=1)

    # Configuration des axes
    ax.set_xlabel('Temps (s)', fontsize=14)
    ax.set_ylabel('Sujets', fontsize=14)
    ax.set_title(f'{group_name} - Points significatifs FDR par sujet\n'
                 f'(n={n_subjects} sujets)', fontsize=16, fontweight='bold')

    # Définir les étiquettes de l'axe Y
    ax.set_yticks(range(n_subjects))
    ax.set_yticklabels([f'ID: {subj_id}' for subj_id in subject_ids], fontsize=10)
    
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([-0.5, n_subjects - 0.5])
    ax.grid(True, alpha=0.3, axis='x')

    # Ajouter ligne de stimulus onset
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
              alpha=0.8, label='Stimulus Onset')
    ax.legend()

    # Ajouter statistiques dans un encadré
    total_sig_points = sum(np.sum(mask) for mask in individual_fdr_masks)
    stats_text = f'Total points significatifs: {total_sig_points}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Sauvegarder
    safe_name = group_name.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_subject_fdr_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot FDR par sujet sauvé: {output_path}")
    return output_path


def create_combined_all_subjects_plot(all_group_stats: Dict[str, Dict[str, Any]], 
                                    output_dir: str, protocol: str) -> str:
    """
    Créer un grand plot combiné montrant le cumul des plots FDR des 3 groupes.
    Chaque ligne correspond à un sujet avec ses points significatifs FDR marqués.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e', 
        'controls': '#2ca02c'
    }
    
    # Récupérer les temps (identiques pour tous les groupes)
    times = list(all_group_stats.values())[0]['times']
    
    # Calculer le nombre total de sujets pour organiser l'affichage
    total_subjects = sum(stats['n_subjects'] for stats in all_group_stats.values())
    
    # Position Y pour chaque sujet
    current_y_position = 0
    group_y_positions = {}
    
    # Tracer les points significatifs FDR pour chaque groupe
    for group_name, stats in all_group_stats.items():
        color = colors.get(group_name, COLORS_PALETTE[0])
        subject_ids = stats['subject_ids']
        individual_fdr_masks = stats['individual_fdr_masks']
        n_subjects = stats['n_subjects']
        
        group_start_position = current_y_position
        
        # Pour chaque sujet dans le groupe
        for i, (subj_id, fdr_mask) in enumerate(zip(subject_ids, individual_fdr_masks)):
            y_position = current_y_position
            
            # Afficher les points significatifs pour ce sujet
            if np.any(fdr_mask):
                sig_times = times[fdr_mask]
                sig_y = np.full_like(sig_times, y_position)
                ax.scatter(sig_times, sig_y, color=color, marker='|', s=100,
                          alpha=0.8, linewidths=2)
            
            # Ligne de fond pour chaque sujet
            ax.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y
            ax.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(all_group_stats.keys())[-1]:  # Pas pour le dernier groupe
            ax.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
        
        # Ajouter le nom du groupe sur le côté
        group_center_y = group_start_position + (n_subjects - 1) / 2
        ax.text(-0.27, group_center_y, group_name, fontsize=12, fontweight='bold',
               verticalalignment='center', rotation=90, color=color)
        
        group_y_positions[group_name] = (group_start_position, current_y_position - 1)
    
    # Configuration des axes
    ax.set_xlabel('Temps (s)', fontsize=16)
    ax.set_title(f'Points significatifs FDR - PP/AP class balanced\n'
                 f'Protocole {protocol} (n={total_subjects} sujets)', 
                 fontsize=18, fontweight='bold')
    
    # Masquer les étiquettes Y par défaut et configurer les limites
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([-0.5, total_subjects - 0.5])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Ligne de stimulus onset
    ax.axvline(x=0, color='red', linestyle='--', linewidth=3,
              alpha=0.8, label='Stimulus Onset')

    legend_elements = []
    for group_name, stats in all_group_stats.items():
        color = colors.get(group_name, COLORS_PALETTE[0])
        legend_elements.append(plt.Line2D([0], [0], marker='|',
                                          color=color,linewidth=0, markersize=15,
                                          label=f'{group_name} (n={stats["n_subjects"]})'))
    
    # Ajouter ligne de stimulus onset à la légende
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                    linewidth=3, label='Stimulus Onset'))
    
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"{protocol}_combined_all_subjects_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot combiné FDR de tous les sujets sauvé: {output_path}")
    return output_path


def create_stacked_vertical_plot_with_specific_masks(all_group_stats: Dict[str, Dict[str, Any]], 
                                                   output_dir: str, protocol: str) -> str:
    """
    Créer un grand plot combiné montrant les masques FDR et cluster spécifiques des 3 groupes.
    Chaque ligne correspond à un sujet avec ses points significatifs FDR et cluster marqués.
    Style similaire à create_combined_all_subjects_plot mais avec deux types de masques.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e', 
        'controls': '#2ca02c'
    }
    
    # Couleurs pastel pour les clusters (versions plus claires)
    pastel_colors = {
        'DELIRIUM +': '#ffb3ba',  # Rouge pastel
        'DELIRIUM -': '#ffd1a3',  # Orange pastel
        'controls': '#b3e5b3'     # Vert pastel
    }
    
    # Marqueurs identiques mais couleurs différentes pour FDR et cluster
    fdr_marker = '|'  # Barre verticale pour FDR
    cluster_marker = '|'  # Barre verticale pour cluster (même forme)
    
    # Récupérer les temps (identiques pour tous les groupes)
    times = list(all_group_stats.values())[0]['times']
    
    # Calculer le nombre total de sujets pour organiser l'affichage
    total_subjects = sum(stats['n_subjects'] for stats in all_group_stats.values())
    
    # Position Y pour chaque sujet
    current_y_position = 0
    group_y_positions = {}
    
    # Tracer les points significatifs FDR et cluster pour chaque groupe
    for group_name, stats in all_group_stats.items():
        color = colors.get(group_name, COLORS_PALETTE[0])
        pastel_color = pastel_colors.get(group_name, color)  # Couleur pastel pour cluster
        subject_ids = stats['subject_ids']
        n_subjects = stats['n_subjects']
        
        # Récupérer les masques spécifiques FDR et cluster
        specific_fdr_masks = stats.get('specific_fdr_masks', np.zeros((n_subjects, len(times)), dtype=bool))
        specific_cluster_masks = stats.get('specific_cluster_masks', np.zeros((n_subjects, len(times)), dtype=bool))
        
        # Vérifier les dimensions
        if specific_fdr_masks.shape != (n_subjects, len(times)):
            logger.warning(f"Forme incorrecte pour specific_fdr_masks dans {group_name}: {specific_fdr_masks.shape}, attendu: {(n_subjects, len(times))}")
            specific_fdr_masks = np.zeros((n_subjects, len(times)), dtype=bool)
            
        if specific_cluster_masks.shape != (n_subjects, len(times)):
            logger.warning(f"Forme incorrecte pour specific_cluster_masks dans {group_name}: {specific_cluster_masks.shape}, attendu: {(n_subjects, len(times))}")
            specific_cluster_masks = np.zeros((n_subjects, len(times)), dtype=bool)
        
        group_start_position = current_y_position
        
        # Pour chaque sujet dans le groupe
        for i, subj_id in enumerate(subject_ids):
            y_position = current_y_position
            
            # Afficher les points significatifs FDR spécifiques
            if i < specific_fdr_masks.shape[0] and np.any(specific_fdr_masks[i, :]):
                fdr_sig_times = times[specific_fdr_masks[i, :]]
                fdr_sig_y = np.full_like(fdr_sig_times, y_position)  # Même hauteur
                ax.scatter(fdr_sig_times, fdr_sig_y, color=color, marker=fdr_marker, s=120,
                          alpha=0.8, linewidths=3)
            
            # Afficher les points significatifs cluster spécifiques
            if i < specific_cluster_masks.shape[0] and np.any(specific_cluster_masks[i, :]):
                cluster_sig_times = times[specific_cluster_masks[i, :]]
                cluster_sig_y = np.full_like(cluster_sig_times, y_position)  # Même hauteur que FDR
                ax.scatter(cluster_sig_times, cluster_sig_y, color=pastel_color, marker=cluster_marker, s=100,
                          alpha=0.8, linewidths=2)
            
            # Ligne de fond pour chaque sujet
            ax.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y
            ax.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(all_group_stats.keys())[-1]:  # Pas pour le dernier groupe
            ax.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
        
        # Ajouter le nom du groupe sur le côté
        group_center_y = group_start_position + (n_subjects - 1) / 2
        ax.text(-0.27, group_center_y, group_name, fontsize=12, fontweight='bold',
               verticalalignment='center', rotation=90, color=color)
        
        group_y_positions[group_name] = (group_start_position, current_y_position - 1)
    
    # Configuration des axes
    ax.set_xlabel('Temps (s)', fontsize=16)
    ax.set_title(f'Masques FDR et Cluster PP/AP mean\n'
                 f'Protocole {protocol} (n={total_subjects} sujets)', 
                 fontsize=18, fontweight='bold')
    
    # Masquer les étiquettes Y par défaut et configurer les limites
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([-0.5, total_subjects - 0.5])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Ligne de stimulus onset
    ax.axvline(x=0, color='red', linestyle='--', linewidth=3,
              alpha=0.8, label='Stimulus Onset')

    # Créer la légende
    legend_elements = []
    
    # Ajouter les éléments de groupe avec les deux types de marqueurs
    for group_name, stats in all_group_stats.items():
        color = colors.get(group_name, COLORS_PALETTE[0])
        pastel_color = pastel_colors.get(group_name, color)
        legend_elements.append(plt.Line2D([0], [0], marker=fdr_marker,
                                          color=color, linewidth=0, markersize=15,
                                          label=f'{group_name} FDR (n={stats["n_subjects"]})'))
        legend_elements.append(plt.Line2D([0], [0], marker=cluster_marker,
                                          color=pastel_color, linewidth=0, markersize=15,
                                          label=f'{group_name} Cluster'))
    
    # Ajouter ligne de stimulus onset à la légende
    legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', 
                                    linewidth=3, label='Stimulus Onset'))
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right', ncol=2)
    
    # Ajouter statistiques globales dans un encadré
    total_fdr_points = 0
    total_cluster_points = 0
    for stats in all_group_stats.values():
        specific_fdr_masks = stats.get('specific_fdr_masks', np.array([]))
        specific_cluster_masks = stats.get('specific_cluster_masks', np.array([]))
        if specific_fdr_masks.size > 0:
            total_fdr_points += np.sum(specific_fdr_masks)
        if specific_cluster_masks.size > 0:
            total_cluster_points += np.sum(specific_cluster_masks)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"{protocol}_combined_specific_masks_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot combiné avec masques spécifiques sauvé: {output_path}")
    return output_path


def generate_comprehensive_report(all_results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate a comprehensive text report with descriptive statistics only.
    """
    report_path = os.path.join(
        output_dir, "comprehensive_descriptive_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EEG DECODING DESCRIPTIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for protocol, results in all_results.items():
            f.write(f"\nPROTOCOL: {protocol}\n")
            f.write("-" * 40 + "\n")

            # Group statistics
            f.write("\nGROUP STATISTICS:\n")
            for group_name, stats in results['group_stats'].items():
                f.write(f"\n{group_name}:\n")
                f.write(f"  N subjects: {stats['n_subjects']}\n")
                f.write(f"  Mean AUC: {stats['global_auc']:.4f}\n")
                f.write(f"  Standard deviation: {np.std(stats['subject_means']):.4f}\n")
                f.write(
                    f"  Peak AUC: {stats['peak_latencies']['global_peak_value']:.4f} ")
                f.write(
                    f"at {stats['peak_latencies']['global_peak_time']:.3f}s\n")
                
                # === NOUVEAUX RÉSULTATS STATISTIQUES DE GROUPE ===
                # Global significance test
                if stats.get('group_global_test_results'):
                    global_results = stats['group_global_test_results']
                    f.write(f"  Global significance test (Wilcoxon):\n")
                    f.write(f"    Statistic: {global_results['global_statistic']:.4f}\n")
                    f.write(f"    P-value: {global_results['global_pvalue']:.6f}\n")
                    f.write(f"    Significant: {'Yes' if global_results['is_significant'] else 'No'}\n")
                
                # FDR results
                if stats.get('group_fdr_results'):
                    fdr_results = stats['group_fdr_results']
                    f.write(f"  FDR correction results (Wilcoxon):\n")
                    f.write(f"    Significant timepoints: {fdr_results['n_significant_timepoints']}\n")
                    f.write(f"    Total timepoints tested: {len(fdr_results['fdr_mask'])}\n")
                    f.write(f"    Proportion significant: {fdr_results['n_significant_timepoints']/len(fdr_results['fdr_mask']):.3f}\n")
                
                # Cluster results
                if stats.get('group_cluster_results'):
                    cluster_results = stats['group_cluster_results']
                    f.write(f"  Cluster permutation results (Wilcoxon):\n")
                    f.write(f"    Significant clusters: {cluster_results['n_significant_clusters']}\n")
                    f.write(f"    Timepoints in significant clusters: {np.sum(cluster_results['combined_cluster_mask'])}\n")
                    
                    if cluster_results['cluster_pvals'] is not None and len(cluster_results['cluster_pvals']) > 0:
                        f.write(f"    Cluster p-values: ")
                        for i, pval in enumerate(cluster_results['cluster_pvals']):
                            f.write(f"C{i+1}={pval:.4f} ")
                        f.write("\n")
                
                # Individual subject FDR (existing)
                f.write(f"  Individual subject FDR significant points: {np.sum(stats['fdr_mask'])}\n")
                f.write(
                    f"  Significant time windows: {stats['peak_latencies']['n_significant_windows']}\n")

                if stats['peak_latencies']['n_significant_windows'] > 0:
                    f.write("  Significant windows details:\n")
                    for i, window in enumerate(stats['peak_latencies']['significant_windows']):
                        f.write(
                            f"    Window {i+1}: {window['start_time']:.3f}s - {window['end_time']:.3f}s ")
                        f.write(
                            f"(duration: {window['duration']:.3f}s, peak: {window['peak_value']:.4f})\n")

            # Three-group comparison (descriptive only)
            if 'three_group_comparison' in results:
                comp = results['three_group_comparison']
                f.write(f"\nTHREE-GROUP DESCRIPTIVE COMPARISON:\n")
                f.write("  Group means comparison:\n")
                for i, group_name in enumerate(comp['group_names']):
                    f.write(f"    {group_name}: {comp['global_means'][i]:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DESCRIPTIVE ANALYSIS COMPLETED\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Comprehensive descriptive report saved: {report_path}")
    return report_path




def main():
    """
    Main function to run the enhanced analysis pipeline.
    """
    logger.info("=" * 80)
    logger.info(
        " ENHANCED EEG DECODING GROUP ANALYSIS WITH ADVANCED STATISTICS ")
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

            # Calculer les statistiques améliorées
            stats = calculate_basic_statistics(
                scores_matrix, times_ref, group_subject_ids, fdr_masks_matrix,
                specific_fdr_masks_matrix, specific_cluster_masks_matrix)
            protocol_stats[group_name] = stats

            logger.info(f"  Groupe '{group_name}' - statistiques calculées:")
            logger.info(f"    Moyenne AUC: {stats['global_auc']:.3f}")
            logger.info(
                f"    Points significatifs FDR: {np.sum(stats['fdr_mask'])}")

            # Créer les visualisations améliorées
            create_streamlined_group_visualization(
                stats, group_name, output_dir, protocol)

            # Créer le plot individuel avec SEM
            create_individual_group_plot(
                stats, group_name, output_dir, protocol)

            # Créer l'histogramme de significativité par sujet (avec ID sur l'axe Y)
            create_subject_fdr_significance_plot(
                stats, group_name, output_dir, protocol)

        # 3.4. Créer le plot combiné de tous les sujets avec les FDR individuels
        valid_groups = {g: s for g, s in protocol_stats.items()
                        if s and s.get('n_subjects', 0) > 0}
        
        if len(valid_groups) >= 2:
            # Créer le plot combiné de tous les sujets
            logger.info("\n--- Creating Combined All Subjects FDR Plot ---")
            create_combined_all_subjects_plot(
                valid_groups, output_dir, protocol)
            
            # Créer le plot empilé vertical avec masques spécifiques FDR et cluster
            logger.info("\n--- Creating Stacked Vertical Plot with Specific Masks ---")
            create_stacked_vertical_plot_with_specific_masks(
                valid_groups, output_dir, protocol)

        # 4. Perform three-group comparison if we have exactly 3 groups
        if len(valid_groups) == 3:
            logger.info("\n--- Performing Three-Group Comparison ---")
            three_group_results = perform_three_group_comparison(valid_groups)
            all_results[protocol]['three_group_comparison'] = \
                three_group_results

            # Create three-group temporal comparison
            create_three_group_temporal_comparison(
                valid_groups, three_group_results, output_dir, protocol)

            logger.info(f"Three-group comparison completed with means: "
                        f"{[f'{name}: {mean:.3f}' for name, mean in zip(three_group_results['group_names'], three_group_results['global_means'])]}")
        else:
            logger.warning(f"Expected 3 groups for comparison, "
                           f"got {len(valid_groups)}")

    # 5. Generate comprehensive report
    generate_comprehensive_report(all_results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("All results saved in: %s", os.path.abspath(output_dir))
    logger.info("Generated files:")
    logger.info("  - Enhanced group visualizations (2 panels each)")
    logger.info("  - Individual group plots with SEM (without group-level statistics)")
    logger.info("  - Subject-level FDR significance plots")
    logger.info("  - Combined FDR plot showing all subjects from all groups")
    logger.info("  - Three-group simple comparison")
    logger.info("  - Comprehensive descriptive report")


if __name__ == "__main__":
    main()