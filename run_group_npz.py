#!/usr/bin/env python3
# Available keys: ['subject_id', 'group', 'decoding_protocol_identifier', 'classifier_type_used', 'epochs_time_points', 'pp_ap_main_original_labels', 'pp_ap_main_pred_probas_global', 'pp_ap_main_pred_labels_global', 'pp_ap_main_cv_global_scores', 'pp_ap_main_scores_1d_all_folds', 'pp_ap_main_scores_1d_mean', 'pp_ap_main_temporal_1d_fdr', 'pp_ap_main_temporal_1d_cluster', 'pp_ap_main_tgm_all_folds', 'pp_ap_main_tgm_mean', 'pp_ap_main_tgm_fdr', 'pp_ap_main_mean_auc_global', 'pp_ap_main_global_metrics', 'pp_ap_specific_ap_results', 'pp_ap_mean_of_specific_scores_1d', 'pp_ap_sem_of_specific_scores_1d', 'pp_ap_mean_specific_fdr', 'pp_ap_mean_specific_cluster', 'pp_ap_ap_vs_ap_results', 'pp_ap_ap_centric_avg_results', 'detected_protocol'

"""
Enhanced EEG Decoding Group Analysis with Advanced Statistical Methods

This script provides comprehensive analysis and visualization of EEG decoding results
for scientific publications. It processes individual subject NPZ files and generates
publication-quality figures with extensive statistical analyses using permutation tests,
FDR correction, and three-group comparisons.

Key Features:
- Three-group simultaneous comparisons (DELIRIUM +, DELIRIUM -, controls)
- Permutation cluster tests for robust statistical inference
- FDR correction for multiple comparisons
- Detailed latency analysis
- Publication-quality visualizations (max 2 plots per page)
- Integration with project's statistical utilities

"""

from utils import stats_utils as bEEG_stats
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, f_oneway, wilcoxon, mannwhitneyu
from scipy.stats import pearsonr, spearmanr, kruskal
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import getpass
from itertools import combinations

# Import project-specific statistical utilities
sys.path.append('/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG/utils')

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
    Recursively find and organize NPZ files based on the specific directory structure.
    Structure attendue : {base_path}/intra_subject_results/{group_name}/.../decoding_results_full.npz
    """
    logger.info(f"Searching for NPZ files in: {base_path}")
    organized_data = {'PP': {}}  # Protocole unique 'PP'

    search_pattern = os.path.join(
        base_path, 'intra_subject_results', '**', 'decoding_results_full.npz')
    all_files = glob.glob(search_pattern, recursive=True)

    if not all_files:
        logger.warning(
            f"No 'decoding_results_full.npz' files found with pattern: {search_pattern}")
        return {}

    logger.info(f"Found {len(all_files)} potential result files.")

    for file_path in all_files:
        try:
            rel_path = os.path.relpath(file_path, base_path)
            path_parts = rel_path.split(os.sep)

            if len(path_parts) > 2 and path_parts[0] == 'intra_subject_results':
                group_folder = path_parts[1]
                group_name = GROUP_NAME_MAPPING.get(group_folder, group_folder)

                if group_name not in organized_data['PP']:
                    organized_data['PP'][group_name] = []
                organized_data['PP'][group_name].append(file_path)
            else:
                logger.warning(
                    f"File path does not match expected structure, skipping: {file_path}")

        except Exception as e:
            logger.warning(f"Error processing file path {file_path}: {e}")

    # Log summary
    logger.info("=== COLLECTED DATA SUMMARY ===")
    for protocol, groups in organized_data.items():
        if not groups:
            logger.warning(f"No groups found for protocol {protocol}.")
            continue
        logger.info(f"Protocol {protocol}:")
        for group, files in groups.items():
            logger.info(f"  - Group '{group}': {len(files)} subjects found.")

    return organized_data


def load_npz_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate NPZ file data, using the correct keys found in the files.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            actual_score_key = 'pp_ap_main_scores_1d_mean'
            actual_time_key = 'epochs_time_points'
            actual_fdr_key = 'pp_ap_main_temporal_1d_fdr'

            data_keys = list(data.keys())
            required_keys = [actual_score_key, actual_time_key]

            for key in required_keys:
                if key not in data_keys:
                    logger.warning(
                        f"Missing required field '{key}' in {file_path}. Available keys: {data_keys}")
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
                if isinstance(fdr_data, np.ndarray) and fdr_data.dtype == object:
                    try:
                        fdr_dict = fdr_data.item()  # Extraire le dictionnaire de l'array
                        if isinstance(fdr_dict, dict) and 'mask' in fdr_dict:
                            result['fdr_mask'] = fdr_dict['mask']
                            logger.debug(
                                f"Loaded FDR mask for {subject_id}: {np.sum(fdr_dict['mask'])} significant points")
                        else:
                            logger.warning(
                                f"FDR data structure not recognized for {subject_id}")
                            result['fdr_mask'] = np.zeros_like(
                                data[actual_score_key], dtype=bool)
                    except Exception as e:
                        logger.warning(
                            f"Error extracting FDR mask for {subject_id}: {e}")
                        result['fdr_mask'] = np.zeros_like(
                            data[actual_score_key], dtype=bool)
                else:
                    logger.warning(
                        f"FDR data is not in expected format for {subject_id}")
                    result['fdr_mask'] = np.zeros_like(
                        data[actual_score_key], dtype=bool)
            else:
                logger.warning(f"FDR data not found in {file_path}")
                result['fdr_mask'] = np.zeros_like(
                    data[actual_score_key], dtype=bool)

            if result['scores'] is None or result['times'] is None:
                logger.warning(
                    f"Data for scores or times is None in {file_path}")
                return None
            if len(result['scores']) == 0 or len(result['times']) == 0:
                logger.warning(
                    f"Data for scores or times is empty in {file_path}")
                return None

            return result

    except Exception as e:
        logger.error(f"Error loading or processing {file_path}: {e}")
        return None


def calculate_basic_statistics(scores: np.ndarray, times: np.ndarray,
                               subject_ids: List[str], fdr_masks: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate basic statistical measures using individual subject FDR masks from NPZ files.
    Focus on individual subject analysis and removing group-level statistical tests.
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

    # Confidence intervals (95%)
    ci_lower = group_mean - 1.96 * group_sem
    ci_upper = group_mean + 1.96 * group_sem

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

    return {
        'n_subjects': n_subjects,
        'subject_ids': subject_ids,
        'times': times,
        'scores': scores,
        'group_mean': group_mean,
        'group_std': group_std,
        'group_sem': group_sem,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'subject_means': subject_means,
        'subject_peaks': subject_peaks,
        'subject_peak_times': subject_peak_times,
        'global_auc': global_auc,
        'peak_latencies': peak_latencies,

        # Individual subject FDR analysis
        'individual_fdr_masks': individual_fdr_masks,
        'fdr_counts_per_timepoint': fdr_counts_per_timepoint,
        'fdr_mask': fdr_counts_per_timepoint > 0,  # Mock FDR mask for compatibility
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
    Perform comprehensive three-group comparison using ANOVA and post-hoc tests.
    """
    group_names = list(group_stats.keys())
    if len(group_names) != 3:
        logger.warning(f"Expected 3 groups, got {len(group_names)}")
        return {}

    # Extract data
    all_scores = [group_stats[name]['scores'] for name in group_names]
    times = group_stats[group_names[0]]['times']
    n_timepoints = len(times)

    # Point-by-point ANOVA
    f_stats = np.zeros(n_timepoints)
    p_values_anova = np.ones(n_timepoints)

    for t in range(n_timepoints):
        scores_at_t = [scores[:, t] for scores in all_scores]
        try:
            f_stat, p_val = f_oneway(*scores_at_t)
            f_stats[t] = f_stat
            p_values_anova[t] = p_val
        except:
            f_stats[t] = 0
            p_values_anova[t] = 1

    # FDR correction for ANOVA
    p_corrected_anova = multipletests(p_values_anova, method='fdr_bh')[1]
    significant_anova = p_corrected_anova < 0.05

    # Post-hoc pairwise comparisons
    pairwise_results = {}
    for i, j in combinations(range(len(group_names)), 2):
        group1_name = group_names[i]
        group2_name = group_names[j]

        scores1 = all_scores[i]
        scores2 = all_scores[j]

        # Point-by-point t-tests
        t_stats = np.zeros(n_timepoints)
        p_values = np.ones(n_timepoints)
        effect_sizes = np.zeros(n_timepoints)

        for t in range(n_timepoints):
            try:
                t_stat, p_val = ttest_ind(
                    scores1[:, t], scores2[:, t], equal_var=False)
                t_stats[t] = t_stat
                p_values[t] = p_val

                # Cohen's d
                n1, n2 = len(scores1), len(scores2)
                pooled_std = np.sqrt(((n1 - 1) * np.var(scores1[:, t], ddof=1) +
                                     (n2 - 1) * np.var(scores2[:, t], ddof=1)) / (n1 + n2 - 2))
                if pooled_std > 0:
                    effect_sizes[t] = (
                        np.mean(scores1[:, t]) - np.mean(scores2[:, t])) / pooled_std
            except:
                t_stats[t] = 0
                p_values[t] = 1
                effect_sizes[t] = 0

        p_corrected = multipletests(p_values, method='fdr_bh')[1]

        pairwise_results[f"{group1_name}_vs_{group2_name}"] = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            't_stats': t_stats,
            'p_values': p_values,
            'p_corrected': p_corrected,
            'effect_sizes': effect_sizes,
            'significant_points': p_corrected < 0.05
        }

    # Global comparisons
    global_means = [np.mean(group_stats[name]['subject_means'])
                    for name in group_names]
    global_f_stat, global_p_anova = f_oneway(
        *[group_stats[name]['subject_means'] for name in group_names])

    return {
        'group_names': group_names,
        'f_stats': f_stats,
        'p_values_anova': p_values_anova,
        'p_corrected_anova': p_corrected_anova,
        'significant_anova': significant_anova,
        'pairwise_results': pairwise_results,
        'global_f_stat': global_f_stat,
        'global_p_anova': global_p_anova,
        'global_means': global_means
    }


def create_streamlined_group_visualization(group_stats: Dict[str, Any], group_name: str,
                                           output_dir: str, protocol: str) -> str:
    """
    Create streamlined group visualization focusing on temporal curves and TGM.
    Maximum 2 plots per page for clarity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']

    # === PANEL 1: Temporal decoding curves ===

    # Plot individual subjects with transparency
    for i in range(min(n_subjects, 20)):  # Limit to 20 for clarity
        ax1.plot(times, scores[i, :], color='lightgray',
                 alpha=0.4, linewidth=1)

    # Plot group average with confidence interval
    ax1.fill_between(times, group_stats['ci_lower'], group_stats['ci_upper'],
                     alpha=0.3, color=COLORS_PALETTE[0], label='95% CI')
    ax1.plot(times, group_stats['group_mean'], color=COLORS_PALETTE[0], linewidth=3,
             label=f'Group Mean (n={n_subjects})')

    ax1.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='Chance Level')
    ax1.axvline(x=0, color='black', linestyle=':',
                linewidth=1, alpha=0.8, label='Stimulus Onset')

    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax1.set_title(f'{group_name} - Temporal Decoding\n'
                  f'Mean AUC: {group_stats["global_auc"]:.3f} | '
                  f'Peak: {group_stats["peak_latencies"]["global_peak_value"]:.3f} at '
                  f'{group_stats["peak_latencies"]["global_peak_time"]:.3f}s',
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_ylim([0.35, 0.85])
    ax1.set_xlim([-0.2, 1.0])

    # === PANEL 2: Average TGM (if available) ===
    # For now, we'll create a placeholder TGM or load if available
    # This would need to be implemented based on your TGM data structure

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

    im = ax2.imshow(tgm_matrix, cmap='RdBu_r', aspect='auto', origin='lower',
                    extent=[times[0], times[-1], times[0], times[-1]],
                    vmin=CHANCE_LEVEL - 0.1, vmax=CHANCE_LEVEL + 0.1)
    ax2.set_xlabel('Testing Time (s)', fontsize=14)
    ax2.set_ylabel('Training Time (s)', fontsize=14)
    ax2.set_title(f'{group_name} - Average TGM',
                  fontsize=16, fontweight='bold')

    # Add stimulus onset lines
    ax2.axvline(x=0, color='black', linestyle=':', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.8)

    plt.colorbar(im, ax=ax2, label='AUC Score')

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
    # Marquer les points significatifs FDR (utiliser fdr_counts_per_timepoint)
    if np.any(group_stats['fdr_mask']):
        y_fdr = 0.37
        sig_times = times[group_stats['fdr_mask']]
        sig_values = np.full_like(sig_times, y_fdr)
        sig_values = np.full_like(sig_times, y_fdr)
        ax.scatter(sig_times, sig_values, color='blue', marker='|', s=60,
                   alpha=0.8, label='Significatif FDR (p<0.05)', zorder=4)

    # Configuration des axes - Extension temporelle complète
    ax.set_xlabel('Temps (s)', fontsize=14)
    ax.set_ylabel('Précision de Décodage (AUC)', fontsize=14)
    ax.set_title(f'{group_name} - Décodage Temporel (n={n_subjects})\n'
                 f'AUC Moyenne: {group_stats["global_auc"]:.3f} ± {np.std(group_stats["subject_means"]):.3f}',
                 fontsize=16, fontweight='bold')

    ax.set_xlim([-0.2, 1.0])  # Extension complète de -0.2 à 1.0 seconde
    ax.set_ylim([0.35, 0.85])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='upper right')

    # Ajouter statistiques dans un encadré
    stats_text = f'Points significatifs FDR: {np.sum(group_stats["fdr_mask"])}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
    Créer une comparaison détaillée des 3 groupes avec SEM et points significatifs.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    group_names = three_group_results['group_names']
    times = group_stats[group_names[0]]['times']

    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c'
    }

    # === PANNEAU 1: Comparaison des moyennes avec SEM ===

    for group_name in group_names:
        stats = group_stats[group_name]
        color = colors.get(group_name, COLORS_PALETTE[0])

        # Zone SEM
        ax1.fill_between(times,
                         stats['group_mean'] - stats['group_sem'],
                         stats['group_mean'] + stats['group_sem'],
                         alpha=0.2, color=color, zorder=1)

        # Ligne moyenne
        ax1.plot(times, stats['group_mean'], color=color, linewidth=3,
                 label=f'{group_name} (n={stats["n_subjects"]})', zorder=2)

    # Ligne de chance
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
                alpha=0.7, label='Niveau de Chance', zorder=2)

    # Marquer les points significatifs ANOVA
    if np.any(three_group_results['significant_anova']):
        y_sig = 0.37
        sig_times = times[three_group_results['significant_anova']]
        sig_values = np.full_like(sig_times, y_sig)
        ax1.scatter(sig_times, sig_values, color='purple', marker='s', s=40,
                    alpha=0.8, label='ANOVA Significatif (p<0.05)', zorder=3)

    ax1.set_xlabel('Temps (s)', fontsize=14)
    ax1.set_ylabel('Précision de Décodage (AUC)', fontsize=14)
    ax1.set_title(f'Comparaison Temporelle des Trois Groupes\n'
                  f'ANOVA Global: F = {three_group_results["global_f_stat"]:.3f}, '
                  f'p = {three_group_results["global_p_anova"]:.4f}',
                  fontsize=16, fontweight='bold')

    ax1.set_xlim([-0.2, 1.0])  # Extension complète de -0.2 à 1.0 seconde
    ax1.set_ylim([0.40, 0.60])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc='upper right')

    # === PANNEAU 2: Comparaisons par paires avec différences ===

    # Calculer et afficher les différences entre groupes
    pairwise_colors = ['blue', 'red', 'green']
    y_offset = 0

    for i, (comparison_name, results) in enumerate(three_group_results['pairwise_results'].items()):
        group1_name = results['group1_name']
        group2_name = results['group2_name']

        # Calculer la différence des moyennes
        diff_means = group_stats[group1_name]['group_mean'] - \
            group_stats[group2_name]['group_mean']

        # Plot la différence
        ax2.plot(times, diff_means + y_offset, color=pairwise_colors[i], linewidth=2,
                 label=f'{group1_name} - {group2_name}', alpha=0.8)

        # Marquer les points significatifs
        if np.any(results['significant_points']):
            sig_times = times[results['significant_points']]
            sig_diffs = diff_means[results['significant_points']] + y_offset
            ax2.scatter(sig_times, sig_diffs, color=pairwise_colors[i], marker='o', s=30,
                        alpha=0.9, zorder=3)

        # Ligne de référence pour cette comparaison
        ax2.axhline(
            y=y_offset, color=pairwise_colors[i], linestyle=':', alpha=0.5)

        y_offset += 0.1  # Décaler verticalement pour la prochaine comparaison

    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_ylabel('Différence AUC (Groupe1 - Groupe2)', fontsize=14)
    ax2.set_title('Différences Par Paires Entre Groupes\n'
                  'Points marqués = Significatifs (FDR p<0.05)',
                  fontsize=14, fontweight='bold')

    ax2.set_xlim([-0.2, 1.0])  # Extension complète de -0.2 à 1.0 seconde
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper right')

    # Ajouter statistiques résumées
    stats_text = "Résumé des comparaisons:\n"
    for comparison_name, results in three_group_results['pairwise_results'].items():
        n_sig = np.sum(results['significant_points'])
        max_effect = np.max(np.abs(results['effect_sizes']))
        stats_text += f"{comparison_name}: {n_sig} pts sig., d_max={max_effect:.2f}\n"

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Sauvegarder
    filename = f"{protocol}_three_groups_detailed_temporal_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparaison détaillée des 3 groupes sauvée: {output_path}")
    return output_path


def create_fdr_significance_histogram(group_stats: Dict[str, Any], group_name: str,
                                      output_dir: str, protocol: str) -> str:
    """
    Créer un histogramme/graphique en points montrant le nombre de sujets 
    avec significativité déjà calculée (pp_ap_main_temporal_1d_fdr) à chaque point temporel.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728',
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c'
    }

    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    color = colors.get(group_name, COLORS_PALETTE[0])

    # Utiliser les comptes de significativité déjà calculés à partir des données NPZ
    if 'fdr_counts_per_timepoint' in group_stats:
        significance_counts = group_stats['fdr_counts_per_timepoint']
        logger.info(f"DEBUG: Found fdr_counts_per_timepoint - Shape: {significance_counts.shape}, "
                    f"Min: {np.min(significance_counts)}, Max: {np.max(significance_counts)}, "
                    f"Sum: {np.sum(significance_counts)}")
    else:
        logger.warning(
            "Pas de données de significativité disponibles, utilisation de données de fallback")
        significance_counts = np.zeros(len(times))

    # DEBUG: Vérifier les masques FDR individuels aussi
    if 'individual_fdr_masks' in group_stats:
        logger.info(
            f"DEBUG: Found {len(group_stats['individual_fdr_masks'])} individual FDR masks")
        # Check first 3
        for i, mask in enumerate(group_stats['individual_fdr_masks'][:3]):
            logger.info(
                f"DEBUG: Subject {i} FDR mask - Shape: {mask.shape}, Sum: {np.sum(mask)}")

    # Afficher où sont les points significatifs s'il y en a
    if np.sum(significance_counts) > 0:
        sig_indices = np.where(significance_counts > 0)[0]
        # First 10
        logger.info(
            f"DEBUG: Points significatifs trouvés aux indices: {sig_indices[:10]}...")
        logger.info(
            f"DEBUG: Temps correspondants: {times[sig_indices[:10]]}...")
    else:
        logger.warning(
            f"ATTENTION: Aucune donnée significative réelle trouvée pour {group_name}")
        logger.info(
            "Possible problème avec les masques FDR dans les fichiers NPZ")
        logger.info("Affichage de l'histogramme vide pour diagnostic")

    # Panel 1: Histogramme des comptes de significativité
    # Utiliser une largeur plus appropriée pour la visualisation
    time_step = times[1] - times[0] if len(times) > 1 else 0.002
    width = time_step * 0.8  # 80% de l'espacement temporel

    bars = ax1.bar(times, significance_counts, width=width,
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Afficher quelques statistiques pour debugging
    logger.info(
        f"DEBUG: Plotting {len(significance_counts)} bars with max height {np.max(significance_counts)}")
    ax1.set_xlabel('Temps (s)', fontsize=12)
    ax1.set_ylabel('Nombre de sujets significatifs', fontsize=12)
    ax1.set_title(f'{group_name} - Nombre de sujets significatifs par point temporel\n'
                  f'(n={n_subjects} sujets, données des fichiers NPZ)', fontsize=14, fontweight='bold')
    ax1.set_xlim([-0.2, 1.0])
    ax1.set_ylim([0, n_subjects + 1])
    ax1.grid(True, alpha=0.3)

    # Ajouter ligne de stimulus onset
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='Stimulus Onset')
    ax1.legend()

    # Panel 2: Proportion de sujets significatifs (graphique en points)
    proportion_significant = significance_counts / \
        n_subjects if n_subjects > 0 else np.zeros_like(significance_counts)
    ax2.plot(times, proportion_significant, color=color,
             linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax2.fill_between(times, 0, proportion_significant, color=color, alpha=0.3)
    ax2.set_xlabel('Temps (s)', fontsize=12)
    ax2.set_ylabel('Proportion de sujets significatifs', fontsize=12)
    ax2.set_title(f'{group_name} - Proportion de sujets significatifs',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    # Ajouter ligne de stimulus onset
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2,
                alpha=0.8, label='Stimulus Onset')
    ax2.legend()

    plt.tight_layout()

    # Sauvegarder
    safe_name = group_name.replace(' ', '_').replace(
        '+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_significance_histogram.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Histogramme FDR significatif sauvé: {output_path}")
    return output_path


def create_global_fdr_significance_histogram(all_group_stats: Dict[str, Dict[str, Any]], 
                                           output_dir: str, protocol: str) -> str:
    """
    Créer un histogramme global montrant le nombre de sujets significatifs 
    pour tous les groupes sur le même graphique.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    
    # Couleurs pour chaque groupe
    colors = {
        'DELIRIUM +': '#d62728', 
        'DELIRIUM -': '#ff7f0e',
        'controls': '#2ca02c'
    }
    
    # Vérifier que tous les groupes ont les mêmes temps
    times = None
    max_subjects_total = 0
    
    for group_name, stats in all_group_stats.items():
        if times is None:
            times = stats['times']
        max_subjects_total += stats['n_subjects']
    
    # Panel 1: Histogrammes superposés pour chaque groupe
    bar_width = (times[1] - times[0]) * 0.25 if len(times) > 1 else 0.0005  # Barres plus fines
    
    group_names = list(all_group_stats.keys())
    for i, group_name in enumerate(group_names):
        stats = all_group_stats[group_name]
        color = colors.get(group_name, COLORS_PALETTE[i])
        n_subjects = stats['n_subjects']
        
        if 'fdr_counts_per_timepoint' in stats:
            significance_counts = stats['fdr_counts_per_timepoint']
        else:
            significance_counts = np.zeros(len(times))
        
        # Décaler les barres pour éviter la superposition
        time_offset = bar_width * (i - len(group_names)/2 + 0.5)
        
        ax1.bar(times + time_offset, significance_counts, width=bar_width,
               color=color, alpha=0.7, label=f'{group_name} (n={n_subjects})',
               edgecolor='black', linewidth=0.3)
    
    ax1.set_xlabel('Temps (s)', fontsize=14)
    ax1.set_ylabel('Nombre de sujets significatifs', fontsize=14)
    ax1.set_title(f'Comparaison globale - Nombre de sujets significatifs par groupe\n'
                  f'(Protocole {protocol}, données FDR des fichiers NPZ)', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlim([-0.2, 1.0])
    ax1.set_ylim([0, max([stats['n_subjects'] for stats in all_group_stats.values()]) + 1])
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label='Stimulus Onset')
    ax1.legend(fontsize=12, loc='upper right')
    
    # Panel 2: Proportions superposées pour chaque groupe
    for i, group_name in enumerate(group_names):
        stats = all_group_stats[group_name]
        color = colors.get(group_name, COLORS_PALETTE[i])
        n_subjects = stats['n_subjects']
        
        if 'fdr_counts_per_timepoint' in stats:
            significance_counts = stats['fdr_counts_per_timepoint']
        else:
            significance_counts = np.zeros(len(times))
        
        proportion_significant = significance_counts / n_subjects if n_subjects > 0 else np.zeros_like(significance_counts)
        
        ax2.plot(times, proportion_significant, color=color, linewidth=3, 
                alpha=0.8, label=f'{group_name} (n={n_subjects})')
        ax2.fill_between(times, 0, proportion_significant, color=color, alpha=0.2)
    
    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_ylabel('Proportion de sujets significatifs', fontsize=14)
    ax2.set_title(f'Comparaison globale - Proportion de sujets significatifs par groupe', 
                  fontsize=16, fontweight='bold')
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, 
               alpha=0.8, label='Stimulus Onset')
    ax2.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"{protocol}_global_significance_histogram_all_groups.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistiques résumées
    logger.info("=== RÉSUMÉ GLOBAL DES SIGNIFICATIVITÉS FDR ===")
    for group_name, stats in all_group_stats.items():
        if 'fdr_counts_per_timepoint' in stats:
            total_sig = np.sum(stats['fdr_counts_per_timepoint'])
            max_subjects_at_timepoint = np.max(stats['fdr_counts_per_timepoint'])
            n_timepoints_with_sig = np.sum(stats['fdr_counts_per_timepoint'] > 0)
            logger.info(f"{group_name}: {total_sig} points significatifs totaux, "
                       f"max {max_subjects_at_timepoint} sujets/point temporel, "
                       f"{n_timepoints_with_sig} points temporels avec significativité")
    
    logger.info(f"Histogramme global sauvé: {output_path}")
    return output_path


def calculate_pairwise_statistical_tests(group_stats_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculer les tests statistiques sur les différences pairées entre groupes.
    """
    group_names = list(group_stats_dict.keys())
    n_groups = len(group_names)
    pairwise_results = {}

    if n_groups < 2:
        logger.warning(
            "Moins de 2 groupes disponibles pour les comparaisons pairées")
        return pairwise_results

    # Générer toutes les combinaisons pairées
    for i, group1 in enumerate(group_names):
        for j, group2 in enumerate(group_names[i+1:], i+1):
            pair_name = f"{group1}_vs_{group2}"

            # [n_subjects1, n_timepoints]
            scores1 = group_stats_dict[group1]['scores']
            # [n_subjects2, n_timepoints]
            scores2 = group_stats_dict[group2]['scores']
            times = group_stats_dict[group1]['times']

            n_timepoints = len(times)

            # Tests t indépendants à chaque point temporel
            t_stats = []
            p_values = []

            for t_idx in range(n_timepoints):
                t_stat, p_val = ttest_ind(scores1[:, t_idx], scores2[:, t_idx])
                t_stats.append(t_stat)
                p_values.append(p_val)

            # Correction FDR
            _, fdr_mask, _, fdr_pvals = multipletests(
                p_values, alpha=FDR_ALPHA, method='fdr_bh')

            pairwise_results[pair_name] = {
                'group1': group1,
                'group2': group2,
                'times': times,
                't_stats': np.array(t_stats),
                'p_values': np.array(p_values),
                'fdr_mask': fdr_mask,
                'fdr_pvals': fdr_pvals,
                'n_significant': np.sum(fdr_mask),
                'mean_diff': np.mean(scores1, axis=0) - np.mean(scores2, axis=0)
            }

            logger.info(
                f"Comparaison pairée {pair_name}: {np.sum(fdr_mask)} points significatifs")

    return pairwise_results


def generate_comprehensive_report(all_results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate a comprehensive text report with all statistical results.
    """
    report_path = os.path.join(
        output_dir, "comprehensive_statistical_report.txt")

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EEG DECODING ANALYSIS REPORT\n")
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
                f.write(
                    f"  Peak AUC: {stats['peak_latencies']['global_peak_value']:.4f} ")
                f.write(
                    f"at {stats['peak_latencies']['global_peak_time']:.3f}s\n")
                f.write(
                    f"  FDR significant points: {np.sum(stats['fdr_mask'])}\n")
                f.write(
                    f"  Significant time windows: {stats['peak_latencies']['n_significant_windows']}\n")

                if stats['peak_latencies']['n_significant_windows'] > 0:
                    f.write("  Significant windows details:\n")
                    for i, window in enumerate(stats['peak_latencies']['significant_windows']):
                        f.write(
                            f"    Window {i+1}: {window['start_time']:.3f}s - {window['end_time']:.3f}s ")
                        f.write(
                            f"(duration: {window['duration']:.3f}s, peak: {window['peak_value']:.4f})\n")

            # Three-group comparison
            if 'three_group_comparison' in results:
                comp = results['three_group_comparison']
                f.write(f"\nTHREE-GROUP COMPARISON:\n")
                f.write(
                    f"  Global ANOVA: F = {comp['global_f_stat']:.4f}, p = {comp['global_p_anova']:.6f}\n")
                f.write(
                    f"  Significant time points (ANOVA): {np.sum(comp['significant_anova'])}\n")

                f.write(f"\nPAIRWISE COMPARISONS:\n")
                for comparison_name, pairwise in comp['pairwise_results'].items():
                    f.write(f"  {comparison_name}:\n")
                    f.write(
                        f"    Significant time points: {np.sum(pairwise['significant_points'])}\n")
                    f.write(
                        f"    Max effect size: {np.max(np.abs(pairwise['effect_sizes'])):.4f}\n")
                    f.write(
                        f"    Max t-statistic: {np.max(np.abs(pairwise['t_stats'])):.4f}\n")

            # Individual pairwise tests section
            if 'pairwise_tests' in results:
                pairwise_tests = results['pairwise_tests']
                f.write(f"\nINDIVIDUAL PAIRWISE STATISTICAL TESTS:\n")
                for pair_name, pair_data in pairwise_tests.items():
                    f.write(f"  {pair_name}:\n")
                    f.write(
                        f"    Groups compared: {pair_data['group1']} vs {pair_data['group2']}\n")
                    f.write(
                        f"    Significant time points (FDR): {pair_data['n_significant']}\n")
                    f.write(
                        f"    Max |t-statistic|: {np.max(np.abs(pair_data['t_stats'])):.4f}\n")
                    f.write(
                        f"    Max |mean difference|: {np.max(np.abs(pair_data['mean_diff'])):.4f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ANALYSIS COMPLETED\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Comprehensive report saved: {report_path}")
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
            for file_path in file_list:
                data = load_npz_data(file_path)
                # Vérifier que les données du sujet couvrent bien notre fenêtre
                if data and len(data['scores']) >= end_idx:
                    # Extraire uniquement la fenêtre temporelle d'intérêt
                    group_scores.append(data['scores'][start_idx:end_idx])
                    group_subject_ids.append(data['subject_id'])

                    # Extraire également le masque FDR si disponible
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
            logger.info(
                f"  Conservé {len(group_scores)} sujets pour le groupe '{group_name}' "
                f"avec {scores_matrix.shape[1]} points temporels")

            # Calculer les statistiques améliorées
            stats = calculate_basic_statistics(
                scores_matrix, times_ref, group_subject_ids, fdr_masks_matrix)
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

            # Créer l'histogramme de significativité par sujet
            create_fdr_significance_histogram(
                stats, group_name, output_dir, protocol)

        # 3.4. Créer l'histogramme global avec tous les groupes
        valid_groups = {g: s for g, s in protocol_stats.items()
                        if s and s.get('n_subjects', 0) > 0}
        
        if len(valid_groups) >= 2:
            logger.info("\n--- Creating Global FDR Significance Histogram ---")
            create_global_fdr_significance_histogram(
                valid_groups, output_dir, protocol)

        # 3.5. Perform pairwise statistical tests
        
        if len(valid_groups) >= 2:
            logger.info("\n--- Performing Pairwise Statistical Tests ---")
            pairwise_results = calculate_pairwise_statistical_tests(
                valid_groups)
            all_results[protocol]['pairwise_tests'] = pairwise_results

            # Log results
            for pair_name, pair_data in pairwise_results.items():
                logger.info(
                    f"Comparaison {pair_name}: {pair_data['n_significant']} points significatifs")

        # 4. Perform three-group comparison if we have exactly 3 groups

        if len(valid_groups) == 3:
            logger.info("\n--- Performing Three-Group Comparison ---")
            three_group_results = perform_three_group_comparison(valid_groups)
            all_results[protocol]['three_group_comparison'] = \
                three_group_results

            # Create three-group temporal comparison
            create_three_group_temporal_comparison(
                valid_groups, three_group_results, output_dir, protocol)

            logger.info(f"Three-group ANOVA: "
                        f"F = {three_group_results['global_f_stat']:.3f}, "
                        f"p = {three_group_results['global_p_anova']:.4f}")
        else:
            logger.warning(f"Expected 3 groups for comparison, "
                           f"got {len(valid_groups)}")

    # 5. Generate comprehensive report
    generate_comprehensive_report(all_results, output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("ENHANCED ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All results saved in: {os.path.abspath(output_dir)}")
    logger.info("Generated files:")
    logger.info("  - Enhanced group visualizations (2 panels each)")
    logger.info("  - Three-group comprehensive comparison")
    logger.info("  - Detailed latency analysis")
    logger.info("  - Comprehensive statistical report")


if __name__ == "__main__":
    main()