#!/usr/bin/env python3
"""
This script provides comprehensive analysis and visualization of EEG decoding results
for scientific publications. It processes individual subject NPZ files and generates
publication-quality figures with extensive statistical analyses.

"""

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('eeg_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Global parameters for publication-quality figures
PUBLICATION_PARAMS = {
    'figure.figsize': (16, 10),
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

# Apply publication parameters
plt.rcParams.update(PUBLICATION_PARAMS)

# Color palettes for consistent visualization
COLORS_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
CHANCE_LEVEL = 0.5


def setup_logging() -> logging.Logger:
    """
    Set up comprehensive logging for the analysis pipeline.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('EEG_Analysis')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('eeg_decoding_analysis.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def find_npz_files(base_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Recursively find and organize NPZ files by protocol and group.
    
    Args:
        base_path (str): Base directory to search for NPZ files
        
    Returns:
        Dict[str, Dict[str, List[str]]]: Organized data structure with protocols and groups
    """
    logger.info(f"Searching for NPZ files in: {base_path}")
    
    organized_data = {}
    
    # Search patterns for different file structures
    search_patterns = [
        '**/*.npz',
        '**/results_*.npz',
        '**/decoding_*.npz',
        '**/subject_*.npz'
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(os.path.join(base_path, pattern), recursive=True)
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    
    logger.info(f"Found {len(all_files)} NPZ files")
    
    # Organize files by protocol and group
    for file_path in all_files:
        try:
            # Extract protocol and group information from path
            rel_path = os.path.relpath(file_path, base_path)
            path_parts = rel_path.split(os.sep)
            
            # Try to identify protocol and group from path structure
            protocol = "Unknown"
            group = "Unknown"
            
            # Common naming patterns
            for part in path_parts:
                if 'protocol' in part.lower() or 'pp' in part.lower():
                    protocol = part
                elif any(keyword in part.lower() for keyword in ['control', 'patient', 'group', 'condition']):
                    group = part
                elif part.endswith('.npz'):
                    # Extract subject ID from filename
                    subject_id = os.path.splitext(part)[0]
                    
            # Initialize nested structure
            if protocol not in organized_data:
                organized_data[protocol] = {}
            if group not in organized_data[protocol]:
                organized_data[protocol][group] = []
                
            organized_data[protocol][group].append(file_path)
            
        except Exception as e:
            logger.warning(f"Error processing file {file_path}: {e}")
            continue
    
    # Log summary
    for protocol, groups in organized_data.items():
        logger.info(f"Protocol {protocol}:")
        for group, files in groups.items():
            logger.info(f"  - {group}: {len(files)} files")
    
    return organized_data


def load_npz_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate NPZ file data.
    
    Args:
        file_path (str): Path to NPZ file
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data or None if invalid
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Validate required fields
        required_fields = ['scores', 'times']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field '{field}' in {file_path}")
                return None
        
        # Convert to dictionary for easier handling
        result = {}
        for key in data.keys():
            result[key] = data[key]
            
        # Extract subject ID from filename
        result['subject_id'] = os.path.splitext(os.path.basename(file_path))[0]
        result['file_path'] = file_path
        
        return result
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None


def calculate_comprehensive_statistics(scores: np.ndarray, times: np.ndarray, 
                                     subject_ids: List[str]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistical measures for group analysis.
    
    Args:
        scores (np.ndarray): Decoding scores array (subjects x time points)
        times (np.ndarray): Time points array
        subject_ids (List[str]): List of subject identifiers
        
    Returns:
        Dict[str, Any]: Comprehensive statistical results
    """
    n_subjects, n_timepoints = scores.shape
    
    # Basic descriptive statistics
    group_mean = np.mean(scores, axis=0)
    group_std = np.std(scores, axis=0, ddof=1)
    group_sem = group_std / np.sqrt(n_subjects)
    
    # Confidence intervals (95%)
    ci_lower = group_mean - 1.96 * group_sem
    ci_upper = group_mean + 1.96 * group_sem
    
    # Individual subject statistics
    subject_means = np.mean(scores, axis=1)
    subject_peaks = np.max(scores, axis=1)
    subject_peak_times = times[np.argmax(scores, axis=1)]
    
    # Global AUC calculation
    global_auc = np.mean(subject_means)
    
    # Statistical tests against chance level
    t_stats = np.zeros(n_timepoints)
    p_values = np.zeros(n_timepoints)
    effect_sizes = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        t_stat, p_val = ttest_1samp(scores[:, t], CHANCE_LEVEL)
        t_stats[t] = t_stat
        p_values[t] = p_val
        
        # Cohen's d effect size
        effect_sizes[t] = (group_mean[t] - CHANCE_LEVEL) / group_std[t]
    
    # Multiple comparisons correction
    p_corrected = multipletests(p_values, method='fdr_bh')[1]
    significant_points = p_corrected < 0.05
    
    # Peak analysis within 0-1s window
    peak_window_mask = (times >= 0) & (times <= 1.0)
    if np.any(peak_window_mask):
        peak_window_scores = scores[:, peak_window_mask]
        peak_window_times = times[peak_window_mask]
        
        # Find peak for each subject
        peak_latencies = []
        peak_amplitudes = []
        
        for subj in range(n_subjects):
            subj_scores = peak_window_scores[subj, :]
            if len(subj_scores) > 0:
                peak_idx = np.argmax(subj_scores)
                peak_latencies.append(peak_window_times[peak_idx])
                peak_amplitudes.append(subj_scores[peak_idx])
        
        peak_latencies = np.array(peak_latencies)
        peak_amplitudes = np.array(peak_amplitudes)
    else:
        peak_latencies = np.array([])
        peak_amplitudes = np.array([])
    
    # Temporal clustering analysis
    cluster_results = perform_cluster_analysis(scores, times, p_corrected)
    
    # Inter-subject variability measures
    variability_measures = calculate_variability_measures(scores, times)
    
    # Reliability analysis
    reliability_measures = calculate_reliability_measures(scores)
    
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
        't_stats': t_stats,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'effect_sizes': effect_sizes,
        'significant_points': significant_points,
        'peak_latencies': peak_latencies,
        'peak_amplitudes': peak_amplitudes,
        'cluster_results': cluster_results,
        'variability_measures': variability_measures,
        'reliability_measures': reliability_measures
    }


def perform_cluster_analysis(scores: np.ndarray, times: np.ndarray, 
                           p_values: np.ndarray, cluster_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Perform cluster-based statistical analysis for temporal data.
    
    Args:
        scores (np.ndarray): Decoding scores (subjects x time)
        times (np.ndarray): Time points
        p_values (np.ndarray): P-values for each time point
        cluster_threshold (float): P-value threshold for cluster formation
        
    Returns:
        Dict[str, Any]: Cluster analysis results
    """
    significant_mask = p_values < cluster_threshold
    
    # Find continuous significant clusters
    clusters = []
    if np.any(significant_mask):
        # Find cluster boundaries
        diff = np.diff(np.concatenate(([False], significant_mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            if end > start:  # Valid cluster
                cluster_times = times[start:end]
                cluster_scores = np.mean(scores[:, start:end], axis=1)
                cluster_p_values = p_values[start:end]
                
                clusters.append({
                    'start_time': cluster_times[0],
                    'end_time': cluster_times[-1],
                    'duration': cluster_times[-1] - cluster_times[0],
                    'start_idx': start,
                    'end_idx': end,
                    'mean_score': np.mean(cluster_scores),
                    'min_p_value': np.min(cluster_p_values),
                    'cluster_mass': np.sum(-np.log10(cluster_p_values))
                })
    
    return {
        'n_clusters': len(clusters),
        'clusters': clusters,
        'total_significant_duration': sum(c['duration'] for c in clusters)
    }


def calculate_variability_measures(scores: np.ndarray, times: np.ndarray) -> Dict[str, Any]:
    """
    Calculate inter-subject variability measures.
    
    Args:
        scores (np.ndarray): Decoding scores (subjects x time)
        times (np.ndarray): Time points
        
    Returns:
        Dict[str, Any]: Variability measures
    """
    n_subjects = scores.shape[0]
    
    # Coefficient of variation across subjects
    cv_across_time = np.std(scores, axis=0) / np.mean(scores, axis=0)
    cv_across_time = np.nan_to_num(cv_across_time)
    
    # Individual subject variability (across time)
    subject_cv = []
    for subj in range(n_subjects):
        subj_scores = scores[subj, :]
        if np.std(subj_scores) > 0:
            subject_cv.append(np.std(subj_scores) / np.mean(subj_scores))
        else:
            subject_cv.append(0)
    
    subject_cv = np.array(subject_cv)
    
    # Range measures
    score_ranges = np.max(scores, axis=1) - np.min(scores, axis=1)
    
    return {
        'cv_across_time': cv_across_time,
        'mean_cv_across_time': np.mean(cv_across_time),
        'subject_cv': subject_cv,
        'mean_subject_cv': np.mean(subject_cv),
        'score_ranges': score_ranges,
        'mean_score_range': np.mean(score_ranges)
    }


def calculate_reliability_measures(scores: np.ndarray) -> Dict[str, Any]:
    """
    Calculate reliability measures for the decoding results.
    
    Args:
        scores (np.ndarray): Decoding scores (subjects x time)
        
    Returns:
        Dict[str, Any]: Reliability measures
    """
    n_subjects, n_timepoints = scores.shape
    
    # Split-half reliability (odd vs even time points)
    if n_timepoints >= 4:
        odd_scores = scores[:, ::2]
        even_scores = scores[:, 1::2]
        
        # Calculate mean scores for each half
        odd_means = np.mean(odd_scores, axis=1)
        even_means = np.mean(even_scores, axis=1)
        
        # Correlation between halves
        if len(odd_means) > 1 and len(even_means) > 1:
            split_half_r, split_half_p = pearsonr(odd_means, even_means)
            
            # Spearman-Brown correction
            split_half_corrected = (2 * split_half_r) / (1 + split_half_r)
        else:
            split_half_r = np.nan
            split_half_p = np.nan
            split_half_corrected = np.nan
    else:
        split_half_r = np.nan
        split_half_p = np.nan
        split_half_corrected = np.nan
    
    # Internal consistency (Cronbach's alpha approximation)
    if n_subjects > 1:
        # Variance of sum scores
        sum_scores = np.sum(scores, axis=1)
        var_sum = np.var(sum_scores, ddof=1)
        
        # Sum of individual variances
        var_individual = np.sum(np.var(scores, axis=0, ddof=1))
        
        # Cronbach's alpha
        if var_sum > 0:
            cronbach_alpha = (n_timepoints / (n_timepoints - 1)) * (1 - var_individual / var_sum)
        else:
            cronbach_alpha = np.nan
    else:
        cronbach_alpha = np.nan
    
    return {
        'split_half_r': split_half_r,
        'split_half_p': split_half_p,
        'split_half_corrected': split_half_corrected,
        'cronbach_alpha': cronbach_alpha
    }


def perform_group_comparison(group1_stats: Dict[str, Any], group2_stats: Dict[str, Any],
                           group1_name: str, group2_name: str) -> Dict[str, Any]:
    """
    Perform comprehensive statistical comparison between two groups.
    
    Args:
        group1_stats (Dict[str, Any]): Statistics for group 1
        group2_stats (Dict[str, Any]): Statistics for group 2
        group1_name (str): Name of group 1
        group2_name (str): Name of group 2
        
    Returns:
        Dict[str, Any]: Comprehensive comparison results
    """
    scores1 = group1_stats['scores']
    scores2 = group2_stats['scores']
    times = group1_stats['times']
    
    n1, n_timepoints = scores1.shape
    n2, _ = scores2.shape
    
    # Point-by-point t-tests
    t_stats = np.zeros(n_timepoints)
    p_values = np.zeros(n_timepoints)
    effect_sizes = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Independent samples t-test
        t_stat, p_val = ttest_ind(scores1[:, t], scores2[:, t])
        t_stats[t] = t_stat
        p_values[t] = p_val
        
        # Cohen's d effect size
        pooled_std = np.sqrt(((n1 - 1) * np.var(scores1[:, t], ddof=1) + 
                             (n2 - 1) * np.var(scores2[:, t], ddof=1)) / (n1 + n2 - 2))
        if pooled_std > 0:
            effect_sizes[t] = (np.mean(scores1[:, t]) - np.mean(scores2[:, t])) / pooled_std
        else:
            effect_sizes[t] = 0
    
    # Multiple comparisons correction
    p_corrected = multipletests(p_values, method='fdr_bh')[1]
    significant_points = p_corrected < 0.05
    
    # Global comparison (mean AUC)
    global_t_stat, global_p_value = ttest_ind(group1_stats['subject_means'], 
                                             group2_stats['subject_means'])
    
    # Effect size for global comparison
    pooled_std_global = np.sqrt(((n1 - 1) * np.var(group1_stats['subject_means'], ddof=1) + 
                                (n2 - 1) * np.var(group2_stats['subject_means'], ddof=1)) / (n1 + n2 - 2))
    if pooled_std_global > 0:
        global_effect_size = (np.mean(group1_stats['subject_means']) - 
                             np.mean(group2_stats['subject_means'])) / pooled_std_global
    else:
        global_effect_size = 0
    
    # Non-parametric tests
    global_u_stat, global_u_p = mannwhitneyu(group1_stats['subject_means'], 
                                             group2_stats['subject_means'], 
                                             alternative='two-sided')
    
    # Peak latency comparison
    if len(group1_stats['peak_latencies']) > 0 and len(group2_stats['peak_latencies']) > 0:
        peak_t_stat, peak_p_value = ttest_ind(group1_stats['peak_latencies'], 
                                             group2_stats['peak_latencies'])
        peak_u_stat, peak_u_p = mannwhitneyu(group1_stats['peak_latencies'], 
                                             group2_stats['peak_latencies'], 
                                             alternative='two-sided')
    else:
        peak_t_stat = peak_p_value = peak_u_stat = peak_u_p = np.nan
    
    # Cluster analysis for group differences
    cluster_results = perform_cluster_analysis(
        np.concatenate([scores1, scores2], axis=0),
        times, p_corrected
    )
    
    return {
        'group1_name': group1_name,
        'group2_name': group2_name,
        'n1': n1,
        'n2': n2,
        't_stats': t_stats,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'effect_sizes': effect_sizes,
        'significant_points': significant_points,
        'global_t_stat': global_t_stat,
        'global_p_value': global_p_value,
        'global_effect_size': global_effect_size,
        'global_u_stat': global_u_stat,
        'global_u_p': global_u_p,
        'peak_t_stat': peak_t_stat,
        'peak_p_value': peak_p_value,
        'peak_u_stat': peak_u_stat,
        'peak_u_p': peak_u_p,
        'cluster_results': cluster_results
    }


def create_publication_figure(group_stats: Dict[str, Any], group_name: str, 
                            output_dir: str, protocol: str) -> str:
    """
    Create publication-quality figure for a single group.
    
    Args:
        group_stats (Dict[str, Any]): Group statistics
        group_name (str): Name of the group
        output_dir (str): Output directory
        protocol (str): Protocol name
        
    Returns:
        str: Path to saved figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    
    # Panel 1: Individual curves and group average
    # Plot ALL individual subject curves
    for i in range(n_subjects):
        ax1.plot(times, scores[i, :], color='lightgray', alpha=0.6, linewidth=1)
    
    # Plot group average with confidence interval
    ax1.fill_between(times, group_stats['ci_lower'], group_stats['ci_upper'], 
                     alpha=0.3, color='blue', label='95% CI')
    ax1.plot(times, group_stats['group_mean'], color='blue', linewidth=3, 
             label=f'Group Mean (n={n_subjects})')
    
    # Add chance level
    ax1.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', linewidth=2, 
                alpha=0.8, label='Chance Level')
    
    # Mark significant time points
    if np.any(group_stats['significant_points']):
        sig_times = times[group_stats['significant_points']]
        sig_scores = group_stats['group_mean'][group_stats['significant_points']]
        ax1.scatter(sig_times, sig_scores, color='red', s=20, zorder=5, 
                   label='Significant (p<0.05, FDR corrected)')
    
    # Formatting
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax1.set_title(f'{group_name} - Temporal Decoding\n'
                  f'Mean AUC: {group_stats["global_auc"]:.3f}', 
                  fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim([0.4, 0.8])
    
    # Add statistical annotations
    max_score = np.max(group_stats['group_mean'])
    ax1.text(0.02, 0.98, f'Max Score: {max_score:.3f}\n'
                         f'Peak Time: {times[np.argmax(group_stats["group_mean"])]:.3f}s\n'
                         f'Effect Size: {np.max(group_stats["effect_sizes"]):.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             fontsize=10)
    
    # Panel 2: Peak latency histogram and statistics
    if len(group_stats['peak_latencies']) > 0:
        ax2.hist(group_stats['peak_latencies'] * 1000, bins=max(5, n_subjects//3), 
                 alpha=0.7, color='blue', edgecolor='black', density=False)
        
        # Add statistics
        mean_latency = np.mean(group_stats['peak_latencies']) * 1000
        std_latency = np.std(group_stats['peak_latencies']) * 1000
        
        ax2.axvline(mean_latency, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_latency:.1f}ms')
        ax2.axvline(mean_latency - std_latency, color='red', linestyle=':', 
                   alpha=0.7, label=f'±1 SD')
        ax2.axvline(mean_latency + std_latency, color='red', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Peak Latency (ms)', fontsize=14)
        ax2.set_ylabel('Number of Subjects', fontsize=14)
        ax2.set_title(f'Peak Latency Distribution\n'
                      f'Mean ± SD: {mean_latency:.1f} ± {std_latency:.1f} ms', 
                      fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        # Add distribution statistics
        ax2.text(0.98, 0.98, f'N = {len(group_stats["peak_latencies"])}\n'
                             f'Range: {np.min(group_stats["peak_latencies"]) * 1000:.1f}-'
                             f'{np.max(group_stats["peak_latencies"]) * 1000:.1f} ms\n'
                             f'Median: {np.median(group_stats["peak_latencies"]) * 1000:.1f} ms',
                 transform=ax2.transAxes, verticalalignment='top', 
                 horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                 fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No peak latencies detected\nin analysis window', 
                 transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('Peak Latency Analysis', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{protocol}_{group_name}_publication_figure.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Publication figure saved: {output_path}")
    return output_path


def create_group_comparison_figure(comparison_results: Dict[str, Any], 
                                 group1_stats: Dict[str, Any], group2_stats: Dict[str, Any],
                                 output_dir: str, protocol: str) -> str:
    """
    Create publication-quality figure comparing two groups.
    
    Args:
        comparison_results (Dict[str, Any]): Comparison statistics
        group1_stats (Dict[str, Any]): Group 1 statistics
        group2_stats (Dict[str, Any]): Group 2 statistics
        output_dir (str): Output directory
        protocol (str): Protocol name
        
    Returns:
        str: Path to saved figure
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    times = group1_stats['times']
    
    # Panel 1: Group means comparison
    ax1.fill_between(times, group1_stats['ci_lower'], group1_stats['ci_upper'], 
                     alpha=0.3, color='blue')
    ax1.plot(times, group1_stats['group_mean'], color='blue', linewidth=3, 
             label=f'{comparison_results["group1_name"]} (n={comparison_results["n1"]})')
    
    ax1.fill_between(times, group2_stats['ci_lower'], group2_stats['ci_upper'], 
                     alpha=0.3, color='red')
    ax1.plot(times, group2_stats['group_mean'], color='red', linewidth=3, 
             label=f'{comparison_results["group2_name"]} (n={comparison_results["n2"]})')
    
    ax1.axhline(y=CHANCE_LEVEL, color='gray', linestyle='--', alpha=0.7, 
                label='Chance Level')
    
    # Mark significant differences
    if np.any(comparison_results['significant_points']):
        sig_times = times[comparison_results['significant_points']]
        y_pos = np.max([np.max(group1_stats['group_mean']), 
                       np.max(group2_stats['group_mean'])]) + 0.02
        ax1.scatter(sig_times, [y_pos] * len(sig_times), color='black', 
                   marker='*', s=50, label='Significant Difference')
    
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax1.set_title('Group Comparison - Temporal Decoding', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.set_ylim([0.4, 0.8])
    
    # Panel 2: Effect sizes
    ax2.plot(times, comparison_results['effect_sizes'], color='black', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=0.2, color='green', linestyle='--', alpha=0.7, label='Small Effect')
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Effect')
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Large Effect')
    
    # Mark significant time points
    if np.any(comparison_results['significant_points']):
        sig_effects = comparison_results['effect_sizes'][comparison_results['significant_points']]
        sig_times = times[comparison_results['significant_points']]
        ax2.scatter(sig_times, sig_effects, color='red', s=30, zorder=5)
    
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel("Cohen's d", fontsize=14)
    ax2.set_title('Effect Size Over Time', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Panel 3: Global AUC comparison
    global_data = [group1_stats['subject_means'], group2_stats['subject_means']]
    global_labels = [comparison_results['group1_name'], comparison_results['group2_name']]
    
    bp = ax3.boxplot(global_data, labels=global_labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    
    # Add individual points
    for i, data in enumerate(global_data):
        y = data
        x = np.random.normal(i+1, 0.04, size=len(y))
        ax3.scatter(x, y, alpha=0.6, s=30, color='black')
    
    ax3.axhline(y=CHANCE_LEVEL, color='gray', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Global AUC', fontsize=14)
    ax3.set_title(f'Global Performance Comparison\n'
                  f't={comparison_results["global_t_stat"]:.3f}, '
                  f'p={comparison_results["global_p_value"]:.6f}\n'
                  f"Cohen's d={comparison_results["global_effect_size"]:.3f}", 
                  fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Peak latency comparison
    if (len(group1_stats['peak_latencies']) > 0 and 
        len(group2_stats['peak_latencies']) > 0):
        
        latency_data = [group1_stats['peak_latencies'] * 1000, 
                       group2_stats['peak_latencies'] * 1000]
        
        bp2 = ax4.boxplot(latency_data, labels=global_labels, patch_artist=True)
        bp2['boxes'][0].set_facecolor('blue')
        bp2['boxes'][1].set_facecolor('red')
        
        # Add individual points
        for i, data in enumerate(latency_data):
            y = data
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax4.scatter(x, y, alpha=0.6, s=30, color='black')
        
        ax4.set_ylabel('Peak Latency (ms)', fontsize=14)
        ax4.set_title(f'Peak Latency Comparison\n'
                      f't={comparison_results["peak_t_stat"]:.3f}, '
                      f'p={comparison_results["peak_p_value"]:.6f}', 
                      fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient peak latency data\nfor comparison', 
                 transform=ax4.transAxes, ha='center', va='center', fontsize=14)
        ax4.set_title('Peak Latency Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{protocol}_{comparison_results['group1_name']}_vs_{comparison_results['group2_name']}_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Group comparison figure saved: {output_path}")
    return output_path


def generate_comprehensive_report(all_results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate comprehensive statistical report for all analyses.
    
    Args:
        all_results (Dict[str, Any]): All analysis results
        output_dir (str): Output directory
        
    Returns:
        str: Path to generated report
    """
    report_path = os.path.join(output_dir, "comprehensive_statistical_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EEG DECODING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"User: {getpass.getuser()}\n")
        f.write(f"Analysis Directory: {output_dir}\n\n")
        
        # Summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        total_subjects = 0
        total_protocols = len(all_results)
        
        for protocol, protocol_results in all_results.items():
            f.write(f"\nProtocol: {protocol}\n")
            
            if 'group_stats' in protocol_results:
                for group_name, stats in protocol_results['group_stats'].items():
                    n_subj = stats['n_subjects']
                    total_subjects += n_subj
                    
                    f.write(f"  {group_name}:\n")
                    f.write(f"    - Subjects: {n_subj}\n")
                    f.write(f"    - Mean AUC: {stats['global_auc']:.4f} ± {np.std(stats['subject_means']):.4f}\n")
                    f.write(f"    - Peak AUC: {np.max(stats['group_mean']):.4f}\n")
                    f.write(f"    - Peak Time: {stats['times'][np.argmax(stats['group_mean'])]:.3f}s\n")
                    f.write(f"    - Max Effect Size: {np.max(stats['effect_sizes']):.3f}\n")
                    
                    if len(stats['peak_latencies']) > 0:
                        f.write(f"    - Peak Latency: {np.mean(stats['peak_latencies']) * 1000:.1f} ± "
                               f"{np.std(stats['peak_latencies']) * 1000:.1f} ms\n")
                    
                    # Reliability measures
                    rel = stats['reliability_measures']
                    if not np.isnan(rel['cronbach_alpha']):
                        f.write(f"    - Cronbach's α: {rel['cronbach_alpha']:.3f}\n")
                    if not np.isnan(rel['split_half_corrected']):
                        f.write(f"    - Split-half reliability: {rel['split_half_corrected']:.3f}\n")
        
        f.write(f"\nTotal Subjects Analyzed: {total_subjects}\n")
        f.write(f"Total Protocols: {total_protocols}\n\n")
        
        # Statistical test results
        f.write("STATISTICAL TEST RESULTS\n")
        f.write("-" * 40 + "\n")
        
        for protocol, protocol_results in all_results.items():
            f.write(f"\nProtocol: {protocol}\n")
            
            # Group comparisons
            if 'comparisons' in protocol_results:
                for comp_name, comp_results in protocol_results['comparisons'].items():
                    f.write(f"\n  Comparison: {comp_name}\n")
                    f.write(f"    Global AUC Comparison:\n")
                    f.write(f"      - t-statistic: {comp_results['global_t_stat']:.4f}\n")
                    f.write(f"      - p-value: {comp_results['global_p_value']:.6f}\n")
                    f.write(f"      - Effect size (Cohen's d): {comp_results['global_effect_size']:.4f}\n")
                    f.write(f"      - Mann-Whitney U: {comp_results['global_u_stat']:.4f}\n")
                    f.write(f"      - U p-value: {comp_results['global_u_p']:.6f}\n")
                    
                    # Significant time points
                    n_sig_points = np.sum(comp_results['significant_points'])
                    total_points = len(comp_results['significant_points'])
                    f.write(f"    Time-point Analysis:\n")
                    f.write(f"      - Significant points: {n_sig_points}/{total_points} "
                           f"({n_sig_points/total_points*100:.1f}%)\n")
                    
                    if n_sig_points > 0:
                        times = protocol_results['group_stats'][comp_results['group1_name']]['times']
                        sig_times = times[comp_results['significant_points']]
                        f.write(f"      - First significant: {np.min(sig_times):.3f}s\n")
                        f.write(f"      - Last significant: {np.max(sig_times):.3f}s\n")
                        f.write(f"      - Max effect size: {np.max(np.abs(comp_results['effect_sizes'])):.3f}\n")
                    
                    # Peak latency comparison
                    if not np.isnan(comp_results['peak_t_stat']):
                        f.write(f"    Peak Latency Comparison:\n")
                        f.write(f"      - t-statistic: {comp_results['peak_t_stat']:.4f}\n")
                        f.write(f"      - p-value: {comp_results['peak_p_value']:.6f}\n")
                    
                    # Cluster analysis
                    cluster_results = comp_results['cluster_results']
                    f.write(f"    Cluster Analysis:\n")
                    f.write(f"      - Number of clusters: {cluster_results['n_clusters']}\n")
                    f.write(f"      - Total significant duration: {cluster_results['total_significant_duration']:.3f}s\n")
                    
                    for i, cluster in enumerate(cluster_results['clusters']):
                        f.write(f"      - Cluster {i+1}: {cluster['start_time']:.3f}-{cluster['end_time']:.3f}s "
                               f"(duration: {cluster['duration']:.3f}s)\n")
        
        # Individual group analyses
        f.write("\nINDIVIDUAL GROUP ANALYSES\n")
        f.write("-" * 40 + "\n")
        
        for protocol, protocol_results in all_results.items():
            if 'group_stats' in protocol_results:
                f.write(f"\nProtocol: {protocol}\n")
                
                for group_name, stats in protocol_results['group_stats'].items():
                    f.write(f"\n  Group: {group_name}\n")
                    f.write(f"    Statistical Tests Against Chance Level:\n")
                    
                    # Significant time points
                    n_sig = np.sum(stats['significant_points'])
                    total_points = len(stats['significant_points'])
                    f.write(f"      - Significant time points: {n_sig}/{total_points} "
                           f"({n_sig/total_points*100:.1f}%)\n")
                    
                    if n_sig > 0:
                        sig_times = stats['times'][stats['significant_points']]
                        f.write(f"      - First significant: {np.min(sig_times):.3f}s\n")
                        f.write(f"      - Last significant: {np.max(sig_times):.3f}s\n")
                        f.write(f"      - Peak t-statistic: {np.max(stats['t_stats']):.3f}\n")
                        f.write(f"      - Min corrected p-value: {np.min(stats['p_corrected'][stats['significant_points']]):.6f}\n")
                    
                    # Variability measures
                    var_measures = stats['variability_measures']
                    f.write(f"    Variability Measures:\n")
                    f.write(f"      - Mean CV across time: {var_measures['mean_cv_across_time']:.3f}\n")
                    f.write(f"      - Mean subject CV: {var_measures['mean_subject_cv']:.3f}\n")
                    f.write(f"      - Mean score range: {var_measures['mean_score_range']:.3f}\n")
                    
                    # Cluster analysis
                    cluster_results = stats['cluster_results']
                    f.write(f"    Cluster Analysis:\n")
                    f.write(f"      - Number of clusters: {cluster_results['n_clusters']}\n")
                    f.write(f"      - Total significant duration: {cluster_results['total_significant_duration']:.3f}s\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Comprehensive report generated: {report_path}")
    return report_path


def process_single_protocol(protocol_name: str, protocol_data: Dict[str, List[str]], 
                          output_dir: str) -> Dict[str, Any]:
    """
    Process all groups within a single protocol.
    
    Args:
        protocol_name (str): Name of the protocol
        protocol_data (Dict[str, List[str]]): Group data for the protocol
        output_dir (str): Output directory
        
    Returns:
        Dict[str, Any]: Protocol analysis results
    """
    logger.info(f"Processing protocol: {protocol_name}")
    
    protocol_results = {
        'protocol_name': protocol_name,
        'group_stats': {},
        'comparisons': {},
        'figures': []
    }
    
    # Process each group
    for group_name, file_paths in protocol_data.items():
        logger.info(f"Processing group: {group_name} ({len(file_paths)} subjects)")
        
        # Load all subject data
        all_scores = []
        all_times = None
        subject_ids = []
        
        for file_path in file_paths:
            data = load_npz_data(file_path)
            if data is not None:
                scores = data['scores']
                times = data['times']
                
                # Ensure consistent time vectors
                if all_times is None:
                    all_times = times
                elif not np.array_equal(times, all_times):
                    logger.warning(f"Inconsistent time vectors in {file_path}")
                    continue
                
                all_scores.append(scores)
                subject_ids.append(data['subject_id'])
        
        if len(all_scores) == 0:
            logger.warning(f"No valid data found for group {group_name}")
            continue
        
        # Convert to array (subjects x time points)
        scores_array = np.array(all_scores)
        
        # Calculate comprehensive statistics
        group_stats = calculate_comprehensive_statistics(scores_array, all_times, subject_ids)
        protocol_results['group_stats'][group_name] = group_stats
        
        # Create publication figure
        fig_path = create_publication_figure(group_stats, group_name, output_dir, protocol_name)
        protocol_results['figures'].append(fig_path)
    
    # Perform group comparisons
    group_names = list(protocol_results['group_stats'].keys())
    
    if len(group_names) >= 2:
        logger.info(f"Performing group comparisons for protocol {protocol_name}")
        
        # Pairwise comparisons
        for i in range(len(group_names)):
            for j in range(i+1, len(group_names)):
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                comparison_results = perform_group_comparison(
                    protocol_results['group_stats'][group1_name],
                    protocol_results['group_stats'][group2_name],
                    group1_name, group2_name
                )
                
                comp_name = f"{group1_name}_vs_{group2_name}"
                protocol_results['comparisons'][comp_name] = comparison_results
                
                # Create comparison figure
                fig_path = create_group_comparison_figure(
                    comparison_results,
                    protocol_results['group_stats'][group1_name],
                    protocol_results['group_stats'][group2_name],
                    output_dir, protocol_name
                )
                protocol_results['figures'].append(fig_path)
    
    logger.info(f"Protocol {protocol_name} processing completed")
    return protocol_results


def main():
    """
    Main function to orchestrate the entire analysis pipeline.
    """
    try:
        logger.info("=" * 60)
        logger.info("EEG DECODING GROUP ANALYSIS - ENHANCED VERSION")
        logger.info("=" * 60)
        
        # Get base directory
        current_user = getpass.getuser()
        base_results_path = f"/Users/{current_user}/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG/results"
        
        if not os.path.exists(base_results_path):
            logger.error(f"Base results directory not found: {base_results_path}")
            return
        
        logger.info(f"Base directory: {base_results_path}")
        
        # Find and organize NPZ files
        organized_data = find_npz_files(base_results_path)
        
        if not organized_data:
            logger.error("No NPZ files found!")
            return
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_results_path, f"enhanced_group_analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Process each protocol
        all_results = {}
        
        for protocol_name, protocol_data in organized_data.items():
            if protocol_data:  # Only process if there's data
                protocol_results = process_single_protocol(protocol_name, protocol_data, output_dir)
                all_results[protocol_name] = protocol_results
        
        # Generate comprehensive report
        report_path = generate_comprehensive_report(all_results, output_dir)
        
        # Summary
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Report file: {report_path}")
        
        total_figures = sum(len(results['figures']) for results in all_results.values())
        logger.info(f"Total figures generated: {total_figures}")
        
        total_comparisons = sum(len(results['comparisons']) for results in all_results.values())
        logger.info(f"Total group comparisons: {total_comparisons}")
        
        logger.info("All publication-quality figures and analyses are ready!")
        
    except Exception as e:
        logger.error(f"Fatal error in main analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
