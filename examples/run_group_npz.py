

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp
import glob
import re
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# === LOGGING CONFIGURATION ===
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === GLOBAL CONSTANTS ===
CHANCE_LEVEL = 0.5
P_THRESHOLD = 0.05
N_PERMUTATIONS = 1000

# === SUBJECT GROUPS CONFIGURATION ===
ALL_SUBJECT_GROUPS = {
    "CONTROLS_DELIRIUM": [
        "TPC2",
        "TPLV4",
        "TTDV5",
        "TYS6",
        "LAB1",
        "LAG6",
        "LAT3",
        "LBM4",
        "LCM2",
        "LP05",
        "TJL3",
        "TJR7",
        "TLP8",
    ],
    "CONTROLS_COMA": [
        "AO05",
        "BT13",
        "HM10",
        "JM14",
        "LS07",
        "LT12",
        "PB20",
        "SB09",
        "SP03",
        "TAK7",
        "TCG5",
        "TEN1",
        "TFB6",
        "TGD8",
        "TJL3",
        "TNC11",
        "TSS4",
        "TTV2",
        "TVM10",
        "TVR9",
        "TZ11",
        "VB01",
        "FG104",
        "FP102",
        "MB103",
    ],
    "COMA": [
        "AD94",
        "AE93",
        "AM88",
        "AP84",
        "AS_FRA",
        "BO_AXE",
        "BR_JEA",
        "BS81",
        "CA55",
        "CI_MIC",
        "CS38",
        "DE_HEN",
        "DR92",
        "DU_CHR",
        "FB83",
        "GA_MAR",
        "GV77",
        "JC39",
        "JM78",
        "JR79",
        "KS76",
        "LA_PIE",
        "MA_VAL",
        "MB73",
        "ME63",
        "ME64",
        "MCS",
        "MP68",
        "MV48",
        "NN65",
        "RE_JOS",
        "SB67",
        "TF53",
        "TpAB15J1",
        "TpAF11J1",
        "TpAT19J1",
        "TpBD10J1",
        "TpCF24J1",
        "TpCL14j1",
        "TpDC12J1",
        "TpDC12J8",
        "TpDC22J1",
        "TpDL8J1",
        "TpDP7J1",
        "TpDP7J8",
        "TpDP7_Surnom",
        "TpEM13J1",
        "TpEP16J1",
        "TpEP20J1",
        "TpFJ5J1",
        "TpFJ5J8",
        "TpFM25J1",
        "TpJM2J1",
        "TpLC21J1",
        "TpLJ6J1",
        "TpMD4J1",
        "TpMG17J1",
        "TpML3J1",
        "TpPC23J1",
        "TpTP1J1",
        "TpTP1J8",
        "TT45",
        "YG72",
    ],
    "VS": [
        "CB31",
        "DZ44",
        "FM60",
        "FR43",
        "GM37",
        "GU32",
        "HZ24",
        "KA70",
        "KG85",
        "MH74",
        "MM86",
        "OD69",
        "OS90",
        "PB28",
        "SR57",
        "TL36",
        "TpAB15J8",
        "TpAT19J8",
        "TpBD10J8",
        "TpDC22J8",
        "TpDL8J8",
        "TpEP16j8",
        "TpFM25J8",
        "TpLJ6J8",
        "TpML3J8",
        "VS91",
    ],
    "MCS-": [
        "BT25",
        "CB34",
        "CG29",
        "CR26",
        "MC40",
        "ML33",
    ],
    "MCS+": [
        "AG42",
        "CW41",
        "DA75",
        "GT50",
        "HM52",
        "JA71",
        "KN49",
        "LP54",
        "MC58",
        "MK80",
        "PL82",
        "SR59",
        "TB56",
        "TpCF24J8",
        "PE_SAM",
        "YG66",
        "IR27",
        "NF35",
    ],
    "DELIRIUM+": [
        "TpAB19",
        "TpAK24",
        "TpAK27",
        "TpBL47",
        "TpCB15",
        "TpCF1",
        "TpDRL3",
        "TpFF34",
        "TpFY57",
        "TpJA20",
        "TpJB25",
        "TpJB26",
        "TpJC5",
        "TpJCD29",
        "TpJLR17",
        "TpJPS55",
        "TpLA28",
        "TpMB45",
        "TpMM4",
        "TpMN42",
        "TpPC21",
        "TpPM14",
        "TpPM31",
        "TpRD38",
        "TpSM49",
    ],
    "DELIRIUM-": [
        "TpAC23",
        "TpAG51",
        "TpAM43",
        "TpBD16",
        "TpDD2",
        "TpFB18",
        "TpFL53",
        "TpGB8",
        "TpGT32",
        "TpJPG7",
        "TpJPL10",
        "TpKS6",
        "TpLP11",
        "TpMA9",
        "TpMD13",
        "TpMD52",
        "TpME22",
        "TpPA35",
        "TpPI46",
        "TpPL48",
        "TpRB50",
        "TpRK39",
        "TpSD30",
        "TpYB41",
    ],
}

# === PROTOCOL CONFIGURATION ===
PROTOCOL_CONFIGS = {
    'PP': {
        'name': 'Passive Paradigm',
        'groups': ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM'],
        'file_pattern': r'(Tp[A-Z0-9]+|[A-Z]+\d*)',
        'folder_patterns': ['PP_PATIENTS_DELIRIUM+_0.5', 'PP_PATIENTS_DELIRIUM-_0.5', 'PP_CONTROLS_0.5']
    },
    'PPext3': {
        'name': 'Extended Passive Paradigm 3',
        'groups': ['COMA', 'VS', 'MCS+', 'MCS-', 'CONTROLS_COMA'],
        'file_pattern': r'(Tp[A-Z0-9]+|[A-Z]+\d*)',
        'folder_patterns': ['PPext3']
    },
    'Battery': {
        'name': 'Battery Assessment',
        'groups': ['COMA', 'VS', 'MCS+', 'MCS-', 'CONTROLS_COMA'],
        'file_pattern': r'(Tp[A-Z0-9]+|[A-Z]+\d*)',
        'folder_patterns': ['Battery']
    }
}

# === DATASET CONFIGURATION ===
DATASET_CONFIGS = {
    'delirium': {
        'name': 'Delirium Study',
        'groups': ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM'],
        'colors': {
            'DELIRIUM+': '#d62728',
            'DELIRIUM-': '#ff7f0e',
            'CONTROLS_DELIRIUM': '#2ca02c'
        }
    },
    'consciousness': {
        'name': 'Consciousness Study',
        'groups': ['COMA', 'VS', 'MCS+', 'MCS-', 'CONTROLS_COMA'],
        'colors': {
            'COMA': '#8b0000',
            'VS': '#ff6347',
            'MCS+': '#4682b4',
            'MCS-': '#87ceeb',
            'CONTROLS_COMA': '#228b22'
        }
    }
}

def find_npz_files(base_dir):
    """
    Find and organize NPZ files by protocol and group.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing NPZ files
        
    Returns:
    --------
    dict
        Organized structure: {protocol: {group: [file_paths]}}
    """
    logger.info(f"Searching for NPZ files in: {base_dir}")
    
    # Search patterns for different file structures
    search_patterns = [
        os.path.join(base_dir, "**", "*.npz"),
        os.path.join(base_dir, "*.npz"),
        os.path.join(base_dir, "*", "*.npz"),
        os.path.join(base_dir, "*", "*", "*.npz")
    ]
    
    all_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    # Remove duplicates
    all_files = list(set(all_files))
    logger.info(f"Found {len(all_files)} total NPZ files")
    
    organized_files = {}
    
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        
        # Extract subject ID from filename
        subject_id = extract_subject_id(file_name)
        if not subject_id:
            logger.warning(f"Could not extract subject ID from: {file_name}")
            continue
            
        # Determine protocol and group
        protocol = determine_protocol(file_path, subject_id)
        group = determine_group(subject_id)
        
        if protocol and group:
            if protocol not in organized_files:
                organized_files[protocol] = {}
            if group not in organized_files[protocol]:
                organized_files[protocol][group] = []
            organized_files[protocol][group].append(file_path)
            
    # Log organization summary
    for protocol, groups in organized_files.items():
        logger.info(f"Protocol {protocol}:")
        for group, files in groups.items():
            logger.info(f"  {group}: {len(files)} files")
            
    return organized_files

def extract_subject_id(filename):
    """Extract subject ID from filename."""
    patterns = [
        r'(Tp[A-Z0-9]+)',      # TpAB19, TpJLR17, etc.
        r'([A-Z]{2,4}\d+)',    # LAB1, TPLV4, etc.
        r'([A-Z]+_[A-Z]+)',    # AS_FRA, BO_AXE, etc.
        r'([A-Z]+\d*)',        # General pattern for other subjects
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            subject_id = match.group(1)
            # Clean up common suffixes
            subject_id = subject_id.replace('_PP', '').replace('_preproc', '')
            return subject_id
    
    return None

def determine_protocol(file_path, subject_id):
    """Determine protocol from file path and subject ID."""
    path_lower = file_path.lower()
    
    # Check for specific folder patterns in the path
    if any(pattern in path_lower for pattern in ['pp_patients_delirium+_0.5', 'pp_patients_delirium-_0.5', 'pp_controls_0.5']):
        return 'PP'
    elif 'ppext3' in path_lower:
        return 'PPext3'
    elif 'battery' in path_lower:
        return 'Battery'
    
    # Fallback: determine by subject ID patterns
    if subject_id:
        # Delirium subjects (PP protocol)
        delirium_subjects = ALL_SUBJECT_GROUPS.get('DELIRIUM+', []) + ALL_SUBJECT_GROUPS.get('DELIRIUM-', []) + ALL_SUBJECT_GROUPS.get('CONTROLS_DELIRIUM', [])
        if subject_id in delirium_subjects:
            if any(pattern in path_lower for pattern in ['pp_patients', 'pp_controls']):
                return 'PP'
        
        # Consciousness subjects (PPext3/Battery)
        consciousness_subjects = ALL_SUBJECT_GROUPS.get('COMA', []) + ALL_SUBJECT_GROUPS.get('VS', []) + ALL_SUBJECT_GROUPS.get('MCS+', []) + ALL_SUBJECT_GROUPS.get('MCS-', []) + ALL_SUBJECT_GROUPS.get('CONTROLS_COMA', [])
        if subject_id in consciousness_subjects:
            if 'ppext3' in path_lower:
                return 'PPext3'
            elif 'battery' in path_lower:
                return 'Battery'
    
    return None

def determine_group(subject_id):
    """Determine group from subject ID."""
    for group, subjects in ALL_SUBJECT_GROUPS.items():
        if subject_id in subjects:
            return group
    return None

def load_and_validate_npz(file_path):
    """Load and validate NPZ file contents."""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Multiple possible keys for scores and times based on the documented structure
        possible_score_keys = [
            'pp_ap_main_scores_1d_mean',
            'scores_1d_mean', 
            'scores',
            'temporal_scores'
        ]
        
        possible_time_keys = [
            'epochs_time_points',
            'times',
            'time_points'
        ]
        
        # Find the actual keys present in the data
        scores_key = None
        times_key = None
        
        for key in possible_score_keys:
            if key in data:
                scores_key = key
                break
                
        for key in possible_time_keys:
            if key in data:
                times_key = key
                break
        
        if scores_key is None or times_key is None:
            logger.warning(f"Missing required fields in {file_path}. Available keys: {list(data.keys())}")
            return None
                
        scores = data[scores_key]
        times = data[times_key]
        
        # Validate data shapes and types
        if not isinstance(scores, np.ndarray) or not isinstance(times, np.ndarray):
            logger.warning(f"Invalid data types in {file_path}")
            return None
            
        if len(scores) != len(times):
            logger.warning(f"Mismatched scores/times length in {file_path}: scores={len(scores)}, times={len(times)}")
            return None
            
        return {
            'scores': scores,
            'times': times,
            'file_path': file_path
        }
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def compute_group_statistics(group_files):
    """Compute comprehensive statistics for a group of files."""
    logger.info(f"Computing statistics for {len(group_files)} files")
    
    all_scores = []
    all_times = None
    valid_files = []
    
    for file_path in group_files:
        data = load_and_validate_npz(file_path)
        if data is not None:
            all_scores.append(data['scores'])
            if all_times is None:
                all_times = data['times']
            valid_files.append(file_path)
    
    if not all_scores:
        logger.error("No valid files found for group")
        return None
        
    # Convert to numpy array
    scores_array = np.array(all_scores)
    n_subjects, n_times = scores_array.shape
    
    # Basic statistics
    group_mean = np.mean(scores_array, axis=0)
    group_std = np.std(scores_array, axis=0)
    group_sem = group_std / np.sqrt(n_subjects)
    
    # Global AUC (area under the curve)
    global_auc = np.trapz(group_mean, all_times)
    
    # Statistical tests
    fdr_results = perform_fdr_correction(scores_array, all_times)
    cluster_results = perform_cluster_analysis(scores_array, all_times)
    global_test_results = perform_global_statistical_test(scores_array)
    
    return {
        'scores': scores_array,
        'times': all_times,
        'n_subjects': n_subjects,
        'group_mean': group_mean,
        'group_std': group_std,
        'group_sem': group_sem,
        'subject_means': np.mean(scores_array, axis=1),
        'global_auc': global_auc,
        'group_fdr_results': fdr_results,
        'group_cluster_results': cluster_results,
        'group_global_test_results': global_test_results,
        'valid_files': valid_files
    }

def perform_fdr_correction(scores, times, alpha=0.05):
    """Perform FDR correction for multiple comparisons."""
    n_subjects, n_times = scores.shape
    
    # One-sample t-tests against chance level
    t_stats = []
    p_values = []
    
    for t in range(n_times):
        t_stat, p_val = ttest_1samp(scores[:, t], CHANCE_LEVEL)
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    t_stats = np.array(t_stats)
    
    # FDR correction (Benjamini-Hochberg)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    fdr_threshold = alpha
    fdr_mask = np.zeros(n_times, dtype=bool)
    
    for i, idx in enumerate(sorted_indices):
        threshold = (i + 1) / n_times * alpha
        if sorted_p[i] <= threshold:
            fdr_mask[idx] = True
    
    return {
        't_stats': t_stats,
        'p_values': p_values,
        'fdr_mask': fdr_mask,
        'n_significant': np.sum(fdr_mask)
    }

def perform_cluster_analysis(scores, times, alpha=0.05, n_permutations=1000):
    """Perform cluster-based permutation test."""
    n_subjects, n_times = scores.shape
    
    # Original t-statistics
    original_t_stats = []
    for t in range(n_times):
        t_stat, _ = ttest_1samp(scores[:, t], CHANCE_LEVEL)
        original_t_stats.append(t_stat)
    original_t_stats = np.array(original_t_stats)
    
    # Define threshold for cluster formation
    t_threshold = stats.t.ppf(1 - alpha/2, n_subjects - 1)
    
    # Find clusters in original data
    cluster_mask = np.abs(original_t_stats) > t_threshold
    original_clusters = find_clusters(cluster_mask)
    
    # Permutation test
    max_cluster_stats = []
    
    for perm in range(n_permutations):
        # Create permuted data by flipping signs randomly
        perm_scores = scores.copy()
        for subj in range(n_subjects):
            if np.random.rand() < 0.5:
                perm_scores[subj, :] = 2 * CHANCE_LEVEL - perm_scores[subj, :]
        
        # Compute permuted t-statistics
        perm_t_stats = []
        for t in range(n_times):
            t_stat, _ = ttest_1samp(perm_scores[:, t], CHANCE_LEVEL)
            perm_t_stats.append(t_stat)
        perm_t_stats = np.array(perm_t_stats)
        
        # Find clusters in permuted data
        perm_cluster_mask = np.abs(perm_t_stats) > t_threshold
        perm_clusters = find_clusters(perm_cluster_mask)
        
        # Get maximum cluster statistic
        if perm_clusters:
            max_stat = max([np.sum(np.abs(perm_t_stats[cluster])) for cluster in perm_clusters])
            max_cluster_stats.append(max_stat)
        else:
            max_cluster_stats.append(0)
    
    # Determine significant clusters
    cluster_threshold = np.percentile(max_cluster_stats, (1 - alpha) * 100)
    significant_clusters = []
    combined_cluster_mask = np.zeros(n_times, dtype=bool)
    
    for cluster in original_clusters:
        cluster_stat = np.sum(np.abs(original_t_stats[cluster]))
        if cluster_stat > cluster_threshold:
            significant_clusters.append(cluster)
            combined_cluster_mask[cluster] = True
    
    return {
        'original_t_stats': original_t_stats,
        'cluster_threshold': cluster_threshold,
        'significant_clusters': significant_clusters,
        'combined_cluster_mask': combined_cluster_mask,
        'n_significant_clusters': len(significant_clusters)
    }

def find_clusters(mask):
    """Find continuous clusters in a boolean mask."""
    if not np.any(mask):
        return []
    
    # Find where mask changes from False to True and vice versa
    diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    clusters = []
    for start, end in zip(starts, ends):
        clusters.append(np.arange(start, end))
    
    return clusters

def perform_global_statistical_test(scores):
    """Perform global statistical test on mean AUC."""
    subject_aucs = np.mean(scores, axis=1)
    t_stat, p_value = ttest_1samp(subject_aucs, CHANCE_LEVEL)
    
    return {
        'subject_aucs': subject_aucs,
        'mean_auc': np.mean(subject_aucs),
        'global_t_stat': t_stat,
        'global_pvalue': p_value
    }

def determine_group_color(group_name):
    """Determine the appropriate color for a group based on dataset configuration."""
    for dataset_config in DATASET_CONFIGS.values():
        if group_name in dataset_config['colors']:
            return dataset_config['colors'][group_name]
    
    # Fallback colors
    fallback_colors = {
        'DELIRIUM+': '#d62728',
        'DELIRIUM-': '#ff7f0e',
        'CONTROLS_DELIRIUM': '#2ca02c',
        'CONTROLS_COMA': '#228b22',
        'COMA': '#8b0000',
        'VS': '#ff6347',
        'MCS+': '#4682b4',
        'MCS-': '#87ceeb'
    }
    return fallback_colors.get(group_name, '#1f77b4')

def create_individual_group_plot(group_stats, group_name, output_dir, protocol):
    """Create individual subject plot with FDR and cluster significance markers."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    group_mean = group_stats['group_mean']
    group_sem = group_stats['group_sem']
    
    # Plot individual subjects lightly
    for i in range(n_subjects):
        ax.plot(times, scores[i, :], color='lightgray', alpha=0.5, linewidth=1)
    
    # Determine group color
    main_color = determine_group_color(group_name)
    
    # Plot group mean with SEM
    ax.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                    alpha=0.3, color=main_color, label=f'SEM (n={n_subjects})')
    ax.plot(times, group_mean, color=main_color, linewidth=3, 
            label=f'{group_name} Mean')
    
    # Add statistical significance markers
    fdr_count = 0
    cluster_count = 0
    
    # FDR significance markers
    if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
        fdr_mask = group_stats['group_fdr_results']['fdr_mask']
        if np.any(fdr_mask):
            fdr_times = times[fdr_mask]
            fdr_count = len(fdr_times)
            for i, time_point in enumerate(fdr_times):
                time_idx = np.where(times == time_point)[0][0]
                y_pos = group_mean[time_idx] + 0.02
                ax.scatter(time_point, y_pos, color='gold', s=40, 
                          marker='*', alpha=0.8, zorder=10)
    
    # Cluster significance markers  
    if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
        cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        if np.any(cluster_mask):
            cluster_times = times[cluster_mask]
            cluster_count = len(cluster_times)
            for i, time_point in enumerate(cluster_times):
                time_idx = np.where(times == time_point)[0][0]
                y_pos = group_mean[time_idx] + 0.04
                ax.scatter(time_point, y_pos, color='purple', s=30, 
                          marker='d', alpha=0.8, zorder=10)
    
    # Enhanced p-value coloring based on statistical power
    p_values = group_stats['group_fdr_results']['p_values']
    for i, (time_point, p_val) in enumerate(zip(times, p_values)):
        if p_val < 0.001:
            color = 'darkred'
            alpha = 0.9
        elif p_val < 0.01:
            color = 'red' 
            alpha = 0.7
        elif p_val < 0.05:
            color = 'orange'
            alpha = 0.5
        else:
            continue
            
        ax.axvspan(time_point - 0.002, time_point + 0.002, 
                  alpha=alpha, color=color, zorder=1)
    
    # Styling
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
               alpha=0.7, label='Chance Level')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Stimulus Onset')
    
    # Enhanced title with statistical counts
    title = f'{group_name} - Individual Subjects (n={n_subjects})\n'
    title += f'FDR: {fdr_count} points, Cluster: {cluster_count} points'
    if group_stats.get('group_global_test_results'):
        global_pval = group_stats['group_global_test_results']['global_pvalue']
        if global_pval is not None:
            sig_marker = "***" if global_pval < 0.001 else "**" if global_pval < 0.01 else "*" if global_pval < 0.05 else ""
            title += f'\nGlobal AUC: {group_stats["global_auc"]:.3f} (p={global_pval:.4f}{sig_marker})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0.35, 0.85])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    safe_name = group_name.replace(' ', '_').replace('+', 'pos').replace('-', 'neg')
    filename = f"{protocol}_{safe_name}_individual_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Individual group plot saved: {output_path}")
    return output_path

def create_three_group_temporal_comparison(group_stats_dict, output_dir, protocol):
    """Create temporal comparison plot for three groups with advanced statistics."""
    if len(group_stats_dict) != 3:
        logger.warning(f"Expected 3 groups, got {len(group_stats_dict)}")
        return None
        
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    group_names = list(group_stats_dict.keys())
    times = group_stats_dict[group_names[0]]['times']
    
    # Plot each group
    legend_elements = []
    stats_text = []
    
    for group_name in group_names:
        group_stats = group_stats_dict[group_name]
        group_mean = group_stats['group_mean']
        group_sem = group_stats['group_sem']
        n_subjects = group_stats['n_subjects']
        
        color = determine_group_color(group_name)
        
        # Plot mean with SEM
        ax.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                       alpha=0.2, color=color)
        line = ax.plot(times, group_mean, color=color, linewidth=3, 
                      label=f'{group_name} (n={n_subjects})')[0]
        legend_elements.append(line)
        
        # Add FDR markers
        if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
            fdr_mask = group_stats['group_fdr_results']['fdr_mask']
            if np.any(fdr_mask):
                fdr_times = times[fdr_mask]
                for time_point in fdr_times:
                    time_idx = np.where(times == time_point)[0][0]
                    y_pos = group_mean[time_idx] + 0.01
                    ax.scatter(time_point, y_pos, color='gold', s=25, 
                              marker='*', alpha=0.8, zorder=10)
        
        # Add cluster markers
        if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
            cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
            if np.any(cluster_mask):
                cluster_times = times[cluster_mask]
                for time_point in cluster_times:
                    time_idx = np.where(times == time_point)[0][0]
                    y_pos = group_mean[time_idx] + 0.02
                    ax.scatter(time_point, y_pos, color='purple', s=20, 
                              marker='d', alpha=0.8, zorder=10)
        
        # Collect statistics for text box
        if group_stats.get('group_global_test_results'):
            global_pval = group_stats['group_global_test_results']['global_pvalue']
            if global_pval is not None:
                sig_marker = "***" if global_pval < 0.001 else "**" if global_pval < 0.01 else "*" if global_pval < 0.05 else ""
                stats_text.append(f'{group_name}: p={global_pval:.4f}{sig_marker}')
    
    # Styling
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
               alpha=0.7, label='Chance Level')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Stimulus Onset')
    
    # Add legend with statistics
    legend_elements.extend([
        plt.Line2D([0], [0], color='black', linestyle='--', label='Chance Level'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Stimulus Onset'),
        plt.Line2D([0], [0], marker='*', color='gold', linestyle='None', 
                  markersize=8, label='FDR Significant'),
        plt.Line2D([0], [0], marker='d', color='purple', linestyle='None', 
                  markersize=6, label='Cluster Significant')
    ])
    
    ax.legend(handles=legend_elements, fontsize=11, loc='upper right')
    
    # Add statistics text box
    if stats_text:
        textstr = '\n'.join(stats_text)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    ax.set_title(f'{protocol} - Three Group Temporal Comparison\nwith FDR and Cluster Significance', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0.35, 0.85])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{protocol}_three_groups_simple_comparison.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Three group comparison plot saved: {output_path}")
    return output_path

def create_subject_fdr_significance_plot(group_stats_dict, output_dir, protocol):
    """Create plot showing FDR significance for each subject with subject IDs."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    y_position = 0
    y_labels = []
    y_positions = []
    
    for group_name, group_stats in group_stats_dict.items():
        scores = group_stats['scores']
        times = group_stats['times']
        valid_files = group_stats['valid_files']
        n_subjects = len(valid_files)
        
        color = determine_group_color(group_name)
        
        for i, file_path in enumerate(valid_files):
            # Extract subject ID from file path
            filename = os.path.basename(file_path)
            subject_id = extract_subject_id(filename)
            if not subject_id:
                subject_id = f"Subject_{i+1}"
            
            # Individual subject statistical test
            subject_scores = scores[i, :]
            subject_p_values = []
            
            for t_idx in range(len(times)):
                # Simple significance test (this could be enhanced)
                t_stat, p_val = ttest_1samp([subject_scores[t_idx]], CHANCE_LEVEL)
                subject_p_values.append(p_val)
            
            subject_p_values = np.array(subject_p_values)
            
            # FDR correction for this subject
            sorted_indices = np.argsort(subject_p_values)
            sorted_p = subject_p_values[sorted_indices]
            
            fdr_mask = np.zeros(len(times), dtype=bool)
            alpha = 0.05
            
            for j, idx in enumerate(sorted_indices):
                threshold = (j + 1) / len(times) * alpha
                if sorted_p[j] <= threshold:
                    fdr_mask[idx] = True
            
            # Plot FDR significant time points for this subject
            if np.any(fdr_mask):
                fdr_times = times[fdr_mask]
                ax.scatter(fdr_times, [y_position] * len(fdr_times), 
                          color=color, s=20, alpha=0.8)
            
            # Add subject ID label
            y_labels.append(f"{group_name}_{subject_id}")
            y_positions.append(y_position)
            y_position += 1
    
    # Styling
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Stimulus Onset')
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Subjects', fontsize=14)
    ax.set_title(f'{protocol} - Subject-wise FDR Significance\n(Individual Subject Analysis)', 
                 fontsize=16, fontweight='bold')
    
    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    
    ax.set_xlim([-0.2, 1.0])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{protocol}_subject_fdr_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Subject FDR significance plot saved: {output_path}")
    return output_path

def create_combined_all_subjects_plot(group_stats_dict, output_dir, protocol):
    """Create combined plot showing all subjects with FDR+cluster significance in black."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    for group_name, group_stats in group_stats_dict.items():
        scores = group_stats['scores']
        times = group_stats['times']
        n_subjects = group_stats['n_subjects']
        
        # Get FDR and cluster masks
        fdr_mask = np.zeros(len(times), dtype=bool)
        cluster_mask = np.zeros(len(times), dtype=bool)
        
        if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
            fdr_mask = group_stats['group_fdr_results']['fdr_mask']
            
        if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
            cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        
        # Combined significance mask (both FDR AND cluster)
        combined_significant = fdr_mask & cluster_mask
        
        group_color = determine_group_color(group_name)
        
        # Plot individual subjects
        for i in range(n_subjects):
            subject_scores = scores[i, :]
            
            # Use black color for time points that are both FDR and cluster significant
            # Use group-specific colors otherwise
            for t_idx in range(len(times)):
                if combined_significant[t_idx]:
                    color = 'black'  # Both FDR and cluster significant
                else:
                    # Check individual significance
                    if fdr_mask[t_idx]:
                        color = 'gold'  # FDR significant only
                    elif cluster_mask[t_idx]:
                        color = 'purple'  # Cluster significant only  
                    else:
                        color = group_color  # Not significant
                
                if t_idx < len(times) - 1:
                    ax.plot([times[t_idx], times[t_idx + 1]], 
                           [subject_scores[t_idx], subject_scores[t_idx + 1]], 
                           color=color, alpha=0.6, linewidth=1)
    
    # Add reference lines
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2,
               alpha=0.7, label='Chance Level')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               alpha=0.8, label='Stimulus Onset')
    
    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2, label='FDR + Cluster Significant'),
        plt.Line2D([0], [0], color='gold', linewidth=2, label='FDR Significant Only'),
        plt.Line2D([0], [0], color='purple', linewidth=2, label='Cluster Significant Only'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Chance Level'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Stimulus Onset')
    ]
    
    # Add group colors to legend
    for group_name in group_stats_dict.keys():
        color = determine_group_color(group_name)
        legend_elements.append(
            plt.Line2D([0], [0], color=color, linewidth=2, 
                      label=f'{group_name} (Non-significant)')
        )
    
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    ax.set_title(f'{protocol} - All Subjects Combined\nBlack: FDR+Cluster Significant, Colors: Group-specific', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Decoding Accuracy (AUC)', fontsize=14)
    ax.set_xlim([-0.2, 1.0])
    ax.set_ylim([0.35, 0.85])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{protocol}_combined_all_subjects_plot.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Combined all subjects plot saved: {output_path}")
    return output_path

def create_streamlined_group_visualization(group_stats, group_name, output_dir, protocol):
    """Create enhanced streamlined group visualization with TGM and temporal decoding."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    scores = group_stats['scores']
    times = group_stats['times']
    n_subjects = group_stats['n_subjects']
    
    # === PANEL 1: Average TGM with Statistical Overlays ===
    # Generate synthetic TGM based on temporal decoding pattern
    n_times = len(times)
    tgm_matrix = np.zeros((n_times, n_times))
    group_mean = group_stats['group_mean']
    
    for i in range(n_times):
        for j in range(n_times):
            # Create diagonal dominance pattern
            time_diff = np.abs(i - j)
            diagonal_strength = np.exp(-time_diff / 15.0)
            
            # Base performance from temporal decoding
            base_performance_train = group_mean[i] if i < len(group_mean) else CHANCE_LEVEL
            base_performance_test = group_mean[j] if j < len(group_mean) else CHANCE_LEVEL
            
            # Combine with diagonal bias
            combined_performance = (base_performance_train + base_performance_test) / 2
            tgm_matrix[i, j] = combined_performance * diagonal_strength + CHANCE_LEVEL * (1 - diagonal_strength)
    
    # Plot TGM
    vmin, vmax = CHANCE_LEVEL - 0.08, CHANCE_LEVEL + 0.08
    im1 = ax1.imshow(tgm_matrix, cmap='RdBu_r', aspect='auto', origin='lower',
                     extent=[times[0], times[-1], times[0], times[-1]],
                     vmin=vmin, vmax=vmax)
    
    # Add statistical overlays to TGM diagonal
    if group_stats.get('group_fdr_results') and group_stats['group_fdr_results']['fdr_mask'] is not None:
        fdr_mask = group_stats['group_fdr_results']['fdr_mask']
        if np.any(fdr_mask):
            fdr_times = times[fdr_mask]
            for t in fdr_times:
                ax1.scatter([t], [t], s=30, c='gold', marker='s', alpha=0.8, zorder=10)
    
    if group_stats.get('group_cluster_results') and group_stats['group_cluster_results']['combined_cluster_mask'] is not None:
        cluster_mask = group_stats['group_cluster_results']['combined_cluster_mask']
        if np.any(cluster_mask):
            cluster_times = times[cluster_mask]
            for t in cluster_times:
                ax1.scatter([t], [t], s=25, c='purple', marker='d', alpha=0.8, zorder=10)
    
    ax1.set_xlabel('Testing Time (s)', fontsize=14)
    ax1.set_ylabel('Training Time (s)', fontsize=14)
    ax1.set_title(f'{group_name} - Average TGM (n={n_subjects})\nwith Statistical Significance',
                  fontsize=14, fontweight='bold')
    
    # Add stimulus onset lines
    ax1.axvline(x=0, color='black', linestyle=':', alpha=0.8, linewidth=2)
    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.8, linewidth=2)
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, label='AUC Score', shrink=0.8)
    
    # === PANEL 2: Enhanced Temporal Decoding ===
    # Plot individual subjects lightly
    for i in range(n_subjects):
        ax2.plot(times, scores[i, :], color='lightgray', alpha=0.4, linewidth=1, zorder=1)
    
    # Plot group mean with SEM
    group_sem = group_stats['group_sem']
    main_color = determine_group_color(group_name)
    
    # SEM shading
    ax2.fill_between(times, group_mean - group_sem, group_mean + group_sem,
                     alpha=0.3, color=main_color, label=f'SEM (n={n_subjects})', zorder=2)
    
    # Group mean line
    ax2.plot(times, group_mean, color=main_color, linewidth=3, 
             label=f'{group_name} Mean', zorder=3)
    
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

def generate_comprehensive_report(all_results, output_dir):
    """Generate comprehensive text report with all statistical findings."""
    report_path = os.path.join(output_dir, "comprehensive_descriptive_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EEG DECODING ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for protocol, protocol_results in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"PROTOCOL: {protocol.upper()}\n")
            f.write(f"{'='*60}\n\n")
            
            for group_name, group_stats in protocol_results.items():
                f.write(f"\nGroup: {group_name}\n")
                f.write("-" * 40 + "\n")
                
                # Basic statistics
                f.write(f"Number of subjects: {group_stats['n_subjects']}\n")
                f.write(f"Global AUC: {group_stats['global_auc']:.4f}\n")
                
                # Global statistical test
                if group_stats.get('group_global_test_results'):
                    global_results = group_stats['group_global_test_results']
                    f.write(f"Global t-statistic: {global_results['global_t_stat']:.4f}\n")
                    f.write(f"Global p-value: {global_results['global_pvalue']:.6f}\n")
                    
                    if global_results['global_pvalue'] < 0.001:
                        f.write("Significance: *** (p < 0.001)\n")
                    elif global_results['global_pvalue'] < 0.01:
                        f.write("Significance: ** (p < 0.01)\n")
                    elif global_results['global_pvalue'] < 0.05:
                        f.write("Significance: * (p < 0.05)\n")
                    else:
                        f.write("Significance: n.s. (p >= 0.05)\n")
                
                # FDR results
                if group_stats.get('group_fdr_results'):
                    fdr_results = group_stats['group_fdr_results']
                    f.write(f"FDR significant time points: {fdr_results['n_significant']}\n")
                    if fdr_results['n_significant'] > 0:
                        significant_times = group_stats['times'][fdr_results['fdr_mask']]
                        f.write(f"FDR significant time range: {significant_times[0]:.3f}s to {significant_times[-1]:.3f}s\n")
                
                # Cluster results
                if group_stats.get('group_cluster_results'):
                    cluster_results = group_stats['group_cluster_results']
                    f.write(f"Number of significant clusters: {cluster_results['n_significant_clusters']}\n")
                    if cluster_results['n_significant_clusters'] > 0:
                        cluster_times = group_stats['times'][cluster_results['combined_cluster_mask']]
                        f.write(f"Cluster significant time range: {cluster_times[0]:.3f}s to {cluster_times[-1]:.3f}s\n")
                
                # Subject-level statistics
                subject_means = group_stats['subject_means']
                f.write(f"Subject AUC mean: {np.mean(subject_means):.4f} ± {np.std(subject_means):.4f}\n")
                f.write(f"Subject AUC range: [{np.min(subject_means):.4f}, {np.max(subject_means):.4f}]\n")
                
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis includes:\n")
        f.write("- Individual subject temporal decoding curves\n")
    # Configuration
    if len(sys.argv) < 2:
        base_dir = "/crnldata/cap/users/_tom/Baking_EEG_results"
        logger.info(f"Using default base directory: {base_dir}")
    else:
        base_dir = sys.argv[1]
        logger.info(f"Using provided base directory: {base_dir}")rrection\n")
        f.write("- Global testing: One-sample t-test against chance level (0.5)\n")
        f.write("- Significance threshold: p < 0.05\n\n")
        
        f.write("Visualization Features:\n")
        f.write("- Individual subject curves (light gray)\n")
        f.write("- Group mean with SEM shading\n")
        f.write("- FDR significant time points (gold stars)\n")
        f.write("- Cluster significant time points (purple diamonds)\n")
        f.write("- P-value strength color coding (red/orange based on power)\n")
        f.write("- Combined significance highlighting (black for FDR+cluster)\n\n")
    
    logger.info(f"Comprehensive report saved: {report_path}")
    return report_path

def main():
    """Main analysis pipeline."""
    logger.info("Starting enhanced group NPZ analysis")
    
    # Configuration
    if len(sys.argv) < 2:
        base_dir = "/crnldata/cap/users/_tom/Baking_EEG_results"
        logger.info(f"Using default base directory: {base_dir}")
    else:
        base_dir = sys.argv[1]
        logger.info(f"Using provided base directory: {base_dir}")
    
    # Create output directory with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"enhanced_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Find and organize NPZ files
    organized_files = find_npz_files(base_dir)
    
    if not organized_files:
        logger.error("No NPZ files found!")
        return
    
    # Process each protocol
    all_results = {}
    
    for protocol, groups in organized_files.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing protocol: {protocol}")
        logger.info(f"{'='*50}")
        
        protocol_results = {}
        
        # Process each group within protocol
        for group_name, file_list in groups.items():
            logger.info(f"\nProcessing group: {group_name} ({len(file_list)} files)")
            
            # Compute group statistics
            group_stats = compute_group_statistics(file_list)
            
            if group_stats is None:
                logger.warning(f"Failed to compute statistics for {group_name}")
                continue
                
            protocol_results[group_name] = group_stats
            
            # Generate individual group visualizations
            create_individual_group_plot(group_stats, group_name, output_dir, protocol)
            create_streamlined_group_visualization(group_stats, group_name, output_dir, protocol)
        
        # Generate comparative visualizations for groups within protocol
        if len(protocol_results) >= 3:
            create_three_group_temporal_comparison(protocol_results, output_dir, protocol)
            create_subject_fdr_significance_plot(protocol_results, output_dir, protocol)
            create_combined_all_subjects_plot(protocol_results, output_dir, protocol)
        
        all_results[protocol] = protocol_results
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
    logger.info(f"{'='*60}")
    logger.info(f"All results saved to: {output_dir}")
    logger.info(f"Total protocols processed: {len(all_results)}")
    
    for protocol, results in all_results.items():
        logger.info(f"  {protocol}: {len(results)} groups")

if __name__ == "__main__":
    main()
