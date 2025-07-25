import os
import sys
import glob
import logging
import warnings
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import re

# --- Configuration du chemin et des imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
baking_eeg_dir = os.path.join(current_dir, '..')
if baking_eeg_dir not in sys.path:
    sys.path.insert(0, baking_eeg_dir)

# Gestion robuste des dépendances optionnelles
try:
    from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
    )
    STATS_UTILS_AVAILABLE = True
except ImportError:
    STATS_UTILS_AVAILABLE = False
    def perform_pointwise_fdr_correction_on_scores(*args, **kwargs):
        # Fonction factice pour éviter un crash si le module est manquant
        return None, np.array([False]), np.array([1.0]), np.array([1.0])

try:
    from config.config import ALL_SUBJECTS_GROUPS
except ImportError:
    ALL_SUBJECTS_GROUPS = None

# --- Configuration Globale ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V17/intra_subject_lg_results"

# --- Constantes et Mappings ---
GROUP_NAME_MAPPING = {
    'COMA': 'Coma', 'CONTROLS': 'Controls', 'DELIRIUM-': 'Delirium -',
    'DELIRIUM+': 'Delirium +', 'MCS': 'MCS', 'VS': 'VS/UWS',
}
GROUP_ORDER = ['CONTROLS', 'DELIRIUM-', 'DELIRIUM+', 'MCS', 'VS', 'COMA']
GROUP_COLORS = {
    'CONTROLS': '#2ca02c', 'DELIRIUM-': '#ff7f0e', 'DELIRIUM+': '#d62728',
    'MCS': '#1f77b4', 'COMA': '#9467bd', 'VS': '#8c564b',
}
CHANCE_LEVEL = 0.5
FDR_ALPHA = 0.05

# === FENÊTRE TEMPORELLE COMMUNE POUR TOUTE L'ANALYSE ===
COMMON_T_MIN = -200  # ms
COMMON_T_MAX = 800   # ms

# Configuration des graphiques
PUBLICATION_PARAMS = {
    'figure.figsize': (16, 9), 'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 20,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14, 'lines.linewidth': 2.5,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1
}
plt.rcParams.update(PUBLICATION_PARAMS)

# Configuration du Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not STATS_UTILS_AVAILABLE:
    logger.warning("Module 'utils.stats_utils' non trouvé. Certaines fonctions statistiques (FDR) seront désactivées.")
if not ALL_SUBJECTS_GROUPS:
    logger.warning("Fichier 'config.config' non trouvé. Le filtrage des sujets par groupe est désactivé.")


# --- Fonctions Utilitaires ---
def extract_subject_id_from_path(file_path):
    if ALL_SUBJECTS_GROUPS:
        try:
            all_ids = set().union(*ALL_SUBJECTS_GROUPS.values())
            for sid in sorted(list(all_ids), key=len, reverse=True):
                if sid in file_path: return sid
        except Exception: pass
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if '_Subj_' in part: return part.split('_Subj_')[1].split('_')[0]
        if 'Subject_' in part: return part.split('Subject_')[1].split('_')[0]
    return os.path.basename(os.path.dirname(file_path))

def find_npz_files(base_path):
    logger.info("Recherche des fichiers NPZ dans: %s", base_path)
    organized_data = {}
    all_files = glob.glob(os.path.join(base_path, '**', '*decoding_results_full.npz'), recursive=True)
    if not all_files:
        logger.warning("Aucun fichier de résultats NPZ trouvé.")
        return {}
    logger.info("%d fichiers de résultats potentiels trouvés.", len(all_files))
    folder_to_group = {
        'COMA_LG_COMA': 'COMA', 'CONTROLS_DELIRIUM_LG_CONTROLS': 'CONTROLS',
        'DELIRIUM-_LG_PATIENTS_DELIRIUM-': 'DELIRIUM-', 'DELIRIUM+_LG_PATIENTS_DELIRIUM+': 'DELIRIUM+',
        'MCS_LG_MCS': 'MCS', 'VS_LG_VS': 'VS',
    }
    for file_path in all_files:
        try:
            path_parts = file_path.split(os.sep)
            group_name, protocol_name = None, None
            for i in range(len(path_parts) - 1, 0, -1):
                folder = path_parts[i]
                if folder in folder_to_group:
                    group_name, protocol_name = folder_to_group[folder], folder
                    break
            if not group_name: continue
            if protocol_name not in organized_data: organized_data[protocol_name] = {}
            if group_name not in organized_data[protocol_name]: organized_data[protocol_name][group_name] = []
            organized_data[protocol_name][group_name].append(file_path)
        except Exception as e:
            logger.warning(f"Erreur lors du traitement du chemin {file_path}: {e}")
    return organized_data

def load_npz_data(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            data_keys = list(data.keys())
            if not any(key.startswith('lg_') for key in data_keys) or 'epochs_time_points' not in data_keys:
                return None
            result = {'subject_id': extract_subject_id_from_path(file_path), 'file_path': file_path, 'times': data['epochs_time_points']}
            for effect in ['local', 'global']:
                prefix = 'lg_ls_ld' if effect == 'local' else 'lg_gs_gd'
                if f'{prefix}_scores_1d_mean' in data_keys:
                    result[f'{effect}_effect_scores'] = data[f'{prefix}_scores_1d_mean']
                    if f'{prefix}_tgm_mean' in data_keys: result[f'{effect}_effect_tgm'] = data[f'{prefix}_tgm_mean']
                    if f'{prefix}_mean_auc_global' in data_keys: result[f'{effect}_effect_auc_global'] = data[f'{prefix}_mean_auc_global']
            if 'local_effect_scores' not in result and 'global_effect_scores' not in result: return None
            return result
    except Exception as e:
        logger.error("Erreur lors du chargement du fichier NPZ %s: %s", file_path, e)
        return None

def filter_group_files_by_config(group_files, group_name):
    if not ALL_SUBJECTS_GROUPS: return group_files
    if group_name not in ALL_SUBJECTS_GROUPS:
        logger.warning(f"Groupe '{group_name}' non trouvé dans la config. Aucun fichier conservé.")
        return []
    allowed_ids = set(ALL_SUBJECTS_GROUPS.get(group_name, []))
    if not allowed_ids: return []
    filtered_files = [fp for fp in group_files if extract_subject_id_from_path(fp) in allowed_ids]
    logger.info(f"Groupe {group_name}: {len(filtered_files)}/{len(group_files)} fichiers conservés après filtrage.")
    return filtered_files


# --- FONCTION CENTRALE CORRIGÉE ---
def analyze_group_data_lg(group_files, group_name):
    """
    Analyse les données d'un groupe en rognant chaque sujet à une fenêtre temporelle
    commune de [-200, 800] ms pour standardiser l'analyse.
    """
    logger.info(f"Analyse LG du groupe {group_name} sur la fenêtre commune [{COMMON_T_MIN}, {COMMON_T_MAX}] ms.")
    
    group_data_list, subject_ids = [], []
    processed_local_scores, processed_global_scores = [], []
    processed_local_tgms, processed_global_tgms = [], []
    local_auc_global_list, global_auc_global_list = [], []
    
    common_time_vector = None

    for file_path in group_files:
        data = load_npz_data(file_path)
        if data is None: continue

        subject_id, raw_times = data['subject_id'], data['times']
        if raw_times is None or len(raw_times) < 2:
            logger.warning(f"Sujet {subject_id} ignoré : données temporelles invalides.")
            continue
            
        times_ms = raw_times * 1000 if np.max(raw_times) < 100 else raw_times
        common_indices = np.where((times_ms >= COMMON_T_MIN) & (times_ms <= COMMON_T_MAX))[0]
        
        if len(common_indices) == 0:
            logger.warning(f"Sujet {subject_id} ignoré : aucune donnée dans la fenêtre commune [{COMMON_T_MIN}, {COMMON_T_MAX}] ms.")
            continue
            
        cropped_times = times_ms[common_indices]
        if common_time_vector is None:
            common_time_vector = cropped_times
            logger.info(f"Vecteur temporel commun défini avec {len(common_time_vector)} points de {cropped_times[0]:.1f} à {cropped_times[-1]:.1f} ms.")
        elif not np.array_equal(common_time_vector, cropped_times):
            logger.error(f"Erreur critique: Le sujet {subject_id} a un vecteur temporel différent après rognage. Vérifiez les sfreq.")
            continue

        subject_ids.append(subject_id)
        group_data_list.append(data)

        for effect in ['local', 'global']:
            scores_key, tgm_key, auc_key = f'{effect}_effect_scores', f'{effect}_effect_tgm', f'{effect}_effect_auc_global'
            if scores_key in data and data[scores_key] is not None:
                if len(data[scores_key]) >= np.max(common_indices) + 1:
                    scores_list = processed_local_scores if effect == 'local' else processed_global_scores
                    scores_list.append(np.nan_to_num(data[scores_key][common_indices], nan=CHANCE_LEVEL))
                    if tgm_key in data and data[tgm_key] is not None:
                        tgm_list = processed_local_tgms if effect == 'local' else processed_global_tgms
                        tgm_list.append(data[tgm_key][np.ix_(common_indices, common_indices)])
                    if auc_key in data:
                        auc_list = local_auc_global_list if effect == 'local' else global_auc_global_list
                        auc_list.append(data[auc_key])

    if not group_data_list:
        logger.warning(f"Aucune donnée valide pour le groupe {group_name} après filtrage temporel.")
        return {}

    result = {'group_name': group_name, 'n_subjects': len(subject_ids), 'subject_ids': subject_ids,
              'times': common_time_vector, 'group_data': group_data_list}

    if processed_local_scores:
        local_matrix = np.array(processed_local_scores)
        result['local_effect'] = {'scores_matrix': local_matrix, 'group_mean': np.nanmean(local_matrix, axis=0),
                                  'group_sem': np.nanstd(local_matrix, axis=0) / np.sqrt(local_matrix.shape[0]),
                                  'auc_global_values': np.array(local_auc_global_list)}
        if processed_local_tgms:
            result['local_effect']['tgm_matrix'] = np.array(processed_local_tgms)
            result['local_effect']['tgm_mean'] = np.nanmean(result['local_effect']['tgm_matrix'], axis=0)

    if processed_global_scores:
        global_matrix = np.array(processed_global_scores)
        result['global_effect'] = {'scores_matrix': global_matrix, 'group_mean': np.nanmean(global_matrix, axis=0),
                                   'group_sem': np.nanstd(global_matrix, axis=0) / np.sqrt(global_matrix.shape[0]),
                                   'auc_global_values': np.array(global_auc_global_list)}
        if processed_global_tgms:
            result['global_effect']['tgm_matrix'] = np.array(processed_global_tgms)
            result['global_effect']['tgm_mean'] = np.nanmean(result['global_effect']['tgm_matrix'], axis=0)

    logger.info(f"Groupe {group_name} analysé - Local: {len(processed_local_scores)}, Global: {len(processed_global_scores)} sujets.")
    return result


# --- Fonctions d'Affichage et d'Analyse ---

def plot_all_groups_comparison(all_groups_data, save_dir, show_plots=True):
    """
    Compare les courbes de décodage de tous les groupes sur un même graphique.
    Affiche les clusters de significativité pour chaque groupe sous forme de barres
    horizontales empilées et colorées au bas du graphique.
    """
    if not all_groups_data: 
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
    
    # --- Configuration pour les barres de significativité empilées ---
    SIGNIFICANCE_AREA_TOP = 0.44 
    BAR_HEIGHT = 0.008
    BAR_GAP = 0.004
    # -------------------------------------------------------------------
    
    for ax, effect in zip(axes, ['local', 'global']):
        title = f'Effet {effect.capitalize()} - Comparaison des Groupes'
        
        # 1. Dessiner les courbes de données en premier
        for group_idx, group_name in enumerate(GROUP_ORDER):
            group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            color = GROUP_COLORS.get(group_name, 'grey')
            
            if group_data and f'{effect}_effect' in group_data:
                data = group_data[f'{effect}_effect']
                times_ms = group_data['times']
                label = f'{mapped_name} (n={group_data["n_subjects"]})'
                ax.plot(times_ms, data['group_mean'], color=color, linewidth=2.5, label=label)
                ax.fill_between(times_ms, data['group_mean'] - data['group_sem'], 
                               data['group_mean'] + data['group_sem'], color=color, alpha=0.2)
                # --- Ajout : calcul et affichage des deux pics principaux avec espacement minimal ---
                try:
                    from scipy.signal import find_peaks
                    curve = data['group_mean']
                    peaks, _ = find_peaks(curve)
                    if len(peaks) < 1:
                        main_peaks = [np.argmax(curve)]
                    else:
                        peak_heights = curve[peaks]
                        first_idx = np.argmax(peak_heights)
                        first_peak = peaks[first_idx]
                        min_dist_ms = 80
                        ms_per_idx = np.mean(np.diff(times_ms))
                        min_dist_idx = int(min_dist_ms / ms_per_idx)
                        # Filtrer les pics dont la latence est > 0 ms
                        distant_peaks = [p for p in peaks if abs(p - first_peak) >= min_dist_idx and times_ms[p] > 0]
                        if times_ms[first_peak] > 0:
                            if distant_peaks:
                                distant_heights = curve[distant_peaks]
                                second_peak = distant_peaks[np.argmax(distant_heights)]
                                main_peaks = [first_peak, second_peak]
                            else:
                                main_peaks = [first_peak]
                        else:
                            # Si le premier pic n'est pas > 0 ms, chercher le prochain pic > 0 ms
                            valid_peaks = [p for p in peaks if times_ms[p] > 0]
                            if valid_peaks:
                                valid_heights = curve[valid_peaks]
                                first_valid_idx = np.argmax(valid_heights)
                                first_valid_peak = valid_peaks[first_valid_idx]
                                distant_peaks = [p for p in valid_peaks if abs(p - first_valid_peak) >= min_dist_idx]
                                if distant_peaks:
                                    distant_heights = curve[distant_peaks]
                                    second_peak = distant_peaks[np.argmax(distant_heights)]
                                    main_peaks = [first_valid_peak, second_peak]
                                else:
                                    main_peaks = [first_valid_peak]
                            else:
                                main_peaks = []
                    for i, peak_idx in enumerate(main_peaks):
                        ax.plot(times_ms[peak_idx], curve[peak_idx], 'o', color=color, markersize=12, markeredgecolor='black', label=None)
                        ax.annotate(f"Pic {i+1}: {int(times_ms[peak_idx])}ms", (times_ms[peak_idx], curve[peak_idx]),
                                    textcoords="offset points", xytext=(0,10+i*18), ha='center', color=color, fontsize=7, fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.7))
                except Exception as e:
                    logger.warning(f"Erreur lors de la détection des pics pour {group_name} ({effect}): {e}")
            else:
                # Afficher dans la légende même si pas de données
                ax.plot([], [], color=color, linewidth=2.5, label=f'{mapped_name} (n=0)')

        # 2. Dessiner les barres de significativité empilées avec les couleurs des groupes
        for group_idx, group_name in enumerate(GROUP_ORDER):
            group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
            color = GROUP_COLORS.get(group_name, 'grey')
            
            if group_data and f'{effect}_effect' in group_data:
                data = group_data[f'{effect}_effect']
                times_ms = group_data['times']
                
                try:
                    scores_matrix = data['scores_matrix']
                    # Utiliser la fonction importée directement
                    _, clusters, cluster_p, _ = perform_cluster_permutation_test(
                        scores_matrix, 
                        chance_level=CHANCE_LEVEL+0.0125, 
                        n_permutations=10000,
                        alternative_hypothesis="greater", 
                        n_jobs=-1
                    )
                    # Calculer la position verticale de la barre pour ce groupe
                    y_top = SIGNIFICANCE_AREA_TOP - (group_idx * (BAR_HEIGHT + BAR_GAP))
                    y_bottom = y_top - BAR_HEIGHT
                    # Dessiner les clusters significatifs avec la couleur du groupe
                    for clu, pval in zip(clusters, cluster_p):
                        if pval < 0.05:
                            ax.fill_between(times_ms, y_bottom, y_top,
                                            where=clu, 
                                            color=color, 
                                            alpha=0.9, 
                                            step='mid',
                                           )
                except Exception as e:
                    logger.warning(f"Cluster permutation test a échoué pour {group_name} ({effect}): {e}")

        # 3. Configuration finale des axes
        ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.6, label='Chance (0.5)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.4)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel('Temps (ms)', fontsize=14)
        ax.set_ylabel('Score AUC', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, linestyle=':', linewidth=0.6)
        
        # Ajuster la limite Y pour inclure les barres de significativité
        num_groups = len(GROUP_ORDER)
        y_min_limit = SIGNIFICANCE_AREA_TOP - (num_groups * (BAR_HEIGHT + BAR_GAP))
        ax.set_ylim([y_min_limit, 0.70])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "all_groups_comparison_stacked_colored_clusters.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique de comparaison sauvegardé : {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_temporal_windows_comparison_boxplots(all_groups_data, save_dir, show_plots=True):
    windows_local = {'(100-120ms)': (100, 120), '(180-200ms)': (180, 200), '(240-260ms)': (240, 260), f'Global (0-{COMMON_T_MAX}ms)': (0, COMMON_T_MAX)}
    windows_global = {'(125-135ms)': (125, 135), '(142-146ms)': (142, 146), '(155-160ms)': (155, 160), '(310-325ms)': (310, 325), '(550-600ms)': (550, 600), f'Global (0-{COMMON_T_MAX}ms)': (0, COMMON_T_MAX)}
    logger.info("Création des boxplots par fenêtres temporelles.")

    for effect_type in ['local', 'global']:
        all_subjects_data = []
        windows = windows_local if effect_type == 'local' else windows_global
        for group_data in all_groups_data:
            effect_key, group_name = f'{effect_type}_effect', group_data['group_name']
            if effect_key not in group_data: continue
            scores_matrix, times_ms = group_data[effect_key]['scores_matrix'], group_data['times']
            for subj_idx, subject_scores in enumerate(scores_matrix):
                subject_id = group_data['subject_ids'][subj_idx]
                for window_name, (start_ms, end_ms) in windows.items():
                    indices = np.where((times_ms >= start_ms) & (times_ms <= end_ms))[0]
                    if len(indices) > 0:
                        all_subjects_data.append({'Group': GROUP_NAME_MAPPING.get(group_name, group_name), 'Subject': subject_id,
                                                  'Window': window_name, 'AUC': np.nanmean(subject_scores[indices])})
        if not all_subjects_data: continue

        df = pd.DataFrame(all_subjects_data)
        ordered_groups = [GROUP_NAME_MAPPING.get(g, g) for g in GROUP_ORDER]
        group_palette = {name: GROUP_COLORS.get(key, '#FFF') for key, name in GROUP_NAME_MAPPING.items()}

        plt.figure(figsize=(16, 9))
        ax = plt.gca()
        sns.boxplot(data=df, x='Window', y='AUC', hue='Group', order=windows.keys(), hue_order=ordered_groups, palette=group_palette, showfliers=False, ax=ax)
        sns.stripplot(data=df, x='Window', y='AUC', hue='Group', order=windows.keys(), hue_order=ordered_groups,
                      dodge=True, jitter=True, marker='o', alpha=0.7, color='black', size=5, legend=False, ax=ax)



        plt.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', label='Chance Level')
        plt.title(f'AUC par Fenêtre Temporelle - Effet {effect_type.capitalize()}', fontsize=18)
        plt.xlabel('Fenêtre Temporelle', fontsize=14); plt.ylabel('Score AUC Moyen', fontsize=14)
        # Adapter la limite Y pour chaque effet
        if effect_type == 'global':
            plt.ylim(0.4, 0.7)
        else:
            plt.ylim(0.4, 0.8)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        handles, labels = plt.gca().get_legend_handles_labels()
        group_counts = df.groupby('Group')['Subject'].nunique()
        legend_labels = [f"{g} (n={group_counts.get(g, 0)})" for g in ordered_groups]
        plt.legend(handles[:len(ordered_groups)], legend_labels, title='Groupe Clinique', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_dir:
            filepath = os.path.join(save_dir, f"boxplot_auc_windows_{effect_type}.png")
            plt.savefig(filepath, dpi=300)
            logger.info(f"Boxplot des fenêtres temporelles sauvegardé : {filepath}")
        if show_plots: plt.show()
        else: plt.close()

def plot_group_tgm_individual(all_groups_data, save_dir, show_plots=True):
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        if group_data['n_subjects'] < 2: continue

        for effect_type in ['local', 'global']:
            effect_key = f'{effect_type}_effect'
            if effect_key not in group_data or 'tgm_mean' not in group_data[effect_key]: continue
            
            tgm_mean, tgm_matrix, times_ms = group_data[effect_key]['tgm_mean'], group_data[effect_key].get('tgm_matrix'), group_data['times']
            
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(tgm_mean, origin='lower', aspect='auto', extent=[times_ms[0], times_ms[-1], times_ms[0], times_ms[-1]], vmin=0.4, vmax=0.62, cmap='RdYlBu_r')
            
            if tgm_matrix is not None and STATS_UTILS_AVAILABLE:
                _, significant_mask, _, _ = perform_pointwise_fdr_correction_on_scores(tgm_matrix, chance_level=CHANCE_LEVEL, alpha_significance_level=FDR_ALPHA,alternative_hypothesis="two-sided")
                if np.any(significant_mask): ax.contour(times_ms, times_ms, significant_mask, levels=[0.5], colors='black', linewidths=2)

            ax.plot([times_ms[0], times_ms[-1]], [times_ms[0], times_ms[-1]], 'k--', alpha=0.5)
            ax.axhline(0, color='black', alpha=0.3); ax.axvline(0, color='black', alpha=0.3)
            ax.set_xlabel('Test Time (ms)'); ax.set_ylabel('Train Time (ms)')
            ax.set_title(f"{GROUP_NAME_MAPPING.get(group_name, group_name)} - {effect_type.capitalize()} TGM (n={group_data['n_subjects']})")
            cbar = plt.colorbar(im, ax=ax); cbar.set_label('AUC Score')
            plt.tight_layout()
            
            if save_dir:
                filepath = os.path.join(save_dir, f"tgm_{effect_type}_group_{group_name}.png")
                plt.savefig(filepath, dpi=300)
                logger.info(f"TGM sauvegardée : {filepath}")
            if show_plots: plt.show()
            else: plt.close()

def add_significance_bars(ax, groups_data, stats_results, y_positions, bar_height=0.02):
    """
    Ajouter des barres de significativité sur un graphique.
    
    Args:
        ax: Axe matplotlib
        groups_data: Données des groupes
        stats_results: Résultats des tests statistiques
        y_positions: Positions y pour les barres
        bar_height: Hauteur des barres
    """
    if not stats_results.get('pairwise_results'):
        return
    
    group_names = stats_results['groups_names']
    pairwise_results = stats_results['pairwise_results']
    
    # Trouver les positions x des groupes
    x_positions = {group_name: i for i, group_name in enumerate(group_names)}
    
    bar_y = max(y_positions) + bar_height
    
    for comparison_key, result in pairwise_results.items():
        # Utiliser la p-value FDR corrigée si disponible, sinon la p-value brute
        p_value = result.get('mannwhitney_pvalue_fdr', result.get('mannwhitney_pvalue', 1.0))
        
        if np.isnan(p_value):
            continue
        
        # Déterminer le niveau de significativité
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            continue  # Ne pas afficher de barre si non significatif
        
        # Extraire les noms des groupes
        group1, group2 = comparison_key.split('_vs_')
        
        if group1 in x_positions and group2 in x_positions:
            x1, x2 = x_positions[group1], x_positions[group2]
            
            # Dessiner la barre horizontale
            ax.plot([x1, x2], [bar_y, bar_y], 'k-', linewidth=1)
            # Dessiner les barres verticales
            ax.plot([x1, x1], [bar_y - bar_height/2, bar_y + bar_height/2], 'k-', linewidth=1)
            ax.plot([x2, x2], [bar_y - bar_height/2, bar_y + bar_height/2], 'k-', linewidth=1)
            
            # Ajouter le texte de significativité
            ax.text((x1 + x2) / 2, bar_y + bar_height/2, significance, 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            bar_y += bar_height * 3  # Espacement entre les barres


def analyze_individual_significance_proportions(all_groups_data, save_dir, show_plots=True):
    """
    Analyse et visualise la significativité des scores AUC pour chaque sujet et chaque groupe.

    Cette fonction génère deux types de graphiques pour les effets 'locaux' et 'globaux' :
    1. Des graphiques en secteurs montrant la proportion de sujets avec un AUC supérieur au niveau de chance.
    2. Des graphiques à barres pour chaque groupe, affichant le score AUC de chaque sujet
       avec une ligne indiquant le niveau de chance pour identifier les sujets significatifs.

    Args:
        all_groups_data (list): Une liste de dictionnaires, chaque dictionnaire contenant les données d'un groupe.
        save_dir (str): Le répertoire où sauvegarder les graphiques générés.
        show_plots (bool): Si True, affiche les graphiques.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Répertoire créé : {save_dir}")

    SIGNIFICANCE_THRESHOLD = CHANCE_LEVEL + 0.0125
    for effect_type in ['local', 'global']:
        groups_analysis = []
        # --- Boucle pour générer les graphiques à barres individuels ---
        for group_data in all_groups_data:
            effect_key = f'{effect_type}_effect'
            if effect_key not in group_data or 'auc_global_values' not in group_data[effect_key]:
                continue

            auc_values = group_data[effect_key]['auc_global_values']
            valid_aucs = auc_values[~np.isnan(auc_values)]
            if len(valid_aucs) == 0:
                continue

            # Préparation des données pour le graphique à barres
            subject_ids = np.arange(len(valid_aucs))
            colors = ['#4ecdc4' if x > SIGNIFICANCE_THRESHOLD else '#ff6b6b' for x in valid_aucs]

            # Création du graphique à barres pour le groupe actuel
            plt.figure(figsize=(12, 7))
            plt.bar(subject_ids, valid_aucs, color=colors, label='Score AUC par sujet')
            plt.axhline(y=SIGNIFICANCE_THRESHOLD, color='r', linestyle='--', label=f'Seuil de significativité ({SIGNIFICANCE_THRESHOLD:.4f})')

            plt.xlabel('ID du Sujet')
            plt.ylabel('Valeur AUC')
            group_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
            plt.title(f'Scores AUC Individuels - {group_name} - Effet {effect_type.capitalize()}')
            plt.legend()
            plt.tight_layout()

            # Sauvegarde du graphique à barres
            if save_dir:
                # Créer des sous-dossiers pour mieux organiser les fichiers
                bar_charts_dir = os.path.join(save_dir, 'bar_charts', effect_type)
                if not os.path.exists(bar_charts_dir):
                    os.makedirs(bar_charts_dir)
                filepath = os.path.join(bar_charts_dir, f"bar_scores_{group_data['group_name']}_{effect_type}.png")
                plt.savefig(filepath, dpi=300)
                logger.info(f"Graphique à barres sauvegardé : {filepath}")

            if show_plots:
                plt.show()
            else:
                plt.close()

            # Collecte des données pour le graphique en secteurs
            significant_count = np.sum(valid_aucs > SIGNIFICANCE_THRESHOLD)
            groups_analysis.append({
                'group_name': group_data['group_name'],
                'n_subjects': len(valid_aucs),
                'n_significant': significant_count
            })

        # --- Création du graphique en secteurs (camembert) comme auparavant ---
        if not groups_analysis:
            continue
        # Trier les analyses selon GROUP_ORDER
        ordered_analysis = []
        for group in GROUP_ORDER:
            for analysis in groups_analysis:
                if analysis['group_name'] == group:
                    ordered_analysis.append(analysis)
        # Si certains groupes ne sont pas dans GROUP_ORDER, les ajouter à la fin
        for analysis in groups_analysis:
            if analysis['group_name'] not in GROUP_ORDER:
                ordered_analysis.append(analysis)
        fig, axes = plt.subplots(1, len(ordered_analysis), figsize=(4 * len(ordered_analysis), 5), squeeze=False)
        for i, analysis in enumerate(ordered_analysis):
            sizes = [analysis['n_significant'], analysis['n_subjects'] - analysis['n_significant']]
            labels = [f'> {CHANCE_LEVEL}', f'<= {CHANCE_LEVEL}']
            pie_colors = ['#4ecdc4', '#ff6b6b']
            axes[0, i].pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
            axes[0, i].set_title(f"{GROUP_NAME_MAPPING.get(analysis['group_name'], analysis['group_name'])}\n(n={analysis['n_subjects']})")
        fig.suptitle(f'Proportion de Sujets Significatifs - Effet {effect_type.capitalize()}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_dir:
            # Créer un sous-dossier pour les graphiques en secteurs
            pie_charts_dir = os.path.join(save_dir, 'pie_charts')
            if not os.path.exists(pie_charts_dir):
                os.makedirs(pie_charts_dir)
            
            filepath = os.path.join(pie_charts_dir, f"significance_proportions_{effect_type}.png")
            plt.savefig(filepath, dpi=300)
            logger.info(f"Graphique des proportions sauvegardé : {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
def perform_statistical_tests(all_groups_data, effect_type='local'):
    """
    Effectuer des tests Mann-Whitney U entre les groupes pour les métriques globales AUC.
    
    Args:
        all_groups_data: Liste des données de tous les groupes
        effect_type: 'local' ou 'global' pour spécifier quel effet analyser
        
    Returns:
        Dictionnaire avec les résultats des tests statistiques
    """
    groups_auc_data = {}
    groups_names = []
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        effect_key = f'{effect_type}_effect'
        if effect_key in group_data and 'auc_global_values' in group_data[effect_key]:
            auc_values = group_data[effect_key]['auc_global_values']
            auc_values = auc_values[~np.isnan(auc_values)]
            if len(auc_values) > 0:
                groups_auc_data[group_name] = auc_values
                groups_names.append(group_name)
    if len(groups_auc_data) < 2:
        logger.warning(f"Pas assez de groupes avec des données AUC pour l'effet {effect_type}")
        return {}
    # Tests Mann-Whitney U par paires
    pairwise_results = {}
    group_names = list(groups_auc_data.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group1, group2 = group_names[i], group_names[j]
            data1, data2 = groups_auc_data[group1], groups_auc_data[group2]
            try:
                u_stat, p_mannwhitney = mannwhitneyu(data1, data2, alternative='two-sided')
            except Exception as e:
                logger.error(f"Erreur dans le test Mann-Whitney entre {group1} et {group2}: {e}")
                u_stat, p_mannwhitney = np.nan, np.nan
            pairwise_results[f"{group1}_vs_{group2}"] = {
                'mannwhitney_stat': u_stat,
                'mannwhitney_pvalue': p_mannwhitney,
                'group1_mean': np.mean(data1),
                'group2_mean': np.mean(data2),
                'group1_std': np.std(data1),
                'group2_std': np.std(data2),
                'group1_n': len(data1),
                'group2_n': len(data2)
            }
    # Correction FDR pour les tests multiples
    if pairwise_results:
        pvalues_mannwhitney = [result['mannwhitney_pvalue'] for result in pairwise_results.values() if not np.isnan(result['mannwhitney_pvalue'])]
        if pvalues_mannwhitney:
            _, pvalues_mannwhitney_corrected = fdrcorrection(pvalues_mannwhitney)
            mannwhitney_idx = 0
            for key, result in pairwise_results.items():
                if not np.isnan(result['mannwhitney_pvalue']):
                    result['mannwhitney_pvalue_fdr'] = pvalues_mannwhitney_corrected[mannwhitney_idx]
                    mannwhitney_idx += 1
                else:
                    result['mannwhitney_pvalue_fdr'] = np.nan
    return {
        'effect_type': effect_type,
        'groups_data': groups_auc_data,
        'groups_names': groups_names,
        'pairwise_results': pairwise_results,
        'n_groups': len(groups_auc_data)
    }


def plot_global_auc_boxplots(all_groups_data, save_dir, show_plots=True):
    """
    Crée des boxplots pour les AUC globaux avec tests statistiques et export des résultats.
    """
    if not all_groups_data:
        logger.warning("Aucune donnée de groupe disponible pour les boxplots")
        return

    for effect_type in ['local', 'global']:
        logger.info(f"Création des boxplots pour l'effet {effect_type}")
        plot_data = []
        group_subject_counts = {}

        # Collecte des données pour chaque groupe
        for group_data in all_groups_data:
            group_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
            effect_key = f'{effect_type}_effect'
            if effect_key in group_data and 'auc_global_values' in group_data[effect_key]:
                auc_values = group_data[effect_key]['auc_global_values']
                auc_values = auc_values[~np.isnan(auc_values)]
                if len(auc_values) > 0:
                    group_subject_counts[group_name] = len(auc_values)
                    for value in auc_values:
                        plot_data.append({'Group': group_name, 'AUC': value})

        if not plot_data:
            logger.warning(f"Pas assez de données pour créer le boxplot de l'effet {effect_type}")
            continue

        df = pd.DataFrame(plot_data)
        ordered_groups = [GROUP_NAME_MAPPING.get(g, g) for g in GROUP_ORDER if GROUP_NAME_MAPPING.get(g, g) in df['Group'].unique()]
        group_palette = {name: GROUP_COLORS.get(key, '#1f77b4') for key, name in GROUP_NAME_MAPPING.items()}

        # Initialiser stats_results AVANT toute utilisation
        stats_results = perform_statistical_tests(all_groups_data, effect_type)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=df, x='Group', y='AUC', order=ordered_groups, ax=ax, palette=group_palette)
        sns.stripplot(data=df, x='Group', y='AUC', order=ordered_groups, ax=ax, color='black', alpha=0.6, size=4, jitter=True)

        ax.set_title(f'Distribution des AUC globaux - Effet {effect_type.capitalize()}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Groupe Clinique', fontsize=14)
        ax.set_ylabel('AUC (Aire sous la courbe)', fontsize=14)
        ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Niveau de chance ({CHANCE_LEVEL})')

        # Légende x avec n sujets
        new_labels = [f"{g} (n={group_subject_counts.get(g, 0)})" for g in ordered_groups]
        ax.set_xticklabels(new_labels, rotation=45, ha='right')

        # Test Wilcoxon vs chance pour chaque groupe (ajouté)
        y_min, y_max = ax.get_ylim()
        y_sig = y_max - 0.04 * (y_max - y_min)
        for idx, group in enumerate(ordered_groups):
            group_aucs = df[df['Group'] == group]['AUC'].values
            group_aucs = group_aucs[~np.isnan(group_aucs)]
            if len(group_aucs) > 1:
                try:
                    stat, pval = wilcoxon(group_aucs - CHANCE_LEVEL, alternative='greater')
                    if pval < 0.001:
                        sig = '***'
                    elif pval < 0.01:
                        sig = '**'
                    elif pval < 0.05:
                        sig = '*'
                    else:
                        sig = ''
                    if sig:
                        ax.text(idx, y_sig, sig, ha='center', va='bottom', fontsize=18, fontweight='bold', color='black')
                except Exception as e:
                    logger.warning(f"Erreur test Wilcoxon pour {group} ({effect_type}): {e}")

        # Affichage des barres de significativité pour les effets global ET local
        if stats_results and stats_results.get('pairwise_results'):
            y_max_auc = df['AUC'].max()
            y_positions = [y_max_auc + 0.05]
            add_significance_bars(ax, df, stats_results, y_positions)

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.02, y_max + 0.15)
        ax.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Barres de comparaison inter-groupes (Mann-Whitney U) au-dessus de chaque paire significative
        if stats_results and 'pairwise_results' in stats_results:
            y_min, y_max = ax.get_ylim()
            # On place les barres au-dessus du max des boxplots
            y_base = y_max + 0.01 * (y_max - y_min)
            bar_height = 0.02 * (y_max - y_min)
            idx_map = {g: i for i, g in enumerate(ordered_groups)}
            # Pour éviter la superposition, on garde une pile de hauteurs par paire
            used_levels = {}
            for key, result in stats_results['pairwise_results'].items():
                g1, g2 = key.split('_vs_')
                if g1 in idx_map and g2 in idx_map:
                    p_corr = result.get('mannwhitney_pvalue_fdr', np.nan)
                    if not np.isnan(p_corr):
                        if p_corr < 0.001:
                            sig = '***'
                        elif p_corr < 0.01:
                            sig = '**'
                        elif p_corr < 0.05:
                            sig = '*'
                        else:
                            sig = ''
                        if sig:
                            x1, x2 = idx_map[g1], idx_map[g2]
                            # Pour chaque paire, on détermine le niveau (empilement) pour éviter la superposition
                            pair = tuple(sorted([x1, x2]))
                            level = used_levels.get(pair, 0)
                            y_bar = y_base + level * (bar_height + 0.01)
                            # Trace la barre horizontale
                            ax.plot([x1, x1, x2, x2], [y_bar, y_bar+bar_height, y_bar+bar_height, y_bar], color='k', linewidth=1.5)
                            ax.text((x1+x2)/2, y_bar+bar_height+0.003, sig, ha='center', va='bottom', fontsize=16, fontweight='bold', color='k')
                            used_levels[pair] = level + 1
            # Ajuste la limite Y pour inclure toutes les barres
            if used_levels:
                max_level = max(used_levels.values())
                ax.set_ylim(y_min, y_base + max_level * (bar_height + 0.03))

        ax.legend(loc='upper right')
        plt.tight_layout()

        if save_dir:
            filename = f"global_auc_boxplot_{effect_type}_effect.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplot sauvegardé: {filepath}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        # Sauvegarde des résultats statistiques
        if save_dir and stats_results:
            stats_filename = f"statistical_results_{effect_type}_effect.json"
            stats_filepath = os.path.join(save_dir, stats_filename)
            stats_to_save = {}
            for key, value in stats_results.items():
                if isinstance(value, np.ndarray):
                    stats_to_save[key] = value.tolist()
                elif isinstance(value, dict):
                    stats_to_save[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            stats_to_save[key][k] = v.tolist()
                        elif isinstance(v, (np.integer, np.floating)):
                            stats_to_save[key][k] = float(v)
                        else:
                            stats_to_save[key][k] = v
                elif isinstance(value, (np.integer, np.floating)):
                    stats_to_save[key] = float(value)
                else:
                    stats_to_save[key] = value
            try:
                with open(stats_filepath, 'w') as f:
                    json.dump(stats_to_save, f, indent=2)
                logger.info(f"Résultats statistiques sauvegardés: {stats_filepath}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde des statistiques: {e}")

def plot_group_individual_curves(group_data, save_dir, show_plots=True):
    """
    Crée des graphiques pour un groupe avec :
    - Les courbes individuelles en arrière-plan.
    - La moyenne du groupe et son erreur standard (SEM) en avant-plan.
    - Des barres empilées colorées en bas du graphique pour indiquer les clusters
      temporels significatifs (p < 0.05) pour chaque sujet individuel.
    """
    group_name = group_data['group_name']
    mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
    times_ms = group_data['times']
    
    if times_ms is None:
        logger.error(f"Pas de données temporelles pour le groupe {group_name}")
        return
    
    group_color = GROUP_COLORS.get(group_name, '#1f77b4')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle(f'Décodage Temporel pour le Groupe : {mapped_name} (n={group_data["n_subjects"]})', 
                 fontsize=22)

    # Configuration pour les barres de significativité empilées
    SIGNIFICANCE_AREA_TOP = 0.37
    BAR_HEIGHT = 0.006
    BAR_GAP = 0.002

    for ax, effect_type in zip(axes, ['local', 'global']):
        effect_key = f'{effect_type}_effect'
        title = f'Effet {effect_type.capitalize()}'
        
        if effect_key in group_data:
            effect_data = group_data[effect_key]
            scores_matrix = effect_data['scores_matrix']
            group_mean = effect_data['group_mean']
            group_sem = effect_data['group_sem']
            n_subjects = scores_matrix.shape[0]

            # 1. Tracer les courbes individuelles
            for i in range(n_subjects):
                ax.plot(times_ms, scores_matrix[i, :], 
                       color=group_color, alpha=0.15, linewidth=1)

            # 2. Tracer la moyenne et l'erreur standard par-dessus
            ax.plot(times_ms, group_mean, color=group_color, alpha=1.0, linewidth=3, 
                    label=f'Moyenne du groupe')
            ax.fill_between(times_ms, group_mean - group_sem, group_mean + group_sem,
                            color=group_color, alpha=0.25)

            # 3. Calculer et dessiner les clusters significatifs pour le GROUPE ENTIER
            if n_subjects >= 2:
                try:
                    _, clusters, cluster_p, _ = perform_cluster_permutation_test(
                        scores_matrix,
                        chance_level=CHANCE_LEVEL,
                        n_permutations=1024,
                        alternative_hypothesis="greater",
                        n_jobs=-1
                    )
                    # Position de la barre de clusters sous la courbe moyenne
                    y_top = SIGNIFICANCE_AREA_TOP - (0 * (BAR_HEIGHT + BAR_GAP))
                    y_bottom = y_top - BAR_HEIGHT
                    for clu, pval in zip(clusters, cluster_p):
                        if pval < 0.05:
                            ax.fill_between(times_ms, y_bottom, y_top,
                                            where=clu,
                                            color=group_color,
                                            alpha=0.8,
                                            step='mid')
                except Exception as e:
                    logger.warning(f"Cluster permutation test groupe échoué ({effect_type}) : {e}")

            # 4. Configuration des axes et légendes
            ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.7, label='Chance')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.4)
            ax.set_xlabel('Temps (ms)', fontsize=14)
            ax.set_ylabel('Score AUC', fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, linestyle=':', linewidth=0.6)
            
            # Ajuster la limite Y pour inclure les barres de significativité individuelles
            y_min_limit = SIGNIFICANCE_AREA_TOP - (n_subjects * (BAR_HEIGHT + BAR_GAP))
            ax.set_ylim([y_min_limit, 0.85])

        else:
            ax.text(0.5, 0.5, 'Aucune donnée pour cet effet', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"group_{group_name.replace('/', '_')}_individual_stacked_clusters.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        logger.info(f"Graphique individuel du groupe sauvegardé : {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def create_temporal_windows_connected_plots(all_groups_data, save_dir, show_plots=True):
    """
    Crée des graphiques avec lignes connectées entre les fenêtres temporelles pour chaque groupe,
    harmonisé avec le style des autres fonctions du script.
    """
    # Fenêtres temporelles d'intérêt (en ms)
    windows_local = {'(100-120ms)': (100, 120), '(180-200ms)': (180, 200), '(240-260ms)': (240, 260), f'Global (0-{COMMON_T_MAX}ms)': (0, COMMON_T_MAX)}
    windows_global = {'(125-135ms)': (125, 135), '(142-146ms)': (142, 146), '(155-160ms)': (155, 160), '(310-325ms)': (310, 325), '(550-600ms)': (550, 600), f'Global (0-{COMMON_T_MAX}ms)': (0, COMMON_T_MAX)}
    window_names_local = list(windows_local.keys())
    window_names_global = list(windows_global.keys())
    x_positions_local = np.arange(len(window_names_local))
    x_positions_global = np.arange(len(window_names_global))

    for effect_type in ['local', 'global']:
        logger.info(f"Création des graphiques connectés pour l'effet {effect_type}")
        windows = windows_local if effect_type == 'local' else windows_global
        window_names = window_names_local if effect_type == 'local' else window_names_global
        x_positions = x_positions_local if effect_type == 'local' else x_positions_global
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            effect_key = f'{effect_type}_effect'
            if effect_key not in group_data:
                logger.warning(f"Pas de données pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            effect_data = group_data[effect_key]
            if 'scores_matrix' not in effect_data:
                logger.warning(f"Pas de matrice de scores pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            scores_matrix = effect_data['scores_matrix']
            times_ms = group_data.get('times')
            n_subjects = scores_matrix.shape[0]
            if times_ms is None:
                logger.warning(f"Pas de données temporelles pour le groupe {group_name}")
                continue
            group_color = GROUP_COLORS.get(group_name, '#1f77b4')

            # Calcul des moyennes par fenêtre pour chaque sujet
            subjects_window_means = []
            for subj_idx in range(n_subjects):
                subject_scores = scores_matrix[subj_idx, :]
                subject_means = []
                for win in window_names:
                    tmin, tmax = windows[win]
                    indices = np.where((times_ms >= tmin) & (times_ms <= tmax))[0]
                    if len(indices) > 0:
                        subject_means.append(np.nanmean(subject_scores[indices]))
                    else:
                        subject_means.append(np.nan)
                subjects_window_means.append(subject_means)
            subjects_window_means = np.array(subjects_window_means)

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 8))
            # Tracer les lignes individuelles
            for subj_means in subjects_window_means:
                valid_idx = ~np.isnan(subj_means)
                if np.sum(valid_idx) > 1:
                    ax.plot(x_positions[valid_idx], np.array(subj_means)[valid_idx], 'o-', color=group_color, alpha=0.3, markersize=7)

            # Moyenne et écart-type du groupe
            group_means = np.nanmean(subjects_window_means, axis=0)
            group_stds = np.nanstd(subjects_window_means, axis=0)
            valid_group_idx = ~np.isnan(group_means)
            if np.sum(valid_group_idx) > 1:
                ax.plot(x_positions[valid_group_idx], group_means[valid_group_idx], 'o-', color='black', alpha=1, linewidth=3, markersize=12, label='Moyenne')
                ax.fill_between(x_positions[valid_group_idx],
                                group_means[valid_group_idx] - group_stds[valid_group_idx],
                                group_means[valid_group_idx] + group_stds[valid_group_idx],
                                color='black', alpha=0.15)

            # Test Wilcoxon vs chance pour chaque fenêtre
            for win_idx, win in enumerate(window_names):
                win_vals = subjects_window_means[:, win_idx]
                valid_vals = win_vals[~np.isnan(win_vals)]
                if len(valid_vals) > 1:
                    try:
                        from scipy.stats import wilcoxon
                        stat, pval = wilcoxon(valid_vals - CHANCE_LEVEL, alternative='two-sided')
                        if pval < 0.001:
                            sig = '***'
                        elif pval < 0.01:
                            sig = '**'
                        elif pval < 0.05:
                            sig = '*'
                        else:
                            sig = ''
                        if sig:
                            ax.text(win_idx, 0.41, sig, ha='center', va='center', fontsize=18, fontweight='bold')
                    except Exception as e:
                        logger.warning(f"Erreur test Wilcoxon pour {mapped_name} {effect_type} {win}: {e}")

            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Niveau de chance')
            ax.set_xlim(-0.5, len(window_names) - 0.5)
            ax.set_ylim(0.4, 0.77)
            ax.set_xticks(x_positions)
            # Affichage des noms de fenêtres en petit pour une meilleure lisibilité
            ax.set_xticklabels(window_names, fontsize=8, rotation=45)
            ax.set_ylabel('Score AUC', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{mapped_name} - {effect_type.capitalize()}\nComparaison des fenêtres temporelles (n={n_subjects})', fontsize=16, fontweight='bold')
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"temporal_windows_connected_{effect_type}_group_{group_name.replace('/', '_')}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique connecté des fenêtres temporelles sauvegardé: {filepath}")
            if show_plots:
                plt.show()
            else:
                plt.close()

    # --- Ajout : graphique multi-groupes pour chaque effet ---
    for effect_type in ['local', 'global']:
        windows = windows_local if effect_type == 'local' else windows_global
        window_names = window_names_local if effect_type == 'local' else window_names_global
        x_positions = x_positions_local if effect_type == 'local' else x_positions_global
        fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(5 * len(GROUP_ORDER), 7), sharey=True)
        for idx, group_name in enumerate(GROUP_ORDER):
            group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
            ax = axes[idx]
            if not group_data or f'{effect_type}_effect' not in group_data:
                ax.set_title(GROUP_NAME_MAPPING.get(group_name, group_name))
                ax.text(0.5, 0.5, 'Aucune donnée', ha='center', va='center', transform=ax.transAxes)
                continue
            effect_data = group_data[f'{effect_type}_effect']
            if 'scores_matrix' not in effect_data:
                ax.set_title(GROUP_NAME_MAPPING.get(group_name, group_name))
                ax.text(0.5, 0.5, 'Aucune donnée', ha='center', va='center', transform=ax.transAxes)
                continue
            scores_matrix = effect_data['scores_matrix']
            times_ms = group_data.get('times')
            n_subjects = scores_matrix.shape[0]
            group_color = GROUP_COLORS.get(group_name, '#1f77b4')
            # Calcul des moyennes par fenêtre pour chaque sujet
            subjects_window_means = []
            for subj_idx in range(n_subjects):
                subject_scores = scores_matrix[subj_idx, :]
                subject_means = []
                for win in window_names:
                    tmin, tmax = windows[win]
                    indices = np.where((times_ms >= tmin) & (times_ms <= tmax))[0]
                    if len(indices) > 0:
                        subject_means.append(np.nanmean(subject_scores[indices]))
                    else:
                        subject_means.append(np.nan)
                subjects_window_means.append(subject_means)
            subjects_window_means = np.array(subjects_window_means)
            # Tracer les points individuels
            for subj_means in subjects_window_means:
                valid_idx = ~np.isnan(subj_means)
                ax.plot(x_positions[valid_idx], np.array(subj_means)[valid_idx], 'o', color=group_color, alpha=0.5, markersize=7)
            # Moyenne et écart-type du groupe
            group_means = np.nanmean(subjects_window_means, axis=0)
            group_stds = np.nanstd(subjects_window_means, axis=0)
            valid_group_idx = ~np.isnan(group_means)
            if np.sum(valid_group_idx) > 1:
                ax.plot(x_positions[valid_group_idx], group_means[valid_group_idx], 'o-', color='black', alpha=1, linewidth=2, markersize=10, label='Moyenne')
                ax.fill_between(x_positions[valid_group_idx],
                                group_means[valid_group_idx] - group_stds[valid_group_idx],
                                group_means[valid_group_idx] + group_stds[valid_group_idx],
                                color='black', alpha=0.12)
            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.set_xticks(x_positions)
            # Affichage des noms de fenêtres en petit pour une meilleure lisibilité
            ax.set_xticklabels(window_names, fontsize=9, rotation=45)
            ax.set_title(f"{GROUP_NAME_MAPPING.get(group_name, group_name)}\n(n={n_subjects})", fontsize=14)
            ax.set_ylim(0.4, 0.77)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f'Points individuels par groupe - {effect_type.capitalize()}', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"temporal_windows_connected_individuals_bygroup_{effect_type}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique points individuels par groupe sauvegardé: {filepath}")
        plt.close()

def display_subject_sampling_frequencies(all_groups_data):
    logger.info("="*80 + "\n===   ANALYSE DÉTAILLÉE DES FRÉQUENCES D'ÉCHANTILLONNAGE (DONNÉES BRUTES)   ===\n" + "="*80)
    overall_sfreq_summary = Counter()
    for group_data in all_groups_data:
        group_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
        logger.info(f"\n--- Groupe : {group_name} (n={group_data['n_subjects']}) ---")
        group_sfreqs = []
        for subject_data in group_data.get('group_data', []):
            subject_id, times = subject_data.get('subject_id', 'ID inconnu'), subject_data.get('times')
            if times is not None and len(times) > 1:
                times_ms = times * 1000 if np.max(times) < 100 else times
                sfreq = round(1000 / np.mean(np.diff(times_ms))) if np.mean(np.diff(times_ms)) > 0 else 0
                group_sfreqs.append(sfreq)
                logger.info(f"  - Sujet: {subject_id:<15} | sfreq: {sfreq} Hz | N_points_bruts: {len(times):<4} | Durée_brute: {times_ms[-1] - times_ms[0]:.0f} ms")
        if group_sfreqs:
            summary_str = ", ".join([f"{freq} Hz ({c} s.)" for freq, c in Counter(group_sfreqs).items()])
            logger.info(f"-> Résumé sfreq du groupe '{group_name}': {summary_str}")
            overall_sfreq_summary.update(group_sfreqs)
    total_subjects = sum(overall_sfreq_summary.values())
    logger.info("\n" + "="*80 + "\n===             RÉSUMÉ GLOBAL DES FRÉQUENCES D'ÉCHANTILLONNAGE             ===\n" + f"Total sujets : {total_subjects}")
    for freq, count in overall_sfreq_summary.items():
        logger.info(f"  - {freq} Hz : {count} sujets ({(count / total_subjects) * 100:.1f}%)")
    logger.info("="*80)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/tom.balay/results/LG_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Les résultats seront sauvegardés dans : {results_dir}")

    organized_data = find_npz_files(BASE_RESULTS_DIR)
    if not organized_data:
        logger.error("Aucun fichier NPZ trouvé. Arrêt.")
        return

    all_groups_data = []
    for protocol_name, groups in organized_data.items():
        for group_name, group_files in groups.items():
            filtered_files = filter_group_files_by_config(group_files, group_name)
            if not filtered_files: continue
            
            group_data = analyze_group_data_lg(filtered_files, group_name)
            if group_data: all_groups_data.append(group_data)
    
    if not all_groups_data:
        logger.error("Aucune donnée de groupe valide n'a pu être chargée.")
        return

    display_subject_sampling_frequencies(all_groups_data)
    plot_global_auc_boxplots(all_groups_data, results_dir, show_plots=False)
    create_temporal_windows_connected_plots(all_groups_data, results_dir, show_plots=False)
    plot_all_groups_comparison(all_groups_data, results_dir, show_plots=False)
    create_temporal_windows_comparison_boxplots(all_groups_data, results_dir, show_plots=False)
    analyze_individual_significance_proportions(all_groups_data, results_dir, show_plots=False)
  #  plot_group_tgm_individual(all_groups_data, results_dir, show_plots=False)
    for group_data in all_groups_data:
        plot_group_individual_curves(group_data, results_dir, show_plots=False)
   
    

    logger.info(f"Analyse terminée. Résultats dans : {results_dir}")

if __name__ == "__main__":
    main()