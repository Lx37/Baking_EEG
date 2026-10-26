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

        return None, np.array([False]), np.array([1.0]), np.array([1.0])

try:
    from config.config import ALL_SUBJECTS_GROUPS
except ImportError:
    ALL_SUBJECTS_GROUPS = None

# --- Configuration Globale ---
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from utils.time_alignment import build_common_time_grid, resample_to_grid
from utils.permutation_stats import (
    load_permutation_results, attach_permutation_results,
    proportion_statistics, proportion_report, benjamini_hochberg, stars,
)


def get_default_results_dir():
    """Racine des resultats individuels (plus de chemin macOS code en dur)."""
    env_dir = os.environ.get("BAKING_EEG_RESULTS_DIR")
    if env_dir:
        return env_dir
    try:
        from getpass import getuser
        from utils.utils import configure_project_paths
        _, base_output = configure_project_paths(getuser())
        return base_output
    except Exception as exc:
        logger.warning("configure_project_paths a echoue (%s) : "
                       "utilisation du repertoire courant.", exc)
        return os.getcwd()


def get_default_output_dir():
    """Dossier de sortie des figures de groupe LG."""
    env_dir = os.environ.get("BAKING_EEG_GROUP_OUTPUT_DIR")
    if env_dir:
        return env_dir
    return os.path.join(get_default_results_dir(), "group_analysis")

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
    """Range les NPZ LG par groupe, d'apres l'identifiant sujet.

    Le groupe est lu dans ALL_SUBJECTS_GROUPS, comme pour l'analyse PP.
    L'ancienne version cherchait des dossiers nommes '<GROUPE>_LG_<...>'
    (ex. 'COMA_LG_COMA') : le code de decodage actuel ecrit dans
    intra_subject_lg_results/<SUJET>/..., donc plus aucun fichier n'etait
    reconnu, et un groupe absent de la table (CONTROLS) etait ignore en silence.
    Quand un sujet a ete relance, seul le NPZ le plus recent est garde.
    """
    logger.info("Recherche des fichiers NPZ dans: %s", base_path)
    # Uniquement les NPZ LG : '*decoding_results_full.npz' ramassait aussi les
    # NPZ PP, et quand le PP d'un sujet etait plus recent, son NPZ LG etait
    # ecarte comme 'ancienne version' -> sujet perdu.
    all_files = glob.glob(os.path.join(base_path, '**', 'lg_decoding_results_full.npz'), recursive=True)
    if not all_files:
        logger.warning("Aucun fichier de résultats NPZ trouvé.")
        return {}
    logger.info("%d fichiers de résultats potentiels trouvés.", len(all_files))

    subject_to_group = {s: g for g, ids in ALL_SUBJECTS_GROUPS.items() for s in ids}
    organized = {}
    seen = set()
    for file_path in sorted(all_files, key=os.path.getmtime, reverse=True):
        parts = file_path.split(os.sep)
        subject_id = next((p for p in reversed(parts) if p in subject_to_group), None)
        subject_id = subject_id or extract_subject_id_from_path(file_path)
        group_name = subject_to_group.get(subject_id)
        if group_name is None:
            logger.warning(f"Sujet '{subject_id}' absent de ALL_SUBJECTS_GROUPS, ignoré : {file_path}")
            continue
        if subject_id in seen:
            logger.info(f"Ancienne version ignorée pour {subject_id} : {file_path}")
            continue
        seen.add(subject_id)
        organized.setdefault(group_name, []).append(file_path)
    logger.info("Sujets retenus par groupe : %s", {g: len(f) for g, f in organized.items()})
    return {'LG': organized} if organized else {}

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
                    if f'{prefix}_cv_global_scores' in data_keys: result[f'{effect}_effect_cv_scores'] = data[f'{prefix}_cv_global_scores']
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



def analyze_group_data_lg(group_files, group_name):
    """Agrege les resultats LG d'un groupe sur une grille temporelle commune.

    Corrections par rapport aux versions <= 2026-01 :

    1. Les sujets dont la grille temporelle differait apres rognage etaient
       purement et simplement ecartes ("vecteur temporel different, Verifiez
       les sfreq"). Ils sont maintenant reinterpoles sur une grille commune
       (cf. utils.time_alignment), donc conserves.
    2. `np.nan_to_num(..., nan=CHANCE_LEVEL)` remplacait les NaN par 0.5, ce
       qui tire la moyenne de groupe vers le niveau de chance au lieu de
       traiter la valeur comme manquante. Les NaN sont desormais propages et
       ignores par les nanmean/nanstd.
    3. `auc_global_values` et `cv_scores_list` n'etaient alimentees que
       lorsque la cle existait, alors que `subject_ids` l'etait toujours :
       les AUC finissaient attribuees au mauvais sujet. Les listes sont
       maintenant strictement paralleles (np.nan / None si absent).
    """
    logger.info(
        "Analyse LG du groupe %s sur la fenetre commune [%s, %s] ms.",
        group_name, COMMON_T_MIN, COMMON_T_MAX)

    group_data_list, subject_ids, subject_times = [], [], []
    raw = {
        'local': {'scores': [], 'tgm': [], 'auc': [], 'cv': []},
        'global': {'scores': [], 'tgm': [], 'auc': [], 'cv': []},
    }

    for file_path in group_files:
        data = load_npz_data(file_path)
        if data is None:
            continue

        subject_id, raw_times = data['subject_id'], data['times']
        if raw_times is None or len(raw_times) < 2:
            logger.warning(
                "Sujet %s ignore : donnees temporelles invalides.", subject_id)
            continue

        times_ms = np.asarray(raw_times, dtype=float)
        times_ms = times_ms * 1000 if np.max(times_ms) < 100 else times_ms
        in_window = (times_ms >= COMMON_T_MIN) & (times_ms <= COMMON_T_MAX)
        if not np.any(in_window):
            logger.warning(
                "Sujet %s ignore : aucune donnee dans la fenetre [%s, %s] ms.",
                subject_id, COMMON_T_MIN, COMMON_T_MAX)
            continue

        has_any_effect = False
        for effect in ('local', 'global'):
            scores = data.get(f'{effect}_effect_scores')
            if scores is None:
                raw[effect]['scores'].append(None)
                raw[effect]['tgm'].append(None)
                raw[effect]['auc'].append(np.nan)
                raw[effect]['cv'].append(None)
                continue
            scores = np.asarray(scores, dtype=float)
            if scores.shape[0] != times_ms.shape[0]:
                logger.warning(
                    "Sujet %s (%s) : %d scores pour %d points temporels, "
                    "effet ignore.", subject_id, effect, scores.shape[0],
                    times_ms.shape[0])
                raw[effect]['scores'].append(None)
                raw[effect]['tgm'].append(None)
                raw[effect]['auc'].append(np.nan)
                raw[effect]['cv'].append(None)
                continue

            has_any_effect = True
            raw[effect]['scores'].append(scores)

            tgm = data.get(f'{effect}_effect_tgm')
            if tgm is not None:
                tgm = np.asarray(tgm, dtype=float)
                if tgm.ndim != 2 or tgm.shape[0] != times_ms.shape[0]:
                    tgm = None
            raw[effect]['tgm'].append(tgm)

            auc = data.get(f'{effect}_effect_auc_global')
            raw[effect]['auc'].append(float(auc) if auc is not None else np.nan)
            raw[effect]['cv'].append(data.get(f'{effect}_effect_cv_scores'))

        if not has_any_effect:
            for effect in ('local', 'global'):
                for key in ('scores', 'tgm', 'auc', 'cv'):
                    raw[effect][key].pop()
            continue

        subject_ids.append(subject_id)
        subject_times.append(times_ms)
        group_data_list.append(data)

    if not group_data_list:
        logger.warning(
            "Aucune donnee valide pour le groupe %s apres filtrage temporel.",
            group_name)
        return {}

    # Grille commune bornee a la fenetre d'analyse LG.
    clipped_times = []
    for times_ms in subject_times:
        in_window = (times_ms >= COMMON_T_MIN) & (times_ms <= COMMON_T_MAX)
        clipped_times.append(times_ms[in_window])
    common_time_vector = build_common_time_grid(clipped_times,
                                                subject_labels=subject_ids)
    if common_time_vector is None:
        logger.warning(
            "Grille temporelle commune impossible pour le groupe %s.",
            group_name)
        return {}

    result = {'group_name': group_name, 'n_subjects': len(subject_ids),
              'subject_ids': subject_ids, 'times': common_time_vector,
              'group_data': group_data_list}

    for effect in ('local', 'global'):
        scores_aligned, tgms_aligned = [], []
        aucs, cvs, kept_ids = [], [], []
        for idx, subject_id in enumerate(subject_ids):
            scores = raw[effect]['scores'][idx]
            if scores is None:
                continue
            resampled = resample_to_grid(scores, subject_times[idx],
                                         common_time_vector)
            if resampled is None or np.all(np.isnan(resampled)):
                continue
            scores_aligned.append(resampled)
            kept_ids.append(subject_id)
            aucs.append(raw[effect]['auc'][idx])
            cvs.append(raw[effect]['cv'][idx])

            tgm = raw[effect]['tgm'][idx]
            if tgm is not None:
                tgm_resampled = resample_to_grid(tgm, subject_times[idx],
                                                 common_time_vector)
                if tgm_resampled is not None:
                    tgms_aligned.append(tgm_resampled)

        if not scores_aligned:
            continue

        matrix = np.array(scores_aligned)
        result[f'{effect}_effect'] = {
            'scores_matrix': matrix,
            'group_mean': np.nanmean(matrix, axis=0),
            'group_sem': np.nanstd(matrix, axis=0) / np.sqrt(matrix.shape[0]),
            'auc_global_values': np.array(aucs, dtype=float),
            'cv_scores_list': cvs,
            'subject_ids': kept_ids,
        }
        if tgms_aligned:
            result[f'{effect}_effect']['tgm_matrix'] = np.array(tgms_aligned)
            result[f'{effect}_effect']['tgm_mean'] = np.nanmean(
                np.array(tgms_aligned), axis=0)

    logger.info(
        "Groupe %s analyse - Local: %d, Global: %d sujets.",
        group_name,
        result.get('local_effect', {}).get('scores_matrix', np.empty((0,))).shape[0],
        result.get('global_effect', {}).get('scores_matrix', np.empty((0,))).shape[0])
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
                        chance_level=CHANCE_LEVEL, 
                        n_permutations=10000,
                        alternative_hypothesis="greater", 
                        n_jobs=-1,
                        random_seed=42
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
                _, significant_mask, _, _ = perform_pointwise_fdr_correction_on_scores(tgm_matrix, chance_level=CHANCE_LEVEL, alpha_significance_level=FDR_ALPHA,alternative_hypothesis="greater")
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

def analyze_individual_significance_proportions(all_groups_data, save_dir, show_plots=True):
    """Significativite individuelle et proportion de sujets significatifs.

    Test de permutation des labels (examples/run_permutation_tests.py), BH sur
    tous les sujets du contraste, IC 95 % de Clopper-Pearson et tests exacts de
    Fisher entre groupes (utils/permutation_stats.py). Remplace le Wilcoxon sur
    les AUC des plis, non valide : les plis partagent leurs essais
    d'entrainement (Stelzer et al. 2013 ; Noirhomme et al. 2014).
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for effect_type in ['local', 'global']:
        effect_key = f'{effect_type}_effect'
        present = [g for g in all_groups_data if effect_key in g]
        groups = {
            g['group_name']: (g[effect_key]['subject_ids'],
                              g[effect_key].get('perm_p', np.full(len(g[effect_key]['subject_ids']), np.nan)))
            for g in present
        }
        stats_res = proportion_statistics(groups, alpha=FDR_ALPHA)
        if stats_res is None:
            logger.error("Effet %s : aucun resultat de permutation, proportions "
                         "non calculees.", effect_type)
            continue
        logger.info("Effet %s\n%s", effect_type, proportion_report(stats_res))
        if save_dir:
            with open(os.path.join(save_dir, f"individual_significance_stats_{effect_type}.json"), 'w') as f:
                json.dump(stats_res, f, indent=2)

        # --- Barres : AUC de chaque sujet, * = significatif ---
        for g in present:
            effect = g[effect_key]
            aucs = np.asarray(effect['auc_global_values'], dtype=float)
            ids = np.asarray(effect['subject_ids'])
            keep = np.isfinite(aucs)
            if not keep.any():
                continue
            aucs, ids = aucs[keep], ids[keep]
            subj = stats_res['subjects']
            is_sig = [subj.get(s, {}).get('significant', False) for s in ids]
            colors = ['#4ecdc4' if sig else ('#ff6b6b' if s in subj else '#bbbbbb')
                      for s, sig in zip(ids, is_sig)]
            x_pos = np.arange(len(aucs))
            plt.figure(figsize=(12, 7))
            bars = plt.bar(x_pos, aucs, color=colors, label='Score AUC par sujet')
            for rect, sig in zip(bars, is_sig):
                if sig:
                    plt.text(rect.get_x() + rect.get_width() / 2., rect.get_height(), '*',
                             ha='center', va='bottom', color='black', fontsize=14, fontweight='bold')
            plt.axhline(y=CHANCE_LEVEL, color='k', linestyle='--', alpha=0.5, label='Chance (0.5)')
            plt.xlabel('ID du Sujet')
            plt.ylabel('AUC (5 plis, 0-800 ms)')
            plt.xticks(x_pos, ids, rotation=45, ha='right')
            group_label = GROUP_NAME_MAPPING.get(g['group_name'], g['group_name'])
            plt.title(f'Scores AUC Individuels - {group_label} - Effet {effect_type.capitalize()}\n'
                      '(* = test de permutation, BH q < 0.05 sur tous les sujets)')
            plt.legend()
            plt.tight_layout()
            if save_dir:
                bar_dir = os.path.join(save_dir, 'bar_charts', effect_type)
                os.makedirs(bar_dir, exist_ok=True)
                filepath = os.path.join(bar_dir, f"bar_scores_{g['group_name']}_{effect_type}.png")
                plt.savefig(filepath, dpi=300)
                logger.info(f"Graphique à barres sauvegardé : {filepath}")
            if show_plots:
                plt.show()
            else:
                plt.close()

        # --- Camemberts : proportion par groupe, IC 95 % ---
        names = ([n for n in GROUP_ORDER if n in stats_res['groups']]
                 + [n for n in stats_res['groups'] if n not in GROUP_ORDER])
        names = [n for n in names if stats_res['groups'][n]['n_tested']]
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 5), squeeze=False)
        for i, name in enumerate(names):
            gs = stats_res['groups'][name]
            lo, hi = gs['ci95_clopper_pearson']
            axes[0, i].pie([gs['n_significant'], gs['n_tested'] - gs['n_significant']],
                           labels=['Significatif\n(q<0.05)', 'Non signif.'], autopct=lambda p: f'{p:.1f}%' if p > 0 else '',
                           colors=['#4ecdc4', '#ff6b6b'], startangle=90)
            axes[0, i].set_title(f"{GROUP_NAME_MAPPING.get(name, name)}\n"
                                 f"(n={gs['n_tested']}, IC95 {100 * lo:.0f}-{100 * hi:.0f} %)")
        fig.suptitle('Proportion de sujets significatifs (test de permutation, BH sur tous '
                     f'les sujets) - Effet {effect_type.capitalize()}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save_dir:
            pie_dir = os.path.join(save_dir, 'pie_charts')
            os.makedirs(pie_dir, exist_ok=True)
            filepath = os.path.join(pie_dir, f"significance_proportions_{effect_type}.png")
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


def plot_global_auc_boxplots(all_groups_data, save_dir, show_plots=True,
                             auc_labels=None):
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
        ax.set_ylabel((auc_labels or {}).get(effect_type, 'AUC'), fontsize=14)
        ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Niveau de chance ({CHANCE_LEVEL})')

        # Légende x avec n sujets
        new_labels = [f"{g} (n={group_subject_counts.get(g, 0)})" for g in ordered_groups]
        ax.set_xticklabels(new_labels, rotation=45, ha='right')

        # Wilcoxon unilateral de chaque groupe contre la chance, corrige BH
        # sur les groupes (6 tests par figure : sans correction, ~1 faux
        # positif sur 3 figures sous l'hypothese nulle).
        y_min, y_max = ax.get_ylim()
        y_sig = y_max - 0.04 * (y_max - y_min)
        raw_p = []
        for group in ordered_groups:
            group_aucs = df[df['Group'] == group]['AUC'].dropna().values
            raw_p.append(wilcoxon(group_aucs - CHANCE_LEVEL, alternative='greater')[1]
                         if len(group_aucs) > 1 else np.nan)
        p_vs_chance = benjamini_hochberg(raw_p)
        for idx, (group, p_raw, p_fdr) in enumerate(zip(ordered_groups, raw_p, p_vs_chance)):
            logger.info("  %s vs chance (%s) : Wilcoxon p=%.4g, p_FDR=%.4g",
                        group, effect_type, p_raw, p_fdr)
            if stars(p_fdr):
                ax.text(idx, y_sig, stars(p_fdr), ha='center', va='bottom',
                        fontsize=18, fontweight='bold', color='black')
        if stats_results:
            stats_results['vs_chance_wilcoxon_bh'] = {
                g: {'p': float(pr), 'p_fdr': float(pf)}
                for g, pr, pf in zip(ordered_groups, raw_p, p_vs_chance)}
        # (L'ancien appel a add_significance_bars() placait les barres selon
        # l'ordre de chargement des groupes, pas celui de l'axe x : les
        # etoiles reliaient de mauvaises boites. Les barres sont tracees
        # ci-dessous, une seule fois, sur les bonnes positions.)

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
                # Cles en noms bruts (DELIRIUM+), axe en noms affiches (Delirium +).
                g1, g2 = (GROUP_NAME_MAPPING.get(g, g) for g in key.split('_vs_'))
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
    - Une barre coloree sous la courbe pour les clusters temporels ou le
      GROUPE decode au-dessus de la chance (test de cluster par permutation
      de signe sur les sujets, 10000 permutations, p < 0.05).
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
                        n_permutations=10000,
                        alternative_hypothesis="greater",
                        n_jobs=-1,
                        random_seed=42
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

            # Wilcoxon unilateral contre la chance par fenetre, BH sur les
            # fenetres de la figure. Les fenetres ont ete choisies d'apres les
            # courbes de groupe : test EXPLORATOIRE (selection circulaire),
            # l'inference temporelle de reference est le test de cluster.
            raw_p = []
            for win_idx in range(len(window_names)):
                vals = subjects_window_means[:, win_idx]
                vals = vals[~np.isnan(vals)]
                raw_p.append(wilcoxon(vals - CHANCE_LEVEL, alternative='greater')[1]
                             if len(vals) > 1 else np.nan)
            for win_idx, p_fdr in enumerate(benjamini_hochberg(raw_p)):
                if stars(p_fdr):
                    ax.text(win_idx, 0.41, stars(p_fdr), ha='center', va='center',
                            fontsize=18, fontweight='bold')

            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Niveau de chance')
            ax.set_xlim(-0.5, len(window_names) - 0.5)
            ax.set_ylim(0.4, 0.77)
            ax.set_xticks(x_positions)
            # Affichage des noms de fenêtres en petit pour une meilleure lisibilité
            ax.set_xticklabels(window_names, fontsize=8, rotation=45)
            ax.set_ylabel('Score AUC', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{mapped_name} - {effect_type.capitalize()}\nFenêtres temporelles (n={n_subjects}) - exploratoire, BH', fontsize=16, fontweight='bold')
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
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyse de groupe LG a partir des NPZ individuels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Racine contenant les resultats individuels LG "
             "(defaut: BAKING_EEG_RESULTS_DIR, sinon configure_project_paths).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Dossier de sortie des figures "
             "(defaut: <results-dir>/group_analysis/LG_analysis_<horodatage>).")
    args = parser.parse_args()

    base_results_dir = args.results_dir or get_default_results_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.output_dir or os.path.join(
        get_default_output_dir(), f"LG_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    logger.info("Racine des résultats individuels : %s", base_results_dir)
    logger.info("Les résultats seront sauvegardés dans : %s", results_dir)

    organized_data = find_npz_files(base_results_dir)
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

    # Test de permutation individuel (examples/run_permutation_tests.py) : il
    # fournit les p individuelles ET l'AUC comparee entre groupes, calculee
    # sur la meme fenetre post-stimulus (0-800 ms) et 5 plis pour tous.
    auc_labels = {}
    for effect_type, contrast in (('local', 'ls_ld'), ('global', 'gs_gd')):
        perm = load_permutation_results(base_results_dir, "LG", contrast)
        for group_data in all_groups_data:
            effect = group_data.get(f'{effect_type}_effect')
            if not effect:
                continue
            effect['perm_auc'], effect['perm_p'] = attach_permutation_results(
                effect['subject_ids'], perm)
            if perm:
                effect['auc_global_values'] = effect['perm_auc']
        if perm:
            auc_labels[effect_type] = "AUC, permutation-test decoding (5-fold, 0-800 ms)"
        else:
            logger.warning(
                "Aucun resultat dans %s/permutation_tests/LG pour %s : boxplots "
                "sur l'AUC CV d'origine (epoque entiere, exploratoire) et pas de "
                "proportions de sujets significatifs. Lancer "
                "submitit/submit_permutation_tests.py --paradigm LG.",
                base_results_dir, contrast)
            auc_labels[effect_type] = "AUC, original CV (exploratory)"

    display_subject_sampling_frequencies(all_groups_data)
    plot_global_auc_boxplots(all_groups_data, results_dir, show_plots=False,
                             auc_labels=auc_labels)
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
