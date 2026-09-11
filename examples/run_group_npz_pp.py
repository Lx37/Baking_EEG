import os
import sys
import glob
import argparse
import logging
from getpass import getuser
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import fdrcorrection
import json
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config.config import ALL_SUBJECTS_GROUPS
from utils.utils import configure_project_paths

from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
 )
from utils.time_alignment import align_subject_curves, resample_to_grid

GROUP_NAME_MAPPING = {
    'COMA': 'COMA',
    'VS': 'VS',
    'DELIRIUM+': 'DELIRIUM+',
    'DELIRIUM-': 'DELIRIUM-',
    'CONTROLS_DELIRIUM': 'CONTROLS',
    'MCS': 'MCS',
    'CONTROLS': 'CONTROLS'
}
GROUP_ORDER = ['CONTROLS', 'DELIRIUM-', 'DELIRIUM+', 'MCS', 'VS', 'COMA']

GROUP_COLORS = {
    'CONTROLS': '#2ca02c', 'DELIRIUM-': '#ff7f0e', 'DELIRIUM+': '#d62728',
    'MCS': '#1f77b4', 'COMA': '#9467bd', 'VS': '#8c564b',
}

# --- Paramètres généraux ---
CHANCE_LEVEL = 0.5
FDR_ALPHA = 0.05

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_default_results_dir():
    """Racine des resultats individuels, deduite de la config utilisateur.

    Remplace le chemin macOS code en dur, inutilisable sur le cluster. Peut
    etre surchargee par --results-dir ou la variable d'environnement
    BAKING_EEG_RESULTS_DIR.
    """
    env_dir = os.environ.get("BAKING_EEG_RESULTS_DIR")
    if env_dir:
        return env_dir
    try:
        _, base_output = configure_project_paths(getuser())
        return base_output
    except Exception as exc:  # configuration utilisateur absente
        logger.warning(
            "configure_project_paths a echoue (%s) : utilisation du "
            "repertoire courant.", exc)
        return os.getcwd()


def get_default_output_dir():
    """Dossier de sortie des figures de groupe."""
    env_dir = os.environ.get("BAKING_EEG_GROUP_OUTPUT_DIR")
    if env_dir:
        return env_dir
    return os.path.join(get_default_results_dir(), "group_analysis")


def extract_subject_id_from_path(file_path):
    import re
    
    # 1. Vérifier si une partie du chemin correspond à un ID connu dans ALL_SUBJECTS_GROUPS
    # Cela permet de gérer des structures de dossiers variables (profondeur différente)
    all_known_ids = set()
    for ids in ALL_SUBJECTS_GROUPS.values():
        all_known_ids.update(ids)
    
    parts = file_path.split(os.sep)
    # On parcourt les dossiers en remontant depuis le fichier
    for part in reversed(parts):
        if part in all_known_ids:
            return part

    # 2. Regex fallback (ancien comportement)
    #Extract subject ID from file path, e.g., "Subj_12345_svc" -> "12345"
    match = re.search(r'Subj_([A-Za-z0-9]+)', file_path) 
    if match:
        return match.group(1)
    # Fallback: ppext3/CW41 -> prend ppext3
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))

def find_pp_ap_npz_files(base_path):
    logger.info("Recherche des fichiers NPZ PP/AP dans: %s", base_path)
    #Organize files by group
    organized_data = {group: [] for group in ALL_SUBJECTS_GROUPS}
    pattern = os.path.join(base_path, 'intra_subject_results', '**', 'decoding_results_full.npz')
    all_npz_files = glob.glob(pattern, recursive=True)

    if not all_npz_files:
        logger.warning("Aucun fichier 'decoding_results_full.npz' trouvé.")
        return {}

    pp_ap_files = []
    for f in all_npz_files:
        try:
            with np.load(f, allow_pickle=True) as data:
                if 'pp_ap_main_scores_1d_mean' in data.keys():
                    pp_ap_files.append(f)
        except Exception as e:
            logger.warning(f"Impossible de lire le fichier {f}: {e}")
    
    logger.info("%d fichiers PP/AP trouvés.", len(pp_ap_files))

    # Suivi des sujets ajoutés pour éviter les doublons
    added_subjects = {group: set() for group in ALL_SUBJECTS_GROUPS}

    for f in pp_ap_files:
        subj_id = extract_subject_id_from_path(f)
        found = False
        for group, ids in ALL_SUBJECTS_GROUPS.items():
            if subj_id in ids:
                if subj_id not in added_subjects[group]:
                    organized_data[group].append(f)
                    added_subjects[group].add(subj_id)
                else:
                    logger.warning(f"Doublon ignoré pour le sujet {subj_id} dans le groupe {group}: {f}")
                found = True
                break
        if not found:
            logger.warning(f"Sujet {subj_id} du fichier {f} non trouvé dans ALL_SUBJECTS_GROUPS.")

    return {g: fl for g, fl in organized_data.items() if fl}

def load_npz_data_pp_ap(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            return {
                'scores_1d_mean': data.get('pp_ap_main_scores_1d_mean'),
                'times': data.get('epochs_time_points'),
                'tgm_mean': data.get('pp_ap_main_tgm_mean'),
                'tgm_all_folds': data.get('pp_ap_main_tgm_all_folds'),
                'auc_global': data.get('pp_ap_main_mean_auc_global'),
                'cv_global_scores': data.get('pp_ap_main_cv_global_scores')
            }
    except Exception as e:
        logger.error("Erreur chargement fichier %s: %s", file_path, e)
        return None

def analyze_group_data_pp_ap(group_files, group_name):
    """Agrege les resultats individuels d'un groupe.

    Deux corrections importantes par rapport aux versions <= 2026-01 :

    1. Alignement temporel. Les courbes n'etaient gardees que si elles
       faisaient exactement 601 points (ou 701, alors TRONQUES) ; toute autre
       longueur entrainait l'exclusion silencieuse du sujet - or la config
       documente des sujets a 401 et 301 points. On reinterpole desormais
       chaque sujet sur une grille temporelle commune (cf.
       utils.time_alignment) : aucun sujet n'est perdu, et l'indice i designe
       le meme instant chez tous.

    2. Listes paralleles. `auc_global_values` et `tgm_matrix` n'etaient
       alimentees que lorsque la valeur existait, alors que `subject_ids`
       l'etait systematiquement : les AUC se retrouvaient decalees et
       attribuees au mauvais sujet. On insere desormais np.nan / None pour
       conserver l'alignement des index.
    """
    logger.info("Analyse du groupe %s (%d sujets) pour PP/AP",
                group_name, len(group_files))

    raw_scores, raw_times, raw_tgms = [], [], []
    all_aucs, subject_ids, all_cv_scores = [], [], []

    for file_path in group_files:
        data = load_npz_data_pp_ap(file_path)
        subj_id = extract_subject_id_from_path(file_path)

        if not data or data['scores_1d_mean'] is None or data['times'] is None:
            logger.warning("Donnees invalides pour le sujet %s dans %s",
                           subj_id, file_path)
            continue

        scores = np.asarray(data['scores_1d_mean'], dtype=float)
        times_subject = np.asarray(data['times'], dtype=float)
        if scores.shape[0] != times_subject.shape[0]:
            logger.warning(
                "Sujet %s ignore : %d scores pour %d points temporels.",
                subj_id, scores.shape[0], times_subject.shape[0])
            continue

        tgm = None
        if data['tgm_all_folds'] is not None:
            tgm_candidate = np.nanmean(np.asarray(data['tgm_all_folds'],
                                                  dtype=float), axis=0)
            if tgm_candidate.ndim == 2 and \
                    tgm_candidate.shape[0] == times_subject.shape[0]:
                tgm = tgm_candidate
            else:
                logger.warning(
                    "Sujet %s : TGM de forme %s incompatible avec %d points "
                    "temporels, TGM ignoree.",
                    subj_id, getattr(tgm_candidate, 'shape', None),
                    times_subject.shape[0])

        raw_scores.append(scores)
        raw_times.append(times_subject)
        raw_tgms.append(tgm)
        subject_ids.append(subj_id)
        # np.nan / None systematiques : les listes restent alignees sur
        # subject_ids meme quand une valeur manque.
        all_aucs.append(float(data['auc_global'])
                        if data['auc_global'] is not None else np.nan)
        all_cv_scores.append(data.get('cv_global_scores'))

    if not raw_scores:
        logger.warning("Aucune donnee valide trouvee pour le groupe %s",
                       group_name)
        return None

    scores_matrix, times, kept_ids, dropped_ids = align_subject_curves(
        raw_scores, raw_times, subject_ids)
    if scores_matrix is None:
        logger.warning(
            "Alignement temporel impossible pour le groupe %s.", group_name)
        return None

    # Repercuter les eventuelles exclusions sur TOUTES les listes paralleles.
    if dropped_ids:
        keep_mask = [sid in set(kept_ids) for sid in subject_ids]
        raw_tgms = [t for t, k in zip(raw_tgms, keep_mask) if k]
        all_aucs = [a for a, k in zip(all_aucs, keep_mask) if k]
        all_cv_scores = [c for c, k in zip(all_cv_scores, keep_mask) if k]
        subject_ids = kept_ids

    aligned_tgms = []
    for tgm, times_subject, subj_id in zip(raw_tgms, raw_times, subject_ids):
        if tgm is None:
            continue
        resampled = resample_to_grid(tgm, times_subject, times)
        if resampled is not None:
            aligned_tgms.append(resampled)

    n_subjects = len(subject_ids)
    logger.info(
        "Groupe %s : %d sujets retenus, %d TGM alignees, %d AUC globales "
        "disponibles.", group_name, n_subjects, len(aligned_tgms),
        int(np.sum(~np.isnan(all_aucs))) if all_aucs else 0)

    return {
        'group_name': group_name,
        'n_subjects': n_subjects,
        'subject_ids': subject_ids,
        'scores_matrix': scores_matrix,
        'group_mean': np.nanmean(scores_matrix, axis=0),
        'group_sem': np.nanstd(scores_matrix, axis=0) / np.sqrt(n_subjects),
        'times': times,
        'tgm_matrix': np.array(aligned_tgms) if aligned_tgms else None,
        'tgm_mean': (np.nanmean(np.array(aligned_tgms), axis=0)
                     if aligned_tgms else None),
        'auc_global_values': np.array(all_aucs, dtype=float),
        'cv_scores_list': all_cv_scores
    }

def plot_group_individual_curves_pp_ap(group_data, save_dir, show_plots=True):
    """
    Affiche les courbes individuelles et la moyenne pour un seul groupe.
    (Cette fonction était déjà correcte).
    """
    group_name = group_data['group_name']
    mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
    times_ms = group_data['times'] * 1000 if np.max(group_data['times']) < 10 else group_data['times']
    group_color = GROUP_COLORS.get(group_name, '#1f77b4')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(group_data['scores_matrix'].shape[0]):
        ax.plot(times_ms, group_data['scores_matrix'][i, :], color=group_color, alpha=0.2, linewidth=1)
    
    ax.plot(times_ms, group_data['group_mean'], color=group_color, alpha=0.9, linewidth=3, label=f'{mapped_name} (n={group_data["n_subjects"]})')
    ax.fill_between(times_ms, group_data['group_mean'] - group_data['group_sem'], group_data['group_mean'] + group_data['group_sem'], color=group_color, alpha=0.3)
    
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.7, label='Chance level')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax.set(xlabel='Time (ms)', ylabel='Score AUC', title=f'PP/AP Decoding Performance - {mapped_name}', ylim=[0.35, 0.80])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"pp_ap_group_{group_name}_curves.png"))
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_all_groups_comparison_pp_ap(all_groups_data, save_dir, show_plots=True):
    """
    Affiche et sauvegarde un graphique comparant les scores de décodage de plusieurs groupes,
    avec des barres de significativité colorées et empilées.


    """
    if not all_groups_data:
        logger.warning("Aucune donnée de groupe fournie à la fonction de traçage.")
        return

    fig, ax = plt.subplots(figsize=(18, 10))

    # --- Configuration pour les barres de significativité empilées ---
    SIGNIFICANCE_AREA_TOP = 0.44
    BAR_HEIGHT = 0.008
    BAR_GAP = 0.004
    # -------------------------------------------------------------------

    # 1. Dessiner les courbes de décodage pour chaque groupe
    for group_name in GROUP_ORDER:
        group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
        mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
        color = GROUP_COLORS.get(group_name, 'grey')
        if group_data:
            group_mean = group_data['group_mean']
            group_sem = group_data['group_sem']
            times_ms = group_data['times'] * 1000 if np.max(group_data['times']) < 10 else group_data['times']
            label = f'{mapped_name} (n={group_data["n_subjects"]})'
            ax.plot(times_ms, group_mean, color=color, linewidth=2.5, label=label)
            ax.fill_between(times_ms, group_mean - group_sem, group_mean + group_sem, color=color, alpha=0.2)
            # --- Ajout détection et annotation des pics principaux ---
            try:
                from scipy.signal import find_peaks
                curve = group_mean
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
                    distant_peaks = [p for p in peaks if abs(p - first_peak) >= min_dist_idx and times_ms[p] > 0]
                    if times_ms[first_peak] > 0:
                        if distant_peaks:
                            distant_heights = curve[distant_peaks]
                            second_peak = distant_peaks[np.argmax(distant_heights)]
                            main_peaks = [first_peak, second_peak]
                        else:
                            main_peaks = [first_peak]
                    else:
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
                logger.warning(f"Erreur lors de la détection des pics pour {group_name}: {e}")
        else:
            # Affiche une entrée dans la légende même si le groupe n'a pas de données
            ax.plot([], [], color=color, linewidth=2.5, label=f'{mapped_name} (n=0)')

    # 2. Dessiner les barres de significativité empilées
    for group_idx, group_name in enumerate(GROUP_ORDER):
        group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
        if not group_data:
            continue
            
        color = GROUP_COLORS.get(group_name, 'grey')
        scores_matrix = group_data['scores_matrix']
        times_ms = group_data['times'] * 1000 if np.max(group_data['times']) < 10 else group_data['times']
        
        try:
            _, clusters, cluster_p_values, _ = perform_cluster_permutation_test(
                scores_matrix,
                chance_level=CHANCE_LEVEL,
                n_permutations=10000,
                alternative_hypothesis="greater",
                n_jobs=-1
            )
            
            # Positionnement vertical des barres pour ce groupe
            y_top = SIGNIFICANCE_AREA_TOP - (group_idx * (BAR_HEIGHT + BAR_GAP))
            y_bottom = y_top - BAR_HEIGHT
            

            # Itérer sur chaque cluster (qui est un masque booléen) et sa p-valeur
            for cluster_mask, pval in zip(clusters, cluster_p_values):
                if pval < 0.05:
                    # Utiliser directement le masque booléen dans 'where'
                    ax.fill_between(times_ms, y_bottom, y_top,
                                    where=cluster_mask,
                                    color=color,
                                    alpha=1.0,  # Rendre la barre opaque pour une bonne visibilité
                                    step='post',
                                    linewidth=0,
                                    zorder=10) # Assurer que la barre est au premier plan


        except Exception as e:
            logger.warning(f"Le test de permutation par cluster a échoué pour {group_name}: {e}")

    # 3. Configuration finale des axes et de la figure
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.6, label=f'Chance ({CHANCE_LEVEL})')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.4)
    ax.set_title('Comparaison des Groupes - Décodage PP/AP', fontsize=18)
    ax.set_xlabel('Temps (ms)', fontsize=14)
    ax.set_ylabel('Score AUC', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, linestyle=':', linewidth=0.6)
    
    # Ajuster la limite inférieure de l'axe Y pour s'assurer que toutes les barres sont visibles
    num_groups = len(GROUP_ORDER)
    y_min_limit = SIGNIFICANCE_AREA_TOP - (num_groups * (BAR_HEIGHT + BAR_GAP)) - 0.01
    ax.set_ylim([y_min_limit, 0.75])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, "pp_ap_all_groups_comparison_stacked_colored_clusters.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique de comparaison sauvegardé : {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_global_auc_boxplots_pp_ap(all_groups_data, save_dir, show_plots=True):
    plot_data = []
    group_subject_counts = defaultdict(int)
    for group_data in all_groups_data:
        if group_data['auc_global_values'] is not None and len(group_data['auc_global_values']) > 0:
            mapped_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
            for value in group_data['auc_global_values']:
                # Les AUC manquantes sont desormais des NaN explicites (listes
                # paralleles) : on ne les compte pas dans le n affiche.
                if value is None or not np.isfinite(value):
                    continue
                group_subject_counts[mapped_name] += 1
                plot_data.append({'Group': mapped_name, 'AUC': float(value)})
    if not plot_data:
        logger.warning("Aucune donnée AUC globale pour les boxplots.")
        return
    df = pd.DataFrame(plot_data)
    order = [g for g in GROUP_ORDER if g in df['Group'].unique()]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    # Boxplot par groupe avec couleur
    # Utilisation directe du dictionnaire GROUP_COLORS pour garantir la bonne association couleur-groupe
    boxplot = sns.boxplot(data=df, x='Group', y='AUC', hue='Group', ax=ax, order=order, palette=GROUP_COLORS, legend=False)
    # Points individuels en couleur de groupe
    for i, group in enumerate(order):
        group_df = df[df['Group'] == group]
        color = GROUP_COLORS.get(group, 'black')
        ax.scatter([i]*len(group_df), group_df['AUC'], color=color, alpha=0.7, s=40, edgecolor='black', zorder=3)
    new_labels = [f"{group} (n={group_subject_counts.get(group, 0)})" for group in order]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(new_labels, rotation=30, ha='right')
    y_max = df['AUC'].max() if len(df) > 0 else 0.8
    y_min = df['AUC'].min() if len(df) > 0 else 0.4
    y_offset = (y_max - y_min) * 0.05
    star_pos = y_max + y_offset

    # Comparaisons deux a deux AVEC correction FDR. Avec 6 groupes cela fait
    # 15 tests : sans correction, environ un "resultat significatif" par
    # figure est attendu sous l'hypothese nulle. Les etoiles affichees
    # correspondent maintenant aux p-values corrigees.
    pairs, raw_pvals = [], []
    for i, g1 in enumerate(order):
        for j, g2 in enumerate(order):
            if j <= i:
                continue
            auc1 = df[df['Group'] == g1]['AUC'].to_numpy()
            auc2 = df[df['Group'] == g2]['AUC'].to_numpy()
            if len(auc1) > 1 and len(auc2) > 1:
                _, p = mannwhitneyu(auc1, auc2, alternative='two-sided')
                pairs.append((i, j, g1, g2, len(auc1), len(auc2)))
                raw_pvals.append(p)

    corrected_pvals = np.array(raw_pvals, dtype=float)
    if raw_pvals:
        _, corrected_pvals = fdrcorrection(np.array(raw_pvals),
                                           alpha=FDR_ALPHA)
        logger.info(
            "Comparaisons inter-groupes (Mann-Whitney, %d tests, FDR "
            "Benjamini-Hochberg alpha=%.2f) :", len(raw_pvals), FDR_ALPHA)
        for (i, j, g1, g2, n1, n2), p_raw, p_fdr in zip(pairs, raw_pvals,
                                                        corrected_pvals):
            logger.info("  %-12s vs %-12s  n=%d/%d  p=%.4f  p_FDR=%.4f%s",
                        g1, g2, n1, n2, p_raw, p_fdr,
                        "  *" if p_fdr < FDR_ALPHA else "")

    annotation_idx = 0
    for (i, j, _g1, _g2, _n1, _n2), p_fdr in zip(pairs, corrected_pvals):
        if p_fdr < FDR_ALPHA:
            x1, x2 = i, j
            y = star_pos + annotation_idx * y_offset * 1.5
            ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01],
                    color='black', linewidth=1.5)
            star = '**' if p_fdr < 0.01 else '*'
            ax.text((x1 + x2) / 2, y, star, ha='center', va='bottom',
                    color='black', fontsize=18, fontweight='bold')
            annotation_idx += 1
    if not annotation_idx and pairs:
        ax.text(0.99, 0.02, "aucune différence significative après FDR",
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                style='italic', color='#555555')

    ax.set(title='PP/AP Global AUC Distribution by Group\n'
                 '(Mann-Whitney, FDR-corrected)',
           xlabel='Clinical Group', ylabel='AUC (Area Under Curve)')
    ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label='Chance Level')
    ax.legend(); plt.tight_layout()
    if save_dir: plt.savefig(os.path.join(save_dir, "pp_ap_global_auc_boxplot.png"))
    if show_plots: plt.show()
    else: plt.close()

def plot_group_tgm_pp_ap(group_data, save_dir, show_plots=True):

    group_name = group_data['group_name']
    mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
    if group_data['n_subjects'] < 2 or group_data['tgm_mean'] is None:
        logger.info(f"TGM non générée pour {mapped_name}: n_sujets < 2 ou données TGM manquantes.")
        return
    tgm_mean = group_data['tgm_mean']
    tgm_matrix = group_data.get('tgm_matrix')
    times_ms = group_data['times'] * 1000 if np.max(group_data['times']) < 10 else group_data['times']

    # Correction FDR pointwise si disponible
    try:
        from stats_utils import perform_pointwise_fdr_correction_on_scores
        STATS_UTILS_AVAILABLE = True
    except ImportError:
        STATS_UTILS_AVAILABLE = False

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(tgm_mean, origin='lower', aspect='auto', extent=[times_ms[0], times_ms[-1], times_ms[0], times_ms[-1]], vmin=0.4, vmax=0.62, cmap='RdYlBu_r')

    if tgm_matrix is not None and STATS_UTILS_AVAILABLE:
        try:
            _, significant_mask, _, _ = perform_pointwise_fdr_correction_on_scores(
                tgm_matrix, chance_level=CHANCE_LEVEL, alpha_significance_level=FDR_ALPHA, alternative_hypothesis="two-sided"
            )
            if np.any(significant_mask):
                ax.contour(times_ms, times_ms, significant_mask, levels=[0.5], colors='black', linewidths=2)
        except Exception as e:
            logger.warning(f"Erreur FDR TGM pour {group_name}: {e}")

    ax.plot([times_ms[0], times_ms[-1]], [times_ms[0], times_ms[-1]], 'k--', alpha=0.5)
    ax.axhline(0, color='black', alpha=0.3); ax.axvline(0, color='black', alpha=0.3)
    ax.set_xlabel('Test Time (ms)'); ax.set_ylabel('Train Time (ms)')
    ax.set_title(f"{mapped_name} - TGM PP/AP (n={group_data['n_subjects']})")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label('AUC Score')
    plt.tight_layout()

    if save_dir:
        filepath = os.path.join(save_dir, f"tgm_pp_ap_group_{group_name}.png")
        plt.savefig(filepath, dpi=300)
        logger.info(f"TGM PP/AP sauvegardée : {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_temporal_windows_connected_plots_pp_ap(all_groups_data, save_dir, show_plots=True):
    windows = {
        'T165-175': (165, 175),
        'T225-235': (225, 235),
        'T290-300': (290, 300),
        'T330-340': (330, 340),
        'T745-755': (745, 755)
    }

    for group_data in all_groups_data:
        group_name = group_data['group_name']
        mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
        scores_matrix, times = group_data['scores_matrix'], group_data['times']
        times_ms = times * 1000 if np.max(times) < 10 else times
        
        subjects_window_means = []
        for subj_idx in range(scores_matrix.shape[0]):
            subj_means = []
            for win_name, (start_ms, end_ms) in windows.items():
                indices = np.where((times_ms >= start_ms) & (times_ms <= end_ms))[0]
                if len(indices) > 0:
                    subj_means.append(np.mean(scores_matrix[subj_idx, indices]))
                else:
                    subj_means.append(np.nan)
            subjects_window_means.append(subj_means)

        fig, ax = plt.subplots(figsize=(8, 8))
        x_pos = np.arange(len(windows))
        group_color = GROUP_COLORS.get(mapped_name, '#1f77b4')


        for subj_means in subjects_window_means:
            ax.plot(x_pos, subj_means, 'o-', color=group_color, alpha=0.5, markersize=6)
     
        group_means = np.nanmean(subjects_window_means, axis=0)
        ax.plot(x_pos, group_means, 'o-', color='black', linewidth=4, markersize=10, label='Group Mean')
        
        ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', label='Chance Level')
        ax.set_xticks(x_pos); ax.set_xticklabels(windows.keys())
        ax.set(ylabel='Mean AUC', title=f'{mapped_name} - AUC Across Temporal Windows (PP/AP)', ylim=(0.4, 0.85))
        ax.legend(); ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_dir: plt.savefig(os.path.join(save_dir, f"temporal_windows_connected_pp_ap_{group_name}.png"))
        if show_plots: plt.show()
        else: plt.close()

    # --- Ajout : graphique multi-groupes PP/AP ---
    window_names = list(windows.keys())
    x_positions = np.arange(len(window_names))
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(5 * len(GROUP_ORDER), 7), sharey=True)
    for idx, group_name in enumerate(GROUP_ORDER):
        group_data = next((g for g in all_groups_data if g['group_name'] == group_name), None)
        ax = axes[idx]
        # Vérification et extraction des données nécessaires
        scores_matrix = None
        times = None
        n_subjects = 0
        if group_data is not None:
            scores_matrix = group_data.get('scores_matrix', None)
            times = group_data.get('times', None)
            if scores_matrix is not None and times is not None:
                n_subjects = scores_matrix.shape[0]
                times_ms = times * 1000 if np.max(times) < 10 else times
            else:
                times_ms = None
        else:
            times_ms = None
        group_color = GROUP_COLORS.get(group_name, '#1f77b4')
        if scores_matrix is None or times_ms is None or n_subjects == 0:
            ax.set_title(GROUP_NAME_MAPPING.get(group_name, group_name))
            ax.text(0.5, 0.5, 'Aucune donnée', ha='center', va='center', transform=ax.transAxes)
            continue
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
        for subj_means in subjects_window_means:
            valid_idx = ~np.isnan(subj_means)
            ax.plot(x_positions[valid_idx], np.array(subj_means)[valid_idx], 'o', color=group_color, alpha=0.5, markersize=7)
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
        ax.set_xticklabels(window_names, fontsize=9, rotation=45)
        ax.set_title(f"{GROUP_NAME_MAPPING.get(group_name, group_name)}\n(n={n_subjects})", fontsize=14)
        ax.set_ylim(0.4, 0.77)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Points individuels par groupe - PP/AP', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "temporal_windows_connected_individuals_bygroup_ppap.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique points individuels par groupe sauvegardé: {filepath}")
    plt.close()

def analyze_individual_significance_proportions_pp_ap(all_groups_data, save_dir, show_plots=True):
    """Proportion de sujets dont le decodage depasse significativement la chance.
    """
    groups_analysis = []

    # Dictionnaire pour stocker les résultats détaillés par sujet pour les bar charts
    detailed_results = {}

    for g_data in all_groups_data:
        group_name = g_data['group_name']
        mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)

        n_subjects = g_data['n_subjects']
        subject_ids = g_data['subject_ids']
        cv_scores_list = g_data.get('cv_scores_list', [None] * n_subjects)
        auc_values = g_data['auc_global_values']

        subject_results = []
        testable_idx, raw_pvals = [], []

        for i in range(n_subjects):
            scores = cv_scores_list[i] if i < len(cv_scores_list) else None
            p_val = None
            if scores is not None and np.size(scores) > 0:
                scores = np.asarray(scores, dtype=float)
                scores = scores[np.isfinite(scores)]
                if scores.size >= 2:
                    try:
                        _, p_val = wilcoxon(scores - CHANCE_LEVEL,
                                            alternative='greater')
                    except ValueError as exc:
                        # Cas degeneres (toutes les differences nulles...).
                        logger.debug("Wilcoxon impossible pour %s : %s",
                                     subject_ids[i], exc)
                        p_val = None

            auc_i = (float(auc_values[i])
                     if auc_values is not None and i < len(auc_values)
                     else np.nan)
            subject_results.append({
                'id': subject_ids[i],
                'auc': auc_i,
                'significant': False,   # rempli apres correction FDR
                'p_value': p_val,
                'p_value_fdr': None,
                'testable': p_val is not None,
            })
            if p_val is not None:
                testable_idx.append(i)
                raw_pvals.append(p_val)

        # Correction FDR sur les sujets testables du groupe.
        n_significant = 0
        if raw_pvals:
            rejected, p_fdr = fdrcorrection(np.array(raw_pvals),
                                            alpha=FDR_ALPHA)
            for pos, subj_idx in enumerate(testable_idx):
                subject_results[subj_idx]['p_value_fdr'] = float(p_fdr[pos])
                subject_results[subj_idx]['significant'] = bool(rejected[pos])
            n_significant = int(np.sum(rejected))

        n_testable = len(testable_idx)
        n_undetermined = n_subjects - n_testable
        if n_undetermined:
            logger.warning(
                "Groupe %s : %d/%d sujets sans scores de CV exploitables, "
                "classes 'indetermines' et exclus du denominateur.",
                mapped_name, n_undetermined, n_subjects)

        percentage_significant = ((n_significant / n_testable) * 100
                                  if n_testable > 0 else 0.0)

        groups_analysis.append({
            'group_name': group_name,
            'mapped_name': mapped_name,
            'n_subjects': n_testable,
            'n_total_subjects': n_subjects,
            'n_undetermined': n_undetermined,
            'n_significant': n_significant,
            'percentage_significant': percentage_significant
        })

        detailed_results[mapped_name] = subject_results

    if not groups_analysis: return

    # --- 1. Pie Charts (Proportions Globales) ---
    # Filtrer pour ne garder que les groupes présents dans GROUP_ORDER
    ordered_analysis = [g for g in groups_analysis if g['mapped_name'] in GROUP_NAME_MAPPING.values()]
    # Trier selon l'ordre défini dans GROUP_ORDER
    ordered_analysis.sort(key=lambda x: GROUP_ORDER.index(x['mapped_name']) if x['mapped_name'] in GROUP_ORDER else 999)

    n_groups = len(ordered_analysis)
    if n_groups > 0:
        fig_pie, axes_pie = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5), squeeze=False)
        
        for i, analysis in enumerate(ordered_analysis):
            mapped_name = analysis['mapped_name']
            sizes = [analysis['n_subjects'] - analysis['n_significant'], analysis['n_significant']]
            labels = ['Non significatif', 'Significatif (FDR)']
            colors = ['#ff6b6b', '#4ecdc4'] # Rouge clair, Cyan clair

            axes_pie[0, i].pie(sizes, autopct='%1.0f%%', startangle=90, colors=colors,
                             textprops={'fontsize': 12, 'fontweight': 'bold'},
                             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            title = f"{mapped_name}\n(n={analysis['n_subjects']})"
            if analysis.get('n_undetermined'):
                title += f"\n+{analysis['n_undetermined']} indéterminés"
            axes_pie[0, i].set_title(title, fontsize=13, fontweight='bold')

        fig_pie.legend(labels, loc='lower center', ncol=2, fontsize=12)
        fig_pie.suptitle(
            'Proportion of Subjects with Significant Decoding\n'
            '(Wilcoxon on CV scores vs chance, FDR-corrected within group)',
            fontsize=15, fontweight='bold', y=1.06)
        plt.tight_layout()
        
        if save_dir: 
            plt.savefig(os.path.join(save_dir, "individual_significance_proportions_pie_pp_ap.png"), bbox_inches='tight')
        if show_plots: 
            plt.show()
        else: 
            plt.close(fig_pie)

    # --- 2. Bar Charts (Détail par Sujet) ---
    # Créer un bar chart pour chaque groupe montrant l'AUC de chaque sujet
    for group_name in GROUP_ORDER:
        mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
        if mapped_name not in detailed_results:
            continue
            
        subjects = detailed_results[mapped_name]
        if not subjects:
            continue
            
        # Trier les sujets par AUC décroissant pour une meilleure lisibilité
        subjects_sorted = sorted(
            subjects,
            key=lambda x: (x['auc'] if np.isfinite(x['auc']) else -np.inf),
            reverse=True)

        ids = [s['id'] for s in subjects_sorted]
        aucs = [s['auc'] for s in subjects_sorted]
        # Gris = pas de scores CV, donc significativite indeterminee.
        colors = ['#4ecdc4' if s['significant']
                  else ('#ff6b6b' if s.get('testable') else '#bbbbbb')
                  for s in subjects_sorted]

        plt.figure(figsize=(max(10, len(subjects)*0.6), 6)) # Largeur adaptative
        bars = plt.bar(range(len(subjects)), aucs, color=colors, edgecolor='black', alpha=0.8)

        # Ligne de chance
        plt.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2, label='Chance Level')

        # Esthétique
        plt.xticks(range(len(subjects)), ids, rotation=45, ha='right', fontsize=10)
        # Bornes calculees sur les donnees : un ylim fixe (0.3, 1.0) masquait
        # sans le dire les sujets dont l'AUC sortait de l'intervalle.
        finite_aucs = [a for a in aucs if np.isfinite(a)]
        if finite_aucs:
            low = min(0.45, min(finite_aucs) - 0.05)
            high = max(0.75, max(finite_aucs) + 0.05)
            plt.ylim(low, high)
        plt.ylabel('Global AUC', fontsize=12)
        plt.title(f'Individual Decoding Performance - {mapped_name}', fontsize=14, fontweight='bold')
        
        # Légende personnalisée
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ecdc4', edgecolor='black',
                  label='Significatif (Wilcoxon, FDR<0.05)'),
            Patch(facecolor='#ff6b6b', edgecolor='black',
                  label='Non significatif'),
            Patch(facecolor='#bbbbbb', edgecolor='black',
                  label='Indéterminé (pas de scores CV)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"individual_auc_bars_{mapped_name}_pp_ap.png"))
        if show_plots:
            plt.show()
        else:
            plt.close()

def create_temporal_windows_comparison_boxplots_pp_ap(all_groups_data, save_dir, show_plots=True):
    """
    Crée des boxplots pour comparer les AUC moyens dans des fenêtres temporelles spécifiques,
    pour chaque groupe, en affichant les points de données individuels (sujets)
    correctement positionnés à l'intérieur de chaque boxplot.
    """
    windows = {
        '165-175 ms': (165, 175),
        '225-235 ms': (225, 235),
        '290-300 ms': (290, 300),
        '330-340 ms': (330, 340),
        '745-755 ms': (745, 755)
    }

    # 1. Préparation des données dans un DataFrame pandas
    all_data = []
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        mapped_group_name = GROUP_NAME_MAPPING.get(group_name, group_name)
        
        scores_matrix = group_data['scores_matrix']
        times = group_data.get('times')
        subject_ids = group_data.get('subject_ids', [f"{group_name}_{i}" for i in range(scores_matrix.shape[0])])

        if times is None:
            logger.warning(f"Pas de vecteur 'times' pour le groupe {group_name}. Ce groupe sera ignoré.")
            continue
            
        times_ms = times * 1000 if np.max(times) < 10 else times

        for subj_idx in range(scores_matrix.shape[0]):
            subject_id = subject_ids[subj_idx]
            subject_scores = scores_matrix[subj_idx, :]
            
            for window_name, (start_ms, end_ms) in windows.items():
                time_mask = (times_ms >= start_ms) & (times_ms <= end_ms)
                if np.any(time_mask):
                    window_auc = np.mean(subject_scores[time_mask])
                    all_data.append({
                        'Group': mapped_group_name,
                        'Subject': subject_id,
                        'Window': window_name,
                        'AUC': window_auc
                    })

    if not all_data:
        logger.warning("Pas de données à afficher après traitement des fenêtres temporelles.")
        return

    df = pd.DataFrame(all_data)
    
    # Déterminer l'ordre des groupes et des fenêtres pour le tracé
    group_order_mapped = [GROUP_NAME_MAPPING.get(g, g) for g in GROUP_ORDER]
    hue_order = [g for g in group_order_mapped if g in df['Group'].unique()]
    x_order = list(windows.keys())
    
    # Utiliser les couleurs mappées pour les groupes présents
    palette = {GROUP_NAME_MAPPING.get(g, g): GROUP_COLORS.get(g) for g in GROUP_ORDER}

    # 2. Création du graphique
    fig, ax = plt.subplots(figsize=(16, 9))


    sns.boxplot(
        data=df, x='Window', y='AUC', hue='Group',
        order=x_order,
        hue_order=hue_order,
        palette=palette,
        ax=ax,
        fliersize=0, # Cacher les points aberrants du boxplot
        boxprops=dict(alpha=0.8),
        whiskerprops=dict(alpha=0.8),
        capprops=dict(alpha=0.8),
        medianprops=dict(alpha=0.9, color="black", linewidth=1.5)
    )

    # b) Superposer les points individuels avec stripplot.
    #    - `dodge=True` est crucial : il sépare les points de chaque groupe pour les aligner
    #      sur le boxplot correspondant.
    #    - `jitter` ajoute une petite dispersion horizontale pour éviter que les points
    #      ne forment une ligne parfaite.
    sns.stripplot(
        data=df, x='Window', y='AUC', hue='Group',
        order=x_order,
        hue_order=hue_order,
        ax=ax,
        dodge=True,
        jitter=0.1,
        alpha=0.7,
        s=4, # Taille des points
        linewidth=0.5,
        edgecolor='black',
        color='black' # Rendre tous les points noirs pour la clarté
    )


    ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.9, linewidth=1.5)
    

    ax.set_xlabel('Fenêtre Temporelle', fontsize=14)
    ax.set_ylabel('Score AUC Moyen', fontsize=14)
    ax.set_title('Comparaison des scores AUC par fenêtre temporelle et par groupe', fontsize=16, fontweight='bold')
    

    handles, labels = ax.get_legend_handles_labels()
    # Garder uniquement les entrées pour les groupes (généralement la première moitié des handles)
    num_groups = len(hue_order)
    ax.legend(handles[:num_groups], labels[:num_groups], title='Groupe', loc='upper right')

    ax.grid(True, axis='y', linestyle=':', linewidth=0.6)
    ax.set_ylim(0.35, 0.85)
    plt.tight_layout()
    
    # 4. Sauvegarde et affichage
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "temporal_windows_comparison_boxplots_corrected.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300)
        logger.info(f"Boxplots de comparaison sauvegardés : {filepath}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def plot_temporal_windows_boxplots_pp_ap(all_groups_data, save_dir, show_plots=True):
    """
    Crée des boxplots pour comparer les AUC par fenêtre temporelle et groupe pour PP/AP.
    """
    windows = {
        'T165-175': (165, 175),
        'T225-235': (225, 235),
        'T290-300': (290, 300),
        'T330-340': (330, 340),
        'T745-755': (745, 755)
    }
    all_data = []
    for group_data in all_groups_data:
        group_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
        scores_matrix = group_data['scores_matrix']
        times = group_data.get('times')
        subject_ids = group_data['subject_ids']
        if times is None:
            continue
        times_ms = times * 1000 if np.max(times) <= 2 else times
        for subj_idx in range(scores_matrix.shape[0]):
            subject_id = subject_ids[subj_idx]
            subject_scores = scores_matrix[subj_idx, :]
            min_length = min(len(times_ms), len(subject_scores))
            times_ms_truncated = times_ms[:min_length]
            subject_scores_truncated = subject_scores[:min_length]
            for win_name, (tmin, tmax) in windows.items():
                idx = np.where((times_ms_truncated >= tmin) & (times_ms_truncated <= tmax))[0]
                if len(idx) > 0:
                    mean_auc = np.nanmean(subject_scores_truncated[idx])
                    all_data.append({'Group': group_name, 'Window': win_name, 'AUC': mean_auc, 'Subject': subject_id})
    if not all_data:
        logger.warning("Aucune donnée pour créer les boxplots des fenêtres temporelles PP/AP")
        return
    df = pd.DataFrame(all_data)
    # Définir l'ordre des groupes selon GROUP_ORDER
    ordered_groups = [group for group in GROUP_ORDER if group in df['Group'].unique()]
    plt.figure(figsize=(14, 8))

    sns.boxplot(data=df, x='Window', y='AUC', hue='Group', hue_order=ordered_groups, palette=GROUP_COLORS)
    # Points individuels en noir
    sns.stripplot(data=df, x='Window', y='AUC', hue='Group', dodge=True, jitter=True, marker='o', alpha=0.7, size=5, hue_order=ordered_groups, color='black')
    plt.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')
    plt.title('AUC par fenêtre temporelle et groupe (PP/AP)', fontsize=16)
    plt.xlabel('Fenêtre temporelle', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Afficher la légende uniquement pour les groupes
    if len(ordered_groups) > 0:
        plt.legend(handles[:len(ordered_groups)], ordered_groups, title='Groupe')
    plt.tight_layout()
    if save_dir:
        filename = "boxplot_auc_windows_ppap.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Boxplot fenêtres temporelles PP/AP sauvegardé: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Analyse de groupe PP/AP a partir des NPZ individuels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Racine contenant intra_subject_results/ "
             "(defaut: BAKING_EEG_RESULTS_DIR, sinon configure_project_paths).")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Dossier de sortie des figures "
             "(defaut: <results-dir>/group_analysis/PP_AP_analysis_<horodatage>).")
    args = parser.parse_args()

    base_results_dir = args.results_dir or get_default_results_dir()
    logger.info("Début de l'analyse PP/AP")
    logger.info("Racine des résultats individuels : %s", base_results_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.output_dir or os.path.join(
        get_default_output_dir(), f"PP_AP_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    organized_files = find_pp_ap_npz_files(base_results_dir)
    if not organized_files:
        logger.error("Aucun fichier NPZ PP/AP trouvé. Arrêt.")
        return

    all_groups_data = []
    for group_name, group_files in organized_files.items():
        group_data = analyze_group_data_pp_ap(group_files, group_name)
        if group_data:
            all_groups_data.append(group_data)
    
    if not all_groups_data:
        logger.error("Aucune donnée de groupe valide n'a pu être chargée. Vérifiez les fichiers.")
        return


    logger.info("Génération des graphiques...")
    for group_data in all_groups_data:
        plot_group_individual_curves_pp_ap(group_data, results_dir, show_plots=False)
       # plot_group_tgm_pp_ap(group_data, results_dir, show_plots=False)

    plot_all_groups_comparison_pp_ap(all_groups_data, results_dir, show_plots=False)
    plot_global_auc_boxplots_pp_ap(all_groups_data, results_dir, show_plots=False)
    create_temporal_windows_connected_plots_pp_ap(all_groups_data, results_dir, show_plots=False)
    analyze_individual_significance_proportions_pp_ap(all_groups_data, results_dir, show_plots=False)
    create_temporal_windows_comparison_boxplots_pp_ap(all_groups_data, results_dir, show_plots=False)
    plot_temporal_windows_boxplots_pp_ap(all_groups_data, results_dir, show_plots=False)
    # Sauvegarde d'un résumé
    summary_data = [{'name': g['group_name'], 'n_subjects': g['n_subjects'], 'subject_ids': g['subject_ids']} for g in all_groups_data]
    with open(os.path.join(results_dir, "analysis_summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Analyse terminée. Résultats sauvegardés dans: {results_dir}")

if __name__ == "__main__":
    main()