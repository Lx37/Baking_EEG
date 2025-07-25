import os
import sys
import glob
import logging
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

from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
 )

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
BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V17"
CHANCE_LEVEL = 0.5
FDR_ALPHA = 0.05


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_subject_id_from_path(file_path):
    import re
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

    for f in pp_ap_files:
        subj_id = extract_subject_id_from_path(f)
        found = False
        for group, ids in ALL_SUBJECTS_GROUPS.items():
            if subj_id in ids:
                organized_data[group].append(f)
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
                'auc_global': data.get('pp_ap_main_mean_auc_global')
            }
    except Exception as e:
        logger.error("Erreur chargement fichier %s: %s", file_path, e)
        return None

def analyze_group_data_pp_ap(group_files, group_name):
    logger.info(f"Analyse du groupe {group_name} ({len(group_files)} sujets) pour PP/AP")
    
    all_scores, all_tgms, all_aucs, subject_ids = [], [], [], []
    times = None
    TARGET_LEN = 601 # Standardisation de la longueur des données

    for file_path in group_files:
        data = load_npz_data_pp_ap(file_path)
        subj_id = extract_subject_id_from_path(file_path)
        
        if data and data['scores_1d_mean'] is not None and data['times'] is not None:
            if data['scores_1d_mean'].shape == (TARGET_LEN,):
                scores = data['scores_1d_mean']
            elif data['scores_1d_mean'].shape == (701,):
                logger.warning(f"Sujet {subj_id} shape (701,) tronqué à (601,).")
                scores = data['scores_1d_mean'][:TARGET_LEN]
            else:
                logger.warning(f"Sujet {subj_id} ignoré: shape inattendu {data['scores_1d_mean'].shape}. Attendu ({TARGET_LEN},) ou (701,).")
                continue

            if times is None:
                times = data['times']
                if times.shape[0] == 701:
                    times = times[:TARGET_LEN]

            all_scores.append(scores)
            subject_ids.append(subj_id)
            if data['tgm_all_folds'] is not None:
                tgm = np.mean(data['tgm_all_folds'], axis=0)
                if tgm.shape[0] == 701:
                    tgm = tgm[:TARGET_LEN, :TARGET_LEN]
                all_tgms.append(tgm)
            if data['auc_global'] is not None:
                all_aucs.append(data['auc_global'])
        else:
            logger.warning(f"Données invalides pour sujet {subj_id} dans {file_path}")

    if not all_scores:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return None

    scores_matrix = np.array(all_scores)
    return {
        'group_name': group_name,
        'n_subjects': len(subject_ids),
        'subject_ids': subject_ids,
        'scores_matrix': scores_matrix,
        'group_mean': np.nanmean(scores_matrix, axis=0),
        'group_sem': np.nanstd(scores_matrix, axis=0) / np.sqrt(len(subject_ids)),
        'times': times,
        'tgm_matrix': np.array(all_tgms) if all_tgms else None,
        'tgm_mean': np.nanmean(np.array(all_tgms), axis=0) if all_tgms else None,
        'auc_global_values': np.array(all_aucs) if all_aucs else None,
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

    *** CODE CORRIGÉ ***
    La correction traite `clusters` comme une liste de masques booléens,
    ce qui est le format de sortie standard de la fonction MNE.
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
            group_subject_counts[mapped_name] += group_data['n_subjects']
            for value in group_data['auc_global_values']:
                plot_data.append({'Group': mapped_name, 'AUC': value})
    if not plot_data:
        logger.warning("Aucune donnée AUC globale pour les boxplots.")
        return
    df = pd.DataFrame(plot_data)
    order = [g for g in GROUP_ORDER if g in df['Group'].unique()]
    palette = [GROUP_COLORS.get(g, '#cccccc') for g in order]
    fig, ax = plt.subplots(figsize=(12, 8))
    # Boxplot par groupe avec couleur
    boxplot = sns.boxplot(data=df, x='Group', y='AUC', ax=ax, order=order, palette=palette)
    # Points individuels en couleur de groupe
    for i, group in enumerate(order):
        group_df = df[df['Group'] == group]
        color = GROUP_COLORS.get(group, 'black')
        ax.scatter([i]*len(group_df), group_df['AUC'], color=color, alpha=0.7, s=40, edgecolor='black', zorder=3)
    new_labels = [f"{group} (n={group_subject_counts.get(group, 0)})" for group in order]
    ax.set_xticklabels(new_labels, rotation=30, ha='right')
    from scipy.stats import mannwhitneyu
    y_max = df['AUC'].max() if len(df) > 0 else 0.8
    y_min = df['AUC'].min() if len(df) > 0 else 0.4
    y_offset = (y_max - y_min) * 0.05
    star_pos = y_max + y_offset
    annotation_idx = 0
    for i, g1 in enumerate(order):
        for j, g2 in enumerate(order):
            if j <= i:
                continue
            auc1 = df[df['Group'] == g1]['AUC']
            auc2 = df[df['Group'] == g2]['AUC']
            if len(auc1) > 1 and len(auc2) > 1:
                stat, p = mannwhitneyu(auc1, auc2, alternative='two-sided')
                if p < 0.05:
                    x1, x2 = i, j
                    y = star_pos + annotation_idx * y_offset * 1.5
                    ax.plot([x1, x1, x2, x2], [y-0.01, y, y, y-0.01], color='black', linewidth=1.5)
                    star = '**' if p < 0.01 else '*'
                    ax.text((x1+x2)/2, y, star, ha='center', va='bottom', color='black', fontsize=18, fontweight='bold')
                    annotation_idx += 1
    ax.set(title='PP/AP Global AUC Distribution by Group', xlabel='Clinical Group', ylabel='AUC (Area Under Curve)')
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
    windows = {'T100': (90, 110), 'T200': (190, 210), 'T300': (290, 310), 'TALL': (0, 600)}

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

def analyze_individual_significance_proportions_pp_ap(all_groups_data, save_dir, show_plots=True):
    groups_analysis = []
    
    # Agréger les données par nom de groupe mappé
    group_data_aggregated = defaultdict(lambda: {'auc_values': [], 'n_subjects': 0})
    for g_data in all_groups_data:
        if g_data['auc_global_values'] is not None and len(g_data['auc_global_values']) > 0:
            mapped_name = GROUP_NAME_MAPPING.get(g_data['group_name'], g_data['group_name'])
            group_data_aggregated[mapped_name]['auc_values'].extend(g_data['auc_global_values'])
            group_data_aggregated[mapped_name]['n_subjects'] += g_data['n_subjects']
            
    # Réorganiser les données selon GROUP_ORDER
    for group_name in GROUP_ORDER:
        if group_name in group_data_aggregated:
            data = group_data_aggregated[group_name]
            auc_values = np.array(data['auc_values'])
            significant_count = np.sum(auc_values > CHANCE_LEVEL)
            total_patients = len(auc_values)
            percentage_significant = (significant_count / total_patients) * 100 if total_patients > 0 else 0
            groups_analysis.append({
                'group_name': group_name,
                'n_subjects': total_patients,
                'n_significant': significant_count,
                'percentage_significant': percentage_significant
            })

    if not groups_analysis: return
    n_groups = len(groups_analysis)
    fig_pie, axes_pie = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5), squeeze=False)
    
    for i, analysis in enumerate(groups_analysis):
        mapped_name = analysis['group_name']
        sizes = [analysis['n_subjects'] - analysis['n_significant'], analysis['n_significant']]
        labels = ['≤ Chance', '> Chance']
        colors = ['#ff6b6b', '#4ecdc4']
        axes_pie[0, i].pie(sizes, autopct='%1.0f%%', startangle=90, colors=colors, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes_pie[0, i].set_title(f"{mapped_name}\n(n={analysis['n_subjects']})")
        
    fig_pie.legend(labels, loc='lower center', ncol=2)
    fig_pie.suptitle('Proportion of Subjects with AUC > Chance (PP/AP)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if save_dir: plt.savefig(os.path.join(save_dir, "individual_significance_proportions_pie_pp_ap.png"))
    if show_plots: plt.show()
    else: plt.close(fig_pie)

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
    logger.info("Début de l'analyse PP/AP")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("/home/tom.balay/results", f"PP_AP_analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    organized_files = find_pp_ap_npz_files(BASE_RESULTS_DIR)
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