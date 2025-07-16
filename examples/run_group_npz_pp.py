import os
import sys
import glob
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

# --- Importer la config des groupes ---
try:
    from config.config import ALL_SUBJECTS_GROUPS
except ImportError:
    print("AVERTISSEMENT: Impossible d'importer ALL_SUBJECTS_GROUPS depuis config.config")
    ALL_SUBJECTS_GROUPS = {}

# --- Paramètres généraux ---
BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V16"
CHANCE_LEVEL = 0.5
FDR_ALPHA = 0.05
PUBLICATION_PARAMS = {
    'figure.figsize': (16, 9), 'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 20,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14, 'lines.linewidth': 2.5,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1
}
plt.rcParams.update(PUBLICATION_PARAMS)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Fonctions utilitaires ---
def extract_subject_id_from_path(file_path):
    """Extrait l'ID sujet du chemin du fichier."""
    # Cherche le pattern 'Subj_<ID>_svc' dans le chemin
    import re
    match = re.search(r'Subj_([A-Za-z0-9]+)_svc', file_path)
    if match:
        return match.group(1)
    # Sinon, prend le nom du dossier parent (souvent l'ID)
    return os.path.basename(os.path.dirname(file_path))

def find_pp_npz_files(base_path):
    """Trouve et organise les fichiers NPZ du protocole PP par groupe clinique."""
    logger.info("Recherche des fichiers NPZ PP dans: %s", base_path)
    organized_data = {group: [] for group in ALL_SUBJECTS_GROUPS}
    pattern = os.path.join(base_path, '**', 'decoding_results_full.npz')
    files = glob.glob(pattern, recursive=True)
    if not files:
        logger.warning("Aucun fichier de résultats NPZ PP trouvé.")
        return {}
    logger.info("%d fichiers de résultats PP trouvés.", len(files))
    # Pour chaque fichier, cherche l'ID dans le chemin et l'associe au bon groupe
    for f in files:
        for group, ids in ALL_SUBJECTS_GROUPS.items():
            for sid in ids:
                if sid in f:
                    organized_data[group].append(f)
                    break  # Un fichier ne va que dans un groupe
    # Retirer les groupes vides
    organized_data = {g: fl for g, fl in organized_data.items() if fl}
    return organized_data

def load_npz_data_pp(file_path):
    """Charge les données NPZ du protocole PP."""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            scores = data['scores'] if 'scores' in data else None
            times = data['times'] if 'times' in data else None
            return {'scores': scores, 'times': times}
    except Exception as e:
        logger.error("Erreur lors du chargement du fichier %s: %s", file_path, e)
        return None

def analyze_group_data_pp(group_files, group_name):
    """Analyse les données d'un groupe pour le protocole PP."""
    logger.info(f"Analyse du groupe {group_name} avec {len(group_files)} sujets")
    group_data = []
    subject_ids = []
    for file_path in group_files:
        data = load_npz_data_pp(file_path)
        if data is not None and data['scores'] is not None:
            group_data.append(data)
            subject_ids.append(os.path.basename(os.path.dirname(file_path)))
    if not group_data:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return {}
    scores_matrix = np.array([d['scores'] for d in group_data])
    times = group_data[0]['times'] if group_data[0]['times'] is not None else None
    group_mean = np.nanmean(scores_matrix, axis=0)
    group_std = np.nanstd(scores_matrix, axis=0)
    group_sem = group_std / np.sqrt(len(group_data))
    return {
        'group_name': group_name,
        'n_subjects': len(group_data),
        'subject_ids': subject_ids,
        'scores_matrix': scores_matrix,
        'group_mean': group_mean,
        'group_std': group_std,
        'group_sem': group_sem,
        'times': times,
        'subject_means': np.nanmean(scores_matrix, axis=1),
        'group_data': group_data
    }

def plot_group_individual_curves_pp(group_data, save_dir, show_plots=True):
    """Trace les courbes individuelles et la moyenne pour un groupe PP."""
    group_name = group_data['group_name']
    times = group_data['times']
    if times is None:
        logger.error(f"Pas de données temporelles pour le groupe {group_name}")
        return
    if np.max(times) <= 2:
        times_ms = times * 1000
    else:
        times_ms = times
    group_color = '#1f77b4'
    individual_alpha = 0.2
    mean_alpha = 0.8
    plt.figure(figsize=(16, 8))
    for i in range(group_data['scores_matrix'].shape[0]):
        plt.plot(times_ms, group_data['scores_matrix'][i], color=group_color, alpha=individual_alpha)
    plt.plot(times_ms, group_data['group_mean'], color=group_color, alpha=mean_alpha, linewidth=3, label=f'{group_name} (n={group_data["n_subjects"]})')
    plt.fill_between(times_ms, group_data['group_mean'] - group_data['group_sem'], group_data['group_mean'] + group_data['group_sem'], color=group_color, alpha=0.3)
    plt.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Score AUC', fontsize=14)
    plt.title(f'PP - {group_name}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 0.8])
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"group_{group_name}_individual_curves_PP.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

# --- Couleurs et mapping groupes (adapter si besoin) ---
GROUP_COLORS = {
    'group_COMA': '#9467bd',
    'group_CONTROLS_COMA': '#2ca02c',
    'group_VS': '#8c564b',
    'group_DELIRIUM+': '#d62728',
    'group_DELIRIUM-': '#ff7f0e',
    'group_CONTROLS_DELIRIUM': '#2ca02c',
    'group_MCS': '#1f77b4',
}

# --- Analyse multi-groupes : courbes moyennes superposées ---
def plot_all_groups_comparison_pp(all_groups_data, save_dir, show_plots=True):
    if not all_groups_data:
        logger.warning("Aucune donnée de groupe fournie pour la comparaison.")
        return
    plt.figure(figsize=(18, 8))
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        color = GROUP_COLORS.get(group_name, '#1f77b4')
        times = group_data['times']
        if np.max(times) <= 2:
            times_ms = times * 1000
        else:
            times_ms = times
        plt.plot(times_ms, group_data['group_mean'], label=f"{group_name} (n={group_data['n_subjects']})", color=color, linewidth=3)
        plt.fill_between(times_ms, group_data['group_mean'] - group_data['group_sem'], group_data['group_mean'] + group_data['group_sem'], color=color, alpha=0.2)
    plt.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Score AUC', fontsize=14)
    plt.title('Comparaison multi-groupes (PP)', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.4, 0.8])
    plt.tight_layout()
    if save_dir:
        filename = "all_groups_comparison_PP.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique de comparaison sauvegardé: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

# --- Boxplots AUC globaux par groupe avec stats ---
def plot_global_auc_boxplots_pp(all_groups_data, save_dir, show_plots=True):
    auc_data = []
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        for auc in group_data['subject_means']:
            auc_data.append({'Group': group_name, 'AUC': auc})
    if len(set([d['Group'] for d in auc_data])) < 2:
        logger.warning("Pas assez de groupes pour boxplot/stats.")
        return
    df = pd.DataFrame(auc_data)
    plt.figure(figsize=(12, 8))
    group_colors = [GROUP_COLORS.get(g, '#1f77b4') for g in df['Group'].unique()]
    sns.boxplot(data=df, x='Group', y='AUC', palette=group_colors)
    sns.stripplot(data=df, x='Group', y='AUC', color='black', alpha=0.6, size=4)
    plt.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')
    plt.title('Distribution des AUC globaux par groupe (PP)', fontsize=16, fontweight='bold')
    plt.xlabel('Groupe clinique', fontsize=14)
    plt.ylabel('AUC (Area Under Curve)', fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()
    if save_dir:
        filename = "boxplot_auc_global_PP.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Boxplot sauvegardé: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()
    # Stats Mann-Whitney U
    from itertools import combinations
    group_names = df['Group'].unique()
    for g1, g2 in combinations(group_names, 2):
        auc1 = df[df['Group'] == g1]['AUC']
        auc2 = df[df['Group'] == g2]['AUC']
        stat, p = mannwhitneyu(auc1, auc2, alternative='two-sided')
        logger.info(f"Mann-Whitney U {g1} vs {g2}: p={p:.4f}")

# --- Analyse sur fenêtres temporelles ---
def plot_temporal_windows_boxplots_pp(all_groups_data, save_dir, show_plots=True):
    windows = {
        'T100': (80, 120),
        'T200': (180, 220),
        'T_all': (0, 600)
    }
    all_data = []
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        times = group_data['times']
        if np.max(times) <= 2:
            times_ms = times * 1000
        else:
            times_ms = times
        for win_name, (tmin, tmax) in windows.items():
            idx = np.where((times_ms >= tmin) & (times_ms <= tmax))[0]
            for subj, subj_scores in enumerate(group_data['scores_matrix']):
                mean_auc = np.nanmean(subj_scores[idx])
                all_data.append({'Group': group_name, 'Window': win_name, 'AUC': mean_auc})
    df = pd.DataFrame(all_data)
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Window', y='AUC', hue='Group')
    plt.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')
    plt.title('AUC par fenêtre temporelle et groupe (PP)', fontsize=16)
    plt.xlabel('Fenêtre temporelle', fontsize=14)
    plt.ylabel('AUC', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        filename = "boxplot_auc_windows_PP.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Boxplot fenêtres temporelles sauvegardé: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

# --- Analyse des proportions de sujets significatifs ---
def analyze_significance_proportions_pp(all_groups_data, save_dir, show_plots=True):
    prop_data = []
    for group_data in all_groups_data:
        group_name = group_data['group_name']
        n = group_data['n_subjects']
        sig_count = 0
        for subj_scores in group_data['scores_matrix']:
            # Test Wilcoxon vs. chance
            try:
                from scipy.stats import wilcoxon
                stat, p = wilcoxon(subj_scores - CHANCE_LEVEL, alternative='greater')
                if p < 0.05:
                    sig_count += 1
            except Exception:
                continue
        prop = sig_count / n if n > 0 else 0
        prop_data.append({'Group': group_name, 'Proportion': prop, 'n': n})
    plt.figure(figsize=(10, 6))
    for d in prop_data:
        plt.bar(d['Group'], d['Proportion'], color=GROUP_COLORS.get(d['Group'], '#1f77b4'))
        plt.text(d['Group'], d['Proportion'] + 0.02, f"{int(d['Proportion']*d['n'])}/{d['n']}", ha='center', fontsize=12)
    plt.ylim(0, 1.05)
    plt.ylabel('Proportion de sujets significatifs (p<0.05)')
    plt.title('Proportion de sujets significatifs par groupe (PP)')
    plt.tight_layout()
    if save_dir:
        filename = "proportion_significatifs_PP.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Proportion sauvegardée: {filepath}")
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    logger.info("Début de l'analyse des données PP")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/tom.balay/results/PP_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    organized_data = find_pp_npz_files(BASE_RESULTS_DIR)
    all_groups_data = []
    for group_name, group_files in organized_data.items():
        group_data = analyze_group_data_pp(group_files, group_name)
        if group_data:
            all_groups_data.append(group_data)
            plot_group_individual_curves_pp(group_data, results_dir, show_plots=False)
    # Analyses multi-groupes
    plot_all_groups_comparison_pp(all_groups_data, results_dir, show_plots=False)
    plot_global_auc_boxplots_pp(all_groups_data, results_dir, show_plots=False)
    plot_temporal_windows_boxplots_pp(all_groups_data, results_dir, show_plots=False)
    analyze_significance_proportions_pp(all_groups_data, results_dir, show_plots=False)
    logger.info("Analyse PP terminée. Courbes et analyses multi-groupes sauvegardées.")

if __name__ == "__main__":
    main()
