import os
import sys
import glob
import logging
import warnings
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import re

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
except ImportError:
    sys.exit(1)


try:
    from config.config import ALL_SUBJECTS_GROUPS
except ImportError:
    print("AVERTISSEMENT: Impossible d'importer ALL_SUBJECTS_GROUPS depuis config.config")



warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)



BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V17/intra_subject_lg_results"

GROUP_NAME_MAPPING = {
    'COMA': 'Coma',
    'CONTROLS_COMA': 'Controls',
    'CONTROLS_DELIRIUM': 'Controls',
    'VS': 'VS/UWS',
    'DELIRIUM+': 'Delirium +',
    'DELIRIUM-': 'Delirium -',
    'MCS': 'MCS',
    'CONTROLS': 'Controls' # Fallback
}

# Définir l'ordre des patients
PATIENT_ORDER = ['Controls', 'Delirium -', 'Delirium +', 'MCS', 'VS/UWS', 'Coma']

GROUP_COLORS = {
    'Controls (Delirium)': '#2ca02c', 'Delirium -': '#ff7f0e', 'Delirium +': '#d62728',
    'Controls (Coma)': '#17becf', 'MCS': '#1f77b4',
    'Coma': '#9467bd', 'VS/UWS': '#8c564b',
    'Controls': '#2ca02c' # Fallback
}

KEY_SUFFIXES = {
    'scores': '_scores_1d', 'times': 'epochs_time_points'
}


CHANCE_LEVEL = 0.5
N_PERMUTATIONS = 1000
FDR_ALPHA = 0.05

PUBLICATION_PARAMS = {
    'figure.figsize': (16, 9), 'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 20,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14, 'lines.linewidth': 2.5,
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1
}
plt.rcParams.update(PUBLICATION_PARAMS)

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def extract_subject_id_from_path(file_path):
    """
    
    Tente d'abord de trouver un ID connu depuis la configuration, puis utilise des méthodes de secours.
    """
    try:
        from config.config import ALL_SUBJECTS_GROUPS
        all_ids = set()
        for group, ids in ALL_SUBJECTS_GROUPS.items():
            all_ids.update(ids)
     
        for sid in sorted(list(all_ids), key=len, reverse=True): 
            if sid in file_path:
                return sid
    except Exception:
        pass 

    # Méthodes de secours basées sur des patterns dans le chemin
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if '_Subj_' in part:
            return part.split('_Subj_')[1].split('_')[0]
        if 'Subject_' in part:
            return part.split('Subject_')[1].split('_')[0]


    return os.path.basename(os.path.dirname(file_path))


def find_npz_files(base_path):
    """

    Trouve et organise les fichiers NPZ par protocole et groupe clinique.
    """
    logger.info("Recherche des fichiers NPZ dans: %s", base_path)
    organized_data = {}
    
   
    all_files = glob.glob(os.path.join(base_path, '**', '*decoding_results_full.npz'), recursive=True)

    if not all_files:
        logger.warning("Aucun fichier de résultats NPZ trouvé.")
        return {}
    logger.info("%d fichiers de résultats potentiels trouvés.", len(all_files))

  
    try:
        from config.config import ALL_SUBJECTS_GROUPS
        known_groups = sorted(list(ALL_SUBJECTS_GROUPS.keys()), key=len, reverse=True)
    except ImportError:
        known_groups = ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM', 'CONTROLS_COMA', 'MCS', 'VS', 'COMA', 'CONTROLS']

    for file_path in all_files:
        try:
            path_parts = file_path.split(os.sep)
            
            group_name = None
            protocol_name = "UnknownProtocol"

            # Itérer sur les parties du chemin pour trouver le nom du groupe et le protocole
            for part in path_parts:
                # Chercher le nom de groupe le plus spécifique en premier
                for known_group in known_groups:
                    if known_group in part:
                        group_name = known_group
                        protocol_name = part # Le nom du dossier est le nom du protocole
                        break
                if group_name:
                    break
            
            if not group_name:
                logger.warning(f"Impossible d'identifier un groupe connu pour le fichier : {file_path}")
                continue

            if protocol_name not in organized_data:
                organized_data[protocol_name] = {}
            if group_name not in organized_data[protocol_name]:
                organized_data[protocol_name][group_name] = []
            
            organized_data[protocol_name][group_name].append(file_path)
            
        except Exception as e:
            logger.warning(f"Erreur lors du traitement du chemin {file_path}: {e}")
            continue

    return organized_data


def load_npz_data(file_path):
    """
   
    Charge et valide les données du fichier NPZ, extrayant les effets locaux et globaux.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            data_keys = list(data.keys())
            
            # Détecter si c'est un protocole LG en se basant sur la présence des clés spécifiques
            is_lg_protocol = any(key.startswith('lg_') for key in data_keys)

            if not is_lg_protocol:
                # Si ce n'est pas un fichier LG, on l'ignore pour cette analyse
                return None

            # Vérifier la présence de la clé temporelle, essentielle pour toute analyse
            if 'epochs_time_points' not in data_keys:
                logger.warning(f"Champ temporel requis 'epochs_time_points' manquant dans {file_path}.")
                return None
                
            subject_id = extract_subject_id_from_path(file_path)

            # Initialiser le dictionnaire de résultats
            result = {
                'subject_id': subject_id,
                'file_path': file_path,
                'times': data['epochs_time_points'],
                'analysis_type': 'lg_protocol'
            }

            # --- Traitement de l'Effet Local (LS vs LD) ---
            if 'lg_ls_ld_scores_1d_mean' in data_keys:
                result['local_effect_scores'] = data['lg_ls_ld_scores_1d_mean']
                if 'lg_ls_ld_tgm_mean' in data_keys:
                    result['local_effect_tgm'] = data['lg_ls_ld_tgm_mean']
                if 'lg_ls_ld_mean_auc_global' in data_keys:
                    result['local_effect_auc_global'] = data['lg_ls_ld_mean_auc_global']
                # On pourrait aussi extraire les données FDR/cluster ici si nécessaire

            # --- Traitement de l'Effet Global (GS vs GD) ---
            if 'lg_gs_gd_scores_1d_mean' in data_keys:
                result['global_effect_scores'] = data['lg_gs_gd_scores_1d_mean']
                if 'lg_gs_gd_tgm_mean' in data_keys:
                    result['global_effect_tgm'] = data['lg_gs_gd_tgm_mean']
                if 'lg_gs_gd_mean_auc_global' in data_keys:
                    result['global_effect_auc_global'] = data['lg_gs_gd_mean_auc_global']
            
            # Vérifier qu'au moins un des deux effets a été trouvé
            if 'local_effect_scores' not in result and 'global_effect_scores' not in result:
                logger.warning(f"Aucune donnée d'effet local ou global trouvée dans le fichier LG : {file_path}")
                return None

            return result

    except Exception as e:
        logger.error("Erreur lors du chargement du fichier NPZ %s: %s", file_path, e)
        return None

def analyze_group_data_lg(group_files, group_name):
    """
    Analyser les données d'un groupe spécifique pour le protocole LG avec effets local et global séparés.
    """
    logger.info(f"Analyse LG du groupe {group_name} avec {len(group_files)} sujets")
    
    group_data_list = []
    subject_ids = []
    local_scores_list = []
    global_scores_list = []
    local_tgm_list = []
    global_tgm_list = []
    local_auc_global_list = []
    global_auc_global_list = []
    
    for file_path in group_files:
        data = load_npz_data(file_path)
        if data is not None:
            group_data_list.append(data)
            subject_ids.append(data['subject_id'])
            
            if 'local_effect_scores' in data and data['local_effect_scores'] is not None:
                local_scores_list.append(data['local_effect_scores'])
                if 'local_effect_tgm' in data and data['local_effect_tgm'] is not None:
                    local_tgm_list.append(data['local_effect_tgm'])
                if 'local_effect_auc_global' in data:
                    local_auc_global_list.append(data['local_effect_auc_global'])
                    
            if 'global_effect_scores' in data and data['global_effect_scores'] is not None:
                global_scores_list.append(data['global_effect_scores'])
                if 'global_effect_tgm' in data and data['global_effect_tgm'] is not None:
                    global_tgm_list.append(data['global_effect_tgm'])
                if 'global_effect_auc_global' in data:
                    global_auc_global_list.append(data['global_effect_auc_global'])
    
    if not group_data_list:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return {}
    
    reference_length = 801
    times = group_data_list[0]['times'][:reference_length] if group_data_list[0]['times'] is not None else None
    
    result = {
        'group_name': group_name,
        'n_subjects': len(group_data_list),
        'subject_ids': subject_ids,
        'times': times,
        'group_data': group_data_list
    }
    

    if local_scores_list:
        local_standardized = []
        for scores in local_scores_list:
            if scores is not None and len(scores) >= reference_length:
                scores_clean = np.nan_to_num(scores[:reference_length], nan=CHANCE_LEVEL)
                local_standardized.append(scores_clean)
        
        if local_standardized:
            local_matrix = np.array(local_standardized)
            result['local_effect'] = {
                'scores_matrix': local_matrix,
                'group_mean': np.nanmean(local_matrix, axis=0),
                'group_std': np.nanstd(local_matrix, axis=0),
                'group_sem': np.nanstd(local_matrix, axis=0) / np.sqrt(len(local_standardized))
            }
            if local_tgm_list:
               
                result['local_effect']['tgm_mean'] = np.nanmean(np.array(local_tgm_list), axis=0)
            if local_auc_global_list:
                result['local_effect']['auc_global_values'] = np.array(local_auc_global_list)

  
    if global_scores_list:
        global_standardized = []
        for scores in global_scores_list:
            if scores is not None and len(scores) >= reference_length:
                scores_clean = np.nan_to_num(scores[:reference_length], nan=CHANCE_LEVEL)
                global_standardized.append(scores_clean)

        if global_standardized:
            global_matrix = np.array(global_standardized)
            result['global_effect'] = {
                'scores_matrix': global_matrix,
                'group_mean': np.nanmean(global_matrix, axis=0),
                'group_std': np.nanstd(global_matrix, axis=0),
                'group_sem': np.nanstd(global_matrix, axis=0) / np.sqrt(len(global_standardized))
            }
            if global_tgm_list:
               
                result['global_effect']['tgm_mean'] = np.nanmean(np.array(global_tgm_list), axis=0)
            if global_auc_global_list:
                result['global_effect']['auc_global_values'] = np.array(global_auc_global_list)

    logger.info(f"Groupe {group_name} analysé - Effet local: {len(local_scores_list)} sujets, Effet global: {len(global_scores_list)} sujets")
    
    return result


def plot_group_individual_curves(group_data, save_dir, show_plots=True):
    """
    Créer des graphiques pour un groupe avec les courbes individuelles en arrière-plan
    et les moyennes en avant-plan pour les effets local et global.
    """
    group_name = group_data['group_name']
    times = group_data['times']
    
    if times is None:
        logger.error(f"Pas de données temporelles pour le groupe {group_name}")
        return
    
    # Convertir les temps en millisecondes si nécessaire
    if np.max(times) <= 2:  # Supposer que c'est en secondes
        times_ms = times * 1000
    else:
        times_ms = times
    
    # Couleur du groupe
    group_color = GROUP_COLORS.get(GROUP_NAME_MAPPING.get(group_name, group_name), '#1f77b4')
    individual_alpha = 0.2
    mean_alpha = 0.8
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True) # Ajout de sharey pour une meilleure comparaison
    
    # Plot 1: Effet Local (LS vs LD)
    if 'local_effect' in group_data and group_data['local_effect']:
        ax1 = axes[0]
        local_data = group_data['local_effect']
        
        # Assurer que les dimensions correspondent
        min_length = min(len(times_ms), local_data['scores_matrix'].shape[1])
        times_ms_truncated = times_ms[:min_length]
        
        # Courbes individuelles en arrière-plan
        for i in range(local_data['scores_matrix'].shape[0]):
            scores_truncated = local_data['scores_matrix'][i, :min_length]
            ax1.plot(times_ms_truncated, scores_truncated, 
                    color=group_color, alpha=individual_alpha, linewidth=1)
        
        # Moyenne du groupe en avant-plan
        group_mean_truncated = local_data['group_mean'][:min_length]
        group_sem_truncated = local_data['group_sem'][:min_length]
        
        ax1.plot(times_ms_truncated, group_mean_truncated, 
                color=group_color, alpha=mean_alpha, linewidth=3, 
                label=f'{GROUP_NAME_MAPPING.get(group_name, group_name)} (n={group_data["n_subjects"]})')
        
        # Bande d'erreur (SEM)
        ax1.fill_between(times_ms_truncated, 
                        group_mean_truncated - group_sem_truncated,
                        group_mean_truncated + group_sem_truncated,
                        color=group_color, alpha=0.3)
        
        ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Time (ms)', fontsize=14)
        ax1.set_ylabel('Score AUC', fontsize=14)
        ax1.set_title(f'Local Effect (LS vs LD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.4, 0.8])
    else:
        axes[0].text(0.5, 0.5, 'No Local Effect Data', 
                    transform=axes[0].transAxes, ha='center', va='center', fontsize=16)
        axes[0].set_title(f'Local Effect (LS vs LD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
    
    # Plot 2: Effet Global (GS vs GD)
    if 'global_effect' in group_data and group_data['global_effect']:
        ax2 = axes[1]
        global_data = group_data['global_effect']
        
        min_length = min(len(times_ms), global_data['scores_matrix'].shape[1])
        times_ms_truncated = times_ms[:min_length]
        
        for i in range(global_data['scores_matrix'].shape[0]):
            scores_truncated = global_data['scores_matrix'][i, :min_length]
            ax2.plot(times_ms_truncated, scores_truncated, 
                    color=group_color, alpha=individual_alpha, linewidth=1)
        
        group_mean_truncated = global_data['group_mean'][:min_length]
        group_sem_truncated = global_data['group_sem'][:min_length]
        
        ax2.plot(times_ms_truncated, group_mean_truncated, 
                color=group_color, alpha=mean_alpha, linewidth=3, 
                label=f'{GROUP_NAME_MAPPING.get(group_name, group_name)} (n={group_data["n_subjects"]})')
        
        ax2.fill_between(times_ms_truncated, 
                        group_mean_truncated - group_sem_truncated,
                        group_mean_truncated + group_sem_truncated,
                        color=group_color, alpha=0.3)
        
        ax2.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time (ms)', fontsize=14)
        ax2.set_title(f'Global Effect (GS vs GD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Global Effect Data', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=16)
        axes[1].set_title(f'Global Effect (GS vs GD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"group_{group_name.replace('/', '_')}_individual_curves.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique sauvegardé: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_all_groups_comparison(all_groups_data, save_dir, show_plots=True):
    """
    Créer des graphiques de comparaison entre tous les groupes pour les effets locaux et globaux.
    """
    if not all_groups_data:
        logger.error("Aucune donnée de groupe fournie")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot 1: Comparaison des effets locaux
    ax1 = axes[0]
    for group_data in all_groups_data:
        if 'local_effect' in group_data:
            group_name = group_data['group_name']
            local_data = group_data['local_effect']
            group_color = GROUP_COLORS.get(GROUP_NAME_MAPPING.get(group_name, group_name), '#1f77b4')
            
            # Obtenir les temps pour ce groupe spécifique
            times = group_data.get('times')
            if times is None:
                logger.warning(f"Pas de données temporelles pour le groupe {group_name}")
                continue
            
            # Convertir les temps en millisecondes si nécessaire
            if np.max(times) <= 2:
                times_ms = times * 1000
            else:
                times_ms = times
            
            # Assurer que les dimensions correspondent
            min_length = min(len(times_ms), len(local_data['group_mean']))
            times_ms_truncated = times_ms[:min_length]
            group_mean_truncated = local_data['group_mean'][:min_length]
            group_sem_truncated = local_data['group_sem'][:min_length]
            
            # Moyenne du groupe
            ax1.plot(times_ms_truncated, group_mean_truncated, 
                    color=group_color, linewidth=3, 
                    label=f'{GROUP_NAME_MAPPING.get(group_name, group_name)} (n={group_data["n_subjects"]})')
            
            # Bande d'erreur (SEM)
            ax1.fill_between(times_ms_truncated, 
                            group_mean_truncated - group_sem_truncated,
                            group_mean_truncated + group_sem_truncated,
                            color=group_color, alpha=0.3)
    
    # Configuration du plot local
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (ms)', fontsize=14)
    ax1.set_ylabel('Score AUC', fontsize=14)
    ax1.set_title('Local Effect Comparison (LS vs LD)', fontsize=18)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.4, 0.8])
    
    # Plot 2: Comparaison des effets globaux
    ax2 = axes[1]
    for group_data in all_groups_data:
        if 'global_effect' in group_data:
            group_name = group_data['group_name']
            global_data = group_data['global_effect']
            group_color = GROUP_COLORS.get(GROUP_NAME_MAPPING.get(group_name, group_name), '#1f77b4')
            
            # Obtenir les temps pour ce groupe spécifique
            times = group_data.get('times')
            if times is None:
                logger.warning(f"Pas de données temporelles pour le groupe {group_name}")
                continue
            
            # Convertir les temps en millisecondes si nécessaire
            if np.max(times) <= 2:
                times_ms = times * 1000
            else:
                times_ms = times
            
            # Assurer que les dimensions correspondent
            min_length = min(len(times_ms), len(global_data['group_mean']))
            times_ms_truncated = times_ms[:min_length]
            group_mean_truncated = global_data['group_mean'][:min_length]
            group_sem_truncated = global_data['group_sem'][:min_length]
            
            # Moyenne du groupe
            ax2.plot(times_ms_truncated, group_mean_truncated, 
                    color=group_color, linewidth=3, 
                    label=f'{GROUP_NAME_MAPPING.get(group_name, group_name)} (n={group_data["n_subjects"]})')
            
            # Bande d'erreur (SEM)
            ax2.fill_between(times_ms_truncated, 
                            group_mean_truncated - group_sem_truncated,
                            group_mean_truncated + group_sem_truncated,
                            color=group_color, alpha=0.3)
    
    # Configuration du plot global
    ax2.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time (ms)', fontsize=14)
    ax2.set_ylabel('Score AUC', fontsize=14)
    ax2.set_title('Global Effect Comparison (GS vs GD)', fontsize=18)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 0.8])
    
    plt.tight_layout()
    
    # Sauvegarder
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = "all_groups_comparison_LG.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Graphique de comparaison sauvegardé: {filepath}")
    
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


def plot_global_auc_boxplots(all_groups_data, save_dir, show_plots=True):
    """
    Créer des boxplots pour les métriques globales AUC avec tests statistiques.
    
    Args:
        all_groups_data: Liste des données de tous les groupes
        save_dir: Répertoire pour sauvegarder les graphiques
        show_plots: Afficher les graphiques
    """
    if not all_groups_data:
        logger.warning("Aucune donnée de groupe disponible pour les boxplots")
        return

    for effect_type in ['local', 'global']:
        logger.info(f"Création des boxplots pour l'effet {effect_type}")
        
        plot_data = []
        group_subject_counts = {}

        for group_data in all_groups_data:
            group_name = GROUP_NAME_MAPPING.get(group_data['group_name'], group_data['group_name'])
            effect_key = f'{effect_type}_effect'
            
            if effect_key in group_data and 'auc_global_values' in group_data[effect_key]:
                auc_values = group_data[effect_key]['auc_global_values']
                auc_values = auc_values[~np.isnan(auc_values)]
                
                if len(auc_values) > 0:
                    group_subject_counts[group_name] = len(auc_values)
                    for value in auc_values:
                        plot_data.append({
                            'Group': group_name,
                            'AUC': value
                        })
        
        if not plot_data:
            logger.warning(f"Pas assez de données pour créer le boxplot de l'effet {effect_type}")
            continue
        
        df = pd.DataFrame(plot_data)

        # Assurer que l'ordre des groupes est respecté
        ordered_groups = [group for group in PATIENT_ORDER if group in df['Group'].unique()]

        stats_results = perform_statistical_tests(all_groups_data, effect_type)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        group_colors = [GROUP_COLORS.get(group, '#1f77b4') for group in ordered_groups]
        
        sns.boxplot(data=df, x='Group', y='AUC', order=ordered_groups, ax=ax, palette=group_colors)
        sns.stripplot(data=df, x='Group', y='AUC', order=ordered_groups, ax=ax, color='black', alpha=0.6, size=4, jitter=True)

        ax.set_title(f'Global AUC Distribution - {effect_type.capitalize()} Effect', fontsize=16, fontweight='bold')
        ax.set_xlabel('Clinical Group', fontsize=14)
        ax.set_ylabel('AUC (Area Under Curve)', fontsize=14)
        ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')

  
        new_labels = [f"{group} (n={group_subject_counts.get(group, 0)})" for group in ordered_groups]
        ax.set_xticklabels(new_labels)

        if stats_results.get('pairwise_results'):
            y_max = df['AUC'].max()
            y_positions = [y_max + 0.05]
            add_significance_bars(ax, df, stats_results, y_positions)

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min - 0.02, y_max + 0.15)
        ax.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
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


def plot_group_tgm_individual(all_groups_data, save_dir, show_plots=True):
    """
    Sauvegarde et affiche la TGM moyenne pour chaque groupe individuellement avec contours significatifs.
    Exclut les groupes avec un seul sujet car les tests statistiques ne sont pas possibles.
    """
    for effect_type in ['local', 'global']:
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            n_subjects = group_data['n_subjects']
            
            # Exclure les groupes avec un seul sujet
            if n_subjects < 2:
                logger.info(f"Groupe {group_name} - {effect_type}: {n_subjects} sujet(s), TGM exclue (minimum 2 sujets requis)")
                continue
                
            effect_key = f'{effect_type}_effect'
            
            if effect_key in group_data and 'tgm_mean' in group_data[effect_key]:
                tgm_mean = group_data[effect_key]['tgm_mean']
                tgm_matrix = group_data[effect_key].get('tgm_matrix')
                times = group_data.get('times')
                
                if times is not None:
                    if np.max(times) <= 2:
                        times_ms = times * 1000
                    else:
                        times_ms = times
                else:
                    times_ms = np.arange(tgm_mean.shape[0])
                
                # Effectuer les tests statistiques si on a la matrice des sujets
                fdr_mask = None
                if tgm_matrix is not None:
                    n_subjects = tgm_matrix.shape[0]
                    if n_subjects > 1:
                        try:
                            # Effectuer le test FDR pour chaque point temporel de la TGM
                            logger.info(f"Effectuation du test FDR pour {group_name} - {effect_type} TGM ({n_subjects} sujets)")
                            
                            # Créer une matrice de test contre le niveau de chance
                            n_train_times, n_test_times = tgm_matrix.shape[1], tgm_matrix.shape[2]
                            fdr_mask = np.zeros((n_train_times, n_test_times), dtype=bool)
                            
                            # Test FDR pour chaque point de la TGM (approche point par point)
                            significant_points = 0
                            total_valid_points = 0
                            
                            for train_idx in range(n_train_times):
                                for test_idx in range(n_test_times):
                                    # Scores de tous les sujets pour ce point temporel
                                    scores_at_point = tgm_matrix[:, train_idx, test_idx]
                                    
                                    # Supprimer les NaN
                                    scores_clean = scores_at_point[~np.isnan(scores_at_point)]
                                    
                                    if len(scores_clean) > 1:  # Besoin d'au moins 2 observations pour le test
                                        total_valid_points += 1
                                        # Test contre le niveau de chance
                                        try:
                                            fdr_result = perform_pointwise_fdr_correction_on_scores(
                                                scores_clean.reshape(-1, 1),  # Shape: (n_subjects, 1) pour un seul point
                                                chance_level=CHANCE_LEVEL,
                                                alpha_significance_level=FDR_ALPHA,
                                                fdr_correction_method="indep",
                                                alternative_hypothesis="two-sided",
                                                statistical_test_type="wilcoxon"
                                            )
                                            
                                            # Récupérer le masque de significativité
                                            _, significant_mask, _, _ = fdr_result
                                            fdr_mask[train_idx, test_idx] = significant_mask[0]
                                            
                                            if significant_mask[0]:
                                                significant_points += 1
                                                
                                        except Exception as e:
                                            logger.warning(f"Erreur FDR pour point ({train_idx}, {test_idx}): {e}")
                                            fdr_mask[train_idx, test_idx] = False
                            
                            logger.info(f"FDR terminé pour {group_name} - {effect_type}: {significant_points} points significatifs sur {total_valid_points} points valides")
                            
                        except Exception as e:
                            logger.error(f"Erreur lors du test FDR pour {group_name} - {effect_type}: {e}")
                            fdr_mask = None
                    else:
                        logger.info(f"Groupe {group_name} - {effect_type}: un seul sujet, TGM exclue de l'analyse")
             
                        continue
                
                # Créer la figure
                fig, ax = plt.subplots(figsize=(8, 7))
                
                # Afficher la TGM
                im = ax.imshow(tgm_mean, origin='lower', aspect='auto',
                              extent=[times_ms[0], times_ms[-1], times_ms[0], times_ms[-1]],
                              vmin=np.nanmin(tgm_mean), vmax=np.nanmax(tgm_mean), cmap='RdYlBu_r')
                
                # Ajouter les contours significatifs si disponibles
                if fdr_mask is not None:
                    # Créer les contours pour les régions significatives
                    contour_x = np.linspace(times_ms[0], times_ms[-1], fdr_mask.shape[1])
                    contour_y = np.linspace(times_ms[0], times_ms[-1], fdr_mask.shape[0])
                    contour_X, contour_Y = np.meshgrid(contour_x, contour_y)
                    
                    # Dessiner les contours significatifs
                    contours = ax.contour(contour_X, contour_Y, fdr_mask.astype(int), 
                                        levels=[0.5], colors='black', linewidths=1, alpha=0.8)
                    
                    # Ajouter une légende pour les contours
                    if len(contours.collections) > 0:
                        ax.text(0.02, 0.98, f'FDR corrected\n(α = {FDR_ALPHA})', 
                               transform=ax.transAxes, fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                               verticalalignment='top')
                
                # Ajouter les lignes de référence
                ax.plot([times_ms[0], times_ms[-1]], [times_ms[0], times_ms[-1]], 'k--', alpha=0.5, linewidth=1)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Personnaliser le graphique
                ax.set_xlabel('Test Time (ms)', fontsize=12)
                ax.set_ylabel('Train Time (ms)', fontsize=12)
                mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
                ax.set_title(f'{mapped_name} - {effect_type.capitalize()} TGM\n(n={group_data["n_subjects"]})', 
                           fontsize=14, fontweight='bold')
                
                # Ajouter la colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Score AUC', rotation=270, labelpad=20)
                
                # Ajuster le layout
                plt.tight_layout()
                
                # Sauvegarder
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    filename = f"tgm_{effect_type}_group_{group_name}_with_fdr.png"
                    filepath = os.path.join(save_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    logger.info(f"TGM individuelle avec FDR sauvegardée: {filepath}")
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()

def create_temporal_windows_comparison_boxplots(all_groups_data, save_dir, show_plots=True):
    """

    
    Args:
        all_groups_data: Liste des données de tous les groupes
        save_dir: Répertoire pour sauvegarder les graphiques
        show_plots: Afficher les graphiques
    """
    # Définir les fenêtres temporelles d'intérêt (en ms)
    windows = {
        'T100': (90, 110),    # Fenêtre autour de 100ms
        'T200': (190, 210),   # Fenêtre autour de 200ms
        'T_all': (0, 800)     # Fenêtre complète
    }
    
    for effect_type in ['local', 'global']:
        logger.info(f"Création des boxplots de comparaison des fenêtres temporelles pour l'effet {effect_type}")
        
        # Collecter toutes les données pour tous les groupes
        all_data = []
        
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            effect_key = f'{effect_type}_effect'
            
            if effect_key not in group_data or 'scores_matrix' not in group_data[effect_key]:
                continue
            
            effect_data = group_data[effect_key]
            scores_matrix = effect_data['scores_matrix']
            times = group_data.get('times')
            n_subjects = scores_matrix.shape[0]
            
            if times is None:
                continue
            
            # Convertir les temps en millisecondes si nécessaire
            if np.max(times) <= 2:
                times_ms = times * 1000
            else:
                times_ms = times
            
            # Calculer les AUC pour chaque sujet et chaque fenêtre
            for subj_idx in range(scores_matrix.shape[0]):
                subject_id = group_data['subject_ids'][subj_idx]
                subject_scores = scores_matrix[subj_idx, :]
                
                # Ajuster les dimensions si nécessaire
                min_length = min(len(times_ms), len(subject_scores))
                times_ms_truncated = times_ms[:min_length]
                subject_scores_truncated = subject_scores[:min_length]
                
                # Calculer les AUC pour chaque fenêtre
                window_aucs = {}
                for window_name, (start_ms, end_ms) in windows.items():
                    start_idx = np.argmin(np.abs(times_ms_truncated - start_ms))
                    end_idx = np.argmin(np.abs(times_ms_truncated - end_ms))
                    
                    if start_idx < end_idx and end_idx <= len(subject_scores_truncated):
                        window_scores = subject_scores_truncated[start_idx:end_idx]
                        window_auc = np.mean(window_scores) 
                        window_aucs[window_name] = window_auc
                
                # Calculer les différences
                if len(window_aucs) >= 2:
                    mapped_group_name = GROUP_NAME_MAPPING.get(group_name, group_name)
                    
                    # T100 vs T_all
                    if 'T100' in window_aucs and 'T_all' in window_aucs:
                        all_data.append({
                            'Group': mapped_group_name,
                            'Subject': subject_id,
                            'Comparison': 'T100',
                            'AUC': window_aucs['T100'],
                            'Window': 'T100'
                        })
                        all_data.append({
                            'Group': mapped_group_name,
                            'Subject': subject_id,
                            'Comparison': 'T_all',
                            'AUC': window_aucs['T_all'],
                            'Window': 'T_all'
                        })
                    
                    # T200 vs T_all
                    if 'T200' in window_aucs and 'T_all' in window_aucs:
                        all_data.append({
                            'Group': mapped_group_name,
                            'Subject': subject_id,
                            'Comparison': 'T200',
                            'AUC': window_aucs['T200'],
                            'Window': 'T200'
                        })
        
        if len(all_data) == 0:
            logger.warning(f"Pas de données pour créer les boxplots des fenêtres temporelles pour l'effet {effect_type}")
            continue
        
        # Créer le DataFrame
        df = pd.DataFrame(all_data)
        
        # Créer les boxplots pour chaque groupe
        unique_groups = df['Group'].unique()
        n_groups = len(unique_groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 8))
        if n_groups == 1:
            axes = [axes]
        
        for idx, group_name in enumerate(unique_groups):
            ax = axes[idx]
            group_data_df = df[df['Group'] == group_name]
            
            # Créer le boxplot pour ce groupe
            windows_to_plot = ['T100', 'T200', 'T_all']
            group_color = GROUP_COLORS.get(group_name, '#1f77b4')
            
            for window_idx, window in enumerate(windows_to_plot):
                window_data = group_data_df[group_data_df['Window'] == window]
                if len(window_data) > 0:
                    # Boxplot
                    bp = ax.boxplot(window_data['AUC'], positions=[window_idx], 
                                   patch_artist=True, widths=0.6)
                    bp['boxes'][0].set_facecolor(group_color)
                    bp['boxes'][0].set_alpha(0.7)
                    
                    # Points individuels
                    for _, row in window_data.iterrows():
                        ax.plot(window_idx, row['AUC'], 'o', color='black', 
                               markersize=4, alpha=0.7)
                    
                    # Connecter les points du même sujet
                    subjects = window_data['Subject'].unique()
                    for subject in subjects:
                        subject_data = group_data_df[group_data_df['Subject'] == subject]
                        if len(subject_data) > 1:
                            subject_data_sorted = subject_data.sort_values('Window')
                            x_coords = [windows_to_plot.index(w) for w in subject_data_sorted['Window']]
                            y_coords = subject_data_sorted['AUC'].values
                            ax.plot(x_coords, y_coords, 'k-', alpha=0.3, linewidth=0.5)
            

            ax.set_xticks(range(len(windows_to_plot)))
            ax.set_xticklabels(windows_to_plot)
            ax.set_ylabel('AUC', fontsize=12)
            ax.set_title(f'{group_name}\n{effect_type.capitalize()} Effect', fontsize=14, fontweight='bold')
            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 0.8)
            
           
            if len(group_data_df) >= 6:  # Au moins 2 sujets avec 3 fenêtres
                # Test de Wilcoxon pour comparer T100 vs T_all et T200 vs T_all
                try:
                    t100_data = group_data_df[group_data_df['Window'] == 'T100']['AUC']
                    t200_data = group_data_df[group_data_df['Window'] == 'T200']['AUC']
                    tall_data = group_data_df[group_data_df['Window'] == 'T_all']['AUC']
                    
                    # Test T100 vs T_all
                    if len(t100_data) > 1 and len(tall_data) > 1:
                        from scipy.stats import wilcoxon
                        _, p_100_all = wilcoxon(t100_data, tall_data)
                        
                        # Ajouter les étoiles de significativité
                        if p_100_all < 0.01:
                            ax.text(0.5, 0.9, '**', transform=ax.transAxes, ha='center', 
                                   fontsize=16, fontweight='bold')
                        elif p_100_all < 0.05:
                            ax.text(0.5, 0.9, '*', transform=ax.transAxes, ha='center', 
                                   fontsize=16, fontweight='bold')
                    
                    # Test T200 vs T_all
                    if len(t200_data) > 1 and len(tall_data) > 1:
                        _, p_200_all = wilcoxon(t200_data, tall_data)
                        
                        # Ajouter les étoiles de significativité
                        if p_200_all < 0.01:
                            ax.text(0.8, 0.9, '**', transform=ax.transAxes, ha='center', 
                                   fontsize=16, fontweight='bold')
                        elif p_200_all < 0.05:
                            ax.text(0.8, 0.9, '*', transform=ax.transAxes, ha='center', 
                                   fontsize=16, fontweight='bold')
                            
                except Exception as e:
                    logger.warning(f"Erreur lors des tests statistiques pour {group_name}: {e}")
        
        plt.tight_layout()
        

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"temporal_windows_comparison_boxplots_{effect_type}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplots de comparaison des fenêtres temporelles sauvegardés: {filepath}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def create_temporal_windows_connected_plots(all_groups_data, save_dir, show_plots=True):
    """
    Créer des graphiques avec connexions entre fenêtres temporelles pour chaque groupe,
    similaires à l'image de référence avec les lignes connectant les points.
    
    Args:
        all_groups_data: Liste des données de tous les groupes
        save_dir: Répertoire pour sauvegarder les graphiques
        show_plots: Afficher les graphiques
    """
    # Définir les fenêtres temporelles d'intérêt (en ms)
    windows = {
        'T100': (80, 120),    # Fenêtre autour de 100ms
        'T200': (180, 220),   # Fenêtre autour de 200ms
        'T_all': (0, 600)     # Fenêtre complète
    }
    
    for effect_type in ['local', 'global']:
        logger.info(f"Création des graphiques connectés pour l'effet {effect_type}")
        
        # Créer une figure pour chaque groupe
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            effect_key = f'{effect_type}_effect'
            
            if effect_key not in group_data:
                logger.warning(f"Pas de données pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            
            effect_data = group_data[effect_key]
            if 'scores_matrix' not in effect_data:
                logger.warning(f"Pas de matrice de scores pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            
            scores_matrix = effect_data['scores_matrix']
            times = group_data.get('times')
            n_subjects = scores_matrix.shape[0]
            
            if times is None:
                logger.warning(f"Pas de données temporelles pour le groupe {group_name}")
                continue
            
            # Convertir les temps en millisecondes si nécessaire
            if np.max(times) <= 2:
                times_ms = times * 1000
            else:
                times_ms = times
            
            # Couleur du groupe
            group_color = GROUP_COLORS.get(GROUP_NAME_MAPPING.get(group_name, group_name), '#1f77b4')
            
            # Calculer les moyennes des fenêtres pour chaque participant
            subjects_window_means = []
            window_names = ['T100', 'T200', 'T_all']
            
            for subj_idx in range(n_subjects):
                subject_scores = scores_matrix[subj_idx, :]
                
                # Ajuster les dimensions si nécessaire
                min_length = min(len(times_ms), len(subject_scores))
                times_ms_truncated = times_ms[:min_length]
                subject_scores_truncated = subject_scores[:min_length]
                
                subject_means = []
                
                # Calculer la moyenne pour chaque fenêtre
                for window_name in window_names:
                    start_ms, end_ms = windows[window_name]
                    
                    # Trouver les indices correspondant aux fenêtres temporelles
                    start_idx = np.argmin(np.abs(times_ms_truncated - start_ms))
                    end_idx = np.argmin(np.abs(times_ms_truncated - end_ms))
                    
                    if start_idx < end_idx and end_idx <= len(subject_scores_truncated):
                        window_scores = subject_scores_truncated[start_idx:end_idx]
                        window_mean = np.mean(window_scores)
                        subject_means.append(window_mean)
                    else:
                        subject_means.append(np.nan)
                
                subjects_window_means.append(subject_means)
            
            # Créer le graphique avec connexions
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Positions x pour les fenêtres temporelles
            x_positions = np.arange(len(window_names))
            
            # Tracer les connexions pour chaque participant
            for subj_idx, subject_means in enumerate(subjects_window_means):
                # Supprimer les NaN pour ce participant
                valid_indices = ~np.isnan(subject_means)
                if np.sum(valid_indices) > 1:  # Au moins 2 points pour tracer une ligne
                    valid_x = x_positions[valid_indices]
                    valid_y = np.array(subject_means)[valid_indices]
                    
                    # Tracer la ligne de connexion
                    ax.plot(valid_x, valid_y, 'o-', color=group_color, 
                           alpha=0.6, linewidth=2, markersize=8)
            
            # Calculer les moyennes de groupe pour chaque fenêtre
            group_means = []
            group_stds = []
            
            for window_idx in range(len(window_names)):
                window_values = [subject_means[window_idx] for subject_means in subjects_window_means]
                valid_values = [v for v in window_values if not np.isnan(v)]
                
                if len(valid_values) > 0:
                    group_means.append(np.mean(valid_values))
                    group_stds.append(np.std(valid_values))
                else:
                    group_means.append(np.nan)
                    group_stds.append(np.nan)
            
            # Tracer la moyenne de groupe avec une ligne plus épaisse
            valid_group_indices = ~np.isnan(group_means)
            if np.sum(valid_group_indices) > 1:
                valid_group_x = x_positions[valid_group_indices]
                valid_group_y = np.array(group_means)[valid_group_indices]
                
                ax.plot(valid_group_x, valid_group_y, 'o-', color='black', 
                       alpha=0.8, linewidth=4, markersize=12, label='Group Mean')
            
            # Tests statistiques pour chaque fenêtre
            for window_idx, window_name in enumerate(window_names):
                window_values = [subject_means[window_idx] for subject_means in subjects_window_means]
                valid_values = [v for v in window_values if not np.isnan(v)]
                
                if len(valid_values) > 1:
                    from scipy.stats import wilcoxon
                    try:
                        # Test contre le niveau de chance
                        diff_from_chance = np.array(valid_values) - CHANCE_LEVEL
                        stat, p_value = wilcoxon(diff_from_chance, alternative='two-sided')
                        
                        # Ajouter l'annotation de significativité
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            continue  # Ne pas afficher de barre si non significatif
                        
                        # Placer l'annotation en bas du graphique
                        ax.text(window_idx, 0.42, significance, ha='center', va='center', 
                               fontsize=16, fontweight='bold')
                        
                    except Exception as e:
                        logger.warning(f"Erreur dans le test statistique pour {window_name}: {e}")
            
            # Ligne de chance
            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Chance Level')
            
            # Personnaliser le graphique
            ax.set_xlim(-0.5, len(window_names) - 0.5)
            ax.set_ylim(0.4, 1.0)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(window_names, fontsize=14)
            ax.set_ylabel('Score AUC', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # Titre
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            ax.set_title(f'{mapped_name} - {effect_type.capitalize()} Effect\nTemporal Windows Comparison (n={n_subjects})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Sauvegarder
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


def analyze_individual_significance_proportions(all_groups_data, save_dir, show_plots=True):
    """
    Analyser les proportions de patients avec des scores significatifs par rapport à la chance
    pour chaque groupe, similaire à la figure de référence avec les graphiques en camembert.
    
    Args:
        all_groups_data: Liste des données de tous les groupes
        save_dir: Répertoire pour sauvegarder les graphiques
        show_plots: Afficher les graphiques
    """
    from scipy.stats import wilcoxon
    for effect_type in ['local', 'global']:
        logger.info(f"Analyse des proportions de significativité pour l'effet {effect_type}")
        groups_analysis = []
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            effect_key = f'{effect_type}_effect'
            if effect_key not in group_data:
                logger.warning(f"Pas de données pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            effect_data = group_data[effect_key]
            if 'auc_global_values' not in effect_data:
                logger.warning(f"Pas de valeurs AUC globales pour l'effet {effect_type} dans le groupe {group_name}")
                continue
            auc_values = effect_data['auc_global_values']
            valid_auc_values = auc_values[~np.isnan(auc_values)]
            if len(valid_auc_values) == 0:
                logger.warning(f"Aucune valeur AUC valide pour le groupe {group_name}")
                continue
            individual_results = []
            significant_count = 0
            # Correction : chaque sujet n'est ajouté qu'une seule fois dans individual_results
            for subj_idx, auc_value in enumerate(valid_auc_values):
                is_auc_above_chance = auc_value > CHANCE_LEVEL
                statistical_threshold = CHANCE_LEVEL + 0.025  # Seuil conservateur
                is_statistically_significant = auc_value > statistical_threshold
                is_significant = is_auc_above_chance and is_statistically_significant
                individual_results.append({
                    'subject_idx': subj_idx,
                    'auc_value': auc_value,
                    'p_value': np.nan,
                    'is_significant': is_significant,
                    'is_statistically_significant': is_statistically_significant,
                    'is_auc_above_chance': is_auc_above_chance,
                    'statistical_threshold': statistical_threshold
                })
                if is_significant:
                    significant_count += 1
            total_patients = len(individual_results)
            percentage_significant = (significant_count / total_patients) * 100 if total_patients > 0 else 0
            group_p_value = np.nan
            if len(valid_auc_values) > 1:
                try:
                    diff_from_chance = valid_auc_values - CHANCE_LEVEL
                    stat, group_p_value = wilcoxon(diff_from_chance, alternative='greater')
                except Exception as e:
                    logger.warning(f"Erreur test groupe {group_name}: {e}")
            groups_analysis.append({
                'group_name': group_name,
                'n_subjects': total_patients,
                'n_significant': significant_count,
                'percentage_significant': percentage_significant,
                'group_mean_auc': np.mean(valid_auc_values),
                'group_std_auc': np.std(valid_auc_values),
                'group_p_value': group_p_value,
                'individual_results': individual_results,
                'auc_values': valid_auc_values
            })
        if not groups_analysis:
            logger.warning(f"Aucune analyse valide pour l'effet {effect_type}")
            continue
        colors_pie = ['#ff6b6b', '#4ecdc4']
        n_groups = len(groups_analysis)
        fig_pie, axes_pie = plt.subplots(1, n_groups, figsize=(4 * n_groups, 5))
        if n_groups == 1:
            axes_pie = [axes_pie]
        for i, group_analysis in enumerate(groups_analysis):
            group_name = group_analysis['group_name']
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            n_significant = group_analysis['n_significant']
            n_total = group_analysis['n_subjects']
            n_non_significant = n_total - n_significant
            sizes = [n_non_significant, n_significant]
            colors = [colors_pie[0], colors_pie[1]]
            wedges, texts, autotexts = axes_pie[i].pie(sizes, labels=None, colors=colors, autopct='%1.0f%%',
                                                      startangle=90, textprops={'fontsize': 12})
            percentage = group_analysis['percentage_significant']
            axes_pie[i].set_title(f'{mapped_name}\n{percentage:.0f}%\nn={n_total}', fontsize=14, fontweight='bold')
        fig_pie.suptitle(f'{effect_type.capitalize()} Effect - Proportion de sujets significatifs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"individual_significance_proportions_pie_{effect_type}_effect.png"
            filepath = os.path.join(save_dir, filename)
            fig_pie.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Camembert proportions sauvegardé: {filepath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_pie)
        # 2. Figure barres individuelles avec proportion
        fig_bar, axes_bar = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5))
        if n_groups == 1:
            axes_bar = [axes_bar]
        for i, group_analysis in enumerate(groups_analysis):
            group_name = group_analysis['group_name']
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            individual_results = group_analysis['individual_results']
            x_positions = np.arange(len(individual_results))
            for j, result in enumerate(individual_results):
                color = colors_pie[1] if result['is_significant'] else colors_pie[0]
                axes_bar[i].bar(j, result['auc_value'], color=color, alpha=0.7)
            axes_bar[i].axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2, alpha=0.7)
            group_mean = group_analysis['group_mean_auc']
            axes_bar[i].axhline(y=group_mean, color='red', linestyle='-', linewidth=2, alpha=0.8)
            axes_bar[i].set_xlabel('Sujet', fontsize=12)
            axes_bar[i].set_ylabel('AUC', fontsize=12)
            axes_bar[i].set_title(f'{mapped_name} - AUC individuels', fontsize=12, fontweight='bold')
            axes_bar[i].set_ylim(0.4, 0.65)
            axes_bar[i].grid(True, alpha=0.3)
            axes_bar[i].set_xticks(x_positions)
            axes_bar[i].set_xticklabels([f'S{j+1}' for j in range(len(individual_results))], rotation=45)
            percentage = group_analysis['percentage_significant']
            axes_bar[i].text(0.5, 0.95, f'Proportion significative: {percentage:.0f}%',
                             transform=axes_bar[i].transAxes, fontsize=13, color=colors_pie[1], ha='center', va='top', fontweight='bold')
        fig_bar.suptitle(f'{effect_type.capitalize()} Effect - Barres individuelles et proportion', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_dir:
            filename = f"individual_significance_proportions_bar_{effect_type}_effect.png"
            filepath = os.path.join(save_dir, filename)
            fig_bar.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Barres individuelles sauvegardées: {filepath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_bar)
        if save_dir:
            results_file = os.path.join(save_dir, f"significance_proportions_{effect_type}_results.json")
            results_to_save = []
            for group_analysis in groups_analysis:
                group_result = {
                    'group_name': group_analysis['group_name'],
                    'mapped_name': GROUP_NAME_MAPPING.get(group_analysis['group_name'], group_analysis['group_name']),
                    'n_subjects': group_analysis['n_subjects'],
                    'n_significant': group_analysis['n_significant'],
                    'percentage_significant': group_analysis['percentage_significant'],
                    'group_mean_auc': float(group_analysis['group_mean_auc']),
                    'group_std_auc': float(group_analysis['group_std_auc']),
                    'group_p_value': float(group_analysis['group_p_value']) if not np.isnan(group_analysis['group_p_value']) else None,
                    'individual_p_values': [float(r['p_value']) if not np.isnan(r['p_value']) else None for r in group_analysis['individual_results']]
                }
                results_to_save.append(group_result)
            with open(results_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            logger.info(f"Résultats numériques sauvegardés: {results_file}")


def filter_group_files_by_config(group_files, group_name):          
    """
    Filtre les fichiers NPZ pour ne garder que ceux dont l'ID sujet est dans la config ALL_SUBJECTS_GROUPS.
    """
    try:
        from config.config import ALL_SUBJECTS_GROUPS
    except ImportError:
        logger.warning("Impossible d'importer ALL_SUBJECTS_GROUPS depuis config.config, tous les fichiers sont conservés.")
        return group_files


    if group_name not in ALL_SUBJECTS_GROUPS:
        logger.warning(f"Le nom de groupe '{group_name}' n'a pas été trouvé dans la configuration. Aucun fichier ne sera conservé pour ce groupe.")
        return []

    allowed_ids = set(ALL_SUBJECTS_GROUPS.get(group_name, []))
    if not allowed_ids:
        logger.warning(f"Aucun ID sujet n'est défini pour le groupe '{group_name}' dans la configuration.")
        return []

    filtered_files = []
    for file_path in group_files:
        subject_id = extract_subject_id_from_path(file_path)
        if subject_id in allowed_ids:
            filtered_files.append(file_path)
    logger.info(f"Groupe {group_name}: {len(filtered_files)}/{len(group_files)} fichiers NPZ conservés après filtrage par config.")
    return filtered_files

def plot_temporal_windows_boxplots(all_groups_data, save_dir, show_plots=True):
    windows = {
        'T100': (90, 110),
        'T200': (190,  210),
        'T_all': (0, 800)
    }
    for effect_type in ['local', 'global']:
        all_data = []
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            mapped_group_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            effect_key = f'{effect_type}_effect'
            if effect_key in group_data and 'scores_matrix' in group_data[effect_key]:
                scores_matrix = group_data[effect_key]['scores_matrix']
                times = group_data['times']
                if np.max(times) <= 2:
                    times_ms = times * 1000
                else:
                    times_ms = times
                for win_name, (tmin, tmax) in windows.items():
                    idx = np.where((times_ms >= tmin) & (times_ms <= tmax))[0]
                    for subj, subj_scores in enumerate(scores_matrix):
                        mean_auc = np.nanmean(subj_scores[idx])
                        all_data.append({'Group': mapped_group_name, 'Window': win_name, 'AUC': mean_auc, 'Subject': subj})
        if not all_data:
            logger.warning(f"Aucune donnée pour créer les boxplots des fenêtres temporelles pour l'effet {effect_type}")
            continue
        df = pd.DataFrame(all_data)
        # Définir l'ordre des groupes selon PATIENT_ORDER
        ordered_groups = [group for group in PATIENT_ORDER if group in df['Group'].unique()]
        plt.figure(figsize=(14, 8))
       
        group_palette = {group: GROUP_COLORS.get(group, '#1f77b4') for group in ordered_groups}
       
        sns.boxplot(data=df, x='Window', y='AUC', hue='Group', hue_order=ordered_groups, palette=group_palette)
        # Points individuels en noir pour meilleure visibilité
        sns.stripplot(data=df, x='Window', y='AUC', hue='Group', dodge=True, jitter=True, marker='o', alpha=0.7, size=5, hue_order=ordered_groups, color='black')
        plt.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')
        plt.title(f'AUC par fenêtre temporelle et groupe ({effect_type})', fontsize=16)
        plt.xlabel('Fenêtre temporelle', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        handles, labels = plt.gca().get_legend_handles_labels()
        # Afficher la légende uniquement pour les groupes
        if len(ordered_groups) > 0:
            plt.legend(handles[:len(ordered_groups)], ordered_groups, title='Groupe')
        plt.tight_layout()
        if save_dir:
            filename = f"boxplot_auc_windows_{effect_type}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Boxplot fenêtres temporelles sauvegardé: {filepath}")
        if show_plots:
            plt.show()
        else:
            plt.close()

def main():
    """
    Fonction principale pour analyser les données LG et générer les graphiques.
    """
    logger.info("Début de l'analyse des données LG")
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/tom.balay/results/LG_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    

    organized_data = find_npz_files(BASE_RESULTS_DIR)
    all_groups_data = []
    
    if not organized_data:
        logger.error("Aucun fichier NPZ n'a été trouvé ou organisé. Arrêt du script.")
        return

    for protocol_name, groups in organized_data.items():
        for group_name, group_files in groups.items():
            
            filtered_files = filter_group_files_by_config(group_files, group_name)
            if not filtered_files:
                logger.info(f"Aucun fichier NPZ à analyser pour le groupe {group_name} (après filtrage)")
                continue
            

            group_data = analyze_group_data_lg(filtered_files, group_name)

            if group_data and (group_data.get('local_effect') or group_data.get('global_effect')):
                all_groups_data.append(group_data)
    
    
    if all_groups_data:
        # Générer les graphiques pour chaque groupe individuellement
        for group_data in all_groups_data:
            plot_group_individual_curves(group_data, results_dir, show_plots=False)

      
        plot_all_groups_comparison(all_groups_data, results_dir, show_plots=False)
        plot_global_auc_boxplots(all_groups_data, results_dir, show_plots=False)
        plot_temporal_windows_boxplots(all_groups_data, results_dir, show_plots=False)
       
        create_temporal_windows_comparison_boxplots(all_groups_data, results_dir, show_plots=False)
        create_temporal_windows_connected_plots(all_groups_data, results_dir, show_plots=False)

        analyze_individual_significance_proportions(all_groups_data, results_dir, show_plots=False)
        plot_group_tgm_individual(all_groups_data, results_dir, show_plots=False)
      
        summary_file = os.path.join(results_dir, "analysis_summary.json")
        summary_data = {
            'timestamp': timestamp,
            'n_groups_processed': len(all_groups_data),
            'groups': [
                {
                    'name': group['group_name'],
                    'n_subjects': group['n_subjects'],
                    'subject_ids': group['subject_ids'],
                    'has_local_effect': 'local_effect' in group and group['local_effect'] is not None,
                    'has_global_effect': 'global_effect' in group and group['global_effect'] is not None,
                }
                for group in all_groups_data
            ]
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        logger.info(f"Analyse terminée. Résultats sauvegardés dans: {results_dir}")
    else:
        logger.error("Aucune donnée de groupe valide n'a pu être chargée après filtrage. Vérifiez vos chemins et la configuration.")


if __name__ == "__main__":
    main()