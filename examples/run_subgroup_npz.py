import os
import sys
import glob
import logging
import warnings
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import fdrcorrection


current_dir = os.path.dirname(os.path.abspath(__file__))
baking_eeg_dir = os.path.join(current_dir, '..')
if baking_eeg_dir not in sys.path:
    sys.path.insert(0, baking_eeg_dir)


# === DÉFINITION DES SOUS-GROUPES SPÉCIFIQUES ===
SUBGROUP_IDS = {
    'COMA': [
        'TF53', 'CA55', 'JA61', 'ME64', 'MP68', 'SV62', 
        'TpAT19J1', 'TpCF24J1', 'TpEM13J1', 'TpEP16J1', 'TT45', 'YG72'
    ],
    'VS': [
        'FM60', 'KA70', 'MH74', 'OD69', 'SR57', 'AG42', 'SM51',
        'TpAB15J8', 'TpAT19J8', 'TpDC22J8', 'TpFM25J8'
    ],
    'MCS': [
        'AE93', 'CW41', 'DA75', 'GT50', 'HM52', 'JA71', 'KN49',
        'LP54', 'MC58', 'TB56', 'VT47', 'YG66'
    ],
    'Del+': [
        'TpAK24', 'TpAK27', 'TpCB15', 'TpDRL3', 'TpJB25', 'TpJLR17',
        'TpMB45', 'TpMN42', 'TpPC21', 'TpPM14', 'TpRD38'
    ],
    'Control': [
        'TWB1', 'TPC2', 'LAB1', 'LCM2', 'LAT3', 'LBM4', 'LPO5',
        'LAG6', 'MB103', 'FP102', 'FG104'
    ]
}


def calculate_pvalue_intensity(pvalues: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Calcule l'intensité des couleurs basée sur les p-values.
    Plus la p-value est petite, plus l'intensité est forte.
    """
    # Normaliser les p-values à une intensité entre 0.3 et 1.0
    # 0.3 pour les p-values élevées, 1.0 pour les p-values très petites
    intensities = np.clip(1.0 - (pvalues / threshold), 0.3, 1.0)
    return intensities


def calculate_score_intensity(scores: np.ndarray, chance_level: float = 0.5) -> np.ndarray:
    """
    Calcule l'intensité des couleurs basée sur les scores de décodage.
    Plus le score est élevé par rapport au chance level, plus l'intensité est forte.
    """
    # Normaliser les scores à une intensité entre 0.3 et 1.0
    score_diff = scores - chance_level
    max_diff = np.max(np.abs(score_diff))
    if max_diff > 0:
        intensities = np.clip(0.3 + 0.7 * (np.abs(score_diff) / max_diff), 0.3, 1.0)
    else:
        intensities = np.full_like(scores, 0.5)
    return intensities


def get_region_label_info() -> Dict[str, Dict[str, Any]]:
    """
    Retourne les informations des régions temporelles pour l'étiquetage des plots.
    """
    return {
        'stimulus_onset': {'time': 0.0, 'label': 'Stimulus', 'color': 'red'},
        'early_response': {'time': 0.1, 'label': 'Early', 'color': 'blue'},
        'late_response': {'time': 0.3, 'label': 'Late', 'color': 'green'}
    }


def add_region_labels_to_plot(ax, times, region_info):
    """
    Ajouter des étiquettes de région sur un plot.
    
    Parameters:
    - ax: matplotlib axis object
    - times: array des temps
    - region_info: dictionnaire des informations de région
    """
    if not region_info:
        return
    
    # Ajouter des zones colorées pour différentes régions temporelles
    for region_name, info in region_info.items():
        start_time = info.get('start', None)
        end_time = info.get('end', None)
        color = info.get('color', 'lightblue')
        alpha = info.get('alpha', 0.1)
        
        if start_time is not None and end_time is not None:
            # Vérifier que les temps sont dans la plage des données
            if start_time <= times[-1] and end_time >= times[0]:
                ax.axvspan(start_time, end_time, alpha=alpha, color=color, 
                          label=region_name if region_name not in [line.get_label() for line in ax.get_lines()] else "")


try:
    from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
    )
except ImportError:
    print("ERREUR FATALE: Le module 'utils.stats_utils' n'a pas pu être importé.")
    print("Veuillez vous assurer que le script est exécuté depuis le dossier racine du projet.")
    print("Exemple de commande: `python examples/run_subgroup_npz.py` (exécutée depuis le dossier 'Baking_EEG')")
    sys.exit(1)


try:
    from config.config import ALL_SUBJECT_GROUPS
except ImportError:
    print("AVERTISSEMENT: Impossible d'importer ALL_SUBJECT_GROUPS depuis config.config")


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


BASE_RESULTS_DIR = "/home/tom.balay/results/BakingEEG_results_organized_by_protocol"

GROUP_NAME_MAPPING = {
    'group_COMA': 'COMA', 'group_CONTROLS_COMA': 'Controls (Coma)', 'group_MCS+': 'MCS+',
    'group_VS': 'VS', 'group_DELIRIUM+': 'Del+', 'group_DELIRIUM-': 'Delirium -',
    'group_CONTROLS_DELIRIUM': 'Controls (Delirium)', 'del': 'Del+', 'nodel': 'Delirium -',
    'control': 'Control', 'controls': 'Control', 'DELIRIUM+': 'Del+', 'DELIRIUM-': 'Delirium -',
    'deliriumpos': 'Del+', 'deliriumneg': 'Delirium -',
    # Groupes spécifiques du protocole LG
    'group_CONTROLS': 'Control', 'group_DEL': 'Del+', 'group_NODEL': 'Delirium -'
}

GROUP_COLORS = {
    'Control': '#2ca02c',
    'Del+': '#d62728',
    'COMA': '#ff7f0e', 
    'VS': '#8b0000',
    'MCS+': '#1f77b4'
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


# === FONCTIONS DE TRAITEMENT ===

def extract_subject_id_from_path(file_path: str) -> str:
    """Extraire l'ID du sujet à partir du chemin du fichier."""
    parts = file_path.split(os.sep)
    # Le sujet devrait être dans l'avant-dernière partie du chemin
    subject_folder = parts[-2]  # ex: "subject_AT19"
    
    # Extraire l'ID du sujet (enlever le préfixe "subject_")
    if subject_folder.startswith('subject_'):
        base_id = subject_folder.replace('subject_', '')
        
        # Pour les sujets avec préfixe Tp, essayer de reconstruire l'ID complet
        # en cherchant dans les noms de fichiers pour trouver le suffixe J1, J8, etc.
        file_name = os.path.basename(file_path)
        
        # Chercher un pattern TpXXXXJY dans le nom du fichier ou du dossier
        tp_pattern = re.search(r'Tp[A-Z0-9]+J\d+', file_name)
        if tp_pattern:
            return tp_pattern.group()
        
        # Sinon, essayer de reconstruire selon les patterns connus
        if base_id.startswith('AT19'):
            return f'Tp{base_id}'
        elif base_id.startswith('CF24'):
            return f'Tp{base_id}'
        elif base_id.startswith('EM13'):
            return f'Tp{base_id}'
        elif base_id.startswith('EP16'):
            return f'Tp{base_id}'
        elif base_id.startswith('AB15'):
            return f'Tp{base_id}'
        elif base_id.startswith('DC22'):
            return f'Tp{base_id}'
        elif base_id.startswith('FM25'):
            return f'Tp{base_id}'
        elif any(base_id.startswith(prefix) for prefix in ['AK24', 'AK27', 'CB15', 'DRL3', 'JB25', 'JLR17', 'MB45', 'MN42', 'PC21', 'PM14', 'RD38']):
            return f'Tp{base_id}'
        else:
            return base_id
    
    return subject_folder


def is_subject_in_subgroup(subject_id: str, subgroup_name: str) -> bool:
    """Vérifier si un sujet appartient au sous-groupe spécifié."""
    if subgroup_name not in SUBGROUP_IDS:
        return False
    return subject_id in SUBGROUP_IDS[subgroup_name]


def find_npz_files_for_subgroups(base_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Trouve et organise les fichiers NPZ par protocole et groupe, filtrant par sous-groupes spécifiés."""
    logger.info("Recherche des fichiers NPZ pour les sous-groupes spécifiés dans: %s", base_path)
    organized_data: Dict[str, Dict[str, List[str]]] = {}
    
    # Chercher les fichiers de résultats standard
    standard_pattern = os.path.join(base_path, '**', 'decoding_results_full.npz')
    standard_files = glob.glob(standard_pattern, recursive=True)
    
    # Chercher les fichiers LG spécifiques
    lg_pattern = os.path.join(base_path, '**', 'lg_decoding_results_full.npz')
    lg_files = glob.glob(lg_pattern, recursive=True)
    
    all_files = standard_files + lg_files

    if not all_files:
        logger.warning("Aucun fichier de résultats NPZ trouvé.")
        return {}
    
    logger.info("%d fichiers de résultats potentiels trouvés.", len(all_files))

    for file_path in all_files:
        try:
            parts = file_path.split(os.sep)
            # ex: .../{protocol}/{group}/{subject}/file.npz
            protocol_name = parts[-4]
            group_folder = parts[-3]
            subject_id = extract_subject_id_from_path(file_path)
            
            # Mapper le nom du groupe
            group_name = GROUP_NAME_MAPPING.get(group_folder, group_folder)
            
            # Vérifier si le sujet appartient à un de nos sous-groupes
            subject_belongs_to_subgroup = False
            final_group_name = None
            
            for subgroup_name in SUBGROUP_IDS.keys():
                if is_subject_in_subgroup(subject_id, subgroup_name):
                    subject_belongs_to_subgroup = True
                    final_group_name = subgroup_name
                    break
            
            if not subject_belongs_to_subgroup:
                continue  # Ignorer ce sujet s'il n'est pas dans nos sous-groupes
            
            if protocol_name not in organized_data:
                organized_data[protocol_name] = {}
            if final_group_name not in organized_data[protocol_name]:
                organized_data[protocol_name][final_group_name] = []
            organized_data[protocol_name][final_group_name].append(file_path)
            
            logger.debug(f"Sujet {subject_id} ajouté au groupe {final_group_name} (protocole {protocol_name})")
            
        except IndexError:
            logger.warning("Impossible de parser le chemin du fichier: %s", file_path)

    # Log du résumé
    for protocol, groups in organized_data.items():
        logger.info(f"Protocole {protocol}:")
        for group, files in groups.items():
            logger.info(f"  - Groupe {group}: {len(files)} sujets")

    return organized_data


def load_npz_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate NPZ file data, using the correct keys found in files.
    Now includes specific FDR and cluster masks and supports LG protocol.
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        keys = list(data.keys())
        
        # Identifier le type de fichier basé sur les clés disponibles
        is_lg_file = any('lg_' in key for key in keys)
        
        result = {
            'raw_data': data,
            'file_path': file_path,
            'keys': keys,
            'analysis_type': 'lg_temporal' if is_lg_file else 'pp_temporal'
        }
        
        # Extraire les données principales selon le type
        if is_lg_file:
            # Données LG
            if 'lg_temporal_decoding_scores_1d' in keys:
                result['scores'] = data['lg_temporal_decoding_scores_1d']
            elif 'lg_scores_1d' in keys:
                result['scores'] = data['lg_scores_1d']
            else:
                logger.warning(f"Aucun score LG trouvé dans {file_path}")
                return None
                
            if 'lg_epochs_time_points' in keys:
                result['times'] = data['lg_epochs_time_points']
            elif 'times' in keys:
                result['times'] = data['times']
        else:
            # Données PP standard
            if 'temporal_decoding_scores_1d' in keys:
                result['scores'] = data['temporal_decoding_scores_1d']
            elif 'scores_1d' in keys:
                result['scores'] = data['scores_1d']
            else:
                logger.warning(f"Aucun score PP trouvé dans {file_path}")
                return None
                
            if 'epochs_time_points' in keys:
                result['times'] = data['epochs_time_points']
            elif 'times' in keys:
                result['times'] = data['times']
        
        # Extraire les masques FDR et cluster si disponibles
        fdr_keys = [k for k in keys if 'fdr' in k.lower() and 'mask' in k.lower()]
        cluster_keys = [k for k in keys if 'cluster' in k.lower() and 'mask' in k.lower()]
        
        if fdr_keys:
            result['fdr_mask'] = data[fdr_keys[0]]
        if cluster_keys:
            result['cluster_mask'] = data[cluster_keys[0]]
        
        # Extraire les p-values si disponibles
        fdr_pval_keys = [k for k in keys if 'fdr' in k.lower() and ('pval' in k.lower() or 'p_val' in k.lower())]
        cluster_pval_keys = [k for k in keys if 'cluster' in k.lower() and ('pval' in k.lower() or 'p_val' in k.lower())]
        
        if fdr_pval_keys:
            result['fdr_pvalues'] = data[fdr_pval_keys[0]]
        if cluster_pval_keys:
            result['cluster_pvalues'] = data[cluster_pval_keys[0]]
        
        return result

    except Exception as e:
        logger.error(f"Erreur lors du chargement de {file_path}: {e}")
        return None


def analyze_subgroup_data(group_files: List[str], group_name: str) -> Dict[str, Any]:
    """
    Analyser les données d'un sous-groupe spécifique et extraire les statistiques.
    """
    logger.info(f"Analyse du sous-groupe {group_name} avec {len(group_files)} sujets")
    
    group_data = []
    subject_ids = []
    fdr_masks = []
    cluster_masks = []
    fdr_pvalues = []
    cluster_pvalues = []
    
    for file_path in group_files:
        data = load_npz_data(file_path)
        if data is not None:
            group_data.append(data)
            subject_ids.append(extract_subject_id_from_path(file_path))
            
            # Ajouter les masques et p-values si disponibles
            fdr_masks.append(data.get('fdr_mask', None))
            cluster_masks.append(data.get('cluster_mask', None))
            fdr_pvalues.append(data.get('fdr_pvalues', None))
            cluster_pvalues.append(data.get('cluster_pvalues', None))
    
    if not group_data:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return None
    
    # Extraire les scores et temps - Standardiser selon le protocole
    is_lg_protocol = any(d.get('analysis_type', '').startswith('lg_') for d in group_data)
    
    if is_lg_protocol:
        target_length = 101  # Longueur standard pour LG
    else:
        target_length = 126  # Longueur standard pour PP
    
    scores_list = []
    times = None
    
    # Adapter la longueur selon les données disponibles
    all_lengths = [len(d['scores']) for d in group_data]
    reference_length = target_length
    
    if target_length in all_lengths:
        reference_length = target_length
    else:
        reference_length = max(set(all_lengths), key=all_lengths.count)
        logger.info(f"Utilisation de la longueur {reference_length} au lieu de {target_length}")
    
    # Standardiser toutes les données à la longueur de référence
    standardized_fdr_masks = []
    standardized_cluster_masks = []
    standardized_fdr_pvalues = []
    standardized_cluster_pvalues = []
    
    for i, d in enumerate(group_data):
        scores = d['scores']
        current_length = len(scores)
        
        if current_length == reference_length:
            scores_list.append(scores)
            if times is None:
                times = d.get('times', np.linspace(-0.2, 0.8, reference_length))
        elif current_length > reference_length:
            # Tronquer
            scores_list.append(scores[:reference_length])
            if times is None:
                times = d.get('times', np.linspace(-0.2, 0.8, reference_length))[:reference_length]
        else:
            # Interpoler
            if times is None:
                times = np.linspace(-0.2, 0.8, reference_length)
            
            original_times = d.get('times', np.linspace(-0.2, 0.8, current_length))
            scores_interp = np.interp(times, original_times, scores)
            scores_list.append(scores_interp)
        
        # Standardiser les masques et p-values
        def standardize_array(arr, target_len):
            if arr is None:
                return np.zeros(target_len, dtype=bool)
            if len(arr) == target_len:
                return arr
            elif len(arr) > target_len:
                return arr[:target_len]
            else:
                # Interpoler pour les masques booléens et p-values
                original_indices = np.linspace(0, 1, len(arr))
                target_indices = np.linspace(0, 1, target_len)
                return np.interp(target_indices, original_indices, arr.astype(float)).astype(arr.dtype)
        
        standardized_fdr_masks.append(standardize_array(fdr_masks[i], reference_length))
        standardized_cluster_masks.append(standardize_array(cluster_masks[i], reference_length))
        standardized_fdr_pvalues.append(standardize_array(fdr_pvalues[i], reference_length))
        standardized_cluster_pvalues.append(standardize_array(cluster_pvalues[i], reference_length))
    
    # Remplacer les masques et p-values originaux par les versions standardisées
    fdr_masks = standardized_fdr_masks
    cluster_masks = standardized_cluster_masks
    fdr_pvalues = standardized_fdr_pvalues
    cluster_pvalues = standardized_cluster_pvalues
    
    scores_matrix = np.array(scores_list)
    
    # Vérifier et nettoyer les NaN dans la matrice des scores
    if np.any(np.isnan(scores_matrix)):
        logger.warning(f"NaN détectés dans les scores du groupe {group_name}, remplacement par la moyenne")
        scores_matrix = np.nan_to_num(scores_matrix, nan=np.nanmean(scores_matrix))
    
    # Calculer les statistiques du groupe avec gestion robuste des NaN
    group_mean = np.nanmean(scores_matrix, axis=0)
    group_std = np.nanstd(scores_matrix, axis=0)
    group_sem = group_std / np.sqrt(len(group_data))
    
    # S'assurer qu'il n'y a pas de NaN dans les résultats finaux
    if np.any(np.isnan(group_mean)):
        logger.warning(f"NaN dans la moyenne du groupe {group_name}")
        group_mean = np.nan_to_num(group_mean, nan=CHANCE_LEVEL)
    
    if np.any(np.isnan(group_sem)):
        logger.warning(f"NaN dans le SEM du groupe {group_name}")
        group_sem = np.nan_to_num(group_sem, nan=0.01)
    
    logger.info(f"Sous-groupe {group_name} statistics: mean range [{np.min(group_mean):.3f}, {np.max(group_mean):.3f}]")
    
    # Compter les sujets significatifs à chaque point temporel
    fdr_count = np.zeros(reference_length)
    cluster_count = np.zeros(reference_length)
    
    for mask in fdr_masks:
        if mask is not None:
            fdr_count += mask.astype(int)
    
    for mask in cluster_masks:
        if mask is not None:
            cluster_count += mask.astype(int)
    
    return {
        'group_name': group_name,
        'n_subjects': len(group_data),
        'subject_ids': subject_ids,
        'scores_matrix': scores_matrix,
        'group_mean': group_mean,
        'group_std': group_std,
        'group_sem': group_sem,
        'times': times,
        'fdr_masks': fdr_masks,
        'cluster_masks': cluster_masks,
        'fdr_pvalues': fdr_pvalues,
        'cluster_pvalues': cluster_pvalues,
        'fdr_count': fdr_count,
        'cluster_count': cluster_count,
        'subject_means': np.nanmean(scores_matrix, axis=1),
        'group_data': group_data
    }


def create_subgroup_comparison_plots(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                    output_dir: str) -> List[str]:
    """
    Créer des plots de comparaison pour un protocole avec les sous-groupes spécifiés.
    """
    protocol_output_dir = os.path.join(output_dir, f"subgroup_protocol_{protocol_name}")
    os.makedirs(protocol_output_dir, exist_ok=True)
    
    saved_plots = []
    
    if not groups_data:
        logger.warning(f"Aucune donnée de groupe pour le protocole {protocol_name}")
        return saved_plots
    
    # Créer les plots de comparaison
    try:
        # Plot des moyennes de groupe avec FDR
        means_plot = create_subgroup_means_with_fdr_plot(protocol_name, groups_data, protocol_output_dir)
        saved_plots.append(means_plot)
        
        # Plot combiné FDR et Cluster
        combined_plot = create_subgroup_combined_fdr_cluster_plot(protocol_name, groups_data, protocol_output_dir)
        saved_plots.append(combined_plot)
        
        logger.info(f"Plots créés pour le protocole {protocol_name}: {len(saved_plots)} fichiers")
        
    except Exception as e:
        logger.error(f"Erreur lors de la création des plots pour {protocol_name}: {e}")
    
    return saved_plots


def create_subgroup_means_with_fdr_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                       output_dir: str) -> str:
    """
    Créer un plot des moyennes de sous-groupe avec les comptages FDR.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # S'assurer que tous les groupes ont la même longueur de temps
    min_length = min(len(data['times']) for data in groups_data.values())
    times = None
    
    # Trouver le premier groupe avec la longueur minimale pour obtenir les temps
    for group_name, data in groups_data.items():
        if len(data['times']) == min_length:
            times = data['times'][:min_length]
            break
    
    if times is None:
        times = np.linspace(-0.2, 0.8, min_length)
    
    region_info = get_region_label_info()
    
    # Subplot 1: Moyennes des sous-groupes
    for group_name, data in groups_data.items():
        group_mean = data['group_mean'][:min_length]
        group_sem = data['group_sem'][:min_length]
        color = GROUP_COLORS.get(group_name, 'gray')
        n_subjects = data['n_subjects']
        
        ax1.plot(times, group_mean, color=color, linewidth=2.5, 
                label=f'{group_name} (n={n_subjects})')
        ax1.fill_between(times, group_mean - group_sem, group_mean + group_sem, 
                        color=color, alpha=0.2)
    
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.7, label='Chance Level')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    ax1.set_ylabel('Accuracy Score', fontsize=14)
    ax1.set_title(f'Sous-groupes Protocole {protocol_name} - Moyennes des Groupes', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Comptages FDR
    for group_name, data in groups_data.items():
        fdr_proportion = data['fdr_count'][:min_length] / data['n_subjects']
        color = GROUP_COLORS.get(group_name, 'gray')
        
        ax2.plot(times, fdr_proportion, color=color, linewidth=2.5, 
                label=f'{group_name}')
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_ylabel('Proportion de sujets significatifs (FDR)', fontsize=14)
    ax2.set_title('Proportions de Sujets Significatifs (FDR)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    
    filename = f"subgroup_{protocol_name}_means_with_fdr.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot des moyennes de sous-groupe avec FDR sauvé: {output_path}")
    return output_path


def create_subgroup_combined_fdr_cluster_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                             output_dir: str) -> str:
    """
    Créer un plot combiné avec FDR et Cluster pour les sous-groupes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # S'assurer que tous les groupes ont la même longueur de temps
    min_length = min(len(data['times']) for data in groups_data.values())
    times = None
    
    # Trouver le premier groupe avec la longueur minimale pour obtenir les temps
    for group_name, data in groups_data.items():
        if len(data['times']) == min_length:
            times = data['times'][:min_length]
            break
    
    if times is None:
        times = np.linspace(-0.2, 0.8, min_length)
    
    total_subjects = sum(data['n_subjects'] for data in groups_data.values())
    
    # === SUBPLOT 1: FDR Spécifique ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        n_subjects = data['n_subjects']
        color = GROUP_COLORS.get(group_name, 'gray')
        
        # Plot pour chaque sujet du groupe
        for i, (subject_id, fdr_mask) in enumerate(zip(data['subject_ids'], data['fdr_masks'])):
            y_pos = current_y_position + i
            
            if fdr_mask is not None:
                fdr_mask_clipped = fdr_mask[:min_length]
                
                # Tracer les points significatifs
                significant_times = times[fdr_mask_clipped.astype(bool)]
                significant_y = np.full_like(significant_times, y_pos)
                
                ax1.scatter(significant_times, significant_y, color=color, s=8, alpha=0.8)
        
        # Ajouter une ligne de séparation entre les groupes
        if current_y_position > 0:
            ax1.axhline(y=current_y_position - 0.5, color='lightgray', linestyle='-', alpha=0.5)
        
        # Etiquette de groupe
        group_y_center = current_y_position + (n_subjects - 1) / 2
        ax1.text(-0.25, group_y_center, f'{group_name}\n(n={n_subjects})', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        current_y_position += n_subjects
    
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.set_ylabel('Sujets', fontsize=14)
    ax1.set_title(f'Sous-groupes {protocol_name} - Significativité FDR par Sujet', fontsize=16, fontweight='bold')
    ax1.set_ylim([-1, total_subjects])
    ax1.grid(True, alpha=0.3)
    
    # === SUBPLOT 2: Cluster Spécifique ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        n_subjects = data['n_subjects']
        color = GROUP_COLORS.get(group_name, 'gray')
        
        # Plot pour chaque sujet du groupe
        for i, (subject_id, cluster_mask) in enumerate(zip(data['subject_ids'], data['cluster_masks'])):
            y_pos = current_y_position + i
            
            if cluster_mask is not None:
                cluster_mask_clipped = cluster_mask[:min_length]
                
                # Tracer les points significatifs
                significant_times = times[cluster_mask_clipped.astype(bool)]
                significant_y = np.full_like(significant_times, y_pos)
                
                ax2.scatter(significant_times, significant_y, color=color, s=8, alpha=0.8)
        
        # Ajouter une ligne de séparation entre les groupes
        if current_y_position > 0:
            ax2.axhline(y=current_y_position - 0.5, color='lightgray', linestyle='-', alpha=0.5)
        
        # Etiquette de groupe
        group_y_center = current_y_position + (n_subjects - 1) / 2
        ax2.text(-0.25, group_y_center, f'{group_name}\n(n={n_subjects})', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        current_y_position += n_subjects
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_ylabel('Sujets', fontsize=14)
    ax2.set_title(f'Sous-groupes {protocol_name} - Significativité Cluster par Sujet', fontsize=16, fontweight='bold')
    ax2.set_ylim([-1, total_subjects])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"subgroup_{protocol_name}_combined_fdr_cluster.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot combiné FDR/Cluster des sous-groupes sauvé: {output_path}")
    return output_path


def run_subgroup_temporal_analysis():
    """
    Fonction principale pour analyser les sous-groupes spécifiés.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/tom.balay/results/subgroup_temporal_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=== DÉBUT DE L'ANALYSE DES SOUS-GROUPES TEMPORELS ===")
    logger.info(f"Répertoire de sortie: {output_dir}")
    
    # Trouver les fichiers NPZ pour les sous-groupes
    organized_files = find_npz_files_for_subgroups(BASE_RESULTS_DIR)
    
    if not organized_files:
        logger.error("Aucun fichier trouvé pour les sous-groupes spécifiés")
        return
    
    all_saved_plots = []
    
    # Analyser chaque protocole
    for protocol_name, groups in organized_files.items():
        logger.info(f"\n=== ANALYSE DU PROTOCOLE {protocol_name} ===")
        
        # Analyser chaque groupe dans le protocole
        protocol_groups_data = {}
        
        for group_name, group_files in groups.items():
            logger.info(f"Analyse du sous-groupe {group_name}...")
            
            group_analysis = analyze_subgroup_data(group_files, group_name)
            if group_analysis is not None:
                protocol_groups_data[group_name] = group_analysis
                
                # Log des IDs des sujets inclus
                logger.info(f"Sujets inclus dans {group_name}: {group_analysis['subject_ids']}")
            else:
                logger.warning(f"Échec de l'analyse pour le groupe {group_name}")
        
        if protocol_groups_data:
            # Créer les plots de comparaison pour ce protocole
            protocol_plots = create_subgroup_comparison_plots(protocol_name, protocol_groups_data, output_dir)
            all_saved_plots.extend(protocol_plots)
        else:
            logger.warning(f"Aucune donnée analysée pour le protocole {protocol_name}")
    
    # Résumé final
    logger.info(f"\n=== RÉSUMÉ DE L'ANALYSE ===")
    logger.info(f"Sous-groupes analysés: {list(SUBGROUP_IDS.keys())}")
    logger.info(f"Protocoles analysés: {list(organized_files.keys())}")
    logger.info(f"Plots générés: {len(all_saved_plots)}")
    logger.info(f"Répertoire de sortie: {output_dir}")
    
    # Sauvegarder un résumé JSON
    summary = {
        'timestamp': timestamp,
        'output_directory': output_dir,
        'subgroups_analyzed': {name: ids for name, ids in SUBGROUP_IDS.items()},
        'protocols_found': list(organized_files.keys()),
        'plots_generated': all_saved_plots,
        'total_subjects_per_group': {}
    }
    
    # Ajouter le compte total de sujets par groupe
    for protocol_name, groups in organized_files.items():
        for group_name, group_files in groups.items():
            if group_name not in summary['total_subjects_per_group']:
                summary['total_subjects_per_group'][group_name] = 0
            summary['total_subjects_per_group'][group_name] += len(group_files)
    
    summary_path = os.path.join(output_dir, 'subgroup_analysis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Résumé sauvé dans: {summary_path}")
    logger.info("=== ANALYSE TERMINÉE ===")


# === POINT D'ENTRÉE PRINCIPAL ===

if __name__ == "__main__":
    run_subgroup_temporal_analysis()
