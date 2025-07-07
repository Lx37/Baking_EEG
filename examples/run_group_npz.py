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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

from scipy.stats import f_oneway
from statsmodels.stats.multitest import fdrcorrection

-
current_dir = os.path.dirname(os.path.abspath(__file__))
baking_eeg_dir = os.path.join(current_dir, '..')
if baking_eeg_dir not in sys.path:
    sys.path.insert(0, baking_eeg_dir)



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
    Retourne les informations sur les régions d'intérêt avec leurs positions temporelles.
    """
    return {
 
    }


try:
    from utils.stats_utils import (
        perform_pointwise_fdr_correction_on_scores,
        perform_cluster_permutation_test,
        compare_global_scores_to_chance
    )
except ImportError:
    print("ERREUR FATALE: Le module 'utils.stats_utils' n'a pas pu être importé.")
    print("Veuillez vous assurer que le script est exécuté depuis le dossier racine du projet.")
    print("Exemple de commande: `python examples/run_group_npz.py` (exécutée depuis le dossier 'Baking_EEG')")
    sys.exit(1)


try:
    from config.config import ALL_SUBJECT_GROUPS
except ImportError:
    print("AVERTISSEMENT: Impossible d'importer ALL_SUBJECT_GROUPS depuis config.config")



warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


-
BASE_RESULTS_DIR = "/home/tom.balay/results/BakingEEG_results_organized_by_protocol"

GROUP_NAME_MAPPING = {
    'group_COMA': 'Coma', 'group_CONTROLS_COMA': 'Controls (Coma)', 'group_MCS+': 'MCS+',
    'group_VS': 'VS/UWS', 'group_DELIRIUM+': 'Delirium +', 'group_DELIRIUM-': 'Delirium -',
    'group_CONTROLS_DELIRIUM': 'Controls (Delirium)', 'del': 'Delirium +', 'nodel': 'Delirium -',
    'control': 'Controls', 'controls': 'Controls', 'DELIRIUM+': 'Delirium +', 'DELIRIUM-': 'Delirium -',
    'deliriumpos': 'Delirium +', 'deliriumneg': 'Delirium -',
    # Groupes spécifiques du protocole LG
    'group_CONTROLS': 'Controls', 'group_DEL': 'Delirium +', 'group_NODEL': 'Delirium -'
}
GROUP_COLORS = {
    'Controls (Delirium)': '#2ca02c', 'Delirium -': '#ff7f0e', 'Delirium +': '#d62728',
    'Controls': '#2ca02c', 'Controls (Coma)': '#2ca02c', 'MCS+': '#1f77b4', 'group_MCS-': '#9467bd',
    'Coma': '#ff7f0e', 'VS/UWS': '#d62728',
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

def find_npz_files(base_path: str) -> Dict[str, Dict[str, List[str]]]:
    """Trouve et organise les fichiers NPZ par protocole et groupe."""
    logger.info("Recherche des fichiers NPZ dans: %s", base_path)
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
            group_name = GROUP_NAME_MAPPING.get(group_folder, group_folder)

            if protocol_name not in organized_data:
                organized_data[protocol_name] = {}
            if group_name not in organized_data[protocol_name]:
                organized_data[protocol_name][group_name] = []
            organized_data[protocol_name][group_name].append(file_path)
        except IndexError:
            logger.warning("Impossible de parser le chemin du fichier: %s", file_path)

    return organized_data


def load_npz_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and validate NPZ file data, using the correct keys found in files.
    Now includes specific FDR and cluster masks and supports LG protocol.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            data_keys = list(data.keys())
            
            # Détecter le protocole en fonction des clés disponibles
            is_lg_protocol = any(key.startswith('lg_') for key in data_keys)
            
            if is_lg_protocol:
                # Pour le protocole LG, il y a deux types de données principales
                # 1. lg_main_scores_1d_mean (LS vs LD)
                # 2. lg_mean_of_specific_scores_1d (GD_ALL vs GS_ALL)
                
                # Priorité à lg_main_scores_1d_mean (LS vs LD)
                if 'lg_main_scores_1d_mean' in data_keys:
                    actual_score_key = 'lg_main_scores_1d_mean'
                    actual_fdr_key = 'lg_main_temporal_1d_fdr'
                    actual_cluster_key = 'lg_main_temporal_1d_cluster'
                    analysis_type = 'lg_main'
                elif 'lg_mean_of_specific_scores_1d' in data_keys:
                    actual_score_key = 'lg_mean_of_specific_scores_1d'
                    actual_fdr_key = 'lg_mean_specific_fdr'
                    actual_cluster_key = 'lg_mean_specific_cluster'
                    analysis_type = 'lg_specific'
                else:
                    logger.warning("No suitable LG score key found in %s. Available keys: %s", file_path, data_keys)
                    return None
                
                actual_time_key = 'epochs_time_points'
                
            else:
                # Protocoles classiques (PP, PPext3, Battery)
                actual_score_key = 'pp_ap_main_scores_1d_mean'
                actual_time_key = 'epochs_time_points'
                # Clés réelles pour FDR et Cluster selon la documentation
                actual_fdr_key = 'pp_ap_main_temporal_1d_fdr'
                actual_cluster_key = 'pp_ap_main_temporal_1d_cluster'
                analysis_type = 'pp_main'
                # Clés TGM si disponibles
                tgm_fdr_key = 'pp_ap_main_tgm_fdr'

            required_keys = [actual_score_key, actual_time_key]

            for key in required_keys:
                if key not in data_keys:
                    logger.warning(
                        "Missing required field '%s' in %s. "
                        "Available keys: %s", key, file_path, data_keys)
                    return None

            # Extraire le nom du sujet à partir du chemin - Méthode améliorée
            path_parts = file_path.split(os.sep)
            subject_id = "Unknown"
            
            # Essayer plusieurs méthodes d'extraction
            for part in path_parts:
                # Méthode 1: _Subj_
                if '_Subj_' in part:
                    subject_id = part.split('_Subj_')[1].split('_')[0]
                    break
                # Méthode 2: Subject_
                elif 'Subject_' in part:
                    subject_id = part.split('Subject_')[1].split('_')[0]
                    break
                # Méthode 3: Subj_
                elif 'Subj_' in part and not part.startswith('_Subj_'):
                    subject_id = part.split('Subj_')[1].split('_')[0]
                    break
                # Méthode 4: Pattern avec chiffres
                elif part.startswith('sub') and any(c.isdigit() for c in part):
                    subject_id = part
                    break
            
            # Si toujours "Unknown", utiliser le nom du dossier parent
            if subject_id == "Unknown":
                for part in reversed(path_parts[:-1]):  # Exclure le fichier lui-même
                    if any(c.isdigit() for c in part):
                        subject_id = part
                        break
            
            # Nettoyer l'ID en enlevant les préfixes communs
            # Traiter tous les cas possibles de préfixes
            if subject_id.startswith('Subject_'):
                subject_id = subject_id.replace('Subject_', '')
            elif subject_id.startswith('Subj_'):
                subject_id = subject_id.replace('Subj_', '')
            elif subject_id.startswith('sub'):
                subject_id = subject_id.replace('sub', '', 1)  # Remplacer seulement la première occurrence
            
            # Nettoyer d'éventuels résidus
            if subject_id.startswith('ject_'):
                subject_id = subject_id.replace('ject_', '')
            if subject_id.startswith('_'):
                subject_id = subject_id[1:]
            
            # S'assurer qu'on a un ID valide
            if not subject_id or subject_id.lower() in ['unknown', 'id']:
                subject_id = os.path.basename(os.path.dirname(file_path))

            result = {
                'scores': data[actual_score_key],
                'times': data[actual_time_key],
                'subject_id': subject_id,
                'file_path': file_path,
                'analysis_type': analysis_type
            }

            # Ajouter FDR temporal si disponible - extraction du dictionnaire
            if actual_fdr_key in data_keys:
                fdr_data = data[actual_fdr_key]
                if isinstance(fdr_data, np.ndarray) and fdr_data.dtype == object:
                    try:
                        fdr_dict = fdr_data.item()
                        if isinstance(fdr_dict, dict):
                            result['fdr_mask'] = fdr_dict.get('mask', np.zeros_like(data[actual_score_key], dtype=bool))
                            result['fdr_pvalues'] = fdr_dict.get('p_values', np.ones_like(data[actual_score_key]))
                            result['fdr_pvalues_raw'] = fdr_dict.get('p_values_raw', np.ones_like(data[actual_score_key]))
                            result['fdr_method'] = fdr_dict.get('method', 'Unknown')
                            logger.debug("Loaded FDR data for %s: %d significant points", 
                                       subject_id, np.sum(result['fdr_mask']))
                        else:
                            logger.warning("FDR data is not a dict for %s, type: %s", subject_id, type(fdr_dict))
                            result['fdr_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                            result['fdr_pvalues'] = np.ones_like(data[actual_score_key])
                    except Exception as e:
                        logger.warning("Error extracting FDR data for %s: %s", subject_id, e)
                        result['fdr_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                        result['fdr_pvalues'] = np.ones_like(data[actual_score_key])
                else:
                    logger.warning("FDR data format not recognized for %s, type: %s", subject_id, type(fdr_data))
                    result['fdr_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                    result['fdr_pvalues'] = np.ones_like(data[actual_score_key])
            else:
                logger.warning("FDR key '%s' not found in %s. Available keys: %s", actual_fdr_key, subject_id, data_keys)
                result['fdr_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                result['fdr_pvalues'] = np.ones_like(data[actual_score_key])

            # Ajouter Cluster temporal si disponible
            if actual_cluster_key in data_keys:
                cluster_data = data[actual_cluster_key]
                if isinstance(cluster_data, np.ndarray) and cluster_data.dtype == object:
                    try:
                        cluster_dict = cluster_data.item()
                        if isinstance(cluster_dict, dict):
                            result['cluster_mask'] = cluster_dict.get('mask', np.zeros_like(data[actual_score_key], dtype=bool))
                            result['cluster_pvalues'] = cluster_dict.get('p_values_all_clusters', np.ones_like(data[actual_score_key]))
                            result['cluster_objects'] = cluster_dict.get('cluster_objects', [])
                            result['cluster_method'] = cluster_dict.get('method', 'Unknown')
                            logger.debug("Loaded Cluster data for %s: %d significant points", 
                                       subject_id, np.sum(result['cluster_mask']))
                        else:
                            logger.warning("Cluster data is not a dict for %s, type: %s", subject_id, type(cluster_dict))
                            result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                            result['cluster_pvalues'] = np.ones_like(data[actual_score_key])
                    except Exception as e:
                        logger.warning("Error extracting Cluster data for %s: %s", subject_id, e)
                        result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                        result['cluster_pvalues'] = np.ones_like(data[actual_score_key])
                else:
                    logger.warning("Cluster data format not recognized for %s, type: %s", subject_id, type(cluster_data))
                    result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                    result['cluster_pvalues'] = np.ones_like(data[actual_score_key])
            else:
                logger.warning("Cluster key '%s' not found in %s. Available keys: %s", actual_cluster_key, subject_id, data_keys)
                result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                result['cluster_pvalues'] = np.ones_like(data[actual_score_key])

            if result['scores'] is None or result['times'] is None:
                logger.warning("Data for scores or times is None in %s",
                               file_path)
                return None
            if len(result['scores']) == 0 or len(result['times']) == 0:
                logger.warning("Data for scores or times is empty in %s",
                               file_path)
                return None

            return result

    except Exception as e:
        logger.error("Error loading NPZ file %s: %s", file_path, e)
        return None


def analyze_group_data(group_files: List[str], group_name: str) -> Dict[str, Any]:
    """
    Analyser les données d'un groupe spécifique et extraire les statistiques.
    """
    logger.info(f"Analyse du groupe {group_name} avec {len(group_files)} sujets")
    
    group_data = []
    subject_ids = []
    fdr_masks = []
    cluster_masks = []
    fdr_pvalues = []
    cluster_pvalues = []
    
    for file_path in group_files:
        data = load_npz_data(file_path)
        if data is not None:
            # Vérifier s'il y a des NaN dans les scores
            if np.any(np.isnan(data['scores'])):
                logger.warning(f"NaN values found in scores for {data['subject_id']} in {group_name}")
                # Remplacer les NaN par la moyenne des valeurs non-NaN ou le niveau de chance
                scores_clean = data['scores'].copy()
                nan_mask = np.isnan(scores_clean)
                if np.all(nan_mask):
                    # Si tous les scores sont NaN, utiliser le niveau de chance
                    scores_clean[:] = CHANCE_LEVEL
                    logger.warning(f"All scores are NaN for {data['subject_id']}, using chance level")
                else:
                    # Remplacer les NaN par la moyenne des valeurs valides
                    valid_mean = np.nanmean(scores_clean)
                    scores_clean[nan_mask] = valid_mean
                    logger.info(f"Replaced {np.sum(nan_mask)} NaN values with mean {valid_mean:.3f} for {data['subject_id']}")
                data['scores'] = scores_clean
            
            group_data.append(data)
            subject_ids.append(data['subject_id'])
            fdr_masks.append(data.get('fdr_mask', np.array([])))
            cluster_masks.append(data.get('cluster_mask', np.array([])))
            fdr_pvalues.append(data.get('fdr_pvalues', np.array([])))
            cluster_pvalues.append(data.get('cluster_pvalues', np.array([])))
    
    if not group_data:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return {}
    
    # Extraire les scores et temps - Standardiser selon le protocole
    # Détecter si c'est le protocole LG basé sur le type d'analyse
    is_lg_protocol = any(d.get('analysis_type', '').startswith('lg_') for d in group_data)
    
    if is_lg_protocol:
        target_length = 801  # Protocole LG utilise 801 points temporels
        logger.info(f"Protocole LG détecté - utilisation de {target_length} points temporels")
    else:
        target_length = 601  # Protocoles classiques utilisent 601 points temporels
        logger.info(f"Protocole classique détecté - utilisation de {target_length} points temporels")
    
    scores_list = []
    times = None
    
    # Adapter la longueur selon les données disponibles
    all_lengths = [len(d['scores']) for d in group_data]
    reference_length = target_length
    
    if target_length in all_lengths:
        logger.info(f"Standardisation à {target_length} points temporels pour le groupe {group_name}")
    else:
        # Trouver la longueur minimale pour éviter les erreurs d'indexation
        min_length = min(all_lengths)
        max_length = max(all_lengths)
        
        if min_length < target_length:
            if max_length >= target_length:
                # Certains sujets ont assez de points, utiliser la longueur cible
                reference_length = target_length
                logger.info(f"Certains sujets ont {target_length} points, standardisation à {target_length} pour {group_name}")
            else:
                # Aucun sujet n'a assez de points
                reference_length = min_length
                logger.warning(f"Longueur cible {target_length} trop grande pour {group_name}, utilisation de {reference_length}")
        else:
            logger.info(f"Standardisation forcée à {target_length} points temporels pour le groupe {group_name}")
    
    # Standardiser toutes les données à la longueur de référence
    standardized_fdr_masks = []
    standardized_cluster_masks = []
    standardized_fdr_pvalues = []
    standardized_cluster_pvalues = []
    
    for i, d in enumerate(group_data):
        scores_truncated = d['scores'][:reference_length]
        scores_list.append(scores_truncated)
        
        if times is None:
            times = d['times'][:reference_length]
        
        # Standardiser les masques FDR
        if len(fdr_masks[i]) >= reference_length:
            standardized_fdr_masks.append(fdr_masks[i][:reference_length])
        else:
            padded_mask = np.zeros(reference_length, dtype=bool)
            if len(fdr_masks[i]) > 0:
                padded_mask[:len(fdr_masks[i])] = fdr_masks[i]
            standardized_fdr_masks.append(padded_mask)
        
        # Standardiser les masques cluster
        if len(cluster_masks[i]) >= reference_length:
            standardized_cluster_masks.append(cluster_masks[i][:reference_length])
        else:
            padded_mask = np.zeros(reference_length, dtype=bool)
            if len(cluster_masks[i]) > 0:
                padded_mask[:len(cluster_masks[i])] = cluster_masks[i]
            standardized_cluster_masks.append(padded_mask)
        
        # Standardiser les p-values FDR
        if len(fdr_pvalues[i]) >= reference_length:
            standardized_fdr_pvalues.append(fdr_pvalues[i][:reference_length])
        else:
            padded_pval = np.ones(reference_length)  # P-value par défaut = 1 (non significatif)
            if len(fdr_pvalues[i]) > 0:
                padded_pval[:len(fdr_pvalues[i])] = fdr_pvalues[i]
            standardized_fdr_pvalues.append(padded_pval)
        
        # Standardiser les p-values cluster
        if len(cluster_pvalues[i]) >= reference_length:
            standardized_cluster_pvalues.append(cluster_pvalues[i][:reference_length])
        else:
            padded_pval = np.ones(reference_length)
            if len(cluster_pvalues[i]) > 0:
                padded_pval[:len(cluster_pvalues[i])] = cluster_pvalues[i]
            standardized_cluster_pvalues.append(padded_pval)
    
    # Remplacer les masques et p-values originaux par les versions standardisées
    fdr_masks = standardized_fdr_masks
    cluster_masks = standardized_cluster_masks
    fdr_pvalues = standardized_fdr_pvalues
    cluster_pvalues = standardized_cluster_pvalues
    
    scores_matrix = np.array(scores_list)
    
    # Vérifier et nettoyer les NaN dans la matrice des scores
    if np.any(np.isnan(scores_matrix)):
        logger.warning(f"NaN values detected in scores matrix for group {group_name}")
        nan_count = np.sum(np.isnan(scores_matrix))
        logger.warning(f"Total NaN values: {nan_count} out of {scores_matrix.size}")
        
        # Remplacer les NaN par le niveau de chance ou interpoler
        scores_matrix = np.where(np.isnan(scores_matrix), CHANCE_LEVEL, scores_matrix)
        logger.info(f"Replaced NaN values with chance level {CHANCE_LEVEL} for group {group_name}")
    
    # Calculer les statistiques du groupe avec gestion robuste des NaN
    group_mean = np.nanmean(scores_matrix, axis=0)
    group_std = np.nanstd(scores_matrix, axis=0)
    group_sem = group_std / np.sqrt(len(group_data))
    
    # S'assurer qu'il n'y a pas de NaN dans les résultats finaux
    if np.any(np.isnan(group_mean)):
        logger.error(f"Group mean contains NaN for {group_name}, replacing with chance level")
        group_mean = np.where(np.isnan(group_mean), CHANCE_LEVEL, group_mean)
    
    if np.any(np.isnan(group_sem)):
        logger.error(f"Group SEM contains NaN for {group_name}, replacing with 0")
        group_sem = np.where(np.isnan(group_sem), 0.0, group_sem)
    
    logger.info(f"Group {group_name} statistics: mean range [{np.min(group_mean):.3f}, {np.max(group_mean):.3f}]")
    
    # Compter les sujets significatifs à chaque point temporel
    fdr_count = np.zeros(reference_length)
    cluster_count = np.zeros(reference_length)
    
    for mask in fdr_masks:
        fdr_count += mask.astype(int)
    
    for mask in cluster_masks:
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
        'group_data': group_data  # Ajouter les données individuelles pour l'analyse LG
    }


def create_protocol_comparison_plots(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                   output_dir: str) -> List[str]:
    """
    Créer des plots de comparaison pour un protocole donné.
    Gère spécialement le protocole LG avec ses deux types d'analyses.
    """
    protocol_output_dir = os.path.join(output_dir, f"protocol_{protocol_name}")
    os.makedirs(protocol_output_dir, exist_ok=True)
    
    saved_plots = []
    
    if not groups_data:
        logger.warning(f"Aucune donnée de groupe pour le protocole {protocol_name}")
        return saved_plots
    
    # Vérifier si c'est le protocole LG
    if protocol_name.lower() == 'lg' or 'lg' in protocol_name.lower():
        logger.info(f"Protocole LG détecté - utilisation de la fonction spécialisée")
        saved_plots = create_lg_protocol_plots(protocol_name, groups_data, protocol_output_dir)
    else:
        # Protocoles classiques (PP, PPext3, Battery)
        # 1. Plot des moyennes de groupe avec FDR général
        plot_path = create_group_means_with_fdr_plot(protocol_name, groups_data, protocol_output_dir)
        if plot_path:
            saved_plots.append(plot_path)
        
        # 2. Plot combiné de tous les sujets avec FDR et Cluster sur la même page
        plot_path = create_combined_subjects_fdr_cluster_plot(protocol_name, groups_data, protocol_output_dir)
        if plot_path:
            saved_plots.append(plot_path)
    
    return saved_plots


def create_group_means_with_fdr_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                   output_dir: str) -> str:
    """
    Créer un plot des moyennes de groupe avec les comptages FDR généraux.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # S'assurer que tous les groupes ont la même longueur de temps
    # Utiliser la longueur minimale commune à tous les groupes
    min_length = min(len(data['times']) for data in groups_data.values())
    times = None
    
    # Trouver le premier groupe avec la longueur minimale pour obtenir les temps
    for group_name, data in groups_data.items():
        if len(data['times']) == min_length:
            times = data['times'][:min_length]
            break
    
    if times is None:
        # Fallback: utiliser le premier groupe et tronquer
        times = list(groups_data.values())[0]['times'][:min_length]
    
    region_info = get_region_label_info()
    
    # Subplot 1: Moyennes des groupes
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        
        # Tronquer les données à la longueur commune
        group_mean = data['group_mean'][:min_length]
        group_sem = data['group_sem'][:min_length]
        
        logger.info(f"Plotting group {group_name}: mean shape {group_mean.shape}, times shape {times.shape}")
        logger.info(f"Group {group_name} mean range: {np.min(group_mean):.3f} to {np.max(group_mean):.3f}")
        
        ax1.plot(times, group_mean, label=f"{group_name} (n={data['n_subjects']})",
                color=color, linewidth=2.5)
        ax1.fill_between(times, 
                        group_mean - group_sem,
                        group_mean + group_sem,
                        alpha=0.2, color=color)
    
    ax1.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.7, label='Chance Level')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    ax1.set_ylabel('Accuracy Score', fontsize=14)
    ax1.set_title(f'Protocole {protocol_name} - Moyennes des Groupes', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les étiquettes de région
    add_region_labels_to_plot(ax1, times, region_info)
    
    # Subplot 2: Comptages FDR
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        
        # Tronquer les données à la longueur commune
        fdr_count = data['fdr_count'][:min_length]
        proportion = fdr_count / data['n_subjects']
        
        ax2.plot(times, proportion, label=f"{group_name} FDR",
                color=color, linewidth=2, marker='o', markersize=3)
    
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    ax2.set_xlabel('Temps (s)', fontsize=14)
    ax2.set_ylabel('Proportion de sujets significatifs (FDR)', fontsize=14)
    ax2.set_title('Proportions de Sujets Significatifs (FDR général)', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Ajouter les étiquettes de région
    add_region_labels_to_plot(ax2, times, region_info)
    
    plt.tight_layout()
    
    filename = f"{protocol_name}_group_means_with_fdr.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot des moyennes de groupe avec FDR sauvé: {output_path}")
    return output_path


def create_combined_subjects_fdr_cluster_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                            output_dir: str) -> str:
    """
    Créer un plot combiné avec FDR et Cluster sur la même page, l'un en dessous de l'autre.
    Inclut les p-values avec intensité selon la significativité et les étiquettes de région.
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
        # Fallback: utiliser le premier groupe et tronquer
        times = list(groups_data.values())[0]['times'][:min_length]
    
    total_subjects = sum(data['n_subjects'] for data in groups_data.values())
    region_info = get_region_label_info()
    
    # === SUBPLOT 1: FDR Spécifique ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        subject_ids = data['subject_ids']
        fdr_masks = data['fdr_masks']
        fdr_pvalues = data.get('fdr_pvalues', [])
        scores_matrix = data['scores_matrix']
        
        group_start_position = current_y_position
        
        # Pour chaque sujet dans le groupe
        for i, (subj_id, fdr_mask) in enumerate(zip(subject_ids, fdr_masks)):
            y_position = current_y_position
            
            # Tronquer le masque à la longueur commune
            fdr_mask_truncated = fdr_mask[:min_length] if len(fdr_mask) >= min_length else np.pad(fdr_mask, (0, min_length - len(fdr_mask)), mode='constant', constant_values=False)
            
            # Afficher les points significatifs FDR pour ce sujet
            if len(fdr_mask_truncated) > 0 and np.any(fdr_mask_truncated):
                sig_times = times[fdr_mask_truncated]
                sig_y = np.full_like(sig_times, y_position)
                
                # Calculer l'intensité basée sur les p-values FDR
                if i < len(fdr_pvalues) and len(fdr_pvalues[i]) > 0:
                    fdr_pvalues_truncated = fdr_pvalues[i][:min_length] if len(fdr_pvalues[i]) >= min_length else np.pad(fdr_pvalues[i], (0, min_length - len(fdr_pvalues[i])), mode='constant', constant_values=1.0)
                    sig_pvalues = fdr_pvalues_truncated[fdr_mask_truncated]
                    intensities = calculate_pvalue_intensity(sig_pvalues, FDR_ALPHA)
                else:
                    # Fallback : intensité basée sur les scores
                    if i < len(scores_matrix):
                        subject_scores = scores_matrix[i][:min_length]
                        sig_scores = subject_scores[fdr_mask_truncated]
                        intensities = calculate_score_intensity(sig_scores, CHANCE_LEVEL)
                    else:
                        intensities = np.full_like(sig_times, 0.8)
                
                # Créer des points avec intensité variable
                for j, (t, y, intensity) in enumerate(zip(sig_times, sig_y, intensities)):
                    ax1.scatter(t, y, color=color, marker='|', s=150,
                              alpha=intensity, linewidths=3)
            
            # Ligne de fond pour chaque sujet
            ax1.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y
            ax1.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(groups_data.keys())[-1]:
            ax1.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
        
        # Ajouter le nom du groupe sur le côté
        group_center_y = group_start_position + (data['n_subjects'] - 1) / 2
        ax1.text(-0.27, group_center_y, group_name, fontsize=12, fontweight='bold',
               verticalalignment='center', rotation=90, color=color)
    
    # Configuration du subplot 1
    ax1.set_ylabel('Sujets', fontsize=14)
    ax1.set_title(f'Protocole {protocol_name} - Points Significatifs FDR\n'
                 f'Tous les sujets (n={total_subjects}) - Intensité selon p-values', 
                 fontsize=16, fontweight='bold')
    ax1.set_yticks([])
    ax1.set_xlim([-0.2, 1.0])
    ax1.set_ylim([-0.5, total_subjects - 0.5])
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8)
    
    # Ajouter les étiquettes de région
    add_region_labels_to_plot(ax1, times, region_info)
    
    # === SUBPLOT 2: Cluster ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        subject_ids = data['subject_ids']
        cluster_masks = data['cluster_masks']
        cluster_pvalues = data.get('cluster_pvalues', [])
        scores_matrix = data['scores_matrix']
        
        group_start_position = current_y_position
        
        # Pour chaque sujet dans le groupe
        for i, (subj_id, cluster_mask) in enumerate(zip(subject_ids, cluster_masks)):
            y_position = current_y_position
            
            # Tronquer le masque à la longueur commune
            cluster_mask_truncated = cluster_mask[:min_length] if len(cluster_mask) >= min_length else np.pad(cluster_mask, (0, min_length - len(cluster_mask)), mode='constant', constant_values=False)
            
            # Afficher les points significatifs cluster pour ce sujet
            if len(cluster_mask_truncated) > 0 and np.any(cluster_mask_truncated):
                sig_times = times[cluster_mask_truncated]
                sig_y = np.full_like(sig_times, y_position)
                
                # Calculer l'intensité basée sur les p-values cluster
                if i < len(cluster_pvalues) and len(cluster_pvalues[i]) > 0:
                    cluster_pvalues_truncated = cluster_pvalues[i][:min_length] if len(cluster_pvalues[i]) >= min_length else np.pad(cluster_pvalues[i], (0, min_length - len(cluster_pvalues[i])), mode='constant', constant_values=1.0)
                    sig_pvalues = cluster_pvalues_truncated[cluster_mask_truncated]
                    intensities = calculate_pvalue_intensity(sig_pvalues, FDR_ALPHA)
                else:
                    # Fallback : intensité basée sur les scores
                    if i < len(scores_matrix):
                        subject_scores = scores_matrix[i][:min_length]
                        sig_scores = subject_scores[cluster_mask_truncated]
                        intensities = calculate_score_intensity(sig_scores, CHANCE_LEVEL)
                    else:
                        intensities = np.full_like(sig_times, 0.8)
                # Créer des points avec intensité variable (barres pour cluster)
                for j, (t, y, intensity) in enumerate(zip(sig_times, sig_y, intensities)):
                    ax2.scatter(t, y, color=color, marker='|', s=150,
                              alpha=intensity, linewidths=3)
            
            # Ligne de fond pour chaque sujet
            ax2.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y
            ax2.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(groups_data.keys())[-1]:
            ax2.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
        
        # Ajouter le nom du groupe sur le côté
        group_center_y = group_start_position + (data['n_subjects'] - 1) / 2
        ax2.text(-0.27, group_center_y, group_name, fontsize=12, fontweight='bold',
               verticalalignment='center', rotation=90, color=color)
    
    # Configuration du subplot 2
    ax2.set_xlabel('Temps (s)', fontsize=16)
    ax2.set_ylabel('Sujets', fontsize=14)
    ax2.set_title(f'Points Significatifs Cluster - Intensité selon p-values', 
                 fontsize=16, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_xlim([-0.2, 1.0])
    ax2.set_ylim([-0.5, total_subjects - 0.5])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=3, alpha=0.8)
    
    # Ajouter les étiquettes de région
    add_region_labels_to_plot(ax2, times, region_info)
    
    # Légende améliorée
    legend_elements = []
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        legend_elements.append(Line2D([0], [0], marker='o', color=color, linewidth=0, 
                                     markersize=10, label=f'{group_name} (n={data["n_subjects"]})'))
    
    legend_elements.append(Line2D([0], [0], marker='|', color='gray', linewidth=0, 
                                 markersize=15, label='FDR Significatif'))
    legend_elements.append(Line2D([0], [0], marker='|', color='gray', linewidth=0, 
                                 markersize=10, label='Cluster Significatif'))
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                 linewidth=3, label='Stimulus Onset'))
    
    # Ajouter légende pour l'intensité
    legend_elements.append(mpatches.Patch(color='gray', alpha=0.3, label='p-value élevée'))
    legend_elements.append(mpatches.Patch(color='gray', alpha=1.0, label='p-value faible'))
    
    ax1.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    
    filename = f"{protocol_name}_combined_subjects_fdr_cluster_pvalues.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot combiné FDR/Cluster avec p-values de tous les sujets sauvé: {output_path}")
    return output_path


def create_lg_protocol_plots(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                           output_dir: str) -> List[str]:
    """
    Créer des plots spécialisés pour le protocole LG avec FDR/Cluster sur une page
    et moyennes sur une autre page.
    """
    logger.info("=== DÉBUT CRÉATION PLOTS LG ===")
    logger.info(f"Protocole: {protocol_name}")
    logger.info(f"Dossier de sortie: {output_dir}")
    logger.info(f"Groupes reçus: {list(groups_data.keys())}")
    
    protocol_output_dir = os.path.join(output_dir, f"protocol_{protocol_name}")
    os.makedirs(protocol_output_dir, exist_ok=True)
    logger.info(f"Dossier protocole créé: {protocol_output_dir}")
    
    saved_plots = []
    
    if not groups_data:
        logger.warning(f"❌ Aucune donnée de groupe pour le protocole {protocol_name}")
        return saved_plots
    
    # Détecter les types d'analyse disponibles dans les données
    logger.info("Appel de detect_lg_analysis_types...")
    analysis_types = detect_lg_analysis_types(groups_data)
    logger.info(f"✓ Types d'analyse LG détectés: {list(analysis_types.keys())}")
    
    if not analysis_types:
        logger.error(f"❌ Aucun type d'analyse détecté pour le protocole LG")
        return saved_plots
    
    # Traiter chaque type d'analyse
    for analysis_type, analysis_data in analysis_types.items():
        logger.info(f"=== TRAITEMENT DE L'ANALYSE: {analysis_type} ===")
        logger.info(f"Groupes dans cette analyse: {list(analysis_data.keys())}")
        
        # Vérifier que l'analyse contient des données
        total_subjects = sum(data.get('n_subjects', 0) for data in analysis_data.values())
        logger.info(f"Total de sujets pour {analysis_type}: {total_subjects}")
        
        if total_subjects == 0:
            logger.warning(f"⚠️ Aucun sujet trouvé pour l'analyse {analysis_type}, passage au suivant")
            continue
        
        logger.info(f"Création des plots pour l'analyse LG: {analysis_type}")
        
        # Déterminer le titre d'analyse
        if analysis_type == 'lg_main':
            analysis_title = "LS vs LD (Local Standard vs Local Deviant)"
        elif analysis_type == 'lg_specific':
            analysis_title = "Moyenne des comparaisons spécifiques LG (lg_mean_of_specific_scores_1d)"
        else:
            analysis_title = analysis_type
        
        logger.info(f"Titre de l'analyse: {analysis_title}")
        
        # 1. Plot des moyennes de groupe (page séparée)
        logger.info("1. Création du plot des moyennes de groupe...")
        try:
            plot_path = create_lg_group_means_plot(protocol_name, analysis_data, protocol_output_dir, analysis_type, analysis_title)
            if plot_path:
                saved_plots.append(plot_path)
                logger.info(f"✓ Plot moyennes créé: {plot_path}")
            else:
                logger.warning(f"⚠️ Échec création plot moyennes pour {analysis_type}")
        except Exception as e:
            logger.error(f"❌ Erreur création plot moyennes pour {analysis_type}: {e}")
        
        # 2. Plot combiné FDR/Cluster pour tous les sujets (page séparée)
        logger.info("2. Création du plot combiné FDR/Cluster...")
        try:
            plot_path = create_lg_combined_fdr_cluster_plot(protocol_name, analysis_data, protocol_output_dir, analysis_type, analysis_title)
            if plot_path:
                saved_plots.append(plot_path)
                logger.info(f"✓ Plot FDR/Cluster créé: {plot_path}")
            else:
                logger.warning(f"⚠️ Échec création plot FDR/Cluster pour {analysis_type}")
        except Exception as e:
            logger.error(f"❌ Erreur création plot FDR/Cluster pour {analysis_type}: {e}")
    
    logger.info(f"=== FIN CRÉATION PLOTS LG ===")
    logger.info(f"Total de plots créés: {len(saved_plots)}")
    for plot in saved_plots:
        logger.info(f"  - {os.path.basename(plot)}")
    
    return saved_plots


def create_lg_group_means_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                             output_dir: str, analysis_type: str, analysis_title: str) -> str:
    """
    Créer un plot des moyennes de groupe pour le protocole LG.
    """
    logger.info(f"=== CRÉATION PLOT MOYENNES LG ({analysis_type}) ===")
    logger.info(f"Groupes reçus: {list(groups_data.keys())}")
    
    if not groups_data:
        logger.error("❌ Pas de données de groupes reçues")
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # S'assurer que tous les groupes ont la même longueur de temps (801 pour LG)
    min_length = min(len(data['times']) for data in groups_data.values())
    logger.info(f"Longueur minimale des temps: {min_length}")
    
    times = None
    
    # Trouver le premier groupe avec la longueur minimale pour obtenir les temps
    for group_name, data in groups_data.items():
        if len(data['times']) == min_length:
            times = data['times'][:min_length]
            logger.info(f"Utilisation des temps du groupe {group_name}")
            break
    
    if times is None:
        # Fallback: utiliser le premier groupe et tronquer
        times = list(groups_data.values())[0]['times'][:min_length]
        logger.warning(f"Utilisation des temps du premier groupe (fallback)")
    
    logger.info(f"Times shape: {times.shape}, range: [{times[0]:.3f}, {times[-1]:.3f}]")
    
    # Plot des moyennes des groupes
    for group_name, data in groups_data.items():
        logger.info(f"Plot du groupe {group_name}:")
        logger.info(f"  - n_subjects: {data.get('n_subjects', 0)}")
        logger.info(f"  - group_mean shape: {data['group_mean'].shape}")
        logger.info(f"  - group_sem shape: {data['group_sem'].shape}")
        
        color = GROUP_COLORS.get(group_name, '#000000')
        
        # Tronquer les données à la longueur commune
        group_mean = data['group_mean'][:min_length]
        group_sem = data['group_sem'][:min_length]
        
        logger.info(f"  - mean range: [{np.min(group_mean):.3f}, {np.max(group_mean):.3f}]")
        logger.info(f"  - sem range: [{np.min(group_sem):.3f}, {np.max(group_sem):.3f}]")
        
        logger.info(f"Plotting LG group {group_name} ({analysis_type}): mean shape {group_mean.shape}, times shape {times.shape}")
        
        ax.plot(times, group_mean, label=f"{group_name} (n={data['n_subjects']})",
                color=color, linewidth=2.5)
        ax.fill_between(times, 
                        group_mean - group_sem,
                        group_mean + group_sem,
                        alpha=0.2, color=color)
        
        logger.info(f"✓ Groupe {group_name} plotté")
    
    ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.7, label='Chance Level')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Decoding Accuracy', fontsize=14)
    ax.set_title(f'Protocol LG - Group Means\n{analysis_title}', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{protocol_name}_{analysis_type}_group_means.png"
    output_path = os.path.join(output_dir, filename)
    logger.info(f"Sauvegarde vers: {output_path}")
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Plot des moyennes de groupe LG ({analysis_type}) sauvé: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
        plt.close()
        return None


def create_lg_combined_fdr_cluster_plot(protocol_name: str, groups_data: Dict[str, Dict[str, Any]], 
                                      output_dir: str, analysis_type: str, analysis_title: str) -> str:
    """
    Créer un plot combiné avec FDR et Cluster sur la même page pour le protocole LG.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # S'assurer que tous les groupes ont la même longueur de temps (801 pour LG)
    min_length = min(len(data['times']) for data in groups_data.values())
    times = None
    
    # Trouver le premier groupe avec la longueur minimale pour obtenir les temps
    for group_name, data in groups_data.items():
        if len(data['times']) == min_length:
            times = data['times'][:min_length]
            break
    
    if times is None:
        # Fallback: utiliser le premier groupe et tronquer
        times = list(groups_data.values())[0]['times'][:min_length]
    
    total_subjects = sum(data['n_subjects'] for data in groups_data.values())
    
    # === SUBPLOT 1: FDR ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        subject_ids = data['subject_ids']
        fdr_masks = data['fdr_masks']
        fdr_pvalues = data.get('fdr_pvalues', [])
        scores_matrix = data['scores_matrix']
        
        group_start_position = current_y_position
        
        # Pour chaque sujet dans le groupe
        for i, (subj_id, fdr_mask) in enumerate(zip(subject_ids, fdr_masks)):
            y_position = current_y_position
            
            # Tronquer le masque à la longueur commune
            fdr_mask_truncated = fdr_mask[:min_length] if len(fdr_mask) >= min_length else np.pad(fdr_mask, (0, min_length - len(fdr_mask)), mode='constant', constant_values=False)
            
            # Afficher les points significatifs FDR pour ce sujet
            if len(fdr_mask_truncated) > 0 and np.any(fdr_mask_truncated):
                sig_times = times[fdr_mask_truncated]
                sig_y = np.full_like(sig_times, y_position)
                
                # Calculer l'intensité basée sur les p-values FDR
                if i < len(fdr_pvalues) and len(fdr_pvalues[i]) > 0:
                    fdr_pvalues_truncated = fdr_pvalues[i][:min_length] if len(fdr_pvalues[i]) >= min_length else np.pad(fdr_pvalues[i], (0, min_length - len(fdr_pvalues[i])), mode='constant', constant_values=1.0)
                    sig_pvalues = fdr_pvalues_truncated[fdr_mask_truncated]
                    intensities = calculate_pvalue_intensity(sig_pvalues, FDR_ALPHA)
                else:
                    # Fallback : intensité basée sur les scores
                    if i < len(scores_matrix):
                        subject_scores = scores_matrix[i][:min_length]
                        sig_scores = subject_scores[fdr_mask_truncated]
                        intensities = calculate_score_intensity(sig_scores, CHANCE_LEVEL)
                    else:
                        intensities = np.full_like(sig_times, 0.8)
                
                # Créer des points avec intensité variable
                for j, (t, y, intensity) in enumerate(zip(sig_times, sig_y, intensities)):
                    ax1.scatter(t, y, color=color, marker='|', s=150,
                              alpha=intensity, linewidths=3)
            
            # Ligne de fond pour chaque sujet
            ax1.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y (juste l'ID)
            ax1.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(groups_data.keys())[-1]:
            ax1.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
    
    # Configuration du subplot FDR
    ax1.set_title(f'Protocol LG - FDR Significance\n{analysis_title}\n'
                 f'All subjects (n={total_subjects}) - Intensity by p-values', 
                 fontsize=16, fontweight='bold')
    ax1.set_ylabel('Subjects', fontsize=14)
    ax1.set_ylim(-0.5, total_subjects - 0.5)
    ax1.set_yticks([])
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.grid(True, alpha=0.3)
    
    # === SUBPLOT 2: Cluster Spécifique ===
    current_y_position = 0
    
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        subject_ids = data['subject_ids']
        cluster_masks = data['cluster_masks']
        cluster_pvalues = data.get('cluster_pvalues', [])
        scores_matrix = data['scores_matrix']
        
        # Pour chaque sujet dans le groupe
        for i, (subj_id, cluster_mask) in enumerate(zip(subject_ids, cluster_masks)):
            y_position = current_y_position
            
            # Tronquer le masque à la longueur commune
            cluster_mask_truncated = cluster_mask[:min_length] if len(cluster_mask) >= min_length else np.pad(cluster_mask, (0, min_length - len(cluster_mask)), mode='constant', constant_values=False)
            
            # Afficher les points significatifs cluster pour ce sujet
            if len(cluster_mask_truncated) > 0 and np.any(cluster_mask_truncated):
                sig_times = times[cluster_mask_truncated]
                sig_y = np.full_like(sig_times, y_position)
                
                # Calculer l'intensité basée sur les p-values cluster
                if i < len(cluster_pvalues) and len(cluster_pvalues[i]) > 0:
                    cluster_pvalues_truncated = cluster_pvalues[i][:min_length] if len(cluster_pvalues[i]) >= min_length else np.pad(cluster_pvalues[i], (0, min_length - len(cluster_pvalues[i])), mode='constant', constant_values=1.0)
                    sig_pvalues = cluster_pvalues_truncated[cluster_mask_truncated]
                    intensities = calculate_pvalue_intensity(sig_pvalues, FDR_ALPHA)
                else:
                    # Fallback : intensité basée sur les scores
                    if i < len(scores_matrix):
                        subject_scores = scores_matrix[i][:min_length]
                        sig_scores = subject_scores[cluster_mask_truncated]
                        intensities = calculate_score_intensity(sig_scores, CHANCE_LEVEL)
                    else:
                        intensities = np.full_like(sig_times, 0.8)
                # Créer des points avec intensité variable (barres pour cluster)
                for j, (t, y, intensity) in enumerate(zip(sig_times, sig_y, intensities)):
                    ax2.scatter(t, y, color=color, marker='|', s=150,
                              alpha=intensity, linewidths=3)
            
            # Ligne de fond pour chaque sujet
            ax2.plot([times[0], times[-1]], [y_position, y_position], 
                   color='lightgray', alpha=0.3, linewidth=1)
            
            # Label du sujet sur l'axe Y (juste l'ID)
            ax2.text(-0.24, y_position, f'{subj_id}', fontsize=8, 
                   verticalalignment='center', color=color, fontweight='bold')
            
            current_y_position += 1
        
        # Ajouter une séparation entre les groupes
        if group_name != list(groups_data.keys())[-1]:
            ax2.axhline(y=current_y_position - 0.5, color='black', linestyle='-', 
                      linewidth=2, alpha=0.5)
    
    # Configuration du subplot Cluster
    ax2.set_title(f'Cluster Significance - Intensity by p-values', 
                 fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Subjects', fontsize=14)
    ax2.set_ylim(-0.5, total_subjects - 0.5)
    ax2.set_yticks([])
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    # Légende améliorée
    legend_elements = []
    for group_name, data in groups_data.items():
        color = GROUP_COLORS.get(group_name, '#000000')
        legend_elements.append(Line2D([0], [0], marker='o', color=color, linewidth=0, 
                                     markersize=10, label=f'{group_name} (n={data["n_subjects"]})'))
    
    legend_elements.append(Line2D([0], [0], marker='|', color='gray', linewidth=0, 
                                 markersize=15, label='FDR Significant'))
    legend_elements.append(Line2D([0], [0], marker='s', color='gray', linewidth=0, 
                                 markersize=10, label='Cluster Significant'))
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                 linewidth=3, label='Stimulus Onset'))
    
    # Ajouter légende pour l'intensité
    import matplotlib.patches as mpatches
    legend_elements.append(mpatches.Patch(color='gray', alpha=0.3, label='High p-value'))
    legend_elements.append(mpatches.Patch(color='gray', alpha=1.0, label='Low p-value'))
    
    ax1.legend(handles=legend_elements, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    
    filename = f"{protocol_name}_{analysis_type}_combined_fdr_cluster.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plot combiné FDR/Cluster LG ({analysis_type}) sauvé: {output_path}")
    return output_path


def extract_lg_specific_data_from_npz(file_path):
    """Extraire les données LG spécifiques depuis le fichier NPZ principal."""
    try:
        data = np.load(file_path, allow_pickle=True)
        
        # Vérifier les clés spécifiques LG selon la structure réelle
        required_keys = ['lg_mean_of_specific_scores_1d', 'epochs_time_points']
        if not all(key in data.files for key in required_keys):
            logger.warning(f"Clés LG spécifiques manquantes dans {file_path}")
            logger.info(f"Clés disponibles: {list(data.files)}")
            return None
        
        # Extraire les données principales
        result = {
            'specific_scores': data['lg_mean_of_specific_scores_1d'],
            'times': data['epochs_time_points'],
            'file_path': file_path
        }
        
        # Ajouter des données optionnelles si disponibles
        if 'lg_sem_of_specific_scores_1d' in data.files:
            result['specific_sem'] = data['lg_sem_of_specific_scores_1d']
        
        if 'lg_specific_comparison_results' in data.files:
            result['comparison_results'] = data['lg_specific_comparison_results']
            logger.info(f"  ✓ lg_specific_comparison_results trouvé: shape {data['lg_specific_comparison_results'].shape}")
        
        # Ajouter les données FDR et cluster spécifiques si disponibles
        if 'lg_mean_specific_fdr' in data.files:
            result['specific_fdr'] = data['lg_mean_specific_fdr']
            logger.info(f"  ✓ lg_mean_specific_fdr trouvé")
        
        if 'lg_mean_specific_cluster' in data.files:
            result['specific_cluster'] = data['lg_mean_specific_cluster']
            logger.info(f"  ✓ lg_mean_specific_cluster trouvé")
        
        # Journaliser les informations sur les données extraites
        logger.info(f"  ✓ Données spécifiques extraites:")
        logger.info(f"    - specific_scores shape: {result['specific_scores'].shape}")
        logger.info(f"    - times shape: {result['times'].shape}")
        logger.info(f"    - times range: [{result['times'][0]:.3f}, {result['times'][-1]:.3f}]")
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données LG spécifiques de {file_path}: {e}")
        return None


def collect_lg_specific_data_from_groups(groups_data):
    """Collecter les données LG spécifiques depuis les données de groupe existantes."""
    logger.info("Collecte des données LG spécifiques depuis les groupes existants...")
    
    lg_specific_data = {}
    
    for group_name, group_data in groups_data.items():
        logger.info(f"Traitement du groupe {group_name} pour les données spécifiques LG")
        
        group_subjects_specific = []
        
        # Pour chaque fichier de sujet dans le groupe
        for subject_file_data in group_data.get('group_data', []):
            if 'file_path' in subject_file_data:
                file_path = subject_file_data['file_path']
                subject_id = subject_file_data.get('subject_id', extract_subject_id(file_path))
                
                # Extraire les données spécifiques LG
                specific_data = extract_lg_specific_data_from_npz(file_path)
                
                if specific_data:
                    group_subjects_specific.append({
                        'subject_id': subject_id,
                        'specific_scores': specific_data['specific_scores'],
                        'specific_sem': specific_data.get('specific_sem'),
                        'times': specific_data['times'],
                        'file_path': file_path
                    })
                    logger.info(f"  ✓ Données spécifiques extraites pour {subject_id}")
                else:
                    logger.warning(f"  ❌ Impossible d'extraire les données spécifiques pour {subject_id}")
        
        if group_subjects_specific:
            lg_specific_data[group_name] = group_subjects_specific
            logger.info(f"Groupe {group_name}: {len(group_subjects_specific)} sujets avec données spécifiques")
    
    return lg_specific_data


def compute_lg_specific_group_data(lg_specific_data):
    """Calculer les moyennes de groupe pour les données LG spécifiques."""
    logger.info("Calcul des moyennes de groupe pour les données LG spécifiques...")
    
    group_averages = {}
    
    for group_name, subjects_list in lg_specific_data.items():
        if not subjects_list:
            continue
            
        logger.info(f"Traitement du groupe {group_name} avec {len(subjects_list)} sujets")
        
        # Collecter tous les scores spécifiques
        all_specific_scores = []
        reference_times = None
        subject_ids = []
        
        for subject_info in subjects_list:
            specific_scores = subject_info['specific_scores']
            times = subject_info['times']
            subject_id = subject_info['subject_id']
            
            if reference_times is None:
                reference_times = times
            elif len(times) == len(reference_times):
                all_specific_scores.append(specific_scores)
                subject_ids.append(subject_id)
            else:
                logger.warning(f"Longueur de temps incompatible pour {subject_id}: {len(times)} vs {len(reference_times)}")
        
        if all_specific_scores and reference_times is not None:
            scores_array = np.array(all_specific_scores)
            
            group_averages[group_name] = {
                'group_mean': np.mean(scores_array, axis=0),
                'group_std': np.std(scores_array, axis=0),
                'group_sem': np.std(scores_array, axis=0) / np.sqrt(len(all_specific_scores)),
                'times': reference_times,
                'n_subjects': len(all_specific_scores),
                'subject_ids': subject_ids,
                'scores_matrix': scores_array,
                # Ajouter des masques fictifs pour la compatibilité
                'fdr_masks': [np.zeros(len(reference_times), dtype=bool) for _ in range(len(all_specific_scores))],
                'cluster_masks': [np.zeros(len(reference_times), dtype=bool) for _ in range(len(all_specific_scores))],
                'subject_means': np.mean(scores_array, axis=1),
                'fdr_pvalues': [np.ones(len(reference_times)) for _ in range(len(all_specific_scores))],
                'cluster_pvalues': [np.ones(len(reference_times)) for _ in range(len(all_specific_scores))],
                'group_data': []
            }
            
            logger.info(f"✓ Groupe {group_name}: moyenne calculée pour {len(all_specific_scores)} sujets")
        else:
            logger.warning(f"❌ Impossible de calculer la moyenne pour le groupe {group_name}")
    
    return group_averages


def detect_lg_analysis_types(groups_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Détecter les types d'analyse disponibles dans les données LG.
    Cette version utilise les vraies données spécifiques lg_mean_of_specific_scores_1d depuis les fichiers NPZ.
    """
    logger.info("=== DÉBUT DÉTECTION DES TYPES D'ANALYSE LG SPÉCIFIQUES ===")
    
    analysis_types = {}
    
    # 1. Analyse principale : utiliser les données déjà disponibles (LS vs LD - lg_main)
    logger.info("=== CRÉATION DE L'ANALYSE PRINCIPALE LG (LS vs LD) ===")
    analysis_types['lg_main'] = groups_data.copy()
    
    # 2. Analyse spécifique : utiliser lg_mean_of_specific_scores_1d depuis les fichiers NPZ
    logger.info("=== CRÉATION DE L'ANALYSE SPÉCIFIQUE LG (Moyenne des comparaisons spécifiques) ===")
    
    # Collecter les données spécifiques LG depuis les groupes existants
    lg_specific_data = collect_lg_specific_data_from_groups(groups_data)
    
    if lg_specific_data:
        # Calculer les moyennes de groupe pour les données spécifiques
        lg_group_averages = compute_lg_specific_group_data(lg_specific_data)
        
        if lg_group_averages:
            analysis_types['lg_specific'] = lg_group_averages
            logger.info(f"✓ Analyse spécifique LG créée avec {len(lg_group_averages)} groupes")
            for group_name, data in lg_group_averages.items():
                logger.info(f"  - {group_name}: {data['n_subjects']} sujets")
        else:
            logger.warning("❌ Aucune moyenne de groupe calculée pour l'analyse spécifique LG")
    else:
        logger.warning("❌ Aucune donnée spécifique LG trouvée dans les groupes")
    
    logger.info("=== FIN DÉTECTION ===")
    logger.info(f"Types d'analyse LG créés: {list(analysis_types.keys())}")
    
    return analysis_types


def run_comprehensive_protocol_analysis(base_results_dir: str = None) -> None:
    """
    Fonction principale pour analyser tous les protocoles et créer les visualisations.
    """
    if base_results_dir is None:
        base_results_dir = BASE_RESULTS_DIR
    
    logger.info("=== DÉBUT DE L'ANALYSE COMPLÈTE PAR PROTOCOLE ===")
    
    # Créer le dossier de sortie principal
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"enhanced_analysis_results_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Trouver tous les fichiers NPZ organisés par protocole
    organized_data = find_npz_files(base_results_dir)
    
    if not organized_data:
        logger.error("Aucune donnée trouvée. Vérifiez le chemin de base.")
        return
    
    logger.info(f"Protocoles trouvés: {list(organized_data.keys())}")
    
    all_results = {}
    
    # Analyser chaque protocole
    for protocol_name, groups in organized_data.items():
        logger.info(f"\n=== ANALYSE DU PROTOCOLE: {protocol_name} ===")
        logger.info(f"Groupes trouvés: {list(groups.keys())}")
        
        # Analyser chaque groupe du protocole
        protocol_groups_data = {}
        for group_name, file_paths in groups.items():
            group_data = analyze_group_data(file_paths, group_name)
            if group_data:
                protocol_groups_data[group_name] = group_data
        
        if not protocol_groups_data:
            logger.warning(f"Aucune donnée valide pour le protocole {protocol_name}")
            continue
        
        # Créer les visualisations pour ce protocole
        logger.info(f"Création des visualisations pour le protocole {protocol_name}")
        saved_plots = create_protocol_comparison_plots(protocol_name, protocol_groups_data, output_base_dir)
        
        all_results[protocol_name] = {
            'groups_data': protocol_groups_data,
            'saved_plots': saved_plots
        }
    
    # Créer un résumé global
    create_global_summary(all_results, output_base_dir)
    
    logger.info(f"\n=== ANALYSE TERMINÉE ===")
    logger.info(f"Résultats sauvés dans: {output_base_dir}")
    logger.info(f"Total protocoles analysés: {len(all_results)}")


def create_global_summary(all_results: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """
    Créer un résumé global de tous les protocoles analysés.
    """
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("=== RÉSUMÉ DE L'ANALYSE COMPLÈTE PAR PROTOCOLE ===\n\n")
        f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== AMÉLIORATIONS APPORTÉES ===\n")
        f.write("- Intensité de couleur basée sur les scores (éloignement du niveau de chance)\n")
        f.write("- Étiquettes de région explicitées (PP/AP main, Early Response, Late Response)\n")
        f.write("- Standardisation à 601 points temporels avec gestion des longueurs variables\n")
        f.write("- Extraction améliorée des IDs de sujet\n")
        f.write("- Plots combinés FDR/Cluster sur la même page\n")
        f.write("- Légendes améliorées avec indication d'intensité\n")
        f.write("- Analyse spécifique LG basée sur lg_mean_of_specific_scores_1d\n\n")
        
        for protocol_name, results in all_results.items():
            f.write(f"PROTOCOLE: {protocol_name}\n")
            f.write("-" * 50 + "\n")
            
            groups_data = results['groups_data']
            total_subjects = sum(data['n_subjects'] for data in groups_data.values())
            
            f.write(f"Nombre total de sujets: {total_subjects}\n")
            f.write(f"Groupes analysés: {len(groups_data)}\n\n")
            
            for group_name, data in groups_data.items():
                f.write(f"  - {group_name}: {data['n_subjects']} sujets\n")
                
                # Calculer le score moyen global avec gestion des NaN
                subject_means = data['subject_means']
                if np.any(np.isnan(subject_means)):
                    valid_means = subject_means[~np.isnan(subject_means)]
                    if len(valid_means) > 0:
                        global_mean = np.mean(valid_means)
                        f.write(f"    Score moyen global: {global_mean:.3f} (sur {len(valid_means)} sujets valides)\n")
                    else:
                        f.write(f"    Score moyen global: N/A (aucun sujet valide)\n")
                else:
                    global_mean = np.mean(subject_means)
                    f.write(f"    Score moyen global: {global_mean:.3f}\n")
                
                f.write(f"    Sujets avec FDR significatif: {np.sum([np.any(mask) for mask in data['fdr_masks']])}\n")
                f.write(f"    Sujets avec cluster significatif: {np.sum([np.any(mask) for mask in data['cluster_masks']])}\n\n")
            
            f.write(f"Visualisations créées: {len(results['saved_plots'])}\n")
            for plot_path in results['saved_plots']:
                f.write(f"  - {os.path.basename(plot_path)}\n")
            f.write("\n" + "="*60 + "\n\n")
    
    logger.info(f"Résumé global sauvé: {summary_file}")


# === FONCTION PRINCIPALE ===
def main():
    """Fonction principale pour exécuter l'analyse."""
    try:
        run_comprehensive_protocol_analysis()
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        raise


def extract_subject_id(file_path):
    """Extraire l'ID du sujet depuis le chemin du fichier (version générale)."""
    filename = os.path.basename(file_path)
    
    # Patterns communs pour tous les types de fichiers
    patterns = [
        r'([A-Z]{2,4}\d+)_.*\.npz',     # Pattern standard de sujet
        r'([A-Z]+\d+)_.*\.npz',         # Pattern alternatif
        r'lg_([A-Z]{2,4}\d+)_',         # Pattern avec préfixe lg_
        r'subject_([A-Z0-9]+)_',        # Pattern avec préfixe subject_
        r'([A-Z]{2,4}\d+)_decoding',    # Pattern avec suffixe decoding
        r'([A-Z]+\d+)\.npz',            # Pattern simple nom.npz
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
    
    # Essayer d'extraire depuis le chemin
    if '/subject_' in file_path:
        match = re.search(r'/subject_([A-Z0-9]+)/', file_path)
        if match:
            return match.group(1)
    
    # Dernier recours : utiliser une partie du nom de fichier
    if '_' in filename:
        parts = filename.split('_')
        for part in parts:
            if re.match(r'^[A-Z]{2,4}\d+$', part):
                return part
    
    logger.warning(f"Impossible d'extraire l'ID du sujet depuis: {file_path}")
    return None


if __name__ == "__main__":
    main()
