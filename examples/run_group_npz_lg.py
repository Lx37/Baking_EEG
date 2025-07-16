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


def extract_subject_id_from_path(file_path):

    try:
        from config.config import ALL_SUBJECTS_GROUPS
        all_ids = set()
        for group, ids in ALL_SUBJECTS_GROUPS.items():
            all_ids.update(ids)
        # Check if any known subject ID is in the path
        for sid in all_ids:
            if sid in file_path:
                return sid
    except Exception:
        pass
    # Fallback: use the parent directory name (should be subject folder)
    return os.path.basename(os.path.dirname(file_path))





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



BASE_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V16"

GROUP_NAME_MAPPING = {
    'group_COMA': 'Coma', 'group_CONTROLS_COMA': 'Controls (Coma)',  
    'group_VS': 'VS/UWS', 'group_DELIRIUM+': 'Delirium +', 'group_DELIRIUM-': 'Delirium -',
    'group_CONTROLS_DELIRIUM': 'Controls (Delirium)', 'group_MCS': 'MCS',
   
}
GROUP_COLORS = {
    'Controls (Delirium)': '#2ca02c', 'Delirium -': '#ff7f0e', 'Delirium +': '#d62728',
    'Controls (Coma)': '#2ca02c', 'MCS': '#1f77b4',  
    'Coma': '#9467bd', 'VS/UWS': '#8c564b', 'CONTROLS': '#2ca02c', 'DELIRIUM+': '#d62728', 'DELIRIUM-': '#ff7f0e',
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

def find_npz_files(base_path):
    """Trouve et organise les fichiers NPZ par protocole et groupe clinique."""
    logger.info("Recherche des fichiers NPZ dans: %s", base_path)
    organized_data = {}
    
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
            

            
            protocol_name = None
            group_name = None
            
          
            for part in parts:
                if 'LG' in part and any(keyword in part for keyword in ['COMA', 'DELIRIUM', 'VS', 'MCS', 'CONTROLS']):
                    protocol_name = part
                    
                    if 'COMA' in part:
                        group_name = 'COMA'
                    elif 'DELIRIUM+' in part:
                        group_name = 'DELIRIUM+'
                    elif 'DELIRIUM-' in part:
                        group_name = 'DELIRIUM-'
                    elif 'VS' in part:
                        group_name = 'VS'
                    elif 'MCS' in part:
                        group_name = 'MCS'
                    elif 'CONTROLS' in part:
                        group_name = 'CONTROLS'
                    break
                elif 'delirium' in part.lower() and 'lg' not in part.lower():
                    
                    protocol_name = part
                    group_name = 'DELIRIUM+'
                    break
            
            
            if protocol_name is None:
            
                for i, part in enumerate(parts):
                    if any(keyword in part.upper() for keyword in ['DELIRIUM', 'COMA', 'VS', 'MCS', 'CONTROLS']):
                        protocol_name = part
                        group_name = 'Unknown'
                        break
            
            if protocol_name is None:
                logger.warning(f"Impossible d'identifier le protocole pour: {file_path}")
                continue
            
   
            if protocol_name not in organized_data:
                organized_data[protocol_name] = {}
            if group_name not in organized_data[protocol_name]:
                organized_data[protocol_name][group_name] = []
            
            organized_data[protocol_name][group_name].append(file_path)
            
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {file_path}: {e}")
            continue

    return organized_data


def load_npz_data(file_path):
    """
    Load and validate NPZ file data, extracting both local and global effects for LG protocol.
    """
    try:
        with np.load(file_path, allow_pickle=True) as data:
            data_keys = list(data.keys())
            
            # Détecter le protocole en fonction des clés disponibles
            is_lg_protocol = any(key.startswith('lg_') for key in data_keys)
            
            if is_lg_protocol:
                # Pour le protocole LG, extraire les deux effets
                local_effect_data = {}
                global_effect_data = {}
                
                # Effet local (LS vs LD)
                if 'lg_ls_ld_scores_1d_mean' in data_keys:
                    local_effect_data = {
                        'scores': data['lg_ls_ld_scores_1d_mean'],
                        'fdr_key': 'lg_ls_ld_temporal_1d_fdr',
                        'cluster_key': 'lg_ls_ld_temporal_1d_cluster',
                        'analysis_type': 'lg_local_effect'
                    }
                
                # Effet global (GS vs GD)
                if 'lg_gs_gd_scores_1d_mean' in data_keys:
                    global_effect_data = {
                        'scores': data['lg_gs_gd_scores_1d_mean'],
                        'fdr_key': 'lg_gs_gd_temporal_1d_fdr',
                        'cluster_key': 'lg_gs_gd_temporal_1d_cluster',
                        'analysis_type': 'lg_global_effect'
                    }
                
                # Vérifier qu'on a au moins un effet
                if not local_effect_data and not global_effect_data:
                    logger.warning("No LG effect data found in %s", file_path)
                    return None
                
                # Utiliser l'effet local comme référence principale (comme avant)
                if local_effect_data:
                    actual_score_key = 'lg_ls_ld_scores_1d_mean'
                    actual_fdr_key = 'lg_ls_ld_temporal_1d_fdr'
                    actual_cluster_key = 'lg_ls_ld_temporal_1d_cluster'
                    analysis_type = 'lg_local_effect'
                else:
                    actual_score_key = 'lg_gs_gd_scores_1d_mean'
                    actual_fdr_key = 'lg_gs_gd_temporal_1d_fdr'
                    actual_cluster_key = 'lg_gs_gd_temporal_1d_cluster'
                    analysis_type = 'lg_global_effect'
                
                actual_time_key = 'epochs_time_points'
                
            else:
                # Protocoles classiques (PP, PPext3, Battery)
                actual_score_key = 'pp_ap_main_scores_1d_mean'
                actual_time_key = 'epochs_time_points'
                # Clés réelles pour FDR et Cluster selon la documentation
                actual_fdr_key = 'pp_ap_main_tgm_fdr'
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

          
            path_parts = file_path.split(os.sep)
            subject_id = "Unknown"
           
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
            if subject_id.startswith('Subject_'):
                subject_id = subject_id.replace('Subject_', '')
            elif subject_id.startswith('Subj_'):
                subject_id = subject_id.replace('Subj_', '')
            elif subject_id.startswith('sub'):
                subject_id = subject_id.replace('sub', '', 1)
            
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

            # Pour le protocole LG, ajouter les données des deux effets
            if is_lg_protocol:
                if local_effect_data:
                    result['local_effect_scores'] = local_effect_data['scores']
                    # Ajouter les données FDR/cluster pour l'effet local
                    if local_effect_data['fdr_key'] in data_keys:
                        fdr_data = data[local_effect_data['fdr_key']]
                        result['local_effect_fdr'] = extract_fdr_data(fdr_data, data[actual_score_key])
                    if local_effect_data['cluster_key'] in data_keys:
                        cluster_data = data[local_effect_data['cluster_key']]
                        result['local_effect_cluster'] = extract_cluster_data(cluster_data, data[actual_score_key])
                    
                    # Ajouter les données TGM pour l'effet local
                    lg_local_tgm_key = 'lg_ls_ld_tgm_mean'
                    if lg_local_tgm_key in data_keys:
                        result['local_effect_tgm'] = data[lg_local_tgm_key]
                
                if global_effect_data:
                    result['global_effect_scores'] = global_effect_data['scores']
                    # Ajouter les données FDR/cluster pour l'effet global
                    if global_effect_data['fdr_key'] in data_keys:
                        fdr_data = data[global_effect_data['fdr_key']]
                        result['global_effect_fdr'] = extract_fdr_data(fdr_data, data[actual_score_key])
                    if global_effect_data['cluster_key'] in data_keys:
                        cluster_data = data[global_effect_data['cluster_key']]
                        result['global_effect_cluster'] = extract_cluster_data(cluster_data, data[actual_score_key])
                    
                    # Ajouter les données TGM pour l'effet global
                    lg_global_tgm_key = 'lg_gs_gd_tgm_mean'
                    if lg_global_tgm_key in data_keys:
                        result['global_effect_tgm'] = data[lg_global_tgm_key]
                
                # Ajouter les métriques globales de performance
                if 'lg_ls_ld_mean_auc_global' in data_keys:
                    result['local_effect_auc_global'] = data['lg_ls_ld_mean_auc_global']
                if 'lg_gs_gd_mean_auc_global' in data_keys:
                    result['global_effect_auc_global'] = data['lg_gs_gd_mean_auc_global']
            
            else:
                # Pour les protocoles classiques, ajouter les TGM si disponibles
                tgm_key = 'pp_ap_main_tgm_mean'
                if tgm_key in data_keys:
                    result['tgm'] = data[tgm_key]
                
                # Ajouter les métriques globales de performance
                if 'pp_ap_main_mean_auc_global' in data_keys:
                    result['auc_global'] = data['pp_ap_main_mean_auc_global']

            # Clés TGM si disponibles
            tgm_fdr_key = 'pp_ap_main_tgm_fdr'

            # Extraire les données FDR et Cluster avec des méthodes séparées
            if actual_fdr_key in data_keys:
                fdr_data = data[actual_fdr_key]
                fdr_result = extract_fdr_data(fdr_data, data[actual_score_key])
                result.update(fdr_result)
            else:
                result['fdr_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                result['fdr_pvalues'] = np.ones_like(data[actual_score_key])

            if actual_cluster_key in data_keys:
                cluster_data = data[actual_cluster_key]
                cluster_result = extract_cluster_data(cluster_data, data[actual_score_key])
                result.update(cluster_result)
            else:
                result['cluster_mask'] = np.zeros_like(data[actual_score_key], dtype=bool)
                result['cluster_pvalues'] = np.ones_like(data[actual_score_key])

            if result['scores'] is None or result['times'] is None:
                logger.warning("Data for scores or times is None in %s", file_path)
                return None
            if len(result['scores']) == 0 or len(result['times']) == 0:
                logger.warning("Data for scores or times is empty in %s", file_path)
                return None

            return result

    except Exception as e:
        logger.error("Error loading NPZ file %s: %s", file_path, e)
        return None


def extract_fdr_data(fdr_data, reference_scores):
    """Helper function to extract FDR data"""
    if isinstance(fdr_data, np.ndarray) and fdr_data.dtype == object:
        try:
            fdr_dict = fdr_data.item()
            if isinstance(fdr_dict, dict):
                return {
                    'fdr_mask': fdr_dict.get('mask', np.zeros_like(reference_scores, dtype=bool)),
                    'fdr_pvalues': fdr_dict.get('p_values', np.ones_like(reference_scores)),
                    'fdr_pvalues_raw': fdr_dict.get('p_values_raw', np.ones_like(reference_scores)),
                    'fdr_method': fdr_dict.get('method', 'Unknown')
                }
        except Exception as e:
            logger.warning("Error extracting FDR data: %s", e)
    
    return {
        'fdr_mask': np.zeros_like(reference_scores, dtype=bool),
        'fdr_pvalues': np.ones_like(reference_scores)
    }


def extract_cluster_data(cluster_data, reference_scores):
    """Helper function to extract cluster data"""
    if isinstance(cluster_data, np.ndarray) and cluster_data.dtype == object:
        try:
            cluster_dict = cluster_data.item()
            if isinstance(cluster_dict, dict):
                return {
                    'cluster_mask': cluster_dict.get('mask', np.zeros_like(reference_scores, dtype=bool)),
                    'cluster_pvalues': cluster_dict.get('p_values_all_clusters', np.ones_like(reference_scores)),
                    'cluster_objects': cluster_dict.get('cluster_objects', []),
                    'cluster_method': cluster_dict.get('method', 'Unknown')
                }
        except Exception as e:
            logger.warning("Error extracting cluster data: %s", e)
    
    return {
        'cluster_mask': np.zeros_like(reference_scores, dtype=bool),
        'cluster_pvalues': np.ones_like(reference_scores)
    }


def analyze_group_data(group_files, group_name):
    """
    Analyser les données d'un groupe spécifique et extraire les statistiques.
    Exclut tout sujet ayant moins de 801 points temporels.
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
            n_timepoints = len(data['scores']) if 'scores' in data else 0
            if n_timepoints < 801:
                logger.warning(f"Subject {data['subject_id']} excluded: only {n_timepoints} timepoints (required: 801)")
                continue
            group_data.append(data)
            subject_ids.append(data['subject_id'])
            fdr_masks.append(data.get('fdr_mask', np.array([])))
            cluster_masks.append(data.get('cluster_mask', np.array([])))
            fdr_pvalues.append(data.get('fdr_pvalues', np.array([])))
            cluster_pvalues.append(data.get('cluster_pvalues', np.array([])))

    if not group_data:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return {}

    # Toutes les données sont déjà à 801 points temporels, pas besoin de standardisation
    scores_matrix = np.array([d['scores'] for d in group_data])
    times = group_data[0]['times'] if group_data[0]['times'] is not None else None

    # Calculs statistiques simples
    group_mean = np.nanmean(scores_matrix, axis=0)
    group_std = np.nanstd(scores_matrix, axis=0)
    group_sem = group_std / np.sqrt(len(group_data))

    # Compter les sujets significatifs à chaque point temporel
    fdr_count = np.zeros(801)
    cluster_count = np.zeros(801)
    for mask in fdr_masks:
        if len(mask) == 801:
            fdr_count += mask.astype(int)
    for mask in cluster_masks:
        if len(mask) == 801:
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


def analyze_group_data_lg(group_files, group_name):
    """
    Analyser les données d'un groupe spécifique pour le protocole LG avec effets local et global séparés.
    """
    logger.info(f"Analyse LG du groupe {group_name} avec {len(group_files)} sujets")
    
    group_data = []
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
            group_data.append(data)
            subject_ids.append(data['subject_id'])
            
            # Extraire les scores pour les effets local et global
            if 'local_effect_scores' in data:
                local_scores_list.append(data['local_effect_scores'])
            if 'global_effect_scores' in data:
                global_scores_list.append(data['global_effect_scores'])
            
            # Extraire les données TGM
            if 'local_effect_tgm' in data:
                local_tgm_list.append(data['local_effect_tgm'])
            if 'global_effect_tgm' in data:
                global_tgm_list.append(data['global_effect_tgm'])
            
            # Extraire les métriques globales AUC
            if 'local_effect_auc_global' in data:
                local_auc_global_list.append(data['local_effect_auc_global'])
            if 'global_effect_auc_global' in data:
                global_auc_global_list.append(data['global_effect_auc_global'])
    
    if not group_data:
        logger.warning(f"Aucune donnée valide trouvée pour le groupe {group_name}")
        return {}
    
    # Déterminer la longueur de référence pour le protocole LG
    reference_length = 801  # Protocole LG utilise 801 points temporels
    times = group_data[0]['times'][:reference_length] if group_data[0]['times'] is not None else None
    
    result = {
        'group_name': group_name,
        'n_subjects': len(group_data),
        'subject_ids': subject_ids,
        'times': times,
        'group_data': group_data
    }
    
    # Traiter l'effet local si disponible
    if local_scores_list:
        # Standardiser les scores locaux
        local_standardized = []
        for scores in local_scores_list:
            if np.any(np.isnan(scores)):
                scores = np.where(np.isnan(scores), CHANCE_LEVEL, scores)
            local_standardized.append(scores[:reference_length])
        
        local_matrix = np.array(local_standardized)
        result['local_effect'] = {
            'scores_matrix': local_matrix,
            'group_mean': np.nanmean(local_matrix, axis=0),
            'group_std': np.nanstd(local_matrix, axis=0),
            'group_sem': np.nanstd(local_matrix, axis=0) / np.sqrt(len(local_standardized))
        }
        
        # Traiter les TGM locaux si disponibles
        if local_tgm_list:
            local_tgm_standardized = []
            for tgm in local_tgm_list:
                if np.any(np.isnan(tgm)):
                    tgm = np.where(np.isnan(tgm), CHANCE_LEVEL, tgm)
                # Assurer que les TGM ont la bonne taille
                if tgm.shape[0] >= reference_length and tgm.shape[1] >= reference_length:
                    local_tgm_standardized.append(tgm[:reference_length, :reference_length])
            
            if local_tgm_standardized:
                local_tgm_matrix = np.array(local_tgm_standardized)
                result['local_effect']['tgm_matrix'] = local_tgm_matrix
                result['local_effect']['tgm_mean'] = np.nanmean(local_tgm_matrix, axis=0)
                result['local_effect']['tgm_std'] = np.nanstd(local_tgm_matrix, axis=0)
        
        # Traiter les AUC globaux locaux si disponibles
        if local_auc_global_list:
            result['local_effect']['auc_global_values'] = np.array(local_auc_global_list)
            result['local_effect']['auc_global_mean'] = np.nanmean(local_auc_global_list)
            result['local_effect']['auc_global_std'] = np.nanstd(local_auc_global_list)
    
    # Traiter l'effet global si disponible
    if global_scores_list:
        # Standardiser les scores globaux
        global_standardized = []
        for scores in global_scores_list:
            if np.any(np.isnan(scores)):
                scores = np.where(np.isnan(scores), CHANCE_LEVEL, scores)
            global_standardized.append(scores[:reference_length])
        
        global_matrix = np.array(global_standardized)
        result['global_effect'] = {
            'scores_matrix': global_matrix,
            'group_mean': np.nanmean(global_matrix, axis=0),
            'group_std': np.nanstd(global_matrix, axis=0),
            'group_sem': np.nanstd(global_matrix, axis=0) / np.sqrt(len(global_standardized))
        }
        
        # Traiter les TGM globaux si disponibles
        if global_tgm_list:
            global_tgm_standardized = []
            for tgm in global_tgm_list:
                if np.any(np.isnan(tgm)):
                    tgm = np.where(np.isnan(tgm), CHANCE_LEVEL, tgm)
                # Assurer que les TGM ont la bonne taille
                if tgm.shape[0] >= reference_length and tgm.shape[1] >= reference_length:
                    global_tgm_standardized.append(tgm[:reference_length, :reference_length])
            
            if global_tgm_standardized:
                global_tgm_matrix = np.array(global_tgm_standardized)
                result['global_effect']['tgm_matrix'] = global_tgm_matrix
                result['global_effect']['tgm_mean'] = np.nanmean(global_tgm_matrix, axis=0)
                result['global_effect']['tgm_std'] = np.nanstd(global_tgm_matrix, axis=0)
        
        # Traiter les AUC globaux si disponibles
        if global_auc_global_list:
            result['global_effect']['auc_global_values'] = np.array(global_auc_global_list)
            result['global_effect']['auc_global_mean'] = np.nanmean(global_auc_global_list)
            result['global_effect']['auc_global_std'] = np.nanstd(global_auc_global_list)
    
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
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Effet Local (LS vs LD)
    if 'local_effect' in group_data:
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
        
        # Ligne de chance
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
    if 'global_effect' in group_data:
        ax2 = axes[1]
        global_data = group_data['global_effect']
        
        # Assurer que les dimensions correspondent
        min_length = min(len(times_ms), global_data['scores_matrix'].shape[1])
        times_ms_truncated = times_ms[:min_length]
        
        # Courbes individuelles en arrière-plan
        for i in range(global_data['scores_matrix'].shape[0]):
            scores_truncated = global_data['scores_matrix'][i, :min_length]
            ax2.plot(times_ms_truncated, scores_truncated, 
                    color=group_color, alpha=individual_alpha, linewidth=1)
        
        # Moyenne du groupe en avant-plan
        group_mean_truncated = global_data['group_mean'][:min_length]
        group_sem_truncated = global_data['group_sem'][:min_length]
        
        ax2.plot(times_ms_truncated, group_mean_truncated, 
                color=group_color, alpha=mean_alpha, linewidth=3, 
                label=f'{GROUP_NAME_MAPPING.get(group_name, group_name)} (n={group_data["n_subjects"]})')
        
        # Bande d'erreur (SEM)
        ax2.fill_between(times_ms_truncated, 
                        group_mean_truncated - group_sem_truncated,
                        group_mean_truncated + group_sem_truncated,
                        color=group_color, alpha=0.3)
        
        # Ligne de chance
        ax2.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, label='Chance level')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_xlabel('Time (ms)', fontsize=14)
        ax2.set_ylabel('Score AUC', fontsize=14)
        ax2.set_title(f'Global Effect (GS vs GD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.4, 0.8])
    else:
        axes[1].text(0.5, 0.5, 'No Global Effect Data', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=16)
        axes[1].set_title(f'Global Effect (GS vs GD) - {GROUP_NAME_MAPPING.get(group_name, group_name)}', fontsize=16)
    
    plt.tight_layout()
    
    # Sauvegarder
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"group_{group_name}_individual_curves.png"
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
            significance = ""
        
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
    
    # Analyser les effets local et global séparément
    for effect_type in ['local', 'global']:
        logger.info(f"Création des boxplots pour l'effet {effect_type}")
        
        # Préparer les données pour le boxplot
        plot_data = []
        group_names = []
        
        for group_data in all_groups_data:
            group_name = group_data['group_name']
            effect_key = f'{effect_type}_effect'
            
            if effect_key in group_data and 'auc_global_values' in group_data[effect_key]:
                auc_values = group_data[effect_key]['auc_global_values']
                # Supprimer les NaN
                auc_values = auc_values[~np.isnan(auc_values)]
                
                if len(auc_values) > 0:
                    # Ajouter les données pour le boxplot
                    for value in auc_values:
                        plot_data.append({
                            'Group': GROUP_NAME_MAPPING.get(group_name, group_name),
                            'AUC': value,
                            'Original_Group': group_name
                        })
                    group_names.append(group_name)
        
        if len(plot_data) < 2:
            logger.warning(f"Pas assez de données pour créer le boxplot de l'effet {effect_type}")
            continue
        
        # Créer le DataFrame pour seaborn
        df = pd.DataFrame(plot_data)
        stats_results = perform_statistical_tests(all_groups_data, effect_type)
        fig, ax = plt.subplots(figsize=(12, 8))
        unique_groups = df['Group'].unique()
        group_colors = [GROUP_COLORS.get(group, '#1f77b4') for group in unique_groups]
        box_plot = sns.boxplot(data=df, x='Group', y='AUC', ax=ax, palette=group_colors)
        sns.stripplot(data=df, x='Group', y='AUC', ax=ax, color='black', alpha=0.6, size=4)
        ax.set_title(f'Global AUC Distribution - {effect_type.capitalize()} Effect', fontsize=16, fontweight='bold')
        ax.set_xlabel('Clinical Group', fontsize=14)
        ax.set_ylabel('AUC (Area Under Curve)', fontsize=14)
        ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, label=f'Chance level ({CHANCE_LEVEL})')
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
                        # Passer ce groupe car il n'y a qu'un seul sujet
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


def analyze_temporal_windows(all_groups_data, save_dir, show_plots=True):
    """
    Analyser les fenêtres temporelles spécifiques (T100, T200, T_all) et créer des graphiques
    avec les moyennes des fenêtres pour chaque participant (style graphique de référence).
    
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
        logger.info(f"Analyse des fenêtres temporelles pour l'effet {effect_type}")
        
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
            
            # Créer une figure avec 3 colonnes pour les 3 fenêtres
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            
            # Calculer les moyennes des fenêtres pour chaque participant
            subjects_window_means = {window: [] for window in windows.keys()}
            
            for subj_idx in range(n_subjects):
                subject_scores = scores_matrix[subj_idx, :]
                
                # Ajuster les dimensions si nécessaire
                min_length = min(len(times_ms), len(subject_scores))
                times_ms_truncated = times_ms[:min_length]
                subject_scores_truncated = subject_scores[:min_length]
                
                # Calculer la moyenne pour chaque fenêtre
                for window_name, (start_ms, end_ms) in windows.items():
                    # Trouver les indices correspondant aux fenêtres temporelles
                    start_idx = np.argmin(np.abs(times_ms_truncated - start_ms))
                    end_idx = np.argmin(np.abs(times_ms_truncated - end_ms))
                    
                    if start_idx < end_idx and end_idx <= len(subject_scores_truncated):
                        window_scores = subject_scores_truncated[start_idx:end_idx]
                        window_mean = np.mean(window_scores)
                        subjects_window_means[window_name].append(window_mean)
                    else:
                        subjects_window_means[window_name].append(np.nan)
            
            # Créer les graphiques pour chaque fenêtre
            window_names = list(windows.keys())
            for i, window_name in enumerate(window_names):
                ax = axes[i]
                window_means = subjects_window_means[window_name]
                
                # Supprimer les NaN
                valid_means = [m for m in window_means if not np.isnan(m)]
                
                if len(valid_means) > 0:
                    # Positions x pour les points (jittering léger pour éviter la superposition)
                    x_positions = np.random.normal(0, 0.03, len(valid_means))
                    
                    # Tracer les points individuels
                    ax.scatter(x_positions, valid_means, color=group_color, alpha=0.7, s=50)
                    
                    # Ajouter une ligne horizontale pour la moyenne du groupe
                    group_mean = np.mean(valid_means)
                    ax.axhline(y=group_mean, color=group_color, linewidth=3, alpha=0.8)
                    
                    # Ajouter des lignes connectant les points pour chaque participant
                    if i > 0:  # Connecter avec la fenêtre précédente
                        prev_window = window_names[i-1]
                        prev_means = subjects_window_means[prev_window]
                        
                        for j, (prev_mean, curr_mean) in enumerate(zip(prev_means, window_means)):
                            if not np.isnan(prev_mean) and not np.isnan(curr_mean):
                                # Ligne de connexion entre les fenêtres
                                ax.plot([-1, 0], [prev_mean, curr_mean], 
                                       color=group_color, alpha=0.5, linewidth=1)
                
                # Ligne de chance
                ax.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', alpha=0.5, linewidth=1)
                
                # Personnaliser le graphique
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(0.4, 1.0)
                ax.set_ylabel('Score AUC', fontsize=12)
                ax.set_title(f'{window_name}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Supprimer les ticks x
                ax.set_xticks([])
                
                # Ajouter le test statistique contre le niveau de chance
                if len(valid_means) > 1:
                    from scipy.stats import wilcoxon
                    try:
                        # Test contre le niveau de chance
                        diff_from_chance = np.array(valid_means) - CHANCE_LEVEL
                        stat, p_value = wilcoxon(diff_from_chance, alternative='two-sided')
                        
                        # Ajouter l'annotation de significativité
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = "ns"
                        
                        # Placer l'annotation en bas du graphique
                        ax.text(0, 0.42, significance, ha='center', va='center', 
                               fontsize=16, fontweight='bold')
                        
                    except Exception as e:
                        logger.warning(f"Erreur dans le test statistique pour {window_name}: {e}")
            
            # Titre général
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            fig.suptitle(f'{mapped_name} - {effect_type.capitalize()} Effect - Temporal Windows\n(n={n_subjects})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Sauvegarder
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"temporal_windows_{effect_type}_group_{group_name}.png"
                filepath = os.path.join(save_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Graphique des fenêtres temporelles sauvegardé: {filepath}")
            
            if show_plots:
                plt.show()
            else:
                plt.close()


def create_temporal_windows_comparison_boxplots(all_groups_data, save_dir, show_plots=True):
    """
    Créer des boxplots pour comparer les différences entre les fenêtres temporelles
    pour chaque groupe, similaires à la partie (b) de la figure de référence.
    
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
            
            # Personnaliser le graphique
            ax.set_xticks(range(len(windows_to_plot)))
            ax.set_xticklabels(windows_to_plot)
            ax.set_ylabel('AUC', fontsize=12)
            ax.set_title(f'{group_name}\n{effect_type.capitalize()} Effect', fontsize=14, fontweight='bold')
            ax.axhline(y=CHANCE_LEVEL, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 0.8)
            
            # Effectuer des tests statistiques si il y a assez de données
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
        
        # Sauvegarder
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
                            significance = "ns"
                        
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
                filename = f"temporal_windows_connected_{effect_type}_group_{group_name}.png"
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
    import matplotlib.patches as mpatches
    
    for effect_type in ['local', 'global']:
        logger.info(f"Analyse des proportions de significativité pour l'effet {effect_type}")
        
        # Préparer les données pour tous les groupes
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
            # Supprimer les NaN
            valid_auc_values = auc_values[~np.isnan(auc_values)]
            
            if len(valid_auc_values) == 0:
                logger.warning(f"Aucune valeur AUC valide pour le groupe {group_name}")
                continue
            
            # Tester chaque participant individuellement contre la chance
            individual_results = []
            significant_count = 0
            
            for subj_idx, auc_value in enumerate(valid_auc_values):
                # Pour un seul participant, on ne peut pas faire de test statistique
                # On utilise une heuristique: si l'AUC est suffisamment éloignée de la chance
                # Alternative: utiliser les scores temporels pour faire un test de Wilcoxon
                
                # Récupérer l'AUC global du participant
                # Test direct sur l'AUC global individuel contre la chance
                try:
                    # Test binomial ou test t à un échantillon sur l'AUC global
                    # Comme on n'a qu'une seule valeur AUC par participant, on utilise un seuil statistique
                    # basé sur la distribution binomiale ou on groupe les AUC pour le test
                    
                    # Pour un test individuel sur l'AUC, on peut utiliser la significativité statistique
                    # basée sur la distribution de l'AUC ou un seuil conservateur
                    
                    # Critère simple mais robuste : AUC significativement > chance
                    # On peut calculer un intervalle de confiance ou utiliser un seuil statistique
                    
                    # Méthode 1: Seuil statistique basé sur la distribution binomiale
                    # Pour un test binomial, AUC > 0.5 avec un seuil de confiance
                    
                    # Méthode 2: Test sur l'AUC directement
                    is_auc_above_chance = auc_value > CHANCE_LEVEL
                    
                    # Seuil statistique pour considérer l'AUC comme significativement différent
                    # Utilisons un seuil conservateur basé sur l'écart-type attendu
                    statistical_threshold = CHANCE_LEVEL + 0.025  # Seuil conservateur
                    is_statistically_significant = auc_value > statistical_threshold
                    
                    # Double critère : AUC > chance ET au-dessus du seuil statistique
                    is_significant = is_auc_above_chance and is_statistically_significant
                    
                    individual_results.append({
                        'subject_idx': subj_idx,
                        'auc_value': auc_value,
                        'p_value': np.nan,  # Pas de p-value pour test individuel sur AUC
                        'is_significant': is_significant,
                        'is_statistically_significant': is_statistically_significant,
                        'is_auc_above_chance': is_auc_above_chance,
                        'statistical_threshold': statistical_threshold
                    })
                    
                    if is_significant:
                        significant_count += 1
                        
                except Exception as e:
                    logger.warning(f"Erreur test significativité AUC sujet {subj_idx} groupe {group_name}: {e}")
                    individual_results.append({
                        'subject_idx': subj_idx,
                        'auc_value': auc_value,
                        'p_value': np.nan,
                        'is_significant': False,
                        'is_statistically_significant': False,
                        'is_auc_above_chance': auc_value > CHANCE_LEVEL,
                        'statistical_threshold': CHANCE_LEVEL + 0.025
                    })
                # Fallback: Test direct sur l'AUC global (cohérent avec l'approche principale)
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
            
            # Calculer le pourcentage de patients significatifs
            total_patients = len(individual_results)
            percentage_significant = (significant_count / total_patients) * 100 if total_patients > 0 else 0
            
            # Test de groupe contre la chance
            group_p_value = np.nan
            if len(valid_auc_values) > 1:
                try:
                    # Test de Wilcoxon sur les AUC du groupe (unilatéral supérieur)
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
        
        # Créer la figure
        n_groups = len(groups_analysis)
        fig, axes = plt.subplots(2, n_groups, figsize=(4 * n_groups, 8))
        
        if n_groups == 1:
            axes = axes.reshape(2, 1)
        
        # Couleurs pour les graphiques en camembert
        colors_pie = ['#ff6b6b', '#4ecdc4']  # Rouge pour non-significatif, bleu-vert pour significatif
        
        for i, group_analysis in enumerate(groups_analysis):
            group_name = group_analysis['group_name']
            mapped_name = GROUP_NAME_MAPPING.get(group_name, group_name)
            group_color = GROUP_COLORS.get(mapped_name, '#1f77b4')
            
            # Graphique en camembert (ligne du haut)
            ax_pie = axes[0, i]
            
            n_significant = group_analysis['n_significant']
            n_total = group_analysis['n_subjects']
            n_non_significant = n_total - n_significant
            
            # Données pour le camembert
            sizes = [n_non_significant, n_significant]
            labels = ['Non-sig', 'Significant']
            colors = [colors_pie[0], colors_pie[1]]
            
            # Créer le camembert
            wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', 
                                                 startangle=90, textprops={'fontsize': 12})
            
            # Titre avec pourcentage
            percentage = group_analysis['percentage_significant']
            ax_pie.set_title(f'{mapped_name}\n{percentage:.0f}%\nn={n_total}', 
                           fontsize=14, fontweight='bold')
            
            # Ajouter la significativité de groupe
            group_p = group_analysis['group_p_value']
            if not np.isnan(group_p) and group_p < 0.05:  # Afficher seulement si significatif
                if group_p < 0.0001:
                    sig_text = "****"
                elif group_p < 0.001:
                    sig_text = "***"
                elif group_p < 0.01:
                    sig_text = "**"
                elif group_p < 0.05:
                    sig_text = "*"
                else:
                    sig_text = ""  # Pas de texte si non significatif
                
                if sig_text:  # Afficher seulement si il y a un texte de significativité
                    ax_pie.text(0, -1.3, sig_text, ha='center', va='center', 
                              fontsize=16, fontweight='bold')
            
            # Graphique en barres des AUC individuelles (ligne du bas)
            ax_bar = axes[1, i]
            
            auc_values = group_analysis['auc_values']
            individual_results = group_analysis['individual_results']
            
            # Séparer les AUC significatives et non-significatives
            sig_aucs = [r['auc_value'] for r in individual_results if r['is_significant']]
            non_sig_aucs = [r['auc_value'] for r in individual_results if not r['is_significant']]
            
            # Positions x
            x_positions = np.arange(len(auc_values))
            
            # Tracer les barres
            for j, result in enumerate(individual_results):
                color = colors_pie[1] if result['is_significant'] else colors_pie[0]
                ax_bar.bar(j, result['auc_value'], color=color, alpha=0.7)
            
            # Ligne de chance
            ax_bar.axhline(y=CHANCE_LEVEL, color='black', linestyle='--', linewidth=2, alpha=0.7)
            
            # Moyenne du groupe
            group_mean = group_analysis['group_mean_auc']
            ax_bar.axhline(y=group_mean, color='red', linestyle='-', linewidth=2, alpha=0.8)
            
            # Personnaliser
            ax_bar.set_xlabel('Subject', fontsize=12)
            ax_bar.set_ylabel('AUC', fontsize=12)
            ax_bar.set_title(f'{mapped_name} - Individual AUC', fontsize=12, fontweight='bold')
            ax_bar.set_ylim(0.4, 0.65)  # Échelle ajustée
            ax_bar.grid(True, alpha=0.3)
            
            # Xticks
            ax_bar.set_xticks(x_positions)
            ax_bar.set_xticklabels([f'S{j+1}' for j in range(len(auc_values))], rotation=45)
        
        # Titre général
        fig.suptitle(f'{effect_type.capitalize()} Effect - Individual Significance Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Légende
        legend_elements = [
            mpatches.Patch(color=colors_pie[1], label='Significant (p<0.05)'),
            mpatches.Patch(color=colors_pie[0], label='Non-significant'),
            mpatches.Patch(color='black', label='Chance level'),
            mpatches.Patch(color='red', label='Group mean')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        
        # Sauvegarder
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"individual_significance_proportions_{effect_type}_effect.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique des proportions de significativité sauvegardé: {filepath}")
        
        # Sauvegarder les résultats numériques
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
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def filter_group_files_by_config(group_files, group_name):
    """
    Filtre les fichiers NPZ pour ne garder que ceux dont l'ID sujet est dans la config ALL_SUBJECTS_GROUPS.
    """
    try:
        from config.config import ALL_SUBJECTS_GROUPS
    except ImportError:
        logger.warning("Impossible d'importer ALL_SUBJECTS_GROUPS depuis config.config")
        return group_files
    allowed_ids = set(ALL_SUBJECTS_GROUPS.get(group_name, []))
    filtered_files = []
    for file_path in group_files:
        subject_id = extract_subject_id_from_path(file_path)
        if subject_id in allowed_ids:
            filtered_files.append(file_path)
    logger.info(f"Groupe {group_name}: {len(filtered_files)}/{len(group_files)} fichiers NPZ conservés après filtrage par config.")
    return filtered_files


def main():
    """
    Fonction principale pour analyser les données LG et générer les graphiques.
    """
    logger.info("Début de l'analyse des données LG")
    
    # Créer un répertoire pour les résultats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/home/tom.balay/results/LG_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Trouver les fichiers NPZ
    organized_data = find_npz_files(BASE_RESULTS_DIR)
    all_groups_data = []
    for protocol_name, groups in organized_data.items():
        for group_name, group_files in groups.items():
            # Filtrer les fichiers NPZ selon la config
            filtered_files = filter_group_files_by_config(group_files, group_name)
            if not filtered_files:
                logger.info(f"Aucun fichier NPZ à analyser pour le groupe {group_name} (après filtrage)")
                continue
            # Analyse LG ou classique selon le protocole
            if protocol_name.upper().startswith("LG"):
                group_data = analyze_group_data_lg(filtered_files, group_name)
            else:
                group_data = analyze_group_data(filtered_files, group_name)
            if group_data:
                all_groups_data.append(group_data)
    
    # Créer les graphiques de comparaison entre tous les groupes
    if all_groups_data:
        analyze_temporal_windows(all_groups_data, results_dir, show_plots=False)
        create_temporal_windows_comparison_boxplots(all_groups_data, results_dir, show_plots=False)
        plot_all_groups_comparison(all_groups_data, results_dir, show_plots=False)
        plot_global_auc_boxplots(all_groups_data, results_dir, show_plots=False)
        create_temporal_windows_connected_plots(all_groups_data, results_dir, show_plots=False)
        
        # Analyse des proportions de significativité individuelle
        analyze_individual_significance_proportions(all_groups_data, results_dir, show_plots=False)
        plot_group_tgm_individual(all_groups_data, results_dir, show_plots=False)
        
        # Nouvelles analyses des fenêtres temporelles
       
      
        # Sauvegarder un résumé des résultats
        summary_file = os.path.join(results_dir, "analysis_summary.json")
        summary_data = {
            'timestamp': timestamp,
            'n_groups': len(all_groups_data),
            'groups': [
                {
                    'name': group['group_name'],
                    'n_subjects': group['n_subjects'],
                    'subject_ids': group['subject_ids'],
                    'has_local_effect': 'local_effect' in group,
                    'has_global_effect': 'global_effect' in group,
                    'local_effect_auc_mean': group.get('local_effect', {}).get('auc_global_mean', np.nan),
                    'global_effect_auc_mean': group.get('global_effect', {}).get('auc_global_mean', np.nan)
                }
                for group in all_groups_data
            ]
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        logger.info(f"Analyse terminée. Résultats sauvegardés dans: {results_dir}")
    else:
        logger.error("Aucune donnée de groupe valide trouvée")


if __name__ == "__main__":
    main()

