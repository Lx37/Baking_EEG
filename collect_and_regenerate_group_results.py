#!/usr/bin/env python3
"""
Script pour collecter tous les fichiers NPZ des analyses individuelles et régénérer 
les courbes de groupe pour le décodage EEG.

Ce script :
1. Scanne récursivement le répertoire des résultats pour trouver les fichiers NPZ
2. Extrait et organise les données de chaque sujet par groupe
3. Recalcule les statistiques de groupe (moyennes, SEM, etc.)
4. Régénère les visualisations de groupe à partir des données existantes
5. Sauvegarde les nouveaux résultats de groupe


"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict, OrderedDict
from getpass import getuser
import traceback
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration des chemins de base
def get_user_paths():
    """Détermine les chemins utilisateur selon la configuration actuelle"""
    current_user = getuser()
    
    # Configuration des chemins utilisateur (basée sur utils.py)
    user_output_results_paths = {
        "tom.balay": "/home/tom.balay/results/Baking_EEG_results_V7",
        "tom": "/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_results_V7",
    }
    
    # Chemins de fallback pour les données
    fallback_paths = [
        "/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_results_V7",
        "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG/results",
        "./results",
    ]
    
    base_output_path = user_output_results_paths.get(current_user)
    
    # Si le chemin utilisateur n'existe pas, chercher des alternatives
    if not base_output_path or not os.path.exists(base_output_path):
        for fallback in fallback_paths:
            if os.path.exists(fallback):
                base_output_path = fallback
                break
    
    return base_output_path, current_user

def setup_logging():
    """Configure le système de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'collect_group_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

class NPZDataCollector:
    """Collecteur et analyseur de fichiers NPZ de décodage EEG"""
    
    def __init__(self, base_results_path: str, logger: logging.Logger):
        self.base_results_path = base_results_path
        self.logger = logger
        self.subjects_data = defaultdict(dict)
        self.group_assignments = {}
        self.protocols_found = set()
        self.classifiers_found = set()
        
        # Configuration des groupes standard (basée sur config.py)
        self.ALL_SUBJECT_GROUPS = {
            'controls': [
                'LC97', 'AG42', 'AS33', 'CC36', 'CM25', 'EB56', 'FC29', 'FS74', 'JB64', 'LE39',
                'LH69', 'MD52', 'MP49', 'MV19', 'NL76', 'PS77', 'RG47', 'SC46', 'SV68', 'XL89'
            ],
            'del': [
                'AD59', 'CB85', 'CF58', 'EG22', 'GC99', 'HM35', 'LC78', 'MC72', 'MH32', 'ML83',
                'MM91', 'MP86', 'PS99', 'RC28', 'YP38'
            ],
            'nodel': [
                'AC55', 'AD94', 'CB58', 'CL81', 'CP45', 'DC69', 'GB89', 'JC93', 'JS26', 'LG62',
                'MJ73', 'MM85', 'MP92', 'MT84', 'RO57', 'VM72'
            ]
        }
    
    def find_npz_files(self) -> List[str]:
        """Trouve tous les fichiers NPZ dans la hiérarchie des résultats"""
        npz_patterns = [
            "**/decoding_results_full.npz",  # Analyses individuelles PP
            "**/lg_decoding_results_full.npz",  # Analyses individuelles LG
            "**/*_results_*.npz",  # Autres formats
            "**/group_*_results_complete.npz"  # Résultats de groupe existants
        ]
        
        found_files = []
        for pattern in npz_patterns:
            full_pattern = os.path.join(self.base_results_path, pattern)
            files = glob.glob(full_pattern, recursive=True)
            found_files.extend(files)
        
        # Éliminer les doublons et trier
        found_files = list(set(found_files))
        found_files.sort()
        
        self.logger.info(f"Trouvé {len(found_files)} fichiers NPZ dans {self.base_results_path}")
        return found_files
    
    def extract_subject_info_from_path(self, file_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extrait l'ID sujet, groupe et protocole depuis le chemin du fichier"""
        path_parts = Path(file_path).parts
        
        # Recherche de l'ID sujet dans le chemin
        subject_id = None
        group = None
        protocol = None
        
        # Patterns d'identification des sujets
        for part in path_parts:
            # Chercher des IDs de type XX## (lettres + chiffres)
            if len(part) >= 4 and part[:2].isalpha() and part[2:4].isdigit():
                potential_subject = part[:4]
                # Vérifier si c'est un sujet connu
                for group_name, subjects in self.ALL_SUBJECT_GROUPS.items():
                    if potential_subject in subjects:
                        subject_id = potential_subject
                        group = group_name
                        break
                if subject_id:
                    break
        
        # Détection du protocole depuis le chemin
        for part in path_parts:
            if 'intra_subject_results' in part.lower():
                protocol = 'PP'  # Par défaut PP pour les analyses individuelles
            elif 'intra_subject_lg' in part.lower():
                protocol = 'LG'
            elif any(x in part.lower() for x in ['pp', 'primary']):
                protocol = 'PP'
            elif any(x in part.lower() for x in ['lg', 'lexical']):
                protocol = 'LG'
        
        # Si pas de protocole détecté, essayer depuis le nom de fichier
        filename = os.path.basename(file_path)
        if not protocol:
            if 'lg_' in filename.lower():
                protocol = 'LG'
            elif any(x in filename.lower() for x in ['decoding_results', 'pp']):
                protocol = 'PP'
        
        return subject_id, group, protocol
    
    def load_and_validate_npz(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Charge et valide un fichier NPZ"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Vérifier que c'est bien un fichier de décodage
            required_keys_pp = [
                'subject_id', 'epochs_time_points', 
                'pp_ap_main_scores_1d_mean', 'pp_ap_main_mean_auc_global'
            ]
            required_keys_lg = [
                'subject_id', 'epochs_time_points',
                'lg_main_scores_1d_mean', 'lg_main_mean_auc_global'
            ]
            
            available_keys = list(data.keys())
            
            # Déterminer le type de fichier
            is_pp = any(key in available_keys for key in required_keys_pp)
            is_lg = any(key in available_keys for key in required_keys_lg)
            
            if not (is_pp or is_lg):
                self.logger.warning(f"Fichier {file_path} ne contient pas les clés de décodage attendues")
                return None
            
            # Convertir en dictionnaire standard Python
            result_dict = {}
            for key in available_keys:
                try:
                    item = data[key]
                    if isinstance(item, np.ndarray) and item.size == 1:
                        # Scalar numpy arrays
                        result_dict[key] = item.item()
                    else:
                        result_dict[key] = item
                except Exception as e:
                    self.logger.warning(f"Erreur lors de la lecture de la clé {key}: {e}")
                    result_dict[key] = None
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de {file_path}: {e}")
            return None
    
    def collect_all_data(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Collecte toutes les données des fichiers NPZ"""
        npz_files = self.find_npz_files()
        
        organized_data = {
            'PP': defaultdict(list),  # Par groupe: controls, del, nodel
            'LG': defaultdict(list)
        }
        
        processed_files = 0
        failed_files = 0
        
        for file_path in npz_files:
            # Éviter les fichiers de groupe déjà générés
            if 'group_' in os.path.basename(file_path) and '_results_complete.npz' in file_path:
                continue
                
            self.logger.info(f"Traitement de: {file_path}")
            
            # Extraire les infos depuis le chemin
            subject_id, group, protocol = self.extract_subject_info_from_path(file_path)
            
            # Charger les données
            data = self.load_and_validate_npz(file_path)
            
            if data is None:
                failed_files += 1
                continue
            
            # Vérifier/corriger les infos extraites avec les données du fichier
            file_subject_id = data.get('subject_id')
            if isinstance(file_subject_id, (np.ndarray, list)):
                file_subject_id = str(file_subject_id[0]) if len(file_subject_id) > 0 else None
            elif file_subject_id is not None:
                file_subject_id = str(file_subject_id)
            
            # Utiliser l'ID du fichier si disponible
            if file_subject_id and not subject_id:
                subject_id = file_subject_id
                # Re-déterminer le groupe
                for group_name, subjects in self.ALL_SUBJECT_GROUPS.items():
                    if subject_id in subjects:
                        group = group_name
                        break
            
            # Déterminer le protocole depuis les données si pas encore fait
            if not protocol:
                if any(key.startswith('pp_ap_') for key in data.keys()):
                    protocol = 'PP'
                elif any(key.startswith('lg_') for key in data.keys()):
                    protocol = 'LG'
            
            if not all([subject_id, group, protocol]):
                self.logger.warning(f"Impossible d'identifier complètement le fichier {file_path}: "
                                  f"subject={subject_id}, group={group}, protocol={protocol}")
                failed_files += 1
                continue
            
            # Ajouter les métadonnées
            data['_file_path'] = file_path
            data['_extracted_subject_id'] = subject_id
            data['_extracted_group'] = group
            data['_extracted_protocol'] = protocol
            
            # Organiser les données
            organized_data[protocol][group].append(data)
            
            # Mémoriser les infos pour les statistiques
            self.protocols_found.add(protocol)
            if group not in self.group_assignments:
                self.group_assignments[group] = []
            if subject_id not in self.group_assignments[group]:
                self.group_assignments[group].append(subject_id)
            
            processed_files += 1
        
        self.logger.info(f"Collecte terminée: {processed_files} fichiers traités, {failed_files} échecs")
        self.logger.info(f"Groupes trouvés: {list(self.group_assignments.keys())}")
        self.logger.info(f"Protocoles trouvés: {list(self.protocols_found)}")
        
        return organized_data
    
    def compute_group_statistics_pp(self, subjects_data: List[Dict]) -> Dict[str, Any]:
        """Calcule les statistiques de groupe pour le protocole PP"""
        group_stats = {
            'n_subjects': len(subjects_data),
            'subject_ids': [],
            'group_mean_auc': np.nan,
            'group_std_auc': np.nan,
            'group_sem_auc': np.nan,
            'time_points': None,
            'group_temporal_mean': None,
            'group_temporal_std': None,
            'group_temporal_sem': None,
            'subject_auc_scores': [],
            'subject_temporal_curves': []
        }
        
        valid_auc_scores = []
        valid_temporal_curves = []
        valid_time_points = []
        
        for subject_data in subjects_data:
            subject_id = subject_data.get('_extracted_subject_id', 'Unknown')
            group_stats['subject_ids'].append(subject_id)
            
            # AUC global
            auc_global = subject_data.get('pp_ap_main_mean_auc_global', np.nan)
            if isinstance(auc_global, np.ndarray):
                auc_global = auc_global.item() if auc_global.size == 1 else np.nanmean(auc_global)
            
            if not np.isnan(auc_global):
                valid_auc_scores.append(auc_global)
                group_stats['subject_auc_scores'].append(auc_global)
            else:
                group_stats['subject_auc_scores'].append(np.nan)
            
            # Courbes temporelles
            temporal_scores = subject_data.get('pp_ap_main_scores_1d_mean')
            time_points = subject_data.get('epochs_time_points')
            
            if temporal_scores is not None and time_points is not None:
                # S'assurer que ce sont des arrays numpy
                if not isinstance(temporal_scores, np.ndarray):
                    temporal_scores = np.array(temporal_scores)
                if not isinstance(time_points, np.ndarray):
                    time_points = np.array(time_points)
                
                # Vérifier la cohérence des dimensions
                if temporal_scores.size == time_points.size and temporal_scores.size > 0:
                    valid_temporal_curves.append(temporal_scores)
                    valid_time_points.append(time_points)
                    group_stats['subject_temporal_curves'].append(temporal_scores)
                else:
                    group_stats['subject_temporal_curves'].append(None)
            else:
                group_stats['subject_temporal_curves'].append(None)
        
        # Calculer les statistiques de groupe
        if valid_auc_scores:
            group_stats['group_mean_auc'] = np.mean(valid_auc_scores)
            group_stats['group_std_auc'] = np.std(valid_auc_scores)
            group_stats['group_sem_auc'] = np.std(valid_auc_scores) / np.sqrt(len(valid_auc_scores))
        
        # Statistiques temporelles
        if valid_temporal_curves and valid_time_points:
            # Vérifier que tous les time_points sont identiques
            time_ref = valid_time_points[0]
            consistent_times = all(np.array_equal(tp, time_ref) for tp in valid_time_points)
            
            if consistent_times:
                temporal_array = np.array(valid_temporal_curves)
                group_stats['time_points'] = time_ref
                group_stats['group_temporal_mean'] = np.nanmean(temporal_array, axis=0)
                group_stats['group_temporal_std'] = np.nanstd(temporal_array, axis=0)
                group_stats['group_temporal_sem'] = np.nanstd(temporal_array, axis=0) / np.sqrt(temporal_array.shape[0])
            else:
                self.logger.warning("Incohérence dans les points temporels entre sujets")
        
        return group_stats
    
    def compute_group_statistics_lg(self, subjects_data: List[Dict]) -> Dict[str, Any]:
        """Calcule les statistiques de groupe pour le protocole LG"""
        group_stats = {
            'n_subjects': len(subjects_data),
            'subject_ids': [],
            'group_mean_auc': np.nan,
            'group_std_auc': np.nan,
            'group_sem_auc': np.nan,
            'time_points': None,
            'group_temporal_mean': None,
            'group_temporal_std': None,
            'group_temporal_sem': None,
            'subject_auc_scores': [],
            'subject_temporal_curves': []
        }
        
        valid_auc_scores = []
        valid_temporal_curves = []
        valid_time_points = []
        
        for subject_data in subjects_data:
            subject_id = subject_data.get('_extracted_subject_id', 'Unknown')
            group_stats['subject_ids'].append(subject_id)
            
            # AUC global
            auc_global = subject_data.get('lg_main_mean_auc_global', np.nan)
            if isinstance(auc_global, np.ndarray):
                auc_global = auc_global.item() if auc_global.size == 1 else np.nanmean(auc_global)
            
            if not np.isnan(auc_global):
                valid_auc_scores.append(auc_global)
                group_stats['subject_auc_scores'].append(auc_global)
            else:
                group_stats['subject_auc_scores'].append(np.nan)
            
            # Courbes temporelles
            temporal_scores = subject_data.get('lg_main_scores_1d_mean')
            time_points = subject_data.get('epochs_time_points')
            
            if temporal_scores is not None and time_points is not None:
                if not isinstance(temporal_scores, np.ndarray):
                    temporal_scores = np.array(temporal_scores)
                if not isinstance(time_points, np.ndarray):
                    time_points = np.array(time_points)
                
                if temporal_scores.size == time_points.size and temporal_scores.size > 0:
                    valid_temporal_curves.append(temporal_scores)
                    valid_time_points.append(time_points)
                    group_stats['subject_temporal_curves'].append(temporal_scores)
                else:
                    group_stats['subject_temporal_curves'].append(None)
            else:
                group_stats['subject_temporal_curves'].append(None)
        
        # Calculer les statistiques de groupe
        if valid_auc_scores:
            group_stats['group_mean_auc'] = np.mean(valid_auc_scores)
            group_stats['group_std_auc'] = np.std(valid_auc_scores)
            group_stats['group_sem_auc'] = np.std(valid_auc_scores) / np.sqrt(len(valid_auc_scores))
        
        # Statistiques temporelles
        if valid_temporal_curves and valid_time_points:
            time_ref = valid_time_points[0]
            consistent_times = all(np.array_equal(tp, time_ref) for tp in valid_time_points)
            
            if consistent_times:
                temporal_array = np.array(valid_temporal_curves)
                group_stats['time_points'] = time_ref
                group_stats['group_temporal_mean'] = np.nanmean(temporal_array, axis=0)
                group_stats['group_temporal_std'] = np.nanstd(temporal_array, axis=0)
                group_stats['group_temporal_sem'] = np.nanstd(temporal_array, axis=0) / np.sqrt(temporal_array.shape[0])
            else:
                self.logger.warning("Incohérence dans les points temporels entre sujets LG")
        
        return group_stats
    
    def create_group_visualization(self, group_stats: Dict[str, Any], group_name: str, 
                                 protocol: str, output_dir: str):
        """Crée les visualisations pour un groupe"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Résultats de Groupe: {group_name.upper()} - Protocole {protocol}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Distribution des scores AUC globaux
        ax1 = axes[0, 0]
        valid_aucs = [auc for auc in group_stats['subject_auc_scores'] if not np.isnan(auc)]
        
        if valid_aucs:
            ax1.hist(valid_aucs, bins=min(10, len(valid_aucs)), alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(group_stats['group_mean_auc'], color='red', linestyle='--', 
                       label=f'Moyenne: {group_stats["group_mean_auc"]:.3f}')
            ax1.axvline(0.5, color='gray', linestyle=':', label='Chance (0.5)')
            ax1.set_xlabel('Score AUC Global')
            ax1.set_ylabel('Nombre de sujets')
            ax1.set_title('Distribution des Performances Globales')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Barplot des AUC par sujet
        ax2 = axes[0, 1]
        subject_ids = group_stats['subject_ids']
        auc_scores = group_stats['subject_auc_scores']
        
        # Trier par performance
        sorted_data = sorted(zip(subject_ids, auc_scores), key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        sorted_subjects, sorted_aucs = zip(*sorted_data) if sorted_data else ([], [])
        
        colors = ['green' if not np.isnan(auc) and auc > 0.5 else 'red' for auc in sorted_aucs]
        bars = ax2.bar(range(len(sorted_subjects)), sorted_aucs, color=colors, alpha=0.7)
        
        ax2.axhline(0.5, color='gray', linestyle=':', label='Chance')
        ax2.axhline(group_stats['group_mean_auc'], color='blue', linestyle='--', 
                   label=f'Moyenne Groupe: {group_stats["group_mean_auc"]:.3f}')
        ax2.set_xlabel('Sujets (triés par performance)')
        ax2.set_ylabel('Score AUC Global')
        ax2.set_title('Performance Individuelle par Sujet')
        ax2.set_xticks(range(len(sorted_subjects)))
        ax2.set_xticklabels(sorted_subjects, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Courbes temporelles moyennes
        ax3 = axes[1, 0]
        if group_stats['time_points'] is not None and group_stats['group_temporal_mean'] is not None:
            time_points = group_stats['time_points']
            mean_curve = group_stats['group_temporal_mean']
            sem_curve = group_stats['group_temporal_sem']
            
            ax3.plot(time_points, mean_curve, 'b-', linewidth=2, label='Moyenne Groupe')
            ax3.fill_between(time_points, mean_curve - sem_curve, mean_curve + sem_curve, 
                           alpha=0.3, color='blue', label='SEM')
            ax3.axhline(0.5, color='gray', linestyle=':', label='Chance')
            ax3.axvline(0, color='black', linestyle='--', alpha=0.5, label='Début Stimulus')
            
            ax3.set_xlabel('Temps (s)')
            ax3.set_ylabel('Score AUC')
            ax3.set_title('Décodage Temporel Moyen')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Trouver et annoter le pic
            peak_idx = np.nanargmax(mean_curve)
            peak_time = time_points[peak_idx]
            peak_auc = mean_curve[peak_idx]
            ax3.annotate(f'Pic: {peak_auc:.3f}\n@ {peak_time:.3f}s', 
                        xy=(peak_time, peak_auc), xytext=(peak_time + 0.1, peak_auc + 0.05),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 4. Courbes individuelles (sous-échantillon)
        ax4 = axes[1, 1]
        if group_stats['time_points'] is not None:
            time_points = group_stats['time_points']
            individual_curves = [curve for curve in group_stats['subject_temporal_curves'] 
                               if curve is not None]
            
            # Afficher max 10 courbes pour la lisibilité
            n_curves_to_show = min(10, len(individual_curves))
            if individual_curves:
                for i, curve in enumerate(individual_curves[:n_curves_to_show]):
                    ax4.plot(time_points, curve, alpha=0.5, linewidth=1, 
                           label=group_stats['subject_ids'][i] if i < len(group_stats['subject_ids']) else f'Sujet {i+1}')
                
                # Ajouter la moyenne en gras
                if group_stats['group_temporal_mean'] is not None:
                    ax4.plot(time_points, group_stats['group_temporal_mean'], 
                           'k-', linewidth=3, label='Moyenne Groupe')
                
                ax4.axhline(0.5, color='gray', linestyle=':', label='Chance')
                ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
                ax4.set_xlabel('Temps (s)')
                ax4.set_ylabel('Score AUC')
                ax4.set_title(f'Courbes Individuelles ({n_curves_to_show} premiers sujets)')
                if n_curves_to_show <= 5:  # Légende seulement si peu de courbes
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        output_file = os.path.join(output_dir, f'group_{group_name}_{protocol}_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualisation sauvegardée: {output_file}")
    
    def save_group_results(self, all_group_stats: Dict, output_dir: str):
        """Sauvegarde les résultats de groupe en formats NPZ et CSV"""
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        summary_data = []
        
        for protocol in all_group_stats:
            for group_name, stats in all_group_stats[protocol].items():
                
                # Sauvegarder en NPZ
                npz_filename = os.path.join(output_dir, f'regenerated_group_{group_name}_{protocol}_results.npz')
                
                save_dict = {
                    'group_name': group_name,
                    'protocol': protocol,
                    'n_subjects': stats['n_subjects'],
                    'subject_ids': stats['subject_ids'],
                    'group_mean_auc': stats['group_mean_auc'],
                    'group_std_auc': stats['group_std_auc'],
                    'group_sem_auc': stats['group_sem_auc'],
                    'subject_auc_scores': np.array(stats['subject_auc_scores']),
                    'regeneration_timestamp': datetime.now().isoformat()
                }
                
                # Ajouter les données temporelles si disponibles
                if stats['time_points'] is not None:
                    save_dict.update({
                        'time_points': stats['time_points'],
                        'group_temporal_mean': stats['group_temporal_mean'],
                        'group_temporal_std': stats['group_temporal_std'],
                        'group_temporal_sem': stats['group_temporal_sem'],
                        'subject_temporal_curves': np.array([curve if curve is not None else np.full(len(stats['time_points']), np.nan) 
                                                            for curve in stats['subject_temporal_curves']])
                    })
                
                np.savez_compressed(npz_filename, **save_dict)
                self.logger.info(f"Résultats NPZ sauvegardés: {npz_filename}")
                
                # Données pour le résumé CSV
                summary_data.append({
                    'Protocol': protocol,
                    'Group': group_name,
                    'N_Subjects': stats['n_subjects'],
                    'Mean_AUC': stats['group_mean_auc'],
                    'Std_AUC': stats['group_std_auc'],
                    'SEM_AUC': stats['group_sem_auc'],
                    'Min_AUC': np.nanmin(stats['subject_auc_scores']) if stats['subject_auc_scores'] else np.nan,
                    'Max_AUC': np.nanmax(stats['subject_auc_scores']) if stats['subject_auc_scores'] else np.nan,
                    'N_Above_Chance': sum(1 for auc in stats['subject_auc_scores'] if not np.isnan(auc) and auc > 0.5),
                    'Peak_Temporal_AUC': np.nanmax(stats['group_temporal_mean']) if stats['group_temporal_mean'] is not None else np.nan,
                    'Peak_Time': stats['time_points'][np.nanargmax(stats['group_temporal_mean'])] if stats['group_temporal_mean'] is not None and stats['time_points'] is not None else np.nan
                })
        
        # Sauvegarder le résumé CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_filename = os.path.join(output_dir, 'group_analysis_summary.csv')
            summary_df.to_csv(csv_filename, index=False)
            self.logger.info(f"Résumé CSV sauvegardé: {csv_filename}")
            
            # Afficher le résumé
            print("\n" + "="*80)
            print("RÉSUMÉ DES RÉSULTATS DE GROUPE RÉGÉNÉRÉS")
            print("="*80)
            print(summary_df.to_string(index=False, float_format='%.3f'))
            print("="*80)

def main():
    """Fonction principale"""
    logger = setup_logging()
    logger.info("Démarrage de la collection et régénération des résultats de groupe")
    
    try:
        # Configuration des chemins
        base_results_path, current_user = get_user_paths()
        
        if not base_results_path or not os.path.exists(base_results_path):
            logger.error(f"Chemin des résultats introuvable: {base_results_path}")
            logger.info("Chemins recherchés:")
            logger.info("  - /Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_results_V7")
            logger.info("  - ./results")
            logger.info("Veuillez vérifier que les résultats d'analyses individuelles existent.")
            return
        
        logger.info(f"Utilisateur: {current_user}")
        logger.info(f"Chemin des résultats: {base_results_path}")
        
        # Initialiser le collecteur
        collector = NPZDataCollector(base_results_path, logger)
        
        # Collecter toutes les données
        logger.info("Phase 1: Collection des fichiers NPZ...")
        organized_data = collector.collect_all_data()
        
        # Vérifier qu'on a des données
        total_subjects = sum(len(subjects) for protocol_data in organized_data.values() 
                           for subjects in protocol_data.values())
        
        if total_subjects == 0:
            logger.error("Aucune donnée de sujet trouvée!")
            logger.info("Vérifiez que les analyses individuelles ont bien été exécutées et sauvegardées.")
            return
        
        logger.info(f"Trouvé {total_subjects} analyses de sujets au total")
        
        # Calculer les statistiques de groupe
        logger.info("Phase 2: Calcul des statistiques de groupe...")
        all_group_stats = {}
        
        for protocol in organized_data:
            all_group_stats[protocol] = {}
            logger.info(f"Traitement du protocole {protocol}...")
            
            for group_name, subjects_data in organized_data[protocol].items():
                if not subjects_data:
                    continue
                    
                logger.info(f"  Groupe {group_name}: {len(subjects_data)} sujets")
                
                if protocol == 'PP':
                    stats = collector.compute_group_statistics_pp(subjects_data)
                elif protocol == 'LG':
                    stats = collector.compute_group_statistics_lg(subjects_data)
                else:
                    logger.warning(f"Protocole inconnu: {protocol}")
                    continue
                
                all_group_stats[protocol][group_name] = stats
                logger.info(f"    AUC moyen: {stats['group_mean_auc']:.3f} ± {stats['group_sem_auc']:.3f}")
        
        # Créer le répertoire de sortie
        output_dir = os.path.join(base_results_path, "regenerated_group_analysis", 
                                 datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Générer les visualisations
        logger.info("Phase 3: Génération des visualisations...")
        for protocol in all_group_stats:
            for group_name, stats in all_group_stats[protocol].items():
                collector.create_group_visualization(stats, group_name, protocol, output_dir)
        
        # Sauvegarder les résultats
        logger.info("Phase 4: Sauvegarde des résultats...")
        collector.save_group_results(all_group_stats, output_dir)
        
        logger.info(f"Analyse terminée avec succès! Résultats dans: {output_dir}")
        
        # Afficher un résumé final
        print(f"\n🎉 RÉGÉNÉRATION TERMINÉE AVEC SUCCÈS!")
        print(f"📁 Résultats sauvegardés dans: {output_dir}")
        print(f"📊 Protocoles traités: {list(all_group_stats.keys())}")
        for protocol in all_group_stats:
            print(f"   {protocol}: {list(all_group_stats[protocol].keys())}")
        
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        logger.error(traceback.format_exc())
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
