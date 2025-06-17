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
            logging.FileHandler(
                f'collect_group_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
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
            "controls": [
                "AO05",
                "BT13",
                "FP102",
                "GA_FRA",
                "HM10",
                "JM14",
                "LAB1",
                "LAG6",
                "LAT3",
                "LBM4",
                "LCM2",
                "LPO5",
                "LS07",
                "LT12",
                "MA_CHA",
                "PB20",
                "SB09",
                "SP03",
                "TAK7",
                "TCG5",
                "TEN1",
                "TFB6",
                "TGD8",
                "TJL3",
                "TJR7",
                "TLB3",
                "TLP8",
                "TNC11",
                "TPC2",
                "TPLV4",
                "TSS4",
                "TTDV5",
                "TTV2",
                "TVR9",
                "TVM10",
                "TWB1",
                "TYS6",
                "TZ11",
                "VB01",
            ],
            "DELIRIUM +": [
                "TpAB19",
                "TpAK24",
                "TpAK27",
                "TpBL47",
                "TpCB15",
                "TpCF1",
                "TpDRL3",
                "TpFF34",
                "TpFY57",
                "TpJA20",
                "TpJB25",
                "TpJB26",
                "TpJC5",
                "TpJCD29",
                "TpJLR17",
                "TpJPS55",
                "TpLA28",
                "TpMB45",
                "TpMM4",
                "TpMN42",
                "TpPC21",
                "TpPM14",
                "TpPM31",
                "TpRD38",
                "TpSM49",
            ],
            "DELIRIUM -": [
                "TpAC23",
                "TpAG51",
                "TpAM43",
                "TpBD16",
                "TpDD2",
                "TpFB18",
                "TpFL53",
                "TpGB8",
                "TpGT32",
                "TpJPG7",
                "TpJPL10",
                "TpKS6",
                "TpLP11",
                "TpMA9",
                "TpMD13",
                "TpMD52",
                "TpME22",
                "TpPA35",
                "TpPI46",
                "TpPL48",
                "TpRB50",
                "TpRK39",
                "TpSD30",
                "TpYB41",
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

        self.logger.info(
            f"Trouvé {len(found_files)} fichiers NPZ dans {self.base_results_path}")
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
                self.logger.warning(
                    f"Fichier {file_path} ne contient pas les clés de décodage attendues")
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
                    self.logger.warning(
                        f"Erreur lors de la lecture de la clé {key}: {e}")
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
            subject_id, group, protocol = self.extract_subject_info_from_path(
                file_path)

            # Charger les données
            data = self.load_and_validate_npz(file_path)

            if data is None:
                failed_files += 1
                continue

            # Vérifier/corriger les infos extraites avec les données du fichier
            file_subject_id = data.get('subject_id')
            if isinstance(file_subject_id, (np.ndarray, list)):
                file_subject_id = str(file_subject_id[0]) if len(
                    file_subject_id) > 0 else None
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

        self.logger.info(
            f"Collecte terminée: {processed_files} fichiers traités, {failed_files} échecs")
        self.logger.info(
            f"Groupes trouvés: {list(self.group_assignments.keys())}")
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
                    group_stats['subject_temporal_curves'].append(
                        temporal_scores)
                else:
                    group_stats['subject_temporal_curves'].append(None)
            else:
                group_stats['subject_temporal_curves'].append(None)

        # Calculer les statistiques de groupe
        if valid_auc_scores:
            group_stats['group_mean_auc'] = np.mean(valid_auc_scores)
            group_stats['group_std_auc'] = np.std(valid_auc_scores)
            group_stats['group_sem_auc'] = np.std(
                valid_auc_scores) / np.sqrt(len(valid_auc_scores))

        # Statistiques temporelles
        if valid_temporal_curves and valid_time_points:
            # Vérifier que tous les time_points sont identiques
            time_ref = valid_time_points[0]
            consistent_times = all(np.array_equal(tp, time_ref)
                                   for tp in valid_time_points)

            if consistent_times:
                temporal_array = np.array(valid_temporal_curves)
                group_stats['time_points'] = time_ref
                group_stats['group_temporal_mean'] = np.nanmean(
                    temporal_array, axis=0)
                group_stats['group_temporal_std'] = np.nanstd(
                    temporal_array, axis=0)
                group_stats['group_temporal_sem'] = np.nanstd(
                    temporal_array, axis=0) / np.sqrt(temporal_array.shape[0])
            else:
                self.logger.warning(
                    "Incohérence dans les points temporels entre sujets")

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
                    group_stats['subject_temporal_curves'].append(
                        temporal_scores)
                else:
                    group_stats['subject_temporal_curves'].append(None)
            else:
                group_stats['subject_temporal_curves'].append(None)

        # Calculer les statistiques de groupe
        if valid_auc_scores:
            group_stats['group_mean_auc'] = np.mean(valid_auc_scores)
            group_stats['group_std_auc'] = np.std(valid_auc_scores)
            group_stats['group_sem_auc'] = np.std(
                valid_auc_scores) / np.sqrt(len(valid_auc_scores))

        # Statistiques temporelles
        if valid_temporal_curves and valid_time_points:
            time_ref = valid_time_points[0]
            consistent_times = all(np.array_equal(tp, time_ref)
                                   for tp in valid_time_points)

            if consistent_times:
                temporal_array = np.array(valid_temporal_curves)
                group_stats['time_points'] = time_ref
                group_stats['group_temporal_mean'] = np.nanmean(
                    temporal_array, axis=0)
                group_stats['group_temporal_std'] = np.nanstd(
                    temporal_array, axis=0)
                group_stats['group_temporal_sem'] = np.nanstd(
                    temporal_array, axis=0) / np.sqrt(temporal_array.shape[0])
            else:
                self.logger.warning(
                    "Incohérence dans les points temporels entre sujets LG")

        return group_stats

    def perform_statistical_tests(self, group_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue les tests statistiques sur les données de groupe"""
        stats_results = {}

        # Test des AUC globaux contre le hasard
        valid_aucs = [
            auc for auc in group_stats['subject_auc_scores'] if not np.isnan(auc)]
        if len(valid_aucs) >= 3:
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(valid_aucs, 0.5)
            stats_results['global_auc_ttest'] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'effect_size': (np.mean(valid_aucs) - 0.5) / np.std(valid_aucs),
                'n_subjects': len(valid_aucs)
            }

        # Tests temporels point par point
        if (group_stats['time_points'] is not None and
                len(group_stats['subject_temporal_curves']) >= 3):
            temporal_array = np.array([curve for curve in group_stats['subject_temporal_curves']
                                       if curve is not None])
            if temporal_array.size > 0:
                # Test t contre le hasard à chaque point temporel
                t_stats = []
                p_vals = []
                for t_idx in range(temporal_array.shape[1]):
                    t_data = temporal_array[:, t_idx]
                    t_data = t_data[~np.isnan(t_data)]
                    if len(t_data) >= 3:
                        t_stat, p_val = stats.ttest_1samp(t_data, 0.5)
                        t_stats.append(t_stat)
                        p_vals.append(p_val)
                    else:
                        t_stats.append(np.nan)
                        p_vals.append(np.nan)

                stats_results['temporal_tests'] = {
                    't_statistics': np.array(t_stats),
                    'p_values': np.array(p_vals),
                    'time_points': group_stats['time_points']
                }

                # Correction FDR
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    valid_p = np.array(p_vals)[~np.isnan(p_vals)]
                    if len(valid_p) > 0:
                        fdr_reject, fdr_pvals = fdrcorrection(
                            valid_p, alpha=0.05)
                        stats_results['temporal_tests']['fdr_corrected'] = {
                            'reject': fdr_reject,
                            'p_values_corrected': fdr_pvals
                        }
                except ImportError:
                    self.logger.warning(
                        "statsmodels non disponible pour correction FDR")

        return stats_results

    def find_peak_latencies(self, group_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Trouve les latences des pics de décodage pour chaque sujet"""
        latency_results = {}

        if (group_stats['time_points'] is not None and
                len(group_stats['subject_temporal_curves']) > 0):

            time_points = group_stats['time_points']
            latencies = []
            peak_values = []

            for i, curve in enumerate(group_stats['subject_temporal_curves']):
                if curve is not None:
                    # Chercher dans une fenêtre temporelle d'intérêt (0-1s par exemple)
                    valid_indices = (time_points >= 0.0) & (time_points <= 1.0)
                    if np.any(valid_indices):
                        window_curve = curve[valid_indices]
                        window_times = time_points[valid_indices]

                        if len(window_curve) > 0 and not np.all(np.isnan(window_curve)):
                            peak_idx = np.nanargmax(window_curve)
                            peak_latency = window_times[peak_idx]
                            peak_value = window_curve[peak_idx]

                            # Seulement si au-dessus du hasard
                            if peak_value > 0.5:
                                latencies.append(peak_latency)
                                peak_values.append(peak_value)

            latency_results = {
                'peak_latencies': np.array(latencies),
                'peak_values': np.array(peak_values),
                'n_subjects_with_peaks': len(latencies)
            }

            if len(latencies) > 0:
                latency_results.update({
                    'mean_latency': np.mean(latencies),
                    'std_latency': np.std(latencies),
                    'median_latency': np.median(latencies)
                })

        return latency_results

    def create_publication_figures(self, group_stats: Dict[str, Any], group_name: str,
                                   protocol: str, output_dir: str, stats_results: Dict[str, Any],
                                   latency_results: Dict[str, Any]):
        """Crée des figures de qualité publication"""

        # Utiliser un style de publication
        plt.style.use('seaborn-v0_8-whitegrid')

        # === FIGURE 1: Courbes temporelles avec tests statistiques ===
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        if group_stats['time_points'] is not None and group_stats['group_temporal_mean'] is not None:
            time_points = group_stats['time_points']
            mean_curve = group_stats['group_temporal_mean']
            sem_curve = group_stats['group_temporal_sem']

            # Toutes les courbes individuelles
            individual_curves = [curve for curve in group_stats['subject_temporal_curves']
                                 if curve is not None]

            # Panel A: Toutes les courbes individuelles + moyenne
            for i, curve in enumerate(individual_curves):
                ax1.plot(time_points, curve, alpha=0.3,
                         linewidth=0.8, color='gray')

            ax1.plot(time_points, mean_curve, 'b-', linewidth=3,
                     label=f'Moyenne Groupe (n={len(individual_curves)})')
            ax1.fill_between(time_points, mean_curve - sem_curve, mean_curve + sem_curve,
                             alpha=0.3, color='blue', label='SEM')
            ax1.axhline(0.5, color='black', linestyle='--',
                        linewidth=1, label='Hasard')
            ax1.axvline(0, color='red', linestyle=':',
                        linewidth=1, label='Onset Stimulus')

            # Annotation du pic
            peak_idx = np.nanargmax(mean_curve)
            peak_time = time_points[peak_idx]
            peak_auc = mean_curve[peak_idx]
            ax1.plot(peak_time, peak_auc, 'ro', markersize=8, zorder=10)
            ax1.annotate(f'Pic: {peak_auc:.3f}\n@ {peak_time*1000:.0f}ms',
                         xy=(peak_time, peak_auc), xytext=(
                             peak_time + 0.2, peak_auc + 0.05),
                         arrowprops=dict(arrowstyle='->', color='red', lw=2),
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="yellow", alpha=0.8),
                         fontsize=12, fontweight='bold')

            ax1.set_xlabel('Temps relatif au stimulus (s)', fontsize=14)
            ax1.set_ylabel('Performance de Classification (AUC)', fontsize=14)
            ax1.set_title(
                f'{group_name.upper()} - Protocole {protocol}', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0.4, 0.8)

            # Panel B: Tests statistiques
            if 'temporal_tests' in stats_results:
                t_stats = stats_results['temporal_tests']['t_statistics']
                p_vals = stats_results['temporal_tests']['p_values']

                # Masque de significativité
                sig_mask = p_vals < 0.05
                ax2.plot(time_points, t_stats, 'k-',
                         linewidth=2, label='Statistique t')
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)

                # Zones significatives
                if np.any(sig_mask):
                    for i in range(len(sig_mask)):
                        if sig_mask[i]:
                            ax2.axvspan(time_points[i] - 0.01, time_points[i] + 0.01,
                                        alpha=0.3, color='red')

                ax2.set_xlabel('Temps relatif au stimulus (s)', fontsize=14)
                ax2.set_ylabel('Statistique t vs hasard', fontsize=14)
                ax2.set_title(
                    'Tests Statistiques Point par Point (p < 0.05)', fontsize=14)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file1 = os.path.join(
            output_dir, f'publication_{group_name}_{protocol}_temporal.png')
        plt.savefig(output_file1, dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        # === FIGURE 2: Distributions et latences ===
        fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(16, 12))

        # Panel A: Distribution des AUC globaux avec statistiques
        valid_aucs = [
            auc for auc in group_stats['subject_auc_scores'] if not np.isnan(auc)]
        if valid_aucs:
            ax3.hist(valid_aucs, bins=max(5, min(15, len(valid_aucs)//2)),
                     alpha=0.7, color='lightblue', edgecolor='black', density=True)
            ax3.axvline(np.mean(valid_aucs), color='red', linestyle='--', linewidth=2,
                        label=f'Moyenne: {np.mean(valid_aucs):.3f}')
            ax3.axvline(0.5, color='black', linestyle=':',
                        linewidth=2, label='Hasard')

            # Test statistique
            if 'global_auc_ttest' in stats_results:
                stats_info = stats_results['global_auc_ttest']
                ax3.text(0.05, 0.95, f't = {stats_info["t_statistic"]:.3f}\np = {stats_info["p_value"]:.6f}\nd = {stats_info["effect_size"]:.3f}',
                         transform=ax3.transAxes, verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.8),
                         fontsize=11)

            ax3.set_xlabel('Score AUC Global', fontsize=14)
            ax3.set_ylabel('Densité', fontsize=14)
            ax3.set_title('Distribution des Performances Globales',
                          fontsize=14, fontweight='bold')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)

        # Panel B: Performance individuelle par sujet
        subject_ids = group_stats['subject_ids']
        auc_scores = group_stats['subject_auc_scores']
        sorted_data = sorted(zip(subject_ids, auc_scores),
                             key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        sorted_subjects, sorted_aucs = zip(
            *sorted_data) if sorted_data else ([], [])

        colors = ['darkgreen' if not np.isnan(auc) and auc > 0.5 else 'darkred'
                  for auc in sorted_aucs]
        bars = ax4.bar(range(len(sorted_subjects)),
                       sorted_aucs, color=colors, alpha=0.7)
        ax4.axhline(0.5, color='black', linestyle=':',
                    linewidth=2, label='Hasard')
        ax4.axhline(np.mean(valid_aucs), color='red', linestyle='--', linewidth=2,
                    label=f'Moyenne: {np.mean(valid_aucs):.3f}')

        # Ajouter les valeurs sur les barres
        for i, (bar, auc) in enumerate(zip(bars, sorted_aucs)):
            if not np.isnan(auc):
                ax4.text(bar.get_x() + bar.get_width()/2., auc + 0.01,
                         f'{auc:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

        ax4.set_xlabel('Sujets (triés par performance)', fontsize=14)
        ax4.set_ylabel('Score AUC Global', fontsize=14)
        ax4.set_title('Performance Individuelle',
                      fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(sorted_subjects)))
        ax4.set_xticklabels(sorted_subjects, rotation=45,
                            ha='right', fontsize=10)
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)

        # Panel C: Histogramme des latences des pics
        if latency_results and 'peak_latencies' in latency_results:
            # Convertir en ms
            latencies = latency_results['peak_latencies'] * 1000
            if len(latencies) > 0:
                ax5.hist(latencies, bins=max(5, len(latencies)//3), alpha=0.7,
                         color='orange', edgecolor='black')
                ax5.axvline(np.mean(latencies), color='red', linestyle='--', linewidth=2,
                            label=f'Moyenne: {np.mean(latencies):.1f}ms')
                ax5.axvline(np.median(latencies), color='blue', linestyle='-', linewidth=2,
                            label=f'Médiane: {np.median(latencies):.1f}ms')

                ax5.set_xlabel('Latence du Pic (ms)', fontsize=14)
                ax5.set_ylabel('Nombre de Sujets', fontsize=14)
                ax5.set_title(f'Distribution des Latences des Pics (n={len(latencies)})',
                              fontsize=14, fontweight='bold')
                ax5.legend(fontsize=12)
                ax5.grid(True, alpha=0.3)

        # Panel D: Relation latence vs performance du pic
        if latency_results and 'peak_latencies' in latency_results:
            latencies = latency_results['peak_latencies'] * 1000
            peak_values = latency_results['peak_values']
            if len(latencies) > 0:
                ax6.scatter(latencies, peak_values, s=60,
                            alpha=0.7, color='purple')

                # Régression linéaire si assez de points
                if len(latencies) >= 3:
                    z = np.polyfit(latencies, peak_values, 1)
                    p = np.poly1d(z)
                    ax6.plot(latencies, p(latencies),
                             "r--", alpha=0.8, linewidth=2)

                    # Calcul de corrélation
                    from scipy.stats import pearsonr
                    r, p_val = pearsonr(latencies, peak_values)
                    ax6.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3f}',
                             transform=ax6.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                ax6.set_xlabel('Latence du Pic (ms)', fontsize=14)
                ax6.set_ylabel('Valeur AUC du Pic', fontsize=14)
                ax6.set_title('Relation Latence-Performance',
                              fontsize=14, fontweight='bold')
                ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file2 = os.path.join(
            output_dir, f'publication_{group_name}_{protocol}_distributions.png')
        plt.savefig(output_file2, dpi=300,
                    bbox_inches='tight', facecolor='white')
        plt.close()

        self.logger.info(
            f"Figures de publication sauvegardées: {output_file1}, {output_file2}")

    def create_inter_group_comparison_figures(self, all_group_stats: Dict, output_dir: str):
        """Crée des figures comparatives entre groupes pour publication scientifique"""

        # Style de publication
        plt.style.use('seaborn-v0_8-whitegrid')

        # Organiser les données par protocole
        for protocol in all_group_stats:
            if len(all_group_stats[protocol]) < 2:
                continue  # Besoin d'au moins 2 groupes pour comparer

            # === FIGURE DE COMPARAISON PRINCIPALE ===
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(16, 12))
            fig.suptitle(f'Comparaison Inter-Groupes - Protocole {protocol}',
                         fontsize=18, fontweight='bold')

            # Couleurs pour chaque groupe
            colors = ['blue', 'red', 'green', 'orange',
                      'purple'][:len(all_group_stats[protocol])]
            group_names = list(all_group_stats[protocol].keys())

            # Panel A: Courbes temporelles moyennes superposées
            for i, (group_name, stats) in enumerate(all_group_stats[protocol].items()):
                if stats['time_points'] is not None and stats['group_temporal_mean'] is not None:
                    time_points = stats['time_points']
                    mean_curve = stats['group_temporal_mean']
                    sem_curve = stats['group_temporal_sem']

                    ax1.plot(time_points, mean_curve, color=colors[i], linewidth=3,
                             label=f'{group_name} (n={stats["n_subjects"]})')
                    ax1.fill_between(time_points,
                                     mean_curve - sem_curve,
                                     mean_curve + sem_curve,
                                     color=colors[i], alpha=0.2)

            ax1.axhline(0.5, color='black', linestyle='--',
                        linewidth=1, label='Hasard')
            ax1.axvline(0, color='gray', linestyle=':',
                        linewidth=1, label='Onset Stimulus')
            ax1.set_xlabel('Temps relatif au stimulus (s)', fontsize=14)
            ax1.set_ylabel('Performance de Classification (AUC)', fontsize=14)
            ax1.set_title('Courbes Temporelles Moyennes par Groupe',
                          fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0.4, 0.8)

            # Panel B: Boxplot des AUC globaux
            global_data = []
            group_labels = []
            for group_name, stats in all_group_stats[protocol].items():
                valid_aucs = [
                    auc for auc in stats['subject_auc_scores'] if not np.isnan(auc)]
                global_data.extend(valid_aucs)
                group_labels.extend([group_name] * len(valid_aucs))

            if global_data:
                import pandas as pd
                df_comparison = pd.DataFrame(
                    {'AUC': global_data, 'Groupe': group_labels})

                # Boxplot avec violinplot superposé
                box_plot = ax2.boxplot([df_comparison[df_comparison['Groupe'] == group]['AUC']
                                       for group in group_names],
                                       labels=group_names, patch_artist=True)

                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax2.axhline(0.5, color='black', linestyle='--',
                            linewidth=1, label='Hasard')
                ax2.set_ylabel('Score AUC Global', fontsize=14)
                ax2.set_xlabel('Groupes', fontsize=14)
                ax2.set_title('Distribution des Performances Globales',
                              fontsize=16, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                # Test ANOVA si plus de 2 groupes
                if len(group_names) > 2:
                    from scipy.stats import f_oneway
                    group_data = [df_comparison[df_comparison['Groupe'] == group]['AUC'].values
                                  for group in group_names]
                    f_stat, p_val = f_oneway(*group_data)
                    ax2.text(0.02, 0.98, f'ANOVA: F={f_stat:.3f}, p={p_val:.6f}',
                             transform=ax2.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                # Tests post-hoc par paires
                elif len(group_names) == 2:
                    from scipy.stats import ttest_ind
                    group1_data = df_comparison[df_comparison['Groupe']
                                                == group_names[0]]['AUC'].values
                    group2_data = df_comparison[df_comparison['Groupe']
                                                == group_names[1]]['AUC'].values
                    t_stat, p_val = ttest_ind(group1_data, group2_data)
                    ax2.text(0.02, 0.98, f't-test: t={t_stat:.3f}, p={p_val:.6f}',
                             transform=ax2.transAxes, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            # Panel C: Latences des pics par groupe
            latences_by_group = {}
            for group_name, stats in all_group_stats[protocol].items():
                latency_results = self.find_peak_latencies(stats)
                if latency_results and 'peak_latencies' in latency_results:
                    latences_by_group[group_name] = latency_results['peak_latencies'] * 1000

            if latences_by_group:
                for i, (group_name, latences) in enumerate(latences_by_group.items()):
                    if len(latences) > 0:
                        ax3.hist(latences, bins=max(3, len(latences)//2), alpha=0.6,
                                 color=colors[i], label=f'{group_name} (n={len(latences)})',
                                 edgecolor='black')

                ax3.set_xlabel('Latence du Pic (ms)', fontsize=14)
                ax3.set_ylabel('Nombre de Sujets', fontsize=14)
                ax3.set_title('Distribution des Latences des Pics',
                              fontsize=16, fontweight='bold')
                ax3.legend(fontsize=12)
                ax3.grid(True, alpha=0.3)

            # Panel D: Heatmap des comparaisons par paires
            if len(group_names) >= 2:
                # Créer une matrice de p-values entre groupes
                from scipy.stats import ttest_ind
                n_groups = len(group_names)
                p_matrix = np.ones((n_groups, n_groups))

                for i in range(n_groups):
                    for j in range(i+1, n_groups):
                        group1_aucs = [auc for auc in all_group_stats[protocol][group_names[i]]['subject_auc_scores']
                                       if not np.isnan(auc)]
                        group2_aucs = [auc for auc in all_group_stats[protocol][group_names[j]]['subject_auc_scores']
                                       if not np.isnan(auc)]

                        if len(group1_aucs) >= 3 and len(group2_aucs) >= 3:
                            _, p_val = ttest_ind(group1_aucs, group2_aucs)
                            p_matrix[i, j] = p_val
                            p_matrix[j, i] = p_val

                # Affichage de la heatmap
                import seaborn as sns
                mask = np.triu(np.ones_like(p_matrix, dtype=bool))
                sns.heatmap(p_matrix, annot=True, fmt='.4f', mask=mask,
                            xticklabels=group_names, yticklabels=group_names,
                            cmap='RdYlBu_r', center=0.05, ax=ax4,
                            cbar_kws={'label': 'p-value'})
                ax4.set_title('P-values des Comparaisons Inter-Groupes',
                              fontsize=14, fontweight='bold')

            plt.tight_layout()
            output_file = os.path.join(
                output_dir, f'inter_group_comparison_{protocol}.png')
            plt.savefig(output_file, dpi=300,
                        bbox_inches='tight', facecolor='white')
            plt.close()

            self.logger.info(
                f"Figure de comparaison inter-groupes sauvegardée: {output_file}")

    def create_group_visualization(self, group_stats: Dict[str, Any], group_name: str,
                                   protocol: str, output_dir: str):
        """Crée des visualisations améliorées pour un groupe avec analyses statistiques complètes"""

        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(
            f"Génération des visualisations pour {group_name} - {protocol}")

        # 1. Effectuer les tests statistiques
        stats_results = self.perform_statistical_tests(group_stats)

        # 2. Analyser les latences des pics
        latency_results = self.find_peak_latencies(group_stats)

        # 3. Créer les figures de publication
        self.create_publication_figures(group_stats, group_name, protocol,
                                        output_dir, stats_results, latency_results)

        # 4. Sauvegarder les données statistiques
        stats_file = os.path.join(
            output_dir, f'stats_{group_name}_{protocol}.npz')
        np.savez_compressed(stats_file,
                            group_stats=group_stats,
                            stats_results=stats_results,
                            latency_results=latency_results)

        self.logger.info(f"Données statistiques sauvegardées: {stats_file}")

    def generate_group_results(self, organized_data: Dict[str, Dict[str, List[Dict]]],
                               output_base_dir: str):
        """Génère tous les résultats de groupe avec visualisations améliorées"""

        all_group_stats = {'PP': {}, 'LG': {}}

        for protocol in organized_data:
            self.logger.info(f"\n=== TRAITEMENT PROTOCOLE {protocol} ===")

            for group_name, subjects_data in organized_data[protocol].items():
                if not subjects_data:
                    continue

                self.logger.info(
                    f"\nCalcul des statistiques pour {group_name} ({len(subjects_data)} sujets)")

                # Calculer les statistiques selon le protocole
                if protocol == 'PP':
                    group_stats = self.compute_group_statistics_pp(
                        subjects_data)
                elif protocol == 'LG':
                    group_stats = self.compute_group_statistics_lg(
                        subjects_data)
                else:
                    continue

                # Sauvegarder les statistiques
                all_group_stats[protocol][group_name] = group_stats

                # Créer le répertoire de sortie pour ce groupe
                group_output_dir = os.path.join(
                    output_base_dir, f"{protocol}_{group_name}")

                # Créer les visualisations améliorées
                self.create_group_visualization(
                    group_stats, group_name, protocol, group_output_dir)

                # Afficher un résumé
                self.logger.info(f"Résumé {group_name}: "
                                 f"AUC moyen = {group_stats.get('group_mean_auc', 'N/A'):.3f}, "
                                 f"N = {group_stats.get('n_subjects', 0)}")

        # Créer les comparaisons inter-groupes
        comparison_output_dir = os.path.join(
            output_base_dir, "comparisons_inter_groupes")
        os.makedirs(comparison_output_dir, exist_ok=True)
        self.create_inter_group_comparison_figures(
            all_group_stats, comparison_output_dir)

        # Sauvegarder les données complètes
        complete_results_file = os.path.join(
            output_base_dir, "all_group_results_complete.npz")
        np.savez_compressed(complete_results_file,
                            all_group_stats=all_group_stats)

        self.logger.info(
            f"Résultats complets sauvegardés: {complete_results_file}")
        return all_group_stats


def main():
    """Fonction principale d'exécution"""
    logger = setup_logging()
    logger.info("=== DÉMARRAGE DU SCRIPT DE COLLECTE ET RÉGÉNÉRATION ===")

    # Déterminer les chemins
    base_results_path, current_user = get_user_paths()

    if not base_results_path:
        logger.error("Aucun répertoire de résultats trouvé!")
        return

    logger.info(f"Utilisateur: {current_user}")
    logger.info(f"Répertoire de base: {base_results_path}")

    # Créer le collecteur
    collector = NPZDataCollector(base_results_path, logger)

    try:
        # Collecter toutes les données
        logger.info("Collecte des données NPZ...")
        organized_data = collector.collect_all_data()

        if not organized_data or not any(organized_data.values()):
            logger.warning("Aucune donnée collectée!")
            return

        # Afficher un résumé de collecte
        for protocol in organized_data:
            for group_name, subjects_list in organized_data[protocol].items():
                logger.info(
                    f"{protocol} - {group_name}: {len(subjects_list)} sujets")

        # Créer le répertoire de sortie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            base_results_path, f"regenerated_group_results_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Répertoire de sortie: {output_dir}")

        # Générer tous les résultats
        logger.info("Génération des résultats de groupe...")
        all_group_stats = collector.generate_group_results(
            organized_data, output_dir)

        # Générer un rapport final
        report_file = os.path.join(output_dir, "rapport_final.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"RAPPORT DE RÉGÉNÉRATION DES RÉSULTATS DE GROUPE\n")
            f.write(
                f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Utilisateur: {current_user}\n")
            f.write(f"Répertoire source: {base_results_path}\n\n")

            f.write("=== RÉSUMÉ DES DONNÉES COLLECTÉES ===\n")
            for protocol in organized_data:
                f.write(f"\nProtocole {protocol}:\n")
                for group_name, subjects_list in organized_data[protocol].items():
                    f.write(f"  - {group_name}: {len(subjects_list)} sujets\n")
                    if group_name in all_group_stats[protocol]:
                        stats = all_group_stats[protocol][group_name]
                        f.write(
                            f"    AUC moyen: {stats.get('group_mean_auc', 'N/A'):.3f}\n")
                        f.write(
                            f"    Sujets: {', '.join(stats.get('subject_ids', []))}\n")

            f.write(f"\n=== FICHIERS GÉNÉRÉS ===\n")
            f.write(f"- Figures de publication par groupe\n")
            f.write(f"- Comparaisons inter-groupes\n")
            f.write(f"- Données statistiques complètes\n")
            f.write(f"- Analyses de latences\n")

        logger.info(f"Rapport final généré: {report_file}")
        logger.info("=== SCRIPT TERMINÉ AVEC SUCCÈS ===")

    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)


if __name__ == "__main__":
    main()
