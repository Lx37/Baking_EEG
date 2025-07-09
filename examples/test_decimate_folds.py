import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import logging
from datetime import datetime
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif
import mne

from scipy import signal
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Baking_EEG._4_decoding_core import run_temporal_decoding_analysis
from utils.loading_PP_utils import load_epochs_data_auto_protocol
from utils.utils import configure_project_paths
from config.config import sfreq as default_sfreq
from utils.stats_utils import perform_pointwise_fdr_correction_on_scores, perform_cluster_permutation_test


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


N_FOLDS_TO_TEST = [2, 5, 7, 9, 13, 16, 20, 24, 28, 32, 36]
SAMPLING_FREQUENCIES = [500, 350, 250, 200, 125, 100]  # Hz
ORIGINAL_FREQ = 500  # Fréquence d'origine assumée


# FEATURE_SELECTION_CONFIGS 

TEST_SUBJECT_FILE = "/mnt/data/tom.balay/data/Baking_EEG_data/PP_PATIENTS_DELIRIUM+_0.5/TpSM49_PP_preproc_noICA_PP-epo_ar.fif"
#/mnt/data/tom.balay/data/Baking_EEG_data/PP_PATIENTS_DELIRIUM+_0.5/TpSM49_PP_preproc_noICA_PP-epo_ar.fif
#/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG_data/ME64_preproc_noICA_PPAP-epo_ar.fif
RANDOM_STATE = 42


def create_empty_result(n_folds, sampling_freq):
    """Crée un résultat vide en cas d'erreur."""
    return {
        'n_folds': n_folds,
        'sampling_freq': sampling_freq,
        'mean_auc': np.nan,
        'std_auc': np.nan,
        'mean_accuracy': np.nan,
        'peak_score': np.nan,
        'peak_time': np.nan,
        'temporal_scores': None,
        'times': None,
        'temporal_scores_all_folds': None,
        'decimation_factor': np.nan,
        'n_trials': 0,
        'n_channels': 0,
        'n_times_original': 0,
        'n_times_decimated': 0,
        'success': False
    }


def decimate_epochs_to_target_freq(epochs, target_freq, original_freq=500):
    """
    Décimate les epochs à la fréquence cible en utilisant MNE resample.
    
    Args:
        epochs (mne.Epochs): Epochs MNE à décimer
        target_freq (float): Fréquence cible en Hz
        original_freq (float): Fréquence d'origine en Hz
        
    Returns:
        mne.Epochs: Epochs décimées
    """
    if target_freq >= original_freq:
        logger.info(f"Fréquence cible {target_freq}Hz >= fréquence originale {original_freq}Hz. Pas de décimation.")
        return epochs
    
    decimation_factor = original_freq / target_freq
    
    if decimation_factor == 1:
        return epochs
    
    logger.info(f"Décimation: {original_freq}Hz -> {target_freq}Hz (facteur: {decimation_factor:.1f})")
    
    epochs_decimated = epochs.copy()
    epochs_decimated = epochs_decimated.resample(sfreq=target_freq, verbose=False)
    
    return epochs_decimated


def run_single_combination_test(epochs, labels, n_folds, sampling_freq, original_freq=500):
    """
    Teste une combinaison spécifique de folds et fréquence d'échantillonnage.
    
    Args:
        epochs (mne.Epochs): Epochs MNE
        labels (np.ndarray): Labels des conditions
        n_folds (int): Nombre de folds pour la validation croisée
        sampling_freq (float): Fréquence d'échantillonnage cible
        original_freq (float): Fréquence d'origine
        
    Returns:
        dict: Résultats du test
    """
    logger.info(f"Test: {n_folds} folds, {sampling_freq}Hz")
    
    try:
       
        epochs_decimated = decimate_epochs_to_target_freq(epochs, sampling_freq, original_freq)
        
       
        X = epochs_decimated.get_data()  # (n_trials, n_channels, n_times)
        y = labels
        
   
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        #cv_splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=5, random_state=RANDOM_STATE)
        

        results = run_temporal_decoding_analysis(
            epochs_data=X,
            target_labels=y_encoded,
            classifier_model_type="svc",
            cross_validation_splitter=cv_splitter,
            use_grid_search=False,
            use_anova_fs_for_temporal_pipelines=False,
            compute_intra_fold_stats=False,
            compute_temporal_generalization_matrix=False,
            n_jobs_external=-1,
            random_state=RANDOM_STATE
        )
        
  
        # Structure : (probas_global_agg, labels_global_agg, cv_global_scores, 
        #             mean_scores_1d, global_metrics, fdr_1d_data, cluster_1d_data, 
        #             scores_1d_all_folds, mean_tgm, fdr_tgm_data, None, tgm_all_folds)
        if isinstance(results, tuple) and len(results) >= 8:
            probas_global = results[0]
            labels_global = results[1] 
            cv_scores = results[2]
            temporal_scores = results[3]  # mean_scores_1d
            global_metrics = results[4]
            all_fold_scores = results[7]  # scores_1d_all_folds
        else:
            logger.error(f"Format de résultats inattendu pour {n_folds} folds, {sampling_freq}Hz")
            return create_empty_result(n_folds, sampling_freq)
        
      
        if cv_scores is None or len(cv_scores) == 0:
            logger.error(f"Scores CV invalides pour {n_folds} folds, {sampling_freq}Hz")
            return create_empty_result(n_folds, sampling_freq)
            
        if temporal_scores is None or len(temporal_scores) == 0:
            logger.error(f"Scores temporels invalides pour {n_folds} folds, {sampling_freq}Hz")
            return create_empty_result(n_folds, sampling_freq)
        
      
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        mean_accuracy = global_metrics.get('accuracy', np.nan) if global_metrics else np.nan
        
      
        n_trials, n_channels, n_times_decimated = X.shape
        n_times_original = len(epochs.times) if epochs is not None else n_times_decimated
        decimation_factor = original_freq / sampling_freq
        
     
        times = epochs_decimated.times
        peak_time_idx = np.argmax(temporal_scores)
        peak_time = times[peak_time_idx]
        peak_score = temporal_scores[peak_time_idx]
        
        return {
            'n_folds': n_folds,
            'sampling_freq': sampling_freq,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'mean_accuracy': mean_accuracy,
            'peak_score': peak_score,
            'peak_time': peak_time,
            'temporal_scores': temporal_scores,
            'times': times,
            'temporal_scores_all_folds': all_fold_scores, 
            'decimation_factor': decimation_factor,
            'n_trials': n_trials,
            'n_channels': n_channels,
            'n_times_original': n_times_original,
            'n_times_decimated': n_times_decimated,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Erreur pour {n_folds} folds, {sampling_freq}Hz: {str(e)}")
        return create_empty_result(n_folds, sampling_freq)


def load_test_data():
    """
    Charge les données de test pour un sujet exemple.
    
    Returns:
        tuple: (epochs, labels, subject_info)
    """
    logger.info(f"Chargement des données depuis {TEST_SUBJECT_FILE}")
    
    try:
     
        if not os.path.exists(TEST_SUBJECT_FILE):
            logger.error(f"Fichier non trouvé: {TEST_SUBJECT_FILE}")
            return create_simulated_data()
        
     
        epochs = mne.read_epochs(TEST_SUBJECT_FILE, preload=True, verbose=False)
        
       
      
        event_names = list(epochs.event_id.keys())
        pp_events = [k for k in event_names if 'PP' in k.upper()]
        ap_events = [k for k in event_names if 'AP' in k.upper()]
        
        logger.info(f"Événements trouvés: {len(event_names)} types")
        logger.info(f"Événements PP: {pp_events[:5]}...")  # Afficher les 5 premiers
        logger.info(f"Événements AP: {ap_events[:5]}...")  # Afficher les 5 premiers
        
        # Créer les labels binaires : PP = 1, AP = 0
        labels = []
        for event in epochs.events[:, 2]:

            event_name = None
            for name, code in epochs.event_id.items():
                if code == event:
                    event_name = name
                    break
            
          
            if event_name and 'PP' in event_name.upper():
                labels.append('PP')
            elif event_name and 'AP' in event_name.upper():
                labels.append('AP')
            else:
            
                labels.append('AP')
        
        labels = np.array(labels)
        
      
        pp_ap_mask = (labels == 'PP') | (labels == 'AP')
        if np.sum(pp_ap_mask) == 0:
            logger.error("Aucun événement PP/AP trouvé")
            return create_simulated_data()
        
        epochs_filtered = epochs[pp_ap_mask]
        labels_filtered = labels[pp_ap_mask]
        
      
        subject_info = {
            'subject_id': 'ME64',
            'n_epochs': len(epochs_filtered),
            'sampling_freq': epochs.info['sfreq'],
            'n_channels': len(epochs.ch_names),
            'event_id': epochs.event_id,
            'pp_count': np.sum(labels_filtered == 'PP'),
            'ap_count': np.sum(labels_filtered == 'AP')
        }
        
        logger.info(f"Données chargées: {len(epochs_filtered)} epochs")
        logger.info(f"PP: {subject_info['pp_count']}, AP: {subject_info['ap_count']}")
        logger.info(f"Fréquence: {epochs.info['sfreq']}Hz, Canaux: {len(epochs.ch_names)}")
        
        return epochs_filtered, labels_filtered, subject_info
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}")
        return create_simulated_data()


def create_simulated_data():
    """
    Crée des données EEG simulées pour les tests.
    
    Returns:
        tuple: (epochs, labels, subject_info)
    """
    logger.info("Création de données EEG simulées")
    

    n_epochs = 200
    n_channels = 64
    sfreq = 500
    tmin, tmax = -0.2, 0.8
    n_times = int((tmax - tmin) * sfreq)
    

    times = np.linspace(tmin, tmax, n_times)
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-6
    
   
    for i in range(n_epochs):
        condition = i % 2
  
        p300_time = 0.3
        p300_idx = np.argmin(np.abs(times - p300_time))
        
        if condition == 0:
         
            amplitude = 5e-6
        else:
           
            amplitude = 2e-6
        
       
        central_channels = slice(20, 40)
        gaussian_wave = amplitude * np.exp(-0.5 * ((times - p300_time) / 0.05) ** 2)
        data[i, central_channels, :] += gaussian_wave
    
    
    ch_names = [f'EEG_{i+1:03d}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
  
    events = np.zeros((n_epochs, 3), dtype=int)
    events[:, 0] = np.arange(n_epochs) * int(sfreq)  
    events[:, 2] = np.array([1, 2] * (n_epochs // 2))  
    

    event_id = {'condition_1': 1, 'condition_2': 2}
    

    epochs = mne.EpochsArray(
        data, info, events=events, tmin=tmin, event_id=event_id, verbose=False
    )
    

    labels = np.array(['condition_1', 'condition_2'] * (n_epochs // 2))
    

    subject_info = {
        'subject_id': 'SIMULATED_DATA',
        'n_epochs': n_epochs,
        'sampling_freq': sfreq,
        'n_channels': n_channels,
        'event_id': event_id,
        'pp_count': n_epochs // 2,  # condition_1
        'ap_count': n_epochs // 2   # condition_2
    }
    
    logger.info(f"Données simulées créées: {n_epochs} epochs, {n_channels} canaux, {sfreq}Hz")
    
    return epochs, labels, subject_info


def run_comprehensive_analysis():
    """
    Exécute l'analyse complète de toutes les combinaisons.
    
    Returns:
        pd.DataFrame: Résultats de tous les tests
    """

    epochs, labels, subject_info = load_test_data()
    
    total_combinations = len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)
    logger.info(f"Début de l'analyse complète: {len(N_FOLDS_TO_TEST)} folds × {len(SAMPLING_FREQUENCIES)} fréquences = {total_combinations} combinaisons")
    
   
    all_results = []
    
  
    for n_folds, sampling_freq in product(N_FOLDS_TO_TEST, SAMPLING_FREQUENCIES):
        result = run_single_combination_test(
            epochs, labels, n_folds, sampling_freq, subject_info['sampling_freq']
        )
        all_results.append(result)
        
        if result['success']:
            logger.info(f"✓ {n_folds} folds, {sampling_freq}Hz: AUC={result['mean_auc']:.3f}±{result['std_auc']:.3f}")
        else:
            logger.error(f"✗ {n_folds} folds, {sampling_freq}Hz: ÉCHEC")
    
    
    results_df = pd.DataFrame(all_results)
    
   
    results_df['subject_id'] = subject_info['subject_id']
    results_df['original_freq'] = subject_info['sampling_freq']
    results_df['decimation_factor'] = results_df['original_freq'] / results_df['sampling_freq']
    
    logger.info(f"Analyse terminée. {results_df['success'].sum()}/{len(results_df)} tests réussis.")
    
    return results_df, subject_info


def create_visualization_page1(results_df, subject_info):
    """
    Crée la page 1 des visualisations: Heatmaps des performances.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Page 1: Performances de Décodage - Sujet {subject_info["subject_id"]}', fontsize=16, fontweight='bold')
    
    # Préparer les données pour les heatmaps
    successful_results = results_df[results_df['success']].copy()
    
    if len(successful_results) == 0:
        plt.text(0.5, 0.5, 'Aucun résultat valide disponible', ha='center', va='center', transform=fig.transFigure, fontsize=20)
        return fig
    

    auc_pivot = successful_results.pivot(index='n_folds', columns='sampling_freq', values='mean_auc')
    std_auc_pivot = successful_results.pivot(index='n_folds', columns='sampling_freq', values='std_auc')
    accuracy_pivot = successful_results.pivot(index='n_folds', columns='sampling_freq', values='mean_accuracy')
    peak_score_pivot = successful_results.pivot(index='n_folds', columns='sampling_freq', values='peak_score')
    
  
    sns.heatmap(auc_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=axes[0,0], cbar_kws={'label': 'AUC'})
    axes[0,0].set_title('AUC Moyenne par Combinaison')
    axes[0,0].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[0,0].set_ylabel('Nombre de folds')
    
  
    sns.heatmap(std_auc_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=axes[0,1], cbar_kws={'label': 'Std AUC'})
    axes[0,1].set_title('Écart-type AUC par Combinaison')
    axes[0,1].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[0,1].set_ylabel('Nombre de folds')
    
   
    sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r',
                ax=axes[1,0], cbar_kws={'label': 'Accuracy'})
    axes[1,0].set_title('Précision Moyenne par Combinaison')
    axes[1,0].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[1,0].set_ylabel('Nombre de folds')
    
  
    sns.heatmap(peak_score_pivot, annot=True, fmt='.3f', cmap='plasma',
                ax=axes[1,1], cbar_kws={'label': 'Peak Score'})
    axes[1,1].set_title('Score de Pic Temporal par Combinaison')
    axes[1,1].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[1,1].set_ylabel('Nombre de folds')
    
    plt.tight_layout()
    return fig


def create_visualization_page2(results_df, subject_info):
    """
    Crée la page 2 des visualisations: Analyses par fréquence et par folds avec statistiques.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    # Calculer les statistiques temporelles
    stats_dict = compute_temporal_statistics(results_df, chance_level=0.5)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Page 2: Analyses Détaillées avec Statistiques - Sujet {subject_info["subject_id"]}', fontsize=16, fontweight='bold')
    

    successful_results = results_df[results_df['success']].copy()
    
    if len(successful_results) == 0:
        plt.text(0.5, 0.5, 'Aucun résultat valide disponible', ha='center', va='center', transform=fig.transFigure, fontsize=20)
        return fig
    
    # Calculer le résumé des significativités
    significance_summary = calculate_significance_summary(stats_dict)
    
    # 1. AUC en fonction du nombre de folds (par fréquence) avec marqueurs de significativité
    for freq in SAMPLING_FREQUENCIES:
        freq_data = successful_results[successful_results['sampling_freq'] == freq]
        if len(freq_data) > 0:
            # Plot principal
            line = axes[0,0].plot(freq_data['n_folds'], freq_data['mean_auc'], 'o-', label=f'{freq}Hz', linewidth=2, markersize=6)
            color = line[0].get_color()
            axes[0,0].fill_between(freq_data['n_folds'], 
                                 freq_data['mean_auc'] - freq_data['std_auc'],
                                 freq_data['mean_auc'] + freq_data['std_auc'], alpha=0.2, color=color)
            
            # Ajouter des marqueurs pour les combinaisons significatives
            for _, row in freq_data.iterrows():
                combination_key = f"{row['n_folds']}_folds_{freq}Hz"
                if combination_key in significance_summary and significance_summary[combination_key]['has_significance']:
                    # Marqueur étoile pour significativité
                    axes[0,0].scatter(row['n_folds'], row['mean_auc'], marker='*', 
                                    s=100, color='red', zorder=5, alpha=0.8)
    
    axes[0,0].set_xlabel('Nombre de folds')
    axes[0,0].set_ylabel('AUC ± std')
    axes[0,0].set_title('AUC vs Nombre de Folds (par fréquence)\n* = significativité temporelle')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    
    # 2. AUC en fonction de la fréquence (par nombre de folds) avec marqueurs de significativité
    for n_folds in N_FOLDS_TO_TEST:
        fold_data = successful_results[successful_results['n_folds'] == n_folds]
        if len(fold_data) > 0:
            # Plot principal
            line = axes[0,1].plot(fold_data['sampling_freq'], fold_data['mean_auc'], 'o-', label=f'{n_folds} folds', linewidth=2, markersize=6)
            color = line[0].get_color()
            axes[0,1].fill_between(fold_data['sampling_freq'], 
                                 fold_data['mean_auc'] - fold_data['std_auc'],
                                 fold_data['mean_auc'] + fold_data['std_auc'], alpha=0.2, color=color)
            
            # Ajouter des marqueurs pour les combinaisons significatives
            for _, row in fold_data.iterrows():
                combination_key = f"{n_folds}_folds_{row['sampling_freq']}Hz"
                if combination_key in significance_summary and significance_summary[combination_key]['has_significance']:
                    # Marqueur étoile pour significativité
                    axes[0,1].scatter(row['sampling_freq'], row['mean_auc'], marker='*', 
                                    s=100, color='red', zorder=5, alpha=0.8)
    
    axes[0,1].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[0,1].set_ylabel('AUC ± std')
    axes[0,1].set_title('AUC vs Fréquence d\'échantillonnage (par folds)\n* = significativité temporelle')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    
    # 3. Heatmap des pourcentages de significativité FDR
    if significance_summary:
        # Créer des matrices pour les heatmaps
        fdr_matrix = np.full((len(N_FOLDS_TO_TEST), len(SAMPLING_FREQUENCIES)), np.nan)
        cluster_matrix = np.full((len(N_FOLDS_TO_TEST), len(SAMPLING_FREQUENCIES)), np.nan)
        
        for combination_key, sig_data in significance_summary.items():
            fold_idx = N_FOLDS_TO_TEST.index(sig_data['n_folds'])
            freq_idx = SAMPLING_FREQUENCIES.index(sig_data['sampling_freq'])
            fdr_matrix[fold_idx, freq_idx] = sig_data['fdr_percent']
            cluster_matrix[fold_idx, freq_idx] = sig_data['cluster_percent']
        
        # Plot FDR heatmap
        im1 = axes[1,0].imshow(fdr_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=100)
        axes[1,0].set_xticks(range(len(SAMPLING_FREQUENCIES)))
        axes[1,0].set_xticklabels(SAMPLING_FREQUENCIES)
        axes[1,0].set_yticks(range(len(N_FOLDS_TO_TEST)))
        axes[1,0].set_yticklabels(N_FOLDS_TO_TEST)
        axes[1,0].set_xlabel('Fréquence d\'échantillonnage (Hz)')
        axes[1,0].set_ylabel('Nombre de folds')
        axes[1,0].set_title('% Temps Significatif (FDR)')
        
        # Ajouter les valeurs dans les cases
        for i in range(len(N_FOLDS_TO_TEST)):
            for j in range(len(SAMPLING_FREQUENCIES)):
                if not np.isnan(fdr_matrix[i, j]):
                    axes[1,0].text(j, i, f'{fdr_matrix[i, j]:.1f}%', 
                                 ha='center', va='center', fontsize=8)
        
        plt.colorbar(im1, ax=axes[1,0], label='% temps FDR significatif')
    else:
        axes[1,0].text(0.5, 0.5, 'Pas de données statistiques', ha='center', va='center', 
                     transform=axes[1,0].transAxes)
        axes[1,0].set_title('% Temps Significatif (FDR)')
    
    # 4. Résumé des meilleures combinaisons
    if significance_summary:
        # Trouver les combinaisons avec le plus de significativité
        sig_combinations = [(k, v) for k, v in significance_summary.items() if v['has_significance']]
        sig_combinations.sort(key=lambda x: x[1]['fdr_percent'] + x[1]['cluster_percent'], reverse=True)
        
        axes[1,1].axis('off')
        
        summary_text = "TOP COMBINAISONS SIGNIFICATIVES:\n\n"
        
        if sig_combinations:
            for i, (combination_key, sig_data) in enumerate(sig_combinations[:8]):  # Top 8
                auc_row = successful_results[
                    (successful_results['n_folds'] == sig_data['n_folds']) & 
                    (successful_results['sampling_freq'] == sig_data['sampling_freq'])
                ]
                if len(auc_row) > 0:
                    auc_value = auc_row.iloc[0]['mean_auc']
                    summary_text += f"{i+1}. {sig_data['n_folds']} folds, {sig_data['sampling_freq']}Hz\n"
                    summary_text += f"   AUC: {auc_value:.3f}\n"
                    summary_text += f"   FDR: {sig_data['fdr_percent']:.1f}% temps\n"
                    summary_text += f"   Cluster: {sig_data['cluster_percent']:.1f}% temps\n\n"
        else:
            summary_text += "Aucune combinaison significative trouvée.\n"
            summary_text += "Toutes les courbes temporelles sont\n"
            summary_text += "non-significatives vs chance level."
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes, 
                     fontsize=10, va='top', ha='left',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1,1].set_title('Résumé Statistique')
    else:
        axes[1,1].text(0.5, 0.5, 'Pas de données statistiques disponibles', 
                     ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Résumé Statistique')
    
    plt.tight_layout()
    return fig


def create_visualization_page3(results_df, subject_info):
    """
    Crée la page 3 des visualisations: Style Dashboard avec décodage temporal principal.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    from matplotlib.gridspec import GridSpec
    import scipy.stats
    
    # Calculer les statistiques temporelles
    stats_dict = compute_temporal_statistics(results_df, chance_level=0.5)
    
    successful_results = results_df[results_df['success'] & results_df['temporal_scores'].notna()].copy()
    
    if len(successful_results) == 0:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.text(0.5, 0.5, 'Aucune donnée temporelle disponible', ha='center', va='center', transform=ax.transAxes, fontsize=20)
        return fig
    
 
    title_main_task = "Class balanced PP vs all AP"
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Dashboard - Subject: {subject_info['subject_id']} - Classifier: SVC\n"
        f"Page 3/3: Overview of - {title_main_task} - Folds & Sampling Frequency Analysis",
        fontsize=16, fontweight="bold",
    )
    
   
    gs = GridSpec(2, 2, figure=fig, height_ratios=[2.5, 1], hspace=0.4, wspace=0.3)
    
    # Plot 1: Décodage temporel principal (prend toute la première ligne)
    ax_temp = fig.add_subplot(gs[0, :])
    

    main_result = None
    for _, row in successful_results.iterrows():
        if row['n_folds'] == 5 and row['sampling_freq'] == 350:
            main_result = row
            break
    
 
    if main_result is None:
        main_result = successful_results.iloc[0]
    
  
    colors = plt.cm.viridis(np.linspace(0, 1, len(successful_results)))
    
    for i, (_, row) in enumerate(successful_results.iterrows()):
        if row['temporal_scores'] is not None and row['times'] is not None:
            label = f"{row['n_folds']} folds, {row['sampling_freq']}Hz"
            alpha = 0.7 if row.name == main_result.name else 0.3
            linewidth = 2.0 if row.name == main_result.name else 1.0
            
            ax_temp.plot(row['times'], row['temporal_scores'], 
                        color=colors[i], alpha=alpha, linewidth=linewidth, label=label)
    
    
    if len(successful_results) > 1:
    
        all_times = []
        all_scores = []
        for _, row in successful_results.iterrows():
            if row['temporal_scores'] is not None and row['times'] is not None:
                all_times.append(row['times'])
                all_scores.append(row['temporal_scores'])
        
        if all_times:
          
            min_time = max(times.min() for times in all_times)
            max_time = min(times.max() for times in all_times)
            common_times = np.linspace(min_time, max_time, 200)
            
          
            interpolated_scores = []
            for times, scores in zip(all_times, all_scores):
                interp_scores = np.interp(common_times, times, scores)
                interpolated_scores.append(interp_scores)
            
          
            if interpolated_scores:
                mean_scores = np.mean(interpolated_scores, axis=0)
                sem_scores = scipy.stats.sem(interpolated_scores, axis=0)
                

                total_epochs = subject_info['n_epochs']
                pp_count = subject_info.get('pp_count', 'N/A')
                ap_count = subject_info.get('ap_count', 'N/A')
                
                mean_label = f'Mean AUC ({total_epochs} epochs: {pp_count}PP+{ap_count}AP)'
                ax_temp.plot(common_times, mean_scores, color='black', linewidth=3.0, label=mean_label)
                

                ax_temp.fill_between(common_times,
                                   mean_scores - sem_scores,
                                   mean_scores + sem_scores,
                                   color='black', alpha=0.2, label='SEM (across combinations)')
    
    # Ajouter les barres de significativité pour chaque courbe
    # On utilise la courbe principale pour les statistiques
    main_combination_key = f"{main_result['n_folds']}_folds_{main_result['sampling_freq']}Hz"
    if main_combination_key in stats_dict:
        main_stats = stats_dict[main_combination_key]
        add_significance_bars_to_axis(ax_temp, main_stats['times'], 
                                    main_stats['fdr_data'], main_stats['cluster_data'])
    
    ax_temp.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Chance (0.5)')
    ax_temp.axvline(0, color='red', linestyle=':', alpha=0.7, label='Stimulus Onset')
    

    ax_temp.set_xlabel('Time (s)')
    ax_temp.set_ylabel('ROC AUC')
    ax_temp.set_title(f'Temporal decoding - {title_main_task}')
    ax_temp.legend(loc='best', fontsize=8)
    ax_temp.grid(True, alpha=0.3)
    
   
    if successful_results['temporal_scores'].notna().any():
        all_scores_flat = []
        for _, row in successful_results.iterrows():
            if row['temporal_scores'] is not None:
                all_scores_flat.extend(row['temporal_scores'])
        
        if all_scores_flat:
            min_score = min(all_scores_flat)
            max_score = max(all_scores_flat)
            ax_temp.set_ylim(min(min_score - 0.05, 0.4), max(max_score + 0.05, 1.0))
    
  
    ax_folds = fig.add_subplot(gs[1, 0])
    
    for freq in SAMPLING_FREQUENCIES:
        freq_data = successful_results[successful_results['sampling_freq'] == freq]
        if len(freq_data) > 0:
            ax_folds.plot(freq_data['n_folds'], freq_data['mean_auc'], 'o-', 
                         label=f'{freq}Hz', linewidth=2, markersize=6)
    
    ax_folds.set_xlabel('Nombre de folds')
    ax_folds.set_ylabel('AUC')
    ax_folds.set_title('AUC vs Folds par Fréquence')
    ax_folds.legend()
    ax_folds.grid(True, alpha=0.3)
    ax_folds.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
  
    ax_freq = fig.add_subplot(gs[1, 1])
    
    for n_folds in N_FOLDS_TO_TEST:
        fold_data = successful_results[successful_results['n_folds'] == n_folds]
        if len(fold_data) > 0:
            ax_freq.plot(fold_data['sampling_freq'], fold_data['mean_auc'], 'o-', 
                        label=f'{n_folds} folds', linewidth=2, markersize=6)
    
    ax_freq.set_xlabel('Fréquence d\'échantillonnage (Hz)')
    ax_freq.set_ylabel('AUC')
    ax_freq.set_title('AUC vs Fréquence par Folds')
    ax_freq.legend()
    ax_freq.grid(True, alpha=0.3)
    ax_freq.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def create_visualization_page4_statistics(results_df, subject_info):
    """
    Crée les pages 4+ des visualisations: Analyse statistique détaillée pour chaque courbe avec SEM.
    Retourne une liste de figures pour les différentes pages.
    Adapté pour gérer un grand nombre de combinaisons (11 folds × 6 fréquences = 66 combinaisons).
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    # Calculer les statistiques temporelles
    stats_dict = compute_temporal_statistics(results_df, chance_level=0.5)
    
    successful_results = results_df[results_df['success'] & results_df['temporal_scores'].notna()].copy()
    
    if len(successful_results) == 0 or len(stats_dict) == 0:
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.text(0.5, 0.5, 'Aucune donnée statistique disponible', ha='center', va='center', transform=ax.transAxes, fontsize=20)
        return [fig]
    
    # Configuration pour l'affichage : 3x3 = 9 plots par page (plus dense pour gérer plus de combinaisons)
    plots_per_page = 9
    n_cols = 3
    n_rows = 3
    
    # Trier les combinaisons pour un affichage cohérent
    sorted_combinations = []
    for combination_key, stats_data in stats_dict.items():
        parts = combination_key.split('_')
        n_folds = int(parts[0])
        sampling_freq = int(parts[2].replace('Hz', ''))
        sorted_combinations.append((n_folds, sampling_freq, combination_key, stats_data))
    
    # Trier par nombre de folds puis par fréquence
    sorted_combinations.sort(key=lambda x: (x[0], x[1]))
    
    # Créer les pages
    figures = []
    n_combinations = len(sorted_combinations)
    n_pages = int(np.ceil(n_combinations / plots_per_page))
    
    logger.info(f"Création de {n_pages} pages de statistiques détaillées pour {n_combinations} combinaisons")
    
    for page_num in range(n_pages):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 18))  # Plus large pour 3x3
        fig.suptitle(f'Page {4 + page_num}: Analyse Temporelle avec SEM et Significativité - Sujet {subject_info["subject_id"]} ({page_num+1}/{n_pages})', 
                    fontsize=16, fontweight='bold')
        
        # S'assurer que axes est un array 2D
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        start_idx = page_num * plots_per_page
        end_idx = min(start_idx + plots_per_page, n_combinations)
        
        plot_idx = 0
        
        for combo_idx in range(start_idx, end_idx):
            n_folds, sampling_freq, combination_key, stats_data = sorted_combinations[combo_idx]
            
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # Trouver les données correspondantes dans results_df
            matching_row = successful_results[
                (successful_results['n_folds'] == n_folds) & 
                (successful_results['sampling_freq'] == sampling_freq)
            ]
            
            if len(matching_row) == 0:
                ax.axis('off')
                plot_idx += 1
                continue
                
            row_data = matching_row.iloc[0]
            times = stats_data['times']
            temporal_scores = row_data['temporal_scores']
            
            # Calculer la SEM à partir de tous les folds
            all_fold_scores = row_data['temporal_scores_all_folds']
            if all_fold_scores is not None:
                if isinstance(all_fold_scores, list):
                    all_fold_scores = np.array(all_fold_scores)
                
                if all_fold_scores.ndim == 1:
                    all_fold_scores = all_fold_scores[np.newaxis, :]
                elif all_fold_scores.ndim > 2:
                    all_fold_scores = all_fold_scores.reshape(all_fold_scores.shape[0], -1)
                
                # Calculer SEM
                sem_scores = scipy.stats.sem(all_fold_scores, axis=0)
                
                # Tracer la courbe principale avec SEM
                ax.plot(times, temporal_scores, 'b-', linewidth=2.0, 
                       label=f'{n_folds}F, {sampling_freq}Hz (AUC: {row_data["mean_auc"]:.3f})')
                
                # Ajouter la zone SEM
                ax.fill_between(times, 
                              temporal_scores - sem_scores,
                              temporal_scores + sem_scores,
                              color='blue', alpha=0.2, label='SEM')
            else:
                # Pas de données de folds multiples, tracer seulement la courbe moyenne
                ax.plot(times, temporal_scores, 'b-', linewidth=2.0, 
                       label=f'{n_folds}F, {sampling_freq}Hz (AUC: {row_data["mean_auc"]:.3f})')
            
            # Ajouter les barres de significativité en bas du plot
            add_significance_bars_to_axis(ax, times, stats_data['fdr_data'], stats_data['cluster_data'])
            
            # Ajouter les lignes de référence
            ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Chance (0.5)')
            ax.axvline(0, color='red', linestyle=':', alpha=0.7, label='Stimulus Onset')
            
            # Formatting - plus compact pour 3x3
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('ROC AUC', fontsize=9)
            ax.set_title(f'{n_folds} folds, {sampling_freq}Hz\nAUC: {row_data["mean_auc"]:.3f}±{row_data["std_auc"]:.3f}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            # Ajuster les limites y pour inclure les barres de significativité
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min - 0.08 * (y_max - y_min), y_max)
            
            # Ajouter des informations statistiques en texte dans le coin
            stats_text = ""
            if stats_data['fdr_data'] and stats_data['fdr_data'].get('mask') is not None:
                n_sig_fdr = np.sum(stats_data['fdr_data']['mask'])
                fdr_percent = (n_sig_fdr / len(times)) * 100
                stats_text += f"FDR: {fdr_percent:.1f}%\n"
            
            if stats_data['cluster_data'] and stats_data['cluster_data'].get('global_mask') is not None:
                n_sig_cluster = np.sum(stats_data['cluster_data']['global_mask'])
                cluster_percent = (n_sig_cluster / len(times)) * 100
                stats_text += f"Cluster: {cluster_percent:.1f}%\n"
            
            if stats_text:
                ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=7)
            
            plot_idx += 1
        
        # Cacher les axes vides
        for i in range(plot_idx, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


def save_results_and_visualizations(results_df, subject_info):
    """
    Sauvegarde les résultats et génère les visualisations.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'test_folds_sampling_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
  
    results_file = os.path.join(output_dir, 'results_summary.csv')
    results_df.to_csv(results_file, index=False)
    logger.info(f"Résultats sauvegardés: {results_file}")
    
   
    logger.info("Génération des visualisations...")
    

    fig1 = create_visualization_page1(results_df, subject_info)
    fig1.savefig(os.path.join(output_dir, 'page1_performance_heatmaps.png'), dpi=300, bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'page1_performance_heatmaps.pdf'), bbox_inches='tight')
 
    fig2 = create_visualization_page2(results_df, subject_info)
    fig2.savefig(os.path.join(output_dir, 'page2_detailed_analysis.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'page2_detailed_analysis.pdf'), bbox_inches='tight')
    

    fig3 = create_visualization_page3(results_df, subject_info)
    fig3.savefig(os.path.join(output_dir, 'page3_temporal_analysis.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'page3_temporal_analysis.pdf'), bbox_inches='tight')
    

    fig_high_folds = create_visualization_high_folds_analysis(results_df, subject_info)
    fig_high_folds.savefig(os.path.join(output_dir, 'page_special_high_folds_analysis.png'), dpi=300, bbox_inches='tight')
    fig_high_folds.savefig(os.path.join(output_dir, 'page_special_high_folds_analysis.pdf'), bbox_inches='tight')
    
    # Pages 4+: Analyses statistiques détaillées (multiples pages - maintenant 3x3 layout)
    page4_figures = create_visualization_page4_statistics(results_df, subject_info)
    for i, fig in enumerate(page4_figures):
        page_num = 4 + i
        fig.savefig(os.path.join(output_dir, f'page{page_num}_statistical_analysis_detailed.png'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(output_dir, f'page{page_num}_statistical_analysis_detailed.pdf'), bbox_inches='tight')
    
    logger.info(f"Généré {len(page4_figures)} pages d'analyses statistiques détaillées")
    
    plt.show()
    
    create_summary_report(results_df, subject_info, output_dir)
    
    logger.info(f"Visualisations sauvegardées dans: {output_dir}")
    
    return output_dir


def create_summary_report(results_df, subject_info, output_dir):
    """
    Crée un rapport de synthèse des résultats.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
        output_dir (str): Répertoire de sortie
    """
    # Calculer les statistiques temporelles pour le rapport
    stats_dict = compute_temporal_statistics(results_df, chance_level=0.5)
    significance_summary = calculate_significance_summary(stats_dict)
    
    report_file = os.path.join(output_dir, 'synthesis_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("RAPPORT DE SYNTHÈSE - TEST FOLDS ET FRÉQUENCES D'ÉCHANTILLONNAGE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Sujet testé: {subject_info['subject_id']}\n")
        f.write(f"Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fréquence originale: {subject_info['sampling_freq']} Hz\n")
        f.write(f"Nombre d'epochs: {subject_info['n_epochs']}\n")
        f.write(f"Nombre de canaux: {subject_info['n_channels']}\n\n")
        
        f.write("PARAMÈTRES TESTÉS:\n")
        f.write(f"- Nombres de folds: {N_FOLDS_TO_TEST}\n")
        f.write(f"- Fréquences d'échantillonnage: {SAMPLING_FREQUENCIES} Hz\n")
        f.write(f"- Total de combinaisons: {len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)}\n")
        f.write(f"- Nouveaux folds ajoutés: [20, 24, 28, 32, 36]\n")
        f.write(f"- Fréquence 500 Hz ajoutée (pas de décimation)\n\n")
        

        successful_results = results_df[results_df['success']]
        f.write("RÉSULTATS GÉNÉRAUX:\n")
        f.write(f"- Tests réussis: {len(successful_results)}/{len(results_df)}\n")
        f.write(f"- Taux de réussite: {len(successful_results)/len(results_df)*100:.1f}%\n")
        f.write(f"- Analyses statistiques: {len(stats_dict)} combinaisons\n\n")
        

        if significance_summary:
            sig_combinations = [v for v in significance_summary.values() if v['has_significance']]
            f.write("SIGNIFICATIVITÉ TEMPORELLE:\n")
            f.write(f"- Combinaisons significatives: {len(sig_combinations)}/{len(significance_summary)}\n")
            f.write(f"- Taux de significativité: {len(sig_combinations)/len(significance_summary)*100:.1f}%\n\n")
            
            if sig_combinations:

                sig_combinations.sort(key=lambda x: x['fdr_percent'] + x['cluster_percent'], reverse=True)
                f.write("TOP 5 COMBINAISONS SIGNIFICATIVES:\n")
                for i, sig_data in enumerate(sig_combinations[:5]):
                    auc_row = successful_results[
                        (successful_results['n_folds'] == sig_data['n_folds']) & 
                        (successful_results['sampling_freq'] == sig_data['sampling_freq'])
                    ]
                    if len(auc_row) > 0:
                        auc_value = auc_row.iloc[0]['mean_auc']
                        f.write(f"{i+1}. {sig_data['n_folds']} folds, {sig_data['sampling_freq']}Hz - AUC: {auc_value:.3f}\n")
                        f.write(f"   FDR significatif: {sig_data['fdr_percent']:.1f}% du temps\n")
                        f.write(f"   Cluster significatif: {sig_data['cluster_percent']:.1f}% du temps\n")
                f.write("\n")
        else:
            f.write("SIGNIFICATIVITÉ TEMPORELLE:\n")
            f.write("- Aucune analyse statistique disponible\n\n")
        
        if len(successful_results) > 0:
            
            best_auc = successful_results.loc[successful_results['mean_auc'].idxmax()]
            f.write("MEILLEURE PERFORMANCE:\n")
            f.write(f"- Configuration: {best_auc['n_folds']} folds, {best_auc['sampling_freq']} Hz\n")
            f.write(f"- AUC: {best_auc['mean_auc']:.3f} ± {best_auc['std_auc']:.3f}\n")
            f.write(f"- Précision: {best_auc['mean_accuracy']:.3f}\n")
            f.write(f"- Score de pic: {best_auc['peak_score']:.3f} à {best_auc['peak_time']:.3f}s\n")
            
           
            best_key = f"{best_auc['n_folds']}_folds_{best_auc['sampling_freq']}Hz"
            if best_key in significance_summary and significance_summary[best_key]['has_significance']:
                best_sig = significance_summary[best_key]
                f.write(f"- Significativité: FDR {best_sig['fdr_percent']:.1f}%, Cluster {best_sig['cluster_percent']:.1f}%\n")
            else:
                f.write(f"- Significativité: Non significative\n")
            f.write("\n")
            
          
            most_stable = successful_results.loc[successful_results['std_auc'].idxmin()]
            f.write("CONFIGURATION LA PLUS STABLE:\n")
            f.write(f"- Configuration: {most_stable['n_folds']} folds, {most_stable['sampling_freq']} Hz\n")
            f.write(f"- AUC: {most_stable['mean_auc']:.3f} ± {most_stable['std_auc']:.3f}\n")
            f.write(f"- Coefficient de variation: {most_stable['std_auc']/most_stable['mean_auc']:.3f}\n\n")
            
            # Analyses par fréquence
            f.write("ANALYSE PAR FRÉQUENCE:\n")
            for freq in SAMPLING_FREQUENCIES:
                freq_data = successful_results[successful_results['sampling_freq'] == freq]
                if len(freq_data) > 0:
                    mean_auc = freq_data['mean_auc'].mean()
                    std_auc = freq_data['mean_auc'].std()
                    # Compter les combinaisons significatives pour cette fréquence
                    freq_sig_count = sum(1 for v in significance_summary.values() 
                                       if v['sampling_freq'] == freq and v['has_significance'])
                    f.write(f"- {freq} Hz: AUC moyen = {mean_auc:.3f} ± {std_auc:.3f} (n={len(freq_data)}, {freq_sig_count} sig.)\n")
            
            f.write("\nANALYSE PAR NOMBRE DE FOLDS:\n")
            for n_folds in N_FOLDS_TO_TEST:
                fold_data = successful_results[successful_results['n_folds'] == n_folds]
                if len(fold_data) > 0:
                    mean_auc = fold_data['mean_auc'].mean()
                    std_auc = fold_data['mean_auc'].std()
                    # Compter les combinaisons significatives pour ce nombre de folds
                    fold_sig_count = sum(1 for v in significance_summary.values() 
                                       if v['n_folds'] == n_folds and v['has_significance'])
                    f.write(f"- {n_folds} folds: AUC moyen = {mean_auc:.3f} ± {std_auc:.3f} (n={len(fold_data)}, {fold_sig_count} sig.)\n")
        
        f.write("\nFICHIERS GÉNÉRÉS:\n")
        f.write("- results_summary.csv: Résultats détaillés\n")
        f.write("- page1_performance_heatmaps.png/.pdf: Heatmaps des performances\n")
        f.write("- page2_detailed_analysis.png/.pdf: Analyses détaillées avec statistiques\n")
        f.write("- page3_temporal_analysis.png/.pdf: Analyses temporelles avec significativité\n")
        f.write("- page4+_statistical_analysis.png/.pdf: Analyses statistiques détaillées (multiples pages)\n")
        f.write("- synthesis_report.txt: Ce rapport de synthèse\n")
    
    logger.info(f"Rapport de synthèse créé: {report_file}")


def compute_temporal_statistics(results_df, chance_level=0.5):
    """
    Calcule les statistiques (FDR et cluster) pour chaque courbe temporelle.
    
    Args:
        results_df (pd.DataFrame): Résultats avec les scores temporels
        chance_level (float): Niveau de chance pour les tests
        
    Returns:
        dict: Dictionnaire avec les statistiques pour chaque combinaison
    """
    logger.info("Calcul des statistiques temporelles...")
    
    stats_dict = {}
    
    # Filtrer les résultats valides avec données temporelles
    valid_results = results_df[
        results_df['success'] & 
        results_df['temporal_scores_all_folds'].notna()
    ].copy()
    
    for idx, row in valid_results.iterrows():
        combination_key = f"{row['n_folds']}_folds_{row['sampling_freq']}Hz"
        
        try:
            # Récupérer les scores de tous les folds
            all_fold_scores = row['temporal_scores_all_folds']
            times = row['times']
            
            if all_fold_scores is None or times is None:
                logger.warning(f"Données manquantes pour {combination_key}")
                continue
                
            # Convertir en array numpy si nécessaire
            if isinstance(all_fold_scores, list):
                all_fold_scores = np.array(all_fold_scores)
            
            # S'assurer que les dimensions sont correctes (n_folds, n_times)
            if all_fold_scores.ndim == 1:
                # Un seul fold, ajouter une dimension
                all_fold_scores = all_fold_scores[np.newaxis, :]
            elif all_fold_scores.ndim > 2:
                # Reshape si nécessaire
                all_fold_scores = all_fold_scores.reshape(all_fold_scores.shape[0], -1)
            
            n_folds, n_times = all_fold_scores.shape
            
            logger.debug(f"Stats pour {combination_key}: {n_folds} folds, {n_times} temps")
            
            # Test FDR pointwise
            try:
                fdr_stats, fdr_mask, fdr_p_corrected, fdr_test_info = perform_pointwise_fdr_correction_on_scores(
                    input_data_array=all_fold_scores,
                    chance_level=chance_level,
                    alpha_significance_level=0.05,
                    fdr_correction_method="indep",
                    alternative_hypothesis="greater",
                    statistical_test_type="wilcoxon"
                )
                
                fdr_data = {
                    'stats': fdr_stats,
                    'mask': fdr_mask,
                    'p_corrected': fdr_p_corrected,
                    'test_info': fdr_test_info
                }
            except Exception as e:
                logger.warning(f"Erreur FDR pour {combination_key}: {e}")
                fdr_data = None
            
            # Test cluster permutation
            try:
                cluster_stats, cluster_masks, cluster_p_values, h0_distribution = perform_cluster_permutation_test(
                    input_data_array=all_fold_scores,
                    chance_level=chance_level,
                    n_permutations=1024,
                    cluster_threshold_config=2.0,  # threshold t-stat
                    alternative_hypothesis="greater",
                    n_jobs=1,
                    random_seed=42
                )
                
                # Créer un masque global pour tous les clusters significatifs
                cluster_global_mask = np.zeros(n_times, dtype=bool)
                if cluster_masks and cluster_p_values is not None:
                    for mask, p_val in zip(cluster_masks, cluster_p_values):
                        if p_val < 0.05 and mask is not None:
                            cluster_global_mask |= mask
                
                cluster_data = {
                    'stats': cluster_stats,
                    'masks': cluster_masks,
                    'p_values': cluster_p_values,
                    'h0_distribution': h0_distribution,
                    'global_mask': cluster_global_mask
                }
            except Exception as e:
                logger.warning(f"Erreur cluster pour {combination_key}: {e}")
                cluster_data = None
            
            stats_dict[combination_key] = {
                'times': times,
                'fdr_data': fdr_data,
                'cluster_data': cluster_data,
                'n_folds': n_folds,
                'n_times': n_times
            }
            
        except Exception as e:
            logger.error(f"Erreur générale stats pour {combination_key}: {e}")
            continue
    
    logger.info(f"Statistiques calculées pour {len(stats_dict)} combinaisons")
    return stats_dict


def calculate_significance_summary(stats_dict):
    """
    Calcule un résumé des statistiques de significativité pour chaque combinaison.
    
    Args:
        stats_dict: Dictionnaire des statistiques temporelles
        
    Returns:
        dict: Résumé avec pourcentages de temps significatifs
    """
    significance_summary = {}
    
    for combination_key, stats_data in stats_dict.items():
        parts = combination_key.split('_')
        n_folds = int(parts[0])
        sampling_freq = int(parts[2].replace('Hz', ''))
        
        n_times = stats_data['n_times']
        
        # Calculer les pourcentages de significativité
        fdr_percent = 0
        cluster_percent = 0
        
        if stats_data['fdr_data'] and stats_data['fdr_data'].get('mask') is not None:
            fdr_percent = (np.sum(stats_data['fdr_data']['mask']) / n_times) * 100
            
        if stats_data['cluster_data'] and stats_data['cluster_data'].get('global_mask') is not None:
            cluster_percent = (np.sum(stats_data['cluster_data']['global_mask']) / n_times) * 100
        
        significance_summary[combination_key] = {
            'n_folds': n_folds,
            'sampling_freq': sampling_freq,
            'fdr_percent': fdr_percent,
            'cluster_percent': cluster_percent,
            'has_significance': fdr_percent > 0 or cluster_percent > 0
        }
    
    return significance_summary


def add_significance_bars_to_axis(ax, times, fdr_data=None, cluster_data=None):
    """
    Ajoute les barres de significativité à un axe matplotlib.
    
    Args:
        ax: Axe matplotlib
        times: Array des temps
        fdr_data: Données FDR (dict avec 'mask')
        cluster_data: Données cluster (dict avec 'global_mask')
    """
    y_min, y_max = ax.get_ylim()
    bar_height = 0.01 * (y_max - y_min)
    
    # Barre FDR (en vert, plus proche du bas)
    if fdr_data and fdr_data.get('mask') is not None:
        fdr_mask = fdr_data['mask']
        if np.any(fdr_mask):
            y_fdr = y_min + 0.01 * (y_max - y_min)
            ax.fill_between(times, y_fdr - bar_height/2, y_fdr + bar_height/2,
                          where=fdr_mask, color='green', alpha=0.8,
                          step='mid', label='FDR p<0.05')
    
    # Barre cluster (en orange, légèrement au-dessus)
    if cluster_data and cluster_data.get('global_mask') is not None:
        cluster_mask = cluster_data['global_mask']
        if np.any(cluster_mask):
            y_cluster = y_min + 0.03 * (y_max - y_min)
            ax.fill_between(times, y_cluster - bar_height/2, y_cluster + bar_height/2,
                          where=cluster_mask, color='orange', alpha=0.8,
                          step='mid', label='Cluster p<0.05')
    


def run_quick_test():
    """
    Exécute un test rapide avec un sous-ensemble de paramètres pour validation.
    """
    logger.info("=== TEST RAPIDE ===")
    
    # Paramètres réduits pour test rapide
    global N_FOLDS_TO_TEST, SAMPLING_FREQUENCIES
    original_folds = N_FOLDS_TO_TEST.copy()
    original_freqs = SAMPLING_FREQUENCIES.copy()
    
    # Réduire les paramètres pour test rapide - inclure quelques nouveaux folds
    N_FOLDS_TO_TEST = [2, 5, 20, 32]  # Mélange ancien/nouveau
    SAMPLING_FREQUENCIES = [500, 250, 125]  # Inclure 500 Hz
    
    try:
        results_df, subject_info = run_comprehensive_analysis()
        output_dir = save_results_and_visualizations(results_df, subject_info)
        logger.info(f"Test rapide terminé. Résultats dans: {output_dir}")
        return output_dir
    finally:
        # Restaurer les paramètres originaux
        N_FOLDS_TO_TEST = original_folds
        SAMPLING_FREQUENCIES = original_freqs


def main():

    
    parser = argparse.ArgumentParser(description='Test de décimation et folds pour EEG')
    parser.add_argument('--quick', action='store_true', 
                       help='Exécuter un test rapide avec paramètres réduits')
    parser.add_argument('--subject-file', type=str, default=None,
                       help='Fichier de données du sujet (optionnel)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Répertoire de sortie (optionnel)')
    
    args = parser.parse_args()
    

    if args.subject_file:
        global TEST_SUBJECT_FILE
        TEST_SUBJECT_FILE = args.subject_file
        logger.info(f"Utilisation du fichier sujet: {TEST_SUBJECT_FILE}")
    
    try:
        if args.quick:
            logger.info("Lancement du test rapide...")
            output_dir = run_quick_test()
        else:
            logger.info("Lancement de l'analyse complète...")
            results_df, subject_info = run_comprehensive_analysis()
            output_dir = save_results_and_visualizations(results_df, subject_info)
        
        logger.info("=" * 50)
        logger.info("ANALYSE TERMINÉE AVEC SUCCÈS")
        logger.info(f"Résultats sauvegardés dans: {output_dir}")
        logger.info("=" * 50)
        

        if args.output_dir:
            import shutil
            final_output = os.path.join(args.output_dir, os.path.basename(output_dir))
            shutil.copytree(output_dir, final_output)
            logger.info(f"Résultats copiés dans: {final_output}")
        
        return output_dir
        
    except KeyboardInterrupt:
        logger.info("Analyse interrompue par l'utilisateur")
        return None
    except Exception as e:
        logger.error(f"Erreur durant l'analyse: {e}", exc_info=True)
        return None


def validate_installation():
    """
    Valide que toutes les dépendances nécessaires sont installées.
    """
    required_modules = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'mne', 'scipy'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Modules manquants: {missing_modules}")
        logger.error("Installez les dépendances avec: pip install numpy pandas matplotlib seaborn scikit-learn mne scipy")
        return False
    

    try:
        from utils.stats_utils import perform_pointwise_fdr_correction_on_scores, perform_cluster_permutation_test
        logger.info("Modules statistiques trouvés")
    except ImportError as e:
        logger.error(f"Modules statistiques manquants: {e}")
        return False
    
    logger.info("Toutes les dépendances sont installées")
    return True


def print_usage_info():
    """
    Affiche les informations d'utilisation du script.
    """
    print("\n" + "="*60)
    print("SCRIPT DE TEST - DÉCIMATION ET FOLDS POUR EEG")
    print("="*60)
    print("\nUsage:")
    print("  python test_decimate_folds.py [options]")
    print("\nOptions:")
    print("  --quick                    Test rapide (2 folds × 2 fréquences)")
    print("  --subject-file PATH        Fichier de données personnalisé")
    print("  --output-dir PATH          Répertoire de sortie personnalisé")
    print("  -h, --help                 Afficher cette aide")
    print("\nExemples:")
    print("  python test_decimate_folds.py --quick")
    print("  python test_decimate_folds.py --subject-file /path/to/data.fif")
    print("  python test_decimate_folds.py --output-dir /path/to/results")
    print("\nConfiguration actuelle:")
    print(f"  - Fichier sujet: {TEST_SUBJECT_FILE}")
    print(f"  - Folds testés: {N_FOLDS_TO_TEST}")
    print(f"  - Fréquences testées: {SAMPLING_FREQUENCIES} Hz")
    print(f"  - FeatureSelection configs: Désactivée")
    print(f"  - Nouveaux hauts folds: [20, 24, 28, 32, 36]")
    print(f"  - Fréquence 500 Hz ajoutée")
    total_combinations = len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)
    print(f"  - Total combinaisons: {total_combinations}")
    print("\nSorties générées:")
    print("  - results_summary.csv: Données tabulaires")
    print("  - page1_performance_heatmaps.png/pdf: Heatmaps performances")
    print("  - page2_detailed_analysis.png/pdf: Analyses détaillées")
    print("  - page3_temporal_analysis.png/pdf: Analyses temporelles")
    print("  - page_special_high_folds_analysis.png/pdf: Analyse hauts folds")
    print("  - page4+_statistical_analysis.png/pdf: Statistiques détaillées (3x3 layout)")
    print("  - synthesis_report.txt: Rapport de synthèse")
    print("="*60)


if __name__ == "__main__":
  
    print_usage_info()
    

    if not validate_installation():
        sys.exit(1)
    

    main()
