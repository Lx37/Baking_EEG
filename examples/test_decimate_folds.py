import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


N_FOLDS_TO_TEST = [2, 5, 7, 9, 13, 16]
SAMPLING_FREQUENCIES = [350, 250, 200, 125, 100]  # Hz
ORIGINAL_FREQ = 500  # Fréquence d'origine assumée


TEST_SUBJECT_FILE = "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG_data/ME64_preproc_noICA_PPAP-epo_ar.fif"
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
        logger.warning(f"Fréquence cible {target_freq}Hz >= fréquence originale {original_freq}Hz. Pas de décimation.")
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
        

        results = run_temporal_decoding_analysis(
            epochs_data=X,
            target_labels=y_encoded,
            classifier_model_type="svc",
            cross_validation_splitter=cv_splitter,
            use_grid_search=False,
            use_csp_for_temporal_pipelines=False,
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
        
        # Vérifier que les résultats sont valides
        if cv_scores is None or len(cv_scores) == 0:
            logger.error(f"Scores CV invalides pour {n_folds} folds, {sampling_freq}Hz")
            return create_empty_result(n_folds, sampling_freq)
            
        if temporal_scores is None or len(temporal_scores) == 0:
            logger.error(f"Scores temporels invalides pour {n_folds} folds, {sampling_freq}Hz")
            return create_empty_result(n_folds, sampling_freq)
        
        # Extraire les métriques principales
        mean_auc = np.mean(cv_scores)
        std_auc = np.std(cv_scores)
        mean_accuracy = global_metrics.get('accuracy', np.nan) if global_metrics else np.nan
        
        # Temps de décodage temporal
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
        'event_id': event_id
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
    
    logger.info(f"Début de l'analyse complète: {len(N_FOLDS_TO_TEST)} folds × {len(SAMPLING_FREQUENCIES)} fréquences = {len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)} combinaisons")
    
   
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
    Crée la page 2 des visualisations: Analyses par fréquence et par folds.
    
    Args:
        results_df (pd.DataFrame): Résultats des tests
        subject_info (dict): Informations sur le sujet
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Page 2: Analyses Détaillées - Sujet {subject_info["subject_id"]}', fontsize=16, fontweight='bold')
    

    successful_results = results_df[results_df['success']].copy()
    
    if len(successful_results) == 0:
        plt.text(0.5, 0.5, 'Aucun résultat valide disponible', ha='center', va='center', transform=fig.transFigure, fontsize=20)
        return fig
    
    # 1. AUC en fonction du nombre de folds (par fréquence)
    for freq in SAMPLING_FREQUENCIES:
        freq_data = successful_results[successful_results['sampling_freq'] == freq]
        if len(freq_data) > 0:
            axes[0,0].plot(freq_data['n_folds'], freq_data['mean_auc'], 'o-', label=f'{freq}Hz', linewidth=2, markersize=6)
            axes[0,0].fill_between(freq_data['n_folds'], 
                                 freq_data['mean_auc'] - freq_data['std_auc'],
                                 freq_data['mean_auc'] + freq_data['std_auc'], alpha=0.2)
    
    axes[0,0].set_xlabel('Nombre de folds')
    axes[0,0].set_ylabel('AUC ± std')
    axes[0,0].set_title('AUC vs Nombre de Folds (par fréquence)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    
    # 2. AUC en fonction de la fréquence (par nombre de folds)
    for n_folds in N_FOLDS_TO_TEST:
        fold_data = successful_results[successful_results['n_folds'] == n_folds]
        if len(fold_data) > 0:
            axes[0,1].plot(fold_data['sampling_freq'], fold_data['mean_auc'], 'o-', label=f'{n_folds} folds', linewidth=2, markersize=6)
            axes[0,1].fill_between(fold_data['sampling_freq'], 
                                 fold_data['mean_auc'] - fold_data['std_auc'],
                                 fold_data['mean_auc'] + fold_data['std_auc'], alpha=0.2)
    
    axes[0,1].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[0,1].set_ylabel('AUC ± std')
    axes[0,1].set_title('AUC vs Fréquence d\'échantillonnage (par folds)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    
    # 3. Temps de pic en fonction de la fréquence
    for n_folds in N_FOLDS_TO_TEST:
        fold_data = successful_results[successful_results['n_folds'] == n_folds]
        if len(fold_data) > 0:
            axes[1,0].plot(fold_data['sampling_freq'], fold_data['peak_time'], 'o-', label=f'{n_folds} folds', linewidth=2, markersize=6)
    
    axes[1,0].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[1,0].set_ylabel('Temps de pic (s)')
    axes[1,0].set_title('Temps de Pic Temporal vs Fréquence')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Distribution des performances
    successful_results['combination'] = successful_results['n_folds'].astype(str) + ' folds'
    sns.boxplot(data=successful_results, x='sampling_freq', y='mean_auc', hue='combination', ax=axes[1,1])
    axes[1,1].set_xlabel('Fréquence d\'échantillonnage (Hz)')
    axes[1,1].set_ylabel('AUC')
    axes[1,1].set_title('Distribution des AUC par Fréquence et Folds')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].grid(True, alpha=0.3)
    
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
                pp_count = subject_info['pp_count']
                ap_count = subject_info['ap_count']
                
                mean_label = f'Mean AUC ({total_epochs} epochs: {pp_count}PP+{ap_count}AP)'
                ax_temp.plot(common_times, mean_scores, color='black', linewidth=3.0, label=mean_label)
                

                ax_temp.fill_between(common_times,
                                   mean_scores - sem_scores,
                                   mean_scores + sem_scores,
                                   color='black', alpha=0.2, label='SEM (across combinations)')
    
  
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
        f.write(f"- Total de combinaisons: {len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)}\n\n")
        
        # Statistiques de réussite
        successful_results = results_df[results_df['success']]
        f.write("RÉSULTATS GÉNÉRAUX:\n")
        f.write(f"- Tests réussis: {len(successful_results)}/{len(results_df)}\n")
        f.write(f"- Taux de réussite: {len(successful_results)/len(results_df)*100:.1f}%\n\n")
        
        if len(successful_results) > 0:
            # Meilleures performances
            best_auc = successful_results.loc[successful_results['mean_auc'].idxmax()]
            f.write("MEILLEURE PERFORMANCE:\n")
            f.write(f"- Configuration: {best_auc['n_folds']} folds, {best_auc['sampling_freq']} Hz\n")
            f.write(f"- AUC: {best_auc['mean_auc']:.3f} ± {best_auc['std_auc']:.3f}\n")
            f.write(f"- Précision: {best_auc['mean_accuracy']:.3f}\n")
            f.write(f"- Score de pic: {best_auc['peak_score']:.3f} à {best_auc['peak_time']:.3f}s\n\n")
            
            # Stabilité (plus faible variabilité)
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
                    f.write(f"- {freq} Hz: AUC moyen = {mean_auc:.3f} ± {std_auc:.3f} (n={len(freq_data)})\n")
            
            f.write("\nANALYSE PAR NOMBRE DE FOLDS:\n")
            for n_folds in N_FOLDS_TO_TEST:
                fold_data = successful_results[successful_results['n_folds'] == n_folds]
                if len(fold_data) > 0:
                    mean_auc = fold_data['mean_auc'].mean()
                    std_auc = fold_data['mean_auc'].std()
                    f.write(f"- {n_folds} folds: AUC moyen = {mean_auc:.3f} ± {std_auc:.3f} (n={len(fold_data)})\n")
        
        f.write("\nFICHIERS GÉNÉRÉS:\n")
        f.write("- results_summary.csv: Résultats détaillés\n")
        f.write("- page1_performance_heatmaps.png/.pdf: Heatmaps des performances\n")
        f.write("- page2_detailed_analysis.png/.pdf: Analyses détaillées\n")
        f.write("- page3_temporal_analysis.png/.pdf: Analyses temporelles\n")
        f.write("- synthesis_report.txt: Ce rapport de synthèse\n")
    
    logger.info(f"Rapport de synthèse créé: {report_file}")


def main():
    """
    Fonction principale d'exécution du script.
    """
    logger.info("=" * 60)
    logger.info("DÉBUT DE L'ANALYSE FOLDS ET FRÉQUENCES D'ÉCHANTILLONNAGE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    try:
      
        results_df, subject_info = run_comprehensive_analysis()
        
      
        output_dir = save_results_and_visualizations(results_df, subject_info)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("ANALYSE TERMINÉE AVEC SUCCÈS")
        logger.info(f"Durée totale: {duration}")
        logger.info(f"Résultats disponibles dans: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
