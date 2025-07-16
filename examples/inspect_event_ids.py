"""
Script d'inspection des event_id dans les fichiers EEG
======================================================

Ce script permet d'explorer et visualiser les event_id présents dans les fichiers
epochés (.fif) pour différents sujets et protocoles.

Usage:
    python -m examples.inspect_event_ids --subject_id LAB1 --group controls
    python -m examples.inspect_event_ids --subject_id TpAB19 --group del --protocol LG
    python -m examples.inspect_event_ids --file_path /path/to/specific/file.fif

"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import mne

# Ajouter le répertoire parent au path Python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import ALL_SUBJECTS_GROUPS
from config.decoding_config import EVENT_ID_LG
from utils.utils import configure_project_paths

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_epoch_files(subject_id, group, protocol="PP", base_path=None):
    """
    Trouve les fichiers d'epochs pour un sujet donné.
    
    Parameters
    ----------
    subject_id : str
        ID du sujet (ex: 'LAB1', 'TpAB19')
    group : str
        Groupe du sujet ('controls', 'del', 'nodel')
    protocol : str
        Protocole ('PP' ou 'LG')
    base_path : str, optional
        Chemin de base des données
        
    Returns
    -------
    list
        Liste des fichiers trouvés
    """
    if base_path is None:
        from getpass import getuser
        base_path, _ = configure_project_paths(getuser())
    
    # Déterminer le dossier de données selon le groupe et protocole
    if protocol.upper() == "PP":
        if group == 'controls':
            data_folder = 'PP_CONTROLS_0.5'
        elif group in ['del', 'nodel']:
            data_folder = f'PP_PATIENTS_{group.upper()}_0.5'
        else:
            raise ValueError(f"Groupe inconnu pour PP: {group}")
    elif protocol.upper() == "LG":
        if group == 'controls':
            data_folder = 'LG_CONTROLS_0.5'
        elif group in ['del', 'nodel']:
            data_folder = f'LG_PATIENTS_{group.upper()}_0.5'
        else:
            raise ValueError(f"Groupe inconnu pour LG: {group}")
    else:
        raise ValueError(f"Protocole inconnu: {protocol}")
    
    data_path = os.path.join(base_path, data_folder, 'data_epochs')
    
    # Chercher les fichiers possibles
    possible_files = [
        f"{subject_id}_{protocol}_preproc_noICA_{protocol}-epo_ar.fif",
        f"{subject_id}_{protocol}_preproc_ICA_{protocol}-epo_ar.fif",
        f"{subject_id}_{protocol}_preproc_{protocol}-epo_ar.fif",
        f"{subject_id}_{protocol}-epo.fif",
        f"{subject_id}-epo.fif"
    ]
    
    found_files = []
    for filename in possible_files:
        full_path = os.path.join(data_path, filename)
        if os.path.exists(full_path):
            found_files.append(full_path)
    
    return found_files


def inspect_epochs_file(file_path, show_events_table=True, show_summary=True):
    """
    Inspecte un fichier d'epochs et affiche les informations sur les event_id.
    
    Parameters
    ----------
    file_path : str
        Chemin vers le fichier .fif
    show_events_table : bool
        Afficher le tableau détaillé des événements
    show_summary : bool
        Afficher le résumé
        
    Returns
    -------
    dict
        Dictionnaire avec les informations extraites
    """
    logger.info(f"Inspection du fichier: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Fichier non trouvé: {file_path}")
        return None
    
    try:
        # Charger les epochs avec le niveau de log minimal
        with mne.utils.use_log_level('warning'):
            epochs = mne.read_epochs(file_path, proj=False, verbose=False, preload=False)
        
        # Extraire les informations
        info = {
            'file_path': file_path,
            'n_epochs': len(epochs),
            'n_channels': len(epochs.ch_names),
            'sampling_freq': epochs.info['sfreq'],
            'tmin': epochs.tmin,
            'tmax': epochs.tmax,
            'event_id': epochs.event_id.copy(),
            'events': epochs.events.copy(),
            'ch_names': epochs.ch_names.copy(),
            'times': epochs.times.copy()
        }
        
        # Analyser les événements
        event_counts = Counter(epochs.events[:, 2])
        
        if show_summary:
            print("\n" + "="*80)
            print(f"RÉSUMÉ DU FICHIER: {os.path.basename(file_path)}")
            print("="*80)
            print(f"Nombre d'epochs: {info['n_epochs']}")
            print(f"Nombre de canaux: {info['n_channels']}")
            print(f"Fréquence d'échantillonnage: {info['sampling_freq']} Hz")
            print(f"Fenêtre temporelle: {info['tmin']:.3f} à {info['tmax']:.3f} s")
            print(f"Durée d'une epoch: {info['tmax'] - info['tmin']:.3f} s")
            
            print(f"\nNOMBRE DE POINTS TEMPORELS: {len(info['times'])}")
            print(f"Résolution temporelle: {1000/info['sampling_freq']:.2f} ms")
        
        if info['event_id']:
            if show_summary:
                print(f"\nEVENT_ID DICTIONARY:")
                print("-" * 40)
                for event_name, event_code in sorted(info['event_id'].items()):
                    count = event_counts.get(event_code, 0)
                    print(f"  {event_name:<25} : {event_code:>3} ({count:>3} epochs)")
                
                print(f"\nSTATISTIQUES DES ÉVÉNEMENTS:")
                print("-" * 40)
                total_events = sum(event_counts.values())
                print(f"Total des événements: {total_events}")
                
                for event_name, event_code in sorted(info['event_id'].items()):
                    count = event_counts.get(event_code, 0)
                    percentage = (count / total_events * 100) if total_events > 0 else 0
                    print(f"  {event_name:<25} : {count:>3}/{total_events} ({percentage:>5.1f}%)")
            
            if show_events_table:
                print(f"\nTABLEAU DÉTAILLÉ DES ÉVÉNEMENTS:")
                print("-" * 60)
                
                # Créer un DataFrame pour une meilleure visualisation
                events_data = []
                for i, (onset, duration, event_code) in enumerate(epochs.events):
                    event_name = None
                    for name, code in info['event_id'].items():
                        if code == event_code:
                            event_name = name
                            break
                    
                    events_data.append({
                        'Index': i,
                        'Onset_Sample': onset,
                        'Duration': duration,
                        'Event_Code': event_code,
                        'Event_Name': event_name or 'Unknown',
                        'Time_s': onset / info['sampling_freq']
                    })
                
                df_events = pd.DataFrame(events_data)
                
                # Afficher les premières et dernières lignes
                print("Premiers événements:")
                print(df_events.head(10).to_string(index=False))
                
                if len(df_events) > 20:
                    print("\n... (événements intermédiaires omis) ...\n")
                    print("Derniers événements:")
                    print(df_events.tail(10).to_string(index=False))
        
        else:
            print("\nAUCUN EVENT_ID TROUVÉ DANS CE FICHIER!")
        
        # Informations sur les canaux
        if show_summary:
            print(f"\nCANAUX ({len(info['ch_names'])}):")
            print("-" * 40)
            
            # Grouper par type de canal
            ch_types = {}
            for ch_name in info['ch_names']:
                ch_type = 'EEG' if ch_name.upper() not in ['EOG', 'ECG', 'EMG'] else ch_name.upper()[:3]
                if ch_type not in ch_types:
                    ch_types[ch_type] = []
                ch_types[ch_type].append(ch_name)
            
            for ch_type, channels in ch_types.items():
                print(f"  {ch_type}: {len(channels)} canaux")
                if len(channels) <= 10:
                    print(f"    {', '.join(channels)}")
                else:
                    print(f"    {', '.join(channels[:5])}, ..., {', '.join(channels[-5:])}")
        
        return info
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inspection du fichier {file_path}: {e}")
        return None


def compare_with_config(event_id_dict, protocol="PP"):
    """
    Compare les event_id trouvés avec ceux définis dans la configuration.
    
    Parameters
    ----------
    event_id_dict : dict
        Dictionnaire des event_id trouvés
    protocol : str
        Protocole utilisé
    """
    print(f"\nCOMPARAISON AVEC LA CONFIGURATION ({protocol.upper()}):")
    print("-" * 60)
    
    if protocol.upper() == "LG" and 'EVENT_ID_LG' in globals():
        config_events = EVENT_ID_LG
        print("Configuration LG trouvée:")
        for name, code in sorted(config_events.items()):
            found = "✓" if name in event_id_dict and event_id_dict[name] == code else "✗"
            print(f"  {found} {name:<25} : {code}")
    else:
        print("Pas de configuration spécifique trouvée pour ce protocole.")
        print("Event_id trouvés dans le fichier:")
        for name, code in sorted(event_id_dict.items()):
            print(f"  • {name:<25} : {code}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(
        description="Inspect event_id in EEG epoch files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Arguments pour spécifier un sujet
    parser.add_argument("--subject_id", type=str,
                        help="Subject ID (e.g., LAB1, TpAB19)")
    parser.add_argument("--group", type=str, choices=['controls', 'del', 'nodel'],
                        help="Subject group")
    parser.add_argument("--protocol", type=str, choices=['PP', 'LG'], default='PP',
                        help="Protocol type")
    
    # Argument pour spécifier directement un fichier
    parser.add_argument("--file_path", type=str,
                        help="Direct path to .fif file")
    
    # Options d'affichage
    parser.add_argument("--no_events_table", action="store_true",
                        help="Don't show detailed events table")
    parser.add_argument("--no_summary", action="store_true",
                        help="Don't show summary")
    parser.add_argument("--compare_config", action="store_true",
                        help="Compare with configuration")
    
    # Option pour lister tous les sujets
    parser.add_argument("--list_subjects", action="store_true",
                        help="List all available subjects by group")
    
    args = parser.parse_args()
    
    if args.list_subjects:
        print("\nSUJETS DISPONIBLES PAR GROUPE:")
        print("="*50)
        for group, subjects in ALL_SUBJECTS_GROUPS.items():
            print(f"\n{group.upper()} ({len(subjects)} sujets):")
            for i, subject in enumerate(subjects, 1):
                print(f"  {i:2d}. {subject}")
        return
    
    # Vérifier les arguments
    if not args.file_path and not (args.subject_id and args.group):
        parser.error("Vous devez spécifier soit --file_path, soit --subject_id et --group")
    
    files_to_inspect = []
    
    if args.file_path:
        # Fichier spécifique
        files_to_inspect = [args.file_path]
    else:
        # Rechercher les fichiers pour le sujet
        try:
            found_files = find_epoch_files(args.subject_id, args.group, args.protocol)
            if not found_files:
                logger.error(f"Aucun fichier trouvé pour le sujet {args.subject_id} "
                           f"(groupe: {args.group}, protocole: {args.protocol})")
                return
            files_to_inspect = found_files
            logger.info(f"Fichiers trouvés pour {args.subject_id}: {len(found_files)}")
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de fichiers: {e}")
            return
    
    # Inspecter chaque fichier
    for i, file_path in enumerate(files_to_inspect):
        if len(files_to_inspect) > 1:
            print(f"\n{'='*80}")
            print(f"FICHIER {i+1}/{len(files_to_inspect)}")
        
        info = inspect_epochs_file(
            file_path,
            show_events_table=not args.no_events_table,
            show_summary=not args.no_summary
        )
        
        if info and args.compare_config:
            compare_with_config(info['event_id'], args.protocol)
    
    print(f"\n{'='*80}")
    print("INSPECTION TERMINÉE")
    print("="*80)


if __name__ == "__main__":
    main()
