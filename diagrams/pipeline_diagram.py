#!/usr/bin/env python3
"""
Visualisation du pipeline d'analyse EEG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os
import sys
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_eeg_pipeline_diagram():
    """Crée un diagramme du pipeline d'analyse EEG."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Définir les étapes du pipeline
    pipeline_steps = [
        {
            'name': 'Données EEG\nBrutes',
            'description': 'Acquisition des\nsignaux EEG',
            'color': '#ff6b6b',
            'position': (1, 9),
            'size': (1.5, 1)
        },
        {
            'name': 'Préprocessing\n(_1_preprocess.py)',
            'description': 'Filtrage, référencement,\nréjection d\'artefacts',
            'color': '#4ecdc4',
            'position': (1, 7),
            'size': (1.5, 1)
        },
        {
            'name': 'Nettoyage\n(_2_cleaning.py)',
            'description': 'ICA, détection\ncanaux défaillants',
            'color': '#45b7d1',
            'position': (1, 5),
            'size': (1.5, 1)
        },
        {
            'name': 'Epoching\n(_3_epoch.py)',
            'description': 'Segmentation\nautour des événements',
            'color': '#96ceb4',
            'position': (1, 3),
            'size': (1.5, 1)
        },
        {
            'name': 'Analyse Temporelle\n(Décodage)',
            'description': 'Classification\ntemporelle',
            'color': '#feca57',
            'position': (4, 7),
            'size': (1.8, 1)
        },
        {
            'name': 'Analyse Spectrale\n(_4_epoch_spectrum.py)',
            'description': 'Analyse fréquentielle\nPSD, TFR',
            'color': '#ff9ff3',
            'position': (4, 5),
            'size': (1.8, 1)
        },
        {
            'name': 'Connectivité\n(_4_connectivity.py)',
            'description': 'Analyses de\nconnectivité',
            'color': '#54a0ff',
            'position': (4, 3),
            'size': (1.8, 1)
        },
        {
            'name': 'Protocole PP\n(loading_PP_utils.py)',
            'description': 'Chargement données\nProtocole Predication',
            'color': '#5f27cd',
            'position': (7, 8),
            'size': (1.8, 0.8)
        },
        {
            'name': 'Protocole LG\n(loading_LG_utils.py)',
            'description': 'Chargement données\nLocal-Global',
            'color': '#00d2d3',
            'position': (7, 6.5),
            'size': (1.8, 0.8)
        },
        {
            'name': 'Core Decoding\n(_4_decoding_core.py)',
            'description': 'Pipeline de\nclassification ML',
            'color': '#ff6348',
            'position': (10, 7),
            'size': (1.8, 1)
        },
        {
            'name': 'Analyses Statistiques\n(stats_utils.py)',
            'description': 'Tests statistiques\nCorrections multiples',
            'color': '#2ed573',
            'position': (7, 4.5),
            'size': (1.8, 1)
        },
        {
            'name': 'Visualisations\n(vizualization_utils.py)',
            'description': 'Graphiques,\nDashboards',
            'color': '#ffa502',
            'position': (10, 4.5),
            'size': (1.8, 1)
        },
        {
            'name': 'Analyse Sujet Unique\n(run_decoding_one_pp.py)',
            'description': 'Pipeline complet\npour un sujet',
            'color': '#3742fa',
            'position': (13, 7),
            'size': (2, 1)
        },
        {
            'name': 'Analyse Groupe\n(run_decoding_one_group_pp.py)',
            'description': 'Analyse niveau\ngroupe + stats',
            'color': '#2f3542',
            'position': (13, 4.5),
            'size': (2, 1)
        }
    ]

    # Dessiner les boîtes pour chaque étape
    boxes = {}
    for step in pipeline_steps:
        x, y = step['position']
        w, h = step['size']

        # Créer une boîte avec coins arrondis
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=step['color'],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(box)

        # Ajouter le titre
        ax.text(
            x, y + 0.1, step['name'],
            ha='center', va='center',
            fontsize=9, fontweight='bold',
            wrap=True
        )

        # Ajouter la description
        ax.text(
            x, y - 0.2, step['description'],
            ha='center', va='center',
            fontsize=7,
            style='italic',
            wrap=True
        )

        boxes[step['name'].split('\n')[0]] = (x, y)

    # Définir les connexions
    connections = [
        ('Données EEG', 'Préprocessing'),
        ('Préprocessing', 'Nettoyage'),
        ('Nettoyage', 'Epoching'),
        ('Epoching', 'Analyse Temporelle'),
        ('Epoching', 'Analyse Spectrale'),
        ('Epoching', 'Connectivité'),
        ('Analyse Temporelle', 'Protocole PP'),
        ('Analyse Temporelle', 'Protocole LG'),
        ('Protocole PP', 'Core Decoding'),
        ('Protocole LG', 'Core Decoding'),
        ('Core Decoding', 'Analyse Sujet Unique'),
        ('Analyse Spectrale', 'Analyses Statistiques'),
        ('Connectivité', 'Analyses Statistiques'),
        ('Analyses Statistiques', 'Visualisations'),
        ('Visualisations', 'Analyse Groupe'),
        ('Analyse Sujet Unique', 'Analyse Groupe')
    ]

    # Dessiner les connexions
    for start, end in connections:
        if start in boxes and end in boxes:
            x1, y1 = boxes[start]
            x2, y2 = boxes[end]

            # Créer une flèche
            arrow = ConnectionPatch(
                (x1, y1), (x2, y2), "data", "data",
                arrowstyle="->", shrinkA=30, shrinkB=30,
                mutation_scale=15, fc="gray", alpha=0.7
            )
            ax.add_patch(arrow)

    # Configuration du graphique
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(1.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Titre
    ax.set_title(
        'Pipeline d\'Analyse EEG - Projet Baking_EEG',
        fontsize=18, fontweight='bold', pad=20
    )

    # Ajouter des annotations pour les phases
    phase_annotations = [
        {'text': 'PRÉTRAITEMENT', 'x': 1, 'y': 10, 'color': '#34495e'},
        {'text': 'ANALYSES', 'x': 4, 'y': 8.5, 'color': '#34495e'},
        {'text': 'PROTOCOLES', 'x': 7, 'y': 9, 'color': '#34495e'},
        {'text': 'CLASSIFICATION', 'x': 10, 'y': 8.5, 'color': '#34495e'},
        {'text': 'SCRIPTS FINAUX', 'x': 13, 'y': 8.5, 'color': '#34495e'}
    ]

    for annotation in phase_annotations:
        ax.text(
            annotation['x'], annotation['y'], annotation['text'],
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=annotation['color'],
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )

    plt.tight_layout()
    return fig


def create_protocol_comparison_diagram():
    """Crée un diagramme de comparaison des protocoles PP et LG."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Protocole PP (Predication Protocol)
    ax1.set_title('Protocole PP (Predication)', fontsize=14, fontweight='bold')

    pp_events = [
        {'name': 'PP/10', 'description': 'Prédiction Standard 1',
            'color': '#3498db', 'y': 3.5},
        {'name': 'PP/20', 'description': 'Prédiction Standard 2',
            'color': '#3498db', 'y': 3},
        {'name': 'PP/30', 'description': 'Prédiction Standard 3',
            'color': '#3498db', 'y': 2.5},
        {'name': 'AP/1X',
            'description': 'Famille AP 1 (X=1-6)', 'color': '#e74c3c', 'y': 2},
        {'name': 'AP/2X',
            'description': 'Famille AP 2 (X=1-6)', 'color': '#e74c3c', 'y': 1.5},
        {'name': 'AP/3X',
            'description': 'Famille AP 3 (X=1-6)', 'color': '#e74c3c', 'y': 1}
    ]

    for event in pp_events:
        # Boîte pour l'événement
        box = FancyBboxPatch(
            (0.1, event['y'] - 0.15), 2, 0.3,
            boxstyle="round,pad=0.05",
            facecolor=event['color'],
            alpha=0.7,
            edgecolor='black'
        )
        ax1.add_patch(box)

        ax1.text(1.1, event['y'], f"{event['name']}: {event['description']}",
                 ha='center', va='center', fontweight='bold', fontsize=10)

    ax1.set_xlim(0, 2.2)
    ax1.set_ylim(0.5, 4)
    ax1.axis('off')

    # Protocole LG (Local-Global)
    ax2.set_title('Protocole LG (Local-Global)',
                  fontsize=14, fontweight='bold')

    lg_events = [
        {'name': 'LS/GS (11)', 'description': 'Local Standard,\nGlobal Standard',
         'color': '#27ae60', 'y': 3.5},
        {'name': 'LS/GD (12)', 'description': 'Local Standard,\nGlobal Deviant',
         'color': '#f39c12', 'y': 3},
        {'name': 'LD/GS (21)', 'description': 'Local Deviant,\nGlobal Standard',
         'color': '#f39c12', 'y': 2},
        {'name': 'LD/GD (22)', 'description': 'Local Deviant,\nGlobal Deviant',
         'color': '#e74c3c', 'y': 1.5}
    ]

    for event in lg_events:
        # Boîte pour l'événement
        box = FancyBboxPatch(
            (0.1, event['y'] - 0.2), 2, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=event['color'],
            alpha=0.7,
            edgecolor='black'
        )
        ax2.add_patch(box)

        ax2.text(1.1, event['y'], f"{event['name']}: {event['description']}",
                 ha='center', va='center', fontweight='bold', fontsize=10)

    ax2.set_xlim(0, 2.2)
    ax2.set_ylim(1, 4)
    ax2.axis('off')

    plt.tight_layout()
    return fig


def main():
    """Fonction principale."""

    logger.info("🎨 Génération des diagrammes de pipeline EEG")

    # Créer le dossier de sortie
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Diagramme du pipeline principal
    logger.info("🔄 Génération du diagramme de pipeline...")
    fig_pipeline = create_eeg_pipeline_diagram()
    pipeline_path = os.path.join(output_dir, "eeg_pipeline_diagram.png")
    fig_pipeline.savefig(pipeline_path, dpi=300, bbox_inches='tight')
    logger.info(f"Pipeline sauvegardé: {pipeline_path}")

    # 2. Diagramme de comparaison des protocoles
    logger.info("📊 Génération du diagramme de protocoles...")
    fig_protocols = create_protocol_comparison_diagram()
    protocols_path = os.path.join(output_dir, "protocols_comparison.png")
    fig_protocols.savefig(protocols_path, dpi=300, bbox_inches='tight')
    logger.info(f"Comparaison protocoles sauvegardée: {protocols_path}")

    # Afficher les graphiques
    plt.show()

    logger.info("✅ Diagrammes de pipeline terminés!")


if __name__ == "__main__":
    main()
