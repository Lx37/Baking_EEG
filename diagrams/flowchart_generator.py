#!/usr/bin/env python3
"""
Générateur de diagrammes de flux (flowcharts) pour le pipeline EEG
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import numpy as np
from pathlib import Path


def create_eeg_pipeline_flowchart():
    """Crée un diagramme de flux du pipeline EEG complet"""

    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Définir les étapes du pipeline
    steps = [
        {"name": "Données EEG Brutes", "type": "data", "pos": (7, 15)},
        {"name": "Préprocessing\n(_1_preprocess.py)",
         "type": "process", "pos": (7, 13.5)},
        {"name": "Nettoyage\n(_2_cleaning.py)",
         "type": "process", "pos": (7, 12)},
        {"name": "Segmentation en Epochs\n(_3_epoch.py)", "type": "process", "pos": (
            7, 10.5)},
        {"name": "Configuration\n(config.py)",
         "type": "config", "pos": (2, 9)},
        {"name": "Chargement des Données\n(loading_PP_utils.py)",
         "type": "process", "pos": (7, 9)},
        {"name": "Décodage Principal\n(_4_decoding_core.py)",
         "type": "process", "pos": (7, 7.5)},
        {"name": "Analyses Statistiques\n(stats_utils.py)", "type": "process", "pos": (
            12, 7.5)},
        {"name": "Connectivité\n(_4_connectivity.py)",
         "type": "process", "pos": (3, 6)},
        {"name": "Spectre des Epochs\n(_4_epoch_spectrum.py)",
         "type": "process", "pos": (11, 6)},
        {"name": "Décodage Cross-Subject\n(cross_subject_decoding.py)",
         "type": "process", "pos": (7, 4.5)},
        {"name": "Visualisation\n(vizualization_utils.py)",
         "type": "output", "pos": (3, 3)},
        {"name": "Résultats Finaux", "type": "output", "pos": (7, 3)},
        {"name": "Statistiques Globales\n(stat_decode_ALL.py)", "type": "output", "pos": (
            11, 3)}
    ]

    # Couleurs selon le type
    colors = {
        "data": "#E3F2FD",
        "process": "#E8F5E8",
        "config": "#FFF3E0",
        "output": "#FFEBEE"
    }

    # Dessiner les étapes
    boxes = {}
    for step in steps:
        x, y = step["pos"]
        step_type = step["type"]

        # Définir la forme selon le type
        if step_type == "data":
            # Forme de données (parallélogramme)
            width, height = 3, 1
            box = patches.Polygon([(x-width/2+0.3, y-height/2), (x+width/2, y-height/2),
                                   (x+width/2-0.3, y+height/2), (x-width/2, y+height/2)],
                                  closed=True, facecolor=colors[step_type], edgecolor='black')
        elif step_type == "process":
            # Rectangle pour processus
            width, height = 3, 1
            box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                 boxstyle="round,pad=0.1", facecolor=colors[step_type],
                                 edgecolor='black', linewidth=1.5)
        elif step_type == "config":
            # Losange pour configuration
            box = patches.RegularPolygon((x, y), 4, radius=0.8, orientation=np.pi/4,
                                         facecolor=colors[step_type], edgecolor='black')
        else:  # output
            # Forme de sortie (rectangle arrondi)
            width, height = 3, 1
            box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                                 boxstyle="round,pad=0.2", facecolor=colors[step_type],
                                 edgecolor='black', linewidth=2)

        ax.add_patch(box)
        boxes[step["name"]] = (x, y)

        # Ajouter le texte
        ax.text(x, y, step["name"], ha='center', va='center', fontsize=9,
                fontweight='bold' if step_type in ['data', 'output'] else 'normal')

    # Définir les connexions
    connections = [
        ("Données EEG Brutes", "Préprocessing\n(_1_preprocess.py)"),
        ("Préprocessing\n(_1_preprocess.py)", "Nettoyage\n(_2_cleaning.py)"),
        ("Nettoyage\n(_2_cleaning.py)", "Segmentation en Epochs\n(_3_epoch.py)"),
        ("Segmentation en Epochs\n(_3_epoch.py)",
         "Chargement des Données\n(loading_PP_utils.py)"),
        ("Configuration\n(config.py)", "Chargement des Données\n(loading_PP_utils.py)"),
        ("Chargement des Données\n(loading_PP_utils.py)",
         "Décodage Principal\n(_4_decoding_core.py)"),
        ("Décodage Principal\n(_4_decoding_core.py)",
         "Analyses Statistiques\n(stats_utils.py)"),
        ("Décodage Principal\n(_4_decoding_core.py)",
         "Connectivité\n(_4_connectivity.py)"),
        ("Décodage Principal\n(_4_decoding_core.py)",
         "Spectre des Epochs\n(_4_epoch_spectrum.py)"),
        ("Analyses Statistiques\n(stats_utils.py)",
         "Décodage Cross-Subject\n(cross_subject_decoding.py)"),
        ("Connectivité\n(_4_connectivity.py)",
         "Visualisation\n(vizualization_utils.py)"),
        ("Décodage Cross-Subject\n(cross_subject_decoding.py)", "Résultats Finaux"),
        ("Analyses Statistiques\n(stats_utils.py)",
         "Statistiques Globales\n(stat_decode_ALL.py)"),
        ("Résultats Finaux", "Visualisation\n(vizualization_utils.py)")
    ]

    # Dessiner les connexions
    for start, end in connections:
        if start in boxes and end in boxes:
            start_pos = boxes[start]
            end_pos = boxes[end]

            # Calculer les points de connexion
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            # Points d'ancrage
            if abs(dx) > abs(dy):  # Connexion horizontale
                start_point = (
                    start_pos[0] + (1.5 if dx > 0 else -1.5), start_pos[1])
                end_point = (
                    end_pos[0] + (-1.5 if dx > 0 else 1.5), end_pos[1])
            else:  # Connexion verticale
                start_point = (
                    start_pos[0], start_pos[1] + (-0.5 if dy < 0 else 0.5))
                end_point = (end_pos[0], end_pos[1] +
                             (0.5 if dy < 0 else -0.5))

            # Dessiner la flèche
            arrow = ConnectionPatch(start_point, end_point, "data", "data",
                                    arrowstyle="->", shrinkA=5, shrinkB=5,
                                    mutation_scale=20, fc="blue", ec="blue", alpha=0.7)
            ax.add_patch(arrow)

    # Ajouter une légende
    legend_elements = [
        patches.Rectangle(
            (0, 0), 1, 1, facecolor=colors["data"], edgecolor='black', label='Données'),
        patches.Rectangle(
            (0, 0), 1, 1, facecolor=colors["process"], edgecolor='black', label='Processus'),
        patches.Rectangle(
            (0, 0), 1, 1, facecolor=colors["config"], edgecolor='black', label='Configuration'),
        patches.Rectangle(
            (0, 0), 1, 1, facecolor=colors["output"], edgecolor='black', label='Sortie')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(0.98, 0.98))

    plt.title("Diagramme de Flux du Pipeline EEG - Projet Baking_EEG",
              fontsize=16, fontweight='bold', pad=20)

    return fig


def create_function_call_flowchart():
    """Crée un diagramme de flux des appels de fonctions principales"""

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Fonctions principales avec leurs appels
    functions = [
        {"name": "main()", "pos": (6, 9), "color": "#FF6B6B"},
        {"name": "load_epochs_data_for_decoding()", "pos": (3, 7.5),
         "color": "#4ECDC4"},
        {"name": "execute_single_subject_decoding()", "pos": (9, 7.5),
         "color": "#4ECDC4"},
        {"name": "perform_pointwise_fdr_correction()", "pos": (6, 6),
         "color": "#45B7D1"},
        {"name": "cross_subject_analysis()", "pos": (3, 4.5), "color": "#FFA07A"},
        {"name": "save_results()", "pos": (9, 4.5), "color": "#FFA07A"},
        {"name": "generate_plots()", "pos": (6, 3), "color": "#DDA0DD"}
    ]

    # Dessiner les fonctions
    for func in functions:
        x, y = func["pos"]
        width, height = 3.5, 0.8

        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=0.1", facecolor=func["color"],
                             edgecolor='black', linewidth=1.5, alpha=0.8)
        ax.add_patch(box)

        ax.text(x, y, func["name"], ha='center',
                va='center', fontsize=10, fontweight='bold')

    # Connexions entre fonctions
    connections = [
        ("main()", "load_epochs_data_for_decoding()"),
        ("main()", "execute_single_subject_decoding()"),
        ("execute_single_subject_decoding()",
         "perform_pointwise_fdr_correction()"),
        ("perform_pointwise_fdr_correction()", "cross_subject_analysis()"),
        ("perform_pointwise_fdr_correction()", "save_results()"),
        ("cross_subject_analysis()", "generate_plots()"),
        ("save_results()", "generate_plots()")
    ]

    func_positions = {func["name"]: func["pos"] for func in functions}

    for start, end in connections:
        start_pos = func_positions[start]
        end_pos = func_positions[end]

        arrow = ConnectionPatch(start_pos, end_pos, "data", "data",
                                arrowstyle="->", shrinkA=40, shrinkB=40,
                                mutation_scale=20, fc="darkblue", ec="darkblue", alpha=0.7)
        ax.add_patch(arrow)

    plt.title("Diagramme de Flux des Appels de Fonctions",
              fontsize=16, fontweight='bold')

    return fig


def create_data_flow_diagram():
    """Crée un diagramme de flux des données"""

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Éléments de données
    data_elements = [
        {"name": "Fichiers EEG\n(.fif, .set)", "pos": (
            2, 8.5), "type": "input"},
        {"name": "Configuration\n(JSON/Python)",
         "pos": (2, 6.5), "type": "config"},
        {"name": "Epochs\nPreprocessed", "pos": (5, 8.5), "type": "data"},
        {"name": "Features\nExtracted", "pos": (8, 8.5), "type": "data"},
        {"name": "Scores de\nDécodage", "pos": (11, 8.5), "type": "result"},
        {"name": "Matrices de\nConnectivité", "pos": (5, 6), "type": "data"},
        {"name": "Spectres de\nPuissance", "pos": (8, 6), "type": "data"},
        {"name": "P-values\nCorrigées", "pos": (11, 6), "type": "result"},
        {"name": "Graphiques\nde Visualisation",
            "pos": (5, 3.5), "type": "output"},
        {"name": "Rapports\nStatistiques", "pos": (8, 3.5), "type": "output"},
        {"name": "Résultats\nFinaux", "pos": (11, 3.5), "type": "output"}
    ]

    # Couleurs selon le type
    colors = {
        "input": "#FFCDD2",
        "config": "#FFF3E0",
        "data": "#E8F5E8",
        "result": "#E3F2FD",
        "output": "#F3E5F5"
    }

    # Dessiner les éléments
    for element in data_elements:
        x, y = element["pos"]
        width, height = 2.5, 1.2

        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                             boxstyle="round,pad=0.1", facecolor=colors[element["type"]],
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(box)

        ax.text(x, y, element["name"], ha='center',
                va='center', fontsize=9, fontweight='bold')

    # Flux de données
    flows = [
        ("Fichiers EEG\n(.fif, .set)", "Epochs\nPreprocessed"),
        ("Configuration\n(JSON/Python)", "Epochs\nPreprocessed"),
        ("Epochs\nPreprocessed", "Features\nExtracted"),
        ("Features\nExtracted", "Scores de\nDécodage"),
        ("Epochs\nPreprocessed", "Matrices de\nConnectivité"),
        ("Epochs\nPreprocessed", "Spectres de\nPuissance"),
        ("Scores de\nDécodage", "P-values\nCorrigées"),
        ("Matrices de\nConnectivité", "Graphiques\nde Visualisation"),
        ("Spectres de\nPuissance", "Rapports\nStatistiques"),
        ("P-values\nCorrigées", "Résultats\nFinaux")
    ]

    element_positions = {elem["name"]: elem["pos"] for elem in data_elements}

    for start, end in flows:
        start_pos = element_positions[start]
        end_pos = element_positions[end]

        arrow = ConnectionPatch(start_pos, end_pos, "data", "data",
                                arrowstyle="->", shrinkA=30, shrinkB=30,
                                mutation_scale=20, fc="green", ec="green", alpha=0.6, linewidth=2)
        ax.add_patch(arrow)

    plt.title("Diagramme de Flux des Données - Pipeline EEG",
              fontsize=16, fontweight='bold')

    return fig


def generate_all_flowcharts():
    """Génère tous les diagrammes de flux"""

    output_dir = Path(
        "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG/analysis_results")
    output_dir.mkdir(exist_ok=True)

    print("🔄 Génération des diagrammes de flux...")

    # Diagramme de flux du pipeline
    fig1 = create_eeg_pipeline_flowchart()
    plt.figure(fig1.number)
    plt.savefig(output_dir / "eeg_pipeline_flowchart.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "eeg_pipeline_flowchart.pdf", bbox_inches='tight')
    plt.close()

    # Diagramme des appels de fonctions
    fig2 = create_function_call_flowchart()
    plt.figure(fig2.number)
    plt.savefig(output_dir / "function_call_flowchart.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "function_call_flowchart.pdf",
                bbox_inches='tight')
    plt.close()

    # Diagramme de flux des données
    fig3 = create_data_flow_diagram()
    plt.figure(fig3.number)
    plt.savefig(output_dir / "data_flow_diagram.png",
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "data_flow_diagram.pdf", bbox_inches='tight')
    plt.close()

    print("✅ Tous les diagrammes de flux ont été générés!")
    print(f"📂 Emplacement: {output_dir}")
    print("   📊 eeg_pipeline_flowchart.png")
    print("   📊 function_call_flowchart.png")
    print("   📊 data_flow_diagram.png")


if __name__ == "__main__":
    print("🎯 GÉNÉRATION DE DIAGRAMMES DE FLUX")
    print("=" * 50)
    generate_all_flowcharts()
