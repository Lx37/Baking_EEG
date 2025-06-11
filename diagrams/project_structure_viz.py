#!/usr/bin/env python3
"""
Visualisation de la structure du projet Baking_EEG
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import logging

# Configuration du chemin pour les imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_project_structure(root_path, max_depth=3):
    """Génère la structure du projet jusqu'à une profondeur donnée."""

    structure = {}
    root = Path(root_path)

    def scan_directory(path, current_depth=0):
        if current_depth >= max_depth:
            return {}

        items = {}
        try:
            for item in path.iterdir():
                if item.name.startswith('.'):
                    continue

                if item.is_dir():
                    items[item.name +
                          '/'] = scan_directory(item, current_depth + 1)
                else:
                    # Déterminer le type de fichier
                    if item.suffix == '.py':
                        items[item.name] = 'python'
                    elif item.suffix in ['.md', '.txt']:
                        items[item.name] = 'docs'
                    elif item.suffix in ['.json', '.yaml', '.yml']:
                        items[item.name] = 'config'
                    else:
                        items[item.name] = 'other'
        except PermissionError:
            pass

        return items

    return scan_directory(root)


def create_structure_visualization(structure, output_path=None):
    """Crée une visualisation de la structure du projet."""

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Couleurs pour différents types
    colors = {
        'python': '#3776ab',    # Bleu Python
        'docs': '#2e8b57',      # Vert pour documentation
        'config': '#ff6347',    # Rouge pour configuration
        'directory': '#9370db',  # Violet pour dossiers
        'other': '#808080'      # Gris pour autres
    }

    def draw_level(items, x_start=0, y_start=0, level=0, parent_width=1.0):
        """Dessine récursivement chaque niveau."""

        if not items:
            return y_start

        y_current = y_start
        item_height = 0.3
        indent = level * 0.1

        for name, content in items.items():
            # Déterminer le type et la couleur
            if name.endswith('/'):
                # C'est un dossier
                item_type = 'directory'
                display_name = name[:-1]  # Retirer le '/'

                # Dessiner le dossier
                rect = mpatches.Rectangle(
                    (x_start + indent, y_current),
                    parent_width - indent, item_height,
                    facecolor=colors[item_type],
                    alpha=0.7,
                    edgecolor='black'
                )
                ax.add_patch(rect)

                # Ajouter le texte
                ax.text(
                    x_start + indent + 0.02, y_current + item_height/2,
                    display_name,
                    fontsize=10 - level,
                    fontweight='bold',
                    va='center'
                )

                y_current -= item_height + 0.05

                # Dessiner récursivement le contenu
                if isinstance(content, dict):
                    y_current = draw_level(
                        content, x_start, y_current,
                        level + 1, parent_width * 0.9
                    )
            else:
                # C'est un fichier
                item_type = content if isinstance(content, str) else 'other'

                # Dessiner le fichier
                rect = mpatches.Rectangle(
                    (x_start + indent, y_current),
                    parent_width - indent, item_height * 0.7,
                    facecolor=colors.get(item_type, colors['other']),
                    alpha=0.5,
                    edgecolor='gray'
                )
                ax.add_patch(rect)

                # Ajouter le texte
                ax.text(
                    x_start + indent + 0.02, y_current + (item_height * 0.7)/2,
                    name,
                    fontsize=8 - level,
                    va='center'
                )

                y_current -= (item_height * 0.7) + 0.03

        return y_current

    # Dessiner la structure
    final_y = draw_level(structure, 0, 0, 0, 0.9)

    # Configuration du graphique
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(final_y - 0.5, 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Titre
    ax.set_title('Structure du Projet Baking_EEG',
                 fontsize=16, fontweight='bold', pad=20)

    # Légende
    legend_elements = [
        mpatches.Patch(color=colors['directory'], alpha=0.7, label='Dossiers'),
        mpatches.Patch(color=colors['python'],
                       alpha=0.5, label='Fichiers Python'),
        mpatches.Patch(color=colors['docs'], alpha=0.5, label='Documentation'),
        mpatches.Patch(color=colors['config'],
                       alpha=0.5, label='Configuration'),
        mpatches.Patch(color=colors['other'],
                       alpha=0.5, label='Autres fichiers')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Sauvegarder si un chemin est fourni
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualisation sauvegardée: {output_path}")

    return fig


def create_module_dependency_graph():
    """Crée un graphique des dépendances entre modules."""

    try:
        import networkx as nx

        # Créer un graphe dirigé
        G = nx.DiGraph()

        # Définir les modules principaux et leurs dépendances
        modules = {
            'config': [],
            'utils': ['config'],
            'base': ['utils', 'config'],
            'Baking_EEG': ['utils', 'config', 'base'],
            'examples': ['Baking_EEG', 'utils', 'config'],
            'test': ['examples', 'Baking_EEG', 'utils'],
            'diagrams': ['Baking_EEG', 'utils']
        }

        # Ajouter les nœuds et arêtes
        for module, deps in modules.items():
            G.add_node(module)
            for dep in deps:
                G.add_edge(dep, module)

        # Créer la visualisation
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Calculer la disposition
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Couleurs pour les différents types de modules
        node_colors = {
            'config': '#ff6347',     # Rouge pour configuration
            'utils': '#3776ab',      # Bleu pour utilitaires
            'base': '#9370db',       # Violet pour base
            'Baking_EEG': '#228b22',  # Vert pour core
            'examples': '#ffa500',   # Orange pour exemples
            'test': '#dc143c',       # Rouge foncé pour tests
            'diagrams': '#4682b4'    # Bleu acier pour diagrammes
        }

        # Dessiner les nœuds
        for node in G.nodes():
            nx.draw_networkx_nodes(
                G, pos, nodelist=[node],
                node_color=node_colors.get(node, '#808080'),
                node_size=2000,
                alpha=0.8,
                ax=ax
            )

        # Dessiner les arêtes
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            alpha=0.6,
            ax=ax
        )

        # Ajouter les labels
        nx.draw_networkx_labels(G, pos, font_size=10,
                                font_weight='bold', ax=ax)

        ax.set_title('Graphique des Dépendances entre Modules',
                     fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        return fig

    except ImportError:
        logger.warning(
            "NetworkX non disponible pour le graphique de dépendances")
        return None


def main():
    """Fonction principale."""

    logger.info("🎨 Génération des visualisations du projet Baking_EEG")

    # Chemin de sortie
    output_dir = Path(SCRIPT_DIR) / "analysis_output"
    output_dir.mkdir(exist_ok=True)

    # 1. Structure du projet
    logger.info("📁 Génération de la structure du projet...")
    structure = get_project_structure(PROJECT_ROOT, max_depth=3)

    fig_structure = create_structure_visualization(
        structure,
        output_dir / "project_structure.png"
    )

    # 2. Graphique de dépendances
    logger.info("🔗 Génération du graphique de dépendances...")
    fig_deps = create_module_dependency_graph()

    if fig_deps:
        fig_deps.savefig(
            output_dir / "module_dependencies.png",
            dpi=300, bbox_inches='tight'
        )
        logger.info(
            f"Graphique de dépendances sauvegardé: {output_dir / 'module_dependencies.png'}")

    # Afficher les graphiques
    plt.show()

    logger.info("✅ Visualisations terminées!")


if __name__ == "__main__":
    main()
