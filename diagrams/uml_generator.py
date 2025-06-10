#!/usr/bin/env python3
"""
Générateur de diagrammes UML et graphiques avancés pour Baking_EEG
"""

import ast
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import networkx as nx
import json
from collections import defaultdict
import numpy as np


class UMLDiagramGenerator:
    """Générateur de diagrammes UML pour Python"""

    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "analysis_results"
        self.output_dir.mkdir(exist_ok=True)

    def analyze_classes_and_functions(self):
        """Analyse toutes les classes et fonctions du projet"""

        analysis_data = {
            'classes': {},
            'functions': {},
            'modules': {},
            'dependencies': defaultdict(list)
        }

        python_files = list(self.project_root.rglob("*.py"))

        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)
                relative_path = py_file.relative_to(self.project_root)
                module_name = str(relative_path.with_suffix(
                    '')).replace(os.sep, '.')

                # Analyser les classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_info = {
                            'module': module_name,
                            'methods': [],
                            'attributes': [],
                            'inheritance': [base.id for base in node.bases if isinstance(base, ast.Name)]
                        }

                        # Analyser les méthodes
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_info = {
                                    'name': item.name,
                                    'args': [arg.arg for arg in item.args.args],
                                    'decorators': [d.id for d in item.decorator_list if isinstance(d, ast.Name)]
                                }
                                class_info['methods'].append(method_info)
                            elif isinstance(item, ast.Assign):
                                for target in item.targets:
                                    if isinstance(target, ast.Name):
                                        class_info['attributes'].append(
                                            target.id)

                        analysis_data['classes'][f"{module_name}.{node.name}"] = class_info

                    # Analyser les fonctions (hors classe)
                    elif isinstance(node, ast.FunctionDef):
                        # Vérifier si c'est une fonction top-level
                        if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                            func_info = {
                                'module': module_name,
                                'args': [arg.arg for arg in node.args.args],
                                'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                            }
                            analysis_data['functions'][f"{module_name}.{node.name}"] = func_info

                    # Analyser les imports pour les dépendances
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis_data['dependencies'][module_name].append(
                                alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis_data['dependencies'][module_name].append(
                                node.module)

                analysis_data['modules'][module_name] = {
                    'path': str(py_file),
                    'lines': len(content.split('\n'))
                }

            except Exception as e:
                print(f"Erreur lors de l'analyse de {py_file}: {e}")

        return analysis_data

    def create_uml_class_diagram(self, analysis_data):
        """Crée un diagramme de classes UML"""

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # Couleurs pour différents types de modules
        colors = {
            'utils': '#E3F2FD',
            'examples': '#E8F5E8',
            'Baking_EEG': '#FFF3E0',
            'config': '#F3E5F5',
            'base': '#FFEBEE'
        }

        classes = analysis_data['classes']
        if not classes:
            # Si pas de classes, créer un diagramme de modules/fonctions
            return self.create_module_function_diagram(analysis_data)

        # Positionner les classes
        positions = {}
        x_offset = 1
        y_offset = 10

        for i, (class_name, class_info) in enumerate(classes.items()):
            module_parts = class_info['module'].split('.')
            base_module = module_parts[0] if module_parts else 'other'

            # Calculer position
            col = i % 4
            row = i // 4
            x = x_offset + col * 3.5
            y = y_offset - row * 2.5

            positions[class_name] = (x, y)

            # Choisir couleur selon le module
            color = colors.get(base_module, '#F5F5F5')

            # Créer le rectangle de la classe
            width = 3
            height = 2

            # Box principale
            class_box = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(class_box)

            # Nom de la classe
            simple_name = class_name.split('.')[-1]
            ax.text(x + width/2, y + height - 0.3, simple_name,
                    ha='center', va='center', fontweight='bold', fontsize=10)

            # Ligne de séparation
            ax.plot([x + 0.1, x + width - 0.1], [y + height - 0.6, y + height - 0.6],
                    'k-', linewidth=1)

            # Méthodes
            methods_text = ""
            # Limiter à 4 méthodes
            for j, method in enumerate(class_info['methods'][:4]):
                if j < 3:
                    methods_text += f"+ {method['name']}()\n"
                elif j == 3:
                    methods_text += f"+ ... ({len(class_info['methods'])} total)"
                    break

            if methods_text:
                ax.text(x + 0.2, y + height - 1.2, methods_text,
                        ha='left', va='top', fontsize=8)

        # Ajouter les relations d'héritage
        for class_name, class_info in classes.items():
            if class_info['inheritance'] and class_name in positions:
                for parent in class_info['inheritance']:
                    parent_full = f"{class_info['module']}.{parent}"
                    if parent_full in positions:
                        # Dessiner flèche d'héritage
                        start_pos = positions[class_name]
                        end_pos = positions[parent_full]

                        arrow = ConnectionPatch(
                            (start_pos[0] + 1.5, start_pos[1] + 2),
                            (end_pos[0] + 1.5, end_pos[1]),
                            "data", "data",
                            arrowstyle="->",
                            shrinkA=5, shrinkB=5,
                            mutation_scale=20,
                            fc="red", ec="red"
                        )
                        ax.add_patch(arrow)

        plt.title("Diagramme de Classes UML - Projet Baking_EEG",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "uml_class_diagram.png",
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "uml_class_diagram.pdf",
                    bbox_inches='tight')
        plt.close()

        return str(self.output_dir / "uml_class_diagram.png")

    def create_module_function_diagram(self, analysis_data):
        """Crée un diagramme des modules et fonctions"""

        fig, ax = plt.subplots(figsize=(20, 14))
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 14)
        ax.axis('off')

        modules = analysis_data['modules']
        functions = analysis_data['functions']

        # Grouper les fonctions par module
        module_functions = defaultdict(list)
        for func_name, func_info in functions.items():
            module_functions[func_info['module']].append(
                func_name.split('.')[-1])

        # Couleurs pour différents types de modules
        colors = {
            'utils': '#E3F2FD',
            'examples': '#E8F5E8',
            'Baking_EEG': '#FFF3E0',
            'config': '#F3E5F5',
            'base': '#FFEBEE'
        }

        # Positionner les modules
        x_offset = 1
        y_offset = 12

        for i, (module_name, module_info) in enumerate(modules.items()):
            if '__pycache__' in module_name or module_name.startswith('test_'):
                continue

            # Calculer position
            col = i % 5
            row = i // 5
            x = x_offset + col * 3.8
            y = y_offset - row * 3

            if y < 1:  # Éviter de sortir du cadre
                break

            # Déterminer la couleur selon le type de module
            base_module = module_name.split(
                '.')[0] if '.' in module_name else module_name.split('/')[0]
            color = colors.get(base_module, '#F5F5F5')

            # Calculer la taille selon le nombre de fonctions
            func_count = len(module_functions[module_name])
            height = max(2, min(2.5, 1.5 + func_count * 0.2))
            width = 3.5

            # Créer le rectangle du module
            module_box = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='navy',
                linewidth=2
            )
            ax.add_patch(module_box)

            # Nom du module
            display_name = module_name.split(
                '.')[-1] if '.' in module_name else os.path.basename(module_name)
            ax.text(x + width/2, y + height - 0.3, display_name,
                    ha='center', va='center', fontweight='bold', fontsize=9)

            # Ligne de séparation
            ax.plot([x + 0.1, x + width - 0.1], [y + height - 0.6, y + height - 0.6],
                    'navy', linewidth=1)

            # Fonctions du module
            functions_in_module = module_functions[module_name]
            functions_text = ""
            # Limiter à 6 fonctions
            for j, func_name in enumerate(functions_in_module[:6]):
                if j < 5:
                    functions_text += f"• {func_name}\n"
                elif j == 5:
                    functions_text += f"• ... (+{len(functions_in_module)-5})"
                    break

            if functions_text:
                ax.text(x + 0.2, y + height - 0.9, functions_text,
                        ha='left', va='top', fontsize=7)

            # Ajouter info sur les lignes de code
            lines_info = f"{module_info['lines']} lignes"
            ax.text(x + width - 0.2, y + 0.1, lines_info,
                    ha='right', va='bottom', fontsize=6, style='italic')

        plt.title("Diagramme des Modules et Fonctions - Projet Baking_EEG",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "module_function_diagram.png",
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "module_function_diagram.pdf",
                    bbox_inches='tight')
        plt.close()

        return str(self.output_dir / "module_function_diagram.png")

    def create_dependency_graph(self, analysis_data):
        """Crée un graphique des dépendances entre modules"""

        G = nx.DiGraph()
        dependencies = analysis_data['dependencies']

        # Ajouter les nœuds et arêtes
        for module, deps in dependencies.items():
            if '__pycache__' not in module:
                G.add_node(module)
                for dep in deps:
                    if not dep.startswith('_'):  # Ignorer les imports privés
                        G.add_edge(module, dep)

        # Créer le graphique
        plt.figure(figsize=(14, 10))

        # Layout avec spring pour une meilleure répartition
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Dessiner les nœuds avec des couleurs différentes
        node_colors = []
        for node in G.nodes():
            if 'utils' in node:
                node_colors.append('#FF6B6B')
            elif 'examples' in node:
                node_colors.append('#4ECDC4')
            elif 'Baking_EEG' in node:
                node_colors.append('#45B7D1')
            elif 'config' in node:
                node_colors.append('#FFA07A')
            else:
                node_colors.append('#DDA0DD')

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=1000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=20, alpha=0.6)

        # Ajouter les labels
        labels = {node: node.split(
            '.')[-1] if '.' in node else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title("Graphique des Dépendances entre Modules",
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "dependency_graph.png",
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "dependency_graph.pdf",
                    bbox_inches='tight')
        plt.close()

        return str(self.output_dir / "dependency_graph.png")

    def create_call_graph(self, analysis_data):
        """Crée un graphique des appels de fonctions"""

        # Pour cette version simplifiée, nous allons créer un graphique basé sur les imports
        plt.figure(figsize=(16, 12))

        # Créer un graphique circulaire des modules principaux
        modules = [name for name in analysis_data['modules'].keys()
                   if not ('__pycache__' in name or name.startswith('test_'))]

        main_modules = [mod for mod in modules if any(
            key in mod for key in ['utils', 'Baking_EEG', 'examples', 'config'])][:10]

        if main_modules:
            # Données pour le graphique circulaire
            sizes = [analysis_data['modules'][mod]['lines']
                     for mod in main_modules]
            colors = plt.cm.Set3(np.linspace(0, 1, len(main_modules)))

            # Créer le graphique en secteurs
            plt.pie(sizes, labels=main_modules, colors=colors,
                    autopct='%1.1f%%', startangle=90)
            plt.title("Répartition des Lignes de Code par Module",
                      fontsize=16, fontweight='bold')
        else:
            plt.text(0.5, 0.5, "Aucun module principal trouvé",
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Graphique des Appels de Fonctions",
                      fontsize=16, fontweight='bold')

        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(self.output_dir / "call_graph.png",
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / "call_graph.pdf", bbox_inches='tight')
        plt.close()

        return str(self.output_dir / "call_graph.png")

    def generate_all_diagrams(self):
        """Génère tous les diagrammes"""

        print("🔍 Analyse des classes et fonctions...")
        analysis_data = self.analyze_classes_and_functions()

        # Sauvegarder les données d'analyse
        with open(self.output_dir / "uml_analysis_data.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print("🎨 Génération des diagrammes...")

        diagrams = {}
        diagrams['uml_class'] = self.create_uml_class_diagram(analysis_data)
        diagrams['module_function'] = self.create_module_function_diagram(
            analysis_data)
        diagrams['dependency'] = self.create_dependency_graph(analysis_data)
        diagrams['call_graph'] = self.create_call_graph(analysis_data)

        print("📊 Statistiques de l'analyse:")
        print(f"   - Classes trouvées: {len(analysis_data['classes'])}")
        print(f"   - Fonctions trouvées: {len(analysis_data['functions'])}")
        print(f"   - Modules analysés: {len(analysis_data['modules'])}")

        return diagrams


if __name__ == "__main__":
    project_root = "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG"
    generator = UMLDiagramGenerator(project_root)

    print("🎯 GÉNÉRATION DE DIAGRAMMES UML ET GRAPHIQUES AVANCÉS")
    print("=" * 60)

    diagrams = generator.generate_all_diagrams()

    print(f"\n✅ Tous les diagrammes ont été générés avec succès!")
    print(f"📂 Emplacement: {generator.output_dir}")
    for name, path in diagrams.items():
        print(f"   📊 {name}: {os.path.basename(path)}")
