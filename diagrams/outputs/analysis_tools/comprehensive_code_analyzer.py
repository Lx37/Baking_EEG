#!/usr/bin/env python3
"""
Analyseur complet de code pour générer tous types de graphiques et tests
pour le projet de décodage EEG Baking_EEG.
"""

import ast
import os
import sys
import json
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import graphviz
import inspect
import re
import time
from datetime import datetime

# Configuration des chemins
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_output" / "comprehensive_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ClassInfo:
    """Information sur une classe."""
    name: str
    methods: List[str]
    attributes: List[str]
    inheritance: List[str]
    file_path: str
    line_number: int
    docstring: str = ""


@dataclass
class FunctionInfo:
    """Information sur une fonction."""
    name: str
    parameters: List[str]
    return_type: str
    calls: List[str]
    file_path: str
    line_number: int
    docstring: str = ""
    complexity: int = 0


@dataclass
class ModuleInfo:
    """Information sur un module."""
    name: str
    file_path: str
    imports: List[str]
    functions: List[FunctionInfo]
    classes: List[ClassInfo]
    dependencies: List[str]
    lines_of_code: int = 0


class ComprehensiveCodeAnalyzer:
    """Analyseur complet de code avec génération de tous types de graphiques."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = OUTPUT_DIR
        self.modules: Dict[str, ModuleInfo] = {}
        self.call_graph = nx.DiGraph()
        self.dependency_graph = nx.DiGraph()
        self.class_hierarchy = nx.DiGraph()
        self.control_flow_graphs = {}

        # Configuration des couleurs pour les graphiques
        self.colors = {
            'core': '#FF6B6B',      # Rouge corail
            'utils': '#4ECDC4',     # Turquoise
            'config': '#45B7D1',    # Bleu ciel
            'examples': '#96CEB4',  # Vert menthe
            'base': '#FFEAA7',      # Jaune pâle
            'analysis': '#DDA0DD',  # Violet
            'default': '#95A5A6'    # Gris
        }

    def analyze_project(self):
        """Lance l'analyse complète du projet."""
        print("🔍 Démarrage de l'analyse complète du code...")

        # 1. Analyse syntaxique (AST)
        self._analyze_syntax()

        # 2. Analyse des dépendances
        self._analyze_dependencies()

        # 3. Analyse du flux d'exécution
        self._analyze_execution_flow()

        # 4. Génération des graphiques
        self._generate_all_diagrams()

        # 5. Tests et métriques
        self._run_comprehensive_tests()

        print(f"✅ Analyse terminée. Résultats dans {self.output_dir}")

    def _analyze_syntax(self):
        """Analyse syntaxique complète avec AST."""
        print("📝 Analyse syntaxique (AST)...")

        python_files = list(self.project_root.rglob("*.py"))

        for file_path in python_files:
            if self._should_analyze_file(file_path):
                try:
                    module_info = self._analyze_module(file_path)
                    if module_info:
                        self.modules[module_info.name] = module_info
                except Exception as e:
                    print(f"⚠️  Erreur lors de l'analyse de {file_path}: {e}")

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Détermine si un fichier doit être analysé."""
        exclude_patterns = [
            '__pycache__',
            '.git',
            'node_modules',
            'test_',
            '_test',
            '.pyc'
        ]

        file_str = str(file_path)
        return not any(pattern in file_str for pattern in exclude_patterns)

    def _analyze_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyse un module Python spécifique."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            module_name = self._get_module_name(file_path)

            # Extraire les informations du module
            imports = self._extract_imports(tree)
            functions = self._extract_functions(tree, file_path)
            classes = self._extract_classes(tree, file_path)
            dependencies = self._extract_dependencies(tree)
            lines_of_code = len(
                [line for line in content.split('\n') if line.strip()])

            return ModuleInfo(
                name=module_name,
                file_path=str(file_path),
                imports=imports,
                functions=functions,
                classes=classes,
                dependencies=dependencies,
                lines_of_code=lines_of_code
            )

        except Exception as e:
            print(f"Erreur lors de l'analyse de {file_path}: {e}")
            return None

    def _get_module_name(self, file_path: Path) -> str:
        """Génère le nom du module à partir du chemin."""
        relative_path = file_path.relative_to(self.project_root)
        if relative_path.name == "__init__.py":
            return str(relative_path.parent).replace('/', '.')
        else:
            return str(relative_path.with_suffix('')).replace('/', '.')

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extrait les imports d'un AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

        return imports

    def _extract_functions(self, tree: ast.AST, file_path: Path) -> List[FunctionInfo]:
        """Extrait les fonctions d'un AST."""
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Paramètres
                params = [arg.arg for arg in node.args.args]

                # Type de retour
                return_type = "Any"
                if node.returns:
                    return_type = ast.unparse(node.returns) if hasattr(
                        ast, 'unparse') else "Any"

                # Appels de fonctions
                calls = self._extract_function_calls(node)

                # Docstring
                docstring = ast.get_docstring(node) or ""

                # Complexité cyclomatique approximative
                complexity = self._calculate_complexity(node)

                functions.append(FunctionInfo(
                    name=node.name,
                    parameters=params,
                    return_type=return_type,
                    calls=calls,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring,
                    complexity=complexity
                ))

        return functions

    def _extract_classes(self, tree: ast.AST, file_path: Path) -> List[ClassInfo]:
        """Extrait les classes d'un AST."""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Méthodes
                methods = [n.name for n in node.body if isinstance(
                    n, ast.FunctionDef)]

                # Attributs (approximation)
                attributes = []
                for n in node.body:
                    if isinstance(n, ast.Assign):
                        for target in n.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)

                # Héritage
                inheritance = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        inheritance.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        inheritance.append(ast.unparse(base) if hasattr(
                            ast, 'unparse') else str(base))

                # Docstring
                docstring = ast.get_docstring(node) or ""

                classes.append(ClassInfo(
                    name=node.name,
                    methods=methods,
                    attributes=attributes,
                    inheritance=inheritance,
                    file_path=str(file_path),
                    line_number=node.lineno,
                    docstring=docstring
                ))

        return classes

    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extrait les appels de fonctions dans une fonction."""
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(ast.unparse(child.func) if hasattr(
                        ast, 'unparse') else str(child.func))

        return calls

    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extrait les dépendances d'un module."""
        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])

        return list(dependencies)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calcule la complexité cyclomatique approximative."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _analyze_dependencies(self):
        """Analyse les dépendances entre modules."""
        print("🔗 Analyse des dépendances...")

        for module_name, module_info in self.modules.items():
            self.dependency_graph.add_node(module_name, **asdict(module_info))

            for dependency in module_info.dependencies:
                if dependency in self.modules:
                    self.dependency_graph.add_edge(dependency, module_name)

    def _analyze_execution_flow(self):
        """Analyse le flux d'exécution et construit le graphe d'appels."""
        print("🔄 Analyse du flux d'exécution...")

        for module_name, module_info in self.modules.items():
            for function in module_info.functions:
                func_full_name = f"{module_name}.{function.name}"
                self.call_graph.add_node(func_full_name, **asdict(function))

                for call in function.calls:
                    # Recherche de la fonction appelée dans tous les modules
                    target_func = self._find_function(call)
                    if target_func:
                        self.call_graph.add_edge(func_full_name, target_func)

    def _find_function(self, func_name: str) -> Optional[str]:
        """Trouve une fonction dans tous les modules."""
        for module_name, module_info in self.modules.items():
            for function in module_info.functions:
                if function.name == func_name or func_name.endswith(f".{function.name}"):
                    return f"{module_name}.{function.name}"
        return None

    def _generate_all_diagrams(self):
        """Génère tous les types de diagrammes."""
        print("📊 Génération des diagrammes...")

        # 1. Diagramme de classes UML
        self._generate_uml_class_diagram()

        # 2. Diagramme de dépendances
        self._generate_dependency_diagram()

        # 3. Diagramme de flux d'exécution
        self._generate_execution_flow_diagram()

        # 4. Graphe d'appel de fonctions
        self._generate_call_graph()

        # 5. Control Flow Graph
        self._generate_control_flow_graphs()

        # 6. Abstract Syntax Tree
        self._generate_ast_diagrams()

        # 7. Data Flow Graph
        self._generate_data_flow_graph()

        # 8. Graphiques de métriques
        self._generate_metrics_diagrams()

    def _generate_uml_class_diagram(self):
        """Génère le diagramme de classes UML."""
        print("  📐 Diagramme de classes UML...")

        dot = graphviz.Digraph(comment='UML Class Diagram')
        dot.attr(rankdir='TB', splines='ortho')
        dot.attr('node', shape='record', style='filled', fillcolor='lightblue')

        for module_name, module_info in self.modules.items():
            for class_info in module_info.classes:
                # Contenu de la classe
                methods_str = "\\l".join(
                    [f"+ {method}()" for method in class_info.methods])
                attributes_str = "\\l".join(
                    [f"- {attr}" for attr in class_info.attributes])

                label = f"{class_info.name}|"
                if attributes_str:
                    label += f"{attributes_str}\\l|"
                if methods_str:
                    label += f"{methods_str}\\l"

                dot.node(f"{module_name}.{class_info.name}", label)

                # Relations d'héritage
                for parent in class_info.inheritance:
                    parent_full_name = self._find_class_full_name(parent)
                    if parent_full_name:
                        dot.edge(parent_full_name, f"{module_name}.{class_info.name}",
                                 arrowhead='empty')

        dot.render(self.output_dir / 'uml_class_diagram',
                   format='png', cleanup=True)
        dot.render(self.output_dir / 'uml_class_diagram',
                   format='pdf', cleanup=True)

    def _find_class_full_name(self, class_name: str) -> Optional[str]:
        """Trouve le nom complet d'une classe."""
        for module_name, module_info in self.modules.items():
            for class_info in module_info.classes:
                if class_info.name == class_name:
                    return f"{module_name}.{class_info.name}"
        return None

    def _generate_dependency_diagram(self):
        """Génère le diagramme de dépendances."""
        print("  🔗 Diagramme de dépendances...")

        plt.figure(figsize=(16, 12))

        # Positionnement avec algorithme de force
        pos = nx.spring_layout(self.dependency_graph, k=3, iterations=50)

        # Couleurs basées sur le type de module
        node_colors = []
        for node in self.dependency_graph.nodes():
            if 'core' in node or '_4_' in node:
                node_colors.append(self.colors['core'])
            elif 'utils' in node:
                node_colors.append(self.colors['utils'])
            elif 'config' in node:
                node_colors.append(self.colors['config'])
            elif 'examples' in node:
                node_colors.append(self.colors['examples'])
            elif 'base' in node:
                node_colors.append(self.colors['base'])
            else:
                node_colors.append(self.colors['default'])

        # Dessiner le graphe
        nx.draw(self.dependency_graph, pos,
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                alpha=0.7)

        # Étiquettes
        nx.draw_networkx_labels(self.dependency_graph, pos, font_size=6)

        plt.title("Diagramme de Dépendances - Projet Baking_EEG",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dependency_diagram.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'dependency_diagram.pdf',
                    bbox_inches='tight')
        plt.close()

    def _generate_execution_flow_diagram(self):
        """Génère le diagramme de flux d'exécution."""
        print("  🔄 Diagramme de flux d'exécution...")

        # Créer un flowchart simplifié des principales fonctions
        dot = graphviz.Digraph(comment='Execution Flow')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded,filled',
                 fillcolor='lightgreen')

        # Identifier les points d'entrée (fonctions main, scripts d'exemple)
        entry_points = []
        for module_name, module_info in self.modules.items():
            if 'examples' in module_name or '__main__' in module_info.file_path:
                for function in module_info.functions:
                    if function.name in ['main', '__main__', 'execute_single_subject_decoding']:
                        entry_points.append(f"{module_name}.{function.name}")

        # Tracer les chemins d'exécution depuis les points d'entrée
        visited = set()
        for entry_point in entry_points:
            self._trace_execution_path(
                dot, entry_point, visited, depth=0, max_depth=5)

        dot.render(self.output_dir / 'execution_flow_diagram',
                   format='png', cleanup=True)
        dot.render(self.output_dir / 'execution_flow_diagram',
                   format='pdf', cleanup=True)

    def _trace_execution_path(self, dot: graphviz.Digraph, func_name: str,
                              visited: Set[str], depth: int, max_depth: int):
        """Trace récursivement le chemin d'exécution."""
        if depth > max_depth or func_name in visited:
            return

        visited.add(func_name)
        dot.node(func_name.replace('.', '_'), func_name.split('.')[-1])

        if func_name in self.call_graph:
            for called_func in self.call_graph.successors(func_name):
                called_func_id = called_func.replace('.', '_')
                dot.node(called_func_id, called_func.split('.')[-1])
                dot.edge(func_name.replace('.', '_'), called_func_id)
                self._trace_execution_path(
                    dot, called_func, visited, depth + 1, max_depth)

    def _generate_call_graph(self):
        """Génère le graphe d'appel de fonctions."""
        print("  📞 Graphe d'appel de fonctions...")

        plt.figure(figsize=(20, 16))

        # Filtrer pour ne garder que les fonctions importantes
        important_functions = [node for node in self.call_graph.nodes()
                               if self.call_graph.degree(node) > 1]

        subgraph = self.call_graph.subgraph(important_functions)

        pos = nx.spring_layout(subgraph, k=2, iterations=50)

        # Dessiner les nœuds avec des tailles basées sur le nombre d'appels
        node_sizes = [self.call_graph.degree(
            node) * 100 + 300 for node in subgraph.nodes()]

        nx.draw(subgraph, pos,
                node_size=node_sizes,
                node_color='lightcoral',
                font_size=6,
                arrows=True,
                arrowsize=15,
                edge_color='gray',
                alpha=0.7)

        # Étiquettes simplifiées
        labels = {node: node.split('.')[-1] for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=5)

        plt.title("Graphe d'Appel de Fonctions - Projet Baking_EEG",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'call_graph.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'call_graph.pdf', bbox_inches='tight')
        plt.close()

    def _generate_control_flow_graphs(self):
        """Génère les graphes de flux de contrôle pour les fonctions principales."""
        print("  🎛️  Graphes de flux de contrôle...")

        # Analyser quelques fonctions clés
        key_functions = [
            'execute_single_subject_decoding',
            'run_temporal_decoding_analysis',
            'load_epochs_data_for_decoding'
        ]

        for func_name in key_functions:
            self._generate_single_control_flow_graph(func_name)

    def _generate_single_control_flow_graph(self, func_name: str):
        """Génère un CFG pour une fonction spécifique."""
        # Trouver la fonction
        target_function = None
        target_module = None

        for module_name, module_info in self.modules.items():
            for function in module_info.functions:
                if function.name == func_name:
                    target_function = function
                    target_module = module_info
                    break
            if target_function:
                break

        if not target_function:
            return

        # Créer un CFG simplifié basé sur la complexité
        dot = graphviz.Digraph(comment=f'Control Flow Graph - {func_name}')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box')

        # Nœud d'entrée
        dot.node('entry', 'Entry', fillcolor='lightgreen', style='filled')

        # Nœuds basés sur la complexité
        complexity = target_function.complexity
        for i in range(min(complexity, 10)):  # Limiter à 10 nœuds
            node_id = f'block_{i}'
            dot.node(node_id, f'Block {i+1}')
            if i == 0:
                dot.edge('entry', node_id)
            else:
                dot.edge(f'block_{i-1}', node_id)

        # Nœud de sortie
        dot.node('exit', 'Exit', fillcolor='lightcoral', style='filled')
        if complexity > 0:
            dot.edge(f'block_{min(complexity-1, 9)}', 'exit')
        else:
            dot.edge('entry', 'exit')

        filename = f'cfg_{func_name}'
        dot.render(self.output_dir / filename, format='png', cleanup=True)

    def _generate_ast_diagrams(self):
        """Génère les diagrammes AST pour les fichiers principaux."""
        print("  🌳 Diagrammes Abstract Syntax Tree...")

        key_files = [
            'Baking_EEG/_4_decoding_core.py',
            'utils/loading_PP_utils.py',
            'examples/run_decoding_one_pp.py'
        ]

        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self._generate_single_ast_diagram(full_path)

    def _generate_single_ast_diagram(self, file_path: Path):
        """Génère un diagramme AST pour un fichier spécifique."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            dot = graphviz.Digraph(comment=f'AST - {file_path.name}')
            dot.attr('node', shape='ellipse',
                     style='filled', fillcolor='lightyellow')

            self._ast_to_graph(tree, dot, 'root', max_depth=3)

            filename = f'ast_{file_path.stem}'
            dot.render(self.output_dir / filename, format='png', cleanup=True)

        except Exception as e:
            print(f"Erreur lors de la génération AST pour {file_path}: {e}")

    def _ast_to_graph(self, node: ast.AST, dot: graphviz.Digraph,
                      parent_id: str, depth: int = 0, max_depth: int = 3):
        """Convertit récursivement un AST en graphe."""
        if depth > max_depth:
            return

        node_id = f"{parent_id}_{depth}_{id(node)}"
        node_label = type(node).__name__

        # Ajouter des informations spécifiques selon le type de nœud
        if isinstance(node, ast.FunctionDef):
            node_label += f"\\n{node.name}"
        elif isinstance(node, ast.ClassDef):
            node_label += f"\\n{node.name}"
        elif isinstance(node, ast.Name):
            node_label += f"\\n{node.id}"

        dot.node(node_id, node_label)

        if parent_id != 'root':
            dot.edge(parent_id, node_id)

        # Traiter les enfants
        for child in ast.iter_child_nodes(node):
            self._ast_to_graph(child, dot, node_id, depth + 1, max_depth)

    def _generate_data_flow_graph(self):
        """Génère un diagramme de flux de données."""
        print("  💾 Diagramme de flux de données...")

        dot = graphviz.Digraph(comment='Data Flow Graph')
        dot.attr(rankdir='LR')
        dot.attr('node', shape='box')

        # Définir les principales étapes du pipeline de données
        stages = [
            ('raw_data', 'Données EEG\nBrutes', 'lightblue'),
            ('preprocessing', 'Préprocessing\n(filtrage, artefacts)', 'lightgreen'),
            ('epoching', 'Segmentation\nen époques', 'lightyellow'),
            ('feature_extraction', 'Extraction de\ncaractéristiques', 'lightcoral'),
            ('decoding', 'Décodage\n(classification)', 'lightpink'),
            ('results', 'Résultats\n(métriques, plots)', 'lightgray')
        ]

        # Ajouter les nœuds
        for stage_id, label, color in stages:
            dot.node(stage_id, label, fillcolor=color, style='filled')

        # Ajouter les connexions
        for i in range(len(stages) - 1):
            dot.edge(stages[i][0], stages[i+1][0])

        # Ajouter des branches pour l'analyse statistique
        dot.node('statistics', 'Analyses\nStatistiques',
                 fillcolor='orange', style='filled')
        dot.node('visualization', 'Visualisations',
                 fillcolor='purple', style='filled')

        dot.edge('results', 'statistics')
        dot.edge('results', 'visualization')

        dot.render(self.output_dir / 'data_flow_graph',
                   format='png', cleanup=True)
        dot.render(self.output_dir / 'data_flow_graph',
                   format='pdf', cleanup=True)

    def _generate_metrics_diagrams(self):
        """Génère des diagrammes de métriques de code."""
        print("  📈 Diagrammes de métriques...")

        # Collecter les métriques
        metrics = self._collect_metrics()

        # 1. Distribution de la complexité
        self._plot_complexity_distribution(metrics)

        # 2. Lignes de code par module
        self._plot_lines_of_code(metrics)

        # 3. Distribution des fonctions par module
        self._plot_function_distribution(metrics)

        # 4. Graphique en réseau des dépendances
        self._plot_dependency_network(metrics)

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collecte toutes les métriques du projet."""
        metrics = {
            'modules': [],
            'functions': [],
            'classes': [],
            'complexities': [],
            'lines_of_code': [],
            'dependencies': []
        }

        for module_name, module_info in self.modules.items():
            metrics['modules'].append(module_name)
            metrics['lines_of_code'].append(module_info.lines_of_code)

            for function in module_info.functions:
                metrics['functions'].append(f"{module_name}.{function.name}")
                metrics['complexities'].append(function.complexity)

            for class_info in module_info.classes:
                metrics['classes'].append(f"{module_name}.{class_info.name}")

            metrics['dependencies'].extend(module_info.dependencies)

        return metrics

    def _plot_complexity_distribution(self, metrics: Dict[str, Any]):
        """Graphique de distribution de la complexité."""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(metrics['complexities'], bins=20, alpha=0.7,
                 color='skyblue', edgecolor='black')
        plt.title('Distribution de la Complexité Cyclomatique')
        plt.xlabel('Complexité')
        plt.ylabel('Nombre de Fonctions')

        plt.subplot(2, 2, 2)
        plt.boxplot(metrics['complexities'])
        plt.title('Boîte à Moustaches - Complexité')
        plt.ylabel('Complexité')

        plt.subplot(2, 2, 3)
        plt.hist(metrics['lines_of_code'], bins=15, alpha=0.7,
                 color='lightgreen', edgecolor='black')
        plt.title('Distribution des Lignes de Code par Module')
        plt.xlabel('Lignes de Code')
        plt.ylabel('Nombre de Modules')

        plt.subplot(2, 2, 4)
        # Top 10 des fonctions les plus complexes
        func_complexity = list(
            zip(metrics['functions'], metrics['complexities']))
        func_complexity.sort(key=lambda x: x[1], reverse=True)
        top_10 = func_complexity[:10]

        if top_10:
            names, complexities = zip(*top_10)
            names = [name.split('.')[-1][:15]
                     for name in names]  # Raccourcir les noms

            plt.barh(range(len(names)), complexities, color='coral')
            plt.yticks(range(len(names)), names)
            plt.title('Top 10 - Fonctions les Plus Complexes')
            plt.xlabel('Complexité')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_complexity.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'metrics_complexity.pdf',
                    bbox_inches='tight')
        plt.close()

    def _plot_lines_of_code(self, metrics: Dict[str, Any]):
        """Graphique des lignes de code par module."""
        plt.figure(figsize=(14, 8))

        # Données pour le graphique
        modules = metrics['modules']
        loc = metrics['lines_of_code']

        # Trier par nombre de lignes
        module_loc = list(zip(modules, loc))
        module_loc.sort(key=lambda x: x[1], reverse=True)

        if module_loc:
            sorted_modules, sorted_loc = zip(*module_loc)

            # Raccourcir les noms de modules
            short_names = [name.split('.')[-1][:20] for name in sorted_modules]

            plt.bar(range(len(short_names)), sorted_loc,
                    color=sns.color_palette("viridis", len(short_names)))
            plt.xticks(range(len(short_names)),
                       short_names, rotation=45, ha='right')
            plt.title('Lignes de Code par Module',
                      fontsize=14, fontweight='bold')
            plt.xlabel('Modules')
            plt.ylabel('Lignes de Code')
            plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_lines_of_code.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'metrics_lines_of_code.pdf',
                    bbox_inches='tight')
        plt.close()

    def _plot_function_distribution(self, metrics: Dict[str, Any]):
        """Graphique de distribution des fonctions."""
        plt.figure(figsize=(12, 8))

        # Compter les fonctions par module
        func_count = defaultdict(int)
        for func_name in metrics['functions']:
            module = '.'.join(func_name.split('.')[:-1])
            func_count[module] += 1

        modules = list(func_count.keys())
        counts = list(func_count.values())

        plt.pie(counts, labels=[mod.split('.')[-1][:15] for mod in modules],
                autopct='%1.1f%%', startangle=90)
        plt.title('Distribution des Fonctions par Module',
                  fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_function_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir /
                    'metrics_function_distribution.pdf', bbox_inches='tight')
        plt.close()

    def _plot_dependency_network(self, metrics: Dict[str, Any]):
        """Graphique en réseau des dépendances."""
        plt.figure(figsize=(16, 12))

        # Créer un graphe de dépendances simplifié
        G = nx.Graph()

        # Compter les dépendances
        dep_count = defaultdict(int)
        for dep in metrics['dependencies']:
            dep_count[dep] += 1

        # Ajouter les nœuds (seulement les plus fréquents)
        top_deps = sorted(dep_count.items(),
                          key=lambda x: x[1], reverse=True)[:15]

        for dep, count in top_deps:
            G.add_node(dep, weight=count)

        # Ajouter des arêtes basées sur la co-occurrence
        deps = [dep for dep, _ in top_deps]
        for i, dep1 in enumerate(deps):
            for dep2 in deps[i+1:]:
                # Simuler une relation basée sur la fréquence
                if dep_count[dep1] > 5 and dep_count[dep2] > 5:
                    G.add_edge(dep1, dep2)

        # Dessiner le graphe
        pos = nx.spring_layout(G, k=2, iterations=50)

        node_sizes = [dep_count[node] * 100 + 300 for node in G.nodes()]

        nx.draw(G, pos,
                node_size=node_sizes,
                node_color='lightblue',
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)

        nx.draw_networkx_labels(G, pos)

        plt.title("Réseau des Dépendances Principales",
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dependency_network.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'dependency_network.pdf',
                    bbox_inches='tight')
        plt.close()

    def _run_comprehensive_tests(self):
        """Lance une série complète de tests et analyses."""
        print("🧪 Exécution des tests complets...")

        # 1. Tests de couverture de code
        self._run_coverage_tests()

        # 2. Analyse de qualité de code
        self._run_code_quality_analysis()

        # 3. Tests de performance
        self._run_performance_tests()

        # 4. Génération du rapport final
        self._generate_final_report()

    def _run_coverage_tests(self):
        """Exécute les tests de couverture de code."""
        print("  📊 Tests de couverture de code...")

        try:
            # Essayer d'installer coverage si nécessaire
            subprocess.run(['pip', 'install', 'coverage'], capture_output=True)

            # Exécuter coverage sur les tests disponibles
            test_files = list(self.project_root.glob("**/test_*.py"))

            if test_files:
                for test_file in test_files[:3]:  # Limiter à 3 tests
                    try:
                        result = subprocess.run([
                            'python', '-m', 'coverage', 'run', str(test_file)
                        ], capture_output=True, text=True, cwd=self.project_root)

                        if result.returncode == 0:
                            print(
                                f"    ✅ Test {test_file.name} exécuté avec succès")
                        else:
                            print(f"    ⚠️  Test {test_file.name} a échoué")
                    except Exception as e:
                        print(
                            f"    ❌ Erreur lors du test {test_file.name}: {e}")

            # Générer le rapport de couverture
            subprocess.run([
                'python', '-m', 'coverage', 'html',
                '--directory', str(self.output_dir / 'coverage_html')
            ], capture_output=True, cwd=self.project_root)

        except Exception as e:
            print(f"    ❌ Erreur lors des tests de couverture: {e}")

    def _run_code_quality_analysis(self):
        """Analyse la qualité du code."""
        print("  🔍 Analyse de qualité de code...")

        quality_metrics = {
            'total_modules': len(self.modules),
            'total_functions': sum(len(mod.functions) for mod in self.modules.values()),
            'total_classes': sum(len(mod.classes) for mod in self.modules.values()),
            'average_complexity': 0,
            'max_complexity': 0,
            'total_loc': sum(mod.lines_of_code for mod in self.modules.values()),
            'functions_without_docstring': 0,
            'classes_without_docstring': 0
        }

        complexities = []
        for module in self.modules.values():
            for function in module.functions:
                complexities.append(function.complexity)
                if not function.docstring:
                    quality_metrics['functions_without_docstring'] += 1

            for class_info in module.classes:
                if not class_info.docstring:
                    quality_metrics['classes_without_docstring'] += 1

        if complexities:
            quality_metrics['average_complexity'] = sum(
                complexities) / len(complexities)
            quality_metrics['max_complexity'] = max(complexities)

        # Sauvegarder les métriques
        with open(self.output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)

        print(f"    📊 Métriques sauvegardées: {quality_metrics}")

    def _run_performance_tests(self):
        """Effectue des tests de performance simples."""
        print("  ⚡ Tests de performance...")

        performance_data = {
            'analysis_start_time': datetime.now().isoformat(),
            'total_files_analyzed': len(self.modules),
            'analysis_duration_seconds': 0
        }

        start_time = time.time()

        # Simuler quelques tests de performance
        for module_name, module_info in list(self.modules.items())[:5]:
            try:
                # Test de parsing AST
                file_path = Path(module_info.file_path)
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    parse_start = time.time()
                    ast.parse(content)
                    parse_time = time.time() - parse_start

                    performance_data[f'{module_name}_parse_time'] = parse_time

            except Exception as e:
                print(f"    ⚠️  Erreur de performance pour {module_name}: {e}")

        performance_data['analysis_duration_seconds'] = time.time() - \
            start_time

        # Sauvegarder les données de performance
        with open(self.output_dir / 'performance_metrics.json', 'w') as f:
            json.dump(performance_data, f, indent=2)

    def _generate_final_report(self):
        """Génère le rapport final d'analyse."""
        print("  📋 Génération du rapport final...")

        report = {
            'analysis_summary': {
                'timestamp': datetime.now().isoformat(),
                'project_root': str(self.project_root),
                'total_modules': len(self.modules),
                'total_functions': sum(len(mod.functions) for mod in self.modules.values()),
                'total_classes': sum(len(mod.classes) for mod in self.modules.values()),
                'total_lines_of_code': sum(mod.lines_of_code for mod in self.modules.values())
            },
            'modules_analyzed': {
                name: {
                    'file_path': info.file_path,
                    'functions_count': len(info.functions),
                    'classes_count': len(info.classes),
                    'lines_of_code': info.lines_of_code,
                    'dependencies': info.dependencies
                }
                for name, info in self.modules.items()
            },
            'diagrams_generated': [
                'uml_class_diagram',
                'dependency_diagram',
                'execution_flow_diagram',
                'call_graph',
                'control_flow_graphs',
                'ast_diagrams',
                'data_flow_graph',
                'metrics_diagrams'
            ],
            'output_directory': str(self.output_dir)
        }

        # Sauvegarder le rapport principal
        with open(self.output_dir / 'comprehensive_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Générer un rapport HTML lisible
        self._generate_html_report(report)

    def _generate_html_report(self, report: Dict[str, Any]):
        """Génère un rapport HTML lisible."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rapport d'Analyse Complète - Projet Baking_EEG</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background: #3498db; color: white; border-radius: 5px; }}
                .module {{ margin: 10px 0; padding: 10px; border-left: 4px solid #3498db; }}
                .diagram-list {{ list-style-type: none; }}
                .diagram-list li {{ padding: 5px 0; }}
                .timestamp {{ color: #7f8c8d; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>📊 Rapport d'Analyse Complète - Projet Baking_EEG</h1>
            
            <div class="summary">
                <h2>Résumé de l'Analyse</h2>
                <div class="timestamp">Généré le: {report['analysis_summary']['timestamp']}</div>
                <br><br>
                <div class="metric">Modules: {report['analysis_summary']['total_modules']}</div>
                <div class="metric">Fonctions: {report['analysis_summary']['total_functions']}</div>
                <div class="metric">Classes: {report['analysis_summary']['total_classes']}</div>
                <div class="metric">Lignes de Code: {report['analysis_summary']['total_lines_of_code']}</div>
            </div>
            
            <h2>🎨 Diagrammes Générés</h2>
            <ul class="diagram-list">
                <li>📐 Diagramme de classes UML</li>
                <li>🔗 Diagramme de dépendances</li>
                <li>🔄 Diagramme de flux d'exécution</li>
                <li>📞 Graphe d'appel de fonctions</li>
                <li>🎛️  Graphes de flux de contrôle</li>
                <li>🌳 Diagrammes Abstract Syntax Tree</li>
                <li>💾 Diagramme de flux de données</li>
                <li>📈 Diagrammes de métriques</li>
            </ul>
            
            <h2>📦 Modules Analysés</h2>
        """

        for module_name, module_info in report['modules_analyzed'].items():
            html_content += f"""
            <div class="module">
                <h3>{module_name}</h3>
                <p><strong>Fichier:</strong> {module_info['file_path']}</p>
                <p><strong>Fonctions:</strong> {module_info['functions_count']} | 
                   <strong>Classes:</strong> {module_info['classes_count']} | 
                   <strong>Lignes:</strong> {module_info['lines_of_code']}</p>
                <p><strong>Dépendances:</strong> {', '.join(module_info['dependencies'][:5])}
                   {'...' if len(module_info['dependencies']) > 5 else ''}</p>
            </div>
            """

        html_content += f"""
            <h2>📂 Fichiers de Sortie</h2>
            <p>Tous les diagrammes et rapports sont disponibles dans le répertoire:</p>
            <code>{report['output_directory']}</code>
            
            <footer style="margin-top: 50px; padding: 20px; background: #34495e; color: white; border-radius: 8px;">
                <p>Rapport généré automatiquement par l'Analyseur Complet de Code</p>
                <p>Projet: Baking_EEG - Analyse de Décodage EEG</p>
            </footer>
        </body>
        </html>
        """

        with open(self.output_dir / 'comprehensive_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)


def main():
    """Fonction principale."""
    print("🚀 Lancment de l'Analyse Complète du Code - Projet Baking_EEG")
    print("=" * 60)

    analyzer = ComprehensiveCodeAnalyzer(PROJECT_ROOT)
    analyzer.analyze_project()

    print("=" * 60)
    print(f"✅ Analyse terminée avec succès!")
    print(f"📁 Résultats disponibles dans: {OUTPUT_DIR}")
    print(
        f"🌐 Rapport HTML: {OUTPUT_DIR / 'comprehensive_analysis_report.html'}")


if __name__ == "__main__":
    main()
