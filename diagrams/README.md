# STRUCTURE DU DOSSIER DIAGRAMS

Ce dossier contient tous les outils de visualisation et d'analyse du projet Baking_EEG.

## Organisation

### 📁 Analysis Tools (Outils d'analyse)
- `ast_visualizer.py` - Visualisation de l'AST (Abstract Syntax Tree)
- `advanced_analysis.py` - Analyses avancées du code
- `flowchart_generator.py` - Génération de diagrammes de flux
- `uml_generator.py` - Génération de diagrammes UML

### 📁 Visualization Scripts (Scripts de visualisation)
- `project_structure_viz.py` - Visualisation de la structure du projet
- `dependency_graph.py` - Graphiques de dépendances
- `pipeline_diagram.py` - Diagrammes de pipeline d'analyse

### 📁 Output Directories (Répertoires de sortie)
- `analysis_output/` - Sorties des analyses
- `analysis_results/` - Résultats d'analyses
- `analysis_tools/` - Outils d'analyse spécialisés

## Utilisation

### Génération de diagrammes
```bash
python diagrams/flowchart_generator.py
python diagrams/uml_generator.py
python diagrams/project_structure_viz.py
```

### Analyses de code
```bash
python diagrams/ast_visualizer.py
python diagrams/advanced_analysis.py
python diagrams/dependency_graph.py
```

## Types de visualisations disponibles

1. **Diagrammes de flux** - Flux de traitement des données EEG
2. **Diagrammes UML** - Architecture des classes et modules
3. **Graphiques de dépendances** - Relations entre modules
4. **Structure du projet** - Organisation hiérarchique des fichiers
5. **Analyses AST** - Structure syntaxique du code

## Configuration

Les scripts de visualisation utilisent les bibliothèques suivantes :
- `matplotlib` - Graphiques de base
- `networkx` - Graphiques de réseau
- `graphviz` - Diagrammes structurés
- `plantuml` - Diagrammes UML (optionnel)

## Maintenance

Pour ajouter une nouvelle visualisation :
1. Créer le script dans le dossier approprié
2. Suivre les conventions de nommage
3. Documenter les paramètres et sorties
4. Mettre à jour ce README
