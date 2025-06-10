#!/usr/bin/env python3
"""
Script d'analyse rapide pour le projet Baking_EEG
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import ast


def quick_analysis():
    print("🔬 Analyse Rapide du Projet Baking_EEG")
    print("=" * 50)

    project_root = Path('/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG')

    # Créer le dossier de résultats
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)

    # Analyser les fichiers Python
    python_files = list(project_root.rglob("*.py"))
    print(f"📁 Fichiers Python trouvés: {len(python_files)}")

    # Collecte des données
    modules = {}
    total_lines = 0
    total_functions = 0

    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = len(content.split('\n'))
            total_lines += lines

            # Compte approximatif des fonctions
            func_count = content.count('def ')
            total_functions += func_count

            relative_path = py_file.relative_to(project_root)
            module_name = str(relative_path.with_suffix(''))

            modules[module_name] = {
                'lines': lines,
                'functions': func_count,
                'path': str(py_file)
            }

        except Exception as e:
            print(f"Erreur avec {py_file}: {e}")

    print(f"📊 Statistiques:")
    print(f"   • Modules: {len(modules)}")
    print(f"   • Fonctions estimées: {total_functions}")
    print(f"   • Lignes totales: {total_lines:,}")

    # Génération du graphique principal
    generate_summary_chart(modules, results_dir, total_lines, total_functions)

    # Génération du diagramme EEG
    generate_eeg_pipeline_diagram(results_dir)

    # Génération du rapport final
    generate_final_report(modules, results_dir, total_lines, total_functions)

    print(f"\n✅ Analyse terminée! Résultats dans: {results_dir}")


def generate_summary_chart(modules, output_dir, total_lines, total_functions):
    """Génère un graphique de résumé."""
    print("📈 Génération du graphique de résumé...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Analyse du Projet Baking_EEG - Vue d\'Ensemble',
                 fontsize=16, fontweight='bold')

    # 1. Top 10 des modules par taille
    sorted_modules = sorted(
        modules.items(), key=lambda x: x[1]['lines'], reverse=True)[:10]
    if sorted_modules:
        names, data = zip(*sorted_modules)
        sizes = [d['lines'] for d in data]

        bars = ax1.barh(range(len(names)), sizes, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Lignes de Code')
        ax1.set_title('Top 10 des Modules par Taille')
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels([n.split('/')[-1] for n in names])

        for i, (bar, size) in enumerate(zip(bars, sizes)):
            width = bar.get_width()
            ax1.text(width + max(sizes) * 0.01, bar.get_y() + bar.get_height()/2,
                     f'{size}', ha='left', va='center', fontsize=8)

    # 2. Distribution de la taille des modules
    module_sizes = [mod['lines'] for mod in modules.values()]
    ax2.hist(module_sizes, bins=15, color='lightcoral',
             alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('Lignes de Code')
    ax2.set_ylabel('Nombre de Modules')
    ax2.set_title('Distribution de la Taille des Modules')
    ax2.axvline(np.mean(module_sizes), color='blue', linestyle='--',
                label=f'Moyenne: {np.mean(module_sizes):.0f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Répartition par catégorie
    categories = {
        'Préprocessing': len([m for m in modules.keys() if 'preprocess' in m.lower()]),
        'Nettoyage': len([m for m in modules.keys() if 'clean' in m.lower()]),
        'Épochage': len([m for m in modules.keys() if 'epoch' in m.lower()]),
        'Décodage': len([m for m in modules.keys() if 'decoding' in m.lower()]),
        'Connectivité': len([m for m in modules.keys() if 'connect' in m.lower()]),
        'Utils': len([m for m in modules.keys() if 'utils' in m.lower()]),
        'Examples': len([m for m in modules.keys() if 'examples' in m.lower()]),
        'Autres': 0
    }

    # Calculer "Autres"
    categorized_count = sum(categories.values())
    categories['Autres'] = len(modules) - categorized_count

    # Filtrer les catégories avec des valeurs > 0
    categories = {k: v for k, v in categories.items() if v > 0}

    if categories:
        labels = list(categories.keys())
        values = list(categories.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        wedges, texts, autotexts = ax3.pie(values, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        ax3.set_title('Répartition des Modules par Catégorie')

    # 4. Métriques de qualité
    quality_scores = {
        'Modularité': min(100, len(modules) / max(1, np.mean(module_sizes)) * 20),
        'Lisibilité': min(100, 100 - np.std(module_sizes) / 5),
        'Structure': min(100, len([m for m in modules.keys() if any(x in m for x in ['utils', 'config', 'examples'])]) / len(modules) * 100),
        'Organisation': min(100, len(categories) * 15)
    }

    metrics = list(quality_scores.keys())
    scores = list(quality_scores.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax4.bar(metrics, scores, color=colors, alpha=0.8)
    ax4.set_ylabel('Score (%)')
    ax4.set_title('Métriques de Qualité Estimées')
    ax4.set_ylim(0, 100)
    ax4.axhline(y=70, color='green', linestyle='--',
                alpha=0.7, label='Seuil "Bon"')

    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{score:.0f}%', ha='center', va='bottom', fontsize=9)

    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "project_summary_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_eeg_pipeline_diagram(output_dir):
    """Génère un diagramme du pipeline EEG."""
    print("🧠 Génération du diagramme du pipeline EEG...")

    fig, ax = plt.subplots(figsize=(16, 10))

    # Définir les étapes du pipeline
    steps = [
        ("Données EEG\nBrutes", (1, 5), "lightcyan"),
        ("Préprocessing", (3, 5), "lightblue"),
        ("Nettoyage ICA", (5, 5), "lightgreen"),
        ("Épochage", (7, 5), "yellow"),
        ("Décodage", (9, 7), "orange"),
        ("Connectivité", (9, 3), "pink"),
        ("Spectral", (9, 1), "lightgray"),
        ("Résultats", (11, 5), "lightcoral")
    ]

    # Dessiner les boîtes
    for step_name, (x, y), color in steps:
        # Boîte
        rect = plt.Rectangle((x-0.7, y-0.5), 1.4, 1,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # Texte
        ax.text(x, y, step_name, ha='center', va='center',
                fontweight='bold', fontsize=10)

    # Dessiner les connexions
    connections = [
        ((1, 5), (3, 5)),   # Brutes -> Preprocessing
        ((3, 5), (5, 5)),   # Preprocessing -> ICA
        ((5, 5), (7, 5)),   # ICA -> Épochage
        ((7, 5), (9, 7)),   # Épochage -> Décodage
        ((7, 5), (9, 3)),   # Épochage -> Connectivité
        ((7, 5), (9, 1)),   # Épochage -> Spectral
        ((9, 7), (11, 5)),  # Décodage -> Résultats
        ((9, 3), (11, 5)),  # Connectivité -> Résultats
        ((9, 1), (11, 5))   # Spectral -> Résultats
    ]

    for (x1, y1), (x2, y2) in connections:
        ax.annotate('', xy=(x2-0.7, y2), xytext=(x1+0.7, y1),
                    arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

    # Configuration
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Pipeline de Traitement EEG - Projet Baking_EEG',
                 fontsize=16, fontweight='bold', pad=20)

    # Légende
    legend_text = """
    Pipeline de traitement des signaux électroencéphalographiques:
    
    1. Données brutes: Fichiers .bdf/.edf, 128 canaux, 512 Hz
    2. Préprocessing: Filtrage, référence, interpolation
    3. Nettoyage ICA: Suppression artéfacts et clignements
    4. Épochage: Fenêtres -0.2 à 1.2s, conditions PP/AP
    5. Analyses parallèles: Décodage (SVM), Connectivité, Spectral
    6. Résultats: Scores, graphiques, statistiques
    """

    ax.text(0.5, 1.5, legend_text, fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "eeg_pipeline_diagram.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_final_report(modules, output_dir, total_lines, total_functions):
    """Génère le rapport HTML final."""
    print("📝 Génération du rapport final...")

    # Calculer quelques statistiques
    avg_size = np.mean([mod['lines'] for mod in modules.values()])
    largest_module = max(modules.items(), key=lambda x: x[1]['lines'])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Analyse Complète - Projet Baking_EEG</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .content {{
                padding: 40px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2.5em;
                font-weight: bold;
                display: block;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }}
            .section h2 {{
                color: #2c3e50;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .image-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .image-container img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .highlight {{
                background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .footer {{
                background: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Analyse Complète</h1>
                <p>Projet Baking_EEG - Décodage de Signaux EEG</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-card">
                        <span class="stat-value">{len(modules)}</span>
                        <span>Modules</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">{total_functions}</span>
                        <span>Fonctions</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">{total_lines:,}</span>
                        <span>Lignes de Code</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-value">{avg_size:.0f}</span>
                        <span>Lignes/Module</span>
                    </div>
                </div>
                
                <div class="highlight">
                    <h3>🎯 Résumé de l'Analyse</h3>
                    <p><strong>Projet:</strong> Système complet de traitement et décodage de signaux électroencéphalographiques</p>
                    <p><strong>Architecture:</strong> Pipeline modulaire avec préprocessing, nettoyage, épochage et analyses multiples</p>
                    <p><strong>Module principal:</strong> {largest_module[0].split('/')[-1]} ({largest_module[1]['lines']} lignes)</p>
                    <p><strong>Couverture:</strong> Décodage, connectivité, analyse spectrale, exemples d'utilisation</p>
                </div>
                
                <div class="section">
                    <h2>📊 Vue d'Ensemble du Projet</h2>
                    <div class="image-container">
                        <img src="project_summary_analysis.png" alt="Résumé du Projet">
                        <p><em>Analyse générale: taille des modules, distribution, catégories et métriques de qualité</em></p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🧠 Pipeline de Traitement EEG</h2>
                    <div class="image-container">
                        <img src="eeg_pipeline_diagram.png" alt="Pipeline EEG">
                        <p><em>Flux de traitement des données: de l'acquisition aux résultats d'analyse</em></p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Fonctionnalités Principales</h2>
                    <ul>
                        <li><strong>Préprocessing:</strong> Filtrage, référencement, interpolation des canaux défaillants</li>
                        <li><strong>Nettoyage ICA:</strong> Suppression automatique des artéfacts oculaires et musculaires</li>
                        <li><strong>Épochage:</strong> Extraction de fenêtres temporelles autour des stimuli</li>
                        <li><strong>Décodage:</strong> Classification par SVM avec validation croisée</li>
                        <li><strong>Connectivité:</strong> Analyse des connexions fonctionnelles entre régions</li>
                        <li><strong>Analyse Spectrale:</strong> Décomposition en bandes de fréquences</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>💡 Points Forts du Code</h2>
                    <ul>
                        <li>Architecture modulaire bien structurée</li>
                        <li>Pipeline de traitement complet et cohérent</li>
                        <li>Utilisation de bibliothèques standards (MNE, scikit-learn)</li>
                        <li>Exemples d'utilisation fournis</li>
                        <li>Gestion de configurations flexibles</li>
                    </ul>
                </div>
                
                <div class="highlight">
                    <h3>🚀 Recommandations</h3>
                    <ul>
                        <li>Ajouter des tests unitaires pour les fonctions critiques</li>
                        <li>Améliorer la documentation des paramètres</li>
                        <li>Implémenter une interface utilisateur pour les analyses</li>
                        <li>Optimiser les performances pour les gros datasets</li>
                    </ul>
                </div>
            </div>
            
            <div class="footer">
                <p>Analyse générée le {__import__('datetime').datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>
                <p>Projet Baking_EEG - Analyse et Décodage de Signaux Électroencéphalographiques</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_dir / "analysis_report.html", 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    quick_analysis()
