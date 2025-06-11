#!/usr/bin/env python3
"""
Script de réorganisation et nettoyage des dossiers test et diagrams
"""

import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reorganize_test_folder(test_path):
    """Réorganise le dossier test avec une structure claire."""

    test_dir = Path(test_path)

    # Créer les sous-dossiers
    subdirs = {
        'core_tests': 'Tests principaux des fonctionnalités',
        'validation': 'Scripts de validation du projet',
        'demos': 'Scripts de démonstration',
        'utilities': 'Utilitaires de test et correction',
        'reports': 'Rapports et résultats de tests'
    }

    for subdir, description in subdirs.items():
        subdir_path = test_dir / subdir
        subdir_path.mkdir(exist_ok=True)

        # Créer un README pour chaque sous-dossier
        readme_content = f"# {subdir.upper()}\n\n{description}\n"
        (subdir_path / 'README.md').write_text(readme_content)

    # Mapping des fichiers vers leurs dossiers
    file_mapping = {
        'core_tests': [
            'test_simple_import.py',
            'test_loading_utils.py',
            'test_loading_lg.py',
            'test_run_group_pp.py',
            'test_import.py'
        ],
        'validation': [
            'simple_validation.py',
            'validate_project.py',
            'final_validation_complete.py',
            'validation_finale_simple.py'
        ],
        'demos': [
            'demo_final.py',
            'demo_group_analysis.py',
            'test_comprehensive.py'
        ],
        'utilities': [
            'fix_format.py',
            'fix_long_lines.py'
        ],
        'reports': [
            'final_validation_report.json',
            'validation_final_simple.json',
            'CORRECTIONS_SUMMARY.md',
            'RAPPORT_FINAL_CORRECTIONS.md',
            'RAPPORT_FINAL_VALIDATION.md',
            'INDEX_COMPLET.md'
        ]
    }

    # Déplacer les fichiers
    for target_dir, files in file_mapping.items():
        target_path = test_dir / target_dir

        for filename in files:
            source_file = test_dir / filename
            if source_file.exists():
                target_file = target_path / filename
                if not target_file.exists():
                    shutil.move(str(source_file), str(target_file))
                    logger.info(f"📁 Déplacé: {filename} → {target_dir}/")
                else:
                    logger.info(
                        f"⚠️ Fichier déjà présent: {target_dir}/{filename}")

    logger.info("✅ Réorganisation du dossier test terminée")


def reorganize_diagrams_folder(diagrams_path):
    """Réorganise le dossier diagrams avec une structure claire."""

    diagrams_dir = Path(diagrams_path)

    # Créer les sous-dossiers
    subdirs = {
        'generators': 'Scripts de génération de diagrammes',
        'analyzers': 'Outils d\'analyse de code',
        'outputs': 'Fichiers de sortie générés',
        'templates': 'Modèles de diagrammes'
    }

    for subdir, description in subdirs.items():
        subdir_path = diagrams_dir / subdir
        subdir_path.mkdir(exist_ok=True)

        # Créer un README pour chaque sous-dossier
        readme_content = f"# {subdir.upper()}\n\n{description}\n"
        (subdir_path / 'README.md').write_text(readme_content)

    # Mapping des fichiers vers leurs dossiers
    file_mapping = {
        'generators': [
            'flowchart_generator.py',
            'uml_generator.py',
            'project_structure_viz.py',
            'pipeline_diagram.py'
        ],
        'analyzers': [
            'ast_visualizer.py',
            'advanced_analysis.py'
        ],
        'outputs': [
            # Les dossiers de sortie existants seront déplacés ici
        ]
    }

    # Déplacer les fichiers
    for target_dir, files in file_mapping.items():
        target_path = diagrams_dir / target_dir

        for filename in files:
            source_file = diagrams_dir / filename
            if source_file.exists():
                target_file = target_path / filename
                if not target_file.exists():
                    shutil.move(str(source_file), str(target_file))
                    logger.info(f"📁 Déplacé: {filename} → {target_dir}/")

    # Déplacer les dossiers de sortie existants
    output_dirs = ['analysis_output', 'analysis_results', 'analysis_tools']
    outputs_path = diagrams_dir / 'outputs'

    for output_dir in output_dirs:
        source_dir = diagrams_dir / output_dir
        if source_dir.exists():
            target_dir = outputs_path / output_dir
            if not target_dir.exists():
                shutil.move(str(source_dir), str(target_dir))
                logger.info(f"📁 Déplacé dossier: {output_dir} → outputs/")

    logger.info("✅ Réorganisation du dossier diagrams terminée")


def create_master_test_script(test_path):
    """Crée un script maître pour lancer tous les tests."""

    master_script_content = '''#!/usr/bin/env python3
"""
Script maître pour lancer tous les tests du projet Baking_EEG
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_script(script_path, description):
    """Lance un script de test et retourne le résultat."""
    
    logger.info(f"🧪 {description}...")
    logger.info(f"   Script: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ SUCCÈS")
            return True
        else:
            logger.error("❌ ÉCHEC")
            logger.error(f"Erreur: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        return False

def main():
    """Fonction principale."""
    
    logger.info("🚀 SUITE DE TESTS COMPLÈTE - BAKING_EEG")
    logger.info("=" * 50)
    
    # Chemin du dossier test
    test_dir = Path(__file__).parent
    
    # Tests à exécuter dans l'ordre
    test_suite = [
        {
            'script': test_dir / 'core_tests' / 'test_simple_import.py',
            'description': 'Tests d\\'imports de base'
        },
        {
            'script': test_dir / 'core_tests' / 'test_loading_lg.py',
            'description': 'Test du protocole Local-Global'
        },
        {
            'script': test_dir / 'core_tests' / 'test_run_group_pp.py',
            'description': 'Test du script d\\'analyse de groupe'
        },
        {
            'script': test_dir / 'validation' / 'simple_validation.py',
            'description': 'Validation simple du projet'
        },
        {
            'script': test_dir / 'demos' / 'demo_group_analysis.py',
            'description': 'Démonstration de l\\'analyse de groupe'
        }
    ]
    
    # Exécuter les tests
    results = []
    
    for test_info in test_suite:
        script_path = test_info['script']
        description = test_info['description']
        
        if script_path.exists():
            success = run_test_script(script_path, description)
            results.append((description, success))
        else:
            logger.warning(f"⚠️ Script manquant: {script_path}")
            results.append((description, False))
        
        logger.info("-" * 40)
    
    # Résumé
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info("=" * 30)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for description, success in results:
        status = "✅" if success else "❌"
        logger.info(f"{status} {description}")
    
    logger.info(f"\\n📈 Score: {passed_tests}/{total_tests} tests réussis")
    
    if passed_tests == total_tests:
        logger.info("🎉 TOUS LES TESTS ONT RÉUSSI!")
    else:
        logger.warning("⚠️ Certains tests ont échoué.")
    
    logger.info("\\n✨ Suite de tests terminée!")

if __name__ == "__main__":
    main()
'''

    master_script_path = Path(test_path) / 'run_all_tests.py'
    master_script_path.write_text(master_script_content)

    # Rendre le script exécutable
    master_script_path.chmod(0o755)

    logger.info(f"✅ Script maître créé: {master_script_path}")


def create_master_diagram_script(diagrams_path):
    """Crée un script maître pour générer tous les diagrammes."""

    master_script_content = '''#!/usr/bin/env python3
"""
Script maître pour générer tous les diagrammes du projet Baking_EEG
"""

import sys
import os
import logging
import subprocess
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_diagram_script(script_path, description):
    """Lance un script de génération de diagramme."""
    
    logger.info(f"🎨 {description}...")
    logger.info(f"   Script: {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max
        )
        
        if result.returncode == 0:
            logger.info("✅ SUCCÈS")
            return True
        else:
            logger.error("❌ ÉCHEC")
            logger.error(f"Erreur: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("⏰ TIMEOUT")
        return False
    except Exception as e:
        logger.error(f"❌ ERREUR: {e}")
        return False

def main():
    """Fonction principale."""
    
    logger.info("🎨 GÉNÉRATION COMPLÈTE DE DIAGRAMMES - BAKING_EEG")
    logger.info("=" * 55)
    
    # Chemin du dossier diagrams
    diagrams_dir = Path(__file__).parent
    
    # Scripts à exécuter
    diagram_scripts = [
        {
            'script': diagrams_dir / 'generators' / 'project_structure_viz.py',
            'description': 'Visualisation de la structure du projet'
        },
        {
            'script': diagrams_dir / 'generators' / 'pipeline_diagram.py',
            'description': 'Diagramme du pipeline d\\'analyse EEG'
        },
        {
            'script': diagrams_dir / 'generators' / 'flowchart_generator.py',
            'description': 'Diagrammes de flux'
        },
        {
            'script': diagrams_dir / 'generators' / 'uml_generator.py',
            'description': 'Diagrammes UML'
        }
    ]
    
    # Créer le dossier de sortie principal
    output_dir = diagrams_dir / 'outputs' / 'all_diagrams'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Exécuter les scripts
    results = []
    
    for script_info in diagram_scripts:
        script_path = script_info['script']
        description = script_info['description']
        
        if script_path.exists():
            success = run_diagram_script(script_path, description)
            results.append((description, success))
        else:
            logger.warning(f"⚠️ Script manquant: {script_path}")
            results.append((description, False))
        
        logger.info("-" * 40)
    
    # Résumé
    logger.info("📊 RÉSUMÉ DE LA GÉNÉRATION")
    logger.info("=" * 35)
    
    total_scripts = len(results)
    successful_scripts = sum(1 for _, success in results if success)
    
    for description, success in results:
        status = "✅" if success else "❌"
        logger.info(f"{status} {description}")
    
    logger.info(f"\\n📈 Score: {successful_scripts}/{total_scripts} diagrammes générés")
    
    if successful_scripts == total_scripts:
        logger.info("🎉 TOUS LES DIAGRAMMES ONT ÉTÉ GÉNÉRÉS!")
    else:
        logger.warning("⚠️ Certains diagrammes ont échoué.")
    
    logger.info(f"\\n📁 Résultats disponibles dans: {output_dir}")
    logger.info("\\n✨ Génération de diagrammes terminée!")

if __name__ == "__main__":
    main()
'''

    master_script_path = Path(diagrams_path) / 'generate_all_diagrams.py'
    master_script_path.write_text(master_script_content)

    # Rendre le script exécutable
    master_script_path.chmod(0o755)

    logger.info(f"✅ Script maître créé: {master_script_path}")


def main():
    """Fonction principale de réorganisation."""

    logger.info("🔄 RÉORGANISATION DES DOSSIERS TEST ET DIAGRAMS")
    logger.info("=" * 55)

    # Chemins des dossiers
    script_dir = Path(__file__).parent.parent  # Retour au dossier racine
    test_path = script_dir / 'test'
    diagrams_path = script_dir / 'diagrams'

    logger.info(f"📁 Dossier test: {test_path}")
    logger.info(f"📁 Dossier diagrams: {diagrams_path}")

    # Réorganiser les dossiers
    if test_path.exists():
        logger.info("\\n🧪 Réorganisation du dossier test...")
        reorganize_test_folder(test_path)
        create_master_test_script(test_path)
    else:
        logger.warning("⚠️ Dossier test non trouvé")

    if diagrams_path.exists():
        logger.info("\\n🎨 Réorganisation du dossier diagrams...")
        reorganize_diagrams_folder(diagrams_path)
        create_master_diagram_script(diagrams_path)
    else:
        logger.warning("⚠️ Dossier diagrams non trouvé")

    logger.info("\\n✅ RÉORGANISATION TERMINÉE!")

    # Instructions pour l'utilisateur
    logger.info("\\n📋 INSTRUCTIONS D'UTILISATION:")
    logger.info("=" * 35)
    logger.info("🧪 Pour lancer tous les tests:")
    logger.info("   python test/run_all_tests.py")
    logger.info("\\n🎨 Pour générer tous les diagrammes:")
    logger.info("   python diagrams/generate_all_diagrams.py")
    logger.info("\\n📁 Structure claire et organisée créée!")


if __name__ == "__main__":
    main()
