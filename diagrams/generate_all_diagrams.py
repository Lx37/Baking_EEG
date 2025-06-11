#!/usr/bin/env python3
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
            'description': 'Diagramme du pipeline d\'analyse EEG'
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
    
    logger.info(f"\n📈 Score: {successful_scripts}/{total_scripts} diagrammes générés")
    
    if successful_scripts == total_scripts:
        logger.info("🎉 TOUS LES DIAGRAMMES ONT ÉTÉ GÉNÉRÉS!")
    else:
        logger.warning("⚠️ Certains diagrammes ont échoué.")
    
    logger.info(f"\n📁 Résultats disponibles dans: {output_dir}")
    logger.info("\n✨ Génération de diagrammes terminée!")

if __name__ == "__main__":
    main()
