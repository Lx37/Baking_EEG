#!/usr/bin/env python3
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
            'description': 'Tests d\'imports de base'
        },
        {
            'script': test_dir / 'core_tests' / 'test_loading_lg.py',
            'description': 'Test du protocole Local-Global'
        },
        {
            'script': test_dir / 'core_tests' / 'test_run_group_pp.py',
            'description': 'Test du script d\'analyse de groupe'
        },
        {
            'script': test_dir / 'validation' / 'simple_validation.py',
            'description': 'Validation simple du projet'
        },
        {
            'script': test_dir / 'demos' / 'demo_group_analysis.py',
            'description': 'Démonstration de l\'analyse de groupe'
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
    
    logger.info(f"\n📈 Score: {passed_tests}/{total_tests} tests réussis")
    
    if passed_tests == total_tests:
        logger.info("🎉 TOUS LES TESTS ONT RÉUSSI!")
    else:
        logger.warning("⚠️ Certains tests ont échoué.")
    
    logger.info("\n✨ Suite de tests terminée!")

if __name__ == "__main__":
    main()
