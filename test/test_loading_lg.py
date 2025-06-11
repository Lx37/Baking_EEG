#!/usr/bin/env python3
"""
Script de test pour le chargement des données Local-Global (LG)
"""

import sys
import os
import logging

# Configuration du chemin pour les imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_lg_loading():
    """Test la fonction de chargement des données LG."""
    
    try:
        from utils.loading_LG_utils import (
            load_epochs_data_for_lg_decoding,
            CONFIG_LOAD_MAIN_LG_DECODING,
            CONFIG_LOAD_LG_COMPARISONS,
            EVENTS_ID_LG
        )
        logger.info("✅ Import des fonctions LG réussi")
        
        # Affichage des configurations
        logger.info("📋 Configuration principale LG:")
        for key, value in CONFIG_LOAD_MAIN_LG_DECODING.items():
            logger.info(f"  - {key}: {value}")
            
        logger.info("📋 Comparaisons LG spécifiques:")
        for key, value in CONFIG_LOAD_LG_COMPARISONS.items():
            logger.info(f"  - {key}: {value}")
            
        logger.info("📋 Event IDs LG:")
        for key, value in EVENTS_ID_LG.items():
            logger.info(f"  - {key}: {value}")
            
        # Test avec des paramètres fictifs
        logger.info("🧪 Test de la fonction de chargement...")
        
        # Ces chemins n'existent probablement pas, mais ça teste la validation
        test_result = load_epochs_data_for_lg_decoding(
            subject_identifier="TEST_SUBJECT",
            group_affiliation="controls",
            base_input_data_path="/tmp/non_existent_path",
            conditions_to_load=CONFIG_LOAD_MAIN_LG_DECODING,
            verbose_logging=True
        )
        
        epochs, data_dict = test_result
        
        if epochs is None and isinstance(data_dict, dict):
            logger.info("✅ Fonction de validation fonctionne (pas de données trouvées, comme attendu)")
        else:
            logger.info("⚠️ Résultat inattendu du test")
            
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du test: {e}")

def show_lg_protocol_info():
    """Affiche les informations sur le protocole Local-Global."""
    
    logger.info("🧠 PROTOCOLE LOCAL-GLOBAL (LG)")
    logger.info("=" * 50)
    logger.info("Le protocole Local-Global teste deux niveaux de traitement:")
    logger.info("  • LOCAL: Comparaison de sons individuels")
    logger.info("    - LS (Local Standard): Son standard local")
    logger.info("    - LD (Local Deviant): Son déviant local")
    logger.info("  • GLOBAL: Comparaison de séquences")
    logger.info("    - GS (Global Standard): Séquence standard globale")
    logger.info("    - GD (Global Deviant): Séquence déviante globale")
    logger.info("")
    logger.info("📊 Comparaisons principales:")
    logger.info("  1. LS vs LD (effet local)")
    logger.info("  2. GS vs GD (effet global)")
    logger.info("  3. Interactions Local × Global")
    logger.info("")
    logger.info("🎯 Conditions spécifiques:")
    logger.info("  • LS/GS (11): Local Standard, Global Standard")
    logger.info("  • LS/GD (12): Local Standard, Global Deviant")
    logger.info("  • LD/GS (21): Local Deviant, Global Standard")
    logger.info("  • LD/GD (22): Local Deviant, Global Deviant")

if __name__ == "__main__":
    logger.info("🚀 Test du module loading_LG_utils")
    logger.info("=" * 50)
    
    show_lg_protocol_info()
    logger.info("")
    test_lg_loading()
    
    logger.info("")
    logger.info("✨ Test terminé!")
