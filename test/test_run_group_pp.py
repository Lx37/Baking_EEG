#!/usr/bin/env python3
"""
Test script for run_decoding_one_group_pp.py
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


def test_group_script_imports():
    """Test les imports du script de groupe."""

    try:
        logger.info("🧪 Test des imports du script run_decoding_one_group_pp...")

        # Test import du script principal
        from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
        logger.info(
            "✅ Import de execute_group_intra_subject_decoding_analysis réussi")

        # Test des imports de configuration
        from config.decoding_config import ALL_SUBJECT_GROUPS
        logger.info("✅ Import des groupes de sujets réussi")
        logger.info(
            f"📋 Groupes disponibles: {list(ALL_SUBJECT_GROUPS.keys())}")

        # Test des fonctions de visualisation
        from utils.vizualization_utils import (
            plot_group_mean_scores_barplot,
            plot_group_temporal_decoding_statistics,
            plot_group_tgm_statistics
        )
        logger.info("✅ Import des fonctions de visualisation de groupe réussi")

        # Test des utils
        from utils.utils import configure_project_paths, setup_analysis_results_directory
        logger.info("✅ Import des fonctions utilitaires réussi")

        return True

    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erreur inattendue: {e}")
        return False


def test_group_function_signature():
    """Test la signature de la fonction principale."""

    try:
        from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
        import inspect

        signature = inspect.signature(
            execute_group_intra_subject_decoding_analysis)
        params = list(signature.parameters.keys())

        logger.info(
            "📋 Paramètres de execute_group_intra_subject_decoding_analysis:")
        for param in params:
            logger.info(f"  - {param}")

        # Vérifier que les paramètres essentiels sont présents
        required_params = [
            'subject_ids_in_group',
            'group_identifier',
            'save_results_flag',
            'generate_plots_flag'
        ]

        for req_param in required_params:
            if req_param in params:
                logger.info(f"✅ Paramètre requis '{req_param}' trouvé")
            else:
                logger.warning(f"⚠️ Paramètre requis '{req_param}' manquant")

        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors du test de signature: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Test du script run_decoding_one_group_pp")
    logger.info("=" * 50)

    success_imports = test_group_script_imports()
    logger.info("")

    success_signature = test_group_function_signature()
    logger.info("")

    if success_imports and success_signature:
        logger.info("✅ Tous les tests ont réussi!")
    else:
        logger.error("❌ Certains tests ont échoué")

    logger.info("✨ Test terminé!")
