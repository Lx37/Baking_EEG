#!/usr/bin/env python3
"""
Script de test pour vérifier que tous les imports fonctionnent correctement
dans les scripts bash de soumission.
"""

import os
import sys

# Ajouter le répertoire parent au path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Baking_EEG"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_imports():
    """Test de tous les imports nécessaires."""
    print("=== Test des imports pour les scripts bash ===")

    errors = []

    # Test des imports de base
    try:
        import submitit
        print("✅ submitit: OK")
    except ImportError as e:
        errors.append(f"❌ submitit: {e}")

    # Test des imports depuis config
    try:
        from config.config import ALL_SUBJECT_GROUPS
        print(f"✅ ALL_SUBJECT_GROUPS: OK ({len(ALL_SUBJECT_GROUPS)} groupes)")
    except ImportError as e:
        errors.append(f"❌ ALL_SUBJECT_GROUPS: {e}")

    try:
        from config.decoding_config import (
            CLASSIFIER_MODEL_TYPE,
            USE_GRID_SEARCH_OPTIMIZATION,
            CONFIG_LOAD_SINGLE_PROTOCOL,
            CHANCE_LEVEL_AUC_SCORE
        )
        print("✅ config.decoding_config: OK")
    except ImportError as e:
        errors.append(f"❌ config.decoding_config: {e}")

    # Test des imports depuis utils
    try:
        from utils.utils import configure_project_paths
        print("✅ utils.utils.configure_project_paths: OK")
    except ImportError as e:
        errors.append(f"❌ utils.utils.configure_project_paths: {e}")

    # Test des imports depuis examples
    try:
        from examples.run_decoding_one_pp import execute_single_subject_decoding
        print("✅ examples.run_decoding_one_pp.execute_single_subject_decoding: OK")
    except ImportError as e:
        errors.append(
            f"❌ examples.run_decoding_one_pp.execute_single_subject_decoding: {e}")

    try:
        from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
        print("✅ examples.run_decoding_one_group_pp.execute_group_intra_subject_decoding_analysis: OK")
    except ImportError as e:
        errors.append(
            f"❌ examples.run_decoding_one_group_pp.execute_group_intra_subject_decoding_analysis: {e}")

    # Résumé
    print("\n=== Résumé des tests ===")
    if errors:
        print(f"❌ {len(errors)} erreurs détectées:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("✅ Tous les imports fonctionnent correctement!")
        return True


def test_configuration_values():
    """Test des valeurs de configuration."""
    print("\n=== Test des valeurs de configuration ===")

    try:
        from config.decoding_config import CONFIG_LOAD_SINGLE_PROTOCOL
        print(f"✅ CONFIG_LOAD_SINGLE_PROTOCOL: {CONFIG_LOAD_SINGLE_PROTOCOL}")
    except ImportError:
        print("❌ CONFIG_LOAD_SINGLE_PROTOCOL non trouvé")
        return False

    try:
        from config.config import ALL_SUBJECT_GROUPS
        for group, subjects in ALL_SUBJECT_GROUPS.items():
            print(f"   Groupe {group}: {len(subjects)} sujets")
    except ImportError:
        print("❌ ALL_SUBJECT_GROUPS non trouvé")
        return False

    return True


if __name__ == "__main__":
    print(f"Répertoire de script: {SCRIPT_DIR}")
    print(f"Racine du projet: {PROJECT_ROOT}")
    print(f"Python path: {sys.path[:3]}...")

    success_imports = test_imports()
    success_config = test_configuration_values()

    if success_imports and success_config:
        print("\n🎉 Tous les tests sont passés avec succès!")
        sys.exit(0)
    else:
        print("\n💥 Certains tests ont échoué!")
        sys.exit(1)
