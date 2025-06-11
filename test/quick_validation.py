#!/usr/bin/env python3
"""
Test simple de validation du projet
"""

import sys
import os

# Configuration du chemin
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_basic_imports():
    """Test des imports de base."""
    print("🧪 Test des imports de base...")

    try:
        from config.config import ALL_SUBJECT_GROUPS
        print(f"✅ Groupes disponibles: {list(ALL_SUBJECT_GROUPS.keys())}")

        from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
        print("✅ Import de la fonction principale réussi")

        import inspect
        sig = inspect.signature(execute_group_intra_subject_decoding_analysis)
        print(f"📋 Fonction avec {len(sig.parameters)} paramètres")

        print("🎉 Tous les imports fonctionnent!")
        return True

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_visualization_imports():
    """Test des imports de visualisation."""
    print("\n🎨 Test des imports de visualisation...")

    try:
        from utils.vizualization_utils import (
            plot_group_mean_scores_barplot,
            plot_group_temporal_decoding_statistics,
            plot_group_tgm_statistics
        )
        print("✅ Fonctions de visualisation importées")

        from utils import stats_utils as bEEG_stats
        print("✅ Utilitaires statistiques importés")

        return True

    except Exception as e:
        print(f"❌ Erreur visualisation: {e}")
        return False


def test_lg_protocol():
    """Test du protocole Local-Global."""
    print("\n🧠 Test du protocole Local-Global...")

    try:
        from utils.loading_LG_utils import (
            load_epochs_data_for_lg_decoding,
            EVENTS_ID_LG
        )
        print("✅ Protocole LG importé")
        print(f"📋 Événements LG: {EVENTS_ID_LG}")

        return True

    except Exception as e:
        print(f"❌ Erreur protocole LG: {e}")
        return False


def main():
    """Fonction principale."""
    print("🚀 VALIDATION RAPIDE DU PROJET BAKING_EEG")
    print("=" * 50)

    tests = [
        ("Imports de base", test_basic_imports),
        ("Visualisations", test_visualization_imports),
        ("Protocole LG", test_lg_protocol)
    ]

    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    print("\n📊 RÉSULTATS")
    print("=" * 30)

    total = len(results)
    passed = sum(1 for _, success in results if success)

    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")

    print(f"\n📈 Score: {passed}/{total} tests réussis")

    if passed == total:
        print("🎉 VALIDATION RÉUSSIE!")
        print("Le projet est prêt à l'emploi.")
    else:
        print("⚠️ Certains tests ont échoué.")

    print("\n✨ Validation terminée!")


if __name__ == "__main__":
    main()
