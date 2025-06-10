#!/usr/bin/env python3
"""Test simple des imports critiques."""

import sys
import os

print("=== DIAGNOSTIC PYTHON ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")

print("\n=== TEST IMPORTS ===")

# Test 1: Import utils.stats_utils
try:
    from utils.stats_utils import perform_pointwise_fdr_correction_on_scores
    print("✅ utils.stats_utils - Import réussi")

    # Test de la fonction
    import numpy as np
    test_data = np.random.rand(5, 10)
    result = perform_pointwise_fdr_correction_on_scores(test_data)
    print(
        f"✅ perform_pointwise_fdr_correction_on_scores - Retourne {len(result)} éléments")

except Exception as e:
    print(f"❌ utils.stats_utils - Erreur: {e}")

# Test 2: Import Baking_EEG._4_decoding_core
try:
    from Baking_EEG._4_decoding_core import run_temporal_decoding_analysis
    print("✅ Baking_EEG._4_decoding_core - Import réussi")
except Exception as e:
    print(f"❌ Baking_EEG._4_decoding_core - Erreur: {e}")

# Test 3: Import vizualization_utils
try:
    from utils.vizualization_utils import create_subject_decoding_dashboard_plots
    import inspect
    sig = inspect.signature(create_subject_decoding_dashboard_plots)
    if 'chance_level_auc_score' in sig.parameters:
        print("✅ create_subject_decoding_dashboard_plots - Paramètre chance_level_auc_score présent")
    else:
        print("❌ create_subject_decoding_dashboard_plots - Paramètre chance_level_auc_score manquant")
except Exception as e:
    print(f"❌ utils.vizualization_utils - Erreur: {e}")

# Test 4: Import examples.run_decoding_one_pp
try:
    from examples.run_decoding_one_pp import parse_arguments
    print("✅ examples.run_decoding_one_pp - Import réussi")
except Exception as e:
    print(f"❌ examples.run_decoding_one_pp - Erreur: {e}")

print("\n=== TEST RÉPERTOIRES ===")
dirs = ['diagrams', 'figures', 'analysis_results',
        'utils', 'Baking_EEG', 'examples']
for d in dirs:
    if os.path.exists(d):
        print(f"✅ {d}/ - Répertoire présent")
    else:
        print(f"❌ {d}/ - Répertoire manquant")

print("\n=== FIN DU TEST ===")
