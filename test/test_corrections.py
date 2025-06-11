#!/usr/bin/env python3
"""
Script de test rapide pour valider les corrections principales
"""

import sys
import os

# Ajouter le projet au PATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print("🔍 Test des corrections principales...")

# Test 1: Import des configurations
try:
    from config.decoding_config import CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
    print("✅ Import CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT - OK")
except ImportError as e:
    print(f"❌ Import CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT - FAILED: {e}")

# Test 2: Import de la fonction de stats corrigée
try:
    from utils.stats_utils import perform_pointwise_fdr_correction_on_scores
    print("✅ Import perform_pointwise_fdr_correction_on_scores - OK")
except ImportError as e:
    print(f"❌ Import perform_pointwise_fdr_correction_on_scores - FAILED: {e}")

# Test 3: Test de la fonction de stats avec des données simples
try:
    import numpy as np
    test_data = np.random.rand(5, 10) + 0.3  # Données de test
    result = perform_pointwise_fdr_correction_on_scores(
        test_data,
        chance_level=0.5,
        alternative_hypothesis="greater"
    )

    if result is not None and len(result) == 3:
        t_obs, fdr_mask, fdr_p = result
        print(
            f"✅ perform_pointwise_fdr_correction_on_scores - Retourne {len(result)} éléments")
        print(f"   - t_obs shape: {t_obs.shape}")
        print(f"   - fdr_mask shape: {fdr_mask.shape}")
        print(f"   - fdr_p shape: {fdr_p.shape}")
    else:
        print(
            f"❌ perform_pointwise_fdr_correction_on_scores - Retour invalide: {type(result)}")
except Exception as e:
    print(f"❌ Test perform_pointwise_fdr_correction_on_scores - FAILED: {e}")

# Test 4: Vérification que le script principal peut être importé
try:
    import examples.run_decoding_one_pp
    print("✅ Import du script principal run_decoding_one_pp - OK")
except ImportError as e:
    print(f"❌ Import du script principal - FAILED: {e}")

print("\n🎯 Tests terminés!")
