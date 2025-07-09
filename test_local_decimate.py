#!/usr/bin/env python3
"""
Script de test simple pour vérifier le bon fonctionnement du test_decimate_folds.py
sans feature selection.
"""

import sys
import os

# Ajouter le chemin vers le projet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from examples.test_decimate_folds import (
    run_quick_test, 
    N_FOLDS_TO_TEST, 
    SAMPLING_FREQUENCIES,
    TEST_SUBJECT_FILE
)

def test_local():
    """Test local rapide."""
    print("=== TEST LOCAL RAPIDE ===")
    print(f"Fichier sujet: {TEST_SUBJECT_FILE}")
    print(f"Folds testés: {N_FOLDS_TO_TEST}")
    print(f"Fréquences testées: {SAMPLING_FREQUENCIES}")
    print("Feature selection: DÉSACTIVÉE")
    print(f"Nouveaux hauts folds: [20, 24, 28, 32, 36]")
    print(f"Fréquence 500 Hz ajoutée (pas de décimation)")
    print(f"Total combinaisons complètes: {len(N_FOLDS_TO_TEST) * len(SAMPLING_FREQUENCIES)}")
    print()
    
    try:
        output_dir = run_quick_test()
        print(f"✅ Test réussi! Résultats dans: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local()
    sys.exit(0 if success else 1)
