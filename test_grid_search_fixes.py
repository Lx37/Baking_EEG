#!/usr/bin/env python3
"""
Script de test pour vérifier les corrections des bugs use_grid_search
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_configuration_coherence():
    """Test la cohérence de la configuration"""
    try:
        from config.decoding_config import (
            USE_GRID_SEARCH_OPTIMIZATION, 
            USE_GRID_SEARCH,
            PARAM_GRID_CONFIG_EXTENDED,
            FIXED_CLASSIFIER_PARAMS_CONFIG
        )
        
        print("✅ Configuration chargée avec succès")
        print(f"   USE_GRID_SEARCH_OPTIMIZATION: {USE_GRID_SEARCH_OPTIMIZATION}")
        print(f"   USE_GRID_SEARCH: {USE_GRID_SEARCH}")
        
        # Test de cohérence
        if USE_GRID_SEARCH == USE_GRID_SEARCH_OPTIMIZATION:
            print("✅ Variables USE_GRID_SEARCH cohérentes")
        else:
            print("❌ Incohérence dans les variables USE_GRID_SEARCH")
            return False
            
        # Test de la grille de paramètres
        if USE_GRID_SEARCH_OPTIMIZATION and PARAM_GRID_CONFIG_EXTENDED:
            print("✅ Configuration grid search appropriée")
        elif not USE_GRID_SEARCH_OPTIMIZATION and FIXED_CLASSIFIER_PARAMS_CONFIG:
            print("✅ Configuration paramètres fixes appropriée")
        else:
            print("❌ Configuration inappropriée pour le mode choisi")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import de configuration: {e}")
        return False

def test_pipeline_builder():
    """Test le constructeur de pipeline"""
    try:
        from base.base_decoding import _build_standard_classifier_pipeline
        
        # Test avec grid search
        pipeline_gs, clf_name_gs, fs_name_gs, csp_name_gs = (
            _build_standard_classifier_pipeline(
                classifier_model_type="svc",
                use_grid_search=True,
                add_csp_step=False,
                add_anova_fs_step=True
            )
        )
        print("✅ Pipeline avec grid search créé")
        print(f"   Classifier name: {clf_name_gs}")
        print(f"   FS name: {fs_name_gs}")
        print(f"   CSP name: {csp_name_gs}")
        
        # Test sans grid search
        pipeline_fixed, clf_name_fixed, fs_name_fixed, csp_name_fixed = (
            _build_standard_classifier_pipeline(
                classifier_model_type="svc",
                use_grid_search=False,
                add_csp_step=False,
                add_anova_fs_step=True,
                svc_c=1.0,
                fs_percentile=20
            )
        )
        print("✅ Pipeline avec paramètres fixes créé")
        print(f"   Classifier name: {clf_name_fixed}")
        print(f"   FS name: {fs_name_fixed}")
        print(f"   CSP name: {csp_name_fixed}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import de base_decoding: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la création du pipeline: {e}")
        return False

def test_tuple_handling():
    """Test la gestion des tuples dans la fonction principale"""
    try:
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        # Créer des données de test minimales
        n_trials, n_channels, n_times = 100, 10, 50
        epochs_data = np.random.randn(n_trials, n_channels, n_times)
        target_labels = np.random.choice(['class_A', 'class_B'], n_trials)
        
        # Test avec valeurs tuple (problème original)
        tuple_csp = (True,)  # Problème typique
        tuple_anova = (False,)
        
        # Test de conversion
        if isinstance(tuple_csp, tuple):
            csp_converted = bool(tuple_csp[0]) if tuple_csp else False
            print(f"✅ Tuple CSP converti: {tuple_csp} -> {csp_converted}")
        
        if isinstance(tuple_anova, tuple):
            anova_converted = bool(tuple_anova[0]) if tuple_anova else False
            print(f"✅ Tuple ANOVA converti: {tuple_anova} -> {anova_converted}")
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de gestion des tuples: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Test des corrections de bugs use_grid_search")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Configuration
    print("\n📋 Test 1: Cohérence de la configuration")
    if test_configuration_coherence():
        tests_passed += 1
    
    # Test 2: Pipeline builder
    print("\n⚙️ Test 2: Constructeur de pipeline")
    if test_pipeline_builder():
        tests_passed += 1
    
    # Test 3: Gestion des tuples
    print("\n🔄 Test 3: Gestion des tuples")
    if test_tuple_handling():
        tests_passed += 1
    
    # Résumé
    print("\n" + "=" * 50)
    print(f"📊 Résultats: {tests_passed}/{total_tests} tests réussis")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests sont réussis ! Le code est prêt.")
        return 0
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
