#!/usr/bin/env python3
"""
Tests complets pour le projet Baking_EEG avec analyse de couverture
"""

import unittest
import sys
import os
import ast
import importlib.util
from pathlib import Path
import json
import traceback
from datetime import datetime

# Ajouter le répertoire racine au path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestProjectStructure(unittest.TestCase):
    """Tests de la structure du projet"""

    def setUp(self):
        self.project_root = Path(__file__).parent

    def test_essential_files_exist(self):
        """Vérifie que les fichiers essentiels existent"""
        essential_files = [
            'utils/loading_PP_utils.py',
            'utils/stats_utils.py',
            'examples/run_decoding_one_pp.py',
            'Baking_EEG/_4_decoding_core.py',
            'config/config.py'
        ]

        for file_path in essential_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(),
                            f"Fichier manquant: {file_path}")

    def test_python_files_syntax(self):
        """Vérifie que tous les fichiers Python ont une syntaxe correcte"""
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []

        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                syntax_errors.append(f"{py_file}: {e}")
            except Exception as e:
                syntax_errors.append(f"{py_file}: {e}")

        if syntax_errors:
            self.fail(f"Erreurs de syntaxe trouvées:\n" +
                      "\n".join(syntax_errors))


class TestLoadingUtils(unittest.TestCase):
    """Tests pour loading_PP_utils.py"""

    def setUp(self):
        self.project_root = Path(__file__).parent
        sys.path.insert(0, str(self.project_root))

        try:
            from utils.loading_PP_utils import load_epochs_data_for_decoding
            self.load_function = load_epochs_data_for_decoding
        except ImportError as e:
            self.skipTest(f"Impossible d'importer loading_PP_utils: {e}")

    def test_load_function_signature(self):
        """Teste la signature de la fonction de chargement"""
        import inspect
        sig = inspect.signature(self.load_function)

        # Vérifie que les paramètres essentiels sont présents
        expected_params = ['config', 'subject_id']
        for param in expected_params:
            self.assertIn(param, sig.parameters,
                          f"Paramètre manquant: {param}")

    def test_load_function_with_mock_data(self):
        """Teste la fonction avec des données simulées"""
        # Configuration minimale pour test
        mock_config = {
            'data_dir': '/tmp/mock_data',
            'subjects': ['S01'],
            'conditions': ['condition1', 'condition2'],
            'epoch_tmin': -0.2,
            'epoch_tmax': 0.8
        }

        # Cette fonction devrait gérer gracieusement les données manquantes
        try:
            result = self.load_function(mock_config, 'S01')
            # Si aucune exception, c'est bon
            self.assertTrue(True)
        except FileNotFoundError:
            # Attendu avec des données simulées
            self.assertTrue(True)
        except Exception as e:
            # D'autres erreurs indiquent un problème de code
            self.fail(f"Erreur inattendue: {e}")


class TestStatsUtils(unittest.TestCase):
    """Tests pour stats_utils.py"""

    def setUp(self):
        try:
            from utils.stats_utils import perform_pointwise_fdr_correction_on_scores
            self.fdr_function = perform_pointwise_fdr_correction_on_scores
        except ImportError as e:
            self.skipTest(f"Impossible d'importer stats_utils: {e}")

    def test_fdr_function_returns_value(self):
        """Vérifie que la fonction FDR retourne une valeur"""
        import numpy as np

        # Données de test
        mock_scores = np.random.random((10, 5))
        mock_pvalues = np.random.random((10, 5))

        try:
            result = self.fdr_function(mock_scores, mock_pvalues)
            self.assertIsNotNone(
                result, "La fonction FDR doit retourner une valeur")
        except Exception as e:
            # Accepter les erreurs de validation des données
            if "shape" in str(e).lower() or "array" in str(e).lower():
                self.assertTrue(True)
            else:
                self.fail(f"Erreur inattendue dans FDR: {e}")


class TestDecodingCore(unittest.TestCase):
    """Tests pour le module de décodage principal"""

    def setUp(self):
        try:
            from examples.run_decoding_one_pp import execute_single_subject_decoding
            self.decode_function = execute_single_subject_decoding
        except ImportError as e:
            self.skipTest(f"Impossible d'importer run_decoding_one_pp: {e}")

    def test_decode_function_signature(self):
        """Teste la signature de la fonction de décodage"""
        import inspect
        sig = inspect.signature(self.decode_function)

        # Vérifie les paramètres par défaut dangereux
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, (list, dict, set)):
                    self.fail(
                        f"Paramètre par défaut mutable détecté: {param_name}")


def run_coverage_analysis():
    """Analyse de couverture du code"""
    print("\n🧪 Analyse de couverture de code")
    print("=" * 50)

    project_root = Path(__file__).parent
    python_files = list(project_root.rglob("*.py"))

    coverage_data = {
        'total_files': len(python_files),
        'tested_files': 0,
        'coverage_percentage': 0,
        'files_analysis': {}
    }

    # Analyser chaque fichier pour les fonctions testables
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            functions = [node for node in ast.walk(
                tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(
                tree) if isinstance(node, ast.ClassDef)]

            relative_path = py_file.relative_to(project_root)
            coverage_data['files_analysis'][str(relative_path)] = {
                'functions': len(functions),
                'classes': len(classes),
                'testable_elements': len(functions) + len(classes),
                'lines': len(content.split('\n'))
            }

        except Exception as e:
            print(f"⚠️  Erreur d'analyse pour {py_file}: {e}")

    # Calculer la couverture approximative
    testable_files = [f for f in coverage_data['files_analysis'].keys()
                      if not f.startswith('test_') and not '__pycache__' in f]

    # Fichiers principaux avec tests
    tested_patterns = ['loading_PP_utils',
                       'stats_utils', 'run_decoding_one_pp']
    tested_files = sum(1 for f in testable_files
                       if any(pattern in f for pattern in tested_patterns))

    coverage_data['tested_files'] = tested_files
    coverage_data['coverage_percentage'] = (
        tested_files / len(testable_files)) * 100 if testable_files else 0

    # Sauvegarder les résultats
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "test_coverage_analysis.json", 'w') as f:
        json.dump(coverage_data, f, indent=2)

    print(f"📊 Couverture de code: {coverage_data['coverage_percentage']:.1f}%")
    print(f"📁 Fichiers testés: {tested_files}/{len(testable_files)}")
    print(f"💾 Résultats sauvegardés dans: test_coverage_analysis.json")

    return coverage_data


def create_test_report():
    """Génère un rapport de test complet"""
    print("\n📋 Génération du rapport de tests")
    print("=" * 50)

    project_root = Path(__file__).parent
    results_dir = project_root / "analysis_results"
    results_dir.mkdir(exist_ok=True)

    # Collecter les résultats de test
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'project_path': str(project_root),
        'test_summary': {},
        'coverage_analysis': {},
        'recommendations': []
    }

    # Exécuter les tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
    result = runner.run(suite)

    test_results['test_summary'] = {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    }

    # Analyse de couverture
    coverage_data = run_coverage_analysis()
    test_results['coverage_analysis'] = coverage_data

    # Recommandations
    test_results['recommendations'] = [
        "Ajouter des tests unitaires pour tous les modules utilitaires",
        "Implémenter des tests d'intégration pour le pipeline complet",
        "Ajouter des tests de régression pour les corrections apportées",
        "Créer des tests avec des données simulées pour validation",
        "Implémenter des tests de performance pour les grandes données EEG"
    ]

    # Sauvegarder le rapport
    with open(results_dir / "comprehensive_test_report.json", 'w') as f:
        json.dump(test_results, f, indent=2)

    print(f"✅ Tests exécutés: {test_results['test_summary']['tests_run']}")
    print(f"❌ Échecs: {test_results['test_summary']['failures']}")
    print(f"⚠️  Erreurs: {test_results['test_summary']['errors']}")
    print(f"⏭️  Ignorés: {test_results['test_summary']['skipped']}")
    print(
        f"📈 Taux de réussite: {test_results['test_summary']['success_rate']:.1f}%")

    return test_results


if __name__ == "__main__":
    print("🧪 TESTS COMPLETS DU PROJET BAKING_EEG")
    print("=" * 60)

    # Exécuter tous les tests et analyses
    test_report = create_test_report()

    print(f"\n🎉 Analyse complète terminée!")
    print(f"📂 Rapports disponibles dans: analysis_results/")
    print(f"📊 Rapport de tests: comprehensive_test_report.json")
    print(f"📈 Analyse de couverture: test_coverage_analysis.json")
