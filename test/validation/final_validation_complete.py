#!/usr/bin/env python3
"""
Validation finale co        ("examples.run_decoding_one_pp", "Main example script"),
        ("Baking_EEG._4_decoding_core", "Core decoding module"),
        ("utils.vizualization_utils", "Visualization utilities"),ète du projet Baking_EEG.
Ce script vérifie que toutes les corrections ont été appliquées avec succès.
"""

import os
import sys
import traceback
import subprocess
import json
from datetime import datetime

# Configuration du répertoire de travail
PROJECT_ROOT = "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG"
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def print_header(title):
    """Affiche un en-tête formaté."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_success(message):
    """Affiche un message de succès."""
    print(f"✅ {message}")


def print_error(message):
    """Affiche un message d'erreur."""
    print(f"❌ {message}")


def print_warning(message):
    """Affiche un message d'avertissement."""
    print(f"⚠️  {message}")


def print_info(message):
    """Affiche un message d'information."""
    print(f"ℹ️  {message}")


def test_imports():
    """Teste tous les imports critiques."""
    print_header("Test des imports critiques")

    tests = [
        ("utils.loading_PP_utils", "Loading utilities"),
        ("utils.stats_utils", "Statistics utilities"),
        ("examples.run_decoding_one_pp", "Main decoding script"),
        ("Baking_EEG._4_decoding", "Core decoding module"),
        ("utils.vizualization_utils", "Visualization utilities"),
    ]

    results = []
    for module, description in tests:
        try:
            __import__(module)
            print_success(f"{description} - Import réussi")
            results.append((module, True, None))
        except Exception as e:
            print_error(f"{description} - Import échoué: {str(e)}")
            results.append((module, False, str(e)))

    return results


def test_function_calls():
    """Teste les appels de fonctions critiques."""
    print_header("Test des fonctions critiques")

    results = []

    # Test 1: stats_utils.perform_pointwise_fdr_correction_on_scores
    try:
        from utils.stats_utils import perform_pointwise_fdr_correction_on_scores
        import numpy as np

        # Test avec des données factices
        test_scores = np.random.rand(10, 100)  # 10 subjects, 100 time points
        result = perform_pointwise_fdr_correction_on_scores(
            test_scores, chance_level=0.5)

        # La fonction retourne maintenant 3 éléments: (t_obs, fdr_mask, fdr_corrected_p_values)
        if result is not None and len(result) == 3:
            print_success(
                "perform_pointwise_fdr_correction_on_scores - Fonction corrigée (retourne 3 éléments)")
            results.append(("fdr_correction", True, None))
        else:
            print_error(
                f"perform_pointwise_fdr_correction_on_scores - Problème de retour: {len(result) if result else 'None'} éléments")
            results.append(("fdr_correction", False, "Invalid return"))
    except Exception as e:
        print_error(
            f"perform_pointwise_fdr_correction_on_scores - Erreur: {str(e)}")
        results.append(("fdr_correction", False, str(e)))

    # Test 2: create_subject_decoding_dashboard_plots parameters
    try:
        from utils.vizualization_utils import create_subject_decoding_dashboard_plots
        import inspect

        # Vérifier la signature de la fonction
        sig = inspect.signature(create_subject_decoding_dashboard_plots)
        params = list(sig.parameters.keys())

        if 'chance_level_auc_score' in params:
            print_success(
                "create_subject_decoding_dashboard_plots - Paramètre chance_level_auc_score présent")
            results.append(("dashboard_params", True, None))
        else:
            print_error(
                "create_subject_decoding_dashboard_plots - Paramètre chance_level_auc_score manquant")
            results.append(("dashboard_params", False, "Missing parameter"))
    except Exception as e:
        print_error(
            f"create_subject_decoding_dashboard_plots - Erreur: {str(e)}")
        results.append(("dashboard_params", False, str(e)))

    return results


def test_code_execution():
    """Teste l'exécution du script principal."""
    print_header("Test d'exécution du script principal")

    try:
        # Test du help du script principal
        result = subprocess.run(
            [sys.executable, "examples/run_decoding_one_pp.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print_success("Script principal - Help menu accessible")
            return ("main_script", True, None)
        else:
            print_error(f"Script principal - Erreur: {result.stderr}")
            return ("main_script", False, result.stderr)
    except Exception as e:
        print_error(f"Script principal - Exception: {str(e)}")
        return ("main_script", False, str(e))


def analyze_code_quality():
    """Analyse la qualité du code."""
    print_header("Analyse de qualité du code")

    files_to_check = [
        "utils/loading_PP_utils.py",
        "utils/stats_utils.py",
        "examples/run_decoding_one_pp.py"
    ]

    results = []

    for file_path in files_to_check:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Compter les lignes corrigées
            lines = content.split('\n')
            long_lines = [i+1 for i,
                          line in enumerate(lines) if len(line) > 79]

            if len(long_lines) == 0:
                print_success(
                    f"{file_path} - Toutes les lignes respectent PEP 8 (<= 79 chars)")
                results.append((file_path, True, "PEP8 compliant"))
            else:
                print_warning(
                    f"{file_path} - {len(long_lines)} lignes dépassent 79 caractères")
                results.append(
                    (file_path, False, f"{len(long_lines)} long lines"))
        else:
            print_error(f"{file_path} - Fichier non trouvé")
            results.append((file_path, False, "File not found"))

    return results


def check_analysis_tools():
    """Vérifie les outils d'analyse créés."""
    print_header("Vérification des outils d'analyse")

    tools = [
        "analysis_tools/comprehensive_code_analyzer.py",
        "analysis_tools/simple_code_analyzer.py",
        "analysis_tools/quick_analyzer.py",
        "flowchart_generator.py",
        "uml_generator.py",
        "test_comprehensive.py"
    ]

    results = []
    existing_tools = 0

    for tool in tools:
        if os.path.exists(tool):
            print_success(f"Outil disponible: {tool}")
            existing_tools += 1
            results.append((tool, True, "Available"))
        else:
            print_warning(f"Outil manquant: {tool}")
            results.append((tool, False, "Missing"))

    print_info(f"Total: {existing_tools}/{len(tools)} outils disponibles")
    return results


def check_generated_outputs():
    """Vérifie les fichiers de sortie générés."""
    print_header("Vérification des fichiers générés")

    output_dirs = [
        "analysis_results",
        "diagrams",
        "figures"
    ]

    results = []
    total_files = 0

    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = []
            for root, dirs, filenames in os.walk(output_dir):
                files.extend([os.path.join(root, f) for f in filenames])

            total_files += len(files)
            print_success(f"{output_dir}/ - {len(files)} fichiers générés")
            results.append((output_dir, True, f"{len(files)} files"))
        else:
            print_warning(f"{output_dir}/ - Répertoire non trouvé")
            results.append((output_dir, False, "Directory missing"))

    print_info(f"Total: {total_files} fichiers de sortie")
    return results


def generate_final_report():
    """Génère le rapport final."""
    print_header("Génération du rapport final")

    # Collecte de tous les résultats de tests
    import_results = test_imports()
    function_results = test_function_calls()
    execution_result = test_code_execution()
    quality_results = analyze_code_quality()
    tools_results = check_analysis_tools()
    output_results = check_generated_outputs()

    # Calcul des statistiques
    total_tests = (len(import_results) + len(function_results) + 1 +
                   len(quality_results) + len(tools_results) + len(output_results))

    passed_tests = (sum(1 for _, success, _ in import_results if success) +
                    sum(1 for _, success, _ in function_results if success) +
                    (1 if execution_result[1] else 0) +
                    sum(1 for _, success, _ in quality_results if success) +
                    sum(1 for _, success, _ in tools_results if success) +
                    sum(1 for _, success, _ in output_results if success))

    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # Création du rapport
    report = {
        "validation_date": datetime.now().isoformat(),
        "project_path": PROJECT_ROOT,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": f"{success_rate:.1f}%",
            "status": "SUCCESS" if success_rate >= 80 else "PARTIAL" if success_rate >= 60 else "FAILED"
        },
        "detailed_results": {
            "imports": [{"module": m, "success": s, "error": e} for m, s, e in import_results],
            "functions": [{"test": t, "success": s, "error": e} for t, s, e in function_results],
            "execution": {"test": execution_result[0], "success": execution_result[1], "error": execution_result[2]},
            "code_quality": [{"file": f, "success": s, "info": i} for f, s, i in quality_results],
            "analysis_tools": [{"tool": t, "success": s, "info": i} for t, s, i in tools_results],
            "generated_outputs": [{"output": o, "success": s, "info": i} for o, s, i in output_results]
        }
    }

    # Sauvegarde du rapport
    report_path = "final_validation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_success(f"Rapport de validation sauvegardé: {report_path}")

    # Affichage du résumé
    print_header("RÉSUMÉ FINAL")

    if success_rate >= 80:
        print_success(f"✨ VALIDATION RÉUSSIE! ✨")
        print_success(
            f"Taux de réussite: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print_info("Le projet Baking_EEG est maintenant fonctionnel et corrigé.")
    elif success_rate >= 60:
        print_warning(f"✨ VALIDATION PARTIELLE ✨")
        print_warning(
            f"Taux de réussite: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print_info("La plupart des corrections ont été appliquées avec succès.")
    else:
        print_error(f"❌ VALIDATION ÉCHOUÉE ❌")
        print_error(
            f"Taux de réussite: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        print_info("Des problèmes critiques subsistent.")

    print(f"\n📊 Statistiques détaillées:")
    print(
        f"   • Imports: {sum(1 for _, s, _ in import_results if s)}/{len(import_results)}")
    print(
        f"   • Fonctions: {sum(1 for _, s, _ in function_results if s)}/{len(function_results)}")
    print(f"   • Exécution: {'✅' if execution_result[1] else '❌'}")
    print(
        f"   • Qualité code: {sum(1 for _, s, _ in quality_results if s)}/{len(quality_results)}")
    print(
        f"   • Outils d'analyse: {sum(1 for _, s, _ in tools_results if s)}/{len(tools_results)}")
    print(
        f"   • Fichiers générés: {sum(1 for _, s, _ in output_results if s)}/{len(output_results)}")

    return report


if __name__ == "__main__":
    print_header("🔬 VALIDATION FINALE DU PROJET BAKING_EEG 🔬")
    print_info(f"Répertoire de travail: {PROJECT_ROOT}")
    print_info(
        f"Date de validation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        report = generate_final_report()
        print_info(
            f"\nValidation terminée. Rapport disponible: final_validation_report.json")

        # Affichage des problèmes restants s'il y en a
        if report["summary"]["status"] != "SUCCESS":
            print_header("⚠️  PROBLÈMES IDENTIFIÉS")
            for category, results in report["detailed_results"].items():
                if isinstance(results, list):
                    for item in results:
                        if not item.get("success", True):
                            print_error(f"{category}: {item}")
                elif isinstance(results, dict):
                    if not results.get("success", True):
                        print_error(f"{category}: {results}")

            print_info(
                "\nPour corriger ces problèmes, consultez le rapport détaillé.")

    except Exception as e:
        print_error(f"Erreur lors de la validation: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
