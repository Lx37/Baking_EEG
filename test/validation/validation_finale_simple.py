#!/usr/bin/env python3
"""
Script de validation finale simplifié - Baking_EEG
Vérifie que toutes les corrections critiques ont été appliquées avec succès.
"""

import os
import sys
import json
from datetime import datetime


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_success(text):
    print(f"✅ {text}")


def print_error(text):
    print(f"❌ {text}")


def print_warning(text):
    print(f"⚠️  {text}")


def print_info(text):
    print(f"ℹ️  {text}")


def test_file_corrections():
    """Vérifie que les corrections de fichiers ont été appliquées."""
    print_header("Vérification des corrections de fichiers")

    results = {
        "files_corrected": 0,
        "total_files": 4,
        "details": []
    }

    # Test 1: stats_utils.py - instruction return ajoutée
    try:
        with open('utils/stats_utils.py', 'r') as f:
            content = f.read()
        if 'return observed_t_values_out, fdr_significant_mask_out, fdr_corrected_p_values_out' in content:
            print_success("stats_utils.py - Instruction return ajoutée")
            results["files_corrected"] += 1
            results["details"].append(
                {"file": "stats_utils.py", "status": "fixed", "issue": "missing return"})
        else:
            print_error("stats_utils.py - Instruction return manquante")
            results["details"].append(
                {"file": "stats_utils.py", "status": "error", "issue": "missing return"})
    except Exception as e:
        print_error(f"stats_utils.py - Erreur lecture: {e}")
        results["details"].append(
            {"file": "stats_utils.py", "status": "error", "issue": str(e)})

    # Test 2: run_decoding_one_pp.py - paramètre corrigé
    try:
        with open('examples/run_decoding_one_pp.py', 'r') as f:
            content = f.read()
        if 'CHANCE_LEVEL_AUC' in content and 'CONFIG_LOAD_MAIN_DECODING' in content:
            print_success("run_decoding_one_pp.py - Paramètres corrigés")
            results["files_corrected"] += 1
            results["details"].append(
                {"file": "run_decoding_one_pp.py", "status": "fixed", "issue": "parameters corrected"})
        else:
            print_error("run_decoding_one_pp.py - Paramètres non corrigés")
            results["details"].append(
                {"file": "run_decoding_one_pp.py", "status": "error", "issue": "parameters not corrected"})
    except Exception as e:
        print_error(f"run_decoding_one_pp.py - Erreur lecture: {e}")
        results["details"].append(
            {"file": "run_decoding_one_pp.py", "status": "error", "issue": str(e)})

    # Test 3: final_validation_complete.py - import corrigé
    try:
        with open('final_validation_complete.py', 'r') as f:
            content = f.read()
        if 'Baking_EEG._4_decoding_core' in content:
            print_success("final_validation_complete.py - Import corrigé")
            results["files_corrected"] += 1
            results["details"].append(
                {"file": "final_validation_complete.py", "status": "fixed", "issue": "import corrected"})
        else:
            print_error("final_validation_complete.py - Import non corrigé")
            results["details"].append(
                {"file": "final_validation_complete.py", "status": "error", "issue": "import not corrected"})
    except Exception as e:
        print_error(f"final_validation_complete.py - Erreur lecture: {e}")
        results["details"].append(
            {"file": "final_validation_complete.py", "status": "error", "issue": str(e)})

    # Test 4: Répertoires créés
    directories = ['diagrams', 'figures', 'analysis_results']
    dirs_created = 0
    for d in directories:
        if os.path.exists(d):
            dirs_created += 1
        else:
            print_warning(f"Répertoire {d}/ manquant")

    if dirs_created == len(directories):
        print_success("Tous les répertoires requis sont présents")
        results["files_corrected"] += 1
        results["details"].append(
            {"file": "directories", "status": "fixed", "issue": "all directories present"})
    else:
        print_warning(
            f"Seulement {dirs_created}/{len(directories)} répertoires présents")
        results["details"].append({"file": "directories", "status": "partial",
                                  "issue": f"{dirs_created}/{len(directories)} directories"})

    return results


def test_basic_functionality():
    """Teste la fonctionnalité de base sans imports lourds."""
    print_header("Test de fonctionnalité de base")

    results = {
        "functional_tests": 0,
        "total_tests": 2,
        "details": []
    }

    # Test 1: Vérification syntaxe Python
    try:
        import py_compile
        files_to_check = [
            'utils/stats_utils.py',
            'examples/run_decoding_one_pp.py',
            'utils/loading_PP_utils.py'
        ]

        compilation_success = 0
        for file_path in files_to_check:
            try:
                py_compile.compile(file_path, doraise=True)
                compilation_success += 1
            except py_compile.PyCompileError as e:
                print_error(f"Erreur compilation {file_path}: {e}")

        if compilation_success == len(files_to_check):
            print_success("Tous les fichiers compilent sans erreur syntaxique")
            results["functional_tests"] += 1
            results["details"].append({"test": "syntax_check", "status": "passed",
                                      "details": f"{compilation_success}/{len(files_to_check)} files"})
        else:
            print_warning(
                f"Compilation: {compilation_success}/{len(files_to_check)} fichiers réussis")
            results["details"].append({"test": "syntax_check", "status": "partial",
                                      "details": f"{compilation_success}/{len(files_to_check)} files"})

    except Exception as e:
        print_error(f"Erreur test compilation: {e}")
        results["details"].append(
            {"test": "syntax_check", "status": "error", "details": str(e)})

    # Test 2: Structure du projet
    required_structure = [
        'utils/',
        'examples/',
        'Baking_EEG/',
        'config/',
        'base/',
        'utils/stats_utils.py',
        'examples/run_decoding_one_pp.py'
    ]

    structure_ok = 0
    for item in required_structure:
        if os.path.exists(item):
            structure_ok += 1
        else:
            print_warning(f"Élément manquant: {item}")

    if structure_ok == len(required_structure):
        print_success("Structure du projet complète")
        results["functional_tests"] += 1
        results["details"].append(
            {"test": "project_structure", "status": "passed", "details": "all required files present"})
    else:
        print_warning(
            f"Structure: {structure_ok}/{len(required_structure)} éléments présents")
        results["details"].append({"test": "project_structure", "status": "partial",
                                  "details": f"{structure_ok}/{len(required_structure)} elements"})

    return results


def generate_final_report(file_results, functionality_results):
    """Génère le rapport final de validation."""
    print_header("RAPPORT FINAL DE VALIDATION")

    total_files_fixed = file_results["files_corrected"]
    total_files = file_results["total_files"]
    total_tests_passed = functionality_results["functional_tests"]
    total_tests = functionality_results["total_tests"]

    overall_score = ((total_files_fixed / total_files) +
                     (total_tests_passed / total_tests)) / 2 * 100

    if overall_score >= 90:
        status = "EXCELLENT ✨"
        emoji = "🎉"
    elif overall_score >= 75:
        status = "TRÈS BON ✅"
        emoji = "✅"
    elif overall_score >= 50:
        status = "SATISFAISANT ⚠️"
        emoji = "⚠️"
    else:
        status = "NÉCESSITE ATTENTION ❌"
        emoji = "❌"

    print(f"{emoji} STATUS: {status}")
    print(f"📊 Score global: {overall_score:.1f}%")
    print(f"🔧 Fichiers corrigés: {total_files_fixed}/{total_files}")
    print(f"⚡ Tests fonctionnels: {total_tests_passed}/{total_tests}")

    # Génération du rapport JSON
    report = {
        "validation_date": datetime.now().isoformat(),
        "overall_score": overall_score,
        "status": status,
        "file_corrections": file_results,
        "functionality_tests": functionality_results,
        "summary": {
            "files_corrected": f"{total_files_fixed}/{total_files}",
            "tests_passed": f"{total_tests_passed}/{total_tests}",
            "recommendation": "Projet prêt pour utilisation" if overall_score >= 75 else "Corrections supplémentaires recommandées"
        }
    }

    with open('validation_final_simple.json', 'w') as f:
        json.dump(report, f, indent=2)

    print_info("Rapport détaillé sauvegardé: validation_final_simple.json")

    return report


def main():
    """Fonction principale de validation."""
    print_header("🔬 VALIDATION FINALE SIMPLIFIÉE - BAKING_EEG")
    print_info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Répertoire: {os.getcwd()}")

    # Tests des corrections de fichiers
    file_results = test_file_corrections()

    # Tests de fonctionnalité de base
    functionality_results = test_basic_functionality()

    # Génération du rapport final
    final_report = generate_final_report(file_results, functionality_results)

    print_header("Validation terminée")
    print_info(
        "Utilisez 'python examples/run_decoding_one_pp.py --help' pour tester le script principal")

    return final_report


if __name__ == "__main__":
    main()
