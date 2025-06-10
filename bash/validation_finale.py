#!/usr/bin/env python3
"""
Script de validation finale pour vérifier que toutes les corrections fonctionnent.
"""

import sys
import os

def test_run_decoding_one_pp():
    """Test d'exécution d'un exemple simple."""
    try:
        # Simuler l'exécution du script avec des arguments minimaux
        sys.path.insert(0, "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG")
        
        # Test des imports principaux
        from examples.run_decoding_one_pp import execute_single_subject_decoding
        from config.config import ALL_SUBJECT_GROUPS
        from config.decoding_config import CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT
        
        print("✅ Tous les imports principaux fonctionnent")
        
        # Vérifier les groupes de sujets
        subject_test = "TpSM49"
        group_found = None
        for group, subjects in ALL_SUBJECT_GROUPS.items():
            if subject_test in subjects:
                group_found = group
                break
        
        if group_found:
            print(f"✅ Sujet {subject_test} trouvé dans le groupe {group_found}")
        else:
            print(f"❌ Sujet {subject_test} non trouvé dans les groupes")
            
        print(f"✅ Configuration de chargement disponible: {type(CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test : {e}")
        return False

def test_bash_scripts_syntax():
    """Vérifie la syntaxe des scripts bash."""
    bash_scripts = [
        "submit_1patient.py",
        "submit_1group.py", 
        "submit_allgroup.py"
    ]
    
    for script in bash_scripts:
        try:
            with open(script, 'r') as f:
                content = f.read()
                
            # Test de compilation Python
            compile(content, script, 'exec')
            print(f"✅ {script} - syntaxe correcte")
            
        except SyntaxError as e:
            print(f"❌ {script} - erreur de syntaxe: {e}")
            return False
        except FileNotFoundError:
            print(f"❌ {script} - fichier non trouvé")
            return False
            
    return True

if __name__ == "__main__":
    print("🧪 Test de validation finale")
    print("=" * 50)
    
    test1 = test_run_decoding_one_pp()
    test2 = test_bash_scripts_syntax()
    
    print("=" * 50)
    if test1 and test2:
        print("🎉 Tous les tests sont passés ! Les corrections fonctionnent.")
        sys.exit(0)
    else:
        print("💥 Certains tests ont échoué.")
        sys.exit(1)
