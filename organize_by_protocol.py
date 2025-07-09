#!/usr/bin/env python3
"""
Script pour organiser les résultats NPZ par protocole et groupe.
Lit l'arborescence existante et crée une nouvelle organisation par protocole.
"""

import os
import sys
import shutil
import glob
import logging
from pathlib import Path
import re
# Import configurations depuis le bon chemin
current_dir = os.path.dirname(os.path.abspath(__file__))
baking_eeg_dir = os.path.join(current_dir, 'Baking_EEG')
sys.path.insert(0, baking_eeg_dir)
# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
try:
    from config.config import ALL_SUBJECT_GROUPS
    logger.info("Configuration importée avec succès")
except ImportError as e:
    logger.error(f"Erreur d'import de la configuration: {e}")
    # Fallback avec configuration basique
    ALL_SUBJECT_GROUPS = {
        'DELIRIUM+': [],
        'DELIRIUM-': [],
        'CONTROLS_DELIRIUM': [],
        'COMA': [],
        'VS': [],
        'MCS+': [],
        'MCS-': [],
        'CONTROLS_COMA': []
    }

# === CONFIGURATION ===
# Chemins de base des résultats existants
#INTRA_SUBJECT_RESULTS_DIR = "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG_results_V12/intra_subject_results"
INTRA_SUBJECT_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V12/intra_subject_results"

#INTRA_SUBJECT_LG_RESULTS_DIR = "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG_results_V12/intra_subject_lg_results"
INTRA_SUBJECT_LG_RESULTS_DIR = "/home/tom.balay/results/Baking_EEG_results_V12/intra_subject_lg_results"
# Nouveau dossier organisé par protocole
#ORGANIZED_RESULTS_DIR = "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG_results_organized_by_protocol"
ORGANIZED_RESULTS_DIR = "/home/tom.balay/results/BakingEEG_results_organized_by_protocol"


def determine_subject_group(subject_id):
    """Détermine le groupe d'un sujet à partir de son ID avec logique améliorée."""
    
    # 1. Recherche directe
    for group_name, subjects in ALL_SUBJECT_GROUPS.items():
        if subject_id in subjects:
            return group_name
    
    # 2. Si pas trouvé, essayer avec préfixe "Tp"
    if not subject_id.startswith('Tp'):
        tp_subject_id = f"Tp{subject_id}"
        for group_name, subjects in ALL_SUBJECT_GROUPS.items():
            if tp_subject_id in subjects:
                return group_name
    
    # 3. Si pas trouvé, essayer sans préfixe "Tp"
    if subject_id.startswith('Tp'):
        no_tp_subject_id = subject_id[2:]  # Enlever "Tp"
        for group_name, subjects in ALL_SUBJECT_GROUPS.items():
            if no_tp_subject_id in subjects:
                return group_name
    
    # 4. Recherche avec des variantes de suffixes (J1, J8, etc.)
    base_patterns = []
    if subject_id.startswith('Tp'):
        # Si c'est TpXXXX, extraire juste XXXX
        base_id = subject_id[2:]
        base_patterns = [
            f"Tp{base_id}J1",
            f"Tp{base_id}J8", 
            f"Tp{base_id}j1",
            f"Tp{base_id}j8",
            base_id
        ]
    else:
        # Si c'est XXXX, essayer toutes les variantes
        base_patterns = [
            f"Tp{subject_id}",
            f"Tp{subject_id}J1",
            f"Tp{subject_id}J8",
            f"Tp{subject_id}j1", 
            f"Tp{subject_id}j8"
        ]
    
    for pattern in base_patterns:
        for group_name, subjects in ALL_SUBJECT_GROUPS.items():
            if pattern in subjects:
                return group_name
    
    # 5. Recherche avec sous-chaines (au cas où il y aurait des caractères supplémentaires)
    for group_name, subjects in ALL_SUBJECT_GROUPS.items():
        for config_subject in subjects:
            # Vérifier si l'ID extrait est une sous-partie de l'ID de la config
            if subject_id in config_subject or config_subject in subject_id:
                # Vérification plus stricte pour éviter les faux positifs
                if len(subject_id) >= 3 and len(config_subject) >= 3:
                    return group_name
    
    return None

def extract_subject_id_from_path(file_path):
    """Extrait l'ID du sujet depuis le chemin du fichier."""
    # Patterns possibles pour extraire l'ID du sujet
    patterns = [
        r'_Subj_([A-Z0-9]+)_',  # Pattern principal
        r'/([A-Z0-9]+)_Group_',  # Alternative
        r'(TpDL8J8)',  # Exemple spécifique vu dans votre structure - avec capture group
        r'([A-Z]{2,4}\d+)',  # Pattern général
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_path)
        if match and match.groups():  # Vérifier qu'il y a bien des groupes
            try:
                return match.group(1)
            except IndexError:
                # Si le pattern match mais n'a pas de groupe 1, passer au suivant
                continue
    
    # Essayer d'extraire depuis le nom du dossier parent
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if '_Subj_' in part:
            try:
                subject_id = part.split('_Subj_')[1].split('_')[0]
                return subject_id
            except:
                continue
    
    return None

def determine_protocol_from_path(file_path):
    """Détermine le protocole depuis le chemin du fichier."""
    path_lower = file_path.lower()
    
    # Vérifier d'abord si c'est dans intra_subject_lg_results
    if 'intra_subject_lg_results' in path_lower:
        return 'LG'
    
    # Vérifier les patterns dans les noms de dossiers/fichiers
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        part_lower = part.lower()
        # Protocole Delirium - patterns spécifiques pour délirium
        if (part_lower.endswith('_delirium') or 
            part_lower.endswith('delirium') or 
            'delirium+_delirium' in part_lower or 
            'delirium-_delirium' in part_lower or
            'controls_delirium' in part_lower):
            return 'Delirium'
        # Protocole PPext3 - tout ce qui contient "ppext3"
        elif 'ppext3' in part_lower:
            return 'PPext3'
        # Protocole Battery - tout ce qui contient "battery"
        elif 'battery' in part_lower:
            return 'Battery'
    
    # Rechercher des indicateurs de protocole dans tout le chemin
    if 'ppext3' in path_lower:
        return 'PPext3'
    elif 'battery' in path_lower:
        return 'Battery'
    elif 'delirium' in path_lower:
        return 'Delirium'
    
    # Fallback basé sur le dossier source
    if 'intra_subject_results' in path_lower:
        # Si c'est dans intra_subject_results, essayer de deviner selon le groupe
        subject_id = extract_subject_id_from_path(file_path)
        if subject_id:
            subject_group = determine_subject_group(subject_id)
            if subject_group in ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM']:
                return 'Delirium'
            elif subject_group in ['COMA', 'VS', 'MCS+', 'MCS-', 'CONTROLS_COMA']:
                # Par défaut PPext3 pour les groupes de conscience
                return 'PPext3'
    
    return 'Unknown'

def find_all_npz_files():
    """Trouve tous les fichiers NPZ de résultats dans les deux dossiers sources."""
    logger.info("Recherche des fichiers NPZ dans les dossiers de résultats...")
    
    all_files = []
    
    # Chercher dans intra_subject_results
    if os.path.exists(INTRA_SUBJECT_RESULTS_DIR):
        logger.info(f"Recherche dans : {INTRA_SUBJECT_RESULTS_DIR}")
        search_patterns = [
            os.path.join(INTRA_SUBJECT_RESULTS_DIR, "**/decoding_results_full.npz"),
            os.path.join(INTRA_SUBJECT_RESULTS_DIR, "**/*results*.npz"),
        ]
        
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
    
    # Chercher dans intra_subject_lg_results
    if os.path.exists(INTRA_SUBJECT_LG_RESULTS_DIR):
        logger.info(f"Recherche dans : {INTRA_SUBJECT_LG_RESULTS_DIR}")
        search_patterns = [
            os.path.join(INTRA_SUBJECT_LG_RESULTS_DIR, "**/decoding_results_full.npz"),
            os.path.join(INTRA_SUBJECT_LG_RESULTS_DIR, "**/*results*.npz"),
        ]
        
        for pattern in search_patterns:
            files = glob.glob(pattern, recursive=True)
            all_files.extend(files)
    
    # Supprimer les doublons
    all_files = list(set(all_files))
    logger.info(f"Trouvé {len(all_files)} fichiers NPZ au total")
    
    return all_files

def organize_files_by_protocol(files_list):
    """Organise les fichiers par protocole et groupe."""
    organized = {}
    unknown_subjects = set()
    
    for file_path in files_list:
        # Extraire les informations
        subject_id = extract_subject_id_from_path(file_path)
        protocol = determine_protocol_from_path(file_path)
        
        if not subject_id:
            logger.warning(f"Impossible d'extraire l'ID du sujet pour : {file_path}")
            continue
        
        group = determine_subject_group(subject_id)
        if not group:
            # Au lieu d'ignorer, créer un groupe "UNKNOWN"
            group = "UNKNOWN"
            unknown_subjects.add(subject_id)
        
        # Organiser par protocole puis par groupe
        if protocol not in organized:
            organized[protocol] = {}
        if group not in organized[protocol]:
            organized[protocol][group] = []
        
        # Stocker le chemin complet du dossier contenant le fichier NPZ
        result_dir = os.path.dirname(file_path)
        organized[protocol][group].append({
            'subject_id': subject_id,
            'file_path': file_path,
            'result_dir': result_dir
        })
    
    # Afficher un résumé des sujets inconnus
    if unknown_subjects:
        logger.warning(f"Trouvé {len(unknown_subjects)} sujets non répertoriés dans la configuration:")
        logger.warning(f"Sujets inconnus: {', '.join(sorted(unknown_subjects))}")
    
    return organized

def copy_organized_structure(organized_data, output_base_dir):
    """Copie les fichiers dans la nouvelle structure organisée."""
    os.makedirs(output_base_dir, exist_ok=True)
    
    for protocol, groups in organized_data.items():
        logger.info(f"\n=== PROTOCOLE: {protocol} ===")
        
        for group, subjects in groups.items():
            logger.info(f"Groupe {group}: {len(subjects)} sujets")
            
            # Créer le dossier de destination
            dest_protocol_dir = os.path.join(output_base_dir, f"protocol_{protocol}")
            dest_group_dir = os.path.join(dest_protocol_dir, f"group_{group}")
            os.makedirs(dest_group_dir, exist_ok=True)
            
            # Copier les dossiers de résultats de chaque sujet
            for subject_data in subjects:
                subject_id = subject_data['subject_id']
                source_dir = subject_data['result_dir']
                
                # Nom du dossier de destination pour ce sujet
                dest_subject_dir = os.path.join(dest_group_dir, f"subject_{subject_id}")
                
                try:
                    # Copier tout le dossier de résultats du sujet
                    if os.path.exists(dest_subject_dir):
                        shutil.rmtree(dest_subject_dir)
                    shutil.copytree(source_dir, dest_subject_dir)
                    logger.info(f"  Copié: {subject_id} -> {dest_subject_dir}")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la copie de {subject_id}: {e}")

def generate_group_analysis_scripts(organized_data, output_base_dir):
    """Génère des scripts d'analyse de groupe pour chaque protocole."""
    
    for protocol, groups in organized_data.items():
        # Créer un script d'analyse pour ce protocole
        script_content = f'''#!/usr/bin/env python3
"""
Script d'analyse de groupe pour le protocole {protocol}
Généré automatiquement
"""


import sys
import os

# Trouver le chemin vers BakingEEG de manière flexible
possible_paths = [
    '/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG',  # Mac local
    '/home/tom.balay/BakingEEG/Baking_EEG',  # Cluster variant 1
    '/home/tom.balay/Baking_EEG/Baking_EEG',  # Cluster variant 2
    '/home/tom.balay/Baking_EEG',  # Cluster root
    '/home/tom.balay/BakingEEG',  # Cluster root variant
    './Baking_EEG',  # Relatif
    '../Baking_EEG',  # Parent directory
    '../../Baking_EEG',  # Two levels up
    '/home/tom.balay/',  # Home directory pour recherche
]

baking_eeg_path = None
for path in possible_paths:
    if os.path.exists(path):
        baking_eeg_path = path
        break

if baking_eeg_path:
    sys.path.insert(0, baking_eeg_path)
    print(f"Chemin BakingEEG trouvé: {{baking_eeg_path}}")
else:
    print("ERREUR: Chemin vers BakingEEG non trouvé!")
    print("Chemins testés:")
    for path in possible_paths:
        print(f"  - {{path}}")
    sys.exit(1)


try:
    # Importer le script d'analyse de groupe existant
    from examples.run_group_npz import main as run_group_analysis
    print("Module run_group_npz importé avec succès")
except ImportError as e:
    print(f"Erreur d'import de run_group_npz: {{e}}")
    print("Tentative d'import alternatif...")
    
    # Import alternatif si le module n'est pas trouvé
    try:
        # Essayer d'importer directement le script
        import run_group_npz
        run_group_analysis = run_group_npz.main
    except ImportError:
        print("Module run_group_npz non trouvé. Veuillez vérifier l'installation.")
        sys.exit(1)

if __name__ == "__main__":
    # Chemin vers les données de ce protocole
    protocol_data_dir = os.path.join(os.path.dirname(__file__), "protocol_{protocol}")
    
    print(f"Lancement de l'analyse de groupe pour le protocole {protocol}")
    print(f"Dossier de données: {{protocol_data_dir}}")
    
    if not os.path.exists(protocol_data_dir):
        print(f"ERREUR: Dossier non trouvé: {{protocol_data_dir}}")
        sys.exit(1)
    
    # Lancer l'analyse en passant le dossier comme argument
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], protocol_data_dir]
    
    try:
        run_group_analysis()
    except Exception as e:
        print(f"Erreur lors de l'exécution: {{e}}")
        sys.argv = original_argv  # Restaurer les arguments originaux
        raise
'''
        
        script_path = os.path.join(output_base_dir, f"analyze_{protocol.lower()}.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Rendre le script exécutable
        os.chmod(script_path, 0o755)
        logger.info(f"Script d'analyse généré: {script_path}")

def create_summary_report(organized_data, output_base_dir):
    """Crée un rapport de synthèse de l'organisation."""
    report_path = os.path.join(output_base_dir, "organization_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT DE SYNTHÈSE - ORGANISATION PAR PROTOCOLE\n")
        f.write("="*80 + "\n\n")
        
        f.write("ORGANISATION EN 4 PROTOCOLES PRINCIPAUX:\n")
        f.write("1. Delirium : Patients avec délire (DELIRIUM+, DELIRIUM-, CONTROLS_DELIRIUM)\n")
        f.write("2. PPext3   : Protocole de conscience étendu (COMA, VS, MCS+, MCS-, CONTROLS_COMA)\n")
        f.write("3. Battery  : Batterie de tests de conscience\n")
        f.write("4. LG       : Local-Global paradigm\n\n")
        
        total_subjects = 0
        unknown_subjects = []
        
        for protocol, groups in organized_data.items():
            f.write(f"PROTOCOLE: {protocol}\n")
            f.write("-"*40 + "\n")
            
            protocol_total = 0
            for group, subjects in groups.items():
                f.write(f"  Groupe {group}: {len(subjects)} sujets\n")
                for subject_data in subjects:
                    f.write(f"    - {subject_data['subject_id']}\n")
                    # Collecter les sujets inconnus
                    if group == "UNKNOWN":
                        unknown_subjects.append(subject_data['subject_id'])
                protocol_total += len(subjects)
            
            f.write(f"  TOTAL {protocol}: {protocol_total} sujets\n\n")
            total_subjects += protocol_total
        
        f.write("="*80 + "\n")
        f.write(f"TOTAL GÉNÉRAL: {total_subjects} sujets\n")
        
        if unknown_subjects:
            f.write(f"SUJETS NON RÉPERTORIÉS: {len(unknown_subjects)}\n")
            
        f.write("="*80 + "\n")
        
        f.write("\nSTRUCTURE CRÉÉE:\n")
        f.write(f"{output_base_dir}/\n")
        protocol_order = ['Delirium', 'PPext3', 'Battery', 'LG']  # Ordre d'affichage
        for protocol in protocol_order:
            if protocol in organized_data:
                f.write(f"├── protocol_{protocol}/\n")
                for group in organized_data[protocol].keys():
                    f.write(f"│   ├── group_{group}/\n")
                    f.write(f"│   │   ├── subject_XXX/\n")
                    f.write(f"│   │   └── ...\n")
                f.write(f"├── analyze_{protocol.lower()}.py\n")
        f.write("└── organization_summary.txt\n")
        
        # Ajouter la liste des sujets inconnus s'il y en a
        if unknown_subjects:
            f.write(f"\nSUJETS NON RÉPERTORIÉS DANS LA CONFIGURATION:\n")
            f.write("-"*50 + "\n")
            for subject in sorted(set(unknown_subjects)):
                f.write(f"  - {subject}\n")
            f.write(f"\nCes {len(set(unknown_subjects))} sujets ont été classés dans le groupe 'UNKNOWN'.\n")
            f.write("Ils doivent être ajoutés à la configuration ALL_SUBJECT_GROUPS si nécessaire.\n")
    
    # Créer aussi un fichier séparé avec la liste des sujets inconnus
    if unknown_subjects:
        unknown_path = os.path.join(output_base_dir, "unknown_subjects.txt")
        with open(unknown_path, 'w') as f:
            f.write("# Sujets non trouvés dans la configuration ALL_SUBJECT_GROUPS\n")
            f.write("# À ajouter manuellement aux groupes appropriés\n\n")
            for subject in sorted(set(unknown_subjects)):
                f.write(f"{subject}\n")
        logger.info(f"Liste des sujets inconnus: {unknown_path}")
    
    logger.info(f"Rapport de synthèse créé: {report_path}")

def test_protocol_detection():
    """Fonction de test pour vérifier la détection des protocoles."""
    test_paths = [
        "/path/to/DELIRIUM+_delirium/results.npz",
        "/path/to/DELIRIUM-_delirium/results.npz", 
        "/path/to/CONTROLS_DELIRIUM_delirium/results.npz",
        "/path/to/COMA_01Hz_ppext3/results.npz",
        "/path/to/battery_test/results.npz",
        "/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG_results_V12/intra_subject_lg_results/test/results.npz"
    ]
    
    print("Test de détection de protocole:")
    print("-" * 50)
    for path in test_paths:
        protocol = determine_protocol_from_path(path)
        print(f"Chemin: {path}")
        print(f"Protocole détecté: {protocol}")
        print()

def test_subject_group_detection():
    """Fonction de test pour vérifier la détection des groupes."""
    # Sujets problématiques mentionnés dans les logs
    problematic_subjects = [
        "ME22",   # Devrait correspondre à TpME22 dans DELIRIUM-
        "PI46",   # Devrait correspondre à TpPI46 dans DELIRIUM-
        "CF1",    # Devrait correspondre à TpCF1 dans DELIRIUM+
        "PC23",   # Devrait correspondre à TpPC23J1 dans COMA
        "ML3",    # Devrait correspondre à TpML3J1 dans COMA
        "EP20",   # Devrait correspondre à TpEP20J1 dans COMA
        "AK27",   # Devrait correspondre à TpAK27 dans DELIRIUM+
        "LC21",   # Devrait correspondre à TpLC21J1 dans COMA
        "RD38",   # Devrait correspondre à TpRD38 dans DELIRIUM+
        "DP7",    # Devrait correspondre à TpDP7J1 ou TpDP7J8 dans COMA
        "GB8",    # Devrait correspondre à TpGB8 dans DELIRIUM-
        "YB41",   # Devrait correspondre à TpYB41 dans DELIRIUM-
        "DL8",    # Devrait correspondre à TpDL8J1 dans COMA ou TpDL8J8 dans VS
        "TCG5",   # Devrait être dans CONTROLS_COMA
        "TpDL8J8", # Devrait être dans VS
    ]
    
    print("Test de détection de groupe pour sujets problématiques:")
    print("=" * 70)
    found_count = 0
    for subject_id in problematic_subjects:
        group = determine_subject_group(subject_id)
        if group:
            found_count += 1
            status = "✓ TROUVÉ"
        else:
            status = "✗ NON TROUVÉ"
        print(f"{subject_id:12} -> {group:20} {status}")
    
    print(f"\nRésumé: {found_count}/{len(problematic_subjects)} sujets trouvés")
    print("=" * 70)
    return found_count, len(problematic_subjects)

def main():
    """Fonction principale d'organisation."""
    logger.info("="*80)
    logger.info("ORGANISATION DES RÉSULTATS PAR PROTOCOLE ET GROUPE")
    logger.info("="*80)
    
    # Vérifier que les dossiers sources existent
    sources_found = []
    if os.path.exists(INTRA_SUBJECT_RESULTS_DIR):
        sources_found.append(INTRA_SUBJECT_RESULTS_DIR)
        logger.info(f"Dossier source trouvé: {INTRA_SUBJECT_RESULTS_DIR}")
    
    if os.path.exists(INTRA_SUBJECT_LG_RESULTS_DIR):
        sources_found.append(INTRA_SUBJECT_LG_RESULTS_DIR)
        logger.info(f"Dossier source trouvé: {INTRA_SUBJECT_LG_RESULTS_DIR}")
    
    if not sources_found:
        logger.error(f"Aucun dossier source trouvé!")
        logger.error(f"Vérifié: {INTRA_SUBJECT_RESULTS_DIR}")
        logger.error(f"Vérifié: {INTRA_SUBJECT_LG_RESULTS_DIR}")
        sys.exit(1)
    
    # 1. Trouver tous les fichiers NPZ
    logger.info("Étape 1: Recherche des fichiers NPZ...")
    all_npz_files = find_all_npz_files()
    
    if not all_npz_files:
        logger.error("Aucun fichier NPZ trouvé!")
        sys.exit(1)
    
    # 2. Organiser par protocole et groupe
    logger.info("Étape 2: Organisation par protocole et groupe...")
    organized_data = organize_files_by_protocol(all_npz_files)
    
    # 3. Afficher le résumé de l'organisation
    logger.info("\n" + "="*60)
    logger.info("RÉSUMÉ DE L'ORGANISATION PAR PROTOCOLE:")
    logger.info("="*60)
    
    # Affichage ordonné des protocoles
    protocol_order = ['Delirium', 'PPext3', 'Battery', 'LG', 'Unknown']
    
    for protocol in protocol_order:
        if protocol in organized_data:
            groups = organized_data[protocol]
            logger.info(f"\nProtocole {protocol}:")
            for group, subjects in groups.items():
                subject_ids = [s['subject_id'] for s in subjects]
                logger.info(f"  {group}: {len(subjects)} sujets ({', '.join(subject_ids[:5])}{'...' if len(subject_ids) > 5 else ''})")
    
    # Afficher les statistiques générales
    total_subjects = sum(len(subjects) for groups in organized_data.values() for subjects in groups.values())
    logger.info(f"\nTOTAL: {total_subjects} sujets dans {len(organized_data)} protocoles")
    
    # 4. Demander confirmation
    response = input(f"\nCopier vers {ORGANIZED_RESULTS_DIR}? (y/n): ")
    if response.lower() != 'y':
        logger.info("Opération annulée.")
        return
    
    # 5. Copier dans la nouvelle structure
    logger.info("Étape 3: Copie des fichiers...")
    copy_organized_structure(organized_data, ORGANIZED_RESULTS_DIR)
    
    # 6. Générer les scripts d'analyse
    logger.info("Étape 4: Génération des scripts d'analyse...")
    generate_group_analysis_scripts(organized_data, ORGANIZED_RESULTS_DIR)
    
    # 7. Créer le rapport de synthèse
    logger.info("Étape 5: Création du rapport de synthèse...")
    create_summary_report(organized_data, ORGANIZED_RESULTS_DIR)
    
    logger.info("\n" + "="*80)
    logger.info("ORGANISATION TERMINÉE AVEC SUCCÈS!")
    logger.info("="*80)
    logger.info(f"Nouvelle structure créée dans: {ORGANIZED_RESULTS_DIR}")
    logger.info("Scripts d'analyse générés pour chaque protocole.")
    logger.info("Consultez organization_summary.txt pour les détails.")

if __name__ == "__main__":
    # Test optionnel - décommenter pour tester
    print("Test de la nouvelle logique de détection:")
    found, total = test_subject_group_detection()
    print(f"\nAvant d'exécuter l'organisation complète, {found}/{total} sujets sont maintenant détectés.")
    input("Appuyez sur Entrée pour continuer avec l'organisation complète...")
    main()
