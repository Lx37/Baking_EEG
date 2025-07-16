#!/usr/bin/env python3
"""
Script pour ranger les fichiers par groupes selon les IDs définis
"""

import os
import shutil
import re
from pathlib import Path

# Définition des groupes
ALL_SUBJECTS_GROUPS = {
    "CONTROLS_DELIRIUM": [
        "TPC2",
        "TPLV4",
        "TTDV5",
        # "TYS6", # Att 401 points temporels pour protocole LG
        "LAB1",
        "LAG6",
        "LAT3",
        "LBM4",
        "LCM2",
        "LP05",
        "TJL3",
        "TJR7",
        "TLP8",
    ],
    "CONTROLS_COMA": [
        "AO05",
        "BT13",
        "HM10",
        "JM14",
        "LS07",
        "LT12",
        "PB20",
        "SB09",
        "SP03",
        "TAK7",
        "TCG5",
        "TEN1",
        "TFB6",
        "TGD8",
        "TJL3",
        "TNC11",
        "TSS4",
        "TTV2",
        "TVM10",
        "TVR9",
        "TZ11",
        "VB01",
        "FG104",
        "FP102",
        "MB103",
    ],
    "COMA": [
        "AD94",
        "AM88",
        "AP84",
        "AS_FRA",
        "BO_AXE",
        "BR_JEA",
        "BS81",
        "CA55",
        "CI_MIC",
        "CS38",
        "DE_HEN",
        "DR92",
        "DU_CHR",
        "FB83",
        "GA_MAR",
        "GV77",
        "JC39",
        "JM78",
        "JR79",
        "KS76",
        "LA_PIE",
        "MA_VAL",
        "MB73",
        "ME63",
        "ME64",
        "MP68",
        "NN65",
        "RE_JOS",
        "SB67",
        "TF53",
        "TpAB15J1",
        "TpAF11J1",
        "TpAT19J1",
        "TpBD10J1",
        "TpCF24J1",
        "TpCL14j1",
        "TpDC12J1",
        "TpDC12J8",
        "TpDC22J1",
        "TpDL8J1",
        "TpDP7J1",
        "TpDP7J8",
        "TpDP7_Surnom",
        "TpEM13J1",
        "TpEP16J1",
        "TpEP20J1",
        "TpFJ5J1",
        "TpFJ5J8",
        "TpFM25J1",
        "TpJM2J1",
        "TpLC21J1",
        "TpLJ6J1",
        "TpMD4J1",
        "TpMG17J1",
        "TpML3J1",
        "TpPC23J1",
        "TpTP1J1",
        "TpTP1J8",
        "TT45",
        "YG72",
    ],
    "VS": [
        "CB31",
        "DZ44",
        "FM60",
        "FR43",
        "GM37",
        "GU32",
        "HZ24",
        "KA70",
        "KG85",
        "MH74",
        "MM86",
        "OD69",
        "OS90",
        "PB28",
        "SR57",
        "TL36",
        "TpAB15J8",
        "TpAT19J8",
        "TpBD10J8",
        "TpDC22J8",
        "TpDL8J8",
        "TpEP16j8",
        "TpFM25J8",
        "TpLJ6J8",
        "TpML3J8",
        "VS91",
    ],
    "MCS-": [
        "BT25",
        "CB34",
        "CG29",
        "CR26",
        "MC40",
        "ML33",
    ],
    "MCS": [
        "AE93",
        "AG42",
        "CW41",
        "DA75",
        "GT50",
        "HM52",
        "JA71",
        "KN49",
        "LP54",
        "MC58",
        "MK80",
        "PL82",
        "SR59",
        "TB56",
        "TpCF24J8",
        "PE_SAM",
        "YG66",
        "IR27",
        "NF35",
    ],
    "DELIRIUM+": [
        "TpAB19",
        "TpAK24",
        "TpAK27",
        "TpBL47",
        "TpCB15",
        "TpCF1",
        "TpDRL3",
        "TpFF34",
        "TpFY57",
        "TpJA20",
        "TpJB25",
        "TpJB26",
        # "TpJC5", 301 points temporels
        "TpJCD29",
        "TpJLR17",
        "TpJPS55",
        "TpLA28",
        "TpMB45",
        "TpMM4",
        "TpMN42",
        "TpPC21",
        "TpPM14",
        "TpPM31",
        "TpRD38",
        "TpSM49",
    ],
    "DELIRIUM-": [
        "TpAC23",
        "TpAG51",
        "TpAM43",
        "TpBD16",
        "TpDD2",
        "TpFB18",
        "TpFL53",
        "TpGB8",
        "TpGT32",
        "TpJPG7",
        "TpJPL10",
        "TpKS6",
        "TpLP11",
        "TpMA9",
        "TpMD13",
        "TpMD52",
        "TpME22",
        "TpPA35",
        "TpPI46",
        "TpPL48",
        "TpRB50",
        "TpRK39",
        "TpSD30",
        "TpYB41",
    ],
}

def extract_subject_id(filename):
    """
    Extrait l'ID du sujet à partir du nom de fichier
    Exemple: AG42_LG_preproc_noICA_LG-epo.fif -> AG42
    """
    # Recherche du pattern ID au début du nom de fichier
    match = re.match(r'^([A-Za-z]+\d+|[A-Za-z]+_[A-Za-z]+|Tp[A-Za-z]+\d+[A-Za-z]*\d*)', filename)
    if match:
        return match.group(1)
    return None

def find_group_for_subject(subject_id):
    """
    Trouve le groupe correspondant à un ID de sujet
    """
    for group_name, subjects in ALL_SUBJECTS_GROUPS.items():
        if subject_id in subjects:
            return group_name
    return None

def organize_files(source_directory, dry_run=True):
    """
    Organise les fichiers dans les dossiers appropriés
    
    Args:
        source_directory (str): Répertoire source contenant les fichiers
        dry_run (bool): Si True, affiche seulement ce qui serait fait sans déplacer les fichiers
    """
    source_path = Path(source_directory)
    
    if not source_path.exists():
        print(f"Erreur: Le répertoire {source_directory} n'existe pas.")
        return
    
    # Statistiques
    moved_files = 0
    unmatched_files = []
    
    # Parcourir tous les fichiers dans le répertoire source
    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            filename = file_path.name
            
            # Extraire l'ID du sujet
            subject_id = extract_subject_id(filename)
            
            if subject_id:
                # Trouver le groupe correspondant
                group = find_group_for_subject(subject_id)
                
                if group:
                    # Créer le nom du dossier de destination
                    dest_folder = f"LG_{group}"
                    dest_path = source_path / dest_folder
                    
                    if dry_run:
                        print(f"[DRY RUN] {filename} -> {dest_folder}/")
                        print(f"  ID: {subject_id} -> Groupe: {group}")
                    else:
                        # Créer le dossier de destination s'il n'existe pas
                        dest_path.mkdir(exist_ok=True)
                        
                        # Déplacer le fichier
                        new_file_path = dest_path / filename
                        try:
                            shutil.move(str(file_path), str(new_file_path))
                            print(f"Déplacé: {filename} -> {dest_folder}/")
                            moved_files += 1
                        except Exception as e:
                            print(f"Erreur lors du déplacement de {filename}: {e}")
                else:
                    unmatched_files.append((filename, subject_id))
                    if dry_run:
                        print(f"[NON TROUVÉ] {filename} (ID: {subject_id}) - Aucun groupe correspondant")
            else:
                unmatched_files.append((filename, "ID non extrait"))
                if dry_run:
                    print(f"[ERREUR] {filename} - Impossible d'extraire l'ID")
    
    # Afficher les statistiques
    print(f"\n=== RÉSUMÉ ===")
    if dry_run:
        print("Mode DRY RUN - Aucun fichier n'a été déplacé")
    else:
        print(f"Fichiers déplacés: {moved_files}")
    
    print(f"Fichiers non traités: {len(unmatched_files)}")
    
    if unmatched_files:
        print("\nFichiers non traités:")
        for filename, reason in unmatched_files:
            print(f"  - {filename} ({reason})")

def main():
    """
    Fonction principale
    """
    # Exemple d'utilisation
    source_directory = "/crnldata/cap/users/_tom/LG_fab/LG_PATIENTS_DoC"
    
    print("=== ORGANISATION DES FICHIERS PAR GROUPE ===")
    print(f"Répertoire source: {source_directory}")
    
    # D'abord, faire un dry run pour voir ce qui serait fait
    print("\n--- DRY RUN (Simulation) ---")
    organize_files(source_directory, dry_run=True)
    
    # Demander confirmation avant de procéder
    print("\n" + "="*50)
    response = input("Voulez-vous procéder au déplacement des fichiers? (oui/non): ").lower().strip()
    
    if response in ['oui', 'o', 'yes', 'y']:
        print("\n--- DÉPLACEMENT DES FICHIERS ---")
        organize_files(source_directory, dry_run=False)
        print("\nTerminé!")
    else:
        print("Opération annulée.")

if __name__ == "__main__":
    main()