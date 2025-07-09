import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime

def compare_sujets_excel():
    """
    Compare les sujets présents dans les dossiers avec ceux du fichier Excel
    pour identifier les fichiers manquants ou les sujets non traités.
    """
    
    # --- Configuration ---
    excel_file = "tableau_tous_les_patients_13-1-25.xlsx"
    base_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/"
    
    # Création du fichier log avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"analyse_sujets_manquants_{timestamp}.log"
    
    # Dossiers à scanner avec leurs protocoles respectifs
    directories_to_scan = {
        # Protocole LG (Local Global)
        "LG_CONTROLS_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_CONTROLS"
        },
        "LG_PATIENTS_DELIRIUM+_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_DELIRIUM+"
        },
        "LG_PATIENTS_DELIRIUM-_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_DELIRIUM-"
        },
        # Protocole PP (Passive Paradigm)
        "PP_CONTROLS_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_CONTROLS"
        },
        "PP_PATIENTS_DELIRIUM+_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_DELIRIUM+"
        },
        "PP_PATIENTS_DELIRIUM-_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_DELIRIUM-"
        },
        # Protocole PPAP (études coma, VS, MCS)
        "PP_CONTROLS_COMA_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_CONTROLS"
        },
        "PP_COMA_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_COMA"
        },
        "PP_COMA_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_COMA"
        },
        "PP_VS_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_VS"
        },
        "PP_VS_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_VS"
        },
        "PP_MCS+_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS+"
        },
        "PP_MCS+_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS+"
        },
        "PP_MCS-_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS-"
        },
        "PP_MCS-_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS-"
        }
    }
    
    def log_and_print(message, log_file_handle=None):
        """Affiche le message et l'écrit dans le log"""
        print(message)
        if log_file_handle:
            log_file_handle.write(message + "\n")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # En-tête du log
            log_and_print("=" * 70, log_f)
            log_and_print(f"🔍 ANALYSE COMPARATIVE SUJETS EXCEL vs FICHIERS", log_f)
            log_and_print(f"📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", log_f)
            log_and_print("=" * 70, log_f)
            log_and_print(f"📁 Dossier base: {base_directory}", log_f)
            log_and_print(f"📋 Fichier Excel: {excel_file}", log_f)
            log_and_print(f"🔍 Dossiers à scanner: {len(directories_to_scan)}", log_f)
            for folder in directories_to_scan.keys():
                log_and_print(f"   - {folder}", log_f)
            log_and_print("=" * 70, log_f)
            
            # --- 1. Lecture du fichier Excel ---
            log_and_print("📖 Lecture du fichier Excel...", log_f)
            if not os.path.exists(excel_file):
                log_and_print(f"❌ Fichier Excel non trouvé: {excel_file}", log_f)
                return
                
            df = pd.read_excel(excel_file, header=1)
            
            # Vérification des colonnes requises
            required_columns = ['ID_sujet', 'Patient / Témoin', 'Diagnostic']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                log_and_print(f"❌ Colonnes manquantes: {', '.join(missing_columns)}", log_f)
                log_and_print(f"📋 Colonnes disponibles: {list(df.columns)}", log_f)
                return
            
            # --- 2. Extraction des sujets du fichier Excel ---
            log_and_print("📋 Extraction des sujets du fichier Excel...", log_f)
            excel_subjects = {}
            
            for index, row in df.iterrows():
                patient_id = str(row.get('ID_sujet', '')).strip()
                patient_temoin = str(row.get('Patient / Témoin', '')).strip()
                diagnostic_raw = str(row.get('Diagnostic', '')).strip()
                
                # Ignorer les lignes invalides
                if not patient_id or patient_id.lower() in ['n/a', 'sortie', 'pas fait', 'debutt lizette', 'debut batterie a lyon']:
                    continue
                    
                excel_subjects[patient_id] = {
                    'patient_temoin': patient_temoin,
                    'diagnostic': diagnostic_raw,
                    'ligne_excel': index + 2  # +2 car header=1 et index commence à 0
                }
            
            log_and_print(f"✅ {len(excel_subjects)} sujets valides trouvés dans le fichier Excel", log_f)
            
            # --- 3. Scan des fichiers dans tous les dossiers ---
            log_and_print(f"\n📁 Scan des dossiers...", log_f)
            
            # Structure: subject_id -> {folder_type: [files]}
            folder_subjects = {}
            total_files = 0
            
            for folder_name, folder_config in directories_to_scan.items():
                folder_path = os.path.join(base_directory, folder_name)
                
                if not os.path.exists(folder_path):
                    log_and_print(f"⚠️  Dossier non trouvé: {folder_path}", log_f)
                    continue
                
                # Scanner le dossier et ses sous-dossiers
                for root, dirs, files in os.walk(folder_path):
                    fif_files = [f for f in files if f.endswith('.fif') and not f.startswith('._')]
                    
                    for filename in fif_files:
                        total_files += 1
                        # Extraire l'ID du sujet
                        subject_id = None
                        
                        for pattern in folder_config["patterns"]:
                            if pattern in filename:
                                subject_id = filename.replace(pattern, '')
                                break
                        
                        if subject_id:
                            if subject_id not in folder_subjects:
                                folder_subjects[subject_id] = {}
                            
                            folder_type = folder_config["type"]
                            if folder_type not in folder_subjects[subject_id]:
                                folder_subjects[subject_id][folder_type] = []
                            
                            folder_subjects[subject_id][folder_type].append({
                                'filename': filename,
                                'folder': folder_name,
                                'path': os.path.join(root, filename)
                            })
            
            log_and_print(f"📄 {total_files} fichiers .fif trouvés au total", log_f)
            log_and_print(f"✅ {len(folder_subjects)} sujets uniques identifiés dans les fichiers", log_f)
            
            # --- 4. Comparaison et analyse ---
            log_and_print("\n" + "="*70, log_f)
            log_and_print("📊 ANALYSE COMPARATIVE", log_f)
            log_and_print("="*70, log_f)
            
            # Sujets dans Excel mais AUCUN fichier trouvé
            sujets_aucun_fichier = []
            for subject_id in excel_subjects:
                if subject_id not in folder_subjects:
                    sujets_aucun_fichier.append(subject_id)
            
            # Sujets dans les fichiers mais pas dans Excel
            sujets_fichiers_orphelins = []
            for subject_id in folder_subjects:
                if subject_id not in excel_subjects:
                    sujets_fichiers_orphelins.append(subject_id)
            
            # Sujets avec au moins un fichier (peu importe le protocole)
            sujets_avec_fichiers = []
            for subject_id in excel_subjects:
                if subject_id in folder_subjects:
                    sujets_avec_fichiers.append(subject_id)
            
            # --- 5. Affichage des résultats ---
            log_and_print(f"\n🟢 SUJETS AVEC FICHIERS ({len(sujets_avec_fichiers)}):", log_f)
            if sujets_avec_fichiers:
                for subject_id in sorted(sujets_avec_fichiers):
                    excel_info = excel_subjects[subject_id]
                    files_info = folder_subjects[subject_id]
                    log_and_print(f"   ✅ {subject_id}", log_f)
                    log_and_print(f"      -> Excel: {excel_info['patient_temoin']} | {excel_info['diagnostic']} (ligne {excel_info['ligne_excel']})", log_f)
                    
                    # Afficher par protocole/type
                    total_files_for_subject = 0
                    for protocol_type, file_list in files_info.items():
                        total_files_for_subject += len(file_list)
                        log_and_print(f"      -> {protocol_type}: {len(file_list)} fichier(s)", log_f)
                        for file_info in file_list:
                            log_and_print(f"         - {file_info['filename']} ({file_info['folder']})", log_f)
                    
                    log_and_print(f"      -> Total: {total_files_for_subject} fichier(s)", log_f)
                    log_and_print("", log_f)
            
            log_and_print(f"\n🔴 SUJETS SANS AUCUN FICHIER ({len(sujets_aucun_fichier)}):", log_f)
            if sujets_aucun_fichier:
                for subject_id in sorted(sujets_aucun_fichier):
                    excel_info = excel_subjects[subject_id]
                    log_and_print(f"   ❌ {subject_id}", log_f)
                    log_and_print(f"      -> Excel: {excel_info['patient_temoin']} | {excel_info['diagnostic']} (ligne {excel_info['ligne_excel']})", log_f)
                    log_and_print(f"      -> Aucun fichier trouvé dans tous les protocoles", log_f)
                    log_and_print("", log_f)
            
            log_and_print(f"\n🟡 FICHIERS ORPHELINS (absents de l'Excel) ({len(sujets_fichiers_orphelins)}):", log_f)
            if sujets_fichiers_orphelins:
                for subject_id in sorted(sujets_fichiers_orphelins):
                    files_info = folder_subjects[subject_id]
                    log_and_print(f"   ⚠️  {subject_id}", log_f)
                    
                    total_files_orphan = 0
                    for protocol_type, file_list in files_info.items():
                        total_files_orphan += len(file_list)
                        log_and_print(f"      -> {protocol_type}: {len(file_list)} fichier(s)", log_f)
                        for file_info in file_list:
                            log_and_print(f"         - {file_info['filename']} ({file_info['folder']})", log_f)
                    
                    log_and_print(f"      -> Total: {total_files_orphan} fichier(s)", log_f)
                    log_and_print("", log_f)
            
            # --- 6. Résumé statistique ---
            log_and_print("\n" + "="*70, log_f)
            log_and_print("📈 RÉSUMÉ STATISTIQUE", log_f)
            log_and_print("="*70, log_f)
            log_and_print(f"📋 Total sujets dans Excel: {len(excel_subjects)}", log_f)
            log_and_print(f"📁 Total sujets avec fichiers: {len(folder_subjects)}", log_f)
            log_and_print(f"✅ Sujets avec au moins un fichier: {len(sujets_avec_fichiers)}", log_f)
            log_and_print(f"❌ Sujets sans aucun fichier: {len(sujets_aucun_fichier)}", log_f)
            log_and_print(f"⚠️  Fichiers orphelins (non dans Excel): {len(sujets_fichiers_orphelins)}", log_f)
            
            taux_completude = (len(sujets_avec_fichiers) / len(excel_subjects)) * 100 if excel_subjects else 0
            log_and_print(f"📊 Taux de couverture (au moins 1 fichier): {taux_completude:.1f}%", log_f)
            
            # --- 7. Analyse par protocole ---
            log_and_print(f"\n📋 ANALYSE PAR PROTOCOLE:", log_f)
            protocol_stats = {}
            
            for subject_id, protocols in folder_subjects.items():
                for protocol_type, file_list in protocols.items():
                    if protocol_type not in protocol_stats:
                        protocol_stats[protocol_type] = {'subjects': set(), 'files': 0}
                    protocol_stats[protocol_type]['subjects'].add(subject_id)
                    protocol_stats[protocol_type]['files'] += len(file_list)
            
            for protocol, stats in sorted(protocol_stats.items()):
                log_and_print(f"   {protocol}:", log_f)
                log_and_print(f"      - {len(stats['subjects'])} sujets", log_f)
                log_and_print(f"      - {stats['files']} fichiers", log_f)
            
            # --- 8. Section spéciale LOG: Liste complète des sujets sans fichiers ---
            log_and_print("\n" + "="*70, log_f)
            log_and_print("📝 LISTE SUJETS SANS AUCUN FICHIER (priorité absolue)", log_f)
            log_and_print("="*70, log_f)
            
            if sujets_aucun_fichier:
                log_and_print("\n🔴 SUJETS À TRAITER EN PRIORITÉ (aucun fichier trouvé):", log_f)
                for i, subject_id in enumerate(sorted(sujets_aucun_fichier), 1):
                    excel_info = excel_subjects[subject_id]
                    log_and_print(f"{i:3d}. {subject_id:<15} | {excel_info['patient_temoin']:<10} | {excel_info['diagnostic']:<15} | Ligne Excel: {excel_info['ligne_excel']}", log_f)
            else:
                log_and_print("\n🎉 Tous les sujets de l'Excel ont au moins un fichier !", log_f)
            
            log_and_print(f"\n🎉 Analyse terminée! Résultats sauvegardés dans: {log_file}", log_f)
        
        print(f"\n📄 Log détaillé sauvegardé dans: {log_file}")
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        # Sauvegarder l'erreur dans le log aussi
        try:
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n❌ ERREUR: {e}\n")
        except:
            pass


def analyser_manquants_detail():
    """
    Analyse détaillée des types de fichiers manquants pour chaque sujet.
    """
    
    excel_file = "tableau_tous_les_patients_13-1-25.xlsx"
    base_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/"
    
    # Création du fichier log détaillé
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"analyse_detaillee_manquants_{timestamp}.log"
    
    # Configuration des protocoles (même que dans compare_sujets_excel)
    directories_to_scan = {
        # Protocole LG (Local Global)
        "LG_CONTROLS_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_CONTROLS"
        },
        "LG_PATIENTS_DELIRIUM+_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_DELIRIUM+"
        },
        "LG_PATIENTS_DELIRIUM-_0.5": {
            "patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", 
                        "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"],
            "type": "LG_DELIRIUM-"
        },
        # Protocole PP (Passive Paradigm)
        "PP_CONTROLS_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_CONTROLS"
        },
        "PP_PATIENTS_DELIRIUM+_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_DELIRIUM+"
        },
        "PP_PATIENTS_DELIRIUM-_0.5": {
            "patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", 
                        "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"],
            "type": "PP_DELIRIUM-"
        },
        # Protocole PPAP (études coma, VS, MCS)
        "PP_CONTROLS_COMA_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_CONTROLS"
        },
        "PP_COMA_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_COMA"
        },
        "PP_COMA_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_COMA"
        },
        "PP_VS_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_VS"
        },
        "PP_VS_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_VS"
        },
        "PP_MCS+_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS+"
        },
        "PP_MCS+_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS+"
        },
        "PP_MCS-_01HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS-"
        },
        "PP_MCS-_1HZ": {
            "patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", 
                        "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"],
            "type": "PPAP_MCS-"
        }
    }
    
    def log_and_print(message, log_file_handle=None):
        """Affiche le message et l'écrit dans le log"""
        print(message)
        if log_file_handle:
            log_file_handle.write(message + "\n")
    
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            # En-tête du log
            log_and_print("=" * 70, log_f)
            log_and_print("🔍 ANALYSE DÉTAILLÉE DES FICHIERS MANQUANTS", log_f)
            log_and_print(f"📅 Date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", log_f)
            log_and_print("=" * 70, log_f)
            log_and_print(f"📁 Dossier base: {base_directory}", log_f)
            log_and_print(f"📋 Fichier Excel: {excel_file}", log_f)
            log_and_print("=" * 70, log_f)
            
            # Lecture Excel
            df = pd.read_excel(excel_file, header=1)
            excel_subjects = []
            excel_subjects_info = {}
            
            for index, row in df.iterrows():
                patient_id = str(row.get('ID_sujet', '')).strip()
                if patient_id and patient_id.lower() not in ['n/a', 'sortie', 'pas fait', 'debutt lizette', 'debut batterie a lyon']:
                    excel_subjects.append(patient_id)
                    excel_subjects_info[patient_id] = {
                        'patient_temoin': str(row.get('Patient / Témoin', '')).strip(),
                        'diagnostic': str(row.get('Diagnostic', '')).strip(),
                        'ligne_excel': index + 2
                    }
            
            log_and_print(f"\n📋 {len(excel_subjects)} sujets à analyser\n", log_f)
            
            # Scan des fichiers existants
            existing_files = {}
            for folder_name, folder_config in directories_to_scan.items():
                folder_path = os.path.join(base_directory, folder_name)
                
                if os.path.exists(folder_path):
                    for root, dirs, files in os.walk(folder_path):
                        fif_files = [f for f in files if f.endswith('.fif') and not f.startswith('._')]
                        
                        for filename in fif_files:
                            # Extraire l'ID du sujet
                            for pattern in folder_config["patterns"]:
                                if pattern in filename:
                                    subject_id = filename.replace(pattern, '')
                                    
                                    if subject_id not in existing_files:
                                        existing_files[subject_id] = {}
                                    
                                    protocol_type = folder_config["type"]
                                    if protocol_type not in existing_files[subject_id]:
                                        existing_files[subject_id][protocol_type] = []
                                    
                                    existing_files[subject_id][protocol_type].append({
                                        'filename': filename,
                                        'folder': folder_name,
                                        'pattern': pattern
                                    })
                                    break
            
            # Compteurs pour statistiques
            total_sujets = len(excel_subjects)
            sujets_avec_fichiers = 0
            sujets_sans_fichiers = 0
            
            # Statistiques par protocole
            protocol_stats = {}
            
            for subject_id in sorted(excel_subjects):
                excel_info = excel_subjects_info[subject_id]
                log_and_print(f"📋 Sujet: {subject_id}", log_f)
                log_and_print(f"   Info Excel: {excel_info['patient_temoin']} | {excel_info['diagnostic']} (ligne {excel_info['ligne_excel']})", log_f)
                
                if subject_id in existing_files:
                    sujets_avec_fichiers += 1
                    files_info = existing_files[subject_id]
                    
                    total_files_subject = 0
                    for protocol_type, file_list in files_info.items():
                        total_files_subject += len(file_list)
                        
                        # Mise à jour des statistiques
                        if protocol_type not in protocol_stats:
                            protocol_stats[protocol_type] = {'subjects': set(), 'files': 0}
                        protocol_stats[protocol_type]['subjects'].add(subject_id)
                        protocol_stats[protocol_type]['files'] += len(file_list)
                        
                        log_and_print(f"   ✅ {protocol_type}: {len(file_list)} fichier(s)", log_f)
                        for file_info in file_list:
                            log_and_print(f"      - {file_info['filename']} ({file_info['folder']})", log_f)
                    
                    log_and_print(f"   📊 Total: {total_files_subject} fichier(s) dans {len(files_info)} protocole(s)", log_f)
                    
                else:
                    sujets_sans_fichiers += 1
                    log_and_print(f"   ❌ Aucun fichier trouvé", log_f)
                
                log_and_print("", log_f)  # Ligne vide
            
            # Résumé des statistiques
            log_and_print("=" * 70, log_f)
            log_and_print("📊 STATISTIQUES DÉTAILLÉES", log_f)
            log_and_print("=" * 70, log_f)
            
            log_and_print(f"📋 Total sujets analysés: {total_sujets}", log_f)
            log_and_print(f"✅ Sujets avec fichiers: {sujets_avec_fichiers} ({sujets_avec_fichiers/total_sujets*100:.1f}%)", log_f)
            log_and_print(f"❌ Sujets sans fichiers: {sujets_sans_fichiers} ({sujets_sans_fichiers/total_sujets*100:.1f}%)", log_f)
            
            log_and_print("\n📋 STATISTIQUES PAR PROTOCOLE:", log_f)
            for protocol, stats in sorted(protocol_stats.items()):
                log_and_print(f"   {protocol}:", log_f)
                log_and_print(f"      ✅ Sujets: {len(stats['subjects'])}", log_f)
                log_and_print(f"      📄 Fichiers: {stats['files']}", log_f)
            
            # Liste des sujets prioritaires à traiter
            log_and_print("\n" + "="*70, log_f)
            log_and_print("🎯 SUJETS PRIORITAIRES À TRAITER", log_f)
            log_and_print("="*70, log_f)
            
            if sujets_sans_fichiers > 0:
                log_and_print(f"\n🔴 PRIORITÉ HAUTE - Sujets sans aucun fichier ({sujets_sans_fichiers}):", log_f)
                for subject_id in sorted(excel_subjects):
                    if subject_id not in existing_files:
                        excel_info = excel_subjects_info[subject_id]
                        log_and_print(f"   ❌ {subject_id} | {excel_info['patient_temoin']} | {excel_info['diagnostic']}", log_f)
            
            # Sujets avec fichiers partiels (par protocole)
            protocoles_incomplets = {}
            for subject_id in existing_files:
                if subject_id in excel_subjects_info:
                    files_info = existing_files[subject_id]
                    # Analyser si certains protocoles sont incomplets
                    for protocol_type, file_list in files_info.items():
                        # Compter les patterns attendus vs trouvés
                        patterns_config = None
                        for folder_name, folder_config in directories_to_scan.items():
                            if folder_config["type"] == protocol_type:
                                patterns_config = folder_config["patterns"]
                                break
                        
                        if patterns_config:
                            patterns_found = set()
                            for file_info in file_list:
                                patterns_found.add(file_info['pattern'])
                            
                            if len(patterns_found) < len(patterns_config):
                                if protocol_type not in protocoles_incomplets:
                                    protocoles_incomplets[protocol_type] = []
                                protocoles_incomplets[protocol_type].append({
                                    'subject_id': subject_id,
                                    'found': len(patterns_found),
                                    'expected': len(patterns_config),
                                    'excel_info': excel_subjects_info[subject_id]
                                })
            
            if protocoles_incomplets:
                log_and_print(f"\n🟡 PRIORITÉ MOYENNE - Sujets avec protocoles incomplets:", log_f)
                for protocol_type, incomplete_list in protocoles_incomplets.items():
                    log_and_print(f"\n   📋 {protocol_type} ({len(incomplete_list)} sujets):", log_f)
                    for item in incomplete_list:
                        subject_id = item['subject_id']
                        excel_info = item['excel_info']
                        log_and_print(f"      ⚠️  {subject_id} ({item['found']}/{item['expected']}) | {excel_info['patient_temoin']} | {excel_info['diagnostic']}", log_f)
            
            log_and_print(f"\n🎉 Analyse détaillée terminée! Log sauvegardé dans: {log_file}", log_f)
        
        print(f"\n📄 Log détaillé sauvegardé dans: {log_file}")
                
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        # Sauvegarder l'erreur dans le log aussi
        try:
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n❌ ERREUR: {e}\n")
        except:
            pass


def generer_csv_manquants():
    """
    Génère un fichier CSV avec la liste des sujets manquants pour faciliter le suivi.
    """
    
    excel_file = "tableau_tous_les_patients_13-1-25.xlsx"
    base_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/"
    
    # Création du fichier CSV avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"sujets_manquants_{timestamp}.csv"
    
    # Configuration des protocoles
    directories_to_scan = {
        "LG_CONTROLS_0.5": {"patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"], "type": "LG_CONTROLS"},
        "LG_PATIENTS_DELIRIUM+_0.5": {"patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"], "type": "LG_DELIRIUM+"},
        "LG_PATIENTS_DELIRIUM-_0.5": {"patterns": ["_LG_preproc_noICA_LG-epo_ar.fif", "_LG_preproc_ICA_LG-epo_ar.fif", "_LG_preproc_noICA_LG-epo.fif", "_LG_preproc_ICA_LG-epo.fif"], "type": "LG_DELIRIUM-"},
        "PP_CONTROLS_0.5": {"patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"], "type": "PP_CONTROLS"},
        "PP_PATIENTS_DELIRIUM+_0.5": {"patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"], "type": "PP_DELIRIUM+"},
        "PP_PATIENTS_DELIRIUM-_0.5": {"patterns": ["_PP_preproc_noICA_PP-epo_ar.fif", "_PP_preproc_ICA_PP-epo_ar.fif", "_PP_preproc_noICA_PP-epo.fif", "_PP_preproc_ICA_PP-epo.fif"], "type": "PP_DELIRIUM-"},
        "PP_CONTROLS_COMA_01HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_CONTROLS"},
        "PP_COMA_01HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_COMA"},
        "PP_COMA_1HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_COMA"},
        "PP_VS_01HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_VS"},
        "PP_VS_1HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_VS"},
        "PP_MCS+_01HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_MCS+"},
        "PP_MCS+_1HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_MCS+"},
        "PP_MCS-_01HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_MCS-"},
        "PP_MCS-_1HZ": {"patterns": ["_preproc_noICA_PPAP-epo_ar.fif", "_preproc_ICA_PPAP-epo_ar.fif", "_preproc_noICA_PPAP-epo.fif", "_preproc_ICA_PPAP-epo.fif"], "type": "PPAP_MCS-"}
    }
    
    try:
        # Lecture Excel
        df = pd.read_excel(excel_file, header=1)
        
        # Scan des fichiers existants
        existing_files = {}
        for folder_name, folder_config in directories_to_scan.items():
            folder_path = os.path.join(base_directory, folder_name)
            
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    fif_files = [f for f in files if f.endswith('.fif') and not f.startswith('._')]
                    
                    for filename in fif_files:
                        for pattern in folder_config["patterns"]:
                            if pattern in filename:
                                subject_id = filename.replace(pattern, '')
                                
                                if subject_id not in existing_files:
                                    existing_files[subject_id] = {}
                                
                                protocol_type = folder_config["type"]
                                if protocol_type not in existing_files[subject_id]:
                                    existing_files[subject_id][protocol_type] = 0
                                
                                existing_files[subject_id][protocol_type] += 1
                                break
        
        # Préparer les données pour le CSV
        csv_data = []
        
        for index, row in df.iterrows():
            patient_id = str(row.get('ID_sujet', '')).strip()
            patient_temoin = str(row.get('Patient / Témoin', '')).strip()
            diagnostic_raw = str(row.get('Diagnostic', '')).strip()
            
            # Ignorer les lignes invalides
            if not patient_id or patient_id.lower() in ['n/a', 'sortie', 'pas fait', 'debutt lizette', 'debut batterie a lyon']:
                continue
            
            # Analyser les fichiers pour ce sujet
            total_files = 0
            protocols_with_files = []
            protocol_details = {}
            
            if patient_id in existing_files:
                for protocol_type, file_count in existing_files[patient_id].items():
                    total_files += file_count
                    protocols_with_files.append(protocol_type)
                    protocol_details[protocol_type] = file_count
            
            # Déterminer le statut
            if total_files == 0:
                statut = "AUCUN_FICHIER"
                priorite = "HAUTE"
            elif len(protocols_with_files) == 1:
                statut = "PROTOCOLE_UNIQUE"
                priorite = "BASSE"  # Changé de MOYENNE à BASSE car au moins 1 fichier = bon
            elif len(protocols_with_files) > 1:
                statut = "MULTI_PROTOCOLES"
                priorite = "AUCUNE"  # Changé de BASSE à AUCUNE car parfait
            else:
                statut = "PARTIEL"
                priorite = "BASSE"  # Changé de MOYENNE à BASSE car au moins 1 fichier = bon
            
            # Créer la ligne CSV
            csv_row = {
                'ID_sujet': patient_id,
                'Patient_Temoin': patient_temoin,
                'Diagnostic': diagnostic_raw,
                'Ligne_Excel': index + 2,
                'Statut': statut,
                'Priorite': priorite,
                'Total_Fichiers': total_files,
                'Nb_Protocoles': len(protocols_with_files),
                'Protocoles': ', '.join(protocols_with_files) if protocols_with_files else 'AUCUN'
            }
            
            # Ajouter les détails par protocole
            all_protocol_types = set()
            for folder_config in directories_to_scan.values():
                all_protocol_types.add(folder_config["type"])
            
            for protocol_type in sorted(all_protocol_types):
                csv_row[f'{protocol_type}_fichiers'] = protocol_details.get(protocol_type, 0)
            
            csv_data.append(csv_row)
        
        # Créer le DataFrame et sauvegarder
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📊 Fichier CSV généré: {csv_file}")
        print(f"📋 {len(csv_data)} sujets analysés")
        
        # Statistiques rapides
        aucun = len([row for row in csv_data if row['Statut'] == 'AUCUN_FICHIER'])
        unique = len([row for row in csv_data if row['Statut'] == 'PROTOCOLE_UNIQUE'])
        multi = len([row for row in csv_data if row['Statut'] == 'MULTI_PROTOCOLES'])
        avec_fichiers = unique + multi
        
        print(f"❌ Aucun fichier: {aucun}")
        print(f"✅ Avec fichiers: {avec_fichiers} (dont {unique} protocole unique, {multi} multi-protocoles)")
        print(f"📊 Taux de couverture: {(avec_fichiers/len(csv_data)*100):.1f}%")
        
        return csv_file
        
    except Exception as e:
        print(f"❌ ERREUR lors de la génération du CSV: {e}")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 COMPARAISON SUJETS EXCEL vs FICHIERS 🔍")
    print("=" * 70)
    
    print("\nChoisissez une option:")
    print("1. Analyse comparative complète (avec log)")
    print("2. Analyse détaillée des fichiers manquants (avec log)")
    print("3. Générer un fichier CSV des sujets manquants")
    print("4. Toutes les analyses (comparaison + détaillée + CSV)")
    
    choix = input("\nVotre choix (1/2/3/4): ").strip()
    
    if choix == "1":
        compare_sujets_excel()
    elif choix == "2":
        analyser_manquants_detail()
    elif choix == "3":
        generer_csv_manquants()
    elif choix == "4":
        print("\n🔄 Exécution de toutes les analyses...\n")
        compare_sujets_excel()
        print("\n" + "="*70)
        analyser_manquants_detail()
        print("\n" + "="*70)
        generer_csv_manquants()
    else:
        print("❌ Choix invalide. Lancement de l'analyse complète par défaut.")
        compare_sujets_excel()
