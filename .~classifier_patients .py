import pandas as pd
import os
import shutil
from pathlib import Path

def classifier_patients():
    """
    Script pour classer les fichiers de patients dans les bons dossiers selon leur diagnostic.
    VERSION CORRIGÉE ET AMÉLIORÉE.
    """

    # --- Configuration ---
    # Assurez-vous que ces chemins et noms de fichiers sont corrects
    excel_file = "tableau_tous_les_patients_13-1-25.xlsx"
    source_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/ACLASSER1HZ"
    base_output_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/classified_patients1HZ"

    # Mapping des diagnostics vers les noms de dossiers
    diagnostic_folders = {
        'coma': 'PP_COMA_1HZ',
        'mcs-': 'PP_MCS-_1HZ',
        'mcs': 'PP_MCS_1HZ',      # Pour 'MCS', 'MCS+', 'EMCS'
        'vs': 'PP_VG_1HZ',
        'témoins': 'PP_CONTROLS_COMA_1HZ'
    }

    try:
        # --- 1. Vérifications initiales ---
        if not os.path.exists(source_directory):
            print(f"❌ Dossier source non trouvé: {source_directory}")
            return

        print("📖 Lecture du fichier Excel...")
        if not os.path.exists(excel_file):
            print(f"❌ Fichier Excel non trouvé: {excel_file}")
            print("💡 Assurez-vous que le fichier est dans le même dossier que le script.")
            return
            
        # FIX: L'en-tête (header) est sur la première ligne (index 0), pas la deuxième.
        df = pd.read_excel(excel_file, header=1)

        # FIX: Utilisation des noms de colonnes exacts (sensible à la casse et aux espaces)
        required_columns = ['ID_sujet', 'Patient / Témoin', 'Diagnostic']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"❌ Colonnes manquantes dans le fichier Excel: {', '.join(missing_columns)}")
            print(f"📋 Colonnes disponibles: {list(df.columns)}")
            return

        # --- 2. Création de la structure des dossiers ---
        print(f"📁 Vérification/Création des dossiers de destination dans:\n   {base_output_directory}")
        os.makedirs(base_output_directory, exist_ok=True)
        for folder_name in diagnostic_folders.values():
            folder_path = os.path.join(base_output_directory, folder_name)
            os.makedirs(folder_path, exist_ok=True)

        # --- 3. Traitement des fichiers ---
        files_moved = 0
        patients_processed = 0
        patients_skipped = 0

        print("\n🔄 Classification des fichiers en cours...")

        for index, row in df.iterrows():
            # FIX: Utilisation des noms de colonnes corrects pour extraire les données
            patient_id = str(row.get('ID_sujet', '')).strip()
            patient_temoin = str(row.get('Patient / Témoin', '')).strip().lower()
            diagnostic_raw = str(row.get('Diagnostic', '')).strip().lower()

            # Ignorer les lignes invalides ou non pertinentes
            if not patient_id or patient_id.lower() in ['n/a', 'sortie', 'pas fait', 'debutt lizette', 'debut batterie a lyon']:
                continue
            
            patients_processed += 1
            target_diagnostic_key = None

            # --- LOGIQUE DE DIAGNOSTIC AMÉLIORÉE ---
            # Priorité 1: Identifier les témoins
            if 'témoin' in patient_temoin:
                target_diagnostic_key = 'témoins'
            # Priorité 2: Identifier le diagnostic du patient de manière flexible
            else:
                if 'mcs-' in diagnostic_raw:
                    target_diagnostic_key = 'mcs-'
                elif any(term in diagnostic_raw for term in ['mcs+', 'emcs']) or diagnostic_raw == 'mcs':
                    target_diagnostic_key = 'mcs'
                elif 'coma' in diagnostic_raw:
                    target_diagnostic_key = 'coma'
                elif 'vs' in diagnostic_raw:
                    target_diagnostic_key = 'vs'
            
            # Si aucun diagnostic correspondant n'est trouvé, passer à la ligne suivante
            if not target_diagnostic_key:
                if diagnostic_raw not in ['n/a', '', 'xx', 'delirium +', 'delirium -']:
                     print(f"⚠️  Ligne {index+2} ({patient_id}): Diagnostic non géré ('{diagnostic_raw}'). Ligne ignorée.")
                patients_skipped += 1
                continue

            # Définir le dossier de destination basé sur la clé de diagnostic trouvée
            destination_folder = os.path.join(base_output_directory, diagnostic_folders[target_diagnostic_key])

            # Chercher les 4 variantes de nom de fichier pour ce patient
            file_patterns = [
                f"{patient_id}_preproc_noICA_PPAP-epo_ar.fif",
                f"{patient_id}_preproc_ICA_PPAP-epo_ar.fif",
                f"{patient_id}_preproc_noICA_PPAP-epo.fif",
                f"{patient_id}_preproc_ICA_PPAP-epo.fif",
            ]

            patient_files_found = 0
            for file_pattern in file_patterns:
                source_file = os.path.join(source_directory, file_pattern)

                if os.path.exists(source_file):
                    destination_file = os.path.join(destination_folder, os.path.basename(source_file))
                    try:
                        shutil.copy2(source_file, destination_file)
                        print(f"✅ Copié: {file_pattern} → {diagnostic_folders[target_diagnostic_key]}")
                        files_moved += 1
                        patient_files_found += 1
                    except Exception as e:
                        print(f"❌ Erreur lors de la copie de {file_pattern}: {e}")
            
            # Afficher un message si aucun fichier n'a été trouvé pour ce patient
            if patient_files_found == 0:
                print(f"   -> Ligne {index+2} ({patient_id}): Aucun fichier correspondant trouvé dans le dossier source.")

        # --- 4. Résumé final ---
        print("\n" + "="*50)
        print("📊 Résumé de l'opération")
        print("="*50)
        print(f"  - Lignes du fichier Excel analysées (avec ID valide): {patients_processed}")
        print(f"  - Sujets classés: {patients_processed - patients_skipped}")
        print(f"  - Sujets ignorés (diagnostic non reconnu/géré): {patients_skipped}")
        print(f"✅ Fichiers copiés avec succès: {files_moved}")
        print("\n🎉 Classification terminée.")

    except FileNotFoundError:
        print(f"❌ ERREUR CRITIQUE: Fichier Excel non trouvé à l'emplacement: {excel_file}")
    except Exception as e:
        print(f"❌ ERREUR INATTENDUE: {e}")


def verifier_structure():
    """
    Fonction pour vérifier la structure actuelle des fichiers avant de lancer le script.
    """
    print("🔍 Vérification préliminaire de la structure...")

    excel_file = "tableau_tous_les_patients_13-1-25.xlsx"
    source_directory = "/crnldata/cap/users/_tom/Baking_EEG_data/ACLASSER1HZ"

    # Vérification du fichier Excel
    if os.path.exists(excel_file):
        print(f"✔️ Fichier Excel trouvé: {excel_file}")
    else:
        print(f"❌ Fichier Excel NON TROUVÉ: {excel_file}")
        excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
        if excel_files:
            print(f"   -> Fichiers Excel trouvés dans ce dossier: {excel_files}")
            print(f"   -> Suggestion: Renommez votre fichier en '{excel_file}' ou modifiez le script.")

    # Vérification du dossier source
    if os.path.exists(source_directory):
        try:
            source_files = os.listdir(source_directory)
            fif_files = [f for f in source_files if f.endswith('.fif')]
            print(f"✔️ Dossier source trouvé: {source_directory}")
            print(f"   -> {len(fif_files)} fichiers .fif y ont été trouvés.")
            if fif_files:
                print(f"      Exemple: {fif_files[0]}")
        except OSError as e:
            print(f"⚠️  Impossible de lire le contenu de {source_directory}: {e}")
    else:
        print(f"❌ Dossier source NON TROUVÉ: {source_directory}")


if __name__ == "__main__":
    print("=" * 50)
    print("🏥 Script de classification des patients EEG 🏥")
    print("=" * 50)

    # Étape 1: Vérification de l'environnement
    verifier_structure()

    print("\n" + "=" * 50)

    # Étape 2: Demande de confirmation à l'utilisateur
    response = input("Voulez-vous procéder à la classification des fichiers? (o/N): ")

    if response.lower() in ['o', 'oui', 'y', 'yes']:
        classifier_patients()
    else:
        print("\n❌ Opération annulée par l'utilisateur.")
        print("\n📋 Instructions pour utiliser ce script:")
        print("1. Placez votre fichier Excel ('tableau_tous_les_patients_13-1-25.xlsx') dans le même dossier que ce script.")
        print("2. Assurez-vous que les chemins 'source_directory' et 'base_output_directory' sont corrects.")
        print("3. Relancez le script et répondez 'o' pour commencer.")