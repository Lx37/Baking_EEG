#!/usr/bin/env python
"""Verifie que l'arborescence de donnees montee localement est exploitable
par Baking_EEG, avant de lancer le pipeline.

Usage :
    python check_data_tree.py [chemin_racine_donnees]

Sans argument, utilise le chemin renvoye par configure_project_paths(getuser()).
Sortie : par groupe et par sujet, le fichier -epo.fif trouve (ou l'absence),
le protocole detecte a partir du nom de fichier, et un recapitulatif final.
"""
import os
import sys
from getpass import getuser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Le script fonctionne qu'il soit place a cote du paquet (…/check_data_tree.py
# + …/Baking_EEG/config/) ou a l'interieur (…/Baking_EEG/check_data_tree.py).
for _candidate in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, "Baking_EEG")):
    if os.path.isdir(os.path.join(_candidate, "config")):
        REPO = _candidate
        break
else:
    print("ERREUR: dossier 'config/' introuvable a cote de ce script.")
    sys.exit(1)
sys.path.insert(0, REPO)

from config.config import ALL_SUBJECTS_GROUPS  # noqa: E402
from utils.utils import configure_project_paths  # noqa: E402
from utils.loading_PP_utils import (  # noqa: E402
    find_epochs_file_with_protocol_detection,
)

# Correspondance groupe -> sous-dossier, recopiee de loading_PP_utils /
# loading_LG_utils (elle n'y est pas exposee sous forme de table reutilisable).
PP_DIRS = {
    "CONTROLS": "PP_CONTROLS_0.5",
    "CONTROLS_DELIRIUM": "PP_CONTROLS_0.5",
    "CONTROLS_COMA": "PP_CONTROLS_COMA_01HZ",
    "COMA": "PP_COMA_01HZ",
    "MCS": "PP_MCS_01HZ",
    "VS": "PP_VS_01HZ",
    "DELIRIUM+": "PP_PATIENTS_DELIRIUM+_0.5",
    "DELIRIUM-": "PP_PATIENTS_DELIRIUM-_0.5",
}
LG_DIRS = {
    "CONTROLS": "LG_CONTROLS_0.5",
    "CONTROLS_DELIRIUM": "LG_CONTROLS_0.5",
    "CONTROLS_COMA": "LG_CONTROLS_0.5",
    "COMA": "LG_COMA",
    "MCS": "LG_MCS",
    "VS": "LG_VS",
    "DELIRIUM+": "LG_PATIENTS_DELIRIUM+_0.5",
    "DELIRIUM-": "LG_PATIENTS_DELIRIUM-_0.5",
}
LG_SUFFIXES = ["ICA_LG", "noICA_LG"]


def find_lg_file(data_root, subject_id):
    """Reproduit la recherche de fichier LG de loading_LG_utils."""
    for s_id in (subject_id, subject_id.replace("Tp", "")):
        for suffix in LG_SUFFIXES:
            base = f"{s_id}_LG_preproc_{suffix}"
            for cand in (f"{base}-epo_ar.fif", f"{base}-epo.fif"):
                full = os.path.join(data_root, cand)
                if os.path.exists(full):
                    return full
    return None


def main():
    if len(sys.argv) > 1:
        base = sys.argv[1]
    else:
        base, _ = configure_project_paths(getuser())

    print(f"Racine des donnees : {base}")
    if not os.path.isdir(base):
        print("  -> INTROUVABLE. Monte les donnees a cet emplacement, "
              "ou passe le bon chemin en argument.")
        return 1
    print()

    totals = {"pp_ok": 0, "pp_ko": 0, "lg_ok": 0, "lg_ko": 0}
    protocols = {}
    missing_dirs = []

    for group, subjects in ALL_SUBJECTS_GROUPS.items():
        pp_dir = os.path.join(base, PP_DIRS.get(group, "???"))
        lg_dir = os.path.join(base, LG_DIRS.get(group, "???"))
        pp_exists = os.path.isdir(pp_dir)
        lg_exists = os.path.isdir(lg_dir)
        if not pp_exists:
            missing_dirs.append(pp_dir)
        if not lg_exists:
            missing_dirs.append(lg_dir)

        print(f"=== {group}  ({len(subjects)} sujets)")
        print(f"    PP: {PP_DIRS.get(group)} {'OK' if pp_exists else 'ABSENT'}"
              f"   |   LG: {LG_DIRS.get(group)} {'OK' if lg_exists else 'ABSENT'}")

        for subj in subjects:
            pp_path, pp_proto = (
                find_epochs_file_with_protocol_detection(pp_dir, subj, False)
                if pp_exists else (None, None)
            )
            lg_path = find_lg_file(lg_dir, subj) if lg_exists else None

            if pp_path:
                totals["pp_ok"] += 1
                protocols[pp_proto] = protocols.get(pp_proto, 0) + 1
            else:
                totals["pp_ko"] += 1
            if lg_path:
                totals["lg_ok"] += 1
            else:
                totals["lg_ko"] += 1

            if not pp_path or not lg_path:
                manque = []
                if not pp_path:
                    manque.append("PP")
                if not lg_path:
                    manque.append("LG")
                print(f"      {subj:12s} MANQUE: {'+'.join(manque)}")

    print()
    print("=== RECAPITULATIF")
    print(f"  PP trouves : {totals['pp_ok']:3d}   manquants : {totals['pp_ko']:3d}")
    print(f"  LG trouves : {totals['lg_ok']:3d}   manquants : {totals['lg_ko']:3d}")
    print(f"  Protocoles PP detectes (par nom de fichier) : {protocols}")
    if missing_dirs:
        print("  Dossiers de groupe absents :")
        for d in sorted(set(missing_dirs)):
            print(f"    - {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
