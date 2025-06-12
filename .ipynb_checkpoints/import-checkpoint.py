#!/bin/bash

echo "=== DIAGNOSTIC DES IMPORTS MANQUANTS ==="
echo

# 1. Localiser le fichier qui contient execute_single_subject_decoding
echo "1. Recherche de execute_single_subject_decoding:"
find /home/tom.balay/Baking_EEG -name "*.py" -exec grep -l "def execute_single_subject_decoding" {} \;
echo

# 2. Recherche de références à _4_decoding_core
echo "2. Fichiers référençant _4_decoding_core:"
find /home/tom.balay/Baking_EEG -name "*.py" -exec grep -l "_4_decoding_core" {} \;
echo

# 3. Recherche de EVENT_ID_LG
echo "3. Fichiers référençant EVENT_ID_LG:"
find /home/tom.balay/Baking_EEG -name "*.py" -exec grep -l "EVENT_ID_LG" {} \;
echo

# 4. Lister tous les modules disponibles dans le projet
echo "4. Structure du projet Baking_EEG:"
find /home/tom.balay/Baking_EEG -type f -name "*.py" | head -20
echo

# 5. Vérifier le contenu du dossier examples
echo "5. Contenu du dossier examples:"
ls -la /home/tom.balay/Baking_EEG/examples/
echo

# 6. Rechercher des fichiers avec des noms similaires à decoding
echo "6. Fichiers contenant 'decoding' dans le nom:"
find /home/tom.balay/Baking_EEG -name "*decoding*" -type f
echo

# 7. Rechercher des fichiers avec des noms similaires à 'core'
echo "7. Fichiers contenant 'core' dans le nom:"
find /home/tom.balay/Baking_EEG -name "*core*" -type f
echo