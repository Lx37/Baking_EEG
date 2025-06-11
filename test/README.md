# STRUCTURE DU DOSSIER TEST

Ce dossier contient tous les tests et validations pour le projet Baking_EEG.

## Organisation

### 📁 Core Tests (Tests principaux)
- `test_simple_import.py` - Test des imports de base  
- `test_loading_utils.py` - Test des utilitaires de chargement de données
- `test_loading_lg.py` - Test spécifique au protocole Local-Global
- `test_run_group_pp.py` - Test du script d'analyse de groupe

### 📁 Validation (Scripts de validation)
- `simple_validation.py` - Validation simple du projet
- `validate_project.py` - Validation complète du projet  
- `final_validation_complete.py` - Validation finale extensive
- `validation_finale_simple.py` - Validation finale simplifiée

### 📁 Demos (Scripts de démonstration)
- `demo_final.py` - Démonstration finale des fonctionnalités
- `test_comprehensive.py` - Tests complets et démonstrations

### 📁 Utilities (Utilitaires de test)
- `fix_format.py` - Correction du format du code
- `fix_long_lines.py` - Correction des lignes trop longues

### 📁 Reports (Rapports)
- `final_validation_report.json` - Rapport JSON de validation
- `validation_final_simple.json` - Rapport JSON simplifié
- `CORRECTIONS_SUMMARY.md` - Résumé des corrections
- `RAPPORT_FINAL_CORRECTIONS.md` - Rapport final des corrections
- `RAPPORT_FINAL_VALIDATION.md` - Rapport final de validation
- `INDEX_COMPLET.md` - Index complet du projet

## Utilisation

### Tests rapides
```bash
python test/test_simple_import.py
python test/test_loading_lg.py
python test/test_run_group_pp.py
```

### Validation complète
```bash
python test/validate_project.py
python test/final_validation_complete.py
```

### Démonstration
```bash
python test/demo_final.py
```

## Maintenance

Pour ajouter un nouveau test :
1. Créer le fichier dans la catégorie appropriée
2. Suivre le modèle des tests existants
3. Mettre à jour ce README
4. Ajouter le test aux scripts de validation globale
