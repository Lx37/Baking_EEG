# EEG Decoding Pipeline - Documentation des Améliorations

## Vue d'ensemble

Ce pipeline de décodage EEG a été considérablement amélioré avec des fonctionnalités avancées d'analyse et de décodage. Les améliorations incluent la correction des erreurs d'import, l'ajout de l'analyse fréquentielle, la reconstruction de source, l'analyse CSP avancée, et la cartographie topographique.

## Corrections apportées

### 1. Correction des erreurs d'import
- **Problème** : Inconsistance entre les imports `Baking_EEG.` et imports relatifs
- **Solution** : Standardisation des imports avec ajout du chemin projet
- **Fichiers modifiés** :
  - `utils/loading_PP_utils.py`
  - `utils/loading_LG_utils.py` 
  - `examples/run_decoding_one_pp.py`

### 2. Amélioration de l'intégration CSP
- Ajout de gestion d'erreur pour l'import MNE-Python
- Support des paramètres CSP configurables
- Intégration dans tous les scripts de décodage

## Nouveaux scripts et fonctionnalités

### 1. `examples/inspect_event_ids.py`
**Fonctionnalité** : Inspection complète des event_id dans les fichiers EEG

**Utilisation** :
```bash
python examples/inspect_event_ids.py --subject_id SUBJECT_01 --group GROUP_NAME
```

**Fonctionnalités** :
- Analyse des dictionnaires d'événements
- Comptage des trials par condition
- Détection des conditions manquantes
- Rapport détaillé des métadonnées

### 2. `examples/run_frequency_domain_decoding.py`
**Fonctionnalité** : Décodage dans le domaine fréquentiel avec analyse spectrale

**Utilisation** :
```bash
python examples/run_frequency_domain_decoding.py --subject_id SUBJECT_01 --group GROUP_NAME
```

**Fonctionnalités** :
- **Analyse spectrale** : Extraction de la puissance par bande de fréquence
  - Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
- **Connectivité fonctionnelle** : Calcul de cohérence et corrélation inter-bandes
- **CSP fréquentiel** : Application de CSP dans le domaine fréquentiel
- **Classification combinée** : Fusion des différents types de features
- **Visualisations** : Spectrogrammes et matrices de connectivité

**Métriques** :
- AUC pour chaque type de features
- Comparaison des performances
- Sauvegarde des résultats structurés

### 3. `examples/run_source_reconstruction_decoding.py`
**Fonctionnalité** : Décodage avec reconstruction de source et visualisation 3D

**Utilisation** :
```bash
python examples/run_source_reconstruction_decoding.py --subject_id SUBJECT_01 --group GROUP_NAME [--subjects_dir FREESURFER_DIR]
```

**Fonctionnalités** :
- **Reconstruction de source** :
  - Modèle de tête sphérique (par défaut)
  - Support FreeSurfer (si disponible)
  - Solutions forward/inverse
  - Application dSPM, MNE, sLORETA
- **Décodage multi-espace** :
  - Espace capteur (traditionnel)
  - Espace source (avancé)
- **Visualisations** :
  - Cartes topographiques 2D
  - Visualisation 3D des sources (si pyvista disponible)
  - Patterns d'activation temporels
- **Features avancées** :
  - Moyennage par ROI
  - Extraction de fenêtres temporelles
  - Statistiques d'activation

### 4. `examples/run_advanced_csp_decoding.py`
**Fonctionnalité** : Analyse CSP avancée avec optimisation et features multiples

**Utilisation** :
```bash
python examples/run_advanced_csp_decoding.py --subject_id SUBJECT_01 --group GROUP_NAME [--use_spoc] [--no_multi_band]
```

**Fonctionnalités** :
- **Optimisation CSP** :
  - Recherche automatique du nombre optimal de composantes
  - Validation croisée pour l'optimisation
  - Comparaison de paramètres
- **CSP multi-méthodes** :
  - CSP standard
  - CSP avec régularisation (empirical, diagonal)
  - CSP multi-bandes fréquentielles
  - SPoC (Source Power Comodulation)
- **Features avancées** :
  - Statistiques temporelles des composantes
  - Features de variance, asymétrie, kurtosis
  - Combinaisons de features
- **Sélection de features** :
  - SelectKBest (sélection univariée)
  - PCA (réduction de dimensionnalité)
  - Comparaison des méthodes
- **Visualisations** :
  - Patterns et filtres CSP
  - Cartes topographiques des composantes
  - Comparaison des performances

### 5. `examples/run_comprehensive_eeg_analysis.py`
**Fonctionnalité** : Pipeline d'analyse comprehensive combinant toutes les méthodes

**Utilisation** :
```bash
python examples/run_comprehensive_eeg_analysis.py --subject_id SUBJECT_01 --group GROUP_NAME
```

**Fonctionnalités** :
- **Orchestration complète** :
  - Analyse temporelle standard
  - Analyse fréquentielle
  - Reconstruction de source
  - CSP avancé
- **Rapport de synthèse** :
  - Comparaison des performances
  - Recommandations automatiques
  - Métriques consolidées
- **Gestion des résultats** :
  - Organisation structurée des fichiers
  - Sauvegarde centralisée
  - Logs détaillés

## Fonctionnalités transversales

### Gestion des erreurs et robustesse
- **Imports optionnels** : Gestion gracieuse des dépendances manquantes (MNE, scipy, etc.)
- **Validation des données** : Vérification de la cohérence des données d'entrée
- **Logging avancé** : Logs détaillés pour le debugging et le suivi
- **Exception handling** : Gestion robuste des erreurs avec continuation de l'analyse

### Visualisations avancées
- **Cartes topographiques** : Visualisation 2D des patterns spatiaux
- **Spectrogrammes** : Analyse temps-fréquence
- **Visualisations 3D** : Reconstruction de source (si supporté)
- **Matrices de connectivité** : Visualisation des interactions
- **Patterns CSP** : Visualisation des filtres spatiaux

### Métriques et évaluation
- **AUC scores** : Métrique principale pour toutes les analyses
- **Validation croisée** : Évaluation robuste des performances
- **Tests statistiques** : Significativité des résultats
- **Comparaisons** : Benchmarking des différentes méthodes

## Structure des résultats

### Organisation des fichiers
```
results/
├── comprehensive_analysis_SUBJECT_ID/
│   ├── temporal_decoding_results.npy
│   ├── frequency_domain_results.npy
│   ├── source_reconstruction_results.npy
│   ├── advanced_csp_results.npy
│   ├── comprehensive_results_SUBJECT_ID.npy
│   ├── analysis_report_SUBJECT_ID.json
│   └── visualizations/
│       ├── topomaps/
│       ├── csp_patterns/
│       ├── spectrograms/
│       └── source_patterns/
```

### Format des résultats
Tous les résultats sont sauvegardés dans des dictionnaires Python structurés contenant :
- **Métriques de performance** : AUC, accuracy, confusion matrix
- **Données techniques** : Features extraites, modèles entraînés
- **Métadonnées** : Paramètres utilisés, temps de traitement
- **Visualisations** : Figures et graphiques

## Recommandations d'utilisation

### Pour un nouveau sujet
1. **Inspection préliminaire** :
   ```bash
   python examples/inspect_event_ids.py --subject_id NEW_SUBJECT --group GROUP_NAME
   ```

2. **Analyse comprehensive** :
   ```bash
   python examples/run_comprehensive_eeg_analysis.py --subject_id NEW_SUBJECT --group GROUP_NAME
   ```

3. **Analyses spécialisées** (si nécessaire) :
   ```bash
   # CSP avancé avec optimisation
   python examples/run_advanced_csp_decoding.py --subject_id NEW_SUBJECT --group GROUP_NAME
   
   # Reconstruction de source
   python examples/run_source_reconstruction_decoding.py --subject_id NEW_SUBJECT --group GROUP_NAME
   ```

### Pour une étude de groupe
```bash
# Analyser tous les sujets d'un groupe
for subject in SUBJECT_LIST; do
    python examples/run_comprehensive_eeg_analysis.py --subject_id $subject --group GROUP_NAME
done

# Puis analyser le groupe
python examples/run_decoding_one_group_pp.py --group_name GROUP_NAME
```

## Paramètres configurables

### Classificateurs supportés
- `logreg` : Régression logistique (défaut)
- `lda` : Analyse discriminante linéaire

### Options CSP
- Nombre de composantes : 2-12 (optimisation automatique)
- Régularisation : None, empirical, diagonal
- Multi-bandes : Support des bandes de fréquence personnalisées

### Reconstruction de source
- Modèles de tête : Sphérique, FreeSurfer
- Méthodes : MNE, dSPM, sLORETA
- ROIs : Support des atlas anatomiques

## Dépendances

### Requises
- numpy, pandas, scipy
- scikit-learn
- matplotlib

### Optionnelles (avec fallback)
- MNE-Python (pour CSP, reconstruction de source, topomaps)
- pyvista (pour visualisation 3D)
- FreeSurfer (pour modèles anatomiques)

## Notes techniques

### Performance
- **Parallélisation** : Support multi-core pour les analyses coûteuses
- **Cache** : Réutilisation des données chargées
- **Optimisation** : Algorithmes efficaces pour les grandes datasets

### Compatibilité
- **Python 3.7+** : Compatible avec les versions récentes
- **Cross-platform** : Fonctionne sur Linux, macOS, Windows
- **Scalabilité** : Adapté aux petites et grandes études

### Extensibilité
- **Architecture modulaire** : Facile d'ajouter de nouvelles méthodes
- **Configuration centralisée** : Paramètres dans config/
- **API cohérente** : Interface standardisée entre modules

## Troubleshooting

### Problèmes courants
1. **Import MNE** : Installer avec `pip install mne`
2. **Visualisations 3D** : Installer pyvista avec `pip install pyvista`
3. **Mémoire** : Réduire la taille des données ou augmenter la RAM
4. **Performances** : Utiliser `--classifier lda` pour des données de grande dimension

### Support
- Logs détaillés dans `logs_*/`
- Messages d'erreur explicites
- Documentation intégrée dans le code
