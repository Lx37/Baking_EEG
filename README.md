# Baking EEG - Analyse de Décodage EEG

Ce projet contient des scripts pour l'analyse de décodage EEG des données de l'expérience de cuisson.

## Structure du Projet

```
Baking_EEG/
├── config/
│   └── decoding_config.py      # Configuration du décodage
├── utils/
│   ├── decoding_utils.py       # Utilitaires de décodage
│   ├── stats_utils.py          # Utilitaires statistiques
│   └── visualization_utils.py  # Utilitaires de visualisation
├── scripts/
│   ├── decoding.py            # Fonctions principales de décodage
│   ├── run_decoding.py        # Script d'exécution du décodage
│   └── decoding_stats.py      # Script d'analyse statistique
└── README.md
```

## Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/votre-username/Baking_EEG.git
cd Baking_EEG
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Décodage Temporel

Pour exécuter le décodage temporel sur les données d'un sujet :

```bash
python scripts/run_decoding.py \
    --data_path /chemin/vers/donnees \
    --output_path /chemin/vers/sortie \
    --subject_id sub-01 \
    --classifier svc \
    --use_grid_search \
    --use_csp \
    --use_anova \
    --n_jobs 4
```

Options disponibles :
- `--data_path` : Chemin vers les données EEG
- `--output_path` : Répertoire de sortie pour les résultats
- `--subject_id` : Identifiant du sujet (ex: sub-01)
- `--classifier` : Type de classifieur (svc, logistic, rf)
- `--use_grid_search` : Utiliser GridSearchCV pour l'optimisation
- `--use_csp` : Utiliser CSP pour la sélection de caractéristiques
- `--use_anova` : Utiliser ANOVA pour la sélection de caractéristiques
- `--n_jobs` : Nombre de jobs pour le parallélisme

### Analyse Statistique

Pour effectuer l'analyse statistique sur les résultats de décodage :

```bash
python scripts/decoding_stats.py \
    --results_dir /chemin/vers/resultats \
    --output_dir /chemin/vers/sortie
```

Options disponibles :
- `--results_dir` : Répertoire contenant les résultats de décodage
- `--output_dir` : Répertoire de sortie pour les résultats statistiques

## Configuration

Les paramètres de configuration sont définis dans `config/decoding_config.py` :

- Paramètres des classifieurs
- Paramètres de validation croisée
- Paramètres statistiques
- Paramètres de visualisation

## Modules

### Utilitaires de Décodage (`utils/decoding_utils.py`)

- Construction de pipelines de classification
- Préparation des poids d'échantillons
- Calcul des métriques de performance

### Utilitaires Statistiques (`utils/stats_utils.py`)

- Tests de permutation par clusters
- Correction FDR
- Comparaison au niveau de chance
- Calcul d'intervalles de confiance
- Calcul de taille d'effet

### Utilitaires de Visualisation (`utils/visualization_utils.py`)

- Visualisation des résultats temporels
- Matrices de confusion
- Importance des caractéristiques
- Résultats cross-subject

## Contribution

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Créer une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 