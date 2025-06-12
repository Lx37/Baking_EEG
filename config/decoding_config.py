# === CONSTANTES DE BASE ===
CHANCE_LEVEL_AUC_SCORE = 0.5
CHANCE_LEVEL_AUC = 0.5  # Pour compatibilité
DEFAULT_CHANCE_LEVEL_AUC = 0.5
DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL = "svc"
# For MNE's internal parallelism within Sliding/Generalizing Estimators
INTERNAL_N_JOBS_FOR_MNE_DECODING = -1

# === CONFIGURATION DES ÉVÉNEMENTS ===
# Codes d'événements pour les comparaisons spécifiques
PP_CODES_FOR_SPECIFIC_COMPARISON = ["PP/10", "PP/20", "PP/30"]

# Familles d'AP pour les comparaisons spécifiques
AP_FAMILIES_FOR_SPECIFIC_COMPARISON = {
    f"AP_FAMILY_{unit_digit}": [
        f"AP/{decade}{unit_digit}" for decade in [1, 2, 3]
    ]
    for unit_digit in range(1, 7)
}
# Example event_id mapping from original data :
# {'PP/10': 110, 'PP/20': 120, 'PP/30': 130, ...
#  'AP/11': 111, 'AP/12': 112, ..., 'AP/36': 136}

# Configuration du chargement des données
CONFIG_LOAD_MAIN_DECODING = {
    "XPP_ALL": "PP/",  # Charge tous les événements commençant par "PP/"
    "XAP_ALL": "AP/",  # Charge tous les événements commençant par "AP/"
}

# Configuration complète pour l'analyse d'un sujet unique
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT = {
    **CONFIG_LOAD_MAIN_DECODING,
    "PP_FOR_SPECIFIC_COMPARISON": PP_CODES_FOR_SPECIFIC_COMPARISON,
    **AP_FAMILIES_FOR_SPECIFIC_COMPARISON,
}
# Local Global protocol event mapping
# LSGD versus LDGD (Local Standard Global Deviant vs Local Deviant Global Deviant)
# LSGS versus LSGD (Local Standard Global Standard vs Local Standard Global Deviant)
# LDGS versus LDGD (Local Deviant Global Standard vs Local Deviant Global Deviant)
# LSGS versus LDGS (Local Standard Global Standard vs Local Deviant Global Standard)
EVENT_ID_LG = {
    'LS/GS': 11,  # Local Standard, Global Standard
    'LS/GD': 12,  # Local Standard, Global Deviant
    'LD/GS': 21,  # Local Deviant, Global Standard
    'LD/GD': 22   # Local Deviant, Global Deviant
}

# Alias pour compatibilité avec les scripts existants (pour bash)
CONFIG_LOAD_SINGLE_PROTOCOL = CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT

# === CONFIGURATION LOCAL-GLOBAL (LG) PROTOCOL ===
# Configuration du chargement des données pour le protocole Local-Global
CONFIG_LOAD_MAIN_LG_DECODING = {
    "LS_ALL": "LS/",  # Charge tous les événements Local Standard
    "LD_ALL": "LD/",  # Charge tous les événements Local Deviant
    "GS_ALL": "/GS",  # Charge tous les événements Global Standard
    "GD_ALL": "/GD",  # Charge tous les événements Global Deviant
}

# Comparaisons spécifiques pour l'analyse LG
CONFIG_LOAD_LG_COMPARISONS = {
    **CONFIG_LOAD_MAIN_LG_DECODING,
    "LSGS": ["LS/GS"],  # Local Standard Global Standard
    "LSGD": ["LS/GD"],  # Local Standard Global Deviant
    "LDGS": ["LD/GS"],  # Local Deviant Global Standard
    "LDGD": ["LD/GD"],  # Local Deviant Global Deviant
}

# Configuration complète pour l'analyse LG d'un sujet unique
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG = {
    **CONFIG_LOAD_MAIN_LG_DECODING,
    **CONFIG_LOAD_LG_COMPARISONS,
}

# === CONFIGURATION GLOBALE DE L'ANALYSE ===
USE_GRID_SEARCH_OPTIMIZATION = False
USE_CSP_FOR_TEMPORAL_PIPELINES = False
USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES = True
N_JOBS_PROCESSING = "auto"
CLASSIFIER_MODEL_TYPE = "svc"

# === PARAMÈTRES DES TESTS STATISTIQUES ===
# Pour les stats sur les CV folds (sujet unique)
N_PERMUTATIONS_INTRA_SUBJECT = 1024
# Pour les stats entre sujets (niveau groupe)
N_PERMUTATIONS_GROUP_LEVEL = 1024

# Configuration du seuillage des clusters dans les tests statistiques
GROUP_LEVEL_STAT_THRESHOLD_TYPE = "tfce"  # 'tfce' ou 'stat'
# Only used if GROUP_LEVEL_STAT_THRESHOLD_TYPE is 'stat'
T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING = None  # e.g., 2.0 or calculated
# Config for TFCE for intra-subject (CV fold) stats
INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG = {"start": 0.1, "step": 0.1}

# === Drapeaux de contrôle pour les étapes d'analyse ===
COMPUTE_INTRA_SUBJECT_STATISTICS = True
COMPUTE_GROUP_LEVEL_STATISTICS = True
SAVE_ANALYSIS_RESULTS = True
GENERATE_PLOTS = True
COMPUTE_TEMPORAL_GENERALIZATION_MATRICES = False


# === Grilles d'hyperparamètres pour GridSearchCV ===
PARAM_GRID_CONFIG_EXTENDED = {
    "svc": {
        'svc_classifier__C': [0.01, 0.1, 1, 10, 100],
        'svc_classifier__kernel': ['linear', 'rbf'],
        'svc_classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'anova_feature_selection__percentile': [10, 20, 30, 50, 75],
        'csp_feature_extraction__n_components': [2, 4, 6, 8, 10, 12]
    },
    "logreg": {
        'logreg_classifier__C': [0.01, 0.1, 1, 10, 100],
        'logreg_classifier__penalty': ['l1', 'l2'],
        'anova_feature_selection__percentile': [10, 20, 30, 50, 75],
        'csp_feature_extraction__n_components': [2, 4, 6, 8, 10, 12]
    },
    "rf": {
        'rf_classifier__n_estimators': [50, 100, 200, 300],
        'rf_classifier__max_depth': [None, 10, 20, 30],
        'anova_feature_selection__percentile': [25, 50, 75, 100],
        'csp_feature_extraction__n_components': [4, 8, 12, 16]
    }
}
CV_FOLDS_FOR_GRIDSEARCH_INTERNAL = 3

# === Hyperparamètres fixes (si USE_GRID_SEARCH_OPTIMIZATION = False) ===
FIXED_CLASSIFIER_PARAMS_CONFIG = {
    "svc": {
        'svc_c': 1.0,
        'svc_kernel': 'linear',
        'svc_gamma': 'scale',
        'fs_percentile': 15,
        'csp_n_components': 4
    },
    "logreg": {
        'logreg_c': 1.0,
        'logreg_penalty': 'l2',
        'fs_percentile': 15,
        'csp_n_components': 4
    },
    "rf": {
        'rf_n_estimators': 100,
        'rf_max_depth': None,
        'fs_percentile': 75,
        'csp_n_components': 8
    }
}
