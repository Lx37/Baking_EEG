# === BASE CONSTANTS ===
CHANCE_LEVEL_AUC = 0.5
P_THRESHOLD = 0.05
N_PERMUTATIONS = 1000
DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL = "svc"
# For MNE's internal parallelism within Sliding/Generalizing Estimators
INTERNAL_N_JOBS_FOR_MNE_DECODING = -1

# Event mapping for different LG condition comparisons:
# LSGD versus LDGD (Local Standard Global Deviant vs Local Deviant
# Global Deviant)
# LSGS versus LSGD (Local Standard Global Standard vs Local Standard
# Global Deviant)
# LDGS versus LDGD (Local Deviant Global Standard vs Local Deviant
# Global Deviant)
# LSGS versus LDGS (Local Standard Global Standard vs Local Deviant
# Global Standard)
EVENT_ID_LG = {
    'LS/GS': 11,  # Local Standard, Global Standard
    'LS/GD': 12,  # Local Standard, Global Deviant
    'LD/GS': 21,  # Local Deviant, Global Standard
    'LD/GD': 22   # Local Deviant, Global Deviant
}



# Local Global protocol event mapping
# Data loading configuration for Local-Global protocol
CONFIG_LOAD_MAIN_LG_DECODING = {
    "LS_ALL": "LS/",  # Load all Local Standard events
    "LD_ALL": "LD/",  # Load all Local Deviant events
    "GS_ALL": "/GS",  # Load all Global Standard events
    "GD_ALL": "/GD",  # Load all Global Deviant events
}

# Specific comparisons for LG analysis
CONFIG_LOAD_LG_COMPARISONS = {
    **CONFIG_LOAD_MAIN_LG_DECODING,
    "LSGS": ["LS/GS"],  # Local Standard Global Standard
    "LSGD": ["LS/GD"],  # Local Standard Global Deviant
    "LDGS": ["LD/GS"],  # Local Deviant Global Standard
    "LDGD": ["LD/GD"],  # Local Deviant Global Deviant
}

# Complete configuration for single subject LG analysis
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT_LG = {
    **CONFIG_LOAD_MAIN_LG_DECODING,
    **CONFIG_LOAD_LG_COMPARISONS,
}

# === GLOBAL ANALYSIS CONFIGURATION ===
USE_GRID_SEARCH_OPTIMIZATION = False
USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES = True
N_JOBS_PROCESSING = "auto"
CLASSIFIER_MODEL_TYPE = "svc"  # Options: "svc", "logreg", "rf"

# === CLASSIFIER CONFIGURATION ===
# Supported classifier types
SUPPORTED_CLASSIFIER_TYPES = {
    "svc": "Support Vector Classifier",
    "logreg": "Logistic Regression",
    "rf": "Random Forest"
}

# === STATISTICAL TESTING PARAMETERS ===
# For stats on CV folds (single subject)
N_PERMUTATIONS_INTRA_SUBJECT = 1024
# For stats between subjects (group level)
N_PERMUTATIONS_GROUP_LEVEL = 1024

# === STATISTICAL TEST CONFIGURATION ===
# Statistical test for FDR pointwise correction: 'wilcoxon' recommended for non-parametric
# For cluster permutation tests: 'ttest' is MNE default, 'wilcoxon' is experimental
STAT_TEST_FOR_FDR_POINTWISE = "wilcoxon"  # Options: "wilcoxon", "ttest"
STAT_TEST_FOR_CLUSTER_PERMUTATION = "wilcoxon"  # Options: "ttest", "wilcoxon"

# Cluster threshold configuration for statistical tests
GROUP_LEVEL_STAT_THRESHOLD_TYPE = "tfce"  # 'tfce' or 'stat'
# Only used if GROUP_LEVEL_STAT_THRESHOLD_TYPE is 'stat'
T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING = None  # e.g., 2.0 or calculated
# Config for TFCE for intra-subject (CV fold) stats
INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG = {"start": 0.1, "step": 0.1}

# === Analysis step control flags ===
COMPUTE_INTRA_SUBJECT_STATISTICS = True
COMPUTE_GROUP_LEVEL_STATISTICS = True
SAVE_ANALYSIS_RESULTS = True
GENERATE_PLOTS = True
COMPUTE_TEMPORAL_GENERALIZATION_MATRICES = True

# Specific TGM configuration by comparison type
# TGM ONLY for main comparison PP vs AP all
COMPUTE_TGM_FOR_MAIN_COMPARISON = True  # PP/all vs AP/all and LD/GD vs LD/GS and GD vs GS
COMPUTE_TGM_FOR_SPECIFIC_COMPARISONS = False  # PP_spec vs AP_families
COMPUTE_TGM_FOR_INTER_FAMILY_COMPARISONS = False  # AP_family vs AP_family


# =========DECODING CORE =============

USE_GRID_SEARCH = USE_GRID_SEARCH_OPTIMIZATION  
USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES = True

# === Hyperparameter grids for GridSearchCV ===
PARAM_GRID_CONFIG_EXTENDED = {
    "svc": {
        'svc_classifier__C': [0.01, 0.1, 1, 10, 100],
        'svc_classifier__kernel': ['linear', 'rbf'],
        'svc_classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'anova_feature_selection__percentile': [10, 20, 30, 50, 75]
    },
    "logreg": {
        'logreg_classifier__C': [0.01, 0.1, 1, 10, 100],
        'logreg_classifier__penalty': ['l1', 'l2'],
        'anova_feature_selection__percentile': [10, 20, 30, 50, 75]
    },
    "rf": {
        'rf_classifier__n_estimators': [50, 100, 200, 300],
        'rf_classifier__max_depth': [None, 10, 20, 30],
        'anova_feature_selection__percentile': [25, 50, 75, 100]
    }
}
CV_FOLDS_FOR_GRIDSEARCH_INTERNAL = 5

# === Fixed hyperparameters (if USE_GRID_SEARCH_OPTIMIZATION = False) ===
# This configuration replaces the previous one
FIXED_CLASSIFIER_PARAMS_CONFIG = {
    "svc": {
        'svc_c': 1.0,
        'svc_kernel': 'linear',
        'svc_gamma': 'scale',
        'fs_percentile': 15
    },
    "logreg": {
        'logreg_c': 1.0,
        'logreg_penalty': 'l2',
        'fs_percentile': 15
    },
    "rf": {
        'rf_n_estimators': 100,
        'rf_max_depth': None,
        'fs_percentile': 75
    }
}

# === ADVANCED FEATURE SELECTION CONFIGURATION ===
# Feature selection methods (CSP functionality removed)
# ANOVA F-score based feature selection is used instead

# === PROTOCOL-SPECIFIC CONFIGURATIONS ===

# Battery Protocol Configuration
# Event structure: AP/1-6/Music/Conv/D-G/V1-3 and PP/Music/Conv/D-G/V1-3
CONFIG_LOAD_BATTERY_PROTOCOL = {
    "XPP_ALL": "PP/",  # All PP events
    "XAP_ALL": "AP/",  # All AP events
    "PP_FOR_SPECIFIC_COMPARISON": "PP/",  # Use all PP events for specific comparisons vs AP families
    # AP families for Battery protocol
    "AP_FAMILY_1": ["AP/1/Music/", "AP/1/Conv/"],
    "AP_FAMILY_2": ["AP/2/Music/", "AP/2/Conv/"],
    "AP_FAMILY_3": ["AP/3/Music/", "AP/3/Conv/"],
    "AP_FAMILY_4": ["AP/4/Music/", "AP/4/Conv/"],
    "AP_FAMILY_5": ["AP/5/Music/", "AP/5/Conv/"],
    "AP_FAMILY_6": ["AP/6/Music/", "AP/6/Conv/"],
}

# PPext3 Protocol Configuration
# Event structure: AP/1-6/Music-Noise/Conv-Dio/D-G/V1-3 and PP/Music-Noise/Conv-Dio/D-G/V1-3
CONFIG_LOAD_PPEXT3_PROTOCOL = {
    "XPP_ALL": "PP/",  # All PP events
    "XAP_ALL": "AP/",  # All AP events
    "PP_FOR_SPECIFIC_COMPARISON": "PP/",  # Use all PP events for specific comparisons vs AP families
    # AP families for PPext3 protocol
    "AP_FAMILY_1": ["AP/1/Music/", "AP/1/Noise/", "AP/1/Conv/", "AP/1/Dio/"],
    "AP_FAMILY_2": ["AP/2/Music/", "AP/2/Noise/", "AP/2/Conv/", "AP/2/Dio/"],
    "AP_FAMILY_3": ["AP/3/Music/", "AP/3/Noise/", "AP/3/Conv/", "AP/3/Dio/"],
    "AP_FAMILY_4": ["AP/4/Music/", "AP/4/Noise/", "AP/4/Conv/", "AP/4/Dio/"],
    "AP_FAMILY_5": ["AP/5/Music/", "AP/5/Noise/", "AP/5/Conv/", "AP/5/Dio/"],
    "AP_FAMILY_6": ["AP/6/Music/", "AP/6/Noise/", "AP/6/Conv/", "AP/6/Dio/"],
}


# === EVENT CONFIGURATION ===
# Event codes for specific comparisons for delirium
PP_CODES_FOR_SPECIFIC_COMPARISON = ["PP/10", "PP/20", "PP/30"]

# AP families for specific comparisons
AP_FAMILIES_FOR_SPECIFIC_COMPARISON = {
    f"AP_FAMILY_{unit_digit}": [
        f"AP/{decade}{unit_digit}" for decade in [1, 2, 3]
    ]
    for unit_digit in range(1, 7)
}
# Example event_id mapping from original data :
# {'PP/10': 110, 'PP/20': 120, 'PP/30': 130, ...
#  'AP/11': 111, 'AP/12': 112, ..., 'AP/36': 136}

# Data loading configuration
CONFIG_LOAD_MAIN_DECODING = {
    "XPP_ALL": "PP/",  # Load all events starting with "PP/"
    "XAP_ALL": "AP/",  # Load all events starting with "AP/"
}

# Complete configuration for single subject analysis
CONFIG_LOAD_DELIRIUM_PROTOCOL = {
    **CONFIG_LOAD_MAIN_DECODING,
    "PP_FOR_SPECIFIC_COMPARISON": PP_CODES_FOR_SPECIFIC_COMPARISON,
    **AP_FAMILIES_FOR_SPECIFIC_COMPARISON,
}

# Alias for compatibility with existing scripts
CONFIG_LOAD_ALL_NEEDED_FOR_SINGLE_SUBJECT = CONFIG_LOAD_DELIRIUM_PROTOCOL



# === DATASET CONFIGURATION ===
DATASET_CONFIGS = {
    'delirium': {
        'name': 'Delirium protocol',
        'groups': ['DELIRIUM+', 'DELIRIUM-', 'CONTROLS_DELIRIUM'],
        'colors': {
            'DELIRIUM+': '#d62728',
            'DELIRIUM-': '#ff7f0e',
            'CONTROLS_DELIRIUM': '#2ca02c'
        }
    },
    'COMA': {
        'name': 'Coma et pas que',
        'groups': ['COMA', 'VS', 'MCS+', 'MCS-', 'CONTROLS_COMA'],
        'colors': {
            'COMA': '#8b0000',
            'VS': '#ff6347',
            'MCS+': '#4682b4',
            'MCS-': '#87ceeb',
            'CONTROLS_COMA': '#228b22'
        }
    }
}
# Function to get protocol-specific configuration


def get_protocol_config(protocol_type):
    """Get loading configuration for specific protocol type.

    Args:
        protocol_type (str): Protocol type ('delirium', 'battery', 'ppext3')

    Returns:
        dict: Protocol-specific loading configuration
    """
    config_map = {
        'delirium': CONFIG_LOAD_DELIRIUM_PROTOCOL,
        'battery': CONFIG_LOAD_BATTERY_PROTOCOL,
        'ppext3': CONFIG_LOAD_PPEXT3_PROTOCOL
    }
    return config_map.get(protocol_type, CONFIG_LOAD_DELIRIUM_PROTOCOL)

# Protocol-specific AP families configuration


def get_protocol_ap_families(protocol_type):
    """Get AP families configuration for specific protocol.

    Args:
        protocol_type (str): Protocol type ('delirium', 'battery', 'ppext3')

    Returns:
        dict: Protocol-specific AP families configuration
    """
    if protocol_type == 'battery':
        return {
            f"AP_FAMILY_{i}": [f"AP/{i}/Music/", f"AP/{i}/Conv/"]
            for i in range(1, 7)
        }
    elif protocol_type == 'ppext3':
        return {
            f"AP_FAMILY_{i}": [f"AP/{i}/Music/", f"AP/{i}/Noise/",
                               f"AP/{i}/Conv/", f"AP/{i}/Dio/"]
            for i in range(1, 7)
        }
    else:  # delirium
        return AP_FAMILIES_FOR_SPECIFIC_COMPARISON


def get_protocol_pp_comparison_events(protocol_type):
    """Get PP events for specific comparisons based on protocol type.

    Args:
        protocol_type (str): Protocol type ('delirium', 'battery', 'ppext3')

    Returns:
        list: List of PP event codes for specific comparisons
    """
    if protocol_type == 'delirium':
        # Delirium protocol uses specific PP codes
        return PP_CODES_FOR_SPECIFIC_COMPARISON  # ["PP/10", "PP/20", "PP/30"]
    elif protocol_type == 'battery':
        # Battery protocol uses different PP structure
        return ["PP/Music/", "PP/Conv/"]
    elif protocol_type == 'ppext3':
        # PPext3 protocol uses extended PP structure  
        return ["PP/Music/", "PP/Noise/", "PP/Conv/", "PP/Dio/"]
    else:
        # Default to delirium for unknown protocols
        return PP_CODES_FOR_SPECIFIC_COMPARISON
