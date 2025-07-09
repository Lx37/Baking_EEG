# === IMPORTS ===
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

# Configuration du logging
logger_base_decoding = logging.getLogger(__name__)

# === CLASSIFIER PIPELINE BUILDER ===


def _build_standard_classifier_pipeline(
    classifier_model_type="svc",
    random_seed_state=42,
    use_grid_search=False,
    add_csp_step=False,
    add_anova_fs_step=False,
    svc_c=1.0, svc_kernel='linear', svc_gamma='scale',
    logreg_c=1.0, logreg_penalty='l2',
    rf_n_estimators=100, rf_max_depth=None,
    fs_percentile=15,
    csp_n_components=4
):
    """
    Construit un pipeline de classification scikit-learn.

    Pipeline: Scaler -> [CSP (optionnel)] -> [ANOVA FS (optionnel)] ->
    Classifier.
    Les hyperparamètres sont pris des arguments si use_grid_search=False,
    sinon ils sont généralement optimisés par GridSearchCV.

    Args:
        classifier_model_type (str): 'svc', 'logreg', ou 'rf'.
        random_seed_state (int): État aléatoire pour la reproductibilité.
        use_grid_search (bool): Si True, les paramètres fixes sont des
            valeurs par défaut.
        add_csp_step (bool): Si True, ajoute CSP.
        add_anova_fs_step (bool): Si True, ajoute la sélection de
            caractéristiques ANOVA.
        svc_c, svc_kernel, ...: Hyperparamètres fixes pour les composants
            si pas de GS.

    Returns:
        tuple: (pipeline, nom_étape_classifier, nom_étape_anova_fs,
                nom_étape_csp)
    """
    log_prefix = ("Pipeline de base pour GridSearchCV" if use_grid_search
                  else "Pipeline à hyperparamètres fixes")
    logger_base_decoding.info(
        "%s: type='%s', CSP: %s, ANOVA FS: %s",
        log_prefix, classifier_model_type, add_csp_step, add_anova_fs_step
    )

    steps = [("scaler", StandardScaler())]
    pipeline_clf_name = ""
    anova_fs_name = None
    csp_name = None


    logger_base_decoding.info("DEBUG: add_csp_step type=%s, value=%s, "
                               "bool()=%s",
                               type(add_csp_step).__name__,
                               repr(add_csp_step),
                               bool(add_csp_step))

    if add_csp_step:
        csp_name = "csp_feature_extraction"
        n_comp = csp_n_components if not use_grid_search else 4
        steps.append((csp_name, CSP(n_components=n_comp, reg=None,
                                    log=True, norm_trace=False)))
        log_detail = ("n_components à optimiser" if use_grid_search
                      else f"n_components fixe={csp_n_components}")
        logger_base_decoding.info(
            "   Étape CSP '%s' ajoutée (%s).", csp_name, log_detail)

    if add_anova_fs_step:
        anova_fs_name = "anova_feature_selection"
        percentile_val = fs_percentile if not use_grid_search else 20
        steps.append((anova_fs_name, SelectPercentile(f_classif,
                                                      percentile=percentile_val)))
        log_detail_fs = ("percentile à optimiser" if use_grid_search
                          else f"percentile fixe={fs_percentile}")
        logger_base_decoding.info("   Étape ANOVA FS '%s' ajoutée (%s).",
                               anova_fs_name, log_detail_fs)

    clf_params = {'probability': True, 'class_weight': "balanced",
                  'random_state': random_seed_state}
    if classifier_model_type.lower() == "svc":
        pipeline_clf_name = "svc_classifier"
        svc_args = {'C': svc_c, 'kernel': svc_kernel, 'gamma': svc_gamma}
        if use_grid_search:
            svc_args = {'C': 1.0, 'kernel': 'linear', 'gamma': 'scale'}
        classifier = SVC(**svc_args, **clf_params)
    elif classifier_model_type.lower() == "logreg":
        pipeline_clf_name = "logreg_classifier"
        logreg_args = {'C': logreg_c, 'penalty': logreg_penalty,
                       'solver': "liblinear", 'max_iter': 2000}
        if use_grid_search:
            logreg_args = {'C': 1.0, 'penalty': 'l2',
                           'solver': "liblinear", 'max_iter': 2000}
        classifier = LogisticRegression(**logreg_args, **clf_params)
    elif classifier_model_type.lower() == "rf":
        pipeline_clf_name = "rf_classifier"
        rf_args = {'n_estimators': rf_n_estimators,
                   'max_depth': rf_max_depth, 'n_jobs': 1}
        if use_grid_search:
            rf_args = {'n_estimators': 100, 'max_depth': None, 'n_jobs': 1}
        del clf_params['probability']
        classifier = RandomForestClassifier(**rf_args, **clf_params)
    else:
        raise ValueError(
            f"Type de classifieur inconnu: '{classifier_model_type}'.")

    steps.append((pipeline_clf_name, classifier))
    logger_base_decoding.info("Structure finale du pipeline: %s", [
                           s[0] for s in steps])
    return Pipeline(steps), pipeline_clf_name, anova_fs_name, csp_name

# === FONCTIONS D'EXÉCUTION DU DÉCODAGE GLOBAL ===


def _execute_global_decoding_for_one_fold(
    estimator_for_fold,
    epochs_data_flat_fold,
    target_labels_encoded_fold,
    train_indices,
    test_indices,
    trial_sample_weights_fold,
    clf_step_name_in_pipeline_for_gs=None
):
    """Exécute le décodage global pour un seul fold CV (helper pour la parallélisation).
    
    Args:
        trial_sample_weights_fold: Can be None, np.ndarray (global weights), or "auto" (fold-specific)
    """
    x_train = epochs_data_flat_fold[train_indices]
    x_test = epochs_data_flat_fold[test_indices]
    y_train = target_labels_encoded_fold[train_indices]
    y_test = target_labels_encoded_fold[test_indices]

    fit_params = {}
    is_gs = isinstance(estimator_for_fold, GridSearchCV)

    # Handle sample weights
    sample_weights_train = None
    sample_weights_test = None
    
    if trial_sample_weights_fold == "auto":
        # Calculate fold-specific weights
        sample_weights_train = calculate_fold_sample_weights(y_train)
        sample_weights_test = calculate_fold_sample_weights(y_test)  # For scoring
    elif isinstance(trial_sample_weights_fold, np.ndarray):
        # Use provided global weights
        sample_weights_train = trial_sample_weights_fold[train_indices]
        sample_weights_test = trial_sample_weights_fold[test_indices]

    # Set up fit parameters for training
    if sample_weights_train is not None:
        if is_gs:
            if clf_step_name_in_pipeline_for_gs:
                fit_params[f"estimator__{clf_step_name_in_pipeline_for_gs}__sample_weight"] = sample_weights_train
        else:
            final_estimator_name = estimator_for_fold.steps[-1][0]
            fit_params[f"{final_estimator_name}__sample_weight"] = sample_weights_train

    try:
        if fit_params:
            estimator_for_fold.fit(x_train, y_train, **fit_params)
        else:
            estimator_for_fold.fit(x_train, y_train)
    except Exception as e_fit:
        logger_base_decoding.error("Erreur pendant l'apprentissage dans le fold: %s",
                                e_fit, exc_info=True)
        return np.array([]), np.array([]), np.nan

    if is_gs:
        logger_base_decoding.debug("Fold Global - Meilleurs paramètres GS: %s, Score: %.4f",
                                estimator_for_fold.best_params_,
                                estimator_for_fold.best_score_)

    predicted_probas_test = estimator_for_fold.predict_proba(x_test)
    predicted_labels_test = estimator_for_fold.predict(x_test)
    roc_auc_fold = np.nan

    if len(np.unique(y_test)) > 1 and predicted_probas_test.shape[1] >= 2:
        try:
            roc_auc_fold = roc_auc_score(
                y_test, predicted_probas_test[:, 1],
                sample_weight=sample_weights_test,  # Use fold-specific weights for scoring too
                average="weighted"
            )
        except ValueError as e_auc:
            logger_base_decoding.debug(
                "Erreur de calcul ROC AUC dans le fold: %s", e_auc)
    elif len(np.unique(y_test)) <= 1:
        logger_base_decoding.debug(
            "ROC AUC ignoré (classe unique dans y_test pour le fold).")

    return predicted_probas_test, predicted_labels_test, roc_auc_fold


def calculate_fold_sample_weights(train_labels_enc):
    """Calculate sample weights for a specific training fold.
    
    This ensures that class balancing is computed based on the actual
    class distribution in each training fold, which is more accurate
    than using global weights when using StratifiedKFold.
    
    Args:
        train_labels_enc (np.ndarray): Encoded labels for training fold
        
    Returns:
        np.ndarray: Sample weights for the training fold
    """
    n_trials_fold = len(train_labels_enc)
    unique_classes = np.unique(train_labels_enc)
    n_classes_fold = len(unique_classes)
    
    if n_classes_fold < 2:
        # Single class, return uniform weights
        return np.ones(n_trials_fold)
    
    # Count occurrences of each class in this fold
    class_counts_fold = np.bincount(train_labels_enc, minlength=max(unique_classes) + 1)
    
    # Calculate balanced weights: inverse frequency weighting
    weights_map_fold = {}
    for cls_idx in unique_classes:
        count = class_counts_fold[cls_idx]
        if count > 0:
            weights_map_fold[cls_idx] = n_trials_fold / (n_classes_fold * count)
        else:
            weights_map_fold[cls_idx] = 0.0
    
    # Apply weights to each sample
    sample_weights_fold = np.array([
        weights_map_fold[label] for label in train_labels_enc
    ])
    
    return sample_weights_fold
