#Tom
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

from config.decoding_config import CLASS_BALANCING_STRATEGY

logger_base_decoding = logging.getLogger(__name__)

# === CLASSIFIER PIPELINE BUILDER ===


def _build_standard_classifier_pipeline(
    classifier_model_type="svc",
    random_seed_state=42,
    use_grid_search=False,
    add_anova_fs_step=False,
    svc_c=1.0,
    svc_kernel='linear',
    svc_gamma='scale',
    logreg_c=1.0,
    logreg_penalty='l2',
    rf_n_estimators=100,
    rf_max_depth=None,
    fs_percentile=15,
    class_balancing_strategy=None,
):
    """
    Construit un pipeline de classification scikit-learn.

    Pipeline: Scaler -> [ANOVA FS (optionnel)] -> Classifier.
    Les hyperparametres sont pris des arguments si use_grid_search=False,
    sinon ils sont generalement optimises par GridSearchCV.

    Rebalancement des classes
    -------------------------
    Un SEUL mecanisme de rebalancement est actif a la fois, pilote par
    `class_balancing_strategy` (defaut: `CLASS_BALANCING_STRATEGY` de la config) :

      - "class_weight"  : le classifieur recoit class_weight="balanced".
                          C'est la strategie par defaut, et la seule qui
                          s'applique de maniere identique au decodage temporel
                          (SlidingEstimator/GeneralizingEstimator, ou l'on ne
                          peut pas passer de sample_weight par fold) et au
                          decodage global. Les deux AUC sont donc comparables.
      - "sample_weight" : pas de class_weight ; des poids equilibres sont
                          calcules par fold et passes a fit().
      - "none"          : aucun rebalancement.

    Cumuler class_weight et sample_weight revenait a appliquer la correction
    deux fois (scikit-learn multiplie les deux) - c'etait le comportement des
    versions <= 2026-01 et il est corrige ici.

    Args:
        classifier_model_type (str): 'svc', 'logreg', ou 'rf'.
        random_seed_state (int): Etat aleatoire pour la reproductibilite.
        use_grid_search (bool): Si True, les parametres fixes sont des
            valeurs par defaut.
        add_anova_fs_step (bool): Si True, ajoute la selection de
            caracteristiques ANOVA.
        svc_c, svc_kernel, ...: Hyperparametres fixes pour les composants
            si pas de GS.
        class_balancing_strategy (str, optional): surcharge ponctuelle de
            CLASS_BALANCING_STRATEGY.

    Returns:
        tuple: (pipeline, nom_etape_classifier, nom_etape_anova_fs, None)
    """
    strategy = (class_balancing_strategy or CLASS_BALANCING_STRATEGY).lower()
    if strategy not in ("class_weight", "sample_weight", "none"):
        logger_base_decoding.warning(
            "class_balancing_strategy='%s' inconnue (attendu: class_weight|"
            "sample_weight|none). Retour a 'class_weight'.", strategy
        )
        strategy = "class_weight"

    log_prefix = ("Pipeline de base pour GridSearchCV" if use_grid_search
                  else "Pipeline a hyperparametres fixes")
    logger_base_decoding.info(
        "%s: type='%s', ANOVA FS: %s, rebalancement: %s",
        log_prefix, classifier_model_type, add_anova_fs_step, strategy
    )

    steps = [("scaler", StandardScaler())]
    pipeline_clf_name = ""
    anova_fs_name = None

    if add_anova_fs_step:
        anova_fs_name = "anova_feature_selection"
        percentile_val = fs_percentile if not use_grid_search else 20
        steps.append((anova_fs_name, SelectPercentile(f_classif,
                                                      percentile=percentile_val)))
        log_detail_fs = ("percentile a optimiser" if use_grid_search
                         else f"percentile fixe={fs_percentile}")
        logger_base_decoding.info("   Etape ANOVA FS '%s' ajoutee (%s).",
                                  anova_fs_name, log_detail_fs)

    # class_weight n'est applique que si c'est la strategie retenue.
    class_weight_value = "balanced" if strategy == "class_weight" else None
    clf_params = {'probability': True, 'class_weight': class_weight_value,
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
        clf_params_lr = dict(clf_params)
        del clf_params_lr['probability']  # LogisticRegression n'a pas ce param
        classifier = LogisticRegression(**logreg_args, **clf_params_lr)
    elif classifier_model_type.lower() == "rf":
        pipeline_clf_name = "rf_classifier"
        rf_args = {'n_estimators': rf_n_estimators,
                   'max_depth': rf_max_depth, 'n_jobs': 1}
        if use_grid_search:
            rf_args = {'n_estimators': 100, 'max_depth': None, 'n_jobs': 1}
        clf_params_rf = dict(clf_params)
        del clf_params_rf['probability']  # RF expose predict_proba nativement
        classifier = RandomForestClassifier(**rf_args, **clf_params_rf)
    else:
        raise ValueError(
            f"Type de classifieur inconnu: '{classifier_model_type}'.")

    steps.append((pipeline_clf_name, classifier))
    logger_base_decoding.info("Structure finale du pipeline: %s",
                              [s[0] for s in steps])
    return Pipeline(steps), pipeline_clf_name, anova_fs_name, None

# === FONCTIONS D'EXECUTION DU DECODAGE GLOBAL ===


def _execute_global_decoding_for_one_fold(
    estimator_for_fold,
    epochs_data_flat_fold,
    target_labels_encoded_fold,
    train_indices,
    test_indices,
    trial_sample_weights_fold=None,
    clf_step_name_in_pipeline_for_gs=None,
    class_balancing_strategy=None,
):
    """Execute le decodage global pour un seul fold CV (helper de parallelisation).

    Args:
        trial_sample_weights_fold: None, np.ndarray (poids globaux, decoupes
            selon train/test), ou la chaine "auto" (poids recalcules par fold).
            Ignore si la strategie de rebalancement n'est pas "sample_weight".
        clf_step_name_in_pipeline_for_gs (str): nom de l'etape classifieur,
            requis pour router sample_weight a travers un GridSearchCV.

    Returns:
        tuple: (probas_test, labels_predits_test, roc_auc_du_fold)

    Notes:
        Le ROC AUC du fold est calcule SANS ponderation, afin d'etre sur la
        meme echelle que l'AUC temporelle produite par cross_val_multiscore
        (qui n'accepte pas de sample_weight). Cf. docstring de
        _build_standard_classifier_pipeline.
    """
    strategy = (class_balancing_strategy or CLASS_BALANCING_STRATEGY).lower()

    x_train = epochs_data_flat_fold[train_indices]
    x_test = epochs_data_flat_fold[test_indices]
    y_train = target_labels_encoded_fold[train_indices]
    y_test = target_labels_encoded_fold[test_indices]

    fit_params = {}
    is_gs = isinstance(estimator_for_fold, GridSearchCV)

    sample_weights_train = None
    if strategy == "sample_weight":
        # isinstance(..., str) AVANT toute comparaison : comparer un ndarray a
        # "auto" renvoie un tableau de booleens et leve ensuite
        # "truth value of an array ... is ambiguous".
        if isinstance(trial_sample_weights_fold, str):
            if trial_sample_weights_fold.lower() == "auto":
                sample_weights_train = calculate_fold_sample_weights(y_train)
            else:
                logger_base_decoding.warning(
                    "trial_sample_weights_fold='%s' non reconnu ; poids ignores.",
                    trial_sample_weights_fold
                )
        elif isinstance(trial_sample_weights_fold, np.ndarray):
            sample_weights_train = trial_sample_weights_fold[train_indices]
    elif isinstance(trial_sample_weights_fold, np.ndarray):
        logger_base_decoding.debug(
            "sample_weights fournis mais strategie='%s' : ignores "
            "(le rebalancement passe par class_weight).", strategy
        )

    if sample_weights_train is not None:
        # Les fit_params d'un Pipeline (meme encapsule dans un GridSearchCV)
        # s'ecrivent "<nom_etape>__<param>". Le prefixe "estimator__" utilise
        # precedemment faisait echouer TOUS les fits, silencieusement convertis
        # en NaN par le except ci-dessous.
        step_name = (clf_step_name_in_pipeline_for_gs if is_gs
                     else estimator_for_fold.steps[-1][0])
        if step_name:
            fit_params[f"{step_name}__sample_weight"] = sample_weights_train
        else:
            logger_base_decoding.warning(
                "Nom d'etape classifieur inconnu : sample_weight non transmis."
            )

    try:
        if fit_params:
            estimator_for_fold.fit(x_train, y_train, **fit_params)
        else:
            estimator_for_fold.fit(x_train, y_train)
    except Exception as e_fit:
        logger_base_decoding.error(
            "Erreur pendant l'apprentissage dans le fold: %s",
            e_fit, exc_info=True)
        return np.array([]), np.array([]), np.nan

    if is_gs:
        logger_base_decoding.debug(
            "Fold Global - Meilleurs parametres GS: %s, Score: %.4f",
            estimator_for_fold.best_params_, estimator_for_fold.best_score_)

    predicted_probas_test = estimator_for_fold.predict_proba(x_test)
    predicted_labels_test = estimator_for_fold.predict(x_test)
    roc_auc_fold = np.nan

    if len(np.unique(y_test)) > 1 and predicted_probas_test.shape[1] >= 2:
        try:
            roc_auc_fold = roc_auc_score(y_test, predicted_probas_test[:, 1])
        except ValueError as e_auc:
            logger_base_decoding.debug(
                "Erreur de calcul ROC AUC dans le fold: %s", e_auc)
    elif len(np.unique(y_test)) <= 1:
        logger_base_decoding.debug(
            "ROC AUC ignore (classe unique dans y_test pour le fold).")

    return predicted_probas_test, predicted_labels_test, roc_auc_fold


def calculate_fold_sample_weights(train_labels_enc):
    """Calcule des poids d'echantillons equilibres pour un fold donne.

    Les poids sont calcules sur la distribution reelle des classes du fold,
    ce qui est plus juste que des poids globaux avec RepeatedStratifiedKFold.

    N'est appele que si CLASS_BALANCING_STRATEGY == "sample_weight" (sinon le
    rebalancement passe par class_weight="balanced", cf.
    _build_standard_classifier_pipeline).

    Args:
        train_labels_enc (np.ndarray): labels encodes du fold d'apprentissage.

    Returns:
        np.ndarray: poids, un par echantillon du fold.
    """
    train_labels_enc = np.asarray(train_labels_enc)
    n_trials_fold = len(train_labels_enc)
    unique_classes = np.unique(train_labels_enc)
    n_classes_fold = len(unique_classes)

    if n_classes_fold < 2:
        return np.ones(n_trials_fold)

    # minlength = max(classe)+1 : indispensable si les classes encodees ne sont
    # pas exactement 0..n-1.
    class_counts_fold = np.bincount(train_labels_enc,
                                    minlength=int(np.max(unique_classes)) + 1)

    weights_map_fold = {}
    for cls_idx in unique_classes:
        count = class_counts_fold[cls_idx]
        weights_map_fold[cls_idx] = (
            n_trials_fold / (n_classes_fold * count) if count > 0 else 0.0
        )

    return np.array([weights_map_fold[label] for label in train_labels_enc])
