#tom

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (RepeatedStratifiedKFold, GroupKFold,
                                     GridSearchCV)
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore)
from mne.parallel import parallel_func
from sklearn.base import clone
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score)

from base.base_decoding import (_build_standard_classifier_pipeline,
                                _execute_global_decoding_for_one_fold)
from utils import stats_utils as bEEG_stats

from config.decoding_config import (DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL,
                                    CHANCE_LEVEL_AUC,
                                    INTERNAL_N_JOBS_FOR_MNE_DECODING,
                                    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
                                    USE_GRID_SEARCH,
                                    COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
                                    CLASS_BALANCING_STRATEGY,
                                    AGGREGATE_CV_REPEATS_BEFORE_STATS,
                                    STAT_TEST_FOR_FDR_POINTWISE,
                                    STAT_TEST_FOR_CLUSTER_PERMUTATION)

logger_decoding_core = logging.getLogger(__name__)


def _get_cv_repeat_structure(cv_splitter, n_total_splits):
    """Retourne (n_repeats, n_splits_par_repetition) pour un splitter donne.

    RepeatedStratifiedKFold enchaine les repetitions : les folds sortent dans
    l'ordre [rep0_split0 ... rep0_split(k-1), rep1_split0 ...]. On peut donc
    remodeler un tableau de folds en (n_repeats, k, ...) puis moyenner sur
    l'axe des repetitions.

    Renvoie (1, n_total_splits) si le splitter n'a pas de repetitions.
    """
    n_repeats = int(getattr(cv_splitter, "n_repeats", 1) or 1)
    if n_repeats <= 1 or n_total_splits % n_repeats != 0:
        return 1, n_total_splits
    return n_repeats, n_total_splits // n_repeats


def _average_over_cv_repeats(fold_array, n_repeats):
    """Moyenne un tableau (n_folds, ...) sur les repetitions du CV.

    Args:
        fold_array (np.ndarray): premier axe = folds, ordonnes par repetition.
        n_repeats (int): nombre de repetitions.

    Returns:
        np.ndarray: (n_splits_par_repetition, ...) - ou l'entree inchangee si
        n_repeats <= 1.
    """
    if n_repeats <= 1 or fold_array is None or fold_array.size == 0:
        return fold_array
    n_folds = fold_array.shape[0]
    if n_folds % n_repeats != 0:
        return fold_array
    reshaped = fold_array.reshape((n_repeats, n_folds // n_repeats)
                                  + fold_array.shape[1:])
    return np.nanmean(reshaped, axis=0)


def _warn_if_underpowered(n_observations, n_features, test_type, alpha=0.05):
    """Avertit si le test demande ne peut PAS atteindre la significativite.

    Un Wilcoxon des rangs signes unilateral sur n observations a une p-value
    minimale atteignable de 1/2**n (cas ou toutes les differences sont du meme
    signe). Avec n=5 -> p_min = 0.031 : aucun point ne peut survivre a une
    correction FDR sur des centaines de points temporels. Le resultat "aucun
    point significatif" est alors une limite du plan de test, pas une absence
    d'effet - il faut le dire explicitement.
    """
    if str(test_type).lower() != "wilcoxon" or n_observations <= 0:
        return
    p_min = 0.5 ** n_observations
    if p_min > alpha:
        logger_decoding_core.warning(
            "Wilcoxon sur n=%d observations : p-value minimale atteignable = "
            "%.4f > alpha=%.2f. AUCUN point ne peut etre declare significatif. "
            "Augmentez le nombre de splits CV ou utilisez "
            "STAT_TEST_FOR_FDR_POINTWISE='ttest'.",
            n_observations, p_min, alpha)
    elif p_min > alpha / max(n_features, 1):
        logger_decoding_core.warning(
            "Wilcoxon sur n=%d observations : p-value minimale atteignable = "
            "%.4f. Sur %d points testes, la correction FDR ne laissera passer "
            "un point que si l'effet est present sur une large portion de la "
            "fenetre. Puissance intra-sujet tres limitee.",
            n_observations, p_min, n_features)


def run_temporal_decoding_analysis(
    epochs_data,  # (n_trials, n_channels, n_times)
    target_labels,  # (n_trials,)
    classifier_model_type=DEFAULT_CLASSIFIER_TYPE_MODULE_INTERNAL,
    use_grid_search=USE_GRID_SEARCH,
    use_anova_fs_for_temporal_pipelines=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
    param_grid_config=None,
    cv_folds_for_gridsearch=3,
    fixed_classifier_params=None,
    cross_validation_splitter=5,
    trial_sample_weights="auto",
    n_jobs_external=1,
    group_labels_for_cv=None,
    compute_intra_fold_stats=True,
    chance_level=CHANCE_LEVEL_AUC,
    n_permutations_for_intra_fold_clusters=1000,
    compute_temporal_generalization_matrix=COMPUTE_TEMPORAL_GENERALIZATION_MATRICES,
    cluster_threshold_config_intra_fold=None,
    random_state=42,
    aggregate_cv_repeats=None,
):
    """
    Coeur du decodage : temporel 1D, TGM, et decodage global.

    Points methodologiques
    ----------------------
    1. Rebalancement des classes : pilote par CLASS_BALANCING_STRATEGY
       (defaut "class_weight"). Un seul mecanisme est actif, cf.
       base.base_decoding._build_standard_classifier_pipeline.

    2. Repetitions de validation croisee : avec
       RepeatedStratifiedKFold(n_splits=k, n_repeats=r), chaque essai est
       teste r fois. Ces k*r folds ne sont PAS des observations
       independantes. Si `aggregate_cv_repeats` est vrai (defaut :
       AGGREGATE_CV_REPEATS_BEFORE_STATS) :
         - les scores par fold sont moyennes sur les r repetitions avant les
           tests statistiques intra-sujet (n effectif = k) ;
         - les probabilites predites sont moyennees sur les repetitions au
           lieu d'etre ecrasees par la derniere.

    3. Test statistique : le test utilise pour la correction FDR ponctuelle et
       pour la permutation par cluster provient de la configuration
       (STAT_TEST_FOR_FDR_POINTWISE / STAT_TEST_FOR_CLUSTER_PERMUTATION) et le
       champ "method" des resultats reflete le test reellement execute.

    Returns:
        tuple de 12 elements (ordre inchange par rapport aux versions
        precedentes) :
          0  probas_global_agg      (n_trials, n_classes)
          1  labels_global_agg      (n_trials,)
          2  cv_global_scores       AUC par fold (moyennees sur repetitions)
          3  mean_scores_1d         (n_times,)
          4  global_metrics         dict
          5  fdr_1d_data            dict
          6  cluster_1d_data        dict
          7  scores_1d_all_folds    (n_folds_effectifs, n_times)
          8  mean_tgm               (n_times, n_times)
          9  fdr_tgm_data           dict
          10 None                   (reserve : clusters TGM)
          11 tgm_all_folds          (n_folds_effectifs, n_times, n_times)
    """
    if aggregate_cv_repeats is None:
        aggregate_cv_repeats = AGGREGATE_CV_REPEATS_BEFORE_STATS

    # Certains appelants historiques transmettent un tuple a la place d'un
    # booleen pour ce drapeau.
    if isinstance(use_anova_fs_for_temporal_pipelines, tuple):
        logger_decoding_core.warning(
            "use_anova_fs_for_temporal_pipelines recu sous forme de tuple "
            "(%s) : conversion en booleen.", use_anova_fs_for_temporal_pipelines
        )
        use_anova_fs_for_temporal_pipelines = (
            bool(use_anova_fs_for_temporal_pipelines[0])
            if use_anova_fs_for_temporal_pipelines else False
        )

    logger_decoding_core.info("--- Analyse de decodage temporel ---")
    logger_decoding_core.info(
        "Clf: %s, GS: %s, ANOVA FS (temporel): %s, rebalancement: %s",
        classifier_model_type, use_grid_search,
        use_anova_fs_for_temporal_pipelines, CLASS_BALANCING_STRATEGY
    )

    if not isinstance(epochs_data, np.ndarray) or epochs_data.ndim != 3:
        logger_decoding_core.error("epochs_data doit etre un tableau 3D.")
        return (None,) * 12
    if not isinstance(target_labels, np.ndarray) or target_labels.ndim != 1 or \
       target_labels.shape[0] != epochs_data.shape[0]:
        logger_decoding_core.error("target_labels incoherent avec epochs_data.")
        return (None,) * 12

    n_trials, n_channels, n_time_points = epochs_data.shape
    label_enc = LabelEncoder()
    target_labels_enc = label_enc.fit_transform(target_labels)
    n_classes = len(label_enc.classes_)

    empty_probas = np.zeros((n_trials, max(2, n_classes)))
    empty_labels = np.zeros(n_trials, dtype=int)
    empty_1d_scores = (np.full(n_time_points, np.nan)
                       if n_time_points > 0 else np.array([]))
    empty_tgm_scores = (np.full((n_time_points, n_time_points), np.nan)
                        if n_time_points > 0 else np.array([[]]))

    empty_results_tuple = (
        empty_probas, empty_labels, np.array([]), empty_1d_scores, {},
        None, None,
        (np.full((1, n_time_points), np.nan) if n_time_points > 0
         else np.array([[]])),
        empty_tgm_scores, None, None,
        (np.full((1, n_time_points, n_time_points), np.nan)
         if n_time_points > 0 else np.array([[[]]])))

    if n_classes < 2:
        logger_decoding_core.error(
            "Au moins 2 classes sont necessaires. Trouve : %d.", n_classes)
        return empty_results_tuple
    if n_time_points == 0:
        logger_decoding_core.error("epochs_data n'a aucun point temporel.")
        return empty_results_tuple

    # --- Pipeline pour MNE (Sliding / Generalizing) ---
    logger_decoding_core.info(
        "Construction du pipeline temporel (avec selection de features)")
    pipeline_mne, clf_name_mne, fs_name_mne, _ = \
        _build_standard_classifier_pipeline(
            classifier_model_type=classifier_model_type,
            use_grid_search=use_grid_search,
            add_anova_fs_step=use_anova_fs_for_temporal_pipelines,
            **(fixed_classifier_params
               if fixed_classifier_params and not use_grid_search else {})
        )

    final_estimator_mne = None
    if use_grid_search:
        current_grid_mne = {}
        if param_grid_config and classifier_model_type in param_grid_config:
            full_grid_for_clf = param_grid_config[classifier_model_type]
            current_grid_mne = {
                k: v for k, v in full_grid_for_clf.items()
                if k.startswith(f"{clf_name_mne}__") or
                (fs_name_mne and k.startswith(f"{fs_name_mne}__"))
            }
            logger_decoding_core.info("param_grid pour GS MNE: %s",
                                      current_grid_mne)
        else:
            logger_decoding_core.warning(
                "Pas de param_grid_config pour '%s' (GS MNE). Defauts basiques.",
                classifier_model_type
            )
            current_grid_mne = {f'{clf_name_mne}__C': [0.1, 1, 10]}
            if fs_name_mne:
                current_grid_mne[f'{fs_name_mne}__percentile'] = [15, 30]

        if not current_grid_mne:
            logger_decoding_core.error(
                "Aucune grille pour le GS MNE (%s). Abandon.",
                classifier_model_type)
            return empty_results_tuple
        gs_cv_n_jobs = (1 if INTERNAL_N_JOBS_FOR_MNE_DECODING != 1
                        else n_jobs_external)
        final_estimator_mne = GridSearchCV(
            estimator=pipeline_mne, param_grid=current_grid_mne,
            scoring='roc_auc', cv=cv_folds_for_gridsearch, n_jobs=gs_cv_n_jobs,
            verbose=0, refit=True,
            error_score=('raise'
                         if logger_decoding_core.getEffectiveLevel() <= logging.DEBUG
                         else np.nan)
        )
    else:
        final_estimator_mne = pipeline_mne

    # --- Pipeline pour le decodage global ---
    logger_decoding_core.info(
        "Construction du pipeline global (avec selection de features)")
    pipeline_global, clf_name_global, fs_name_global, _ = \
        _build_standard_classifier_pipeline(
            classifier_model_type=classifier_model_type,
            use_grid_search=use_grid_search,
            add_anova_fs_step=use_anova_fs_for_temporal_pipelines,
            **(fixed_classifier_params
               if fixed_classifier_params and not use_grid_search else {})
        )
    final_estimator_global = None
    if use_grid_search:
        current_grid_global = {}
        if param_grid_config and classifier_model_type in param_grid_config:
            full_grid_for_clf = param_grid_config[classifier_model_type]
            current_grid_global = {
                k: v for k, v in full_grid_for_clf.items()
                if k.startswith(f"{clf_name_global}__") or
                (fs_name_global and k.startswith(f"{fs_name_global}__"))
            }
            logger_decoding_core.info(
                "param_grid (classifieur+FS) pour GS global: %s",
                current_grid_global)
        else:
            logger_decoding_core.warning(
                "Pas de param_grid_config pour '%s' (GS global). Defauts basiques.",
                classifier_model_type
            )
            current_grid_global = {f'{clf_name_global}__C': [0.1, 1, 10]}
            if fs_name_global:
                current_grid_global[f'{fs_name_global}__percentile'] = [15, 30]
        if not current_grid_global:
            logger_decoding_core.error(
                "Aucune grille pour le GS global (%s). Abandon.",
                classifier_model_type)
            return empty_results_tuple
        gs_cv_n_jobs_global = 1 if n_jobs_external != 1 else -1
        final_estimator_global = GridSearchCV(
            estimator=pipeline_global, param_grid=current_grid_global,
            scoring='roc_auc', cv=cv_folds_for_gridsearch,
            n_jobs=gs_cv_n_jobs_global, verbose=0, refit=True,
            error_score=('raise'
                         if logger_decoding_core.getEffectiveLevel() <= logging.DEBUG
                         else np.nan)
        )
    else:
        final_estimator_global = pipeline_global

    # --- Choix du splitter de validation croisee ---
    actual_cv_splitter = None
    if isinstance(cross_validation_splitter, int):
        n_splits_req = cross_validation_splitter
        if group_labels_for_cv is not None and \
                len(np.unique(group_labels_for_cv)) >= 2:
            n_groups = len(np.unique(group_labels_for_cv))
            # n_splits doit etre calcule AVANT d'etre teste : la version
            # precedente lisait une variable non affectee (UnboundLocalError).
            n_splits_group = min(n_splits_req, n_groups)
            if n_splits_group >= 2:
                actual_cv_splitter = GroupKFold(n_splits=n_splits_group)
            else:
                logger_decoding_core.warning(
                    "GroupKFold impossible avec %d splits demandes pour %d "
                    "groupes. Bascule sur RepeatedStratifiedKFold.",
                    n_splits_req, n_groups)

        if actual_cv_splitter is None:
            min_class_count = np.min(np.bincount(target_labels_enc))
            n_splits = (min(n_splits_req, min_class_count)
                        if min_class_count >= 2 else 0)
            if n_splits >= 2:
                actual_cv_splitter = RepeatedStratifiedKFold(
                    n_splits=n_splits, n_repeats=5, random_state=random_state)
    elif (hasattr(cross_validation_splitter, "split") and
          hasattr(cross_validation_splitter, "get_n_splits")):
        actual_cv_splitter = cross_validation_splitter

    if actual_cv_splitter is None or actual_cv_splitter.get_n_splits(
            X=epochs_data, y=target_labels_enc,
            groups=group_labels_for_cv) < 2:
        logger_decoding_core.error(
            "Configuration CV invalide : moins de 2 splits possibles. Abandon.")
        return empty_results_tuple

    num_cv_splits = actual_cv_splitter.get_n_splits(
        X=epochs_data, y=target_labels_enc, groups=group_labels_for_cv)
    n_repeats, n_splits_per_repeat = _get_cv_repeat_structure(
        actual_cv_splitter, num_cv_splits)

    logger_decoding_core.info(
        "Strategie CV : %s, %d folds au total (%d repetitions x %d splits).",
        type(actual_cv_splitter).__name__, num_cv_splits, n_repeats,
        n_splits_per_repeat)
    if n_repeats > 1:
        if aggregate_cv_repeats:
            logger_decoding_core.info(
                "Les %d repetitions seront moyennees avant les statistiques "
                "intra-sujet (n effectif = %d folds independants).",
                n_repeats, n_splits_per_repeat)
        else:
            logger_decoding_core.warning(
                "aggregate_cv_repeats=False : les %d folds seront traites "
                "comme independants alors qu'ils ne le sont pas. Les p-values "
                "intra-sujet seront anticonservatrices.", num_cv_splits)

    # --- Poids d'echantillons ---
    # Le rebalancement par defaut passe par class_weight="balanced" (applique
    # a l'identique dans MNE et dans le decodage global). Les sample_weight ne
    # sont utilises que si CLASS_BALANCING_STRATEGY == "sample_weight".
    effective_sample_weights = None
    use_fold_specific_weights = False
    if CLASS_BALANCING_STRATEGY == "sample_weight":
        if isinstance(trial_sample_weights, str) and \
                trial_sample_weights.lower() == "auto":
            use_fold_specific_weights = True
            logger_decoding_core.info(
                "Poids d'echantillons recalcules par fold.")
        elif isinstance(trial_sample_weights, np.ndarray):
            if trial_sample_weights.shape == (n_trials,):
                effective_sample_weights = trial_sample_weights
                logger_decoding_core.info(
                    "Poids d'echantillons fournis par l'utilisateur.")
            else:
                logger_decoding_core.warning(
                    "Dimensions de sample_weights incoherentes. Desactives.")
    else:
        logger_decoding_core.info(
            "Rebalancement via class_weight : sample_weight non utilise.")

    # --- Conteneurs de resultats ---
    scores_1d_all_folds = np.full((num_cv_splits, n_time_points), np.nan)
    mean_scores_1d = np.full(n_time_points, np.nan)
    tgm_all_folds = np.full(
        (num_cv_splits, n_time_points, n_time_points), np.nan)
    mean_tgm = np.full((n_time_points, n_time_points), np.nan)
    fdr_1d_data, cluster_1d_data, fdr_tgm_data = None, None, None
    probas_global_sum = np.zeros((n_trials, max(2, n_classes)))
    probas_global_count = np.zeros(n_trials)
    labels_votes = np.zeros((n_trials, max(2, n_classes)))
    cv_global_scores = np.array([])
    global_metrics = {}

    cv_groups_mne = (group_labels_for_cv
                     if isinstance(actual_cv_splitter, GroupKFold) else None)
    fit_params_mne = {}
    if effective_sample_weights is not None and not use_grid_search and clf_name_mne:
        fit_params_mne[f"{clf_name_mne}__sample_weight"] = effective_sample_weights

    # --- Decodage temporel 1D (SlidingEstimator) ---
    logger_decoding_core.info("Decodage temporel 1D (SlidingEstimator)...")
    sliding_decoder = SlidingEstimator(
        base_estimator=clone(final_estimator_mne),
        n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING, scoring="roc_auc",
        verbose=False
    )
    try:
        scores_1d_cv_raw = cross_val_multiscore(
            sliding_decoder, epochs_data, target_labels_enc,
            cv=actual_cv_splitter, groups=cv_groups_mne,
            fit_params=fit_params_mne if fit_params_mne else None,
            n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING
        )
        if scores_1d_cv_raw is not None and len(scores_1d_cv_raw) > 0:
            scores_1d_temp = np.asarray(scores_1d_cv_raw)
            if scores_1d_temp.ndim == 2 and \
               scores_1d_temp.shape == (num_cv_splits, n_time_points):
                scores_1d_all_folds = scores_1d_temp
                mean_scores_1d = np.nanmean(scores_1d_all_folds, axis=0)
            else:
                logger_decoding_core.error(
                    "Dimensions inattendues pour les scores 1D. "
                    "Attendu (%d, %d), obtenu %s.",
                    num_cv_splits, n_time_points, scores_1d_temp.shape)
        peak_auc = (np.nanmax(mean_scores_1d)
                    if mean_scores_1d.size > 0 and
                    not np.all(np.isnan(mean_scores_1d)) else "N/A")
        logger_decoding_core.info(
            "Decodage 1D termine. AUC moyenne au pic : %s",
            f"{peak_auc:.3f}" if isinstance(peak_auc, float) else peak_auc)
    except Exception as e_temporal:
        logger_decoding_core.error(
            "Erreur pendant le decodage temporel 1D : %s",
            e_temporal, exc_info=True)

    # --- TGM (GeneralizingEstimator) ---
    if compute_temporal_generalization_matrix:
        logger_decoding_core.info("Calcul de la TGM (GeneralizingEstimator)...")
        tgm_decoder = GeneralizingEstimator(
            clone(final_estimator_mne),
            n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING, scoring="roc_auc",
            verbose=False
        )
        try:
            tgm_cv_raw = cross_val_multiscore(
                tgm_decoder, epochs_data, target_labels_enc,
                cv=actual_cv_splitter, groups=cv_groups_mne,
                fit_params=fit_params_mne if fit_params_mne else None,
                n_jobs=INTERNAL_N_JOBS_FOR_MNE_DECODING
            )
            if tgm_cv_raw is not None and len(tgm_cv_raw) > 0:
                tgm_temp = np.asarray(tgm_cv_raw)
                if tgm_temp.ndim == 3 and tgm_temp.shape == (
                        num_cv_splits, n_time_points, n_time_points):
                    tgm_all_folds = tgm_temp
                    mean_tgm = np.nanmean(tgm_all_folds, axis=0)
                else:
                    logger_decoding_core.error(
                        "Dimensions inattendues pour la TGM. "
                        "Attendu (%d, %d, %d), obtenu %s.",
                        num_cv_splits, n_time_points, n_time_points,
                        tgm_temp.shape)
            peak_tgm_auc = (np.nanmax(mean_tgm)
                            if mean_tgm.size > 0 and
                            not np.all(np.isnan(mean_tgm)) else "N/A")
            logger_decoding_core.info(
                "TGM terminee. AUC moyenne au pic : %s",
                f"{peak_tgm_auc:.3f}" if isinstance(peak_tgm_auc, float)
                else peak_tgm_auc)
        except Exception as e_tgm:
            logger_decoding_core.error(
                "Erreur pendant le calcul de la TGM : %s", e_tgm, exc_info=True)
    else:
        logger_decoding_core.info("Calcul de la TGM desactive.")

    # --- Decodage global (features aplaties) ---
    logger_decoding_core.info(
        "Decodage global (n_jobs_folds=%s)...", n_jobs_external)
    epochs_flat_global = epochs_data.reshape(n_trials, -1)
    parallel_global, pfunc_global, _ = parallel_func(
        _execute_global_decoding_for_one_fold, n_jobs=n_jobs_external,
        verbose=0
    )
    cv_splits_global = list(actual_cv_splitter.split(
        epochs_flat_global, target_labels_enc, groups=cv_groups_mne))

    weights_arg = (effective_sample_weights
                   if not use_fold_specific_weights else "auto")
    global_fold_outputs = parallel_global(
        pfunc_global(
            clone(final_estimator_global),
            epochs_flat_global,
            target_labels_enc,
            train_idx,
            test_idx,
            weights_arg,
            clf_name_global,
        ) for train_idx, test_idx in cv_splits_global
    )

    cv_scores_list_g = []
    for i_fold, (_, test_idx_g) in enumerate(cv_splits_global):
        if i_fold >= len(global_fold_outputs):
            logger_decoding_core.error(
                "Sortie manquante pour le fold global %d.", i_fold)
            cv_scores_list_g.append(np.nan)
            continue

        probas_f, preds_f, score_f = global_fold_outputs[i_fold]

        # Accumulation (somme + compteur) au lieu d'ecrasement : avec
        # RepeatedStratifiedKFold chaque essai est teste n_repeats fois et
        # l'ancienne affectation directe ne conservait que la derniere.
        if isinstance(probas_f, np.ndarray) and probas_f.ndim == 2 and \
                probas_f.shape[0] == len(test_idx_g):
            if probas_f.shape[1] == probas_global_sum.shape[1]:
                probas_global_sum[test_idx_g] += probas_f
                probas_global_count[test_idx_g] += 1
            elif probas_f.shape[1] == 1 and probas_global_sum.shape[1] == 2:
                probas_global_sum[test_idx_g, 1] += probas_f.ravel()
                probas_global_sum[test_idx_g, 0] += 1 - probas_f.ravel()
                probas_global_count[test_idx_g] += 1
            else:
                logger_decoding_core.warning(
                    "Fold %d : dimensions de probabilites incoherentes "
                    "(agregat %d, fold %d). Fold ignore pour l'agregation.",
                    i_fold, probas_global_sum.shape[1], probas_f.shape[1])
        else:
            logger_decoding_core.warning(
                "Fold %d : probabilites invalides, agregation ignoree.", i_fold)

        # Vote majoritaire sur les repetitions pour les labels predits.
        if isinstance(preds_f, np.ndarray) and \
                preds_f.shape[0] == len(test_idx_g):
            for pos, trial_idx in enumerate(test_idx_g):
                pred_class = int(preds_f[pos])
                if 0 <= pred_class < labels_votes.shape[1]:
                    labels_votes[trial_idx, pred_class] += 1
        else:
            logger_decoding_core.warning(
                "Fold %d : predictions invalides, agregation ignoree.", i_fold)

        cv_scores_list_g.append(score_f)

    with np.errstate(invalid="ignore"):
        probas_global_agg = np.where(
            probas_global_count[:, None] > 0,
            probas_global_sum / np.maximum(probas_global_count, 1)[:, None],
            0.0,
        )
    labels_global_agg = np.argmax(labels_votes, axis=1).astype(int)
    n_never_tested = int(np.sum(probas_global_count == 0))
    if n_never_tested:
        logger_decoding_core.warning(
            "%d essais n'ont jamais ete en test : exclus des metriques "
            "globales agregees.", n_never_tested)
    tested_mask = probas_global_count > 0

    cv_global_scores = np.asarray(cv_scores_list_g, dtype=float)
    if aggregate_cv_repeats and n_repeats > 1:
        cv_global_scores = _average_over_cv_repeats(cv_global_scores, n_repeats)

    mean_auc_g = (np.nanmean(cv_global_scores)
                  if cv_global_scores.size > 0 and
                  not np.all(np.isnan(cv_global_scores)) else "N/A")
    std_auc_g = (np.nanstd(cv_global_scores)
                 if cv_global_scores.size > 0 and
                 not np.all(np.isnan(cv_global_scores)) else "N/A")
    logger_decoding_core.info(
        "Decodage global termine. AUC CV moyenne : %s +/- %s (n=%d folds)",
        f"{mean_auc_g:.3f}" if isinstance(mean_auc_g, float) else mean_auc_g,
        f"{std_auc_g:.3f}" if isinstance(std_auc_g, float) else std_auc_g,
        cv_global_scores.size,
    )

    # --- Metriques globales sur les predictions agregees ---
    try:
        y_true_eval = target_labels_enc[tested_mask]
        y_pred_eval = labels_global_agg[tested_mask]
        probas_eval = probas_global_agg[tested_mask]

        if probas_eval.ndim == 2 and probas_eval.shape[1] >= 2 and \
                len(np.unique(y_true_eval)) > 1 and \
                not np.all(np.isnan(probas_eval)):
            global_metrics["roc_auc"] = roc_auc_score(
                y_true_eval, probas_eval[:, 1])
        else:
            global_metrics["roc_auc"] = np.nan
            logger_decoding_core.debug(
                "ROC AUC global agrege ignore (probabilites ou labels "
                "invalides).")

        avg_mode = "binary" if n_classes == 2 else "weighted"
        pos_label_val = 1 if avg_mode == "binary" else None
        m_args_common = {"y_true": y_true_eval, "y_pred": y_pred_eval,
                         "zero_division": 0}

        global_metrics.update({
            "accuracy": accuracy_score(y_true_eval, y_pred_eval),
            "precision": (precision_score(**m_args_common, average=avg_mode,
                                          pos_label=pos_label_val)
                          if avg_mode == "binary"
                          else precision_score(**m_args_common, average=avg_mode)),
            "recall": (recall_score(**m_args_common, average=avg_mode,
                                    pos_label=pos_label_val)
                       if avg_mode == "binary"
                       else recall_score(**m_args_common, average=avg_mode)),
            "f1_score": (f1_score(**m_args_common, average=avg_mode,
                                  pos_label=pos_label_val)
                         if avg_mode == "binary"
                         else f1_score(**m_args_common, average=avg_mode)),
            "balanced_accuracy": balanced_accuracy_score(y_true_eval,
                                                         y_pred_eval),
        })
        global_metrics["n_trials_evaluated"] = int(np.sum(tested_mask))
        global_metrics["n_cv_repeats_averaged"] = int(n_repeats)
        logger_decoding_core.info(
            "Metriques globales agregees : %s",
            {k: (f"{v:.3f}" if isinstance(v, float) and pd.notna(v) else v)
             for k, v in global_metrics.items()})
    except Exception as e_metrics:
        logger_decoding_core.error(
            "Erreur de calcul des metriques globales : %s",
            e_metrics, exc_info=True)
        for metric_key in ["accuracy", "precision", "recall", "f1_score",
                           "balanced_accuracy"]:
            global_metrics.setdefault(metric_key, np.nan)

    # --- Agregation des folds avant statistiques intra-sujet ---
    scores_1d_for_stats = scores_1d_all_folds
    tgm_for_stats = tgm_all_folds
    if aggregate_cv_repeats and n_repeats > 1:
        scores_1d_for_stats = _average_over_cv_repeats(
            scores_1d_all_folds, n_repeats)
        tgm_for_stats = _average_over_cv_repeats(tgm_all_folds, n_repeats)

    fdr_test_name = STAT_TEST_FOR_FDR_POINTWISE
    cluster_test_name = STAT_TEST_FOR_CLUSTER_PERMUTATION
    cluster_stat_fun = bEEG_stats.get_stat_function_for_cluster_test(
        cluster_test_name)
   
    effective_cluster_test = ("wilcoxon" if cluster_stat_fun is not None
                              else "ttest")

    if compute_intra_fold_stats and scores_1d_for_stats.shape[0] > 1 and \
            n_time_points > 0:
        _warn_if_underpowered(scores_1d_for_stats.shape[0], n_time_points,
                              fdr_test_name)
        if not np.all(np.isnan(scores_1d_for_stats)):
            _, fdr_mask_1d, fdr_p_1d, fdr_test_info = \
                bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                    scores_1d_for_stats, chance_level,
                    alternative_hypothesis="greater",
                    statistical_test_type=fdr_test_name
                )
            fdr_1d_data = {
                "mask": fdr_mask_1d,
                "p_values": fdr_p_1d,
                "p_values_raw": fdr_test_info.get("p_values_raw", fdr_p_1d),
                "method": f"FDR_CV_Folds_1D_{fdr_test_name}",
                "test_info": fdr_test_info,
                "n_observations": int(scores_1d_for_stats.shape[0]),
                "cv_repeats_averaged": int(n_repeats) if aggregate_cv_repeats else 1,
            }

            t_obs_clu, clu_1d, p_clu_1d, clu_info = \
                bEEG_stats.perform_cluster_permutation_test(
                    scores_1d_for_stats,
                    chance_level,
                    n_permutations_for_intra_fold_clusters,
                    cluster_threshold_config_intra_fold,
                    "greater",
                    INTERNAL_N_JOBS_FOR_MNE_DECODING,
                    random_seed=random_state,
                    stat_fun=cluster_stat_fun,
                )
            combined_mask_clu1d = np.zeros(n_time_points, dtype=bool)
            sig_clu_obj1d = []
            if clu_1d and p_clu_1d is not None:
                for i, cluster_mask in enumerate(clu_1d):
                    if i < len(p_clu_1d) and p_clu_1d[i] < 0.05:
                        sig_clu_obj1d.append(cluster_mask)
                        combined_mask_clu1d = np.logical_or(
                            combined_mask_clu1d, cluster_mask)

            cluster_1d_data = {
                "mask": combined_mask_clu1d,
                "cluster_objects": sig_clu_obj1d,
                "p_values_all_clusters": p_clu_1d,
                "cluster_info": clu_info,
                "observed_stats": t_obs_clu,
                "method": f"CluPerm_CV_Folds_1D_{effective_cluster_test}",
                "n_observations": int(scores_1d_for_stats.shape[0]),
                "cv_repeats_averaged": int(n_repeats) if aggregate_cv_repeats else 1,
            }

        if compute_temporal_generalization_matrix and \
                tgm_for_stats.size > 0 and \
                not np.all(np.isnan(tgm_for_stats)) and \
                tgm_for_stats.shape[0] > 1:
            n_f, n_tr, n_te = tgm_for_stats.shape
            if n_tr > 0 and n_te > 0:
                flat_tgm_scores = tgm_for_stats.reshape(n_f, n_tr * n_te)
                _, fdr_mask_tgm_flat, pvals_tgm_flat, fdr_test_info_tgm = \
                    bEEG_stats.perform_pointwise_fdr_correction_on_scores(
                        flat_tgm_scores, chance_level,
                        alternative_hypothesis="greater",
                        statistical_test_type=fdr_test_name
                    )
                fdr_tgm_data = {
                    "mask": (fdr_mask_tgm_flat.reshape(n_tr, n_te)
                             if fdr_mask_tgm_flat is not None and
                             hasattr(fdr_mask_tgm_flat, 'reshape') else None),
                    "p_values": (pvals_tgm_flat.reshape(n_tr, n_te)
                                 if pvals_tgm_flat is not None and
                                 hasattr(pvals_tgm_flat, 'reshape') else None),
                    "p_values_raw": (
                        fdr_test_info_tgm.get("p_values_raw", pvals_tgm_flat)
                        .reshape(n_tr, n_te)
                        if pvals_tgm_flat is not None and
                        hasattr(pvals_tgm_flat, 'reshape') else None),
                    "method": f"FDR_CV_Folds_TGM_{fdr_test_name}",
                    "test_info": fdr_test_info_tgm,
                    "n_observations": int(n_f),
                }
            else:
                logger_decoding_core.warning(
                    "Dimensions TGM nulles, FDR sur TGM ignoree.")
            logger_decoding_core.info(
                "Permutation par cluster sur la TGM intra-sujet : IGNOREE "
                "(cout de calcul).")
    else:
        logger_decoding_core.info(
            "Statistiques intra-sujet ignorees (drapeau desactive ou pas "
            "assez de folds / points temporels).")

    return (
        probas_global_agg, labels_global_agg, cv_global_scores,
        mean_scores_1d, global_metrics,
        fdr_1d_data, cluster_1d_data,
        scores_1d_for_stats, mean_tgm,
        fdr_tgm_data, None,
        tgm_for_stats
    )
