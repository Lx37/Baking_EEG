import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score,
                             roc_curve, confusion_matrix)
from sklearn.base import clone
from mne.parallel import parallel_func

import os
import logging
from datetime import datetime

import multiprocessing as mp

# ---------------------------
# Logger Configuration
# ---------------------------
log_dir = './logs/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, 'log_' + datetime.now().strftime('%Y-%m-%d.log')),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Intra-Subject Decoding Functions
# ---------------------------

def _decode_window_one_fold(clf, X, y, train, test, sample_weight):
    """
    Decoding for a specific fold of cross-validation.

    Parameters
    ----------
    clf : estimator
        Classifier to use (clone of the original pipeline)
    X : array
        Input data
    y : array
        Class labels
    train : array
        Indices of training samples for this fold
    test : array
        Indices of test samples for this fold
    sample_weight : array
        Sample weights

    Returns
    -------
    this_probas : array
        Predicted probabilities for the test data
    prediction : array
        Binary predictions for the test data
    score : float
        AUC score for this fold
    """
    
    if 'svc' in clf.named_steps:
        clf.fit(X[train], y[train], **{'svc__sample_weight': sample_weight[train]})
    else:
        clf.fit(X[train], y[train])


    this_probas = clf.predict_proba(X[test])
    prediction = clf.predict(X[test])

    score = roc_auc_score(
        y_true=y[test],
        y_score=this_probas[:, 1],
        sample_weight=sample_weight[test],
        average='weighted'
    )

    return this_probas, prediction, score

def decode_window(X, y, clf=None, cv=5, sample_weight='auto', n_jobs='auto', labels=None):
    """Decode entire window

    Parameters
    ----------
    X : np.ndarray of float, shape(n_samples, n_sensors, n_times)
        The data.
    y : np.ndarray of int, shape(n_samples,)
        The response vector.
    clf : instance of BaseEstimator | None
        The classifier. If None, defaults to a Pipeline.
    cv : cross validation object | None
        The cross validation. If None, defaults to stratified K-folds
        with 10 folds.
    sample_weight : np.ndarray of float, shape(n_samples,)
        The sample weights to deal with class imbalance.
        if 'auto' computes sample weights to balance

    Returns
    -------
    probas : np.ndarray of float, shape(n_samples,)
        The predicted probabilities for each sample.
    predictions : np.ndarray of int, shape(n_samples,)
        The class predictions.
    scores : np.ndarray of float, shape(n_resamples,)
        The score at each resampling iteration.
    score_times: 
    """
    if n_jobs == 'auto':
        try:
            n_jobs = mp.cpu_count()
            logger.info(f'Autodetected number of jobs: {n_jobs}')
        except Exception:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1

    if clf is None:
        scaler = StandardScaler()
        transform = SelectPercentile(f_classif, percentile=15)
        svc = SVC(C=1, kernel='linear', probability=True, class_weight='balanced')
        clf = Pipeline([('scaler', scaler), ('anova', transform), ('svc', svc)])

    if cv is None or isinstance(cv, int):
        n_splits = cv if isinstance(cv, int) else 5
        cv = (StratifiedKFold(n_splits=int(min(n_splits, len(y) / 2)),
                              shuffle=True, random_state=42)
              if labels is None else GroupKFold(n_splits=n_splits))

    if isinstance(sample_weight, str) and sample_weight == 'auto':
        sample_weight = np.zeros(len(y), dtype=float)
        for this_y in np.unique(y):
            sample_weight[y == this_y] = 1.0 / np.sum(y == this_y)

    n_trials, n_channels, n_times = X.shape
    y = LabelEncoder().fit_transform(y)
    scores_time = np.zeros(n_times)

    # Temporal decoding: compute an average score for each time point
    for t in range(n_times):
        X_t = X[:, :, t].reshape(n_trials, -1)
        fold_scores = []
        for train_idx, test_idx in cv.split(X_t, y, labels):
            _, _, score = _decode_window_one_fold(clone(clf), X_t, y, train_idx, test_idx, sample_weight)
            fold_scores.append(score)
        scores_time[t] = np.mean(fold_scores)

    # Global decoding 
    X_flat = X.reshape(n_trials, -1)
    probas = np.zeros((y.shape[0], 2), dtype=float)
    predictions = np.zeros(y.shape, dtype=int)
    scores = []

    parallel, pfunc, _ = parallel_func(_decode_window_one_fold, n_jobs)
    out = parallel(pfunc(clone(clf), X_flat, y, train_idx, test_idx, sample_weight)
                   for train_idx, test_idx in cv.split(X_flat, y, labels))

    for (fold, (train_idx, test_idx)), (probas_, predicts_, score_) in zip(
            enumerate(cv.split(X_flat, y, labels)), out):
        probas[test_idx] = probas_
        predictions[test_idx] = predicts_
        scores.append(score_)
        logger.info(f"Fold {fold+1}/{cv.get_n_splits()} AUC: {score_:.3f}")


    overall_accuracy = accuracy_score(y, predictions)
    overall_precision = precision_score(y, predictions)
    overall_recall = recall_score(y, predictions)
    overall_f1 = f1_score(y, predictions)
    overall_balanced_accuracy = balanced_accuracy_score(y, predictions)
    overall_roc_auc = roc_auc_score(y, probas[:, 1])

    logger.info("Overall decoding metrics:")
    logger.info(f"Accuracy: {overall_accuracy:.3f}")
    logger.info(f"Precision: {overall_precision:.3f}")
    logger.info(f"Recall: {overall_recall:.3f}")
    logger.info(f"F1 Score: {overall_f1:.3f}")
    logger.info(f"Balanced Accuracy: {overall_balanced_accuracy:.3f}")
    logger.info(f"ROC AUC: {overall_roc_auc:.3f}")

    metrics = {
        "accuracy": overall_accuracy,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "balanced_accuracy": overall_balanced_accuracy,
        "roc_auc": overall_roc_auc
    }

    return probas, predictions, np.array(scores), scores_time, metrics

# ---------------------------
# Inter-subject decoding functions
# ---------------------------
def decode_cross_subject_fold(X_train, y_train, X_test, y_test, test_subject, group_name, protocol, base_path, save=True):
    """
    Decoding for a fold in a cross-subject context.

    Parameters
    ----------
    X_train : array, shape (n_trials_train, n_channels, n_times)
        Training EEG data.
    y_train : array, shape (n_trials_train,)
        Training class labels.
    X_test : array, shape (n_trials_test, n_channels, n_times)
        Test EEG data.
    y_test : array, shape (n_trials_test,)
        Test class labels.
    test_subject : str
        Identifier of the test subject.
    group_name : str
        Name of the group.
    protocol : str
        Name of the protocol.
    base_path : str
        Base path for the data.
    save : bool
        Indicates whether the results should be saved.

    Returns
    -------
    score : float
        Global AUC score.
    metrics : dict
        Performance metrics.
    clf : estimator
        Trained classifier.
    probas : array
        Predicted probabilities.
    predictions : array
        Binary predictions.
    scores_time : array
        AUC scores for each time point.
    """
    try:
        
    
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        

        clf = make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=15),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )

       
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        clf.fit(X_train_flat, y_train)

        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        probas = clf.predict_proba(X_test_flat)
        predictions = clf.predict(X_test_flat)

        score = roc_auc_score(y_test, probas[:, 1])
        metrics = {
            "roc_auc": score,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions)
        }

       
        n_times = X_test.shape[2]
        scores_time = np.zeros(n_times)

      
        parallel, pfunc, _ = parallel_func(_compute_score_at_time, n_jobs=-1)
        scores_time = parallel(pfunc(t, X_train, y_train, X_test, y_test, clf) for t in range(n_times))

        return score, metrics, clf, probas, predictions, scores_time

    except Exception as e:
        logger.error(f"Error during cross-subject decoding: {e}")
        raise

def _compute_score_at_time(t, X_train, y_train, X_test, y_test, clf):
    """
    Compute the AUC score for a specific time point.

    Parameters
    ----------
    t : int
        The time point index.
    X_train : array, shape (n_trials_train, n_channels, n_times)
        Training EEG data.
    y_train : array, shape (n_trials_train,)
        Training class labels.
    X_test : array, shape (n_trials_test, n_channels, n_times)
        Test EEG data.
    y_test : array, shape (n_trials_test,)
        Test class labels.
    clf : estimator
        The classifier to use.

    Returns
    -------
    float
        The mean AUC score for the given time point.
    """
    X_train_t = X_train[:, :, t].reshape(X_train.shape[0], -1)
    X_test_t = X_test[:, :, t].reshape(X_test.shape[0], -1)

    fold_scores = []
    for train_idx, test_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_test_t, y_test):
       
        clf_t = clone(clf)

        clf_t.fit(X_train_t, y_train)

        probas_t = clf_t.predict_proba(X_test_t[test_idx])
     
        score_t = roc_auc_score(y_test[test_idx], probas_t[:, 1])
        fold_scores.append(score_t)

    return np.mean(fold_scores)

# ---------------------------
# Plotting Functions
# ---------------------------

def plot_decoding_results_dashboard(epochs, X, y, probas, predictions, cv_scores,
                                    scores_time, subject_id, group, save_dir=None):
    """
    Dashboard of decoding results with a maximum of 4 plots per page.
    """

    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # ====================== PAGE 1 ======================
    fig1 = plt.figure(figsize=(15, 10))
    fig1.suptitle(f"Dashboard - {subject_id} in group {group} (1/2)",
                  fontsize=16, fontweight='bold')

    gs1 = GridSpec(2, 2, figure=fig1)

    # 1/ Temporal decoding (scores_time)
    ax_temporal = fig1.add_subplot(gs1[0, :])
    ax_temporal.plot(epochs.times, scores_time, 'b-', linewidth=2.5)
    std_scores = np.std(scores_time) * 0.5
    ax_temporal.fill_between(
        epochs.times,
        scores_time - std_scores,
        scores_time + std_scores,
        color='blue', alpha=0.2
    )
    ax_temporal.axhline(0.5, color='k', linestyle='--', label='Chance level')
    ax_temporal.axvline(0, color='r', linestyle='--', label='Stimulus')
    max_time_idx = np.argmax(scores_time)
    max_score = scores_time[max_time_idx]
    max_time = epochs.times[max_time_idx]
    ax_temporal.plot(max_time, max_score, 'ro', markersize=8)
    ax_temporal.annotate(
        f'Max: {max_score:.3f} @ {max_time * 1000:.0f}ms',
        xy=(max_time, max_score),
        xytext=(max_time + 0.1, max_score),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5)
    )
    ax_temporal.set_xlabel('Time (s)', fontsize=12)
    ax_temporal.set_ylabel('AUC Score', fontsize=12)
    ax_temporal.set_ylim(0.35, 1.0)
    ax_temporal.set_title('Temporal Decoding', fontsize=14)
    ax_temporal.legend(loc='lower right')
    ax_temporal.grid(True, alpha=0.3)

    # 2/ Barplot of CV scores
    ax_cv = fig1.add_subplot(gs1[1, 0])
    ax_cv.bar(range(len(cv_scores)), cv_scores, color='skyblue')
    ax_cv.axhline(np.mean(cv_scores), color='r', linestyle='--',
                  label=f'Mean: {np.mean(cv_scores):.3f}')
    ax_cv.set_xlabel('Fold')
    ax_cv.set_ylabel('AUC Score')
    ax_cv.set_title('CV Scores', fontsize=14)
    ax_cv.legend()

    # 3/ ROC Curve
    ax_roc = fig1.add_subplot(gs1[1, 1])
    fpr, tpr, _ = roc_curve(y, probas[:, 1])
    roc_auc = roc_auc_score(y, probas[:, 1])
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate', fontsize=10)
    ax_roc.set_ylabel('True Positive Rate', fontsize=10)
    ax_roc.set_title('ROC Curve', fontsize=14)
    ax_roc.legend(loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    page1_path = os.path.join(save_dir, f"dashboard_{subject_id}_{group}_page1.png")
    fig1.savefig(page1_path, dpi=300, bbox_inches='tight')

    # ====================== PAGE 2 ======================
    fig2 = plt.figure(figsize=(15, 10))
    fig2.suptitle(f"Dashboard - {subject_id} in group {group} (2/2)",
                  fontsize=16, fontweight='bold')

    gs2 = GridSpec(2, 2, figure=fig2)

    # 1/ Confusion Matrix
    ax_cm = fig2.add_subplot(gs2[0, 0])
    cm = confusion_matrix(y, predictions, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel('Prediction', fontsize=10)
    ax_cm.set_ylabel('Actual', fontsize=10)
    ax_cm.set_title('Confusion Matrix', fontsize=14)
    ax_cm.set_xticklabels(['Class 0', 'Class 1'])
    ax_cm.set_yticklabels(['Class 0', 'Class 1'])

    # 2/ Probability Distribution
    ax_dist = fig2.add_subplot(gs2[0, 1])
    for i, cls in enumerate(['Class 0', 'Class 1']):
        sns.kdeplot(probas[y == i, 1], ax=ax_dist,
                    label=f'{cls} (true)', fill=True, alpha=0.5)
    ax_dist.axvline(x=0.5, color='red', linestyle='--', label='Decision threshold')
    ax_dist.set_xlabel('Predicted Probability (class 1)', fontsize=10)
    ax_dist.set_ylabel('Density', fontsize=10)
    ax_dist.set_title('Probability Distribution', fontsize=14)
    ax_dist.legend()

    # 3/ Global Metrics
    ax_metrics = fig2.add_subplot(gs2[1, :])
    metrics = {
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1": f1_score(y, predictions),
        "Accuracy": accuracy_score(y, predictions),
        "Balanced\nAccuracy": balanced_accuracy_score(y, predictions),
        "AUC": roc_auc
    }
    keys = list(metrics.keys())
    values = [metrics[k] for k in keys]
    bars = ax_metrics.bar(keys, values, color='skyblue')
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_title('Global Metrics', fontsize=14)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_metrics.text(i, height + 0.02, f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    page2_path = os.path.join(save_dir, f"dashboard_{subject_id}_{group}_page2.png")
    fig2.savefig(page2_path, dpi=300, bbox_inches='tight')

    plt.close('all')
    return os.path.dirname(page1_path)

def plot_group_results(subject_scores, group_name, save_dir=None):
    """
    Visualize the results of a group.
    """
    valid_data = [(subject, score) for subject, score in subject_scores.items() if score is not None]
    if valid_data:
        subjects, scores = zip(*valid_data)
    else:
        logger.warning(f"No valid data to visualize for group {group_name}")
        return
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(subjects)), scores, color='skyblue')
    plt.axhline(np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
    plt.axhline(0.5, color='k', linestyle=':', label='Chance')
    plt.xticks(range(len(subjects)), subjects, rotation=90)
    plt.ylabel('AUC Score')
    plt.ylim(0.4, 1.0)
    plt.title(f'Decoding Scores - Group {group_name}')
    plt.legend()
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'group_{group_name}_scores.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()
    # TODO: rajouter plus de fonctionnalités pour la visualisation de groupe
