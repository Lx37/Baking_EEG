import os
import logging
from datetime import datetime
import multiprocessing as mp
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline, make_pipeline
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score,
                             roc_curve, confusion_matrix)
from sklearn.base import clone
from mne.parallel import parallel_func


log_dir = './logs/'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(
        log_dir, 'log_' + datetime.now().strftime('%Y-%m-%d.log')),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------
# Intra-subject decoding functions
# ---------------------------


def _decode_window_one_fold(clf, X, y, train, test, sample_weight):
    """Perform decoding for a single fold of cross-validation (Helper function).

    This function encapsulates the fitting and prediction steps for one
    split of the data during cross-validation within the `decode_window` function.
    It handles potential sample weighting specifically for SVC.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline or estimator
        A clone of the main classifier pipeline/estimator to be trained.
    X : np.ndarray
        The input feature data for this fold (potentially already subsetted or transformed).
    y : np.ndarray
        The target labels for this fold.
    train : np.ndarray
        Indices of the training samples for this fold.
    test : np.ndarray
        Indices of the test samples for this fold.
    sample_weight : np.ndarray or None
        Weights assigned to each sample, used during fitting if not None.

    Returns
    -------
    this_probas : np.ndarray
        Predicted probabilities for the positive class on the test set. Shape (n_test_samples, 2).
    prediction : np.ndarray
        Predicted class labels (0 or 1) for the test set. Shape (n_test_samples,).
    score : float
        The ROC AUC score calculated for this fold on the test set.
    """

    # The 'svc__sample_weight' syntax is specific to accessing parameters of a step named 'svc' in a Pipeline.
    if 'svc' in clf.named_steps and sample_weight is not None:
        # Fit the pipeline using training data and corresponding sample weights.
        clf.fit(X[train], y[train], **
                {'svc__sample_weight': sample_weight[train]})
    else:
        # Fit the pipeline using training data without sample weights (or if sample_weight is None).
        clf.fit(X[train], y[train])

    # Predict probabilities on the test set. Returns shape (n_test_samples, n_classes)
    # #where we've got (probability to belong to the class 0, proba to belong class 1)
    this_probas = clf.predict_proba(X[test])
    # It assigns the class with the highest probability to each sample
    prediction = clf.predict(X[test])

    # Calculate the ROC AUC score for this fold.
    # We use the probabilities of the positive class (index 1).
    score = roc_auc_score(
        y_true=y[test],
        y_score=this_probas[:, 1],  # Probability of the positive class
        sample_weight=sample_weight[test] if sample_weight is not None else None,
        average='weighted'
    )

    return this_probas, prediction, score


def decode_window(X, y, clf=None, cv=5, sample_weight='auto', n_jobs='auto', labels=None):
    """Decode EEG data using cross-validation, calculating global and temporal scores.

    This function performs classification on epoch data. It handles:
    - Setting up a default classification pipeline (StandardScaler + SelectPercentile + SVC) if none is provided.
    - Configuring cross-validation (StratifiedKFold by default, or GroupKFold if labels are provided).
    - Calculating sample weights to address class imbalance if 'auto'.
    - Performing temporal decoding: evaluating classification performance at each time point individually.
    - Performing global decoding: evaluating performance using features from all time points flattened.
    - Utilizing parallel processing to speed up computations.
    - Calculating various performance metrics (ROC AUC, accuracy, precision, recall, F1, balanced accuracy).  

    Parameters
    ----------
    X : np.ndarray
        The EEG data epochs. Shape (n_epochs, n_channels, n_times).
    y : np.ndarray
        The target labels corresponding to the epochs. Shape (n_epochs,).
    clf : sklearn estimator or None, optional
        The classifier pipeline to use. If None, a default SVC pipeline is created. Default is None.
    cv : int or cross-validation generator, optional
        The cross-validation strategy. If int, specifies the number of folds for StratifiedKFold.
        If None, defaults to 5-fold StratifiedKFold. Can also be a specific CV object like GroupKFold. Default is 5.
    sample_weight : 'auto' or np.ndarray or None, optional
        Strategy for sample weighting.
        - 'auto': Calculate weights inversely proportional to class frequencies.
        - None: No sample weighting.
        Default is 'auto'.
    n_jobs : int or 'auto', optional
        Number of CPU cores to use for parallel processing. 'auto' attempts to use all available cores. Default is 'auto'.
    labels : np.ndarray or None, optional
        Group labels for each epoch, used for GroupKFold cross-validation if provided. Shape (n_epochs,). Default is None.

    Returns
    -------
    probas : np.ndarray
        Predicted probabilities for the positive class for each epoch (from global decoding). Shape (n_epochs, 2).
    predictions : np.ndarray
        Predicted class labels for each epoch (from global decoding). Shape (n_epochs,).
    scores : np.ndarray
        The ROC AUC score obtained for each fold of the cross-validation (global decoding). Shape (n_splits,).
    scores_time : np.ndarray
        The average ROC AUC score calculated at each time point across CV folds (temporal decoding). Shape (n_times,).
    metrics : dict
        A dictionary containing overall performance metrics calculated on the aggregated out-of-fold predictions
        from the global decoding (e.g., 'accuracy', 'precision', 'recall', 'f1', 'roc_auc').
    """
    # Configure the number of parallel jobs.
    if n_jobs == 'auto':
        try:
            # Try to detect the number of available CPU cores.
            n_jobs = mp.cpu_count()
            logger.info(
                f'Autodetected number of jobs for parallel processing: {n_jobs}')
        except NotImplementedError:
            # Fallback if CPU count detection fails.
            logger.warning(
                'Cannot automatically detect number of jobs. Defaulting to 1.')
            n_jobs = 1
    elif not isinstance(n_jobs, int) or n_jobs < 1:
        logger.warning(f'Invalid n_jobs value ({n_jobs}). Defaulting to 1.')
        n_jobs = 1

    # Define a default classifier pipeline if none is provided by the user.
    if clf is None:
        logger.info(
            "No classifier provided, using default pipeline: StandardScaler -> SelectPercentile(15%) -> SVC(linear, balanced)")
        scaler = StandardScaler()  # Standardize features (mean=0, variance=1)
        # Select the top 15% of features based on ANOVA F-test score.
        # This helps reduce dimensionality and potentially noise.
        transform = SelectPercentile(f_classif, percentile=15)
        # Linear Support Vector Classifier.
        # probability=True is needed for predict_proba and ROC AUC.
        # class_weight='balanced' helps handle imbalanced classes.
        svc = SVC(C=1, kernel='linear', probability=True,
                  class_weight='balanced')
        # Chain the steps together.
        clf = Pipeline(
            [('scaler', scaler), ('anova', transform), ('svc', svc)])

    # Configure the cross-validation strategy.
    if cv is None or isinstance(cv, int):
        # If cv is an integer or None, set up KFold.
        # Default to 5 folds if cv is None or invalid int.
        n_splits = cv if isinstance(cv, int) else 5

        # StratifiedKFold maintains class proportions in folds, crucial for classification.
        # GroupKFold ensures all samples from a group are in the same fold (train or test).
        cv = (GroupKFold(n_splits=int(min(n_splits, len(np.unique(labels)))))
              if labels is not None
              else StratifiedKFold(n_splits=int(min(n_splits, np.sum(y == y[0]), np.sum(y != y[0]))),  # Ensure n_splits <= n_samples_per_class
                                   shuffle=True, random_state=42))
        logger.info(
            f"Using CV strategy: {type(cv).__name__} with {cv.get_n_splits()} splits.")

    if isinstance(sample_weight, str) and sample_weight == 'auto':
        logger.info(
            "Calculating sample weights automatically to balance classes.")
        # Calculate weights inversely proportional to class frequency.
        n_samples = len(y)
        n_classes = len(np.unique(y))
        sample_weight = np.zeros(n_samples, dtype=float)
        for this_y in np.unique(y):
            class_indices = (y == this_y)
            class_count = np.sum(class_indices)
            if class_count > 0:
                sample_weight[class_indices] = n_samples / \
                    (n_classes * class_count)
            else:
                logger.warning(
                    f"Class {this_y} has zero samples, cannot compute weight.")
        # Normalize weights
        # sample_weight /= np.mean(sample_weight)
    elif sample_weight is not None and not isinstance(sample_weight, np.ndarray):
        logger.warning(
            "Invalid sample_weight provided, ignoring sample weights.")
        sample_weight = None

    # Get data dimensions.
    n_trials, n_channels, n_times = X.shape
    # Encode labels to integers (0 and 1) if they are not already.
    le = LabelEncoder()
    y = le.fit_transform(y)
    logger.info(
        f"Data shape: {n_trials} trials, {n_channels} channels, {n_times} time points.")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # --- Temporal Decoding ---
    # Purpose: Evaluate classification performance independently at each time point.
    logger.info("Starting temporal decoding (per time point)...")
    # Array to store the average score for each time point.
    scores_time = np.zeros(n_times)
    temporal_start_time = time.time()

    # Helper function for parallel processing of temporal decoding.
    def _temporal_decode_time_point(t):
        # Reshape data for the current time point t: (n_trials, n_channels)
        X_t = X[:, :, t]  # Use all channels at this time point
        fold_scores_t = []
        # Perform cross-validation for this specific time point.
        for train_idx, test_idx in cv.split(X_t, y, labels):
            # We need a fresh clone for each fold and time point if the classifier has state.
            # we have a blank clf without the hyperparameters being changed
            clf_t = clone(clf)
            _, _, score_t = _decode_window_one_fold(
                clf_t, X_t, y, train_idx, test_idx, sample_weight)
            fold_scores_t.append(score_t)

        return np.mean(fold_scores_t)

    # Set up parallel execution using mne's helper.
    parallel, pfunc, n_jobs = parallel_func(
        _temporal_decode_time_point, n_jobs=n_jobs, verbose=0)
    logger.info(f"Running temporal decoding in parallel with {n_jobs} jobs.")
    # Execute the decoding for all time points in parallel.
    scores_time = np.array(parallel(pfunc(t) for t in range(n_times)))
    logger.info(
        f"Temporal decoding finished in {time.time() - temporal_start_time:.2f} seconds.")
    logger.info(
        f"Peak temporal AUC: {np.max(scores_time):.3f} at time index {np.argmax(scores_time)}")

    # --- Global Decoding ---
   # Here we evaluate performance using features from all time points combined.
    logger.info("Starting global decoding (using all time points)...")
    global_start_time = time.time()
    # Reshape data by flattening channels and time points for each trial because SVM need 2D data form
    # Shape becomes (n_trials, n_channels * n_times).
    X_flat = X.reshape(n_trials, -1)

    # Probabilities for each class as before
    probas = np.zeros((n_trials, len(le.classes_)), dtype=float)
    # Predicted label for each trial
    predictions = np.zeros(n_trials, dtype=int)
    scores = []

    # Set up parallel execution for the cross-validation folds.
    parallel, pfunc, n_jobs = parallel_func(
        _decode_window_one_fold, n_jobs=n_jobs, verbose=0)
    logger.info(f"Running global decoding CV in parallel with {n_jobs} jobs.")
    # Execute the decoding for each fold in parallell
    out = parallel(pfunc(clone(clf), X_flat, y, train_idx, test_idx, sample_weight)
                   for train_idx, test_idx in cv.split(X_flat, y, labels))

    # Process results from each parallel job (each fold).
    for fold_idx, ((train_idx, test_idx), (probas_, predicts_, score_)) in enumerate(zip(
            cv.split(X_flat, y, labels), out)):

        probas[test_idx] = probas_
        predictions[test_idx] = predicts_
        scores.append(score_)
        logger.info(
            f"Global decoding Fold {fold_idx+1}/{cv.get_n_splits()} AUC: {score_:.3f}")

    # Convert scores list to a numpy array.
    scores = np.array(scores)
    logger.info(
        f"Global decoding finished in {time.time() - global_start_time:.2f} seconds.")
    logger.info(
        f"Mean global CV AUC: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    # --- Calculate Overall Metrics ---
    # Calculate metrics based on the out-of-fold predictions aggregated across all folds.
    try:
        # Use proba of positive class
        overall_roc_auc = roc_auc_score(y, probas[:, 1], average='weighted')
        overall_accuracy = accuracy_score(y, predictions)
        # Handle case where prediction counts are zero
        overall_precision = precision_score(y, predictions, zero_division=0)
        overall_recall = recall_score(y, predictions, zero_division=0)
        overall_f1 = f1_score(y, predictions, zero_division=0)
        overall_balanced_accuracy = balanced_accuracy_score(y, predictions)

        logger.info(
            "Overall global decoding metrics (based on aggregated CV predictions):")
        logger.info(f"  Accuracy:          {overall_accuracy:.3f}")
        logger.info(f"  Precision:         {overall_precision:.3f}")
        logger.info(f"  Recall:            {overall_recall:.3f}")
        logger.info(f"  F1 Score:          {overall_f1:.3f}")
        logger.info(f"  Balanced Accuracy: {overall_balanced_accuracy:.3f}")
        logger.info(f"  ROC AUC:           {overall_roc_auc:.3f}")

        # Store metrics in a dictionary for easy return.
        metrics = {
            "accuracy": overall_accuracy,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "balanced_accuracy": overall_balanced_accuracy,
            "roc_auc": overall_roc_auc
        }
    except Exception as e:
        logger.error(f"Could not calculate overall metrics: {e}")
        metrics = {}

    return probas, predictions, scores, scores_time, metrics

# ---------------------------
# Inter-subject decoding functions
# ---------------------------


def _compute_score_at_time(t, X_train, y_train, X_test, y_test, clf_prototype):
    """Compute the AUC score for a specific time point in cross-subject decoding.

    This helper function is used within 'decode_cross_subject_fold' to calculate
    time-resolved performance. It trains a new classifier specifically on the
    data from time point `t` of the training subjects and evaluates it on the
    data from time point `t` of the test subject. An internal cross-validation
    loop on the test data is used here to get a potentially more stable score estimate
    for that single time point on the test subject, although this is computationally
    more intensive than a single prediction.

    Parameters
    ----------
    t : int
        The time point index to evaluate.
    X_train : np.ndarray
        Training EEG data (n_trials_train, n_channels, n_times).
    y_train : np.ndarray
        Training class labels (n_trials_train,).
    X_test : np.ndarray
        Test EEG data (n_trials_test, n_channels, n_times).
    y_test : np.ndarray
        Test class labels (n_trials_test,).
    clf_prototype : sklearn estimator
        A *clone* of the classifier pipeline structure (not yet fitted).

    Returns
    -------
    float
        The mean AUC score calculated for the given time point `t`, averaged
        over internal cross-validation folds on the test set.
    """
    # Extract data for the specific time point `t`.
    X_train_t = X_train[:, :, t]  # Shape (n_trials_train, n_channels)
    X_test_t = X_test[:, :, t]   # Shape (n_trials_test, n_channels)

    fold_scores_t = []
    # Define an inner cross-validation strategy (e.g., 5-fold) to apply on the test subject's data at time t.
    # This aims to provide a more robust score estimate for this time point on this subject.
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=t)

    for _, inner_test_idx in inner_cv.split(X_test_t, y_test):
        try:

            clf_t = clone(clf_prototype)
            # Train the classifier on the training data from time point `t`.
            clf_t.fit(X_train_t, y_train)

            # Predict probabilities on the inner fold's test samples from the test subject at time `t`.
            probas_t = clf_t.predict_proba(X_test_t[inner_test_idx])
            score_t = roc_auc_score(y_test[inner_test_idx], probas_t[:, 1])
            fold_scores_t.append(score_t)
        except Exception as e:
            logger.info(f"Error computing score at time {t}, inner fold: {e}")
            fold_scores_t.append(0.5)

    # Return the average score across the inner folds for time point `t`.
    # Return chance if all inner folds failed
    return np.mean(fold_scores_t) if fold_scores_t else 0.5


def decode_cross_subject_fold(X_train, y_train, X_test, y_test, test_subject, group_name, protocol, base_path, save=True):
    """Perform decoding for one fold of a cross-subject (e.g., LOSO) analysis.

    This function takes pre-defined training and testing data (where the test data
    comes from a single held-out subject), trains a classifier on the training data,
    and evaluates it on the test data. It calculates both global performance metrics
    and time-resolved scores for the test subject.

    Parameters
    ----------
    X_train : np.ndarray
        Training EEG data from multiple subjects. Shape (n_trials_train, n_channels, n_times).
    y_train : np.ndarray
        Training class labels. Shape (n_trials_train,).
    X_test : np.ndarray
        Test EEG data from the held-out subject. Shape (n_trials_test, n_channels, n_times).
    y_test : np.ndarray
        Test class labels for the held-out subject. Shape (n_trials_test,).
    test_subject : str
        Identifier of the subject used as the test set.
    group_name : str
        Name of the group or analysis context.
    protocol : str
        Name of the experimental protocol.
    base_path : str
        Base path for potentially loading auxiliary data or saving (though saving is often handled outside).
    save : bool, optional
        Indicates whether results specific to this fold should be saved (often False, as aggregation happens later). Default is True.

    Returns
    -------
    score : float
        The global ROC AUC score obtained on the test subject.
    metrics : dict
        A dictionary of various performance metrics (accuracy, precision, etc.) on the test subject.
    clf : sklearn estimator
        The classifier pipeline trained on the X_train data.
    probas : np.ndarray
        Predicted probabilities for the test subject's epochs. Shape (n_trials_test, 2).
    predictions : np.ndarray
        Predicted class labels for the test subject's epochs. Shape (n_trials_test,).
    scores_time : np.ndarray
        Time-resolved AUC scores calculated for the test subject. Shape (n_times,).

    Raises
    ------
    Exception
        Propagates exceptions occurring during fitting or prediction.
    """
    fold_start_time = time.time()
    logger.info(
        f"Cross-subject fold: Training on {X_train.shape[0]} trials, Testing on {test_subject} ({X_test.shape[0]} trials)")
    try:
        # Ensure labels are numpy arrays of integers.
        y_train = LabelEncoder().fit_transform(np.array(y_train))
        y_test = LabelEncoder().fit_transform(np.array(y_test))

        # Define the standard classification pipeline for cross-subject analysis.
        clf_prototype = make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=15),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )

        # --- Global Decoding for this Fold ---
        logger.info("Fitting global model for cross-subject fold...")
        clf = clone(clf_prototype)  # Clone the structure before fitting.
        # Flatten training data: (n_trials_train, n_channels * n_times)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        # Train the classifier on the aggregated training data.
        clf.fit(X_train_flat, y_train)
        logger.info("Global model fitting complete.")

        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        probas = clf.predict_proba(X_test_flat)
        predictions = clf.predict(X_test_flat)

        # Calculate global performance metrics for the test subject.
        # AUC based on positive class probability
        score = roc_auc_score(y_test, probas[:, 1])
        metrics = {
            "roc_auc": score,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, predictions)
        }
        logger.info(
            f"Global metrics for test subject {test_subject}: AUC={score:.3f}, Acc={metrics['accuracy']:.3f}")

        # --- Temporal Decoding for this fold ---
        logger.info(
            f"Starting temporal decoding for test subject {test_subject}...")

        n_times = X_test.shape[2]
        scores_time = np.zeros(n_times)

        # Setup parallel execution for computing scores at each time point.
        parallel, pfunc, n_jobs = parallel_func(
            _compute_score_at_time, n_jobs=-1, verbose=0)
        logger.info(
            f"Running temporal decoding in parallel with {n_jobs} jobs.")
        temporal_results = parallel(pfunc(t, X_train, y_train, X_test, y_test, clone(clf_prototype))
                                    for t in range(n_times))
        # Collect results from parallel jobs.
        scores_time = np.array(temporal_results)

        logger.info(
            f"Temporal decoding for {test_subject} complete. Peak AUC: {np.max(scores_time):.3f}")
        logger.info(
            f"Cross-subject fold processed in {time.time() - fold_start_time:.2f} seconds.")

        return score, metrics, clf, probas, predictions, scores_time

    except Exception as e:

        logger.error(
            f"Error during cross-subject decoding fold for test subject {test_subject}: {e}", exc_info=True)

        raise

# ---------------------------
# Plotting Functions
# ---------------------------


def plot_decoding_results_dashboard(epochs, X, y, probas, predictions, cv_scores,
                                    scores_time, subject_id, group, save_dir=None):
    """Generate a multi-page dashboard visualizing decoding results for a subject.

    Creates a set of plots summarizing the outcome of a decoding analysis
    (either intra-subject or one fold of cross-subject). Includes temporal
    decoding scores, cross-validation fold scores, ROC curve, confusion matrix,
    probability distributions, and global performance metrics.

    Parameters
    ----------
    epochs : mne.Epochs
        The MNE Epochs object associated with the data (used for time axis).
    X : np.ndarray
        The input data used for decoding (primarily for context, not directly plotted).
    y : np.ndarray
        The true labels used for decoding.
    probas : np.ndarray
        Predicted probabilities (n_epochs, n_classes), typically from global decoding.
    predictions : np.ndarray
        Predicted labels (n_epochs,), typically from global decoding.
    cv_scores : np.ndarray or list
        Scores obtained for each fold of cross-validation (e.g. AUC).
        For cross-subject fold plots, this might contain only one score.
    scores_time : np.ndarray
        Time-resolved scores (e.g. AUC) over the epoch duration (n_times,).
    subject_id : str
        Identifier of the subject for titles and filenames.
    group : str
        Group information (e.g., 'controls', 'del', 'cross-subject test') for context.
    save_dir : str or None, optional
        Directory where the generated plot files should be saved.Default is None.

    Returns
    -------
    str or None
        The directory where plots were saved, or None if not saved.
    """
    logger.info(
        f"Generating dashboard plots for {subject_id} (Group: {group})")

    le = LabelEncoder()
    le.fit(y)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True) 
        logger.info(f"Plots will be saved in: {save_dir}")
    else:
        save_dir = os.getcwd()
        logger.info(
            "Save directory not specified, plots might not be saved to file.")

    # Use a non-interactive backend to prevent plots from popping up if running in batch.
    plt.switch_backend('Agg')  # Use 'Agg' for file saving without display (for cluster)

    # ====================== PAGE 1 ======================
    try:
        fig1 = plt.figure(figsize=(15, 10))  # 15 inch large by 10 height inch 
        # Main title for the first page.
        fig1.suptitle(f"Decoding Dashboard - Subject: {subject_id} (Group: {group}) - Page 1/2",
                      fontsize=16, fontweight='bold')

        # Use GridSpec for flexible subplot layout.
        # Give more height to temporal plot because we need to seedetails
        gs1 = GridSpec(2, 2, figure=fig1, height_ratios=[2, 1])

        # --- Plot 1: Temporal Decoding Scores ---

        ax_temporal = fig1.add_subplot(gs1[0, :])
        # Plot the mean score over time.
        ax_temporal.plot(epochs.times, scores_time, 'b-',
                         linewidth=2.0, label='Mean AUC over time')
        std_scores = np.std(scores_time) * 0.5
        ax_temporal.fill_between(epochs.times, scores_time - std_scores,
                                 scores_time + std_scores, color='blue', alpha=0.2, label='Std dev')
        ax_temporal.axhline(0.5, color='k', linestyle='--',
                            linewidth=1.0, label='Chance level (AUC=0.5)')
        ax_temporal.axvline(0, color='r', linestyle=':',
                            linewidth=1.0, label='Stimulus onset (t=0)')
        # Highlight the maximum score point.
        if len(scores_time) > 0:
            max_time_idx = np.argmax(scores_time)
            max_score = scores_time[max_time_idx]
            max_time = epochs.times[max_time_idx]
            ax_temporal.plot(max_time, max_score, 'ro',
                             markersize=6, label=f'Max AUC: {max_score:.3f}')   # Annotate the max score point in red for clarity 
            ax_temporal.annotate(
                f'{max_score:.3f} @ {max_time * 1000:.0f}ms',
                xy=(max_time, max_score),
                # Adjust text position slightly
                xytext=(max_time + 0.05, max_score + 0.02),
          
            )
        ax_temporal.set_xlabel(
            'Time relative to stimulus onset (s)', fontsize=12)
        ax_temporal.set_ylabel('ROC AUC Score', fontsize=12)
        ax_temporal.set_ylim(min(0.4, np.min(scores_time)-0.05) if len(scores_time) > 0 else 0.4,  # Adjust ylim dynamically between 0.4 and 1
                             max(1.0, np.max(scores_time)+0.05) if len(scores_time) > 0 else 1.0)
        ax_temporal.set_title('Temporal Decoding Performance', fontsize=14)
        # Let matplotlib choose best legend location
        ax_temporal.legend(loc='best')
        ax_temporal.grid(True, linestyle=':', alpha=0.6)

        # --- Plot 2: Cross-Validation Scores (Global Decoding) ---
        ax_cv = fig1.add_subplot(gs1[1, 0])  # Bottom-left subplot
        n_folds = len(cv_scores)
        if n_folds > 0:
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            ax_cv.bar(range(1, n_folds + 1), cv_scores,
                      color='skyblue', label='Fold Score')
            # Add mean line.
            ax_cv.axhline(mean_cv_score, color='r', linestyle='--',
                          label=f'Mean: {mean_cv_score:.3f}')
            ax_cv.set_xlabel('Cross-Validation Fold', fontsize=10)
            ax_cv.set_ylabel('ROC AUC Score', fontsize=10)
            # Ensure ticks for each fold if not too many folds
            ax_cv.set_xticks(range(1, n_folds + 1))
            ax_cv.set_ylim(0.0, 1.05)
            ax_cv.set_title(
                f'Global Decoding CV Scores ({n_folds} folds)', fontsize=14)
            ax_cv.legend()
        else:
            ax_cv.text(0.5, 0.5, 'CV scores not available',
                       horizontalalignment='center', verticalalignment='center')
            ax_cv.set_title('Global Decoding CV Scores', fontsize=14)

        # --- Plot 3: ROC Curve (Global Decoding) ---
        ax_roc = fig1.add_subplot(gs1[1, 1])  # Bottom-right subplot
        if probas is not None and len(y) > 0:
            # Use probability of positive class
            fpr, tpr, thresholds = roc_curve(y, probas[:, 1])
            # Calculate AUC from the global predictions/probabilities.
            # Should ideally match the mean CV score if stable, but calculated on aggregated predictions here.
            global_auc = roc_auc_score(y, probas[:, 1])
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (Aggregated AUC = {global_auc:.3f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=1,
                        linestyle=':', label='Chance (AUC = 0.5)')  # Chance line
            ax_roc.set_xlim([-0.02, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel(
                'False positive rate (1 - specificity) = FPR', fontsize=10)
            ax_roc.set_ylabel('True positive rate (sensitivity = TPR)', fontsize=10)
            ax_roc.set_title(
                'Receiver operating characteristic (ROC)', fontsize=14)
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle=':', alpha=0.6)
        else:
            ax_roc.text(0.5, 0.5, 'ROC data not available',
                        horizontalalignment='center', verticalalignment='center')
            ax_roc.set_title(
                'Receiver Operating Characteristic (ROC)', fontsize=14)

        # Adjust layout to prevent overlap.
        # Adjust rect to make space for suptitle
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        if save_dir:
            page1_path = os.path.join(
                save_dir, f"dashboard_{subject_id}_{group}_page1.png")
            try:
                fig1.savefig(page1_path, dpi=300, bbox_inches='tight')
                logger.info(f"Dashboard page 1 saved to {page1_path}")
            except Exception as e:
                logger.error(f"Failed to save dashboard page 1: {e}")

    except Exception as e:
        logger.error(
            f"Error creating dashboard page 1 for {subject_id}: {e}", exc_info=True)
    finally:
        plt.close(fig1) 

    # ====================== PAGE 2 ======================
    try:
        fig2 = plt.figure(figsize=(15, 10))
        fig2.suptitle(f"Decoding Dashboard - Subject: {subject_id} (Group: {group}) - Page 2/2",
                      fontsize=16, fontweight='bold')

        gs2 = GridSpec(2, 2, figure=fig2)  # 2x2 layout for page 2

        # --- Plot 4: Confusion Matrix (Global Decoding) ---
        ax_cm = fig2.add_subplot(gs2[0, 0])  # Top-left subplot
        if predictions is not None and len(y) > 0:
            # Compute confusion matrix, normalized to show percentages (recall) per true class.
            cm = confusion_matrix(y, predictions, normalize='true')
            # Use seaborn heatmap for better visualization.
            sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', ax=ax_cm,  # Show percentages
                        xticklabels=le.classes_, yticklabels=le.classes_,
                        cbar=False)  

            ax_cm.set_xlabel('Predicted Label', fontsize=10)
            ax_cm.set_ylabel('True Label', fontsize=10)
            ax_cm.set_title('Normalized Confusion Matrix', fontsize=14)
        else:
            ax_cm.text(0.5, 0.5, 'Confusion matrix data not available',
                       horizontalalignment='center', verticalalignment='center')
            ax_cm.set_title('Normalized Confusion Matrix', fontsize=14)

        # --- Plot 5: Probability Distribution (Global Decoding) ---
        ax_dist = fig2.add_subplot(gs2[0, 1])  # Top-right subplot
        if probas is not None and len(y) > 0:
            # Plot kernel density estimates of the predicted probabilities for each true class.
            for i, cls_name in enumerate(le.classes_):
                sns.kdeplot(probas[y == i, 1], ax=ax_dist,  # Use probability of positive class (class 1)
                            label=f'True Class: {cls_name}', fill=True, alpha=0.5, bw_adjust=0.5)  # Adjust bandwidth for smoothness
            # Add decision threshold line.
            ax_dist.axvline(x=0.5, color='red', linestyle='--',
                            linewidth=1.0, label='Decision threshold (0.5)')
            ax_dist.set_xlabel(
                'Predicted Probability (for Class 1)', fontsize=10)
            ax_dist.set_ylabel('Density', fontsize=10)
            ax_dist.set_title(
                'Distribution of Predicted Probabilities', fontsize=14)
            ax_dist.legend()
            ax_dist.grid(True, linestyle=':', alpha=0.6)
        else:
            ax_dist.text(0.5, 0.5, 'Probability data not available',
                         horizontalalignment='center', verticalalignment='center')
            ax_dist.set_title(
                'Distribution of Predicted Probabilities', fontsize=14)

        # --- Plot 6: Global Performance Metrics Summary ---
        ax_metrics = fig2.add_subplot(gs2[1, :])  # Span bottom row
        metrics_to_plot = {}
        if predictions is not None and probas is not None and len(y) > 0:
            try:
                metrics_to_plot = {
                    "Accuracy": accuracy_score(y, predictions),
                    "Balanced Acc": balanced_accuracy_score(y, predictions),
                    # Assuming 1 is positive class
                    "Precision": precision_score(y, predictions, pos_label=1, zero_division=0),
                    "Recall (Sens)": recall_score(y, predictions, pos_label=1, zero_division=0),
                    "F1 Score": f1_score(y, predictions, pos_label=1, zero_division=0),
                    "AUC": roc_auc_score(y, probas[:, 1])
                }
            except Exception as metric_error:
                logger.warning(
                    f"Could not calculate some metrics for plotting: {metric_error}")

        if metrics_to_plot:
            keys = list(metrics_to_plot.keys())
            values = [metrics_to_plot[k] for k in keys]
            bars = ax_metrics.bar(keys, values, color='lightcoral')
            ax_metrics.set_ylim(0, 1.05)
            ax_metrics.set_ylabel('Score', fontsize=10)
            ax_metrics.set_title(
                'Overall Performance Metrics (Global Decoding)', fontsize=14)
            # Add text labels above bars.
            for bar in bars:
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            # Keep labels horizontal if few enough metrics
            ax_metrics.tick_params(axis='x', rotation=0)
        else:
            ax_metrics.text(0.5, 0.5, 'Metrics data not available',
                            horizontalalignment='center', verticalalignment='center')
            ax_metrics.set_title(
                'Overall Performance Metrics (Global Decoding)', fontsize=14)

        # Adjust layout for page 2.
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            page2_path = os.path.join(
                save_dir, f"dashboard_{subject_id}_{group}_page2.png")
            try:
                fig2.savefig(page2_path, dpi=200, bbox_inches='tight')
                logger.info(f"Dashboard page 2 saved to {page2_path}")
            except Exception as e:
                logger.error(f"Failed to save dashboard page 2: {e}")

    except Exception as e:
        logger.error(
            f"Error creating dashboard page 2 for {subject_id}: {e}", exc_info=True)
    finally:
        plt.close(fig2) 

    logger.info(f"Finished generating dashboard plots for {subject_id}")
    return save_dir 


def plot_group_results(subject_scores, group_name, save_dir=None):
    """Visualize the decoding scores for a group of subjects.

    Creates a bar plot showing the primary decoding score (assumed to be AUC
    or similar, passed in `subject_scores`) for each subject in the group.
    Includes lines for mean score and chance level.

    Parameters
    ----------
    subject_scores : dict
        A dictionary where keys are subject identifiers (str) and values are
        their corresponding decoding scores (float or None). None values are excluded.
    group_name : str
        The name of the group for the plot title and filename.
    save_dir : str or None, optional
        Directory where the plot file should be saved. If None, the plot is shown
        but may not be saved to a file. Default is None.
    """
    # Filter out subjects with None scores, as they cannot be plotted.
    valid_data = {subj: score for subj,
                  score in subject_scores.items() if score is not None}

    if not valid_data:
        logger.warning(
            f"No valid scores found for group {group_name}. Skipping group plot.")
        print(f"Warning: No valid scores to plot for group {group_name}.")
        return  # Do not attempt to plot if no valid data exists.

    subjects = list(valid_data.keys())
    scores = list(valid_data.values())
    n_subjects = len(subjects)

    logger.info(
        f"Generating group results plot for {group_name} ({n_subjects} valid subjects)")

    # Use Agg backend for saving without display
    plt.switch_backend('Agg')
    # Adjust width based on number of subjects
    fig_group, ax_group = plt.subplots(figsize=(max(8, n_subjects * 0.6), 6))

    # Create the bar plot.
    bars = ax_group.bar(range(n_subjects), scores,
                        color='skyblue', label='Subject Score')
    # Calculate and plot the mean score.
    mean_score = np.mean(scores)
    ax_group.axhline(mean_score, color='r', linestyle='--', linewidth=1.5,
                     label=f'Mean: {mean_score:.3f}')
    # Plot the chance level (assuming 0.5 for AUC in binary classification).
    ax_group.axhline(0.5, color='k', linestyle=':',
                     linewidth=1.0, label='Chance Level (0.5)')

    # Set labels and title.
    ax_group.set_ylabel('Decoding Score (e.g., ROC AUC)', fontsize=12)
    ax_group.set_xlabel('Subject ID', fontsize=12)
    ax_group.set_title(
        f'Decoding Performance - Group: {group_name}', fontsize=14, fontweight='bold')
    
    ax_group.set_xticks(range(n_subjects))
    # Rotate labels for readability if many subjects.
    ax_group.set_xticklabels(subjects, rotation=90, fontsize=9)
    ax_group.set_ylim(min(0.4, np.min(scores) - 0.05) if scores else 0.4,  # Dynamic y-axis lower limit
                      max(1.0, np.max(scores) + 0.05) if scores else 1.0)  # Dynamic y-axis upper limit

   
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax_group.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Place legend outside plot area
    ax_group.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_group.grid(True, axis='y', linestyle=':', alpha=0.6)

    # Adjust layout to make space for legend outside
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save the plot if a directory is specified.
    if save_dir:
        os.makedirs(save_dir, exist_ok=True) 
        save_path = os.path.join(
            save_dir, f'group_{group_name}_scores_summary.png')
        try:
            fig_group.savefig(save_path, dpi=200, bbox_inches='tight')
            logger.info(f"Group results plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save group plot: {e}")
    else:
        # plt.show() # Uncomment if interactive display is desired and backend supports it.
        pass

    plt.close(fig_group)  
    # TODO: I want to add more advanced group visualizations (e.g., boxplots, violin plots later).
