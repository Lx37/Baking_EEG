"""Alignement temporel des courbes de decodage entre sujets.

Contexte
--------
Tous les sujets n'ont pas le meme nombre de points temporels : selon le
protocole et la frequence d'echantillonnage on rencontre 601, 701, 401 (TYS6 en
LG) ou 301 points (TpJC5 en PP).

L'ancienne strategie des scripts de groupe etait :

    TARGET_LEN = 601
    if scores.shape == (601,):   garder
    elif scores.shape == (701,): scores = scores[:601]   # troncature
    else:                        continue                # sujet SUPPRIME

Deux problemes :

1. La troncature suppose que les deux vecteurs partagent la meme grille
   temporelle sur leurs 601 premiers points. Si les frequences
   d'echantillonnage different, l'indice i ne designe pas le meme instant
   d'un sujet a l'autre : on moyenne des instants differents.
2. Tout sujet d'une autre longueur etait ecarte silencieusement du groupe,
   alors que la config documente elle-meme l'existence de tels sujets.

Ce module remplace la troncature par une interpolation lineaire sur une grille
temporelle commune, definie par l'intersection des fenetres de tous les sujets.
Aucun sujet n'est exclu pour une simple question de longueur, et un point
d'indice i correspond au meme instant chez tout le monde.
"""

import logging

import numpy as np

logger_time_alignment = logging.getLogger(__name__)


def build_common_time_grid(time_vectors, target_n_points=None,
                           subject_labels=None):
    """Construit une grille temporelle commune a partir de plusieurs sujets.

    La grille couvre l'intersection des fenetres temporelles (max des debuts,
    min des fins), avec une resolution egale a la resolution mediane observee
    (ou `target_n_points` points si precise).

    Args:
        time_vectors (list[np.ndarray]): un vecteur de temps par sujet.
        target_n_points (int, optional): nombre de points impose.
        subject_labels (list[str], optional): identifiants, pour les messages.

    Returns:
        np.ndarray | None: grille commune, ou None si aucune intersection.
    """
    valid = [(np.asarray(t, dtype=float), i)
             for i, t in enumerate(time_vectors)
             if t is not None and np.size(t) >= 2]
    if not valid:
        logger_time_alignment.error(
            "Aucun vecteur temps exploitable : grille commune impossible.")
        return None

    starts = np.array([t[0] for t, _ in valid])
    stops = np.array([t[-1] for t, _ in valid])
    t_start, t_stop = float(np.max(starts)), float(np.min(stops))

    if not np.isfinite(t_start) or not np.isfinite(t_stop) or t_stop <= t_start:
        logger_time_alignment.error(
            "Fenetres temporelles disjointes (intersection [%.3f, %.3f]) : "
            "grille commune impossible.", t_start, t_stop)
        return None

    if target_n_points is None:
        # Resolution mediane des sujets, ramenee a la fenetre commune.
        steps = [float(np.median(np.diff(t))) for t, _ in valid]
        median_step = float(np.median(steps))
        if median_step <= 0:
            median_step = (t_stop - t_start) / 600.0
        target_n_points = int(round((t_stop - t_start) / median_step)) + 1
    target_n_points = max(int(target_n_points), 2)

    grid = np.linspace(t_start, t_stop, target_n_points)

    n_trimmed = sum(1 for t, _ in valid
                    if t[0] < t_start - 1e-9 or t[-1] > t_stop + 1e-9)
    logger_time_alignment.info(
        "Grille temporelle commune : [%.3f, %.3f] s, %d points "
        "(pas %.4f s) sur %d sujets.",
        t_start, t_stop, target_n_points, grid[1] - grid[0], len(valid))
    if n_trimmed:
        logger_time_alignment.info(
            "%d sujet(s) ont une fenetre plus large : ils sont rogne(s) a "
            "l'intersection commune.", n_trimmed)

    if subject_labels is not None:
        lengths = {}
        for t, idx in valid:
            lengths.setdefault(len(t), []).append(
                subject_labels[idx] if idx < len(subject_labels) else str(idx))
        if len(lengths) > 1:
            logger_time_alignment.info(
                "Longueurs temporelles heterogenes (reinterpolees) : %s",
                {n: (subs if len(subs) <= 6 else subs[:6] + ["..."])
                 for n, subs in sorted(lengths.items())})

    return grid


def resample_to_grid(values, source_times, target_times):
    """Reinterpole des scores sur une grille temporelle cible.

    Gere les vecteurs 1D (n_times,) et les matrices carrees de generalisation
    temporelle (n_times, n_times), interpolees sur les deux axes.

    Les NaN sont propages : un point cible entoure de NaN reste NaN.

    Args:
        values (np.ndarray): (n_times,) ou (n_times, n_times).
        source_times (np.ndarray): grille d'origine, longueur n_times.
        target_times (np.ndarray): grille cible.

    Returns:
        np.ndarray | None: valeurs reinterpolees, ou None si incoherent.
    """
    if values is None or source_times is None or target_times is None:
        return None
    values = np.asarray(values, dtype=float)
    source_times = np.asarray(source_times, dtype=float)
    target_times = np.asarray(target_times, dtype=float)

    if values.ndim == 1:
        if values.shape[0] != source_times.shape[0]:
            logger_time_alignment.warning(
                "Longueurs incoherentes (scores %d, temps %d) : "
                "reinterpolation impossible.",
                values.shape[0], source_times.shape[0])
            return None
        finite = np.isfinite(values)
        if finite.sum() < 2:
            return np.full(target_times.shape, np.nan)
        out = np.interp(target_times, source_times[finite], values[finite],
                        left=np.nan, right=np.nan)
        # Un point cible situe dans un trou de NaN de la source doit rester NaN.
        nan_source = np.interp(target_times, source_times,
                               (~finite).astype(float), left=1.0, right=1.0)
        out[nan_source > 0.5] = np.nan
        return out

    if values.ndim == 2:
        if values.shape[0] != source_times.shape[0] or \
                values.shape[1] != source_times.shape[0]:
            logger_time_alignment.warning(
                "Matrice %s incompatible avec %d points temporels : "
                "reinterpolation impossible.",
                values.shape, source_times.shape[0])
            return None
        # Interpolation ligne par ligne puis colonne par colonne.
        intermediate = np.vstack([
            resample_to_grid(values[i, :], source_times, target_times)
            for i in range(values.shape[0])
        ])
        result = np.column_stack([
            resample_to_grid(intermediate[:, j], source_times, target_times)
            for j in range(intermediate.shape[1])
        ])
        return result

    logger_time_alignment.warning(
        "Dimension %dD non geree par resample_to_grid.", values.ndim)
    return None


def align_subject_curves(curves, time_vectors, subject_labels=None,
                         target_n_points=None):
    """Aligne les courbes de plusieurs sujets sur une grille commune.

    Args:
        curves (list[np.ndarray]): une courbe (ou TGM) par sujet.
        time_vectors (list[np.ndarray]): le vecteur temps de chaque sujet.
        subject_labels (list[str], optional): identifiants sujets.
        target_n_points (int, optional): nombre de points impose.

    Returns:
        tuple: (matrice_alignee, grille_commune, labels_conserves,
                labels_ecartes)
        `matrice_alignee` vaut None si aucun sujet n'a pu etre aligne.
    """
    labels = list(subject_labels) if subject_labels is not None else [
        str(i) for i in range(len(curves))]

    grid = build_common_time_grid(time_vectors, target_n_points, labels)
    if grid is None:
        return None, None, [], labels

    aligned, kept, dropped = [], [], []
    for curve, times, label in zip(curves, time_vectors, labels):
        resampled = resample_to_grid(curve, times, grid)
        if resampled is None or np.all(np.isnan(resampled)):
            dropped.append(label)
            continue
        aligned.append(resampled)
        kept.append(label)

    if dropped:
        logger_time_alignment.warning(
            "%d sujet(s) ecarte(s) faute de donnees exploitables : %s",
            len(dropped), dropped)

    if not aligned:
        return None, grid, [], dropped

    return np.array(aligned), grid, kept, dropped
