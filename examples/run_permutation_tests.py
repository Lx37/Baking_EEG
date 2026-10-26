#!/usr/bin/env python
"""Test de permutation des labels, sujet par sujet.

Pourquoi ce script
------------------
Les plis d'une validation croisee partagent leurs essais d'entrainement : les
AUC des plis ne sont pas independantes et un Wilcoxon (ou un t-test) sur ces
AUC donne trop de faux positifs, quel que soit le nombre de plis (Nadeau &
Bengio 2003 ; Stelzer et al. 2013 ; Noirhomme et al. 2014). En simulation sur
nos dimensions (64/64 essais, pas d'effet), ce Wilcoxon declarait 22 % (k=5)
a 30 % (k=10) des sujets significatifs au lieu de 5 %.

Le test valide au niveau individuel consiste a refaire EXACTEMENT la meme
validation croisee apres avoir permute les labels des essais, un grand nombre
de fois, et a situer l'AUC observee dans cette distribution nulle (Ojala &
Garriga 2010). p = (b + 1) / (m + 1), b = nombre de permutations dont l'AUC
est >= a l'AUC observee, m = nombre de permutations (Phipson & Smyth 2010) ;
c'est la formule de sklearn.model_selection.permutation_test_score.

Ce qui est decode
-----------------
Meme pipeline que le decodage global de run_decoding_one_pp/lg :
StandardScaler -> SelectPercentile(f_classif, 15 %) -> SVC lineaire C=1,
class_weight='balanced', sur les essais aplatis (electrodes x temps). Deux
differences, voulues :
  - fenetre post-stimulus fixe, identique pour tous les sujets d'un paradigme
    (PP : 0-1000 ms ; LG : 0-800 ms apres le 5e son) ;
  - 5 plis pour TOUS les sujets (le minimum des patients DoC), si bien que
    l'AUC obtenue est comparable d'un groupe a l'autre. L'AUC est calculee
    sur la fonction de decision du SVC (pas de calibration de Platt, inutile
    pour un score de rang).

Usage
-----
    python examples/run_permutation_tests.py --paradigm PP --subject TJR7 --group CONTROLS
    python examples/run_permutation_tests.py --paradigm LG --subject TpSM49 --group DELIRIUM+

Sortie : <resultats>/permutation_tests/<PP|LG>/<SUJET>_<contraste>.npz (+ .json)
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
from getpass import getuser

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold, permutation_test_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.base_decoding import _build_standard_classifier_pipeline  # noqa: E402
from config.decoding_config import (  # noqa: E402
    CLASSIFIER_MODEL_TYPE, FIXED_CLASSIFIER_PARAMS_CONFIG,
    USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
)
from utils.loading_LG_utils import load_epochs_data_for_lg_decoding  # noqa: E402
from utils.loading_PP_utils import load_epochs_data_auto_protocol  # noqa: E402
from utils.permutation_stats import PERMUTATION_SUBDIR  # noqa: E402
from utils.utils import configure_project_paths  # noqa: E402

logger = logging.getLogger(__name__)

N_FOLDS = 5
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# contraste -> (classe 0, classe 1) ; codes evenements MNE identiques a ceux du
# decodage (PP = classe positive, comme dans run_decoding_one_pp).
PARADIGMS = {
    "PP": {
        "loader": load_epochs_data_auto_protocol,
        "conditions": {"XAP_ALL": "AP/", "XPP_ALL": "PP/"},
        "contrasts": {"pp_ap": ("XAP_ALL", "XPP_ALL")},
        "window_s": (0.0, 1.0),
    },
    "LG": {
        "loader": load_epochs_data_for_lg_decoding,
        "conditions": {"LS_ALL": "LS/", "LD_ALL": "LD/",
                       "GS_ALL": "/GS", "GD_ALL": "/GD"},
        "contrasts": {"ls_ld": ("LS_ALL", "LD_ALL"),
                      "gs_gd": ("GS_ALL", "GD_ALL")},
        "window_s": (0.0, 0.8),
    },
}


def build_pipeline():
    """Pipeline du decodage global, sans calibration de Platt."""
    params = FIXED_CLASSIFIER_PARAMS_CONFIG[CLASSIFIER_MODEL_TYPE]
    pipeline, clf_name, _, _ = _build_standard_classifier_pipeline(
        classifier_model_type=CLASSIFIER_MODEL_TYPE,
        random_seed_state=RANDOM_SEED,
        use_grid_search=False,
        add_anova_fs_step=USE_ANOVA_FS_FOR_TEMPORAL_PIPELINES,
        **params,
    )
    if CLASSIFIER_MODEL_TYPE == "svc":
        pipeline.set_params(**{f"{clf_name}__probability": False})
    return pipeline


def permutation_test_subject(x_class0, x_class1, times, window_s,
                             n_permutations=N_PERMUTATIONS, n_jobs=-1,
                             n_folds=N_FOLDS, seed=RANDOM_SEED):
    """AUC observee et p de permutation pour un sujet et un contraste.

    x_class0, x_class1 : (n_essais, n_electrodes, n_temps)
    """
    tmask = (times >= window_s[0] - 1e-9) & (times <= window_s[1] + 1e-9)
    if not tmask.any():
        raise ValueError(f"Aucun echantillon dans la fenetre {window_s} s.")
    X = np.concatenate([x_class0, x_class1])[:, :, tmask]
    X = X.reshape(X.shape[0], -1)
    y = np.r_[np.zeros(len(x_class0), int), np.ones(len(x_class1), int)]

    k = min(n_folds, len(x_class0), len(x_class1))
    if k < 2:
        raise ValueError(f"Trop peu d'essais ({len(x_class0)}/{len(x_class1)}).")
    if k < n_folds:
        logger.warning("Seulement %d plis possibles (classe minoritaire).", k)

    auc, null, p_value = permutation_test_score(
        build_pipeline(), X, y,
        cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed),
        n_permutations=n_permutations, scoring="roc_auc",
        n_jobs=n_jobs, random_state=seed,
    )
    return {
        "auc": float(auc), "p_value": float(p_value),
        "null_distribution": null.astype(np.float32),
        "n_trials_class0": len(x_class0), "n_trials_class1": len(x_class1),
        "n_folds": k, "n_permutations": n_permutations,
        "window_s": np.asarray(window_s, float), "n_features": X.shape[1],
    }


def _code_version():
    """Commit git (si dispo) et empreinte de ce script, pour la tracabilite."""
    try:
        commit = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    with open(__file__, "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()[:12]
    return commit or "inconnu", sha1


def run_subject(paradigm, subject_id, group, base_input_path=None,
                base_output_path=None, n_permutations=N_PERMUTATIONS,
                n_jobs=-1):
    """Charge les epochs d'un sujet et teste chaque contraste du paradigme.

    Renvoie la liste des fichiers NPZ ecrits.
    """
    if not base_input_path or not base_output_path:
        cfg_in, cfg_out = configure_project_paths(getuser())
        base_input_path = base_input_path or cfg_in
        base_output_path = base_output_path or cfg_out

    spec = PARADIGMS[paradigm]
    epochs, data, protocol = spec["loader"](
        subject_id, group, base_input_path, spec["conditions"], False)
    if epochs is None:
        raise FileNotFoundError(
            f"Epochs introuvables pour {subject_id} ({group}, {paradigm}).")

    out_dir = os.path.join(base_output_path, PERMUTATION_SUBDIR, paradigm)
    os.makedirs(out_dir, exist_ok=True)
    commit, script_sha1 = _code_version()
    written = []

    for contrast, (key0, key1) in spec["contrasts"].items():
        x0, x1 = data.get(key0), data.get(key1)
        if x0 is None or x1 is None or len(x0) == 0 or len(x1) == 0:
            logger.error("%s %s : classe vide (%s=%s, %s=%s), contraste saute.",
                         subject_id, contrast, key0, getattr(x0, "shape", None),
                         key1, getattr(x1, "shape", None))
            continue
        logger.info("%s %s : %d vs %d essais, %d permutations...",
                    subject_id, contrast, len(x0), len(x1), n_permutations)
        res = permutation_test_subject(x0, x1, epochs.times, spec["window_s"],
                                       n_permutations, n_jobs)
        meta = {
            "subject_id": subject_id, "group": group, "paradigm": paradigm,
            "contrast": contrast, "class0": key0, "class1": key1,
            "detected_protocol": str(protocol),
            "sfreq": float(epochs.info["sfreq"]),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
            "code_commit": commit, "script_sha1": script_sha1,
            "sklearn_version": sklearn.__version__,
        }
        path = os.path.join(out_dir, f"{subject_id}_{contrast}.npz")
        np.savez(path, **res, **meta)
        summary = {**meta, **{k: v for k, v in res.items()
                              if k != "null_distribution"}}
        summary["window_s"] = list(spec["window_s"])
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("%s %s : AUC=%.3f  p=%.4f  (k=%d)  -> %s", subject_id,
                    contrast, res["auc"], res["p_value"], res["n_folds"], path)
        written.append(path)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paradigm", required=True, choices=sorted(PARADIGMS))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--group", required=True,
                        help="Groupe de ALL_SUBJECTS_GROUPS (sert a trouver le fichier).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    written = run_subject(args.paradigm, args.subject, args.group,
                          args.data_dir, args.results_dir,
                          args.n_permutations, args.n_jobs)
    if not written:
        sys.exit(1)


if __name__ == "__main__":
    main()
