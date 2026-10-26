#!/usr/bin/env python
"""Submit one Slurm job per subject for the individual permutation test.

See examples/run_permutation_tests.py for the method. The results are stored
in <results>/permutation_tests/<PP|LG>/ and are read by the group scripts
run_group_npz_pp.py and run_group_npz_lg.py.
By default, PP AND LG are submitted together, one job per subject and per
paradigm, all at once: Slurm runs them in parallel.

Usage :
    python submitit/submit_permutation_tests.py --dry-run
    python submitit/submit_permutation_tests.py
    python submitit/submit_permutation_tests.py --missing-only
    python submitit/submit_permutation_tests.py --paradigm LG --groups COMA VS
    python submitit/submit_permutation_tests.py --paradigm PP --subjects TJR7

Environment variables: BAKING_EEG_VENV_ACTIVATE, BAKING_EEG_PROJECT_ROOT,
BAKING_EEG_SLURM_ACCOUNT, BAKING_EEG_TIMEOUT_MIN, BAKING_EEG_DATA_DIR,
BAKING_EEG_RESULTS_DIR
"""
import argparse
import getpass
import logging
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import ALL_SUBJECTS_GROUPS  # noqa: E402
from examples.run_permutation_tests import N_PERMUTATIONS, PARADIGMS  # noqa: E402
from utils.permutation_stats import PERMUTATION_SUBDIR  # noqa: E402
from utils.utils import configure_project_paths  # noqa: E402

logger = logging.getLogger(__name__)

N_CPUS = 40
MEMORY = "60G"
PARTITION = "CPU"
VENV = os.environ.get("BAKING_EEG_VENV_ACTIVATE",
                      "/home/tom.balay/.venvs/py3.11_cluster/bin/activate")
WORKER_ROOT = os.environ.get("BAKING_EEG_PROJECT_ROOT", PROJECT_ROOT)
SETUP = [
    f'cd "{WORKER_ROOT}" || exit 1',
    "module purge",
    f'source "{VENV}" || exit 1',
    f'export PYTHONPATH="{WORKER_ROOT}${{PYTHONPATH:+:$PYTHONPATH}}"',
    "export PYTHONUNBUFFERED=1",
    'python -c "import mne, sklearn; print(mne.__version__, sklearn.__version__)" || exit 1',
]


def run_on_worker(paradigm, subject_id, group, data_dir, results_dir,
                  n_permutations, n_jobs):
    """Execute sur le noeud : import local pour ne charger mne que la-bas."""
    import logging as _logging
    import sys as _sys
    root = os.environ.get("BAKING_EEG_PROJECT_ROOT", WORKER_ROOT)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    from examples.run_permutation_tests import run_subject
    return run_subject(paradigm, subject_id, group, data_dir, results_dir,
                       n_permutations, n_jobs)


def select_subjects(groups=None, subjects=None):
    """{sujet: groupe}. Un sujet liste dans deux groupes n'est soumis qu'une
    fois (meme fichier de sortie) ; il est relu dans chaque groupe ensuite."""
    unknown = sorted(set(groups or []) - set(ALL_SUBJECTS_GROUPS))
    known_ids = {s for ids in ALL_SUBJECTS_GROUPS.values() for s in ids}
    unknown += sorted(set(subjects or []) - known_ids)
    if unknown:
        sys.exit(f"Inconnus dans ALL_SUBJECTS_GROUPS : {unknown}")
    selected = {}
    for group, ids in ALL_SUBJECTS_GROUPS.items():
        if groups and group not in groups:
            continue
        for sid in ids:
            if subjects and sid not in subjects:
                continue
            if sid in selected:
                logger.warning("%s est dans %s et %s : soumis une seule fois.",
                               sid, selected[sid], group)
                continue
            selected[sid] = group
    return selected


def has_data(paradigm, subject_id, group, data_dir):
    """Le fichier d'epochs existe-t-il ? (meme recherche que check_data_tree.py)"""
    from check_data_tree import LG_DIRS, PP_DIRS, find_lg_file
    from utils.loading_PP_utils import find_epochs_file_with_protocol_detection
    if paradigm == "PP":
        path, _ = find_epochs_file_with_protocol_detection(
            os.path.join(data_dir, PP_DIRS.get(group, "?")), subject_id, False)
        return path is not None
    return find_lg_file(os.path.join(data_dir, LG_DIRS.get(group, "?")), subject_id) is not None


def is_done(results_dir, paradigm, subject_id):
    out = os.path.join(results_dir, PERMUTATION_SUBDIR, paradigm)
    return all(os.path.exists(os.path.join(out, f"{subject_id}_{c}.npz"))
               for c in PARADIGMS[paradigm]["contrasts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paradigm", nargs="+", choices=sorted(PARADIGMS),
                        default=["PP", "LG"],
                        help="Paradigme(s) a lancer (defaut : PP et LG).")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--missing-only", action="store_true",
                        help="Ne soumet que les sujets sans resultat complet.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    data_dir, results_dir = configure_project_paths(getpass.getuser())
    logger.info("Donnees : %s | resultats : %s", data_dir, results_dir)
    check_files = os.path.isdir(data_dir)
    if not check_files:
        logger.warning("Dossier de donnees introuvable ici : pas de verification "
                       "des fichiers (normal en dry-run hors cluster).")

    # Une liste de (paradigme, sujet, groupe) : un job par element.
    tasks = []
    for paradigm in args.paradigm:
        selected = select_subjects(args.groups, args.subjects)
        if args.missing_only:
            selected = {s: g for s, g in selected.items()
                        if not is_done(results_dir, paradigm, s)}
        if check_files:
            no_data = [s for s, g in selected.items()
                       if not has_data(paradigm, s, g, data_dir)]
            if no_data:
                logger.warning("%s : %d sujet(s) sans fichier, non soumis : %s",
                               paradigm, len(no_data), ", ".join(no_data))
            selected = {s: g for s, g in selected.items() if s not in no_data}
        logger.info("%s : %d sujets a soumettre (%d permutations) : %s",
                    paradigm, len(selected), args.n_permutations,
                    ", ".join(f"{s}({g})" for s, g in selected.items()))
        tasks += [(paradigm, s, g) for s, g in selected.items()]

    logger.info("Total : %d jobs.", len(tasks))
    if args.dry_run or not tasks:
        return
    if not check_files:
        sys.exit(f"Dossier de donnees introuvable : {data_dir}")

    import submitit
    log_dir = os.path.join(PROJECT_ROOT, "submitit", "logs_submitit_jobs",
                           f"perm_{'_'.join(args.paradigm)}_{datetime.now():%Y%m%d_%H%M%S}")
    executor = submitit.AutoExecutor(folder=log_dir)
    account = os.environ.get("BAKING_EEG_SLURM_ACCOUNT", "tom.balay")
    executor.update_parameters(
        timeout_min=int(os.environ.get("BAKING_EEG_TIMEOUT_MIN", 24 * 60)),
        slurm_setup=SETUP, slurm_partition=PARTITION, slurm_mem=MEMORY,
        slurm_cpus_per_task=N_CPUS,
        slurm_job_name="perm_BakingEEG",
        slurm_additional_parameters={"account": account} if account else {},
    )
    # Soumission groupee : tous les jobs entrent dans la file en une fois et
    # Slurm les execute en parallele selon les noeuds libres.
    with executor.batch():
        jobs = [(paradigm, sid, executor.submit(
                    run_on_worker, paradigm, sid, group, data_dir, results_dir,
                    args.n_permutations, N_CPUS))
                for paradigm, sid, group in tasks]
    logger.info("%d jobs soumis (ids %s ... %s). Logs : %s", len(jobs),
                jobs[0][2].job_id, jobs[-1][2].job_id, log_dir)

    failed = []
    for paradigm, sid, job in jobs:
        try:
            job.result()
        except Exception as exc:  # noqa: BLE001 - on veut la liste complete
            logger.error("%s %s (job %s) a echoue : %s", paradigm, sid, job.job_id, exc)
            failed.append(f"{paradigm}:{sid}")
    logger.info("Termines : %d/%d. Echecs : %s", len(jobs) - len(failed),
                len(jobs), failed or "aucun")
    if failed:
        logger.info("Relance des seuls manquants : python "
                    "submitit/submit_permutation_tests.py --missing-only")


if __name__ == "__main__":
    main()
