import os
import logging
from datetime import datetime
import sys
import argparse
import time

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)
logname = os.path.join(log_dir, datetime.now().strftime('log_%Y-%m-%d.log'))
logging.basicConfig(
    filename=logname,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

import pandas as pd
import mne
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from getpass import getuser
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Baking_EEG'))
from _4_decoding import (
    decode_window,
    plot_decoding_results_dashboard,
    plot_group_results,
    decode_cross_subject_fold
)

SUBJECTS = {
    'controls': ['LAB1', 'LAG6', 'LAT3', 'LBM4', 'LCM2',
                 'LPO5', 'TJL3', 'TJR7', 'TLP8', 'TPC2',
                 'TPLV4', 'TTDV5', 'TYS6'],
    'del': ['TpAB19', 'TpAG51', 'TpAK24', 'TpAK27', 'TpCB15',
            'TpCF1', 'TpDRL3', 'TpJB25', 'TpJB26', 'TpJC5',
            'TpJLR17', 'TpJPS55', 'TpKT33', 'TpLA28', 'TpMB45',
            'TpMM4', 'TpMN42', 'TpPC21', 'TpPM14', 'TpPM31',
            'TpRD38', 'TpSM49'],
    'nodel': ['TpAC23', 'TpAM43', 'TpBD16', 'TpBL47', 'TpCG36',
              'TpFF34', 'TpFL53', 'TpGB8', 'TpGT32', 'TpJA20',
              'TpJPG7', 'TpJPL10', 'TpKS6', 'TpLP11', 'TpMD13',
              'TpMD52', 'TpME22', 'TpPA35', 'TpPI46', 'TpPL48',
              'TpRB50', 'TpRK39', 'TpSD30', 'TpYB41']
}

def configure_paths(user):
    """Configure les chemins en fonction de l'utilisateur."""
    if user == 'tkz':
        base_path = '/home/tkz/Projets/0_FPerrin_FFerre_2024_Baking_EEG_CAP/Baking_EEG_data'
    elif user == 'adminlocal':
        base_path = 'C:\\Users\\adminlocal\\Desktop\\ConnectDoc\\EEG_2025_CAP_FPerrin_Vera'
    elif user.lower() == 'tom':
        base_path = '/Users/tom/Desktop/ENSC/2A/PII/Tom/Baking_EEG_data'
    else:
        base_path = os.path.join(os.path.expanduser('~'), 'Baking_EEG_data')
    logger.info(f'Configured paths for user {user}')
    return base_path

def get_patient_info(subject_id, protocol, base_path):
    """Crée un dictionnaire avec les informations du patient."""
    return {
        'ID_patient': subject_id,
        'protocol': protocol,
        'data_dir': base_path,
        'data_save_dir': base_path
    }

def setup_results_dir(patient_info, group_name):
    """Configure le dossier pour sauvegarder les résultats par groupe."""
    results_dir = os.path.join(
        patient_info['data_save_dir'],
        'decoding_results',
        group_name, 
        f"{patient_info['ID_patient']}_{patient_info['protocol']}"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_epochs_for_decoding(subject_id, group, base_path, verbose=True):
    """
    Charge les epochs pour le décodage en tenant compte des fichiers avec ou sans ICA.
    On s'assure aussi que les données EEG sont toujours en 3D (n_epochs, n_channels, n_times).
    """
    start_time = time.time()
    if group == 'controls':
        data_path = os.path.join(base_path, 'PP_CONTROLS_0.5', 'data_epochs')
    elif group in ['del', 'nodel']:
        data_path = os.path.join(base_path, f'PP_PATIENTS_{group.upper()}_0.5', 'data_epochs')
    else:
        raise ValueError(f"Unknown group: {group}")


    # if verbose:
    #     print(f"Fichiers disponibles dans {data_path} :")
    #     print(os.listdir(data_path))

    file_noICA = os.path.join(data_path, subject_id + '_PP_preproc_noICA_PP-epo_ar.fif')
    file_ICA = os.path.join(data_path, subject_id + '_PP_preproc_ICA_PP-epo_ar.fif')

    if os.path.exists(file_noICA):
        fif_fname = file_noICA
    elif os.path.exists(file_ICA):
        fif_fname = file_ICA
    else:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le sujet {subject_id} dans {data_path}")

    if verbose:
        print(f"\nProcessing subject {subject_id} from group {group}")
        print(f"Loading file: {fif_fname}")


    with mne.utils.use_log_level('warning'):
        epochs = mne.read_epochs(fif_fname, proj=False, verbose=verbose, preload=True)

    
    # if "PP" not in epochs.event_id or "AP" not in epochs.event_id:
    #     raise ValueError(f"Les conditions 'PP' ou 'AP' sont manquantes dans les données pour le sujet {subject_id}.")

    XPP = epochs["PP"].pick(picks="eeg").get_data(copy=False)
    XAP = epochs["AP"].pick(picks="eeg").get_data(copy=False)

    if verbose:
        print('XPP.shape : ', XPP.shape)
        print('XAP.shape : ', XAP.shape)
        print("Number of epochs:", len(epochs))

    return epochs, XPP, XAP

def run_decoding_on_subject(subject_id, protocol='PP',
                            save=True, verbose=True, plot=True, group=None, base_path=None):
    """Fonction pour exécuter le décodage sur un seul sujet (décodage intra-sujet)."""
    total_start = time.time()
    try:
        if base_path is None:
            base_path = configure_paths(getuser())
        patient_info = get_patient_info(subject_id, protocol, base_path)
        if verbose:
            logger.info(f"Decoding for subject {subject_id} (group: {group}, protocol: {protocol})")

        epochs, XPP, XAP = load_epochs_for_decoding(subject_id, group, base_path, verbose)

    
        X = np.concatenate([XAP, XPP], axis=0)
        y = np.concatenate([np.zeros(XAP.shape[0]), np.ones(XPP.shape[0])])
        assert X.shape[0] == y.shape[0], f"X.shape: {X.shape}, y.shape: {y.shape}"

        if verbose:
            print('X.shape : ', X.shape)
            print('y.shape : ', y.shape)

        clf = make_pipeline(
            StandardScaler(),
            #SelectPercentile(f_classif, percentile=15),
            SVC(kernel='linear', probability=True, class_weight='balanced')
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        probas, predictions, cv_scores, scores_time, metrics = decode_window(
            X, y, clf=clf, cv=cv, n_jobs='auto'
        )

        if verbose:
            print(f"Total time elapsed: {time.time() - total_start:.2f} sec")
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean score: {np.mean(cv_scores):.3f}, Std: {np.std(cv_scores):.3f}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name}: {metric_value:.3f}")

        if save:
            results_dir = setup_results_dir(patient_info, group)
            results = {
                'subject_id': subject_id,
                'group': group,
                'protocol': protocol,
                'probas': probas,
                'predictions': predictions,
                'cv_scores': cv_scores,
                'scores_time': scores_time,
                'metrics': metrics,
            }
            np.save(os.path.join(results_dir, f"decoding_{subject_id}_{group}_results.npy"), results)
            csv_results = {
                'subject_id': subject_id,
                'group': group,
                'protocol': protocol,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
            }
            csv_results.update(metrics)
            df = pd.DataFrame([csv_results])
            csv_path = os.path.join(results_dir, f"decoding_{subject_id}_{group}_results.csv")
            df.to_csv(csv_path, index=False)
            if verbose:
                logger.info(f"Results saved in {results_dir}")

        if plot:
            plot_decoding_results_dashboard(
                epochs, X, y, probas, predictions, cv_scores, np.array(scores_time),
                subject_id, group, results_dir if save else None
            )

        return np.mean(cv_scores)

    except Exception as e:
        logger.error(f"Error decoding subject {subject_id}: {e}", exc_info=True)
        print(f"Error decoding subject {subject_id}: {e}")
        return None

def run_on_group(subjects, group_name, protocol='PP',
                 save=True, verbose=True, plot=True, base_path=None):
    """Décodage sur un groupe de sujets (décodage intra-sujet) pour un même groupe."""
    total_start = time.time()
    if base_path is None:
        base_path = configure_paths(getuser())

    if verbose:
        logger.info(f"Decoding for group {group_name} ({len(subjects)} subjects)")
        print(f"Decoding for group {group_name} ({len(subjects)} subjects)")

    results = {}
    for i, subject_id in enumerate(subjects, 1):
        if verbose:
            print(f"Subject {i}/{len(subjects)}: {subject_id}")
        score = run_decoding_on_subject(subject_id, protocol,
                                        save, verbose, plot, group_name, base_path)
        results[subject_id] = score

    valid_scores = [score for score in results.values() if score is not None]
    if valid_scores:
        mean_group_score = np.mean(valid_scores)
        std_group_score = np.std(valid_scores)
        if verbose:
            logger.info(f"Group {group_name} results:")
            logger.info(f"Mean score: {mean_group_score:.3f} ± {std_group_score:.3f}")
            print(f"Group {group_name} results:")
            print(f"Mean score: {mean_group_score:.3f} ± {std_group_score:.3f}")
            print(f"Total time : {time.time() - total_start:.2f} sec")
        if save:
            group_results_dir = os.path.join(base_path, 'decoding_results', f'group_{group_name}')
            os.makedirs(group_results_dir, exist_ok=True)
            group_results = {
                'group': group_name,
                'protocol': protocol,
                'subject_scores': results,
                'mean_score': mean_group_score,
                'std_score': std_group_score
            }
            np.save(os.path.join(group_results_dir, f"group_{group_name}_decoding_results.npy"), group_results)
            df = pd.DataFrame([
                {'subject_id': subject_id, 'score': score, 'group': group_name}
                for subject_id, score in results.items()
            ])
            df.to_csv(os.path.join(group_results_dir, f'group_{group_name}_scores.csv'), index=False)
            if verbose:
                logger.info(f"Group results saved in {group_results_dir}")
        if plot:
            plot_group_results(results, group_name, save_dir=group_results_dir if save else None)

    return results

def run_cross_subject(subjects, group_name, protocol='PP', save=True, verbose=True, plot=True, base_path=None):
    """
    Décodage cross-subject avec Leave-One-Group-Out (LOGO) pour le décodage inter-sujet.
    """
    total_start = time.time()
    if base_path is None:
        base_path = configure_paths(getuser())

    logger.info(f"Début du décodage cross-subject pour le groupe {group_name}")
    print(f"\n=== Décodage cross-subject ({group_name}) ===")

    subjects = np.array(subjects)


    subject_data = {}
    for subject_id in subjects:
        try:
            epochs, XPP, XAP = load_epochs_for_decoding(subject_id, group_name, base_path, verbose)
            subject_data[subject_id] = {
                'epochs': epochs,
                'XAP': XAP,
                'XPP': XPP
            }
            if verbose:
                print(f"{subject_id} chargé | AP: {XAP.shape} PP: {XPP.shape}")
        except Exception as e:
            logger.error(f"Échec chargement {subject_id}: {str(e)}")
            print(f"{subject_id} non chargé: {str(e)}")
            continue

    if not subject_data:
        logger.error("Aucune donnée valide ! abandon.")
        raise ValueError("Aucun sujet valide pour le décodage")

  
    if save:
        results_dir = os.path.join(base_path, 'decoding_results', f'cross_decoding_{group_name}')
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None

    logo = LeaveOneGroupOut()
    scores = {}
    metrics_all = {}
    models = {}

    indices = np.arange(len(subjects))  

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(indices, groups=indices), 1):
        test_subject = subjects[test_idx[0]]
        if verbose:
            print(f"\n--- Fold {fold_idx}/{len(subjects)} - Test: {test_subject} Train : {', '.join(subjects[train_idx])} --- ")

        if test_subject not in subject_data:
            print(f" Données manquantes pour {test_subject} - skip")
            continue

        X_train, y_train = [], []
        for subj in subjects[train_idx]:
            data = subject_data.get(subj)
            if data:
             
                X = np.concatenate([data['XAP'], data['XPP']], axis=0)
                y = np.concatenate([np.zeros(data['XAP'].shape[0]), np.ones(data['XPP'].shape[0])])
                y = y.astype(int)  
                X_train.append(X)
                y_train.append(y)

        if not X_train:
            print(" Aucune donnée d'entraînement valide - Skip")
            continue

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0).astype(int) 
        test_data = subject_data[test_subject]
        X_test = np.concatenate([test_data['XAP'], test_data['XPP']], axis=0)
        y_test = np.concatenate([
            np.zeros(test_data['XAP'].shape[0]),
            np.ones(test_data['XPP'].shape[0])
        ]).astype(int)  

        try:
            result = decode_cross_subject_fold(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_subject=test_subject,
                group_name=group_name,
                protocol=protocol,
                base_path=base_path,
                save=False
            )
        except Exception as e:
            logger.error(f"ERREUR fold {fold_idx}: {str(e)}")
            print(f" Erreur lors du décodage: {str(e)}")
            continue

        if result is not None:
            score, metrics, model, probas, predictions, scores_time = result
            scores[test_subject] = score
            metrics_all[test_subject] = metrics
            models[test_subject] = model

            if verbose:
                print(f"Score AUC pour {test_subject}: {score:.3f}")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.3f}")

            if plot:
                plot_decoding_results_dashboard(
                    subject_data[test_subject]['epochs'],
                    X_test, y_test, probas, predictions,
                    [score], scores_time,
                    test_subject, group_name,
                    results_dir if save else None
                )

           
            subject_results_dir = os.path.join(results_dir, f'{test_subject}_{protocol}_cross')
            os.makedirs(subject_results_dir, exist_ok=True)

    if not scores:
        logger.error("Aucun résultat valide obtenu !")
        return None, None, None

    mean_score = np.mean(list(scores.values()))
    std_score = np.std(list(scores.values()))

    if save:
        df = pd.DataFrame({
            'subject': list(scores.keys()),
            'score': list(scores.values()),
            'group': group_name
        })
        df.to_csv(os.path.join(results_dir, 'scores.csv'), index=False)

        pd.DataFrame(metrics_all).T.to_csv(os.path.join(results_dir, 'metrics.csv'))

        for subj, model in models.items():
            joblib.dump(model, os.path.join(results_dir, f'model_{subj}.joblib'))

    if plot and scores:
        try:
            plot_group_results(
                scores,
                group_name,
                save_dir=results_dir if save else None
            )
        except Exception as e:
            logger.error(f"Erreur visualisation: {str(e)}")

    total_time = time.time() - total_start
    logger.info(f"Temps total: {total_time / 60:.1f} min")
    print("\n=== Résultats finaux ===")
    print(f"Moyenne AUC: {mean_score:.3f} ± {std_score:.3f}")
    print(f"Temps total: {total_time / 60:.1f} minutes")

    return mean_score, std_score, models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Module de Décodage EEG')
    parser.add_argument('--mode', type=str, choices=['one', 'group', 'cross'],
                        help='Choisissez le mode d\'exécution : "one" pour un sujet, "group" pour tous les sujets d\'un groupe, "cross" pour cross-subject')
    args = parser.parse_args()

    user = getuser()
    base_path = configure_paths(user)

    if args.mode == 'one':
        print("\nIdentifiants disponibles :")
        print("controls :", SUBJECTS['controls'])
        print("del      :", SUBJECTS['del'])
        print("nodel    :", SUBJECTS['nodel'])
        subject_id = input("\nEntrez l'identifiant du sujet (exemple : TJR7) : ").strip()
        group = input("Entrez le groupe ('controls', 'del' ou 'nodel') [défaut : controls] : ").strip() or "controls"
        run_decoding_on_subject(subject_id, protocol='PP', group=group, base_path=base_path)

    elif args.mode == 'group':
        print("\nIdentifiants disponibles :")
        print("controls :", SUBJECTS['controls'])
        print("del      :", SUBJECTS['del'])
        print("nodel    :", SUBJECTS['nodel'])
        group = input("Entrez le groupe pour ces sujets ('controls', 'del' ou 'nodel') : ").strip()
        if group in SUBJECTS:
            subjects_list = SUBJECTS[group]
            run_on_group(subjects_list, group, protocol='PP', base_path=base_path)
        else:
            print(f"Groupe inconnu : {group}. Veuillez choisir parmi 'controls', 'del' ou 'nodel'.")

    elif args.mode == 'cross':
        print("\nIdentifiants disponibles :")
        print("controls :", SUBJECTS['controls'])
        print("del      :", SUBJECTS['del'])
        print("nodel    :", SUBJECTS['nodel'])
        print("\nEntrez les paramètres pour le cross-subject sous la forme :")
        print("<sujets>,<sujets>,... <groupe>")
        print("Exemple : TpAB19,TpAG51,TpAK27,TpCB15,TpCF1,TpDRL3,TpJB25,TpJB26,TpJLR17,TpJPS55,TpKT33,TpLA28,TpMB45,TpMM4,TpMN42,TpPC21,TpPM14,TpPM31,TpRD38,TpSM49 del")
        cross_input = input("Saisie : ").strip()
        try:
            parts = cross_input.split()
            if len(parts) != 2:
                raise ValueError("Format invalide. Vous devez saisir une liste de sujets et un groupe.")
            subjects_list = [s.strip() for s in parts[0].split(',')]
            group = parts[1]
            run_cross_subject(subjects_list, group, protocol='PP', base_path=base_path)
        except Exception as e:
            print("Erreur dans la saisie. Veuillez respecter le format indiqué.", e)


# if args.load_model:
#     # Code pour charger un modèle existant et l'utiliser pour prédire
#     print(f"Chargement du modèle {args.load_model}")
#     model = load(args.load_model)