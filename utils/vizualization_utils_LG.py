#tom
import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from matplotlib.patches import Rectangle


logger_viz_lg = logging.getLogger(__name__)



FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 11
DPI_VALUE = 150

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
except NameError:
    PROJECT_ROOT = os.path.abspath('.')
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    logger_viz_lg.info(f"__file__ not defined. Using project root: {PROJECT_ROOT}")


try:

    from utils import stats_utils as bEEG_stats
except ImportError:
    logger_viz_lg.warning(
        "Could not import 'stats_utils'. Statistical tests will be disabled.")
    bEEG_stats = None


# ==============================================================================
# FONCTION PRINCIPALE REFACTORISÉE
# ==============================================================================

def create_subject_decoding_dashboard_plots_lg(
    main_epochs_time_points,
    classifier_name_for_title,
    subject_identifier,
    group_identifier,
    output_directory_path,
    results_data,
    chance_level=0.5,
    protocol_type="LG"
):
    """
    Crée un tableau de bord complet pour les résultats de décodage d'un sujet.

    Cette version refactorisée utilise un dictionnaire `results_data` pour
    passer toutes les données de résultats, ce qui simplifie grandement la
    signature de la fonction et améliore la maintenabilité.

    Args:
        main_epochs_time_points (array): Vecteur de temps pour les époques.
        classifier_name_for_title (str): Nom du classifieur pour les titres.
        subject_identifier (str): Identifiant du sujet.
        group_identifier (str): Identifiant du groupe.
        output_directory_path (str): Chemin pour sauvegarder les graphiques.
        results_data (dict): Dictionnaire contenant tous les résultats de décodage.
        chance_level (float): Niveau de chance pour les scores (ex: 0.5 pour AUC).
        protocol_type (str): Type de protocole (non utilisé actuellement mais conservé).
    """
    logger_viz_lg.info(
        f"Début de la création des graphiques pour le sujet {subject_identifier}...")

    # Configuration du style des graphiques
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_directory_path, exist_ok=True)

    # --- Configuration des comparaisons à tracer ---
    # C'est ici que nous définissons chaque graphique à générer.
    # Pour ajouter une nouvelle comparaison, il suffit d'ajouter une entrée à cette liste.
    comparison_configs = [
        {
            "name": "LD_vs_LS",
            "title": "Local Deviant vs Local Standard",
            "include_tgm": True,
            "data_keys": {
                "mean_scores": "lg_ls_ld_scores_1d_mean",
                "all_folds_scores": "lg_ls_ld_scores_1d_all_folds",
                "original_labels": "lg_ls_ld_original_labels",
                "tgm_scores": "lg_ls_ld_tgm_mean",
                "tgm_fdr_data": "lg_ls_ld_tgm_fdr",
            }
        },
        {
            "name": "GD_vs_GS",
            "title": "Global Deviant vs Global Standard",
            "include_tgm": True,
            "data_keys": {
                "mean_scores": "lg_gs_gd_scores_1d_mean",
                "all_folds_scores": "lg_gs_gd_scores_1d_all_folds",
                "original_labels": "lg_gs_gd_original_labels",
                "tgm_scores": "lg_gs_gd_tgm_mean",
                "tgm_fdr_data": "lg_gs_gd_tgm_fdr",
            }
        },
        {
            "name": "LSGS_vs_LSGD",
            "title": "Local Std: Global Std vs Global Dev",
            "include_tgm": False,
            "data_keys": {
                "mean_scores": "lg_lsgs_vs_lsgd_scores_1d_mean",
                # Pas de folds pour les comparaisons spécifiques, donc on ne les met pas
            }
        },
        {
            "name": "LDGS_vs_LDGD",
            "title": "Local Dev: Global Std vs Global Dev",
            "include_tgm": False,
            "data_keys": {
                "mean_scores": "lg_ldgs_vs_ldgd_scores_1d_mean",
            }
        },
        {
            "name": "LSGS_vs_LDGS",
            "title": "Global Std: Local Std vs Local Dev",
            "include_tgm": False,
            "data_keys": {
                "mean_scores": "lg_lsgs_vs_ldgs_scores_1d_mean",
            }
        },
        {
            "name": "LSGD_vs_LDGD",
            "title": "Global Dev: Local Std vs Local Dev",
            "include_tgm": False,
            "data_keys": {
                "mean_scores": "lg_lsgd_vs_ldgd_scores_1d_mean",
            }
        },
    ]

    # --- Boucle pour générer chaque graphique de comparaison ---
    for config in comparison_configs:
        try:
            _create_lg_comparison_page(
                times=main_epochs_time_points,
                comparison_config=config,
                results_data=results_data,
                subject_id=subject_identifier,
                group_id=group_identifier,
                output_dir=output_directory_path,
                classifier_name=classifier_name_for_title,
                chance_level=chance_level,
            )
        except Exception as e:
            logger_viz_lg.error(
                f"Erreur lors de la création de la page de comparaison '{config['name']}' "
                f"pour le sujet {subject_identifier}: {e}", exc_info=True)

    logger_viz_lg.info(
        f"Graphiques créés avec succès pour le sujet {subject_identifier}.")


# ==============================================================================
# FONCTIONS AUXILIAIRES (HELPERS)
# ==============================================================================

def _create_lg_comparison_page(
    times,
    comparison_config,
    results_data,
    subject_id,
    group_id,
    output_dir,
    classifier_name,
    chance_level,
):
    """
    Crée une page (un fichier image) contenant les graphiques pour une comparaison spécifique.
    """
    comparison_name = comparison_config["name"]
    comparison_title = comparison_config["title"]
    data_keys = comparison_config["data_keys"]

    # --- Préparation des données pour cette comparaison ---
    # Récupère les données depuis le dictionnaire global `results_data`
    # en utilisant les clés définies dans la configuration. `.get(key)` renvoie
    # `None` si la clé n'existe pas, évitant ainsi les erreurs.
    plot_data = {key: results_data.get(value) for key, value in data_keys.items()}
    
    # Si des données de folds sont disponibles, on peut calculer les stats
    if plot_data.get("all_folds_scores") is not None:
        stats_results = _apply_statistical_tests_to_scores(
            plot_data["all_folds_scores"], times
        )
        plot_data.update(stats_results) # Ajoute 'fdr_data' et 'cluster_data'

    # --- Création de la figure ---
    # Un seul graphique si TGM n'est pas inclus, deux sinon.
    if comparison_config["include_tgm"]:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=DPI_VALUE)
        ax_auc, ax_tgm = axes
    else:
        fig, ax_auc = plt.subplots(1, 1, figsize=(10, 7), dpi=DPI_VALUE)
        ax_tgm = None
    
    fig.suptitle(
        f"{comparison_title}\nSujet: {subject_id} | Groupe: {group_id} | Classifieur: {classifier_name}",
        fontsize=FONT_SIZE_TITLE + 2
    )

    # --- Graphique 1: Courbe temporelle de l'AUC ---
    _plot_lg_temporal_auc(
        ax=ax_auc,
        times=times,
        mean_scores=plot_data.get("mean_scores"),
        all_folds_scores=plot_data.get("all_folds_scores"),
        fdr_data=plot_data.get("fdr_data"),
        cluster_data=plot_data.get("cluster_data"),
        original_labels=plot_data.get("original_labels"),
        title="Décodage Temporel (AUC)",
        chance_level=chance_level
    )

    # --- Graphique 2: Matrice de Généralisation Temporelle (TGM) ---
    if ax_tgm and plot_data.get("tgm_scores") is not None:
        _plot_lg_tgm(
            ax=ax_tgm,
            times=times,
            tgm_scores=plot_data.get("tgm_scores"),
            tgm_fdr_data=plot_data.get("tgm_fdr_data"),
            title="Généralisation Temporelle (TGM)",
            chance_level=chance_level
        )
    elif ax_tgm:
        # Afficher un message si TGM était attendu mais les données sont absentes
        ax_tgm.text(0.5, 0.5, "Données TGM non disponibles",
                    ha='center', va='center', fontsize=FONT_SIZE_LABEL, alpha=0.5)
        ax_tgm.set_xticks([])
        ax_tgm.set_yticks([])

    # --- Sauvegarde de la figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste pour le suptitle
    filename = f"lg_{comparison_name}_{subject_id}_{group_id}_{classifier_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    logger_viz_lg.info(f"Graphique de comparaison '{comparison_name}' sauvegardé: {filepath}")


def _plot_lg_temporal_auc(ax, times, mean_scores, all_folds_scores, fdr_data,
                          cluster_data, original_labels, title, chance_level):
    """Trace la courbe AUC temporelle avec les statistiques et les détails."""
    if mean_scores is None:
        ax.text(0.5, 0.5, "Données AUC non disponibles",
                ha='center', va='center', fontsize=FONT_SIZE_LABEL, alpha=0.5)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        return

    # --- Tracé des folds individuels (si disponibles) ---
    if all_folds_scores is not None and all_folds_scores.ndim == 2:
        n_folds = all_folds_scores.shape[0]
        ax.plot(times, all_folds_scores.T, color='gray', alpha=0.2, linewidth=0.8)
        # Ajoute une seule ligne à la légende pour tous les folds
        ax.plot([], [], color='gray', alpha=0.4, linewidth=1, label=f'Folds individuels (N={n_folds})')

    # --- Tracé de la courbe moyenne ---
    ax.plot(times, mean_scores, 'b-', linewidth=2.5, label='Moyenne des folds')

    # --- Tracé de l'intervalle de confiance (si folds disponibles) ---
    if all_folds_scores is not None and all_folds_scores.ndim == 2:
        sem = scipy.stats.sem(all_folds_scores, axis=0, nan_policy='omit')
        ci_95 = 1.96 * sem
        ax.fill_between(times, mean_scores - ci_95, mean_scores + ci_95,
                        color='blue', alpha=0.2, label='IC 95%')

    # --- Lignes de référence ---
    ax.axhline(chance_level, color='r', linestyle='--', linewidth=1.5, label=f'Chance ({chance_level})')
    ax.axvline(0, color='k', linestyle=':', linewidth=1.5, label='Stimulus Onset')
    
    y_min, _ = ax.get_ylim()
    y_sig_pos = y_min + 0.01

    # --- Marqueurs de significativité FDR ---
    if fdr_data and fdr_data.get('mask') is not None:
        if np.any(fdr_data['mask']):
            ax.fill_between(times, y_sig_pos, y_sig_pos + 0.01, where=fdr_data['mask'],
                            color='green', step='mid', label='FDR p<0.05')
        else:
            ax.plot([], [], color='green', label='FDR (non-significatif)')
    else:
        ax.plot([], [], color='gray', linestyle=':', label='FDR (N/A)')

    # --- Marqueurs de significativité Cluster ---
    if cluster_data and cluster_data.get('mask') is not None:
        y_sig_pos += 0.015 # Position un peu au-dessus du FDR
        if np.any(cluster_data['mask']):
            ax.fill_between(times, y_sig_pos, y_sig_pos + 0.01, where=cluster_data['mask'],
                            color='orange', step='mid', label='Cluster p<0.05')
        else:
            ax.plot([], [], color='orange', label='Cluster (non-significatif)')
    else:
        ax.plot([], [], color='gray', linestyle='-.', label='Cluster (N/A)')
        
    # --- Annotation du pic AUC ---
    peak_auc = np.max(mean_scores)
    peak_time = times[np.argmax(mean_scores)]
    ax.text(0.98, 0.98, f'Pic AUC: {peak_auc:.3f}\nà {peak_time:.3f}s',
            transform=ax.transAxes, va='top', ha='right', fontsize=FONT_SIZE_LEGEND - 1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # --- Mise en forme ---
    plot_title = f"{title}\n{_get_epochs_info_for_labels_lg(original_labels)}"
    ax.set_title(plot_title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel('Temps (s)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Score AUC', fontsize=FONT_SIZE_LABEL)
    ax.legend(loc='lower right', fontsize=FONT_SIZE_LEGEND)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(times[0], times[-1])


def _plot_lg_tgm(ax, times, tgm_scores, tgm_fdr_data, title, chance_level):
    """Trace la matrice de généralisation temporelle (TGM)."""
    if tgm_scores is None or tgm_scores.ndim != 2:
        ax.text(0.5, 0.5, "Données TGM non valides",
                ha='center', va='center', fontsize=FONT_SIZE_LABEL, alpha=0.5)
        ax.set_title(title, fontsize=FONT_SIZE_TITLE)
        return

    # --- Heatmap de la TGM ---
    vmax = np.max([abs(tgm_scores.max() - chance_level), abs(tgm_scores.min() - chance_level)])
    vmin, vmax = chance_level - vmax, chance_level + vmax
    
    im = ax.imshow(tgm_scores, interpolation='lanczos', cmap='RdBu_r',
                   extent=[times[0], times[-1], times[0], times[-1]],
                   origin='lower', aspect='auto', vmin=vmin, vmax=vmax)

    # --- Contours de significativité (si disponibles) ---
    if tgm_fdr_data and tgm_fdr_data.get('mask') is not None:
        mask = tgm_fdr_data['mask']
        if np.any(mask) and mask.shape == tgm_scores.shape:
            ax.contour(times, times, mask, levels=[0.5], colors='black',
                       linewidths=1, linestyles='solid')

    # --- Mise en forme ---
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel('Temps de Test (s)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Temps d'Entraînement (s)", fontsize=FONT_SIZE_LABEL)
    ax.axvline(0, color='k', linestyle=':', alpha=0.8)
    ax.axhline(0, color='k', linestyle=':', alpha=0.8)

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score AUC', fontsize=FONT_SIZE_LABEL)
    cbar.ax.axhline(chance_level, color='k', linestyle='--', lw=1)


def _apply_statistical_tests_to_scores(all_folds_scores, times, chance_level=0.5, n_permutations=1024):
    """Applique les tests statistiques (FDR, Cluster) sur les scores des folds."""
    results = {'fdr_data': None, 'cluster_data': None}
    if not bEEG_stats or all_folds_scores is None or all_folds_scores.ndim != 2:
        return results

    try:
        # --- Test FDR ---
        _, fdr_mask, fdr_p, fdr_info = bEEG_stats.perform_pointwise_fdr_correction_on_scores(
            all_folds_scores, chance_level=chance_level, alternative_hypothesis="greater")
        results['fdr_data'] = {'mask': fdr_mask, 'p_values': fdr_p, 'info': fdr_info}

        # --- Test par permutation de clusters ---
        _, clusters, cluster_p, _ = bEEG_stats.perform_cluster_permutation_test(
            all_folds_scores, chance_level=chance_level, n_permutations=n_permutations,
            alternative_hypothesis="greater")
        
        cluster_mask = np.zeros(len(times), dtype=bool)
        if clusters:
            significant_clusters = [c for c, p in zip(clusters, cluster_p) if p < 0.05]
            if significant_clusters:
                for c in significant_clusters:
                    cluster_mask[c] = True
        results['cluster_data'] = {'mask': cluster_mask, 'p_values': cluster_p}
        
    except Exception as e:
        logger_viz_lg.error(f"Erreur lors de l'application des tests statistiques: {e}", exc_info=True)

    return results


def _get_epochs_info_for_labels_lg(original_labels):
    """Formate une chaîne d'information sur le nombre et la répartition des époques."""
    if original_labels is None or len(original_labels) == 0:
        return "N époques: N/A"
    
    try:
        total_epochs = len(original_labels)
        unique_labels, counts = np.unique(original_labels, return_counts=True)
        
        if len(unique_labels) > 1:
            info_parts = []
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_epochs) * 100
                info_parts.append(f"Classe {int(label)}: {count} ({percentage:.1f}%)")
            return f"N époques: {total_epochs} [{', '.join(info_parts)}]"
        else:
            return f"N époques: {total_epochs} (une seule classe)"
            
    except Exception:
        return f"N époques: {len(original_labels)}"