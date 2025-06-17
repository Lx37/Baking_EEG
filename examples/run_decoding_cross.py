

# --- Cross-Subject Decoding ---
def execute_group_cross_subject_decoding_analysis(
    subject_ids_for_cs_set,
    cross_subject_set_identifier,
    decoding_protocol_identifier="PP_AP_General_CS",
    save_results_flag=True,
    enable_verbose_logging=True,
    generate_plots_flag=True,
    base_input_data_path=None,
    base_output_results_path=None,
    n_jobs_for_fold_temporal_decoding=4,
    compute_group_stats_for_cs_flag=True,
    classifier_type_for_cs_runs=CLASSIFIER_MODEL_TYPE,
    n_perms_for_cs_group_cluster_test=N_PERMUTATIONS_GROUP_LEVEL,
    cs_group_cluster_test_threshold_method=GROUP_LEVEL_STAT_THRESHOLD_TYPE,
    cs_group_cluster_test_t_thresh_value=T_THRESHOLD_FOR_GROUP_STAT_CLUSTERING,
    use_grid_search_for_cs=False,
    use_csp_for_cs_temporal=False,
    # ANOVA FS for CS is implicitly handled by _4_decoding._build_standard_classifier_pipeline
    # based on use_csp_for_cs_temporal (if CSP is false, ANOVA might be true)
    param_grid_config_for_cs=None,
    cv_folds_for_gs_cs=3,
    fixed_params_for_cs=None,
):
    """Executes cross-subject decoding for a given set of subjects."""
    # Parameter Validation
    if not isinstance(subject_ids_for_cs_set, list) or len(subject_ids_for_cs_set) < 2:
        logger_run_decoding.error(
            "subject_ids_for_cs_set must be a list with at least 2 subjects."
        )
        return np.nan, np.nan, None  # Mean AUC, Std AUC, Other results (None)
    if not isinstance(cross_subject_set_identifier, str) or not cross_subject_set_identifier:
        logger_run_decoding.error(
            "cross_subject_set_identifier must be a non-empty string.")
        return np.nan, np.nan, None

    total_cs_analysis_start_time = time.time()
    if not base_input_data_path or not base_output_results_path:
        current_user = getuser()
        cfg_input, cfg_output = configure_project_paths(current_user)
        base_input_data_path = base_input_data_path or cfg_input
        base_output_results_path = base_output_results_path or cfg_output

    logger_run_decoding.info(
        "Starting CROSS-SUBJECT decoding: %s (%d subjects, Clf: %s, GS: %s, CSP (temp): %s)",
        cross_subject_set_identifier, len(subject_ids_for_cs_set),
        classifier_type_for_cs_runs, use_grid_search_for_cs, use_csp_for_cs_temporal
    )

    subject_data_cache = {}
    valid_subject_ids_for_cs = []
    # Pre-load data for all subjects in the set
    for subj_id_cs in subject_ids_for_cs_set:
        try:
            # Determine actual group for loading based on ALL_SUBJECT_GROUPS mapping
            actual_group_for_loading = cross_subject_set_identifier  # Default
            for grp_name, s_list in ALL_SUBJECT_GROUPS.items():
                if subj_id_cs in s_list:
                    actual_group_for_loading = grp_name
                    break
            epochs_obj, data_dict = load_epochs_data_for_decoding(
                subj_id_cs, actual_group_for_loading, base_input_data_path,
                CONFIG_LOAD_MAIN_DECODING, verbose_logging=False  # Less verbose for pre-loading
            )
            if epochs_obj is None:
                logger_run_decoding.warning(
                    "Skipping subject %s from CS set '%s' (data load error).",
                    subj_id_cs, cross_subject_set_identifier
                )
                continue
            xpp_cs = data_dict.get("XPP_ALL")
            xap_cs = data_dict.get("XAP_ALL")
            if xpp_cs is None or xap_cs is None or xpp_cs.size == 0 or xap_cs.size == 0:
                logger_run_decoding.warning(
                    "Skipping subject %s from CS set '%s' (empty XPP/XAP data).",
                    subj_id_cs, cross_subject_set_identifier
                )
                continue
            subject_data_cache[subj_id_cs] = {
                "epochs_obj": epochs_obj, "XAP_data": xap_cs, "XPP_data": xpp_cs
            }
            valid_subject_ids_for_cs.append(subj_id_cs)
        except Exception as e_load:
            logger_run_decoding.error(
                "Failed to load data for subject %s for CS set '%s': %s. Skipping.",
                subj_id_cs, cross_subject_set_identifier, e_load, exc_info=True
            )

    if len(valid_subject_ids_for_cs) < 2:
        logger_run_decoding.error(
            "Not enough valid subjects (%d) for CS analysis in set '%s'. Aborting CS.",
            len(valid_subject_ids_for_cs), cross_subject_set_identifier
        )
        return np.nan, np.nan, None

    cs_results_dir_suffix = (f"{cross_subject_set_identifier}_"
                             f"{classifier_type_for_cs_runs}_"
                             f"GS{use_grid_search_for_cs}_"
                             f"CSP{use_csp_for_cs_temporal}")
    cs_results_main_dir = None
    if save_results_flag or generate_plots_flag:
        cs_results_main_dir = setup_analysis_results_directory(
            base_output_results_path, "cross_subject_analysis_results", cs_results_dir_suffix
        )

    loso_cv = LeaveOneGroupOut()
    # Groups for LOSO are just indices of valid subjects
    subject_indices_loso = np.arange(len(valid_subject_ids_for_cs))
    valid_subject_ids_arr = np.array(valid_subject_ids_for_cs)

    cs_fold_results = {
        "fold_global_auc_scores": {}, "fold_global_metrics_maps": {},
        "fold_temporal_scores_1d_list": [], "fold_time_points_list": [],
        "fold_test_subject_ids_processed": []
    }

    # Prepare classifier specific params for run_cross_subject_decoding_for_fold
    current_fixed_params_for_cs_clf = None
    current_param_grid_for_cs_clf = None
    if use_grid_search_for_cs:
        if param_grid_config_for_cs and classifier_type_for_cs_runs in param_grid_config_for_cs:
            current_param_grid_for_cs_clf = param_grid_config_for_cs  # Pass full grid
        # else: _4_decoding will use its defaults if any
    else:  # Not using GridSearch
        if fixed_params_for_cs and classifier_type_for_cs_runs in fixed_params_for_cs:
            current_fixed_params_for_cs_clf = fixed_params_for_cs[classifier_type_for_cs_runs]
        # else: _4_decoding will use its defaults if any

    for fold_num, (train_subj_indices, test_subj_idx_tuple) in enumerate(
            loso_cv.split(X=np.zeros(len(valid_subject_ids_for_cs)), groups=subject_indices_loso), 1):
        test_subj_id = valid_subject_ids_arr[test_subj_idx_tuple[0]]
        train_subj_ids = valid_subject_ids_arr[train_subj_indices]

        logger_run_decoding.info(
            "\n--- CS Fold %d/%d (Set: '%s') --- Test Subject: %s ---",
            fold_num, len(
                valid_subject_ids_for_cs), cross_subject_set_identifier, test_subj_id
        )
        x_train_fold_list, y_train_fold_list_orig = [], []
        for tr_subj_id in train_subj_ids:
            if tr_subj_id in subject_data_cache:
                tr_data = subject_data_cache[tr_subj_id]
                if tr_data["XAP_data"].size > 0 and tr_data["XPP_data"].size > 0:
                    x_train_fold_list.append(
                        np.concatenate(
                            [tr_data["XAP_data"], tr_data["XPP_data"]])
                    )
                    y_train_fold_list_orig.append(np.concatenate([
                        np.zeros(tr_data["XAP_data"].shape[0]),
                        np.ones(tr_data["XPP_data"].shape[0])
                    ]))

        if not x_train_fold_list:
            logger_run_decoding.warning(
                "CS Fold %d (Test: %s): No valid training data found. Skipping fold.",
                fold_num, test_subj_id
            )
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue

        x_train_fold = np.concatenate(x_train_fold_list)
        y_train_fold_orig = np.concatenate(y_train_fold_list_orig).astype(int)

        test_data_cache_entry = subject_data_cache[test_subj_id]
        if test_data_cache_entry["XAP_data"].size == 0 or \
           test_data_cache_entry["XPP_data"].size == 0:
            logger_run_decoding.warning(
                "CS Fold %d: Test subject %s has empty XAP/XPP data. Skipping fold.",
                fold_num, test_subj_id
            )
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue

        x_test_fold = np.concatenate([
            test_data_cache_entry["XAP_data"], test_data_cache_entry["XPP_data"]
        ])
        y_test_fold_orig = np.concatenate([
            np.zeros(test_data_cache_entry["XAP_data"].shape[0]),
            np.ones(test_data_cache_entry["XPP_data"].shape[0])
        ]).astype(int)

        if len(np.unique(y_train_fold_orig)) < 2 or len(np.unique(y_test_fold_orig)) < 2:
            logger_run_decoding.warning(
                "CS Fold %d (Test: %s): Train or test set has fewer than 2 classes. "
                "Train classes: %s, Test classes: %s. Skipping fold.",
                fold_num, test_subj_id, np.unique(
                    y_train_fold_orig), np.unique(y_test_fold_orig)
            )
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
            continue

        try:
            fold_auc, fold_metrics, _, fold_probas, fold_labels, fold_scores_1d = \
                run_cross_subject_decoding_for_fold(
                    training_epochs_data=x_train_fold,
                    training_original_labels=y_train_fold_orig,
                    testing_epochs_data=x_test_fold,
                    testing_original_labels=y_test_fold_orig,
                    testing_subject_identifier=test_subj_id,
                    group_identifier=cross_subject_set_identifier,  # Set ID
                    decoding_protocol_identifier=decoding_protocol_identifier,
                    classifier_model_type_cs=classifier_type_for_cs_runs,
                    use_grid_search_cs=use_grid_search_for_cs,
                    use_csp_in_pipeline_cs=use_csp_for_cs_temporal,
                    # ANOVA implicitly handled by _4_decoding based on CSP flag
                    param_grid_config_cs=current_param_grid_for_cs_clf,
                    cv_folds_for_gridsearch_cs=cv_folds_for_gs_cs,
                    fixed_classifier_params_cs=current_fixed_params_for_cs_clf,
                    n_jobs_for_temporal_decoding=n_jobs_for_fold_temporal_decoding
                )

            cs_fold_results["fold_global_auc_scores"][test_subj_id] = fold_auc
            cs_fold_results["fold_global_metrics_maps"][test_subj_id] = fold_metrics or {
            }
            if pd.notna(fold_auc):
                cs_fold_results["fold_test_subject_ids_processed"].append(
                    test_subj_id)
                if fold_scores_1d is not None and not np.all(np.isnan(fold_scores_1d)):
                    cs_fold_results["fold_temporal_scores_1d_list"].append(
                        fold_scores_1d)
                    cs_fold_results["fold_time_points_list"].append(
                        test_data_cache_entry["epochs_obj"].times.copy()
                    )
                if generate_plots_flag and test_data_cache_entry["epochs_obj"] is not None and \
                   cs_results_main_dir:
                    fold_plot_dir = os.path.join(
                        cs_results_main_dir, "dashboards_cs_folds", f"test_subject_{test_subj_id}"
                    )
                    os.makedirs(fold_plot_dir, exist_ok=True)
                    create_subject_decoding_dashboard_plots(
                        main_epochs_time_points=test_data_cache_entry["epochs_obj"].times,
                        main_original_labels_array=y_test_fold_orig,
                        main_predicted_probabilities_global=fold_probas,
                        main_predicted_labels_global=fold_labels,
                        main_cross_validation_global_scores=(np.array([fold_auc]) if pd.notna(fold_auc)
                                                             else np.array([np.nan])),
                        main_temporal_scores_1d_all_folds=(fold_scores_1d[np.newaxis, :]
                                                           if fold_scores_1d is not None else None),
                        main_mean_temporal_decoding_scores_1d=fold_scores_1d,
                        # No intra-fold stats for CS fold plots
                        main_temporal_1d_fdr_sig_data=None,
                        main_temporal_1d_cluster_sig_data=None,
                        main_mean_temporal_generalization_matrix_scores=None,  # No TGM in CS fold
                        main_tgm_fdr_sig_data=None,
                        classifier_name_for_title=classifier_type_for_cs_runs,
                        subject_identifier=test_subj_id,
                        group_identifier=f"{cross_subject_set_identifier}_CS_TestFold",
                        output_directory_path=fold_plot_dir,
                        chance_level_auc=CHANCE_LEVEL_AUC,
                        specific_ap_decoding_results=None,  # No specific tasks in CS
                        mean_of_specific_scores_1d=None,
                        sem_of_specific_scores_1d=None,
                        mean_specific_fdr_sig_data=None,
                        mean_specific_cluster_sig_data=None,
                        ap_vs_ap_decoding_results=None  # No AP vs AP in CS
                    )
        except Exception as e_cs_fold:
            logger_run_decoding.error(
                "Error during CS Fold %d (Test: %s): %s",
                fold_num, test_subj_id, e_cs_fold, exc_info=True
            )
            cs_fold_results["fold_global_auc_scores"][test_subj_id] = np.nan
