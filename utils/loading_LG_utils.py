
import os
import time
import logging
import numpy as np
import mne
from config.config import ALL_SUBJECT_GROUPS
from config.decoding_config import CONFIG_LOAD_MAIN_LG_DECODING, CONFIG_LOAD_LG_COMPARISONS, EVENT_ID_LG
logger_data_loading = logging.getLogger(__name__)


def load_epochs_data_for_lg_decoding(
    subject_identifier,
    group_affiliation,
    base_input_data_path,
    conditions_to_load=None,
    verbose_logging=True,
):
    """Load preprocessed MNE Epochs data for Local-Global protocol analysis.

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation (e.g., 'controls',
                                 'del'). Used for path determination.
        base_input_data_path (str): Root directory for input data.
        conditions_to_load (dict, optional): Specifies conditions to extract.
            Keys: custom names for extracted data arrays.
            Values: MNE event specifiers (e.g., 'LS/GS',
            ['LS/GS', 'LS/GD'], or 'LS/').
            Defaults to `CONFIG_LOAD_MAIN_LG_DECODING` if None.
        verbose_logging (bool): If True, enables detailed logging.

    Returns:
        tuple: (mne.Epochs or None, dict, str)
            - mne.Epochs object if loading successful, else None.
            - Dictionary: keys are `condition_name` from `conditions_to_load`,
              values are NumPy arrays of epoch data
              (n_epochs, n_channels, n_times).
              Empty array if a condition has no data.
            - String: detected protocol identifier (e.g., "LG")
    """
    # Parameter validation
    if not isinstance(subject_identifier, str) or not subject_identifier:
        logger_data_loading.error(
            "Subject identifier must be a non-empty string.")
        return None, {}, "unknown"
    if not isinstance(group_affiliation, str) or not group_affiliation:
        logger_data_loading.error(
            "Group affiliation must be a non-empty string.")
        return None, {}, "unknown"
    if (not isinstance(base_input_data_path, str) or
            not os.path.isdir(base_input_data_path)):
        logger_data_loading.error(
            "Base input data path '%s' is not a valid directory.",
            base_input_data_path
        )
        return None, {}, "unknown", "unknown"
    if (conditions_to_load is not None and
            not isinstance(conditions_to_load, dict)):
        logger_data_loading.error(
            "'conditions_to_load' must be a dict or None. "
            "Received type: %s.", type(conditions_to_load)
        )
        return None, {}, "unknown"

    start_time = time.time()
    group_affiliation_lower = group_affiliation.lower()
    data_root_path = None

    # --- Path determination logic for LG protocol ---
    # Only 3 real LG directories exist: LG_CONTROLS_0.5, LG_PATIENTS_DELIRIUM+_0.5, LG_PATIENTS_DELIRIUM-_0.5
    if group_affiliation_lower == "controls":
        potential_path = os.path.join(base_input_data_path, "LG_CONTROLS_0.5")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["delirium+", "delirium +", "patients_delirium+"]:
        potential_path = os.path.join(
            base_input_data_path, "LG_PATIENTS_DELIRIUM+_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["delirium-", "delirium -", "patients_delirium-"]:
        potential_path = os.path.join(
            base_input_data_path, "LG_PATIENTS_DELIRIUM-_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path

    # Fallback path logic if primary path fails
    if not data_root_path:
        detected_group = next(
            (grp for grp, s_list in ALL_SUBJECT_GROUPS.items()
             if subject_identifier in s_list), None,
        )
        if detected_group:
            logger_data_loading.warning(
                "Original group path for '%s' not found for subject '%s'. "
                "Subject found in group '%s'. Using its path convention.",
                group_affiliation, subject_identifier, detected_group
            )
            detected_group_lower = detected_group.lower()
            # Map to one of the 3 real LG directories only
            if detected_group_lower == "controls_delirium" or detected_group_lower == "controls_coma":
                data_root_path = os.path.join(
                    base_input_data_path, "LG_CONTROLS_0.5")
            elif detected_group_lower == "delirium+":
                data_root_path = os.path.join(
                    base_input_data_path, "LG_PATIENTS_DELIRIUM+_0.5"
                )
            elif detected_group_lower == "delirium-":
                data_root_path = os.path.join(
                    base_input_data_path, "LG_PATIENTS_DELIRIUM-_0.5"
                )
        else:  # Generic fallback - try to map to one of the 3 real directories
            if "control" in group_affiliation.lower():
                potential_path_generic = os.path.join(
                    base_input_data_path, "LG_CONTROLS_0.5"
                )
                if os.path.isdir(potential_path_generic):
                    data_root_path = potential_path_generic
            elif "delirium" in group_affiliation.lower():
                if "+" in group_affiliation or "delirium+" in group_affiliation.lower():
                    alt_path = os.path.join(
                        base_input_data_path, "LG_PATIENTS_DELIRIUM+_0.5")
                elif "-" in group_affiliation or "delirium-" in group_affiliation.lower():
                    alt_path = os.path.join(
                        base_input_data_path, "LG_PATIENTS_DELIRIUM-_0.5")
                else:
                    alt_path = None

                if alt_path and os.path.isdir(alt_path):
                    data_root_path = alt_path

    if not data_root_path or not os.path.isdir(data_root_path):
        logger_data_loading.error(
            "Data directory for subject '%s' (group: '%s') not found. "
            "Attempted path: '%s'. Base input: '%s'.",
            subject_identifier, group_affiliation, data_root_path,
            base_input_data_path
        )
        return None, {}, "unknown"

   
    possible_subject_ids = [subject_identifier,
                            subject_identifier.replace("Tp", "")]
    # Priority order: ICA_ar > ICA > noICA_ar > noICA
    possible_suffixes = ["ICA_LG", "noICA_LG"]  # Order by preference ICA first
    fname_candidates = []
    for s_id_cand in possible_subject_ids:
        for suffix_cand in possible_suffixes:
            # Format: {subject_id}_LG_preproc_{suffix}-epo.fif ou {subject_id}_LG_preproc_{suffix}-epo_ar.fif
            base_name = f"{s_id_cand}_LG_preproc_{suffix_cand}"
            # Priority: first _ar then without _ar for each suffix
            fname_candidates.extend([
                os.path.join(data_root_path, f"{base_name}-epo_ar.fif"),
                os.path.join(data_root_path, f"{base_name}-epo.fif")
            ])
    epochs_fif_filename = next(
        (f for f in fname_candidates if os.path.exists(f)), None
    )

    if not epochs_fif_filename:
        logger_data_loading.error(
            "No preprocessed LG epoch FIF file found for '%s' "
            "in '%s'. Checked %d candidates (first 5: %s).",
            subject_identifier, data_root_path,
            len(fname_candidates), fname_candidates[:5]
        )
        return None, {}, "unknown"

    if verbose_logging:
        logger_data_loading.info(
            "Loading LG epoch data for subject '%s' from: %s",
            subject_identifier, epochs_fif_filename
        )
    try:
        # Suppress MNE's verbose logging
        with mne.utils.use_log_level("WARNING"):
            epochs_object = mne.read_epochs(
                epochs_fif_filename, proj=False, verbose=False, preload=True
            )
        if epochs_object and verbose_logging:
            logger_data_loading.info(
                "LG Event IDs available for %s:", subject_identifier)
            if epochs_object.event_id:
                for desc, code in epochs_object.event_id.items():
                    logger_data_loading.info("  - '%s': %d", desc, code)
            else:
                logger_data_loading.info(
                    "  - No event_id found in epochs_object.")
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logger_data_loading.error(
            "Failed to read LG epochs file '%s' for subject '%s': %s",
            epochs_fif_filename, subject_identifier, e, exc_info=True,
        )
        return None, {}, "unknown"

    num_eeg_channels = len(epochs_object.copy().pick(picks="eeg").ch_names)
    num_time_points = len(epochs_object.times)
    extracted_data = {}
    actual_conditions_to_process = (
        conditions_to_load if conditions_to_load is not None
        else CONFIG_LOAD_MAIN_LG_DECODING
    )

    if verbose_logging:
        logger_data_loading.info(
            "Processing LG conditions: %s", list(
                actual_conditions_to_process.keys())
        )

    # --- Data extraction per condition ---
    for condition_name, specifier in actual_conditions_to_process.items():
        event_keys_to_select = []
        empty_data_array = np.empty((0, num_eeg_channels, num_time_points))

        if isinstance(specifier, str):
            if specifier.endswith(("/", "*")):
                prefix = specifier.rstrip("/*")
                event_keys_to_select = [
                    k for k in epochs_object.event_id if k.startswith(prefix)
                ]
            elif specifier.startswith("/"):
                # Handle suffix matching (e.g., "/GS" matches "LS/GS", "LD/GS")
                suffix = specifier
                event_keys_to_select = [
                    k for k in epochs_object.event_id if k.endswith(suffix)
                ]
            elif specifier in epochs_object.event_id:
                event_keys_to_select = [specifier]
        elif isinstance(specifier, list):
            event_keys_to_select = [
                k for k in specifier if k in epochs_object.event_id
            ]

        if not event_keys_to_select and specifier:
            if verbose_logging:
                logger_data_loading.info(
                    "Aucune donnée trouvée pour '%s' (condition: %s) "
                    "chez sujet %s. Événements disponibles: %s. "
                    "Ceci peut être normal selon le protocole du sujet.",
                    specifier, condition_name, subject_identifier,
                    list(epochs_object.event_id.keys())
                )
            extracted_data[condition_name] = empty_data_array
            continue

        try:
            if not event_keys_to_select:
                extracted_data[condition_name] = empty_data_array
                continue

            selected_epochs = epochs_object[event_keys_to_select]
            if len(selected_epochs) > 0:
                extracted_data[condition_name] = selected_epochs.pick(
                    picks="eeg"
                ).get_data(copy=False)  # Avoid unnecessary copy
                if verbose_logging:
                    logger_data_loading.debug(
                        "  Extracted %d epochs for '%s'. Shape: %s",
                        extracted_data[condition_name].shape[0],
                        condition_name,
                        extracted_data[condition_name].shape
                    )
            else:
                extracted_data[condition_name] = empty_data_array
        except KeyError:
            extracted_data[condition_name] = empty_data_array
            if verbose_logging:
                logger_data_loading.debug(
                    "KeyError (no events) for specifier '%s' (condition: %s) "
                    "for subject %s.",
                    specifier, condition_name, subject_identifier
                )
        except (ValueError, RuntimeError) as e_ex:
            logger_data_loading.error(
                "Error extracting LG data for condition '%s' (subject %s): %s",
                condition_name, subject_identifier, e_ex, exc_info=True,
            )
            extracted_data[condition_name] = empty_data_array

    total_loaded_epochs = sum(
        arr.shape[0] for arr in extracted_data.values()
        if hasattr(arr, "ndim") and arr.ndim == 3
    )
    if total_loaded_epochs == 0 and any(actual_conditions_to_process.values()):
        logger_data_loading.error(
            "CRITICAL: No LG data loaded for ANY specified conditions "
            "for subject %s.", subject_identifier
        )

    for cn, da in extracted_data.items():
        if da.size == 0 and actual_conditions_to_process.get(cn):
            logger_data_loading.warning(
                "Empty data array for LG condition '%s' (specifier: '%s') "
                "for subject %s.", cn, actual_conditions_to_process.get(cn),
                subject_identifier
            )

    if verbose_logging:
        shapes_log = ", ".join(
            [f"{name}={arr.shape}" for name, arr in extracted_data.items()]
        )
        logger_data_loading.info(
            "  LG Data shapes for '%s': %s. Loaded in %.2fs",
            subject_identifier, shapes_log, time.time() - start_time
        )
    
    # Determine detected protocol based on the data path
    detected_protocol = "LG"  # Default for Local-Global protocol
    if data_root_path:
        if "CONTROLS" in data_root_path:
            detected_protocol = "LG_CONTROLS"
        elif "PATIENTS_DELIRIUM+" in data_root_path:
            detected_protocol = "LG_PATIENTS_DELIRIUMM+"
        elif "PATIENTS_DELIRIUM-" in data_root_path:
            detected_protocol = "LG_PATIENTS_DELIRIUM-"
    
    return epochs_object, extracted_data, detected_protocol
