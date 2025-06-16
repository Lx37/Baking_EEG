# --- Data loading function for PP
# DELIRIUM protocol
import os
import time
import logging
import numpy as np
import mne
from config.config import ALL_SUBJECT_GROUPS
from config.decoding_config import CONFIG_LOAD_MAIN_DECODING
logger_data_loading = logging.getLogger(__name__)


def find_epochs_file_with_protocol_detection(
    data_root_path, subject_identifier, verbose_logging=True
):
    """Find epochs file with protocol-specific folder structure detection.

    Searches for epochs files in both legacy structure (data_epochs/) and
    new protocol-specific structures (Battery/, PPext3/ subfolders).

    Args:
        data_root_path (str): Root path of the data directory
        subject_identifier (str): Subject ID to search for
        verbose_logging (bool): Enable verbose logging

    Returns:
        tuple: (file_path, detected_protocol)
            - file_path: Full path to found epochs file or None
            - detected_protocol: 'legacy', 'battery', 'ppext3', or None
    """
    possible_subject_ids = [
        subject_identifier, subject_identifier.replace("Tp", "")
    ]

    # Define search patterns for different file naming conventions
    file_patterns = {
        'legacy': [
            "{subject_id}_PP_preproc_noICA_PP-epo_ar.fif",
            "{subject_id}_PP_preproc_ICA_PP-epo_ar.fif",
            "{subject_id}_PP_preproc-epo_ar.fif"
        ],
        'new_format': [
            "{subject_id}_preproc_noICA_PPAP-epo_ar.fif",
            "{subject_id}_preproc_ICA_PPAP-epo_ar.fif",
            "{subject_id}_preproc_PPAP-epo_ar.fif"
        ]
    }

    # Define search locations in order of preference
    search_locations = [
        # Protocol-specific subdirectories (new structure)
        ('battery', ['Battery', 'battery']),
        ('ppext3', ['PPext3', 'ppext3', 'PPExt3']),
        # Legacy data_epochs directory
        ('legacy', ['data_epochs'])
    ]

    for protocol_type, subdirs in search_locations:
        for subdir in subdirs:
            epochs_dir = os.path.join(data_root_path, subdir)
            if not os.path.isdir(epochs_dir):
                continue

            if verbose_logging:
                logger_data_loading.debug(
                    "Searching in %s directory: %s", protocol_type, epochs_dir
                )

            # Try both legacy and new file naming patterns
            for pattern_type, patterns in file_patterns.items():
                for subject_id in possible_subject_ids:
                    for pattern in patterns:
                        filename = pattern.format(subject_id=subject_id)
                        full_path = os.path.join(epochs_dir, filename)

                        if os.path.exists(full_path):
                            if verbose_logging:
                                logger_data_loading.info(
                                    "Found epochs file: %s (protocol: %s, "
                                    "pattern: %s)",
                                    full_path, protocol_type, pattern_type
                                )
                            return full_path, protocol_type

    # If nothing found, return None
    if verbose_logging:
        logger_data_loading.warning(
            "No epochs file found for subject '%s' in '%s'",
            subject_identifier, data_root_path
        )
    return None, None


def load_epochs_data_for_decoding_delirium(
    subject_identifier,
    group_affiliation,
    base_input_data_path,
    conditions_to_load=None,
    verbose_logging=True,
):
    """Load preprocessed MNE Epochs data for a subject and extract data.

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation (e.g., 'controls',
                                 'del'). Used for path determination.
        base_input_data_path (str): Root directory for input data.
        conditions_to_load (dict, optional): Specifies conditions to extract.
            Keys: custom names for extracted data arrays.
            Values: MNE event specifiers (e.g., 'Event/Type',
            ['Event/1', 'Event/2'], or 'Event/Prefix/').
            Defaults to `CONFIG_LOAD_MAIN_DECODING` if None.
        verbose_logging (bool): If True, enables detailed logging.

    Returns:
        tuple: (mne.Epochs or None, dict)
            - mne.Epochs object if loading successful, else None.
            - Dictionary: keys are `condition_name` from `conditions_to_load`,
              values are NumPy arrays of epoch data
              (n_epochs, n_channels, n_times).
              Empty array if a condition has no data.
    """
    # Parameter validation
    if not isinstance(subject_identifier, str) or not subject_identifier:
        logger_data_loading.error(
            "Subject identifier must be a non-empty string.")
        return None, {}
    if not isinstance(group_affiliation, str) or not group_affiliation:
        logger_data_loading.error(
            "Group affiliation must be a non-empty string.")
        return None, {}
    if (not isinstance(base_input_data_path, str) or
            not os.path.isdir(base_input_data_path)):
        logger_data_loading.error(
            "Base input data path '%s' is not a valid directory.",
            base_input_data_path
        )
        return None, {}
    if (conditions_to_load is not None and
            not isinstance(conditions_to_load, dict)):
        logger_data_loading.error(
            "'conditions_to_load' must be a dict or None. "
            "Received type: %s.", type(conditions_to_load)
        )
        return None, {}

    start_time = time.time()
    group_affiliation_lower = group_affiliation.lower()
    data_root_path = None

    # --- Path determination logic ---
    if group_affiliation_lower == "controls":
        potential_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["del", "nodel"]:
        potential_path = os.path.join(
            base_input_data_path,
            f"PP_PATIENTS_{group_affiliation.upper()}_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    # New group support: COMA, MCS, MCS-, VG with 1HZ frequency
    elif group_affiliation_lower in ["coma"]:
        potential_path = os.path.join(base_input_data_path, "PP_COMA_1HZ")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["mcs"]:
        potential_path = os.path.join(base_input_data_path, "PP_MCS_1HZ")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["mcs-", "mcs_minus"]:
        potential_path = os.path.join(base_input_data_path, "PP_MCS-_1HZ")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_affiliation_lower in ["vg"]:
        potential_path = os.path.join(base_input_data_path, "PP_VG_1HZ")
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
            group_affiliation_lower = detected_group.lower()
            if group_affiliation_lower == "controls":
                data_root_path = os.path.join(
                    base_input_data_path, "PP_CONTROLS_0.5")
            elif group_affiliation_lower in ["del", "nodel"]:
                data_root_path = os.path.join(
                    base_input_data_path,
                    f"PP_PATIENTS_{detected_group.upper()}_0.5"
                )
            elif group_affiliation_lower == "coma":
                data_root_path = os.path.join(
                    base_input_data_path, "PP_COMA_1HZ")
            elif group_affiliation_lower == "mcs":
                data_root_path = os.path.join(
                    base_input_data_path, "PP_MCS_1HZ")
            elif group_affiliation_lower in ["mcs-", "mcs_minus"]:
                data_root_path = os.path.join(
                    base_input_data_path, "PP_MCS-_1HZ")
            elif group_affiliation_lower == "vg":
                data_root_path = os.path.join(
                    base_input_data_path, "PP_VG_1HZ")
        else:  # Generic fallback
            # Try new 1HZ format first
            potential_paths_1hz = [
                os.path.join(
                    base_input_data_path,
                    f"PP_{group_affiliation.upper()}_1HZ"
                ),
                os.path.join(
                    base_input_data_path,
                    f"PP_{group_affiliation.upper()}-_1HZ"
                )
            ]
            for potential_path in potential_paths_1hz:
                if os.path.isdir(potential_path):
                    data_root_path = potential_path
                    break

            # Try original 0.5 format if 1HZ not found
            if not data_root_path:
                potential_path_generic = os.path.join(
                    base_input_data_path,
                    f"PP_{group_affiliation.upper()}_0.5"
                )
                if os.path.isdir(potential_path_generic):
                    data_root_path = potential_path_generic
                else:
                    data_root_path = base_input_data_path  # Last resort

    if not data_root_path or not os.path.isdir(data_root_path):
        logger_data_loading.error(
            "Data directory for subject '%s' (group: '%s') not found. "
            "Attempted path: '%s'. Base input: '%s'.",
            subject_identifier, group_affiliation, data_root_path,
            base_input_data_path
        )
        return None, {}  # Or raise FileNotFoundError

    epochs_file_path_base = os.path.join(data_root_path, "data_epochs")

    # Use new file detection function for multiple protocols/structures
    epochs_fif_filename, file_protocol_type = (
        find_epochs_file_with_protocol_detection(
            data_root_path, subject_identifier, verbose_logging
        )
    )

    # Legacy fallback if new detection fails
    if not epochs_fif_filename:
        possible_subject_ids = [
            subject_identifier, subject_identifier.replace("Tp", "")
        ]
        possible_suffixes = ["noICA_PP", "ICA_PP", ""]
        fname_candidates = []
        for s_id_cand in possible_subject_ids:
            for suffix_cand in possible_suffixes:
                base_name = f"{s_id_cand}_PP_preproc"
                if suffix_cand:
                    base_name += f"_{suffix_cand}"
                fname_candidates.append(
                    os.path.join(
                        epochs_file_path_base, f"{base_name}-epo_ar.fif"
                    )
                )
        epochs_fif_filename = next(
            (f for f in fname_candidates if os.path.exists(f)), None
        )
        file_protocol_type = 'legacy_fallback'

    if not epochs_fif_filename:
        logger_data_loading.error(
            "No preprocessed epoch FIF file found for '%s' "
            "in '%s'. Tried protocol-specific detection and legacy fallback.",
            subject_identifier, data_root_path
        )
        return None, {}  # Or raise FileNotFoundError

    if verbose_logging:
        logger_data_loading.info(
            "Loading epoch data for subject '%s' from: %s (file protocol: %s)",
            subject_identifier, epochs_fif_filename, file_protocol_type
        )
    try:
        # Suppress MNE's verbose logging
        with mne.utils.use_log_level("WARNING"):
            epochs_object = mne.read_epochs(
                epochs_fif_filename, proj=False, verbose=False, preload=True
            )
        if epochs_object and verbose_logging:
            logger_data_loading.info(
                "Event IDs available for %s:", subject_identifier)
            if epochs_object.event_id:
                for desc, code in epochs_object.event_id.items():
                    logger_data_loading.info("  - '%s': %d", desc, code)
            else:
                logger_data_loading.info(
                    "  - No event_id found in epochs_object.")
    except (FileNotFoundError, IOError, ValueError, RuntimeError) as e:
        logger_data_loading.error(
            "Failed to read epochs file '%s' for subject '%s': %s",
            epochs_fif_filename, subject_identifier, e, exc_info=True,
        )
        return None, {}

    num_eeg_channels = len(epochs_object.copy().pick(picks="eeg").ch_names)
    num_time_points = len(epochs_object.times)
    extracted_data = {}
    actual_conditions_to_process = (
        conditions_to_load if conditions_to_load is not None
        else CONFIG_LOAD_MAIN_DECODING
    )

    if verbose_logging:
        logger_data_loading.info(
            "Processing conditions: %s", list(
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
            elif specifier in epochs_object.event_id:
                event_keys_to_select = [specifier]
        elif isinstance(specifier, list):
            event_keys_to_select = [
                k for k in specifier if k in epochs_object.event_id
            ]

        if not event_keys_to_select and specifier:
            if verbose_logging:
                logger_data_loading.warning(
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
                "Error extracting data for condition '%s' (subject %s): %s",
                condition_name, subject_identifier, e_ex, exc_info=True,
            )
            extracted_data[condition_name] = empty_data_array

    total_loaded_epochs = sum(
        arr.shape[0] for arr in extracted_data.values()
        if hasattr(arr, "ndim") and arr.ndim == 3
    )
    if total_loaded_epochs == 0 and any(actual_conditions_to_process.values()):
        logger_data_loading.error(
            "CRITICAL: No data loaded for ANY specified conditions "
            "for subject %s.", subject_identifier
        )

    for cn, da in extracted_data.items():
        if da.size == 0 and actual_conditions_to_process.get(cn):
            logger_data_loading.warning(
                "Empty data array for condition '%s' (specifier: '%s') "
                "for subject %s.", cn, actual_conditions_to_process.get(cn),
                subject_identifier
            )

    if verbose_logging:
        shapes_log = ", ".join(
            [f"{name}={arr.shape}" for name, arr in extracted_data.items()]
        )
        logger_data_loading.info(
            "  Data shapes for '%s': %s. Loaded in %.2fs",
            subject_identifier, shapes_log, time.time() - start_time
        )
    return epochs_object, extracted_data


# the classifier outputs a predicted probability of belonging to the standard
# class (S). Note that the probability of belonging to the other, deviant
# class (D) c
# PPext3

def load_epochs_data_for_decoding_battery(
    subject_identifier,
    group_affiliation,
    base_input_data_path,
    conditions_to_load=None,
    verbose_logging=True,
):
    """Load preprocessed MNE Epochs data for Battery protocol subjects.

    Battery Protocol Event Structure:
    - AP: AP/1-6/Music/Conv/D-G/V1-3 and AP/1-6/Music/Conv/D-G/V1-3
    - PP: PP/Music/Conv/D-G/V1-3 and PP/Music/Conv/D-G/V1-3
    - Expected: 128 epochs total, 126 channels, 500Hz sampling,
      -0.2 to 1.0s window

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation (e.g.,
                                 'controls', 'del').
        base_input_data_path (str): Root directory for input data.
        conditions_to_load (dict, optional): Specifies conditions to extract.
            Keys: custom names for extracted data arrays.
            Values: MNE event specifiers (e.g., 'AP/', 'PP/', etc.).
            Defaults to Battery-specific configuration if None.
        verbose_logging (bool): If True, enables detailed logging.

    Returns:
        tuple: (mne.Epochs or None, dict)
            - mne.Epochs object if loading successful, else None.
            - Dictionary: keys are condition names, values are NumPy arrays
              of epoch data (n_epochs, n_channels, n_times).
    """
    # Default Battery protocol conditions if none specified
    if conditions_to_load is None:
        conditions_to_load = {
            'AP_events': 'AP/',  # All AP events (AP/1-6/Music/Conv/D-G/V1-3)
            'PP_events': 'PP/',  # All PP events (PP/Music/Conv/D-G/V1-3)
            'AP_Music': 'AP/Music/',  # AP Music events
            'AP_Conv': 'AP/Conv/',   # AP Conversation events
            'PP_Music': 'PP/Music/',  # PP Music events
            'PP_Conv': 'PP/Conv/',   # PP Conversation events
        }

    if verbose_logging:
        logger_data_loading.info(
            "Loading Battery protocol data for subject '%s' (group: %s)",
            subject_identifier, group_affiliation
        )

    # Use the main loading function with Battery-specific conditions
    epochs_object, extracted_data = load_epochs_data_for_decoding_delirium(
        subject_identifier=subject_identifier,
        group_affiliation=group_affiliation,
        base_input_data_path=base_input_data_path,
        conditions_to_load=conditions_to_load,
        verbose_logging=verbose_logging
    )

    # Battery protocol validation
    if epochs_object is not None and verbose_logging:
        total_epochs = sum(
            arr.shape[0] for arr in extracted_data.values()
            if hasattr(arr, 'ndim') and arr.ndim == 3
        )
        logger_data_loading.info(
            "Battery protocol loaded: %d total epochs for subject '%s'",
            total_epochs, subject_identifier
        )

        # Log expected Battery event structure
        if epochs_object.event_id:
            battery_events = [
                k for k in epochs_object.event_id.keys()
                if k.startswith(('AP/', 'PP/'))
            ]
            logger_data_loading.info(
                "Battery events detected: %d events (%s)",
                len(battery_events),
                (battery_events[:5] if len(battery_events) > 5
                 else battery_events)
            )

    return epochs_object, extracted_data


def load_epochs_data_for_decoding_ppext3(
    subject_identifier,
    group_affiliation,
    base_input_data_path,
    conditions_to_load=None,
    verbose_logging=True,
):
    """Load preprocessed MNE Epochs data for PPext3 protocol subjects.

    PPext3 Protocol Event Structure:
    - AP: AP/1-6/Music-Noise/Conv-Dio/D-G/V1-3
    - PP: PP/Music-Noise/Conv-Dio/D-G/V1-3
    - Expected: 278 epochs total, 126 channels, 500Hz sampling,
      -0.2 to 1.0s window

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation (e.g.,
                                 'controls', 'del').
        base_input_data_path (str): Root directory for input data.
        conditions_to_load (dict, optional): Specifies conditions to extract.
            Keys: custom names for extracted data arrays.
            Values: MNE event specifiers (e.g., 'AP/', 'PP/', etc.).
            Defaults to PPext3-specific configuration if None.
        verbose_logging (bool): If True, enables detailed logging.

    Returns:
        tuple: (mne.Epochs or None, dict)
            - mne.Epochs object if loading successful, else None.
            - Dictionary: keys are condition names, values are NumPy arrays
              of epoch data (n_epochs, n_channels, n_times).
    """
    # Default PPext3 protocol conditions if none specified
    if conditions_to_load is None:
        conditions_to_load = {
            # All AP events (AP/1-6/Music-Noise/Conv-Dio/D-G/V1-3)
            'AP_events': 'AP/',
            # All PP events (PP/Music-Noise/Conv-Dio/D-G/V1-3)
            'PP_events': 'PP/',
            'AP_Music': 'AP/Music/',     # AP Music events
            'AP_Noise': 'AP/Noise/',     # AP Noise events
            'AP_Conv': 'AP/Conv/',       # AP Conversation events
            'AP_Dio': 'AP/Dio/',         # AP Dialogue events
            'PP_Music': 'PP/Music/',     # PP Music events
            'PP_Noise': 'PP/Noise/',     # PP Noise events
            'PP_Conv': 'PP/Conv/',       # PP Conversation events
            'PP_Dio': 'PP/Dio/',         # PP Dialogue events
        }

    if verbose_logging:
        logger_data_loading.info(
            "Loading PPext3 protocol data for subject '%s' (group: %s)",
            subject_identifier, group_affiliation
        )

    # Use the main loading function with PPext3-specific conditions
    epochs_object, extracted_data = load_epochs_data_for_decoding_delirium(
        subject_identifier=subject_identifier,
        group_affiliation=group_affiliation,
        base_input_data_path=base_input_data_path,
        conditions_to_load=conditions_to_load,
        verbose_logging=verbose_logging
    )

    # PPext3 protocol validation
    if epochs_object is not None and verbose_logging:
        total_epochs = sum(
            arr.shape[0] for arr in extracted_data.values()
            if hasattr(arr, 'ndim') and arr.ndim == 3
        )
        logger_data_loading.info(
            "PPext3 protocol loaded: %d total epochs for subject '%s'",
            total_epochs, subject_identifier
        )

        # Log expected PPext3 event structure
        if epochs_object.event_id:
            ppext3_events = [
                k for k in epochs_object.event_id.keys()
                if k.startswith(('AP/', 'PP/'))
            ]
            music_noise_events = [
                k for k in ppext3_events
                if 'Music' in k or 'Noise' in k
            ]
            conv_dio_events = [
                k for k in ppext3_events
                if 'Conv' in k or 'Dio' in k
            ]

            logger_data_loading.info(
                "PPext3 events detected: %d total, %d Music/Noise, "
                "%d Conv/Dio",
                len(ppext3_events), len(music_noise_events),
                len(conv_dio_events)
            )

    return epochs_object, extracted_data


def detect_protocol_type(epochs_object, file_protocol_hint=None):
    """Automatically detect protocol type based on event IDs and file location.

    Args:
        epochs_object (mne.Epochs): The loaded epochs object.
        file_protocol_hint (str, optional): Protocol hint from file location
                                             detection.

    Returns:
        str: Detected protocol type ('delirium', 'battery', 'ppext3',
             'unknown')
    """
    if epochs_object is None or not hasattr(epochs_object, 'event_id'):
        return 'unknown'

    event_keys = list(epochs_object.event_id.keys())

    # Use file protocol hint if available and matches event structure
    if file_protocol_hint in ['battery', 'ppext3']:
        # Verify the hint matches the actual event structure
        has_music_noise = any('Music' in k or 'Noise' in k for k in event_keys)
        has_conv_dio = any('Conv' in k and 'Dio' in k for k in event_keys)

        if file_protocol_hint == 'ppext3' and has_music_noise and has_conv_dio:
            return 'ppext3'
        elif (file_protocol_hint == 'battery' and has_music_noise and
              not has_conv_dio):
            return 'battery'

    # Check for PPext3 protocol (Music-Noise and Conv-Dio patterns)
    has_music_noise = any('Music' in k or 'Noise' in k for k in event_keys)
    has_conv_dio = any('Conv' in k and 'Dio' in k for k in event_keys)

    if has_music_noise and has_conv_dio:
        return 'ppext3'

    # Check for Battery protocol (Music/Conv but no Noise/Dio)
    has_music_conv = any('Music' in k or 'Conv' in k for k in event_keys)
    has_ap_pp_structure = any(k.startswith(('AP/', 'PP/')) for k in event_keys)

    if has_music_conv and has_ap_pp_structure and not has_conv_dio:
        return 'battery'

    # Check for standard delirium protocol
    has_delirium_events = any(
        k in ['AP', 'PP'] or k.startswith(('AP/', 'PP/'))
        for k in event_keys
    )

    if has_delirium_events and not has_music_conv:
        return 'delirium'

    return 'unknown'


def load_epochs_data_auto_protocol(
    subject_identifier,
    group_affiliation,
    base_input_data_path,
    conditions_to_load=None,
    verbose_logging=True,
):
    """Automatically detect protocol and load data using appropriate function.

    Args:
        subject_identifier (str): The ID of the subject.
        group_affiliation (str): The group affiliation.
        base_input_data_path (str): Root directory for input data.
        conditions_to_load (dict, optional): Custom conditions to load.
        verbose_logging (bool): Enable detailed logging.

    Returns:
        tuple: (mne.Epochs or None, dict, str)
            - mne.Epochs object if successful
            - Dictionary of extracted data
            - Detected protocol type
    """
    # First, load with delirium function to detect protocol and get file info
    epochs_temp, _ = load_epochs_data_for_decoding_delirium(
        subject_identifier=subject_identifier,
        group_affiliation=group_affiliation,
        base_input_data_path=base_input_data_path,
        conditions_to_load={'temp': 'AP'},  # Minimal load for detection
        verbose_logging=False
    )

    # Get file protocol hint from the temporary load (context)
    # We'll need to detect this from the file path structure
    data_root_path = None
    file_protocol_hint = None

    # Reconstruct the data path to get file protocol hint
    group_affiliation_lower = group_affiliation.lower()
    if group_affiliation_lower == "controls":
        data_root_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
    elif group_affiliation_lower in ["del", "nodel"]:
        data_root_path = os.path.join(
            base_input_data_path,
            f"PP_PATIENTS_{group_affiliation.upper()}_0.5"
        )
    elif group_affiliation_lower in ["coma"]:
        data_root_path = os.path.join(base_input_data_path, "PP_COMA_1HZ")
    elif group_affiliation_lower in ["mcs"]:
        data_root_path = os.path.join(base_input_data_path, "PP_MCS_1HZ")
    elif group_affiliation_lower in ["mcs-", "mcs_minus"]:
        data_root_path = os.path.join(base_input_data_path, "PP_MCS-_1HZ")
    elif group_affiliation_lower in ["vg"]:
        data_root_path = os.path.join(base_input_data_path, "PP_VG_1HZ")

    if data_root_path and os.path.isdir(data_root_path):
        _, file_protocol_hint = find_epochs_file_with_protocol_detection(
            data_root_path, subject_identifier, verbose_logging=False
        )

    # Detect protocol type with file hint
    protocol_type = detect_protocol_type(epochs_temp, file_protocol_hint)

    if verbose_logging:
        logger_data_loading.info(
            "Auto-detected protocol '%s' for subject '%s' (file hint: %s)",
            protocol_type, subject_identifier, file_protocol_hint
        )

    # Load data with appropriate function
    if protocol_type == 'battery':
        epochs_object, extracted_data = load_epochs_data_for_decoding_battery(
            subject_identifier, group_affiliation, base_input_data_path,
            conditions_to_load, verbose_logging
        )
    elif protocol_type == 'ppext3':
        epochs_object, extracted_data = load_epochs_data_for_decoding_ppext3(
            subject_identifier, group_affiliation, base_input_data_path,
            conditions_to_load, verbose_logging
        )
    else:  # delirium or unknown
        epochs_object, extracted_data = load_epochs_data_for_decoding_delirium(
            subject_identifier, group_affiliation, base_input_data_path,
            conditions_to_load, verbose_logging
        )

    return epochs_object, extracted_data, protocol_type
