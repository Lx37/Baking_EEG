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
from config.decoding_config import CONFIG_LOAD_BATTERY_PROTOCOL
from config.decoding_config import CONFIG_LOAD_PPEXT3_PROTOCOL
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

    # Define search patterns - Priority order: ICA_ar > ICA > noICA_ar > noICA
    file_patterns = [
        "{subject_id}_PP_preproc_ICA_PP-epo_ar.fif",      # 1st priority: ICA with _ar
        "{subject_id}_PP_preproc_ICA_PP-epo.fif",         # 2nd priority: ICA without _ar
        "{subject_id}_PP_preproc_noICA_PP-epo_ar.fif",    # 3rd priority: noICA with _ar
        "{subject_id}_PP_preproc_noICA_PP-epo.fif"        # 4th priority: noICA without _ar
    ]
    
    battery_patterns = [
        "{subject_id}_preproc_ICA_PPAP-epo_ar.fif",       # 1st priority: ICA with _ar
        "{subject_id}_preproc_ICA_PPAP-epo.fif",          # 2nd priority: ICA without _ar
        "{subject_id}_preproc_noICA_PPAP-epo_ar.fif",     # 3rd priority: noICA with _ar
        "{subject_id}_preproc_noICA_PPAP-epo.fif"         # 4th priority: noICA without _ar
    ]

    # Check Battery subdirectory first
    battery_path = os.path.join(data_root_path, "Battery")
    if os.path.isdir(battery_path):
        if verbose_logging:
            logger_data_loading.debug("Searching in Battery directory: %s", battery_path)
        
        for subject_id in possible_subject_ids:
            for pattern in battery_patterns:
                filename = pattern.format(subject_id=subject_id)
                full_path = os.path.join(battery_path, filename)
                if verbose_logging:
                    logger_data_loading.debug("  Trying Battery file: %s", full_path)
                if os.path.exists(full_path):
                    if verbose_logging:
                        logger_data_loading.info(
                            "Found epochs file: %s (protocol: battery)",
                            full_path
                        )
                    return full_path, 'battery'
                    
    # Check PPext3 subdirectory
    ppext3_path = os.path.join(data_root_path, "PPext3")
    if os.path.isdir(ppext3_path):
        if verbose_logging:
            logger_data_loading.debug("Searching in PPext3 directory: %s", ppext3_path)
        
        for subject_id in possible_subject_ids:
            for pattern in battery_patterns:  # Same format as Battery
                filename = pattern.format(subject_id=subject_id)
                full_path = os.path.join(ppext3_path, filename)
                if verbose_logging:
                    logger_data_loading.debug("  Trying PPext3 file: %s", full_path)
                if os.path.exists(full_path):
                    if verbose_logging:
                        logger_data_loading.info(
                            "Found epochs file: %s (protocol: ppext3)",
                            full_path
                        )
                    return full_path, 'ppext3'
    # Check root directory (legacy format)
    if verbose_logging:
        logger_data_loading.debug("Searching in root directory: %s", data_root_path)
    
    for subject_id in possible_subject_ids:
        for pattern in file_patterns:
            filename = pattern.format(subject_id=subject_id)
            full_path = os.path.join(data_root_path, filename)
            if verbose_logging:
                logger_data_loading.debug("  Trying legacy file: %s", full_path)
            if os.path.exists(full_path):
                if verbose_logging:
                    logger_data_loading.info(
                        "Found epochs file: %s (protocol: legacy)",
                        full_path
                    )
                return full_path, 'legacy'

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
    
    # First, try to find the subject in ALL_SUBJECT_GROUPS to get the exact group name
    detected_group = next(
        (grp for grp, s_list in ALL_SUBJECT_GROUPS.items()
         if subject_identifier in s_list), None,
    )
    
    # Use detected group or provided group_affiliation
    group_to_use = detected_group if detected_group else group_affiliation.upper()
    
    data_root_path = None
    if group_to_use == "CONTROLS_DELIRIUM":
        potential_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_to_use == "COMA":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_COMA{freq_suffix}")
            if verbose_logging:
                logger_data_loading.debug("Trying COMA path: %s (exists: %s)", 
                                        potential_path, os.path.isdir(potential_path))
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "MCS+":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_MCS+{freq_suffix}")
            if verbose_logging:
                logger_data_loading.debug("Trying MCS+ path: %s (exists: %s)", 
                                        potential_path, os.path.isdir(potential_path))
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "MCS-":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_MCS-{freq_suffix}")
            if verbose_logging:
                logger_data_loading.debug("Trying MCS- path: %s (exists: %s)", 
                                        potential_path, os.path.isdir(potential_path))
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "VS":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_VS{freq_suffix}")
            if verbose_logging:
                logger_data_loading.debug("Trying VS path: %s (exists: %s)", 
                                        potential_path, os.path.isdir(potential_path))
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "VS":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_VS{freq_suffix}")
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "DELIRIUM+":
        potential_path = os.path.join(
            base_input_data_path,
            "PP_PATIENTS_DELIRIUM+_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    elif group_to_use == "DELIRIUM-":
        potential_path = os.path.join(
            base_input_data_path,
            "PP_PATIENTS_DELIRIUM-_0.5"
        )
        if os.path.isdir(potential_path):
            data_root_path = potential_path
    
    # Fallback for special cases or legacy group names
    if not data_root_path:
        group_affiliation_lower = group_affiliation.lower()
        if group_affiliation_lower == "controls_coma" or group_to_use == "CONTROLS_COMA":
            potential_path = os.path.join(base_input_data_path, "PP_CONTROLS_COMA_01HZ")
            if os.path.isdir(potential_path):
                data_root_path = potential_path

    if not data_root_path or not os.path.isdir(data_root_path):
        logger_data_loading.error(
            "Data directory for subject '%s' (group: '%s') not found. "
            "Attempted path: '%s'. Base input: '%s'.",
            subject_identifier, group_affiliation, data_root_path,
            base_input_data_path
        )
        return None, {}  # Or raise FileNotFoundError

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
        # Priority order: ICA_ar > ICA > noICA_ar > noICA
        possible_suffixes = ["ICA_PP", "noICA_PP"]
        fname_candidates = []
        for s_id_cand in possible_subject_ids:
            for suffix_cand in possible_suffixes:
                base_name = f"{s_id_cand}_PP_preproc_{suffix_cand}"
                # Priority: first _ar then without _ar for each suffix
                fname_candidates.extend([
                    os.path.join(data_root_path, f"{base_name}-epo_ar.fif"),
                    os.path.join(data_root_path, f"{base_name}-epo.fif")
                ])
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
                # Detailed event inspection for debugging
                inspect_epochs_events(epochs_fif_filename, subject_identifier, verbose=True)
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
    
    # Adaptively adjust conditions based on available events
    if conditions_to_load is None:
        actual_conditions_to_process = adapt_loading_conditions_to_available_events(
            epochs_object, subject_identifier, verbose_logging
        )
    else:
        actual_conditions_to_process = conditions_to_load

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
                
                # Fallback: try simple event names without prefixes for legacy data
                if not event_keys_to_select and prefix in ['PP', 'AP']:
                    # For legacy data, try just 'PP' or 'AP' as exact match
                    if prefix in epochs_object.event_id:
                        event_keys_to_select = [prefix]
                    else:
                        # Try numbered variants like PP1, AP1, etc.
                        numbered_variants = [
                            k for k in epochs_object.event_id 
                            if k.startswith(prefix) and k[len(prefix):].isdigit()
                        ]
                        if numbered_variants:
                            event_keys_to_select = numbered_variants
                        else:
                            # Try with underscore variants PP_1, AP_1, etc.
                            underscore_variants = [
                                k for k in epochs_object.event_id 
                                if k.startswith(prefix + '_') or k.startswith(prefix + '/')
                            ]
                            event_keys_to_select = underscore_variants
                            
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
    # Import configuration here to avoid circular imports
   
    
    # Default Battery protocol conditions if none specified
    if conditions_to_load is None:
        conditions_to_load = CONFIG_LOAD_BATTERY_PROTOCOL

    if verbose_logging:
        logger_data_loading.info(
            "Loading Battery protocol data for subject '%s' (group: %s)",
            subject_identifier, group_affiliation
        )
        logger_data_loading.info(
            "Battery protocol conditions: %s", list(conditions_to_load.keys())
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
    # Import configuration here to avoid circular imports
  
    
    # Default PPext3 protocol conditions if none specified
    if conditions_to_load is None:
        conditions_to_load = CONFIG_LOAD_PPEXT3_PROTOCOL

    if verbose_logging:
        logger_data_loading.info(
            "Loading PPext3 protocol data for subject '%s' (group: %s)",
            subject_identifier, group_affiliation
        )
        logger_data_loading.info(
            "PPext3 protocol conditions: %s", list(conditions_to_load.keys())
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

    Protocol-specific patterns:
    - PPext3: Has both Music-Noise AND Conv-Dio events (278 epochs expected)
    - Battery: Has Music/Conv events but NO Noise/Dio events (128 epochs expected)  
    - Delirium: Simple AP/PP structure without Music/Conv/Noise/Dio

    Args:
        epochs_object (mne.Epochs): The loaded epochs object.
        file_protocol_hint (str, optional): Protocol hint from file location
                                             detection ('battery', 'ppext3', 'legacy').

    Returns:
        str: Detected protocol type ('delirium', 'battery', 'ppext3', 'unknown')
    """
    if epochs_object is None or not hasattr(epochs_object, 'event_id'):
        return 'unknown'

    event_keys = list(epochs_object.event_id.keys())
    
    # Log all events for debugging
    logger_data_loading.debug("Event keys for protocol detection: %s", event_keys)
    
    # Analyze event patterns with more specificity
    has_music = any('Music' in k for k in event_keys)
    has_noise = any('Noise' in k for k in event_keys) 
    has_conv = any('Conv' in k for k in event_keys)
    has_dio = any('Dio' in k for k in event_keys)
    has_ap_slash_structure = any(k.startswith('AP/') for k in event_keys)
    has_pp_slash_structure = any(k.startswith('PP/') for k in event_keys)
    has_simple_ap = 'AP' in event_keys
    has_simple_pp = 'PP' in event_keys
    
    # Count total events for additional validation
    total_events = len(epochs_object.events) if hasattr(epochs_object, 'events') else 0
    
    logger_data_loading.debug(
        "Protocol detection analysis: Music=%s, Noise=%s, Conv=%s, Dio=%s, "
        "AP/=%s, PP/=%s, simple_AP=%s, simple_PP=%s, total_events=%d",
        has_music, has_noise, has_conv, has_dio, has_ap_slash_structure, 
        has_pp_slash_structure, has_simple_ap, has_simple_pp, total_events
    )
    
    # === PPext3 Protocol Detection (Most Specific) ===
    # PPext3 MUST have BOTH Music-Noise AND Conv-Dio combinations
    if (has_music and has_noise and has_conv and has_dio and 
        has_ap_slash_structure and has_pp_slash_structure):
        
        # Verify typical PPext3 event structure
        ppext3_events = [k for k in event_keys if k.startswith(('AP/', 'PP/'))]
        music_noise_events = [k for k in ppext3_events if 'Music' in k or 'Noise' in k]
        conv_dio_events = [k for k in ppext3_events if 'Conv' in k or 'Dio' in k]
        
        # PPext3 should have both types of events
        if len(music_noise_events) > 0 and len(conv_dio_events) > 0:
            logger_data_loading.info(
                "Detected PPext3 protocol: %d Music/Noise events, %d Conv/Dio events, %d total events",
                len(music_noise_events), len(conv_dio_events), total_events
            )
            return 'ppext3'
    
    # === Battery Protocol Detection ===
    # Battery has Music/Conv but NO Noise/Dio
    if (has_music and has_conv and not has_noise and not has_dio and
        has_ap_slash_structure and has_pp_slash_structure):
        
        # Verify typical Battery event structure
        battery_events = [k for k in event_keys if k.startswith(('AP/', 'PP/'))]
        music_conv_events = [k for k in battery_events if 'Music' in k or 'Conv' in k]
        
        if len(music_conv_events) > 0:
            logger_data_loading.info(
                "Detected Battery protocol: %d Music/Conv events, %d total events",
                len(music_conv_events), total_events
            )
            return 'battery'
    
    # === Legacy/Delirium Protocol Detection ===
    # Simple AP/PP structure without Music/Conv/Noise/Dio
    if ((has_simple_ap and has_simple_pp) or 
        (has_ap_slash_structure and has_pp_slash_structure)) and \
       not any(x in k for k in event_keys for x in ['Music', 'Conv', 'Noise', 'Dio']):
        
        logger_data_loading.info(
            "Detected Delirium/Legacy protocol: simple AP/PP structure, %d total events",
            total_events
        )
        return 'delirium'
    
    # === Use file protocol hint as fallback if structure is ambiguous ===
    if file_protocol_hint in ['battery', 'ppext3', 'legacy']:
        logger_data_loading.warning(
            "Protocol detection ambiguous, using file hint: %s (events: %s)",
            file_protocol_hint, event_keys[:5]
        )
        if file_protocol_hint == 'legacy':
            return 'delirium'
        return file_protocol_hint
    
    # === Final fallback: detect any AP/PP structure ===
    if any(k in ['AP', 'PP'] or k.startswith(('AP/', 'PP/')) for k in event_keys):
        logger_data_loading.warning(
            "Defaulting to delirium protocol for unrecognized AP/PP structure: %s",
            event_keys[:5]
        )
        return 'delirium'
    
    logger_data_loading.warning(
        "Could not detect protocol type. Available events: %s", event_keys
    )
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
    # First, try to find the subject in ALL_SUBJECT_GROUPS to get the exact group name
    detected_group = next(
        (grp for grp, s_list in ALL_SUBJECT_GROUPS.items()
         if subject_identifier in s_list), None,
    )
    
    # Use detected group or provided group_affiliation
    group_to_use = detected_group if detected_group else group_affiliation.upper()
    
    if group_to_use == "CONTROLS_DELIRIUM":
        data_root_path = os.path.join(base_input_data_path, "PP_CONTROLS_0.5")
    elif group_to_use == "COMA":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_COMA{freq_suffix}")
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "MCS+":
        # Try both 01HZ and 1HZ variants
        for freq_suffix in ["_01HZ", "_1HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_MCS+{freq_suffix}")
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "MCS-":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_MCS-{freq_suffix}")
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "VS":
        # Try both 1HZ and 01HZ variants (prioritize _1HZ based on actual data structure)
        for freq_suffix in ["_1HZ", "_01HZ"]:
            potential_path = os.path.join(base_input_data_path, f"PP_VS{freq_suffix}")
            if os.path.isdir(potential_path):
                data_root_path = potential_path
                break
    elif group_to_use == "DELIRIUM+":
        data_root_path = os.path.join(
            base_input_data_path,
            "PP_PATIENTS_DELIRIUM+_0.5"
        )
    elif group_to_use == "DELIRIUM-":
        data_root_path = os.path.join(
            base_input_data_path,
            "PP_PATIENTS_DELIRIUM-_0.5"
        )
    
    # Fallback for special cases
    if not data_root_path or not os.path.isdir(data_root_path):
        group_affiliation_lower = group_affiliation.lower()
        if group_affiliation_lower == "controls_coma" or group_to_use == "CONTROLS_COMA":
            data_root_path = os.path.join(base_input_data_path, "PP_CONTROLS_COMA_01HZ")

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


def inspect_epochs_events(epochs_file_path, subject_identifier, verbose=True):
    """Inspect available events in an epochs file for debugging.
    
    Args:
        epochs_file_path (str): Path to the epochs file
        subject_identifier (str): Subject ID for logging
        verbose (bool): Enable verbose output
        
    Returns:
        dict: Dictionary with event_id information
    """
    try:
        with mne.utils.use_log_level("WARNING"):
            epochs = mne.read_epochs(epochs_file_path, proj=False, verbose=False, preload=False)
        
        event_info = {
            'event_id': epochs.event_id,
            'n_events': len(epochs.events),
            'event_counts': {}
        }
        
        # Count events per type
        for event_name, event_code in epochs.event_id.items():
            count = sum(epochs.events[:, 2] == event_code)
            event_info['event_counts'][event_name] = count
        
        if verbose:
            logger_data_loading.info(
                "=== EVENT INSPECTION FOR %s ===", subject_identifier
            )
            logger_data_loading.info("File: %s", epochs_file_path)
            logger_data_loading.info("Total events: %d", event_info['n_events'])
            logger_data_loading.info("Event types and counts:")
            for event_name, count in event_info['event_counts'].items():
                logger_data_loading.info("  - '%s' (code %d): %d epochs", 
                                       event_name, epochs.event_id[event_name], count)
                                       
        return event_info
        
    except Exception as e:
        logger_data_loading.error(
            "Failed to inspect events for %s: %s", subject_identifier, e
        )
        return {}


def adapt_loading_conditions_to_available_events(epochs_object, subject_identifier, verbose_logging=True):
    """Adapt loading conditions based on actually available events in the data.
    
    Args:
        epochs_object (mne.Epochs): The loaded epochs object
        subject_identifier (str): Subject ID for logging
        verbose_logging (bool): Enable detailed logging
        
    Returns:
        dict: Adapted conditions configuration
    """
    if not epochs_object or not hasattr(epochs_object, 'event_id'):
        return CONFIG_LOAD_MAIN_DECODING
    
    available_events = list(epochs_object.event_id.keys())
    adapted_conditions = {}
    
    # Detect protocol type to adapt AP family logic
    detected_protocol = detect_protocol_type(epochs_object, None)
    
    # Try to find PP events
    pp_events = [k for k in available_events if 'PP' in k.upper()]
    if pp_events:
        if any(k.startswith('PP/') for k in pp_events):
            adapted_conditions['XPP_ALL'] = 'PP/'
        elif 'PP' in available_events:
            adapted_conditions['XPP_ALL'] = 'PP'
        else:
            # Use all PP-containing events
            adapted_conditions['XPP_ALL'] = pp_events
    
    # Try to find AP events  
    ap_events = [k for k in available_events if 'AP' in k.upper()]
    if ap_events:
        if any(k.startswith('AP/') for k in ap_events):
            adapted_conditions['XAP_ALL'] = 'AP/'
        elif 'AP' in available_events:
            adapted_conditions['XAP_ALL'] = 'AP'
        else:
            # Use all AP-containing events
            adapted_conditions['XAP_ALL'] = ap_events
    
    # Try to find specific PP events for comparison (protocol-specific)
    if detected_protocol == 'battery':
        # Battery protocol: look for PP/Music/ and PP/Conv/
        pp_specific = [k for k in available_events if k.startswith('PP/') and ('Music' in k or 'Conv' in k)]
        if pp_specific:
            adapted_conditions['PP_FOR_SPECIFIC_COMPARISON'] = ['PP/Music/', 'PP/Conv/']
    elif detected_protocol == 'ppext3':
        # PPext3 protocol: look for PP/Music/, PP/Noise/, PP/Conv/, PP/Dio/
        pp_specific = [k for k in available_events if k.startswith('PP/') and any(x in k for x in ['Music', 'Noise', 'Conv', 'Dio'])]
        if pp_specific:
            adapted_conditions['PP_FOR_SPECIFIC_COMPARISON'] = ['PP/Music/', 'PP/Noise/', 'PP/Conv/', 'PP/Dio/']
    else:
        # Delirium protocol: look for PP/10, PP/20, PP/30
        pp_specific = [k for k in available_events if k.startswith('PP/') and any(x in k for x in ['10', '20', '30'])]
        if pp_specific:
            adapted_conditions['PP_FOR_SPECIFIC_COMPARISON'] = pp_specific
    
    # Try to find AP families (protocol-specific logic)
    for family_num in range(1, 7):
        if detected_protocol == 'battery':
            # Battery protocol: AP/1/Music/, AP/1/Conv/, etc.
            family_events = [k for k in available_events 
                            if k.startswith(f'AP/{family_num}/') and ('Music' in k or 'Conv' in k)]
            if family_events:
                adapted_conditions[f'AP_FAMILY_{family_num}'] = [f'AP/{family_num}/Music/', f'AP/{family_num}/Conv/']
        elif detected_protocol == 'ppext3':
            # PPext3 protocol: AP/1/Music/, AP/1/Noise/, AP/1/Conv/, AP/1/Dio/, etc.
            family_events = [k for k in available_events 
                            if k.startswith(f'AP/{family_num}/') and any(x in k for x in ['Music', 'Noise', 'Conv', 'Dio'])]
            if family_events:
                adapted_conditions[f'AP_FAMILY_{family_num}'] = [f'AP/{family_num}/Music/', f'AP/{family_num}/Noise/', 
                                                                 f'AP/{family_num}/Conv/', f'AP/{family_num}/Dio/']
        else:
            # Delirium protocol: AP/11, AP/21, AP/31, etc.
            family_events = [k for k in available_events 
                            if k.startswith('AP/') and k.endswith(str(family_num))]
            if family_events:
                adapted_conditions[f'AP_FAMILY_{family_num}'] = family_events
    
    if verbose_logging:
        logger_data_loading.info(
            "Adapted loading conditions for %s (protocol: %s): %s", 
            subject_identifier, detected_protocol, adapted_conditions
        )
        logger_data_loading.info(
            "Available events were: %s", available_events
        )
    
    # Fallback to original config if nothing found
    if not adapted_conditions:
        if verbose_logging:
            logger_data_loading.warning(
                "No PP/AP events found for %s, using default config", 
                subject_identifier
            )
        return CONFIG_LOAD_MAIN_DECODING
    
    return adapted_conditions
