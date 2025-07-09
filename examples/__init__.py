# -*- coding: utf-8 -*-
"""
Examples module for Baking_EEG package.

This module contains example scripts for various EEG analysis workflows:
- PP (Predictive Processing) protocol analysis
- LG (Local-Global) protocol analysis  
- Group-level analyses
- Single subject analyses
"""

__version__ = "1.0.0"
__author__ = "Baking_EEG Team"

# Expose main functions for easier imports
try:
    from .run_decoding_one_pp import execute_single_subject_decoding
    __all__ = ['execute_single_subject_decoding']
except ImportError:
    # If modules aren't available yet, that's OK
    __all__ = []

try:
    from .run_decoding_one_lg import execute_single_subject_lg_decoding
    __all__.append('execute_single_subject_lg_decoding')
except ImportError:
    pass

try:
    from .run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
    __all__.append('execute_group_intra_subject_decoding_analysis')
except ImportError:
    pass

try:
    from .run_decoding_one_group_lg import execute_group_intra_subject_lg_decoding_analysis
    __all__.append('execute_group_intra_subject_lg_decoding_analysis')
except ImportError:
    pass
