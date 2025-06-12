"""
Utilities module for Baking_EEG package.

This module contains various utility functions for EEG data processing,
visualization, and statistical analysis.
"""

# Make key functions available at package level
try:
    from .utils import configure_project_paths, setup_analysis_results_directory
    from . import stats_utils
    from . import loading_PP_utils
    from . import loading_LG_utils
    
    # Import visualization utilities
    try:
        from . import vizualization_utils_PP
        from . import visualization_utils_LG
    except ImportError:
        # Handle case where visualization modules might have dependencies not installed
        pass
        
except ImportError as e:
    import warnings
    warnings.warn(f"Some utilities could not be imported: {e}")

__all__ = [
    'configure_project_paths',
    'setup_analysis_results_directory', 
    'stats_utils',
    'loading_PP_utils',
    'loading_LG_utils'
]
