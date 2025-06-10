#!/usr/bin/env python3
"""
Test script for loading_PP_utils.py
Tests the corrected code functionality
"""

import sys
import os

# Add current directory to path for imports
sys.path.append('.')


def test_imports():
    """Test all imports from loading_PP_utils.py"""
    print("🧪 Testing imports...")

    try:
        import numpy as np
        print("✅ NumPy import: OK")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    try:
        import mne
        print("✅ MNE import: OK")
    except ImportError as e:
        print(f"❌ MNE import failed: {e}")
        return False

    try:
        from config.config import ALL_SUBJECT_GROUPS
        print("✅ Config import: OK")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False

    try:
        from config.decoding_config import CONFIG_LOAD_MAIN_DECODING
        print("✅ Decoding config import: OK")
    except ImportError as e:
        print(f"❌ Decoding config import failed: {e}")
        return False

    try:
        from utils.loading_PP_utils import load_epochs_data_for_decoding
        print("✅ Main function import: OK")
    except ImportError as e:
        print(f"❌ Main function import failed: {e}")
        return False

    return True


def test_function_signature():
    """Test the function signature"""
    print("\n🧪 Testing function signature...")

    try:
        from utils.loading_PP_utils import load_epochs_data_for_decoding
        import inspect

        sig = inspect.signature(load_epochs_data_for_decoding)
        params = list(sig.parameters.keys())

        expected_params = [
            'subject_identifier',
            'group_affiliation',
            'base_input_data_path',
            'conditions_to_load',
            'verbose_logging'
        ]

        if params == expected_params:
            print("✅ Function signature: OK")
            print(f"   Parameters: {params}")
            return True
        else:
            print(f"❌ Function signature mismatch")
            print(f"   Expected: {expected_params}")
            print(f"   Got: {params}")
            return False

    except Exception as e:
        print(f"❌ Function signature test failed: {e}")
        return False


def test_parameter_validation():
    """Test parameter validation with invalid inputs"""
    print("\n🧪 Testing parameter validation...")

    try:
        from utils.loading_PP_utils import load_epochs_data_for_decoding

        # Test with invalid subject_identifier
        result = load_epochs_data_for_decoding(
            "", "controls", "/tmp", None, False)
        if result == (None, {}):
            print("✅ Empty subject_identifier validation: OK")
        else:
            print("❌ Empty subject_identifier validation failed")
            return False

        # Test with invalid group_affiliation
        result = load_epochs_data_for_decoding(
            "test_subject", "", "/tmp", None, False)
        if result == (None, {}):
            print("✅ Empty group_affiliation validation: OK")
        else:
            print("❌ Empty group_affiliation validation failed")
            return False

        # Test with invalid path
        result = load_epochs_data_for_decoding(
            "test_subject", "controls", "/nonexistent", None, False)
        if result == (None, {}):
            print("✅ Invalid path validation: OK")
        else:
            print("❌ Invalid path validation failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Parameter validation test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting tests for loading_PP_utils.py corrections\n")

    tests = [
        test_imports,
        test_function_signature,
        test_parameter_validation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The corrections are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the corrections.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
