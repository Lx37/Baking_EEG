#!/usr/bin/env python3
"""Test rapide des imports principaux."""

import sys
import os

# Ajouter le bon chemin
PROJECT_ROOT = "/Users/tom/Desktop/ENSC/Stage CAP/BakingEEG/Baking_EEG"
sys.path.insert(0, PROJECT_ROOT)

try:
    from examples.run_decoding_one_pp import execute_single_subject_decoding
    print("✅ execute_single_subject_decoding importé")
except ImportError as e:
    print(f"❌ execute_single_subject_decoding: {e}")

try:
    from examples.run_decoding_one_group_pp import execute_group_intra_subject_decoding_analysis
    print("✅ execute_group_intra_subject_decoding_analysis importé")
except ImportError as e:
    print(f"❌ execute_group_intra_subject_decoding_analysis: {e}")

try:
    from config.config import ALL_SUBJECT_GROUPS
    print(f"✅ ALL_SUBJECT_GROUPS importé ({len(ALL_SUBJECT_GROUPS)} groupes)")
except ImportError as e:
    print(f"❌ ALL_SUBJECT_GROUPS: {e}")

print("🎉 Test terminé!")
