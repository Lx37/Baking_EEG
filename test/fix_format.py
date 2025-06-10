#!/usr/bin/env python3
"""
Script pour corriger automatiquement le formatage du fichier run_decoding_one_pp.py
"""

import re
import os


def fix_long_lines_in_function_calls(content):
    """Corrige les lignes trop longues dans les appels de fonction"""

    # Pattern pour les appels de fonction avec de nombreux paramètres
    patterns = [
        # logger calls
        (r'logger_run_one\.(info|warning|error)\(\s*"([^"]+)",\s*(.+)\)',
         lambda m: f'logger_run_one.{m.group(1)}(\n            "{m.group(2)}",\n            {m.group(3)})'),

        # Long dictionary definitions
        (r'(\s+)(["\w_]+): ([^,]+), (["\w_]+): ([^,]+),',
         lambda m: f'{m.group(1)}{m.group(2)}: {m.group(3)},\n{m.group(1)}{m.group(4)}: {m.group(5)},'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content


def fix_imports(content):
    """Corrige les imports redondants"""
    lines = content.split('\n')

    # Supprimer les imports redondants de 'os'
    seen_imports = set()
    fixed_lines = []

    for line in lines:
        # Si c'est un import simple
        if line.strip().startswith('import ') and not 'from' in line:
            module = line.strip().split()[1].split('.')[0]
            if module not in seen_imports:
                seen_imports.add(module)
                fixed_lines.append(line)
            elif module == 'os' and line.strip() == 'import os':
                # Skip duplicate os import
                continue
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_trailing_whitespace(content):
    """Supprime les espaces en fin de ligne"""
    lines = content.split('\n')
    return '\n'.join(line.rstrip() for line in lines)


def fix_line_length_basic(content):
    """Corrige quelques lignes trop longues de base"""
    fixes = [
        # Corrections basiques de longueur
        ('cluster_threshold_config_intra_fold = INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG',
         'cluster_threshold_config_intra_fold = (\n            INTRA_FOLD_CLUSTER_THRESHOLD_CONFIG\n        )'),

        # Corrections de commentaires longs
        ('# param_grid_config_for_subject est déjà conditionné par USE_GRID_SEARCH_OPTIMIZATION',
         '# param_grid_config_for_subject est déjà conditionné par\n        # USE_GRID_SEARCH_OPTIMIZATION'),

        ('# fixed_params_for_subject est déjà conditionné par not USE_GRID_SEARCH_OPTIMIZATION',
         '# fixed_params_for_subject est déjà conditionné par\n        # not USE_GRID_SEARCH_OPTIMIZATION'),
    ]

    for old, new in fixes:
        content = content.replace(old, new)

    return content


def main():
    file_path = '/Users/tom/Desktop/ENSC/Stage CAP/Baking_EEG/examples/run_decoding_one_pp.py'

    print("🔧 Correction automatique du formatage...")

    # Lire le fichier
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Appliquer les corrections
    content = fix_imports(content)
    content = fix_trailing_whitespace(content)
    content = fix_line_length_basic(content)

    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✅ Corrections automatiques appliquées")
    print("⚠️  Il reste probablement des corrections manuelles à faire")


if __name__ == "__main__":
    main()
