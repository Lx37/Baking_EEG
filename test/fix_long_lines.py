#!/usr/bin/env python3
"""
Script pour corriger automatiquement les lignes trop longues (> 79 chars) 
dans les fichiers Python tout en préservant la fonctionnalité.
"""

import re
import os


def fix_long_lines_in_file(file_path, max_line_length=79):
    """Corrige les lignes trop longues dans un fichier Python."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    fixes_applied = 0
    
    for i, line in enumerate(lines):
        original_line = line.rstrip()
        
        if len(original_line) <= max_line_length:
            fixed_lines.append(line)
            continue
        
        # Types de corrections à appliquer
        fixed_line = fix_line(original_line, max_line_length)
        
        if fixed_line != original_line:
            fixes_applied += 1
            print(f"Ligne {i+1}: {len(original_line)} -> {len(fixed_line.split('\n')[0])} chars")
        
        # Ajouter la ligne corrigée (peut être multi-lignes)
        if '\n' in fixed_line:
            fixed_lines.extend([l + '\n' for l in fixed_line.split('\n')[:-1]])
            if fixed_line.split('\n')[-1]:  # Si la dernière ligne n'est pas vide
                fixed_lines.append(fixed_line.split('\n')[-1] + '\n')
        else:
            fixed_lines.append(fixed_line + '\n')
    
    # Sauvegarder le fichier corrigé
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    return fixes_applied


def fix_line(line, max_length=79):
    """Corrige une ligne trop longue selon différentes stratégies."""
    
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # 1. Docstrings et commentaires
    if '"""' in stripped or "'''" in stripped:
        return fix_docstring(line, max_length)
    
    if stripped.startswith('#'):
        return fix_comment(line, max_length)
    
    # 2. Imports
    if stripped.startswith('from ') or stripped.startswith('import '):
        return fix_import(line, max_length)
    
    # 3. Chaînes de caractères longues
    if any(quote in stripped for quote in ['"', "'"]):
        return fix_string_line(line, max_length)
    
    # 4. Appels de fonction avec paramètres multiples
    if '(' in stripped and ')' in stripped and ',' in stripped:
        return fix_function_call(line, max_length)
    
    # 5. Expressions conditionnelles ou assignments
    if any(op in stripped for op in [' and ', ' or ', ' = ', ' += ', ' -= ']):
        return fix_expression(line, max_length)
    
    # 6. Listes et dictionnaires
    if any(bracket in stripped for bracket in ['[', '{']) and any(bracket in stripped for bracket in [']', '}']):
        return fix_collection(line, max_length)
    
    # Si aucune règle ne s'applique, essayer une cassure générique
    return fix_generic(line, max_length)


def fix_docstring(line, max_length):
    """Corrige les docstrings trop longues."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    if '"""' in stripped or "'''" in stripped:
        # Ne pas casser les délimiteurs de docstring
        return line.rstrip()
    
    # Pour le contenu de docstring, casser aux espaces
    if len(stripped) > max_length - len(indent):
        words = stripped.split()
        lines = []
        current_line = indent
        
        for word in words:
            if len(current_line + word + ' ') <= max_length:
                current_line += word + ' '
            else:
                lines.append(current_line.rstrip())
                current_line = indent + word + ' '
        
        if current_line.strip():
            lines.append(current_line.rstrip())
        
        return '\n'.join(lines)
    
    return line.rstrip()


def fix_comment(line, max_length):
    """Corrige les commentaires trop longs."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    if len(line.rstrip()) <= max_length:
        return line.rstrip()
    
    # Extraire le contenu du commentaire
    comment_start = stripped.find('#')
    comment_content = stripped[comment_start+1:].strip()
    
    # Casser le commentaire en plusieurs lignes
    words = comment_content.split()
    lines = []
    current_line = indent + '# '
    
    for word in words:
        if len(current_line + word + ' ') <= max_length:
            current_line += word + ' '
        else:
            lines.append(current_line.rstrip())
            current_line = indent + '# ' + word + ' '
    
    if current_line.strip() != indent.strip() + '#':
        lines.append(current_line.rstrip())
    
    return '\n'.join(lines)


def fix_import(line, max_length):
    """Corrige les imports trop longs."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    if 'from ' in stripped and ' import ' in stripped:
        parts = stripped.split(' import ')
        from_part = parts[0]
        import_part = parts[1]
        
        if ',' in import_part:
            # Import multiple - casser par virgules
            imports = [item.strip() for item in import_part.split(',')]
            
            if len(from_part + ' import (' + import_part + ')') <= max_length:
                return f"{indent}{from_part} import ({import_part})"
            else:
                # Multi-lignes
                result = f"{indent}{from_part} import (\n"
                for imp in imports:
                    result += f"{indent}    {imp},\n"
                result += f"{indent})"
                return result
    
    return line.rstrip()


def fix_string_line(line, max_length):
    """Corrige les lignes contenant des chaînes de caractères longues."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # Si c'est une f-string ou logger, préserver la structure
    if any(pattern in stripped for pattern in ['f"', "f'", 'logger', '.format(']):
        # Essayer de casser aux espaces dans la chaîne
        for quote in ['"', "'"]:
            if quote in stripped:
                quote_start = stripped.find(quote)
                quote_end = stripped.rfind(quote)
                if quote_start != quote_end and quote_end - quote_start > 50:
                    # Casser la chaîne
                    before = stripped[:quote_start+1]
                    content = stripped[quote_start+1:quote_end]
                    after = stripped[quote_end:]
                    
                    if len(content) > 40:
                        mid = len(content) // 2
                        space_pos = content.find(' ', mid)
                        if space_pos != -1:
                            part1 = content[:space_pos]
                            part2 = content[space_pos+1:]
                            return f"{indent}{before}{part1} \"\n{indent}    \"{part2}{after}"
    
    return line.rstrip()


def fix_function_call(line, max_length):
    """Corrige les appels de fonction avec beaucoup de paramètres."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # Trouver les paramètres de fonction
    paren_start = stripped.find('(')
    paren_end = stripped.rfind(')')
    
    if paren_start != -1 and paren_end != -1 and ',' in stripped[paren_start:paren_end]:
        func_name = stripped[:paren_start+1]
        params = stripped[paren_start+1:paren_end]
        after_params = stripped[paren_end:]
        
        # Séparer les paramètres
        param_list = []
        current_param = ""
        paren_count = 0
        
        for char in params:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                param_list.append(current_param.strip())
                current_param = ""
                continue
            current_param += char
        
        if current_param.strip():
            param_list.append(current_param.strip())
        
        if len(param_list) > 1:
            # Multi-lignes
            result = f"{indent}{func_name}\n"
            for param in param_list:
                result += f"{indent}    {param},\n"
            result = result.rstrip(',\n') + '\n' + f"{indent}{after_params}"
            return result
    
    return line.rstrip()


def fix_expression(line, max_length):
    """Corrige les expressions longues."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # Casser aux opérateurs logiques
    for op in [' and ', ' or ']:
        if op in stripped:
            parts = stripped.split(op)
            if len(parts) == 2:
                return f"{indent}{parts[0].strip()} \\\n{indent}    {op.strip()} {parts[1].strip()}"
    
    # Casser aux assignations
    if ' = ' in stripped and not any(op in stripped for op in ['==', '!=', '<=', '>=']):
        eq_pos = stripped.find(' = ')
        before = stripped[:eq_pos]
        after = stripped[eq_pos+3:]
        
        if len(after) > 40:
            return f"{indent}{before} = \\\n{indent}    {after}"
    
    return line.rstrip()


def fix_collection(line, max_length):
    """Corrige les listes/dictionnaires longs."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    # Pour les listes simples avec virgules
    if '[' in stripped and ']' in stripped and ',' in stripped:
        bracket_start = stripped.find('[')
        bracket_end = stripped.rfind(']')
        
        before = stripped[:bracket_start]
        content = stripped[bracket_start+1:bracket_end]
        after = stripped[bracket_end:]
        
        items = [item.strip() for item in content.split(',')]
        if len(items) > 2:
            result = f"{indent}{before}[\n"
            for item in items:
                if item:
                    result += f"{indent}    {item},\n"
            result += f"{indent}]{after}"
            return result
    
    return line.rstrip()


def fix_generic(line, max_length):
    """Correction générique - casser à un point logique."""
    stripped = line.strip()
    indent = line[:len(line) - len(line.lstrip())]
    
    if len(stripped) <= max_length - len(indent):
        return line.rstrip()
    
    # Essayer de casser après une virgule
    comma_positions = [i for i, char in enumerate(stripped) if char == ',']
    for pos in reversed(comma_positions):
        if pos < max_length - len(indent) - 10:
            part1 = stripped[:pos+1]
            part2 = stripped[pos+1:].strip()
            return f"{indent}{part1}\n{indent}    {part2}"
    
    # Essayer de casser après un espace proche de la limite
    target_pos = max_length - len(indent) - 5
    space_pos = stripped.rfind(' ', 0, target_pos)
    
    if space_pos > max_length // 2:
        part1 = stripped[:space_pos]
        part2 = stripped[space_pos+1:]
        return f"{indent}{part1} \\\n{indent}    {part2}"
    
    return line.rstrip()


def main():
    """Fonction principale."""
    files_to_fix = [
        'utils/stats_utils.py',
        'examples/run_decoding_one_pp.py'
    ]
    
    total_fixes = 0
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"\n🔧 Correction de {file_path}...")
            fixes = fix_long_lines_in_file(file_path)
            print(f"   {fixes} lignes corrigées")
            total_fixes += fixes
        else:
            print(f"❌ Fichier non trouvé: {file_path}")
    
    print(f"\n✅ Total: {total_fixes} lignes corrigées")


if __name__ == "__main__":
    main()
