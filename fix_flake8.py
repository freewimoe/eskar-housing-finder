#!/usr / bin / env python3
"""
Automated Flake8 Code Style Fixer
Fixes common flake8 issues in Python files
"""

import os
import re
import glob

def fix_whitespace_issues(content):
    """Fix whitespace - related issues"""
    lines = content.split('\n')
    fixed_lines = []

    for i, line in enumerate(lines):
        # Remove trailing whitespace (W291)
        line = line.rstrip()

        # Remove whitespace from blank lines (W293)
        if line.strip() == '':
            line = ''

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_blank_lines(content):
    """Fix blank line issues (E302, E305)"""
    lines = content.split('\n')
    fixed_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this is a function or class definition
        if (line.startswith('def ') or line.startswith('class ')) and i > 0:
            # Check if we need 2 blank lines before
            prev_lines = []
            j = i - 1
            while j >= 0 and lines[j].strip() == '':
                prev_lines.append('')
                j -= 1

            # If previous non - empty line exists and we don't have 2 blank lines
            if j >= 0 and len(prev_lines) < 2:
                # Add necessary blank lines
                while len(prev_lines) < 2:
                    prev_lines.append('')

                # Remove old blank lines and add correct number
                while fixed_lines and fixed_lines[-1] == '':
                    fixed_lines.pop()

                fixed_lines.extend(['', ''])

        fixed_lines.append(lines[i])
        i += 1

    return '\n'.join(fixed_lines)

def fix_arithmetic_operators(content):
    """Fix missing whitespace around arithmetic operators (E226)"""
    # Fix common patterns like x + y -> x + y, x * y -> x * y
    patterns = [
        (r'(\w)(\+)(\w)', r'\1 \2 \3'),
        (r'(\w)(\-)(\w)', r'\1 \2 \3'),
        (r'(\w)(\*)(\w)', r'\1 \2 \3'),
        (r'(\w)(/)(\w)', r'\1 \2 \3'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content

def remove_unused_imports(content):
    """Remove some obvious unused imports"""
    lines = content.split('\n')
    fixed_lines = []

    # Common unused imports to remove
    unused_patterns = [
        r'import datetime\.datetime',
        r'import random$',
    ]

    for line in lines:
        should_remove = False
        for pattern in unused_patterns:
            if re.search(pattern, line):
                should_remove = True
                break

        if not should_remove:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_file(filepath):
    """Fix a single Python file"""
    print(f"Fixing {filepath}...")

    with open(filepath, 'r', encoding='utf - 8') as f:
        content = f.read()

    original_content = content

    # Apply fixes
    content = fix_whitespace_issues(content)
    content = fix_arithmetic_operators(content)
    content = remove_unused_imports(content)

    # Only write if content changed
    if content != original_content:
        with open(filepath, 'w', encoding='utf - 8') as f:
            f.write(content)
        print(f"  ✅ Fixed {filepath}")
    else:
        print(f"  ⏭️  No changes needed for {filepath}")

def main():
    """Main function"""
    print("🔧 ESKAR Housing Finder - Automated Flake8 Fixer")
    print("=" * 50)

    # Get all Python files
    python_files = []
    for pattern in ['*.py', 'src/**/*.py', 'tests/**/*.py']:
        python_files.extend(glob.glob(pattern, recursive=True))

    for filepath in python_files:
        if os.path.isfile(filepath):
            fix_file(filepath)

    print("\n✅ Flake8 auto - fix completed!")
    print("Run 'python -m flake8' again to check remaining issues.")

if __name__ == "__main__":
    main()
