#!/usr/bin/env python3
"""Fix the missing newline in the __init__ method"""

def fix_esk_location():
    """Add the missing newline before self.esk_location"""
    
    # Read the file
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix the problematic line
    problem_text = "Karlsruhe (KORREKTE Koordinaten)self.esk_location"
    fixed_text = "Karlsruhe (KORREKTE Koordinaten)\n        self.esk_location"
    
    if problem_text in content:
        content = content.replace(problem_text, fixed_text)
        print("✅ Fixed missing newline before self.esk_location")
    else:
        print("❌ Could not find the problematic text")
        return
    
    # Write back
    with open('data_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File updated successfully")

if __name__ == "__main__":
    fix_esk_location()
