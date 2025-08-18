#!/usr/bin/env python3
"""Script to fix the __init__ method by rewriting it properly"""

def fix_init_method():
    """Fix the __init__ method by adding RNG initialization"""
    
    # Read the current file
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the __init__ method
    start = content.find('def __init__(self):')
    if start == -1:
        print("❌ No __init__ method found")
        return
    
    # Find the start of the method body (after the :)
    method_start = content.find(':', start) + 1
    
    # Find the next method to get the end of __init__
    after_start = start + len('def __init__(self):')
    next_def = content.find('\n    def ', after_start)
    
    if next_def == -1:
        print("❌ Could not find end of __init__ method")
        return
    
    # Get the original method body (everything after the colon and before next method)
    original_body = content[method_start:next_def]
    
    # Create the new method body with RNG initialization
    new_body = '''
        """Initialize the data generator with ESK coordinates and configuration"""
        # Initialize PEP8-compliant random number generator
        import numpy as np
        self.rng = np.random.default_rng(42)  # For reproducible results
        
        # European School Karlsruhe coordinates - Albert-Schweitzer-Str. 1, 76139 Karlsruhe (KORREKTE Koordinaten)''' + original_body[original_body.find('self.esk_location'):]
    
    # Replace the method body in the content
    new_content = content[:method_start] + new_body + content[next_def:]
    
    # Write the corrected content back
    with open('data_generator.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully fixed __init__ method")
    
    # Verify the fix
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        new_content = f.read()
    
    if 'self.rng = np.random.default_rng(42)' in new_content:
        print("✅ RNG initialization confirmed")
    else:
        print("❌ RNG initialization not found")

if __name__ == "__main__":
    fix_init_method()
