#!/usr/bin/env python3
"""
Quick fix for remaining np.random calls in data_generator.py
"""

def fix_np_random_calls():
    """Replace all np.random calls with self.rng calls"""
    
    file_path = 'data_generator.py'
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original np.random calls
    import re
    original_count = len(re.findall(r'np\.random\.[a-zA-Z_]+', content))
    print(f"Found {original_count} np.random calls to fix")
    
    # Replace all np.random calls
    replacements = [
        ('np.random.choice', 'self.rng.choice'),
        ('np.random.normal', 'self.rng.normal'),
        ('np.random.seed(42)', '# Seed is set in __init__ with self.rng'),
    ]
    
    for old, new in replacements:
        count_before = content.count(old)
        content = content.replace(old, new)
        count_after = content.count(old)
        replaced = count_before - count_after
        if replaced > 0:
            print(f"  Replaced {replaced} instances of '{old}' with '{new}'")
    
    # Write back the corrected content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Verify no np.random calls remain
    final_count = len(re.findall(r'np\.random\.[a-zA-Z_]+', content))
    print(f"After replacement: {final_count} np.random calls remain")
    
    if final_count == 0:
        print("✅ All np.random calls successfully replaced!")
    else:
        print("⚠️  Some np.random calls still remain")

if __name__ == "__main__":
    fix_np_random_calls()
