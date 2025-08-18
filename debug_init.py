#!/usr/bin/env python3
"""Script to debug and fix the __init__ method in data_generator.py"""

def debug_init_method():
    """Read and analyze the __init__ method"""
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the __init__ method
    start = content.find('def __init__(self):')
    if start == -1:
        print("❌ No __init__ method found")
        return
        
    # Find the next method or class to get the end of __init__
    after_start = start + len('def __init__(self):')
    next_def = content.find('\n    def ', after_start)
    next_class = content.find('\nclass ', after_start)
    
    # Get the smaller of the two (whichever comes first)
    if next_def == -1:
        end = next_class if next_class != -1 else len(content)
    elif next_class == -1:
        end = next_def
    else:
        end = min(next_def, next_class)
    
    init_method = content[start:end]
    print("Current __init__ method:")
    print("=" * 50)
    print(init_method)
    print("=" * 50)
    
    # Check if it contains rng initialization
    if 'self.rng' in init_method:
        print("✅ Contains self.rng initialization")
    else:
        print("❌ Missing self.rng initialization")
        
    if 'import numpy' in init_method:
        print("✅ Contains numpy import")
    else:
        print("❌ Missing numpy import")

if __name__ == "__main__":
    debug_init_method()
