#!/usr/bin/env python3
"""Script to replace the incorrect neighborhood_weights"""

def fix_neighborhood_weights():
    """Fix the neighborhood_weights to have 16 values instead of 6"""
    
    # Read the file
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the old 6-value array with the new 16-value array
    old_weights = "[0.35, 0.25, 0.15, 0.12, 0.08, 0.05]"
    new_weights = '''[
                0.202,  # Weststadt - closest to ESK
                0.152,  # Südstadt - family-friendly
                0.121,  # Innenstadt-West - walking distance
                0.101,  # Durlach - good families area
                0.081,  # Oststadt - good transport
                0.051,  # Mühlburg - affordable
                0.020,  # Stutensee - growing area
                0.020,  # Bruchsal - growing area
                0.020,  # Weingarten (Baden) - affordable families
                0.061,  # Waldstadt - popular with SAP families
                0.040,  # Nordstadt - central location
                0.030,  # Nordweststadt - family-friendly
                0.020,  # Eggenstein-Leopoldshafen - near KIT
                0.040,  # Pfinztal - suburban families
                0.030,  # Grötzingen - quiet residential
                0.010   # Graben-Neudorf - rural option
            ]'''
    
    if old_weights in content:
        content = content.replace(old_weights, new_weights)
        print(f"✅ Replaced old 6-value weights with new 16-value weights")
    else:
        print(f"❌ Could not find old weights pattern: {old_weights}")
        return
    
    # Write back
    with open('data_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ File updated successfully")

if __name__ == "__main__":
    fix_neighborhood_weights()
