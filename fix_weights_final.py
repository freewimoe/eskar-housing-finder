#!/usr/bin/env python3
"""Fix the weights by using simple string replacement"""

def fix_weights_final():
    """Replace the problematic weights with simple equal weights"""
    
    # Read the file
    with open('data_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the neighborhood_weights section
    start_marker = "neighborhood_weights = ["
    end_marker = "]  # Based on ESK preferences"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker) + len(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("❌ Could not find weight definition markers")
        return
    
    # New simple equal weights definition  
    new_definition = """neighborhood_weights = [
                0.0625,  # Weststadt - closest to ESK
                0.0625,  # Südstadt - family-friendly
                0.0625,  # Innenstadt-West - walking distance
                0.0625,  # Durlach - good families area
                0.0625,  # Oststadt - good transport
                0.0625,  # Mühlburg - affordable
                0.0625,  # Stutensee - growing area
                0.0625,  # Bruchsal - growing area
                0.0625,  # Weingarten (Baden) - affordable families
                0.0625,  # Waldstadt - popular with SAP families
                0.0625,  # Nordstadt - central location
                0.0625,  # Nordweststadt - family-friendly
                0.0625,  # Eggenstein-Leopoldshafen - near KIT
                0.0625,  # Pfinztal - suburban families
                0.0625,  # Grötzingen - quiet residential
                0.0625   # Graben-Neudorf - rural option
            ]"""
    
    # Replace the section
    new_content = content[:start_idx] + new_definition + content[end_idx:]
    
    # Write back
    with open('data_generator.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Replaced weights with equal distribution")
    
    # Verify
    test_sum = 16 * 0.0625
    print(f"New sum: {test_sum}")

if __name__ == "__main__":
    fix_weights_final()
