#!/usr/bin/env python3
"""Create exactly normalized weights for the 16 neighborhoods"""

import numpy as np

def create_perfect_weights():
    """Create weights that sum to exactly 1.0"""
    
    # Original desired weights (not normalized)
    desired_weights = [
        0.20,  # Weststadt - closest to ESK
        0.15,  # Südstadt - family-friendly
        0.12,  # Innenstadt-West - walking distance
        0.10,  # Durlach - good families area
        0.08,  # Oststadt - good transport
        0.05,  # Mühlburg - affordable
        0.02,  # Stutensee - growing area
        0.02,  # Bruchsal - growing area
        0.02,  # Weingarten (Baden) - affordable families
        0.06,  # Waldstadt - popular with SAP families
        0.04,  # Nordstadt - central location
        0.03,  # Nordweststadt - family-friendly
        0.02,  # Eggenstein-Leopoldshafen - near KIT
        0.04,  # Pfinztal - suburban families
        0.03,  # Grötzingen - quiet residential
        0.01   # Graben-Neudorf - rural option
    ]
    
    # Normalize to sum exactly to 1.0
    weights_array = np.array(desired_weights)
    normalized_weights = weights_array / weights_array.sum()
    
    print("Perfect normalized weights:")
    for i, weight in enumerate(normalized_weights):
        print(f"                {weight:.10f},  # {i+1}")
    
    print(f"\nSum: {normalized_weights.sum():.15f}")
    print(f"Count: {len(normalized_weights)}")
    
    # Test that they work with numpy choice
    neighborhoods = [f"neighborhood_{i}" for i in range(16)]
    try:
        rng = np.random.default_rng(42)
        result = rng.choice(neighborhoods, p=normalized_weights)
        print(f"✅ Test successful: chose {result}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    return normalized_weights

if __name__ == "__main__":
    create_perfect_weights()
