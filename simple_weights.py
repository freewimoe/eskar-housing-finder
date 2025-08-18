#!/usr/bin/env python3
"""Simple approach - use equal weights for now"""

def create_simple_weights():
    """Create equal weights for all 16 neighborhoods"""
    
    # Create 16 equal weights that sum to exactly 1.0
    weight = 1.0 / 16
    weights_str = ""
    for i in range(16):
        weights_str += f"                {weight:.10f},  # Neighborhood {i+1}\n"
    
    # Remove the last comma and newline, add closing bracket
    weights_str = weights_str.rstrip(",  # Neighborhood 16\n") + "   # Graben-Neudorf - rural option\n            ]"
    
    print("Equal weights array:")
    print(weights_str)
    print(f"\nEach weight: {weight:.10f}")
    print(f"Sum: {16 * weight:.15f}")

if __name__ == "__main__":
    create_simple_weights()
