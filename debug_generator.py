#!/usr/bin/env python3
"""Debug script to test ESKARDataGenerator initialization"""

print("Testing ESKARDataGenerator initialization...")

try:
    from data_generator import ESKARDataGenerator
    print("✅ Import successful")
    
    print("Creating generator...")
    generator = ESKARDataGenerator()
    print("✅ Generator created")
    
    print("Checking attributes...")
    print(f"Has rng: {hasattr(generator, 'rng')}")
    print(f"Has esk_location: {hasattr(generator, 'esk_location')}")
    
    if hasattr(generator, 'rng'):
        print(f"RNG type: {type(generator.rng)}")
    else:
        print("❌ RNG not found")
        
    # Let's check what attributes the generator actually has
    print("Generator attributes:")
    for attr in dir(generator):
        if not attr.startswith('_'):
            print(f"  {attr}: {type(getattr(generator, attr))}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
