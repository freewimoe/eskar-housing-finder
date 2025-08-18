#!/usr / bin / env python3
"""Test script for regional expansion verification"""

from data_generator import ESKARDataGenerator

def test_regional_expansion():
    """Test the new Stutensee - Bruchsal - Durlach regions"""
    print("🔧 ESKAR UX IMPROVEMENTS - REGIONAL EXPANSION TEST")
    print("=" * 50)

    # Initialize generator
    gen = ESKARDataGenerator()

    print(f"📊 Total neighborhoods: {len(gen.neighborhoods)}")
    print()

    # Test new regions
    new_regions = ['Stutensee', 'Bruchsal', 'Weingarten (Baden)']
    print("🆕 New regions added:")
    for name in new_regions:
        if name in gen.neighborhoods:
            data = gen.neighborhoods[name]
            print(f"  ✅ {name}:")
            print(f"     📍 Commute to ESK: {data['commute_time_esk']} min")
            print(f"     💰 Avg price / sqm: €{data['avg_price_per_sqm']}")
            print(f"     👨‍👩‍👧‍👦 Current ESK families: {data['current_esk_families']}")
            print(f"     🔒 Safety rating: {data['safety_rating']}/10")
        else:
            print(f"  ❌ {name}: NOT FOUND")

    print()
    print("📋 All neighborhoods:")
    for name in sorted(gen.neighborhoods.keys()):
        esk_families = gen.neighborhoods[name]['current_esk_families']
        commute = gen.neighborhoods[name]['commute_time_esk']
        print(f"  • {name}: {esk_families} ESK families, {commute}min commute")

if __name__ == "__main__":
    test_regional_expansion()
