#!/usr / bin / env python3
"""
Unit tests for Data Generation functionality
Tests synthetic housing data creation and validation
"""

import unittest
import pandas as pd
from pathlib import Path

class TestDataGeneration(unittest.TestCase):
    """Test suite for Data Generation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.sample_neighborhoods = [
            'Weststadt', 'Südstadt', 'Innenstadt - West',
            'Durlach', 'Oststadt', 'Mühlburg'
        ]
        self.property_types = ['house', 'apartment']

    def test_property_generation_structure(self):
        """Test generated property has correct structure"""
        # Mock property generation
        property_data = {
            'neighborhood': random.choice(self.sample_neighborhoods),
            'property_type': random.choice(self.property_types),
            'bedrooms': random.randint(2, 5),
            'sqft': random.randint(50, 300),
            'garden': random.choice([True, False]),
            'garage': random.choice([True, False]),
            'price': random.randint(300000, 1200000),
            'latitude': 49.0 + random.uniform(-0.05, 0.05),
            'longitude': 8.4 + random.uniform(-0.05, 0.05),
            'safety_score': round(random.uniform(7.0, 9.5), 1),
            'current_esk_families': random.randint(0, 8)
        }

        # Test required fields exist
        required_fields = [
            'neighborhood', 'property_type', 'bedrooms', 'sqft',
            'garden', 'garage', 'price', 'latitude', 'longitude',
            'safety_score', 'current_esk_families'
        ]

        for field in required_fields:
            self.assertIn(field, property_data)

    def test_property_data_types(self):
        """Test property data types are correct"""
        property_data = {
            'neighborhood': 'Weststadt',
            'property_type': 'house',
            'bedrooms': 4,
            'sqft': 150,
            'garden': True,
            'garage': False,
            'price': 650000,
            'latitude': 49.0123,
            'longitude': 8.4567,
            'safety_score': 8.5,
            'current_esk_families': 3
        }

        # Test data types
        self.assertIsInstance(property_data['neighborhood'], str)
        self.assertIsInstance(property_data['property_type'], str)
        self.assertIsInstance(property_data['bedrooms'], int)
        self.assertIsInstance(property_data['sqft'], int)
        self.assertIsInstance(property_data['garden'], bool)
        self.assertIsInstance(property_data['garage'], bool)
        self.assertIsInstance(property_data['price'], int)
        self.assertIsInstance(property_data['latitude'], float)
        self.assertIsInstance(property_data['longitude'], float)
        self.assertIsInstance(property_data['safety_score'], float)
        self.assertIsInstance(property_data['current_esk_families'], int)

    def test_property_value_ranges(self):
        """Test property values are within expected ranges"""
        # Generate test data
        bedrooms = random.randint(2, 5)
        sqft = random.randint(50, 300)
        price = random.randint(300000, 1200000)
        safety_score = round(random.uniform(7.0, 9.5), 1)
        current_esk_families = random.randint(0, 8)

        # Test ranges
        self.assertIn(bedrooms, range(2, 6))
        self.assertIn(sqft, range(50, 301))
        self.assertIn(price, range(300000, 1200001))
        self.assertGreaterEqual(safety_score, 7.0)
        self.assertLessEqual(safety_score, 9.5)
        self.assertIn(current_esk_families, range(0, 9))

    def test_neighborhood_validity(self):
        """Test neighborhood names are valid Karlsruhe districts"""
        valid_neighborhoods = [
            'Weststadt', 'Südstadt', 'Innenstadt - West',
            'Durlach', 'Oststadt', 'Mühlburg'
        ]

        for neighborhood in valid_neighborhoods:
            self.assertIsInstance(neighborhood, str)
            self.assertGreater(len(neighborhood), 0)
            # Test neighborhood is a known Karlsruhe district
            self.assertIn(neighborhood, [
                'Weststadt', 'Südstadt', 'Innenstadt - West',
                'Durlach', 'Oststadt', 'Mühlburg'
            ])

    def test_coordinate_validity(self):
        """Test coordinates are within Karlsruhe bounds"""
        # Karlsruhe approximate bounds
        karlsruhe_lat_min, karlsruhe_lat_max = 48.95, 49.15
        karlsruhe_lon_min, karlsruhe_lon_max = 8.25, 8.55

        # Generate test coordinates
        test_lat = 49.0 + random.uniform(-0.05, 0.05)
        test_lon = 8.4 + random.uniform(-0.05, 0.05)

        # Test coordinates are reasonable for Karlsruhe
        self.assertGreaterEqual(test_lat, karlsruhe_lat_min)
        self.assertLessEqual(test_lat, karlsruhe_lat_max)
        self.assertGreaterEqual(test_lon, karlsruhe_lon_min)
        self.assertLessEqual(test_lon, karlsruhe_lon_max)

    def test_esk_distance_calculation(self):
        """Test ESK distance calculation for generated properties"""
        import math

        # ESK coordinates (as used in the application)
        esk_lat, esk_lon = 49.0464700608647, 8.44612290974462

        # Test property coordinates
        test_lat, test_lon = 49.0123, 8.4567

        # Calculate distance using Haversine formula
        R = 6371  # Earth radius in km
        dlat = math.radians(test_lat - esk_lat)
        dlon = math.radians(test_lon - esk_lon)
        a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
             math.cos(math.radians(esk_lat)) * math.cos(math.radians(test_lat)) *
             math.sin(dlon / 2) * math.sin(dlon / 2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # Test distance is reasonable
        self.assertIsInstance(distance, float)
        self.assertGreater(distance, 0)
        self.assertLess(distance, 20)  # Should be within 20km for Karlsruhe area

    def test_esk_suitability_scoring(self):
        """Test ESK suitability score calculation logic"""
        # Test property for scoring
        test_property = {
            'bedrooms': 4,
            'garden': True,
            'garage': True,
            'safety_score': 8.5,
            'current_esk_families': 3,
            'distance_to_esk': 2.5
        }

        # Mock ESK suitability calculation
        base_score = 50

        # Distance factor (closer is better)
        distance_score = max(0, 30 - (test_property['distance_to_esk'] * 5))

        # Feature bonuses
        feature_score = 0
        if test_property['garden']: feature_score += 5
        if test_property['garage']: feature_score += 5
        if test_property['bedrooms'] >= 3: feature_score += 10

        # Community factor
        community_score = min(15, test_property['current_esk_families'] * 2)

        # Safety factor
        safety_score = (test_property['safety_score'] - 7) * 5

        total_score = base_score + distance_score + feature_score + community_score + safety_score
        final_score = min(100, max(0, total_score))

        # Test score is valid
        self.assertIsInstance(final_score, (int, float))
        self.assertGreaterEqual(final_score, 0)
        self.assertLessEqual(final_score, 100)

        # Test score makes sense for good property
        self.assertGreater(final_score, 70)  # Should be high for good ESK property

if __name__ == '__main__':
    print("🏗️ ESKAR Housing Finder - Data Generation Tests")
    print("=" * 50)

    # Run tests
    unittest.main(verbosity=2)
