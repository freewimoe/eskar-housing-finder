"""
ESKAR Housing Data Generator
European School Karlsruhe Housing Data Generator

Creates realistic Karlsruhe housing data optimized for ESK families.
Based on real market research and ESK community insights.

Author: Friedrich-Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime
import os

class ESKARDataGenerator:
    """Generates ESK-specific housing data for Karlsruhe"""
    
    def __init__(self):
        # European School Karlsruhe coordinates (CORRECTED: Albert-Schweitzer-Str. 1, 76139 Karlsruhe)
        self.esk_location = {"lat": 49.04637, "lon": 8.44805, "name": "European School Karlsruhe"}
        
        # Major employers for ESK families (updated with realistic Karlsruhe employers)
        self.major_employers = {
            'sap_se_walldorf': {"lat": 49.2933, "lon": 8.6428, "name": "SAP SE Walldorf"},
            'sap_labs_karlsruhe': {"lat": 49.0233, "lon": 8.4103, "name": "SAP Labs Karlsruhe"},
            'ionos_se': {"lat": 49.0089, "lon": 8.3858, "name": "Ionos SE"},
            'kit_campus_south': {"lat": 49.0069, "lon": 8.4037, "name": "KIT Campus Süd"},
            'kit_campus_north': {"lat": 49.0943, "lon": 8.4347, "name": "KIT Campus Nord"},
            'forschungszentrum': {"lat": 49.0930, "lon": 8.4279, "name": "Forschungszentrum Karlsruhe"},
            'enbw_karlsruhe': {"lat": 49.0158, "lon": 8.4044, "name": "EnBW Karlsruhe"},
            'dm_drogerie': {"lat": 49.0125, "lon": 8.4200, "name": "dm-drogerie markt"},
            'siemens_karlsruhe': {"lat": 49.0200, "lon": 8.3800, "name": "Siemens Karlsruhe"}
        }
        
        # ESK family preferred neighborhoods (based on real data)
        self.neighborhoods = {
            'Weststadt': {
                'current_esk_families': 45,
                'avg_price_per_sqm': 4200,
                'commute_time_esk': 12,
                'safety_rating': 9.2,
                'international_community': 8.5,
                'family_friendliness': 9.0,
                'public_transport': 8.8,
                'base_lat': 49.0069,
                'base_lon': 8.3737
            },
            'Südstadt': {
                'current_esk_families': 38,
                'avg_price_per_sqm': 4000,
                'commute_time_esk': 15,
                'safety_rating': 8.8,
                'international_community': 8.2,
                'family_friendliness': 8.5,
                'public_transport': 9.0,
                'base_lat': 49.0000,
                'base_lon': 8.3937
            },
            'Innenstadt-West': {
                'current_esk_families': 28,
                'avg_price_per_sqm': 4800,
                'commute_time_esk': 8,
                'safety_rating': 8.5,
                'international_community': 7.5,
                'family_friendliness': 7.8,
                'public_transport': 9.5,
                'base_lat': 49.0069,
                'base_lon': 8.3937
            },
            'Durlach': {
                'current_esk_families': 22,
                'avg_price_per_sqm': 3600,
                'commute_time_esk': 25,
                'safety_rating': 9.0,
                'international_community': 7.0,
                'family_friendliness': 9.2,
                'public_transport': 7.5,
                'base_lat': 48.9969,
                'base_lon': 8.4737
            },
            'Oststadt': {
                'current_esk_families': 15,
                'avg_price_per_sqm': 3800,
                'commute_time_esk': 18,
                'safety_rating': 8.3,
                'international_community': 7.2,
                'family_friendliness': 8.0,
                'public_transport': 8.0,
                'base_lat': 49.0100,
                'base_lon': 8.4200
            },
            'Mühlburg': {
                'current_esk_families': 12,
                'avg_price_per_sqm': 3400,
                'commute_time_esk': 20,
                'safety_rating': 8.0,
                'international_community': 6.8,
                'family_friendliness': 8.2,
                'public_transport': 7.8,
                'base_lat': 49.0169,
                'base_lon': 8.3637
            },
            # NEW EXPANDED REGION: Stutensee-Bruchsal Area
            'Stutensee': {
                'current_esk_families': 18,
                'avg_price_per_sqm': 3200,
                'commute_time_esk': 22,
                'safety_rating': 9.1,
                'international_community': 6.5,
                'family_friendliness': 9.3,
                'public_transport': 7.2,
                'base_lat': 49.0911,
                'base_lon': 8.4530
            },
            'Bruchsal': {
                'current_esk_families': 14,
                'avg_price_per_sqm': 2900,
                'commute_time_esk': 35,
                'safety_rating': 8.7,
                'international_community': 6.2,
                'family_friendliness': 8.8,
                'public_transport': 8.9,
                'base_lat': 49.1240,
                'base_lon': 8.5980
            },
            'Ettlingen': {
                'current_esk_families': 16,
                'avg_price_per_sqm': 3500,
                'commute_time_esk': 28,
                'safety_rating': 8.9,
                'international_community': 7.1,
                'family_friendliness': 9.0,
                'public_transport': 8.3,
                'base_lat': 48.9456,
                'base_lon': 8.4044
            },
            # ADDITIONAL KARLSRUHE NEIGHBORHOODS (for realistic coverage)
            'Waldstadt': {
                'current_esk_families': 8,
                'avg_price_per_sqm': 3300,
                'commute_time_esk': 15,
                'safety_rating': 9.0,
                'international_community': 6.8,
                'family_friendliness': 8.8,
                'public_transport': 8.1,
                'base_lat': 49.0420,
                'base_lon': 8.4580
            },
            'Nordstadt': {
                'current_esk_families': 11,
                'avg_price_per_sqm': 3600,
                'commute_time_esk': 10,
                'safety_rating': 8.4,
                'international_community': 7.5,
                'family_friendliness': 8.1,
                'public_transport': 9.0,
                'base_lat': 49.0250,
                'base_lon': 8.3950
            },
            'Nordweststadt': {
                'current_esk_families': 9,
                'avg_price_per_sqm': 3500,
                'commute_time_esk': 12,
                'safety_rating': 8.6,
                'international_community': 7.2,
                'family_friendliness': 8.5,
                'public_transport': 8.7,
                'base_lat': 49.0180,
                'base_lon': 8.3700
            },
            'Neureut': {
                'current_esk_families': 6,
                'avg_price_per_sqm': 3100,
                'commute_time_esk': 18,
                'safety_rating': 8.8,
                'international_community': 6.5,
                'family_friendliness': 9.1,
                'public_transport': 7.9,
                'base_lat': 49.0580,
                'base_lon': 8.3950
            },
            'Eggenstein': {
                'current_esk_families': 4,
                'avg_price_per_sqm': 2950,
                'commute_time_esk': 20,
                'safety_rating': 8.9,
                'international_community': 6.3,
                'family_friendliness': 8.7,
                'public_transport': 7.6,
                'base_lat': 49.0830,
                'base_lon': 8.4050
            },
            'Blankenloch': {
                'current_esk_families': 5,
                'avg_price_per_sqm': 3000,
                'commute_time_esk': 22,
                'safety_rating': 8.7,
                'international_community': 6.4,
                'family_friendliness': 8.9,
                'public_transport': 7.8,
                'base_lat': 49.0900,
                'base_lon': 8.4200
            }
        }
        
        # ESK family budget ranges (based on typical salaries)
        self.budget_ranges = {
            'entry_level': (300000, 450000),    # Young researchers, PhD students
            'mid_level': (450000, 650000),      # Experienced developers, scientists
            'senior_level': (650000, 900000),   # Senior positions, team leaders
            'executive': (900000, 1500000)      # Management, professors
        }

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in kilometers"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

    def calculate_esk_suitability_score(self, property_data, neighborhood_info):
        """Calculate ESK Family Suitability Score (1-10)"""
        weights = {
            'distance_to_esk': 0.25,        # Very important for daily school commute
            'employer_accessibility': 0.20,  # Important for work commute
            'international_community': 0.15, # ESK families appreciate international environment
            'family_amenities': 0.15,       # Playgrounds, pediatricians, etc.
            'safety': 0.10,                 # Safety for children
            'public_transport': 0.10,       # Public transport for teenagers
            'property_quality': 0.05        # Property quality
        }
        
        score = 0
        
        # Distance to ESK score (closer = better)
        distance_to_esk = property_data.get('distance_to_esk', 10)
        if distance_to_esk <= 1:
            distance_score = 10
        elif distance_to_esk <= 3:
            distance_score = 9
        elif distance_to_esk <= 5:
            distance_score = 7
        elif distance_to_esk <= 10:
            distance_score = 5
        else:
            distance_score = 2
        
        # Employer accessibility
        avg_employer_distance = property_data.get('avg_employer_distance', 20)
        employer_score = max(1, 10 - (avg_employer_distance / 3))
        
        # International community score
        international_score = neighborhood_info['international_community']
        
        # Family amenities score
        family_score = neighborhood_info['family_friendliness']
        
        # Safety score
        safety_score = neighborhood_info['safety_rating']
        
        # Public transport score
        transport_score = neighborhood_info['public_transport']
        
        # Property quality (bedrooms, size, garden)
        bedrooms = property_data.get('bedrooms', 3)
        garden = property_data.get('garden', 0)
        property_score = min(10, (bedrooms * 2) + (garden * 2) + 2)
        
        # Calculate weighted score
        score = (
            weights['distance_to_esk'] * distance_score +
            weights['employer_accessibility'] * employer_score +
            weights['international_community'] * international_score +
            weights['family_amenities'] * family_score +
            weights['safety'] * safety_score +
            weights['public_transport'] * transport_score +
            weights['property_quality'] * property_score
        )
        
        return round(min(10.0, max(1.0, score)), 1)

    def generate_housing_dataset(self, n_samples=300):
        """Generate realistic ESK-optimized housing dataset"""
        np.random.seed(42)  # For reproducible results
        
        print(f"🏫 Generating {n_samples} ESK-optimized properties...")
        
        data = []
        
        for i in range(n_samples):
            # Select neighborhood based on ESK family distribution
            neighborhood_names = list(self.neighborhoods.keys())
            # Updated weights for 15 neighborhoods (sum = 1.0) - Matching actual neighborhood order
            # Higher weights for popular ESK areas, moderate for established areas, lower for newer/distant areas
            neighborhood_weights = [
                0.197,  # Weststadt (still popular, close to ESK)
                0.158,  # Südstadt (popular, central)
                0.119,  # Innenstadt-West (central, convenient)
                0.099,  # Durlach (established, good transport)
                0.069,  # Oststadt (good area)
                0.050,  # Mühlburg (affordable option)
                0.040,  # Stutensee (expanded region)
                0.010,  # Bruchsal (furthest option)
                0.020,  # Ettlingen (southern option)
                0.040,  # Waldstadt (northern, quiet)
                0.079,  # Nordstadt (central-north, good access)
                0.059,  # Nordweststadt (decent residential)
                0.030,  # Neureut (northern suburb)
                0.015,  # Eggenstein (near research centers)
                0.015   # Blankenloch (northern area)
            ]
            
            neighborhood = np.random.choice(neighborhood_names, p=neighborhood_weights)
            neighborhood_info = self.neighborhoods[neighborhood]
            
            # Generate property characteristics
            property_type = np.random.choice(['house', 'apartment'], p=[0.4, 0.6])
            
            # Bedrooms (ESK families typically need 2-5 bedrooms)
            bedrooms = np.random.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1])
            
            # Size based on property type
            if property_type == 'house':
                sqft = np.random.normal(140, 40)  # Houses: 100-220 sqm typically
                garden = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% of houses have gardens
            else:
                sqft = np.random.normal(90, 25)   # Apartments: 60-130 sqm typically
                garden = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% have balconies
            
            sqft = max(50, min(sqft, 300))  # Reasonable bounds
            
            # Generate coordinates around neighborhood center
            lat = neighborhood_info['base_lat'] + np.random.normal(0, 0.01)
            lon = neighborhood_info['base_lon'] + np.random.normal(0, 0.015)
            
            # Calculate distance to ESK
            distance_to_esk = self.calculate_distance(
                lat, lon, 
                self.esk_location['lat'], self.esk_location['lon']
            )
            
            # Calculate average distance to major employers
            employer_distances = []
            for employer_id, employer_info in self.major_employers.items():
                distance = self.calculate_distance(
                    lat, lon,
                    employer_info['lat'], employer_info['lon']
                )
                employer_distances.append(distance)
            
            avg_employer_distance = np.mean(employer_distances)
            
            # Property data for ESK score calculation
            property_data = {
                'distance_to_esk': distance_to_esk,
                'avg_employer_distance': avg_employer_distance,
                'bedrooms': bedrooms,
                'garden': garden
            }
            
            # Calculate ESK suitability score
            esk_score = self.calculate_esk_suitability_score(property_data, neighborhood_info)
            
            # Price calculation based on market data
            base_price_per_sqm = neighborhood_info['avg_price_per_sqm']
            
            # Price variations
            price_variation = np.random.normal(1.0, 0.15)
            price_per_sqm = base_price_per_sqm * price_variation
            
            # Premium for high ESK scores
            if esk_score >= 8.5:
                price_per_sqm *= 1.1
            elif esk_score >= 7.5:
                price_per_sqm *= 1.05
            
            total_price = price_per_sqm * sqft
            
            # Create property record
            property_record = {
                'id': f'ESKAR_{i+1:03d}',
                'neighborhood': neighborhood,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'sqft': round(sqft),
                'garden': garden,
                'price': round(total_price),
                'price_per_sqm': round(price_per_sqm),
                'distance_to_esk': round(distance_to_esk, 1),
                'distance_to_center': round(self.calculate_distance(lat, lon, 49.0069, 8.4037), 1),
                'avg_employer_distance': round(avg_employer_distance, 1),
                'esk_suitability_score': esk_score,
                'safety_score': neighborhood_info['safety_rating'],
                'international_community_score': neighborhood_info['international_community'],
                'family_amenities_score': neighborhood_info['family_friendliness'],
                'public_transport_score': neighborhood_info['public_transport'],
                'current_esk_families': neighborhood_info['current_esk_families'],
                'commute_time_esk': neighborhood_info['commute_time_esk'],
                'lat': round(lat, 6),
                'lon': round(lon, 6)
            }
            
            data.append(property_record)
        
        df = pd.DataFrame(data)
        
        # Display statistics
        print(f"✅ {n_samples} ESK-optimized properties generated!")
        print(f"📊 Average ESK Score: {df['esk_suitability_score'].mean():.1f}/10")
        print(f"🏠 {len(df[df['property_type'] == 'house'])} houses, {len(df[df['property_type'] == 'apartment'])} apartments")
        print(f"🎯 {len(df[df['distance_to_esk'] <= 2])} properties within 2km of ESK")
        
        # Neighborhood distribution
        print("\n📈 Neighborhood Distribution:")
        for neighborhood in df['neighborhood'].unique():
            subset = df[df['neighborhood'] == neighborhood]
            avg_price = subset['price'].mean()
            count = len(subset)
            print(f"   {neighborhood}: {count} properties (⌀ €{avg_price:,.0f})")
        
        return df
    
    def generate_dataset(self, n_samples=300):
        """Alias for generate_housing_dataset for compatibility"""
        return self.generate_housing_dataset(n_samples)

    def save_dataset(self, df, filename='eskar_housing_data.csv'):
        """Save dataset to CSV file"""
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Dataset saved: {filepath}")
        return filepath

def main():
    """Main function to generate and save ESKAR dataset"""
    print("🏫 ESKAR Housing Data Generator")
    print("=" * 60)
    
    generator = ESKARDataGenerator()
    
    # Generate dataset
    df = generator.generate_housing_dataset(n_samples=300)
    
    # Save dataset
    generator.save_dataset(df)
    
    # Top ESK recommendations
    print(f"\n🎯 Top ESK Properties:")
    top_properties = df.nlargest(5, 'esk_suitability_score')[
        ['neighborhood', 'esk_suitability_score', 'distance_to_esk', 'price']
    ]
    print(top_properties.to_string(index=False))

if __name__ == "__main__":
    main()
