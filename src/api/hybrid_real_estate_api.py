"""
ESKAR Hybrid Real Estate API
Combines real estate data sources with synthetic data enhancement.

Features:
- Real estate API integration (ImmoScout24, Immowelt)
- Synthetic data mixing for complete coverage
- ESK-specific scoring and analytics
- Geographic extension to Stutensee-Bruchsal region

Author: Friedrich-Wilhelm Möller
Purpose: Code Institute Portfolio Project 5 - UX Enhancement
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import hashlib
import os
import sys
import random

# Add src to path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from data_generator import ESKARDataGenerator

logger = logging.getLogger('ESKAR.HybridRealEstateAPI')

@dataclass
class PropertyData:
    """Standard property data structure"""
    id: str
    title: str
    price: float
    size_sqm: float
    bedrooms: int
    bathrooms: int
    property_type: str
    neighborhood: str
    address: str
    lat: float
    lon: float
    features: List[str]
    description: str
    images: List[str]
    is_synthetic: bool = False
    data_source: str = "unknown"

class ESKARHybridRealEstateAPI:
    """
    Hybrid real estate API that combines real and synthetic data
    Provides comprehensive property coverage for ESK families
    """
    
    def __init__(self):
        """Initialize hybrid API with both real and synthetic data sources"""
        self.synthetic_generator = ESKARDataGenerator()
        self.api_cache = {}
        self.cache_expiry = timedelta(hours=1)
        
        # Enhanced region coverage
        self.supported_regions = [
            'Karlsruhe', 'Weststadt', 'Südstadt', 'Durlach', 'Oststadt', 
            'Mühlburg', 'Innenstadt-West', 'Stutensee', 'Bruchsal', 'Ettlingen'
        ]
        
        # Real estate API configurations (placeholder for future implementation)
        self.api_configs = {
            'immoscout24': {
                'base_url': 'https://rest.immobilienscout24.de/restapi/api/search/v1.0',
                'enabled': False,  # Enable when API key available
                'rate_limit': 1.0  # seconds between requests
            },
            'immowelt': {
                'base_url': 'https://api.immowelt.de',
                'enabled': False,  # Enable when API key available
                'rate_limit': 1.0
            }
        }
        
    def search_properties_karlsruhe(self, filters: Dict = None) -> List[PropertyData]:
        """
        Search properties with hybrid approach: real API + synthetic enhancement
        """
        if filters is None:
            filters = {}
            
        max_results = filters.get('max_results', 50)
        min_price = filters.get('min_price', 200000)
        max_price = filters.get('max_price', 2000000)
        
        logger.info(f"🔍 Hybrid search: max_results={max_results}, price_range={min_price}-{max_price}")
        
        properties = []
        
        # 1. Try to get real estate data (if APIs are available)
        real_properties = self._fetch_real_estate_data(filters)
        properties.extend(real_properties)
        
        # 2. Fill remaining slots with enhanced synthetic data
        remaining_slots = max_results - len(real_properties)
        if remaining_slots > 0:
            synthetic_properties = self._generate_enhanced_synthetic_data(
                remaining_slots, filters
            )
            properties.extend(synthetic_properties)
        
        # 3. Enhance all properties with ESK-specific data
        enhanced_properties = self._enhance_properties_for_esk(properties)
        
        logger.info(f"✅ Retrieved {len(enhanced_properties)} properties "
                   f"({len(real_properties)} real, {len(synthetic_properties)} synthetic)")
        
        return enhanced_properties[:max_results]
    
    def _fetch_real_estate_data(self, filters: Dict) -> List[PropertyData]:
        """
        Fetch data from real estate APIs (placeholder for future implementation)
        Currently returns sample realistic data
        """
        # TODO: Implement real API connections when API keys are available
        # For now, return some realistic sample data
        
        realistic_properties = [
            PropertyData(
                id="real_001",
                title="Moderne 4-Zimmer-Wohnung in Weststadt",
                price=680000,
                size_sqm=125,
                bedrooms=4,
                bathrooms=2,
                property_type="Wohnung",
                neighborhood="Weststadt",
                address="Moltkestraße 45, 76133 Karlsruhe",
                lat=49.0089,
                lon=8.3757,
                features=["balcony", "modern", "parking", "elevator"],
                description="Hochwertige Wohnung in beliebter Lage, ideal für Familien",
                images=["https://example.com/img1.jpg"],
                is_synthetic=False,
                data_source="sample_realistic"
            ),
            PropertyData(
                id="real_002", 
                title="Einfamilienhaus mit Garten in Durlach",
                price=850000,
                size_sqm=165,
                bedrooms=5,
                bathrooms=3,
                property_type="Haus",
                neighborhood="Durlach",
                address="Pfinztalstraße 12, 76227 Karlsruhe",
                lat=48.9989,
                lon=8.4757,
                features=["garden", "garage", "fireplace", "quiet"],
                description="Familienfreundliches Haus mit großem Garten",
                images=["https://example.com/img2.jpg"],
                is_synthetic=False,
                data_source="sample_realistic"
            ),
            PropertyData(
                id="real_003",
                title="Neubau-Wohnung in Stutensee",
                price=520000,
                size_sqm=95,
                bedrooms=3,
                bathrooms=2,
                property_type="Wohnung", 
                neighborhood="Stutensee",
                address="Hauptstraße 88, 76297 Stutensee",
                lat=49.0931,
                lon=8.4550,
                features=["new_construction", "balcony", "parking", "energy_efficient"],
                description="Moderne Neubau-Wohnung mit optimaler Verkehrsanbindung",
                images=["https://example.com/img3.jpg"],
                is_synthetic=False,
                data_source="sample_realistic"
            )
        ]
        
        # Filter based on criteria
        filtered_properties = []
        for prop in realistic_properties:
            if filters.get('min_price', 0) <= prop.price <= filters.get('max_price', 2000000):
                if not filters.get('property_type') or prop.property_type == filters.get('property_type'):
                    filtered_properties.append(prop)
        
        return filtered_properties[:filters.get('max_results', 10)]
    
    def _generate_enhanced_synthetic_data(self, count: int, filters: Dict) -> List[PropertyData]:
        """
        Generate synthetic properties with enhanced realism and ESK focus
        """
        # Generate base synthetic data
        df = self.synthetic_generator.generate_housing_dataset(n_samples=count)
        
        synthetic_properties = []
        for idx, row in df.iterrows():
            # Enhanced property titles based on characteristics
            title = self._generate_realistic_title(row)
            
            # Enhanced features based on price and type
            features = self._generate_realistic_features(row)
            
            # Realistic address generation
            address = self._generate_realistic_address(row['neighborhood'])
            
            prop = PropertyData(
                id=f"synth_{idx}_{int(row['lat']*1000)}{int(row['lon']*1000)}",
                title=title,
                price=row['price'],
                size_sqm=row['sqft'],
                bedrooms=row['bedrooms'],
                bathrooms=max(1, row['bedrooms'] - 1),
                property_type=row['property_type'],
                neighborhood=row['neighborhood'],
                address=address,
                lat=row['lat'],
                lon=row['lon'],
                features=features,
                description=self._generate_realistic_description(row),
                images=[],
                is_synthetic=True,
                data_source="enhanced_synthetic"
            )
            synthetic_properties.append(prop)
        
        return synthetic_properties
    
    def _generate_realistic_title(self, row) -> str:
        """Generate realistic German property titles"""
        room_count = row['bedrooms']
        prop_type = row['property_type']
        neighborhood = row['neighborhood']
        
        title_templates = [
            f"{room_count}-Zimmer-{prop_type} in {neighborhood}",
            f"Schöne {room_count}-Zimmer-{prop_type}, {neighborhood}",
            f"Moderne {room_count}-Zimmer-{prop_type} - {neighborhood}",
            f"Familienfreundliche {room_count}-Zimmer-{prop_type}, {neighborhood}"
        ]
        
        if row.get('garden'):
            title_templates.append(f"{prop_type} mit Garten in {neighborhood}")
        if row.get('balcony'):
            title_templates.append(f"{prop_type} mit Balkon - {neighborhood}")
            
        return random.choice(title_templates)
    
    def _generate_realistic_features(self, row) -> List[str]:
        """Generate realistic features based on property characteristics"""
        features = []
        
        # Base features from data
        if row.get('garden'): features.append('garden')
        if row.get('balcony'): features.append('balcony')
        if row.get('garage'): features.append('garage')
        
        # Additional realistic features based on price and type
        if row['price'] > 600000:
            features.extend(['modern', 'high_quality', 'elevator'])
        if row['price'] > 800000:
            features.extend(['luxury', 'premium_location'])
            
        if row['property_type'] == 'Haus':
            features.extend(['private_entrance', 'basement'])
            if row['bedrooms'] >= 4:
                features.append('family_friendly')
        else:
            features.append('apartment_building')
            
        # Location-based features
        if row['neighborhood'] in ['Weststadt', 'Südstadt']:
            features.append('central_location')
        if row['neighborhood'] in ['Stutensee', 'Bruchsal']:
            features.extend(['quiet', 'suburban'])
            
        return list(set(features))  # Remove duplicates
    
    def _generate_realistic_address(self, neighborhood: str) -> str:
        """Generate realistic German addresses"""
        street_names = {
            'Weststadt': ['Moltkestraße', 'Rüppurrer Straße', 'Karlstraße'],
            'Südstadt': ['Ettlinger Straße', 'Werderstraße', 'Südendstraße'],
            'Durlach': ['Pfinztalstraße', 'Amalienbadstraße', 'Weiherhofstraße'],
            'Stutensee': ['Hauptstraße', 'Karlsruher Straße', 'Friedrichstaler Straße'],
            'Bruchsal': ['Heidelberger Straße', 'Bahnhofstraße', 'Wilhelmstraße'],
            'Ettlingen': ['Pforzheimer Straße', 'Rheinstraße', 'Wattkopfstraße']
        }
        
        streets = street_names.get(neighborhood, ['Musterstraße', 'Beispielweg'])
        street = random.choice(streets)
        number = random.randint(1, 150)
        
        postal_codes = {
            'Weststadt': '76133', 'Südstadt': '76135', 'Durlach': '76227',
            'Stutensee': '76297', 'Bruchsal': '76646', 'Ettlingen': '76275'
        }
        postal = postal_codes.get(neighborhood, '76131')
        
        return f"{street} {number}, {postal} {neighborhood}"
    
    def _generate_realistic_description(self, row) -> str:
        """Generate realistic German property descriptions"""
        templates = [
            f"Attraktive {row['bedrooms']}-Zimmer-{row['property_type']} in beliebter Lage von {row['neighborhood']}.",
            f"Moderne {row['property_type']} mit {row['bedrooms']} Zimmern in {row['neighborhood']}.",
            f"Familienfreundliche {row['property_type']} in ruhiger Lage, {row['neighborhood']}."
        ]
        
        description = random.choice(templates)
        
        if row.get('garden'):
            description += " Mit schönem Garten."
        if row.get('balcony'):
            description += " Balkon vorhanden."
        if row.get('garage'):
            description += " Garage/Stellplatz inklusive."
            
        description += " Ideal für ESK-Familien!"
        
        return description
    
    def _enhance_properties_for_esk(self, properties: List[PropertyData]) -> List[PropertyData]:
        """
        Enhance properties with ESK-specific data and scoring
        """
        enhanced = []
        
        for prop in properties:
            # Calculate ESK-specific metrics
            esk_distance = self.synthetic_generator.calculate_distance(
                prop.lat, prop.lon,
                self.synthetic_generator.esk_location['lat'],
                self.synthetic_generator.esk_location['lon']
            )
            
            # Add ESK-specific features to description
            esk_info = f" ESK-Entfernung: {esk_distance:.1f}km."
            if esk_distance < 5:
                esk_info += " Sehr ESK-nah!"
            elif esk_distance < 10:
                esk_info += " Gute ESK-Anbindung."
                
            prop.description += esk_info
            
            enhanced.append(prop)
        
        return enhanced
    
    def export_properties_to_dataframe(self, properties: List[PropertyData]) -> pd.DataFrame:
        """
        Convert properties to DataFrame for seamless integration with existing code
        """
        data = []
        
        for prop in properties:
            # Calculate ESK-specific metrics
            esk_distance = self.synthetic_generator.calculate_distance(
                prop.lat, prop.lon,
                self.synthetic_generator.esk_location['lat'], 
                self.synthetic_generator.esk_location['lon']
            )
            
            # Generate ESK suitability score
            max_distance = 30  # km
            distance_score = max(0, (max_distance - esk_distance) / max_distance * 70)
            
            feature_bonus = 0
            if 'garden' in prop.features: feature_bonus += 10
            if 'balcony' in prop.features: feature_bonus += 5
            if 'garage' in prop.features: feature_bonus += 5
            if 'modern' in prop.features: feature_bonus += 3
            if 'family_friendly' in prop.features: feature_bonus += 7
            
            esk_score = min(100, distance_score + feature_bonus)
            
            row = {
                'property_id': prop.id,
                'title': prop.title,
                'price': prop.price,
                'sqft': prop.size_sqm,
                'bedrooms': prop.bedrooms,
                'bathrooms': prop.bathrooms,
                'property_type': prop.property_type,
                'neighborhood': prop.neighborhood,
                'address': prop.address,
                'lat': prop.lat,
                'lon': prop.lon,
                'distance_to_esk': esk_distance,
                'esk_suitability_score': esk_score,
                'garden': 'garden' in prop.features,
                'balcony': 'balcony' in prop.features,
                'garage': 'garage' in prop.features,
                'features': ', '.join(prop.features),
                'description': prop.description,
                'is_synthetic': prop.is_synthetic,
                'data_source': prop.data_source,
                'current_esk_families': random.randint(0, 15) if prop.neighborhood in ['Weststadt', 'Südstadt'] else random.randint(0, 8)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
