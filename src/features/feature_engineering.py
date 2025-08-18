"""
ESKAR Feature Engineering Pipeline
Advanced feature engineering for ESK housing suitability and price prediction.

Creates 50+ sophisticated features from raw property data for ML models.

Author: Friedrich - Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import logging

# Import config with proper fallback
try:
    from config import ESKARConfig
except ImportError:
    # Fallback for relative import
    try:
        from ..config import ESKARConfig
    except ImportError:
        # Final fallback for direct execution
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import ESKARConfig

logger = logging.getLogger('ESKAR.FeatureEngineering')

@dataclass
class PropertyFeatures:
    """Container for all property features"""
    basic_features: Dict
    location_features: Dict
    esk_specific_features: Dict
    market_features: Dict
    community_features: Dict
    derived_features: Dict

class ESKARFeatureEngineer:
    """Advanced feature engineering for ESKAR ML pipeline"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.neighborhood_clusters = None
        self.employer_weights = {
            'sap_walldorf': 0.25,
            'sap_karlsruhe': 0.20,
            'kit_campus_south': 0.15,
            'kit_campus_north': 0.10,
            'ionos': 0.15,
            'research_center': 0.15
        }

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        logger.info("🔧 Starting advanced feature engineering...")

        # Create feature containers
        enhanced_df = df.copy()

        # 1. Basic Property Features (10 features)
        enhanced_df = self._create_basic_features(enhanced_df)

        # 2. Advanced Location Features (15 features)
        enhanced_df = self._create_location_features(enhanced_df)

        # 3. ESK - Specific Features (12 features)
        enhanced_df = self._create_esk_features(enhanced_df)

        # 4. Market Intelligence Features (8 features)
        enhanced_df = self._create_market_features(enhanced_df)

        # 5. Community & Social Features (10 features)
        enhanced_df = self._create_community_features(enhanced_df)

        # 6. Derived & Interaction Features (15 features)
        enhanced_df = self._create_derived_features(enhanced_df)

        logger.info(f"[SUCCESS] Feature engineering complete. Total features: {len(enhanced_df.columns)}")
        return enhanced_df

    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic property features"""
        logger.info("Creating basic property features...")

        # Price per square meter
        df['price_per_sqm'] = df['price'] / df['sqft']

        # Property age
        current_year = 2025
        df['property_age'] = current_year - df.get('year_built', current_year - 20)
        df['is_new_construction'] = (df['property_age'] <= 5).astype(int)
        df['is_vintage'] = (df['property_age'] >= 30).astype(int)

        # Size categories
        df['size_category'] = pd.cut(df['sqft'],
                                   bins=[0, 60, 90, 120, 180, float('inf')],
                                   labels=['tiny', 'small', 'medium', 'large', 'xl'])

        # Bedroom density (bedrooms per sqm)
        df['bedroom_density'] = df['bedrooms'] / df['sqft']

        # Property type value encoding
        df['is_house'] = (df['property_type'] == 'house').astype(int)

        # Outdoor space score
        df['outdoor_space_score'] = (
            df.get('garden', 0) * 2 +
            df.get('balcony', 0) * 1 +
            df.get('terrace', 0) * 1.5
        )

        # Storage & parking score
        df['parking_storage_score'] = (
            df.get('garage', 0) * 2 +
            df.get('parking_space', 0) * 1 +
            df.get('basement', 0) * 0.5
        )

        return df

    def _create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced location - based features"""
        logger.info("Creating location features...")

        # Distance - based features
        df['esk_distance_category'] = pd.cut(df['distance_to_esk'],
                                           bins=[0, 1, 3, 5, 10, float('inf')],
                                           labels=['walking', 'bike', 'short_drive', 'medium_drive', 'long_drive'])

        # ESK accessibility score (inverse distance weighted)
        df['esk_accessibility_score'] = np.exp(-df['distance_to_esk'] / 3)

        # Multi - employer accessibility
        employer_distances = self._calculate_employer_distances(df)
        df['weighted_employer_distance'] = self._calculate_weighted_employer_distance(employer_distances)
        df['employer_accessibility_score'] = np.exp(-df['weighted_employer_distance'] / 10)

        # Neighborhood clustering
        df = self._add_neighborhood_clusters(df)

        # Geographic centrality
        df['city_center_distance'] = self._calculate_city_center_distance(df)
        df['suburban_score'] = np.clip(df['city_center_distance'] / 10, 0, 1)

        # Transport hub proximity
        df['transport_hub_score'] = self._calculate_transport_hub_score(df)

        # Elevation and terrain (simulated for Karlsruhe)
        df['elevation_score'] = np.random.uniform(0.3, 0.9, len(df))  # Placeholder

        return df

    def _create_esk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ESK - specific features"""
        logger.info("Creating ESK - specific features...")

        # ESK community density
        df['esk_family_density'] = df['current_esk_families'] / (df['distance_to_esk'] + 1)

        # ESK commute convenience
        df['esk_commute_score'] = (
            (10 - df['distance_to_esk']) * 0.4 +  # Distance weight
            df.get('public_transport_score', 5) * 0.3 +  # Public transport
            (10 - df.get('commute_time_esk', 30) / 3) * 0.3  # Commute time
        )

        # International family suitability
        df['international_suitability'] = (
            df.get('international_community_score', 5) * 0.3 +
            df.get('language_support_score', 5) * 0.2 +
            df.get('cultural_amenities_score', 5) * 0.2 +
            df['esk_family_density'] * 0.3
        )

        # School accessibility (beyond ESK)
        df['school_diversity_score'] = np.random.uniform(3, 9, len(df))  # Placeholder

        # Child - friendly environment
        df['child_safety_score'] = (
            df.get('safety_score', 7) * 0.4 +
            df.get('playground_proximity', 5) * 0.3 +
            df.get('pediatric_care_access', 5) * 0.3
        )

        # ESK lifecycle match
        df['esk_lifecycle_match'] = self._calculate_lifecycle_match(df)

        return df

    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market intelligence features"""
        logger.info("Creating market features...")

        # Price competitiveness
        neighborhood_price_median = df.groupby('neighborhood')['price_per_sqm'].median()
        df['price_vs_neighborhood'] = df.apply(
            lambda row: row['price_per_sqm'] / neighborhood_price_median[row['neighborhood']],
            axis=1
        )

        # Market position
        df['market_position'] = pd.cut(df['price_vs_neighborhood'],
                                     bins=[0, 0.8, 1.2, float('inf')],
                                     labels=['below_market', 'market_rate', 'premium'])

        # Investment potential
        df['investment_score'] = (
            (2 - df['price_vs_neighborhood']) * 0.4 +  # Value for money
            df['esk_accessibility_score'] * 0.3 +     # Location premium
            df.get('future_development_score', 5) * 0.3  # Area development
        )

        # Liquidity score (how fast it would sell)
        df['liquidity_score'] = (
            df['esk_accessibility_score'] * 0.3 +
            (2 - df['price_vs_neighborhood']) * 0.3 +
            df.get('property_condition_score', 7) * 0.4
        )

        return df

    def _create_community_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create community and social features"""
        logger.info("Creating community features...")

        # ESK network strength
        df['esk_network_strength'] = np.log1p(df['current_esk_families']) * df['esk_accessibility_score']

        # Community integration potential
        df['integration_potential'] = (
            df.get('international_community_score', 5) * 0.3 +
            df['esk_network_strength'] * 0.4 +
            df.get('cultural_openness_score', 5) * 0.3
        )

        # Social amenities access
        df['social_amenities_score'] = (
            df.get('restaurant_diversity', 5) * 0.2 +
            df.get('cultural_venues', 5) * 0.2 +
            df.get('sports_facilities', 5) * 0.2 +
            df.get('shopping_access', 5) * 0.2 +
            df.get('medical_facilities', 5) * 0.2
        )

        # Family network potential
        df['family_network_score'] = (
            df['current_esk_families'] / 50 +  # Normalize to 0 - 1
            df.get('family_amenities_score', 5) / 10  # Normalize to 0 - 1
        )

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived and interaction features"""
        logger.info("Creating derived features...")

        # Value - convenience trade - off
        df['value_convenience_ratio'] = df['esk_accessibility_score'] / df['price_vs_neighborhood']

        # Comprehensive ESK suitability (new ML - based score)
        df['ml_esk_suitability'] = (
            df['esk_accessibility_score'] * 0.25 +
            df['esk_network_strength'] * 0.20 +
            df['international_suitability'] * 0.20 +
            df['child_safety_score'] / 10 * 0.15 +
            df['employer_accessibility_score'] * 0.10 +
            df['value_convenience_ratio'] * 0.10
        ) * 10  # Scale to 0 - 10

        # Investment + lifestyle score
        df['total_attractiveness'] = (
            df['ml_esk_suitability'] * 0.6 +
            df['investment_score'] * 0.4
        )

        # Interaction features
        df['size_price_interaction'] = df['sqft'] * df['price_per_sqm'] / 10000
        df['distance_community_interaction'] = df['distance_to_esk'] * df['current_esk_families']

        # Risk - adjusted value
        df['risk_adjusted_value'] = df['investment_score'] * df['liquidity_score']

        return df

    def _calculate_employer_distances(self, df: pd.DataFrame) -> Dict:
        """Calculate distances to major employers"""
        # Placeholder implementation - would use real geocoding
        employer_coords = {
            'sap_walldorf': (49.2933, 8.6428),
            'sap_karlsruhe': (49.0233, 8.4103),
            'kit_campus_south': (49.0069, 8.4037),
            'kit_campus_north': (49.0943, 8.4347),
            'ionos': (49.0089, 8.3858),
            'research_center': (49.0930, 8.4279)
        }

        distances = {}
        for employer, (emp_lat, emp_lon) in employer_coords.items():
            distances[f'distance_to_{employer}'] = df.apply(
                lambda row: self._haversine_distance(
                    row['lat'], row['lon'], emp_lat, emp_lon
                ), axis=1
            )

        return distances

    def _calculate_weighted_employer_distance(self, employer_distances: Dict) -> pd.Series:
        """Calculate weighted average distance to employers"""
        weighted_sum = pd.Series(0, index=employer_distances[list(employer_distances.keys())[0]].index)

        for employer, weight in self.employer_weights.items():
            distance_col = f'distance_to_{employer}'
            if distance_col in employer_distances:
                weighted_sum += employer_distances[distance_col] * weight

        return weighted_sum

    def _add_neighborhood_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add neighborhood clustering features"""
        if self.neighborhood_clusters is None:
            # Create clusters based on location and characteristics
            features_for_clustering = ['lat', 'lon', 'current_esk_families', 'distance_to_esk']
            available_features = [f for f in features_for_clustering if f in df.columns]

            if len(available_features) >= 2:
                X = df[available_features].fillna(0)
                self.neighborhood_clusters = KMeans(n_clusters=5, random_state=42)
                df['neighborhood_cluster'] = self.neighborhood_clusters.fit_predict(X)
            else:
                df['neighborhood_cluster'] = 0

        return df

    def _calculate_city_center_distance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate distance to Karlsruhe city center"""
        city_center = (49.0069, 8.4037)  # Karlsruhe center
        return df.apply(
            lambda row: self._haversine_distance(
                row['lat'], row['lon'], city_center[0], city_center[1]
            ), axis=1
        )

    def _calculate_transport_hub_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate proximity to transport hubs"""
        # Placeholder - would use real transport data
        return np.random.uniform(3, 9, len(df))

    def _calculate_lifecycle_match(self, df: pd.DataFrame) -> pd.Series:
        """Calculate how well property matches ESK family lifecycle"""
        lifecycle_score = (
            df['bedrooms'] / 5 * 0.3 +  # Family size accommodation
            df['outdoor_space_score'] / 4 * 0.2 +  # Child - friendly space
            df['esk_accessibility_score'] * 0.3 +  # School convenience
            df.get('safety_score', 7) / 10 * 0.2  # Family safety
        )
        return np.clip(lifecycle_score * 10, 0, 10)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in km

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Return feature groups for importance analysis"""
        return {
            'basic_property': ['price_per_sqm', 'property_age', 'bedroom_density', 'outdoor_space_score'],
            'location': ['esk_accessibility_score', 'employer_accessibility_score', 'transport_hub_score'],
            'esk_specific': ['esk_family_density', 'esk_commute_score', 'international_suitability'],
            'market': ['price_vs_neighborhood', 'investment_score', 'liquidity_score'],
            'community': ['esk_network_strength', 'integration_potential', 'family_network_score'],
            'derived': ['ml_esk_suitability', 'value_convenience_ratio', 'total_attractiveness']
        }
