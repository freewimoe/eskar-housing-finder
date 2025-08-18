"""
ESKAR Configuration Management
Centralized configuration for ML pipeline, APIs, and deployment settings.

Author: Friedrich - Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

@dataclass
class MLConfig:
    """Machine Learning Pipeline Configuration"""

    # Model Parameters
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Ensemble Models
    rf_n_estimators: int = 200
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5

    xgb_n_estimators: int = 300
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.1

    lgb_n_estimators: int = 250
    lgb_max_depth: int = 10
    lgb_learning_rate: float = 0.05

    # Feature Engineering
    max_features: int = 50
    feature_selection_threshold: float = 0.01

    # Model Performance Thresholds
    min_r2_score: float = 0.75
    min_mae_threshold: float = 50000  # €50k MAE acceptable
    min_mape_threshold: float = 0.15  # 15% MAPE acceptable

@dataclass
class DataConfig:
    """Data Pipeline Configuration"""

    # File Paths
    raw_data_dir: str = "data / raw"
    processed_data_dir: str = "data / processed"
    interim_data_dir: str = "data / interim"

    # Dataset Configuration
    min_dataset_size: int = 500
    max_dataset_size: int = 10000

    # ESK - Specific Parameters
    max_esk_distance: float = 15.0  # km
    esk_coordinates: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.esk_coordinates is None:
            self.esk_coordinates = {"lat": 49.0464700608647, "lon": 8.44612290974462}

@dataclass
class APIConfig:
    """External API Configuration"""

    # ImmoScout24 API
    immoscout_base_url: str = "https://rest.immobilienscout24.de / restapi / api"
    immoscout_timeout: int = 30
    immoscout_rate_limit: int = 100  # requests per hour

    # OpenStreetMap Nominatim
    osm_base_url: str = "https://nominatim.openstreetmap.org"
    osm_timeout: int = 10

    # Geocoding Service
    geocoding_cache_ttl: int = 86400  # 24 hours

    # API Keys (from environment)
    immoscout_api_key: Optional[str] = None

    def __post_init__(self):
        self.immoscout_api_key = os.getenv('IMMOSCOUT_API_KEY')

@dataclass
class AppConfig:
    """Streamlit Application Configuration"""

    # UI Settings
    page_title: str = "ESKAR Housing Finder"
    page_icon: str = "🏫"
    layout: str = "wide"

    # Performance
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 100

    # Display Settings
    max_properties_display: int = 20
    default_map_zoom: int = 12

    # ESK Branding
    primary_color: str = "#1e3a8a"
    secondary_color: str = "#3b82f6"

class ESKARConfig:
    """Main ESKAR Configuration Manager"""

    def __init__(self):
        self.ml = MLConfig()
        self.data = DataConfig()
        self.api = APIConfig()
        self.app = AppConfig()
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('eskar.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ESKAR')

    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Validate ML parameters
            assert 0 < self.ml.test_size < 1, "Test size must be between 0 and 1"
            assert self.ml.cv_folds > 1, "CV folds must be > 1"

            # Validate data parameters
            assert self.data.min_dataset_size > 0, "Min dataset size must be positive"
            assert self.data.max_esk_distance > 0, "Max ESK distance must be positive"

            # Validate coordinates
            lat = self.data.esk_coordinates['lat']
            lon = self.data.esk_coordinates['lon']
            assert -90 <= lat <= 90, "Latitude must be between -90 and 90"
            assert -180 <= lon <= 180, "Longitude must be between -180 and 180"

            self.logger.info("[SUCCESS] Configuration validation successful")
            return True

        except AssertionError as e:
            self.logger.error(f"[ERROR] Configuration validation failed: {e}")
            return False

    def get_model_save_path(self, model_name: str, version: str = "latest") -> str:
        """Get standardized path for saving models"""
        return f"models / versioned/{version}/{model_name}.joblib"

    def get_data_path(self, filename: str, data_type: str = "processed") -> str:
        """Get standardized path for data files"""
        if data_type == "raw":
            return f"{self.data.raw_data_dir}/{filename}"
        elif data_type == "interim":
            return f"{self.data.interim_data_dir}/{filename}"
        else:
            return f"{self.data.processed_data_dir}/{filename}"

# Global configuration instance
config = ESKARConfig()

# Validate configuration on import
if not config.validate_config():
    raise ValueError("ESKAR configuration validation failed")
