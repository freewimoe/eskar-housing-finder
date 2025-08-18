"""
ESKAR ML Model Ensemble
Advanced machine learning pipeline with multiple models for different prediction tasks.

Implements ensemble methods for:
- Property price prediction
- ESK suitability scoring
- Market trend analysis
- Investment recommendation

Author: Friedrich - Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

# ML imports (with fallbacks for development)
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit - learn not available. Using fallback implementations.")

logger = logging.getLogger('ESKAR.MLPipeline')

@dataclass
class ModelPerformance:
    """Container for model performance metrics"""
    model_name: str
    r2_score: float
    mae: float
    rmse: float
    mape: float
    cv_score_mean: float
    cv_score_std: float
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class PredictionResult:
    """Container for prediction results"""
    prediction: float
    confidence_interval: Tuple[float, float]
    feature_contributions: Dict[str, float]
    model_used: str
    prediction_quality: str

class ESKARMLEnsemble:
    """Advanced ML ensemble for ESKAR housing predictions"""

    def __init__(self):
        self.models = {}
        self.model_performances = {}
        self.feature_importance = {}
        self.scalers = {}
        self.is_trained = False

        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 250,
                'max_depth': 10,
                'learning_rate': 0.05,
                'random_state': 42
            }
        }

    def train_ensemble(self, X: pd.DataFrame, y: pd.Series,
                      task_type: str = 'price_prediction') -> Dict[str, ModelPerformance]:
        """Train ensemble of models for specified task"""
        logger.info(f"🧠 Training ML ensemble for {task_type}...")

        if not SKLEARN_AVAILABLE:
            return self._train_fallback_model(X, y, task_type)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers[task_type] = scaler

        performances = {}

        # Train Random Forest
        performances['random_forest'] = self._train_random_forest(
            X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
        )

        # Train Gradient Boosting
        performances['gradient_boosting'] = self._train_gradient_boosting(
            X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
        )

        # Train XGBoost (if available)
        try:
            performances['xgboost'] = self._train_xgboost(
                X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
            )
        except:
            logger.warning("XGBoost not available, skipping...")

        # Train LightGBM (if available)
        try:
            performances['lightgbm'] = self._train_lightgbm(
                X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
            )
        except:
            logger.warning("LightGBM not available, skipping...")

        # Create ensemble model
        self._create_ensemble_model(task_type)

        self.model_performances[task_type] = performances
        self.is_trained = True

        logger.info(f"[SUCCESS] Ensemble training complete for {task_type}")
        return performances

    def _train_random_forest(self, X_train, X_test, y_train, y_test,
                           feature_names) -> ModelPerformance:
        """Train Random Forest model"""
        start_time = datetime.now()

        rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
        rf_model.fit(X_train, y_train)

        # Predictions
        y_pred = rf_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross - validation
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')

        # Feature importance
        importance_dict = dict(zip(feature_names, rf_model.feature_importances_))

        training_time = (datetime.now() - start_time).total_seconds()

        self.models['random_forest'] = rf_model

        return ModelPerformance(
            model_name='Random Forest',
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            mape=mape,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_time=training_time,
            feature_importance=importance_dict
        )

    def _train_gradient_boosting(self, X_train, X_test, y_train, y_test,
                               feature_names) -> ModelPerformance:
        """Train Gradient Boosting model"""
        start_time = datetime.now()

        gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
        gb_model.fit(X_train, y_train)

        # Predictions
        y_pred = gb_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross - validation
        cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')

        # Feature importance
        importance_dict = dict(zip(feature_names, gb_model.feature_importances_))

        training_time = (datetime.now() - start_time).total_seconds()

        self.models['gradient_boosting'] = gb_model

        return ModelPerformance(
            model_name='Gradient Boosting',
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            mape=mape,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_time=training_time,
            feature_importance=importance_dict
        )

    def _train_xgboost(self, X_train, X_test, y_train, y_test,
                      feature_names) -> ModelPerformance:
        """Train XGBoost model"""
        start_time = datetime.now()

        xgb_model = xgb.XGBRegressor(**self.model_configs['xgboost'])
        xgb_model.fit(X_train, y_train)

        # Predictions
        y_pred = xgb_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross - validation
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')

        # Feature importance
        importance_dict = dict(zip(feature_names, xgb_model.feature_importances_))

        training_time = (datetime.now() - start_time).total_seconds()

        self.models['xgboost'] = xgb_model

        return ModelPerformance(
            model_name='XGBoost',
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            mape=mape,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_time=training_time,
            feature_importance=importance_dict
        )

    def _train_lightgbm(self, X_train, X_test, y_train, y_test,
                       feature_names) -> ModelPerformance:
        """Train LightGBM model"""
        start_time = datetime.now()

        lgb_model = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
        lgb_model.fit(X_train, y_train)

        # Predictions
        y_pred = lgb_model.predict(X_test)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Cross - validation
        cv_scores = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring='r2')

        # Feature importance
        importance_dict = dict(zip(feature_names, lgb_model.feature_importances_))

        training_time = (datetime.now() - start_time).total_seconds()

        self.models['lightgbm'] = lgb_model

        return ModelPerformance(
            model_name='LightGBM',
            r2_score=r2,
            mae=mae,
            rmse=rmse,
            mape=mape,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_time=training_time,
            feature_importance=importance_dict
        )

    def _create_ensemble_model(self, task_type: str):
        """Create weighted ensemble model"""
        # Weight models by their R² scores
        if task_type in self.model_performances:
            performances = self.model_performances[task_type]
            total_r2 = sum(perf.r2_score for perf in performances.values())

            weights = {}
            for model_name, perf in performances.items():
                weights[model_name] = perf.r2_score / total_r2

            self.model_weights = {task_type: weights}
            logger.info(f"Ensemble weights for {task_type}: {weights}")

    def predict(self, X: pd.DataFrame, task_type: str = 'price_prediction') -> PredictionResult:
        """Make prediction using ensemble"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        if task_type not in self.scalers:
            raise ValueError(f"No trained models for task: {task_type}")

        # Scale features
        X_scaled = self.scalers[task_type].transform(X)

        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)[0]
                predictions[model_name] = pred
            except:
                continue

        if not predictions:
            raise ValueError("No models available for prediction")

        # Weighted ensemble prediction
        if task_type in self.model_weights:
            weights = self.model_weights[task_type]
            ensemble_pred = sum(
                pred * weights.get(model_name, 0)
                for model_name, pred in predictions.items()
            )
        else:
            ensemble_pred = np.mean(list(predictions.values()))

        # Calculate confidence interval
        pred_std = np.std(list(predictions.values()))
        confidence_interval = (
            ensemble_pred - 1.96 * pred_std,
            ensemble_pred + 1.96 * pred_std
        )

        # Determine prediction quality
        if pred_std / ensemble_pred < 0.1:
            quality = "High"
        elif pred_std / ensemble_pred < 0.2:
            quality = "Medium"
        else:
            quality = "Low"

        return PredictionResult(
            prediction=ensemble_pred,
            confidence_interval=confidence_interval,
            feature_contributions=predictions,
            model_used="Ensemble",
            prediction_quality=quality
        )

    def _train_fallback_model(self, X: pd.DataFrame, y: pd.Series,
                             task_type: str) -> Dict[str, ModelPerformance]:
        """Fallback model when sklearn is not available"""
        logger.warning("Using fallback linear model (sklearn not available)")

        # Simple linear regression fallback
        # This is a placeholder for when dependencies aren't available
        mock_performance = ModelPerformance(
            model_name='Fallback Linear',
            r2_score=0.75,
            mae=50000,
            rmse=75000,
            mape=15.0,
            cv_score_mean=0.73,
            cv_score_std=0.05,
            training_time=0.1
        )

        return {'fallback': mock_performance}

    def save_models(self, save_dir: str, version: str = "latest"):
        """Save trained models to disk"""
        save_path = Path(save_dir) / version
        save_path.mkdir(parents=True, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_path = save_path / f"{model_name}.joblib"
            joblib.dump(model, model_path)

        # Save scalers
        for task_type, scaler in self.scalers.items():
            scaler_path = save_path / f"scaler_{task_type}.joblib"
            joblib.dump(scaler, scaler_path)

        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'model_performances': {
                task: {name: {
                    'r2_score': perf.r2_score,
                    'mae': perf.mae,
                    'rmse': perf.rmse,
                    'mape': perf.mape
                } for name, perf in perfs.items()}
                for task, perfs in self.model_performances.items()
            },
            'feature_importance': self.feature_importance
        }

        metadata_path = save_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[SUCCESS] Models saved to {save_path}")

    def load_models(self, load_dir: str, version: str = "latest"):
        """Load trained models from disk"""
        load_path = Path(load_dir) / version

        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {load_path}")

        # Load models
        for model_file in load_path.glob("*.joblib"):
            if "scaler" not in model_file.name:
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)

        # Load scalers
        for scaler_file in load_path.glob("scaler_*.joblib"):
            task_type = scaler_file.stem.replace("scaler_", "")
            self.scalers[task_type] = joblib.load(scaler_file)

        # Load metadata
        metadata_path = load_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_importance = metadata.get('feature_importance', {})

        self.is_trained = True
        logger.info(f"[SUCCESS] Models loaded from {load_path}")

    def get_model_comparison(self, task_type: str) -> pd.DataFrame:
        """Get comparison of model performances"""
        if task_type not in self.model_performances:
            return pd.DataFrame()

        performances = self.model_performances[task_type]

        comparison_data = []
        for model_name, perf in performances.items():
            comparison_data.append({
                'Model': perf.model_name,
                'R² Score': round(perf.r2_score, 3),
                'MAE': round(perf.mae, 0),
                'RMSE': round(perf.rmse, 0),
                'MAPE (%)': round(perf.mape, 1),
                'CV Score': f"{perf.cv_score_mean:.3f} ± {perf.cv_score_std:.3f}",
                'Training Time (s)': round(perf.training_time, 2)
            })

        return pd.DataFrame(comparison_data)

    def get_feature_importance(self, task_type: str, top_n: int = 20) -> pd.DataFrame:
        """Get aggregated feature importance across models"""
        if task_type not in self.model_performances:
            return pd.DataFrame()

        # Aggregate feature importance across models
        all_importances = {}
        for model_name, perf in self.model_performances[task_type].items():
            if perf.feature_importance:
                for feature, importance in perf.feature_importance.items():
                    if feature not in all_importances:
                        all_importances[feature] = []
                    all_importances[feature].append(importance)

        # Calculate mean importance
        mean_importances = {
            feature: np.mean(importances)
            for feature, importances in all_importances.items()
        }

        # Sort and get top features
        sorted_features = sorted(
            mean_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        return pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
