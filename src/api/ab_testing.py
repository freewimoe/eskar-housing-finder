"""
ESKAR A / B Testing Framework
Advanced experimentation platform for ML model optimization.

Features:
- Multi - armed bandit testing
- ML model comparison
- Feature flag management
- Statistical significance testing
- Real - time experiment monitoring

Author: Friedrich - Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import sqlite3
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path
from scipy import stats
from datetime import datetime, timedelta
import json

logger = logging.getLogger('ESKAR.ABTesting')

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class ExperimentType(Enum):
    ML_MODEL_COMPARISON = "ml_model_comparison"
    FEATURE_FLAG = "feature_flag"
    UI_VARIANT = "ui_variant"
    ALGORITHM_COMPARISON = "algorithm_comparison"

@dataclass
class ExperimentVariant:
    """Individual variant in an A / B test"""
    variant_id: str
    name: str
    description: str
    config: Dict[str, Any]
    traffic_allocation: float  # 0.0 to 1.0
    is_control: bool = False

@dataclass
class Experiment:
    """A / B test experiment definition"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[ExperimentVariant]
    status: ExperimentStatus
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    success_metric: str
    minimum_sample_size: int
    confidence_level: float
    created_by: str
    metadata: Dict[str, Any]

@dataclass
class ExperimentEvent:
    """Individual event in an experiment"""
    event_id: str
    experiment_id: str
    variant_id: str
    user_session: str
    event_type: str  # 'exposure', 'conversion', 'interaction'
    event_value: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class ExperimentResult:
    """Statistical results of an experiment"""
    experiment_id: str
    variant_results: Dict[str, Dict[str, float]]
    statistical_significance: bool
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float
    winner: Optional[str]
    analysis_timestamp: datetime

class ESKARABTestingFramework:
    """Advanced A / B testing and experimentation platform"""

    def __init__(self, db_path: str = "data / experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)

        # Initialize database
        self._init_database()

        # Active experiments cache
        self._active_experiments = {}
        self._load_active_experiments()

        # Statistical settings
        self.default_confidence_level = 0.95
        self.default_minimum_effect_size = 0.05

    def _init_database(self):
        """Initialize SQLite database for experiments"""
        with sqlite3.connect(self.db_path) as conn:
            # Experiments table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    experiment_type TEXT NOT NULL,
                    variants TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    success_metric TEXT NOT NULL,
                    minimum_sample_size INTEGER NOT NULL,
                    confidence_level REAL NOT NULL,
                    created_by TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            ''')

            # Experiment events table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_events (
                    event_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    user_session TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_value REAL NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

            # Experiment results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiment_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    variant_results TEXT NOT NULL,
                    statistical_significance BOOLEAN NOT NULL,
                    confidence_interval TEXT NOT NULL,
                    p_value REAL NOT NULL,
                    effect_size REAL NOT NULL,
                    winner TEXT,
                    analysis_timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            ''')

            conn.commit()

    def create_ml_model_experiment(self, name: str, description: str,
                                  model_configs: List[Dict], success_metric: str = 'user_rating',
                                  minimum_sample_size: int = 100) -> str:
        """Create ML model comparison experiment"""
        experiment_id = f"ml_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(name) % 1000:03d}"

        # Create variants for each model
        variants = []
        traffic_per_variant = 1.0 / len(model_configs)

        for i, config in enumerate(model_configs):
            variant = ExperimentVariant(
                variant_id=f"model_{i + 1}",
                name=config.get('name', f"Model {i + 1}"),
                description=config.get('description', ''),
                config=config,
                traffic_allocation=traffic_per_variant,
                is_control=(i == 0)  # First model is control
            )
            variants.append(variant)

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=ExperimentType.ML_MODEL_COMPARISON,
            variants=variants,
            status=ExperimentStatus.DRAFT,
            start_date=None,
            end_date=None,
            success_metric=success_metric,
            minimum_sample_size=minimum_sample_size,
            confidence_level=self.default_confidence_level,
            created_by="ESKAR_System",
            metadata={'created_for': 'ml_model_optimization'}
        )

        self._save_experiment(experiment)
        logger.info(f"🧪 Created ML model experiment: {name} with {len(variants)} variants")
        return experiment_id

    def create_feature_experiment(self, name: str, description: str,
                                 feature_variants: List[Dict], success_metric: str = 'user_satisfaction',
                                 minimum_sample_size: int = 50) -> str:
        """Create feature flag experiment"""
        experiment_id = f"feat_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(name) % 1000:03d}"

        variants = []
        traffic_per_variant = 1.0 / len(feature_variants)

        for i, variant_config in enumerate(feature_variants):
            variant = ExperimentVariant(
                variant_id=f"feature_{i + 1}",
                name=variant_config.get('name', f"Variant {i + 1}"),
                description=variant_config.get('description', ''),
                config=variant_config,
                traffic_allocation=traffic_per_variant,
                is_control=(i == 0)
            )
            variants.append(variant)

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=ExperimentType.FEATURE_FLAG,
            variants=variants,
            status=ExperimentStatus.DRAFT,
            start_date=None,
            end_date=None,
            success_metric=success_metric,
            minimum_sample_size=minimum_sample_size,
            confidence_level=self.default_confidence_level,
            created_by="ESKAR_System",
            metadata={'created_for': 'feature_optimization'}
        )

        self._save_experiment(experiment)
        logger.info(f"🚀 Created feature experiment: {name} with {len(variants)} variants")
        return experiment_id

    def start_experiment(self, experiment_id: str) -> bool:
        """Start running an experiment"""
        experiment = self._load_experiment(experiment_id)
        if not experiment:
            logger.error(f"Experiment {experiment_id} not found")
            return False

        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} cannot be started (status: {experiment.status})")
            return False

        # Validate experiment setup
        total_traffic = sum(v.traffic_allocation for v in experiment.variants)
        if abs(total_traffic - 1.0) > 0.001:
            logger.error(f"Traffic allocation must sum to 1.0, got {total_traffic}")
            return False

        # Start experiment
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()

        self._update_experiment(experiment)
        self._active_experiments[experiment_id] = experiment

        logger.info(f"[START] Started experiment: {experiment.name}")
        return True

    def assign_variant(self, experiment_id: str, user_session: str) -> Optional[str]:
        """Assign user to experiment variant using consistent hashing"""
        if experiment_id not in self._active_experiments:
            return None

        experiment = self._active_experiments[experiment_id]

        # Use consistent hashing for variant assignment
        hash_input = f"{experiment_id}_{user_session}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0

        # Assign based on traffic allocation
        cumulative_allocation = 0.0
        for variant in experiment.variants:
            cumulative_allocation += variant.traffic_allocation
            if normalized_hash <= cumulative_allocation:
                # Log exposure event
                self._log_exposure_event(experiment_id, variant.variant_id, user_session)
                return variant.variant_id

        # Fallback to control
        control_variant = next((v for v in experiment.variants if v.is_control), experiment.variants[0])
        self._log_exposure_event(experiment_id, control_variant.variant_id, user_session)
        return control_variant.variant_id

    def log_conversion_event(self, experiment_id: str, variant_id: str,
                           user_session: str, conversion_value: float,
                           metadata: Dict = None):
        """Log a conversion event for statistical analysis"""
        event = ExperimentEvent(
            event_id=f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(user_session) % 1000:03d}",
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_session=user_session,
            event_type='conversion',
            event_value=conversion_value,
            metadata=metadata or {},
            timestamp=datetime.now()
        )

        self._save_event(event)
        logger.debug(f"📊 Logged conversion: {experiment_id}/{variant_id} = {conversion_value}")

    def log_interaction_event(self, experiment_id: str, variant_id: str,
                            user_session: str, interaction_value: float,
                            interaction_type: str, metadata: Dict = None):
        """Log user interaction event"""
        event_metadata = metadata or {}
        event_metadata['interaction_type'] = interaction_type

        event = ExperimentEvent(
            event_id=f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(user_session) % 1000:03d}",
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_session=user_session,
            event_type='interaction',
            event_value=interaction_value,
            metadata=event_metadata,
            timestamp=datetime.now()
        )

        self._save_event(event)
        logger.debug(f"👆 Logged interaction: {experiment_id}/{variant_id} = {interaction_value}")

    def _log_exposure_event(self, experiment_id: str, variant_id: str, user_session: str):
        """Log user exposure to variant"""
        event = ExperimentEvent(
            event_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(user_session) % 1000:03d}",
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_session=user_session,
            event_type='exposure',
            event_value=1.0,
            metadata={},
            timestamp=datetime.now()
        )

        self._save_event(event)

    def analyze_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Perform statistical analysis of experiment results"""
        experiment = self._load_experiment(experiment_id)
        if not experiment:
            return None

        # Get conversion data
        with sqlite3.connect(self.db_path) as conn:
            events_df = pd.read_sql_query('''
                SELECT variant_id, event_type, event_value, user_session
                FROM experiment_events
                WHERE experiment_id = ? AND event_type IN ('exposure', 'conversion')
                ORDER BY timestamp
            ''', conn, params=[experiment_id])

        if events_df.empty:
            logger.warning(f"No data found for experiment {experiment_id}")
            return None

        # Calculate conversion rates by variant
        variant_results = {}

        for variant in experiment.variants:
            variant_data = events_df[events_df['variant_id'] == variant.variant_id]

            exposures = variant_data[variant_data['event_type'] == 'exposure']['user_session'].nunique()
            conversions = variant_data[variant_data['event_type'] == 'conversion']

            if exposures > 0:
                conversion_rate = len(conversions) / exposures
                avg_conversion_value = conversions['event_value'].mean() if not conversions.empty else 0

                variant_results[variant.variant_id] = {
                    'exposures': exposures,
                    'conversions': len(conversions),
                    'conversion_rate': conversion_rate,
                    'avg_conversion_value': avg_conversion_value,
                    'is_control': variant.is_control
                }

        # Statistical significance testing
        control_variant = next((v for v in experiment.variants if v.is_control), experiment.variants[0])
        control_data = variant_results.get(control_variant.variant_id)

        if not control_data or control_data['exposures'] < 10:
            logger.warning(f"Insufficient data for statistical analysis in experiment {experiment_id}")
            return None

        # Find best variant and calculate statistical significance
        best_variant = None
        best_conversion_rate = 0
        p_value = 1.0
        confidence_interval = (0.0, 0.0)
        effect_size = 0.0

        for variant_id, data in variant_results.items():
            if data['conversion_rate'] > best_conversion_rate:
                best_conversion_rate = data['conversion_rate']
                best_variant = variant_id

            # Compare with control using two - proportion z - test
            if variant_id != control_variant.variant_id and data['exposures'] >= 10:
                control_successes = control_data['conversions']
                control_trials = control_data['exposures']
                variant_successes = data['conversions']
                variant_trials = data['exposures']

                if control_trials > 0 and variant_trials > 0:
                    # Two - proportion z - test
                    p1 = control_successes / control_trials
                    p2 = variant_successes / variant_trials

                    pooled_p = (control_successes + variant_successes) / (control_trials + variant_trials)
                    se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / control_trials + 1 / variant_trials))

                    if se > 0:
                        z_score = abs(p2 - p1) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                        # Effect size (Cohen's h)
                        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

                        # Confidence interval for difference
                        margin_error = stats.norm.ppf(0.975) * se
                        confidence_interval = (p2 - p1 - margin_error, p2 - p1 + margin_error)

        # Determine statistical significance
        statistical_significance = p_value < (1 - experiment.confidence_level)

        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval,
            p_value=p_value,
            effect_size=effect_size,
            winner=best_variant if statistical_significance else None,
            analysis_timestamp=datetime.now()
        )

        # Save results
        self._save_experiment_result(result)

        logger.info(f"📈 Analyzed experiment {experiment_id}: "
                   f"{'Significant' if statistical_significance else 'Not significant'} "
                   f"(p={p_value:.4f})")

        return result

    def get_experiment_dashboard(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment dashboard data"""
        experiment = self._load_experiment(experiment_id)
        if not experiment:
            return {}

        # Get latest analysis
        result = self.analyze_experiment(experiment_id)

        # Get event timeline
        with sqlite3.connect(self.db_path) as conn:
            timeline_df = pd.read_sql_query('''
                SELECT DATE(timestamp) as date,
                       variant_id,
                       event_type,
                       COUNT(*) as count
                FROM experiment_events
                WHERE experiment_id = ?
                GROUP BY DATE(timestamp), variant_id, event_type
                ORDER BY date
            ''', conn, params=[experiment_id])

        dashboard = {
            'experiment': asdict(experiment),
            'status': experiment.status.value,
            'duration_days': 0,
            'results': asdict(result) if result else None,
            'timeline_data': timeline_df.to_dict('records') if not timeline_df.empty else [],
            'recommendations': self._generate_experiment_recommendations(experiment, result)
        }

        if experiment.start_date:
            duration = datetime.now() - experiment.start_date
            dashboard['duration_days'] = duration.days

        return dashboard

    def _generate_experiment_recommendations(self, experiment: Experiment,
                                           result: Optional[ExperimentResult]) -> List[str]:
        """Generate actionable recommendations based on experiment results"""
        recommendations = []

        if not result:
            recommendations.append("Collect more data before making decisions")
            return recommendations

        # Check sample size
        total_exposures = sum(v.get('exposures', 0) for v in result.variant_results.values())
        if total_exposures < experiment.minimum_sample_size:
            recommendations.append(f"Continue experiment - need {experiment.minimum_sample_size - total_exposures} more exposures")

        # Statistical significance
        if result.statistical_significance:
            if result.winner:
                recommendations.append(f"[SUCCESS] Implement variant {result.winner} - statistically significant improvement")
            else:
                recommendations.append("[WARNING] Results are significant but no clear winner")
        else:
            if result.p_value > 0.8:
                recommendations.append("[STOP] Consider stopping - very unlikely to find significance")
            else:
                recommendations.append("[CONTINUE] Continue collecting data - trending toward significance")

        # Effect size consideration
        if abs(result.effect_size) < 0.1:
            recommendations.append("[WARNING] Effect size is small - consider practical significance")
        elif abs(result.effect_size) > 0.5:
            recommendations.append("[HIGH IMPACT] Large effect size detected - high business impact expected")

        return recommendations

    def _save_experiment(self, experiment: Experiment):
        """Save experiment to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiments
                (experiment_id, name, description, experiment_type, variants, status,
                 start_date, end_date, success_metric, minimum_sample_size,
                 confidence_level, created_by, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment.experiment_id,
                experiment.name,
                experiment.description,
                experiment.experiment_type.value,
                json.dumps([asdict(v) for v in experiment.variants]),
                experiment.status.value,
                experiment.start_date.isoformat() if experiment.start_date else None,
                experiment.end_date.isoformat() if experiment.end_date else None,
                experiment.success_metric,
                experiment.minimum_sample_size,
                experiment.confidence_level,
                experiment.created_by,
                json.dumps(experiment.metadata),
                datetime.now().isoformat()
            ))
            conn.commit()

    def _update_experiment(self, experiment: Experiment):
        """Update existing experiment"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE experiments SET
                    name = ?, description = ?, status = ?, start_date = ?,
                    end_date = ?, variants = ?, metadata = ?
                WHERE experiment_id = ?
            ''', (
                experiment.name,
                experiment.description,
                experiment.status.value,
                experiment.start_date.isoformat() if experiment.start_date else None,
                experiment.end_date.isoformat() if experiment.end_date else None,
                json.dumps([asdict(v) for v in experiment.variants]),
                json.dumps(experiment.metadata),
                experiment.experiment_id
            ))
            conn.commit()

    def _load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM experiments WHERE experiment_id = ?
            ''', (experiment_id,))
            row = cursor.fetchone()

        if not row:
            return None

        # Parse row data
        experiment_data = dict(zip([col[0] for col in cursor.description], row))

        # Convert JSON fields
        variants_data = json.loads(experiment_data['variants'])
        variants = [ExperimentVariant(**v) for v in variants_data]

        return Experiment(
            experiment_id=experiment_data['experiment_id'],
            name=experiment_data['name'],
            description=experiment_data['description'],
            experiment_type=ExperimentType(experiment_data['experiment_type']),
            variants=variants,
            status=ExperimentStatus(experiment_data['status']),
            start_date=datetime.fromisoformat(experiment_data['start_date']) if experiment_data['start_date'] else None,
            end_date=datetime.fromisoformat(experiment_data['end_date']) if experiment_data['end_date'] else None,
            success_metric=experiment_data['success_metric'],
            minimum_sample_size=experiment_data['minimum_sample_size'],
            confidence_level=experiment_data['confidence_level'],
            created_by=experiment_data['created_by'],
            metadata=json.loads(experiment_data['metadata'])
        )

    def _load_active_experiments(self):
        """Load all active experiments into cache"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT experiment_id FROM experiments
                WHERE status = ?
            ''', (ExperimentStatus.RUNNING.value,))

            for (experiment_id,) in cursor.fetchall():
                experiment = self._load_experiment(experiment_id)
                if experiment:
                    self._active_experiments[experiment_id] = experiment

    def _save_event(self, event: ExperimentEvent):
        """Save experiment event to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiment_events
                (event_id, experiment_id, variant_id, user_session,
                 event_type, event_value, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.experiment_id,
                event.variant_id,
                event.user_session,
                event.event_type,
                event.event_value,
                json.dumps(event.metadata),
                event.timestamp.isoformat()
            ))
            conn.commit()

    def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO experiment_results
                (experiment_id, variant_results, statistical_significance,
                 confidence_interval, p_value, effect_size, winner, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.experiment_id,
                json.dumps(result.variant_results),
                result.statistical_significance,
                json.dumps(result.confidence_interval),
                result.p_value,
                result.effect_size,
                result.winner,
                result.analysis_timestamp.isoformat()
            ))
            conn.commit()
