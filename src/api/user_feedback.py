"""
ESKAR User Feedback System
Advanced feedback collection and analysis for ML model improvement.

Features:
- User satisfaction tracking
- Property rating system
- Search experience feedback
- ML prediction accuracy feedback
- Analytics dashboard

Author: Friedrich-Wilhelm Möller
Purpose: Code Institute Portfolio Project 5
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger('ESKAR.UserFeedback')

@dataclass
class FeedbackRecord:
    """Individual feedback record"""
    feedback_id: str
    user_session: str
    feedback_type: str  # 'property_rating', 'search_satisfaction', 'prediction_accuracy'
    property_id: Optional[str]
    rating: int  # 1-5 scale
    comments: str
    metadata: Dict[str, Any]
    timestamp: datetime
    processed: bool = False

@dataclass
class UserSession:
    """User session tracking"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    searches_performed: int
    properties_viewed: int
    predictions_requested: int
    satisfaction_score: Optional[int]
    user_type: str  # 'esk_family', 'general_user', 'researcher'

@dataclass
class ModelFeedback:
    """ML model performance feedback"""
    model_name: str
    prediction_id: str
    predicted_score: float
    user_rating: int
    property_id: str
    feedback_timestamp: datetime
    accuracy_score: float

class ESKARFeedbackSystem:
    """Advanced feedback collection and analysis system"""
    
    def __init__(self, db_path: str = None):
        # Use absolute path for database
        if db_path is None:
            base_path = Path(__file__).resolve().parent.parent.parent
            self.db_path = base_path / "data" / "feedback.db"
        else:
            self.db_path = Path(db_path)
        
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Feedback categories
        self.feedback_categories = {
            'property_rating': {
                'description': 'Rate property suitability for ESK families',
                'scale': 'Very Poor (1) to Excellent (5)'
            },
            'search_satisfaction': {
                'description': 'Overall search experience satisfaction',
                'scale': 'Very Dissatisfied (1) to Very Satisfied (5)'
            },
            'prediction_accuracy': {
                'description': 'How accurate were our AI predictions?',
                'scale': 'Very Inaccurate (1) to Very Accurate (5)'
            },
            'feature_usefulness': {
                'description': 'How useful are specific app features?',
                'scale': 'Not Useful (1) to Very Useful (5)'
            }
        }
    
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.db_path) as conn:
            # Feedback records table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS feedback_records (
                    feedback_id TEXT PRIMARY KEY,
                    user_session TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    property_id TEXT,
                    rating INTEGER NOT NULL,
                    comments TEXT,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # User sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    searches_performed INTEGER DEFAULT 0,
                    properties_viewed INTEGER DEFAULT 0,
                    predictions_requested INTEGER DEFAULT 0,
                    satisfaction_score INTEGER,
                    user_type TEXT
                )
            ''')
            
            # Model feedback table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    prediction_id TEXT NOT NULL,
                    predicted_score REAL NOT NULL,
                    user_rating INTEGER NOT NULL,
                    property_id TEXT NOT NULL,
                    feedback_timestamp TEXT NOT NULL,
                    accuracy_score REAL
                )
            ''')
            
            conn.commit()
    
    def start_user_session(self, user_type: str = 'esk_family') -> str:
        """Start a new user session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 10000:04d}"
        
        session = UserSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            searches_performed=0,
            properties_viewed=0,
            predictions_requested=0,
            satisfaction_score=None,
            user_type=user_type
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO user_sessions 
                (session_id, start_time, user_type, searches_performed, 
                 properties_viewed, predictions_requested)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session.session_id, session.start_time.isoformat(), 
                  session.user_type, 0, 0, 0))
            conn.commit()
        
        logger.info(f"[SESSION] Started user session: {session_id}")
        return session_id
    
    def update_session_activity(self, session_id: str, activity_type: str):
        """Update session activity counters"""
        with sqlite3.connect(self.db_path) as conn:
            if activity_type == 'search':
                conn.execute('''
                    UPDATE user_sessions 
                    SET searches_performed = searches_performed + 1 
                    WHERE session_id = ?
                ''', (session_id,))
            elif activity_type == 'view_property':
                conn.execute('''
                    UPDATE user_sessions 
                    SET properties_viewed = properties_viewed + 1 
                    WHERE session_id = ?
                ''', (session_id,))
            elif activity_type == 'request_prediction':
                conn.execute('''
                    UPDATE user_sessions 
                    SET predictions_requested = predictions_requested + 1 
                    WHERE session_id = ?
                ''', (session_id,))
            
            conn.commit()
    
    def end_user_session(self, session_id: str, satisfaction_score: Optional[int] = None):
        """End user session with optional satisfaction rating"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE user_sessions 
                SET end_time = ?, satisfaction_score = ?
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), satisfaction_score, session_id))
            conn.commit()
        
        logger.info(f"[SUCCESS] Ended user session: {session_id}")
    
    def collect_property_feedback(self, session_id: str, property_id: str, 
                                 rating: int, comments: str = "", 
                                 metadata: Dict = None) -> str:
        """Collect feedback about a specific property"""
        feedback_id = f"prop_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(property_id) % 1000:03d}"
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            user_session=session_id,
            feedback_type='property_rating',
            property_id=property_id,
            rating=rating,
            comments=comments,
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        self._save_feedback(feedback)
        logger.info(f"📊 Collected property feedback: {property_id} rated {rating}/5")
        return feedback_id
    
    def collect_search_feedback(self, session_id: str, satisfaction_rating: int,
                               search_criteria: Dict, results_count: int,
                               comments: str = "") -> str:
        """Collect feedback about search experience"""
        feedback_id = f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(session_id) % 1000:03d}"
        
        metadata = {
            'search_criteria': search_criteria,
            'results_count': results_count,
            'search_timestamp': datetime.now().isoformat()
        }
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            user_session=session_id,
            feedback_type='search_satisfaction',
            property_id=None,
            rating=satisfaction_rating,
            comments=comments,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        self._save_feedback(feedback)
        logger.info(f"[FEEDBACK] Collected search feedback: {satisfaction_rating}/5 satisfaction")
        return feedback_id
    
    def collect_prediction_feedback(self, session_id: str, model_name: str,
                                   prediction_id: str, predicted_score: float,
                                   property_id: str, user_rating: int,
                                   comments: str = "") -> str:
        """Collect feedback about ML prediction accuracy"""
        feedback_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(prediction_id) % 1000:03d}"
        
        # Calculate accuracy score (how close prediction was to user rating)
        # Convert user rating (1-5) to same scale as prediction (0-1)
        user_score_normalized = (user_rating - 1) / 4.0
        accuracy_score = 1.0 - abs(predicted_score - user_score_normalized)
        
        # Save to model feedback table
        model_feedback = ModelFeedback(
            model_name=model_name,
            prediction_id=prediction_id,
            predicted_score=predicted_score,
            user_rating=user_rating,
            property_id=property_id,
            feedback_timestamp=datetime.now(),
            accuracy_score=accuracy_score
        )
        
        self._save_model_feedback(model_feedback)
        
        # Also save as general feedback
        metadata = {
            'model_name': model_name,
            'prediction_id': prediction_id,
            'predicted_score': predicted_score,
            'accuracy_score': accuracy_score
        }
        
        feedback = FeedbackRecord(
            feedback_id=feedback_id,
            user_session=session_id,
            feedback_type='prediction_accuracy',
            property_id=property_id,
            rating=user_rating,
            comments=comments,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        self._save_feedback(feedback)
        logger.info(f"🤖 Collected prediction feedback: {user_rating}/5 accuracy, score: {accuracy_score:.3f}")
        return feedback_id
    
    def _save_feedback(self, feedback: FeedbackRecord):
        """Save feedback record to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO feedback_records 
                (feedback_id, user_session, feedback_type, property_id, 
                 rating, comments, metadata, timestamp, processed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id,
                feedback.user_session,
                feedback.feedback_type,
                feedback.property_id,
                feedback.rating,
                feedback.comments,
                json.dumps(feedback.metadata),
                feedback.timestamp.isoformat(),
                feedback.processed
            ))
            conn.commit()
    
    def _save_model_feedback(self, model_feedback: ModelFeedback):
        """Save model-specific feedback"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_feedback 
                (model_name, prediction_id, predicted_score, user_rating, 
                 property_id, feedback_timestamp, accuracy_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_feedback.model_name,
                model_feedback.prediction_id,
                model_feedback.predicted_score,
                model_feedback.user_rating,
                model_feedback.property_id,
                model_feedback.feedback_timestamp.isoformat(),
                model_feedback.accuracy_score
            ))
            conn.commit()
    
    def get_feedback_analytics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall ratings by type
            ratings_df = pd.read_sql_query('''
                SELECT feedback_type, rating, COUNT(*) as count
                FROM feedback_records 
                WHERE timestamp >= ?
                GROUP BY feedback_type, rating
            ''', conn, params=[cutoff_date.isoformat()])
            
            # Session analytics
            sessions_df = pd.read_sql_query('''
                SELECT user_type, 
                       AVG(searches_performed) as avg_searches,
                       AVG(properties_viewed) as avg_properties_viewed,
                       AVG(predictions_requested) as avg_predictions,
                       AVG(satisfaction_score) as avg_satisfaction,
                       COUNT(*) as total_sessions
                FROM user_sessions 
                WHERE start_time >= ?
                GROUP BY user_type
            ''', conn, params=[cutoff_date.isoformat()])
            
            # Model accuracy
            model_accuracy_df = pd.read_sql_query('''
                SELECT model_name,
                       AVG(accuracy_score) as avg_accuracy,
                       AVG(user_rating) as avg_user_rating,
                       COUNT(*) as total_predictions
                FROM model_feedback 
                WHERE feedback_timestamp >= ?
                GROUP BY model_name
            ''', conn, params=[cutoff_date.isoformat()])
        
        analytics = {
            'period_days': days_back,
            'summary': {
                'total_feedback_records': len(ratings_df),
                'total_sessions': len(sessions_df),
                'avg_rating_overall': ratings_df['rating'].mean() if not ratings_df.empty else 0
            },
            'ratings_by_type': {},
            'session_analytics': {},
            'model_performance': {}
        }
        
        # Process ratings by type
        for feedback_type in ratings_df['feedback_type'].unique():
            type_data = ratings_df[ratings_df['feedback_type'] == feedback_type]
            analytics['ratings_by_type'][feedback_type] = {
                'avg_rating': type_data['rating'].mean(),
                'total_responses': type_data['count'].sum(),
                'distribution': dict(zip(type_data['rating'], type_data['count']))
            }
        
        # Process session analytics
        for _, row in sessions_df.iterrows():
            analytics['session_analytics'][row['user_type']] = {
                'avg_searches': row['avg_searches'],
                'avg_properties_viewed': row['avg_properties_viewed'],
                'avg_predictions': row['avg_predictions'],
                'avg_satisfaction': row['avg_satisfaction'],
                'total_sessions': row['total_sessions']
            }
        
        # Process model performance
        for _, row in model_accuracy_df.iterrows():
            analytics['model_performance'][row['model_name']] = {
                'avg_accuracy': row['avg_accuracy'],
                'avg_user_rating': row['avg_user_rating'],
                'total_predictions': row['total_predictions']
            }
        
        return analytics
    
    def get_improvement_suggestions(self) -> List[Dict[str, str]]:
        """Generate improvement suggestions based on feedback"""
        analytics = self.get_feedback_analytics()
        suggestions = []
        
        # Check overall satisfaction
        overall_rating = analytics['summary']['avg_rating_overall']
        if overall_rating < 3.5:
            suggestions.append({
                'type': 'critical',
                'area': 'User Satisfaction',
                'suggestion': f'Overall satisfaction is low ({overall_rating:.1f}/5). Consider UX improvements.',
                'priority': 'high'
            })
        
        # Check model accuracy
        model_performance = analytics.get('model_performance', {})
        for model_name, performance in model_performance.items():
            if performance['avg_accuracy'] < 0.7:
                suggestions.append({
                    'type': 'model',
                    'area': f'ML Model: {model_name}',
                    'suggestion': f'Model accuracy is {performance["avg_accuracy"]:.2f}. Consider retraining.',
                    'priority': 'medium'
                })
        
        # Check search satisfaction
        search_ratings = analytics.get('ratings_by_type', {}).get('search_satisfaction', {})
        if search_ratings and search_ratings['avg_rating'] < 3.5:
            suggestions.append({
                'type': 'feature',
                'area': 'Search Experience',
                'suggestion': 'Search satisfaction is low. Improve search filters and results.',
                'priority': 'high'
            })
        
        return suggestions
    
    def export_feedback_data(self, output_path: str = "data/feedback_export.csv"):
        """Export all feedback data for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            # Join feedback with session data
            query = '''
                SELECT f.*, s.user_type, s.searches_performed, s.properties_viewed
                FROM feedback_records f
                LEFT JOIN user_sessions s ON f.user_session = s.session_id
                ORDER BY f.timestamp DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            df.to_csv(output_path, index=False)
            
        logger.info(f"📤 Exported feedback data to {output_path}")
        return output_path
